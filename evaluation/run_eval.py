"""
uv run python evaluation/run_eval.py

Args:
    [--task]: cvqa, cvqa_blind, xmmmu, ...
    [--backend]: vllm | hf | hf-multimodal
    [--batch-size]: auto | 1 (vllm | hf)
    [--limit]: int  (num samples for quick tests)
    [--output-dir]: str
    [--model-name]: str (HF repo id OR local path to save_pretrained output)

"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _model_output_slug(model_name: str, model_subfolder: str | None = None) -> str:
    slug = model_name.replace("/", "__")
    if model_subfolder:
        slug += "__" + model_subfolder.strip("/").replace("/", "__")
    return slug


def _resolve_task_sample_count(task_manager, task_name: str) -> tuple[str, int]:
    """Return the concrete lm-eval task name and eval split length.

    This lets chunked eval use the same dataset resolution path as lm-eval
    itself: task YAML -> datasets.load_dataset/custom_dataset -> HF datasets
    cache. It avoids requiring task-specific local metadata files.
    """
    loaded_tasks = task_manager.load_task_or_group([task_name])
    if len(loaded_tasks) != 1:
        task_names = ", ".join(sorted(loaded_tasks))
        raise ValueError(
            "--chunk-size currently supports a single concrete task. "
            f"{task_name!r} resolved to: {task_names}"
        )

    resolved_name, task = next(iter(loaded_tasks.items()))
    try:
        total_samples = len(task.eval_docs)
    except TypeError as exc:
        raise TypeError(
            f"Task {resolved_name!r} does not expose a sized eval dataset; "
            "chunked evaluation requires len(task.eval_docs)."
        ) from exc

    return resolved_name, total_samples


def _resolve_chunked_task_sample_count(task_manager, task_name: str) -> tuple[str, int]:
    if task_name == "cvqa":
        from datasets import load_dataset_builder

        builder = load_dataset_builder("afaji/cvqa")
        return "cvqa", builder.info.splits["test"].num_examples

    return _resolve_task_sample_count(task_manager, task_name)


def _chunk_metadata(task_name: str, start: int, end: int) -> dict[str, int] | None:
    if task_name != "cvqa":
        return None

    return {"cvqa_chunk_start": start, "cvqa_chunk_end": end}


def main():
    parser = argparse.ArgumentParser(description="lm-eval runner for Tiny Aya Vision benchmarks.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="hf-multimodal", choices=["vllm", "hf", "hf-multimodal", "tiny-aya-vision"])
    parser.add_argument("--batch-size", type=str, default="1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None, help="Max samples per inference pass; iterates all chunks automatically")
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    parser.add_argument("--model-subfolder", type=str, default=None, help="Optional HF Hub subfolder to load from inside --model-name")
    parser.add_argument("--log-samples", action="store_true", help="Log per-question results")
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template")
    parser.add_argument("--skip-registration", action="store_true", help="Skip TinyAyaVision Auto class registration (use for external baseline models)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype (bfloat16, float16, auto, etc.)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Pass trust_remote_code=True to model loader")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    logger.info(f"Starting evaluation for task: {args.task}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Model: {args.model_name}")
    if args.model_subfolder:
        logger.info(f"Model subfolder: {args.model_subfolder}")
    logger.info(f"=========================================")

    # Register TinyAyaVision with HuggingFace Auto classes so lm-eval can
    # load it via AutoModelForCausalLM.from_pretrained / AutoConfig.
    if not args.skip_registration:
        import models  # noqa: F401 — triggers Auto class registration

    # Register the custom tiny-aya-vision lm-eval backend (via @register_model).
    if args.backend == "tiny-aya-vision":
        import evaluation.tiny_aya_vision_lm_eval  # noqa: F401

    import lm_eval
    import lm_eval.tasks

    model_args = f"pretrained={args.model_name},dtype={args.dtype},trust_remote_code={args.trust_remote_code}"
    if args.model_subfolder:
        model_args += f",subfolder={args.model_subfolder}"
    if args.backend == "vllm":
        model_args += ",tensor_parallel_size=1"

    task_manager = lm_eval.tasks.TaskManager(include_path="evaluation/tasks")

    eval_kwargs = dict(
        tasks=[args.task],
        batch_size=args.batch_size,
        task_manager=task_manager,
        log_samples=True,  # always collect per-sample results
    )

    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    if args.apply_chat_template:
        eval_kwargs["apply_chat_template"] = True

    if args.chunk_size:
        # --- Chunked evaluation ---
        # Create the model once, then reuse it across all chunks.
        # simple_evaluate accepts a pre-initialized LM object so the model
        # is never reloaded between chunks.
        import lm_eval.api.registry

        lm = lm_eval.api.registry.get_model(args.backend).create_from_arg_string(
            model_args, {"batch_size": args.batch_size}
        )

        resolved_task_name, total_samples = _resolve_chunked_task_sample_count(
            task_manager, args.task
        )
        if args.limit is not None:
            total_samples = min(total_samples, args.limit)
        eval_kwargs.pop("limit", None)  # mutually exclusive with samples= in lm-eval
        logger.info(f"Dataset size: {total_samples} | chunk size: {args.chunk_size}")

        import json as _json2
        from pathlib import Path as _Path

        model_name_sanitized = _model_output_slug(args.model_name, args.model_subfolder)
        chunk_suffix = f"_limit{args.limit}" if args.limit is not None else ""
        chunk_output_dir = (
            _Path(args.output_dir)
            / model_name_sanitized
            / f"{args.task}_chunks{chunk_suffix}"
        )
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        samples_jsonl = chunk_output_dir / "samples.jsonl"

        # Resume from a previous partial run: count already-written samples.
        n_done = 0
        if samples_jsonl.exists():
            with open(samples_jsonl) as _f:
                n_done = sum(1 for line in _f if line.strip())
            logger.info(f"Resuming: {n_done} samples already written, skipping to chunk {n_done // args.chunk_size + 1}")

        sample_metrics: dict[str, list] = {}
        n_chunks = (total_samples + args.chunk_size - 1) // args.chunk_size
        eval_start_time = time.monotonic()

        for chunk_idx in range(n_chunks):
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, total_samples)

            if end <= n_done:
                logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: already done, skipping")
                continue

            logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: samples {start}–{end - 1}")

            chunk_metadata = _chunk_metadata(resolved_task_name, start, end)
            chunk_kwargs = {
                **eval_kwargs,
                "model": lm,
                "task_manager": lm_eval.tasks.TaskManager(
                    include_path="evaluation/tasks",
                    metadata=chunk_metadata,
                ),
            }
            if chunk_metadata is None:
                chunk_kwargs["samples"] = {resolved_task_name: list(range(start, end))}

            try:
                chunk_results = lm_eval.simple_evaluate(**chunk_kwargs)
            except BaseException:
                logger.error(f"Chunk {chunk_idx + 1} failed:")
                traceback.print_exc()
                sys.exit(1)

            # Append this chunk's samples to the running JSONL immediately.
            chunk_samples = (chunk_results.get("samples") or {}).get(
                resolved_task_name, []
            )
            with open(samples_jsonl, "a") as _f:
                for s in chunk_samples:
                    _f.write(_json2.dumps(s, default=str, ensure_ascii=False) + "\n")

            for s in chunk_samples:
                for k, v in s.items():
                    if isinstance(v, (int, float)) and k != "doc_id":
                        sample_metrics.setdefault(k, []).append(v)

            n_so_far = n_done + (chunk_idx + 1 - n_done // args.chunk_size) * args.chunk_size
            running_metrics = {k: sum(vs) / len(vs) for k, vs in sample_metrics.items()}
            elapsed = time.monotonic() - eval_start_time
            logger.info(
                f"Chunk {chunk_idx + 1}/{n_chunks} done | "
                f"samples so far: {min(end, total_samples)} | "
                f"elapsed: {elapsed:.1f}s | "
                f"running metrics: {running_metrics}"
            )
            # Write running aggregate after every chunk.
            with open(chunk_output_dir / "running_results.json", "w") as _f:
                _json2.dump({
                    "chunks_done": chunk_idx + 1,
                    "samples_done": min(end, total_samples),
                    "total_samples": total_samples,
                    "elapsed_seconds": round(elapsed, 1),
                    "metrics": running_metrics,
                }, _f, indent=2)

        # Final aggregate from all per-sample metric values in samples.jsonl
        if not sample_metrics:
            # We resumed and skipped all chunks — recompute from JSONL.
            with open(samples_jsonl) as _f:
                for line in _f:
                    line = line.strip()
                    if not line:
                        continue
                    s = _json2.loads(line)
                    for k, v in s.items():
                        if isinstance(v, (int, float)) and k != "doc_id":
                            sample_metrics.setdefault(k, []).append(v)
            # Read elapsed from the checkpoint written by the original run.
            running_results_path = chunk_output_dir / "running_results.json"
            if running_results_path.exists():
                with open(running_results_path) as _f:
                    elapsed = _json2.load(_f).get("elapsed_seconds", 0.0)
            else:
                elapsed = 0.0

        final_metrics = {k: sum(vs) / len(vs) for k, vs in sample_metrics.items()}
        total_elapsed = round(time.monotonic() - eval_start_time if sample_metrics else elapsed, 1)
        logger.info(f"Final aggregated metrics: {final_metrics} | total elapsed: {total_elapsed}s")
        results = {"results": {args.task: final_metrics}, "samples": None}  # samples on disk

    else:
        # --- Single-pass evaluation ---
        eval_kwargs["model"] = args.backend
        eval_kwargs["model_args"] = model_args

        try:
            results = lm_eval.simple_evaluate(**eval_kwargs)
        except BaseException:
            logger.error("Evaluation failed with exception:")
            traceback.print_exc()
            sys.exit(1)

    if args.output_dir:
        import json
        from pathlib import Path

        # Store results under output_dir/model_name/
        model_name_sanitized = _model_output_slug(args.model_name, args.model_subfolder)
        output_path = Path(args.output_dir) / model_name_sanitized
        output_path.mkdir(parents=True, exist_ok=True)

        task_results = results.get("results", {})

        # Compute aggregate score for the group across sub-tasks
        group_name = args.task
        if group_name in task_results:
            subtask_scores = {}
            for key, metrics in task_results.items():
                if key == group_name:
                    continue
                for metric_name, value in metrics.items():
                    if "stderr" in metric_name or not isinstance(value, (int, float)):
                        continue
                    subtask_scores.setdefault(metric_name, []).append(value)

            aggregated = {}
            for metric_name, values in subtask_scores.items():
                aggregated[metric_name] = sum(values) / len(values)
            # Only overwrite group metrics when there are real subtask aggregates.
            # In chunked single-task runs, task_results[group_name] already contains
            # the correct final metrics and subtask_scores is empty.
            if aggregated:
                task_results[group_name] = aggregated
                logger.info(f"Aggregate {group_name}: {aggregated}")

        chunk_suffix = f"_limit{args.limit}" if args.limit is not None else ""

        # Save aggregated results (overall + per-language)
        with open(output_path / f"{args.task}_results{chunk_suffix}.json", "w") as f:
            json.dump(task_results, f, indent=2, ensure_ascii=False)

        # Save per-task sample-level JSONL files
        if results.get("samples"):
            for task_name, task_samples in results["samples"].items():
                with open(output_path / f"samples_{task_name}{chunk_suffix}.jsonl", "w") as f:
                    for sample in task_samples:
                        f.write(json.dumps(sample, default=str, ensure_ascii=False) + "\n")

        logger.info(f"Results and samples saved to {output_path}/")


if __name__ == "__main__":
    main()
