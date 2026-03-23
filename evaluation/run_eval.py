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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="lm-eval runner for Tiny Aya Vision benchmarks.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="hf-multimodal", choices=["vllm", "hf", "hf-multimodal"])
    parser.add_argument("--batch-size", type=str, default="1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
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
    logger.info(f"=========================================")

    # Register TinyAyaVision with HuggingFace Auto classes so lm-eval can
    # load it via AutoModelForCausalLM.from_pretrained / AutoConfig.
    if not args.skip_registration:
        import models  # noqa: F401 — triggers Auto class registration

    import lm_eval
    import lm_eval.tasks

    model_args = f"pretrained={args.model_name},dtype={args.dtype},trust_remote_code={args.trust_remote_code}"
    if args.backend == "vllm":
        model_args += ",tensor_parallel_size=1"

    eval_kwargs = dict(
        model=args.backend,
        model_args=model_args,
        tasks=[args.task],
        batch_size=args.batch_size,
        task_manager=lm_eval.tasks.TaskManager(include_path="evaluation/tasks"),
        log_samples=args.log_samples,
    )

    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    if args.apply_chat_template:
        eval_kwargs["apply_chat_template"] = True

    results = lm_eval.simple_evaluate(**eval_kwargs)

    from lm_eval.utils import make_table
    logger.info(make_table(results))

    if args.output_dir:
        import json
        from pathlib import Path

        # Store results under output_dir/model_name/
        model_name_sanitized = args.model_name.replace("/", "__")
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
            task_results[group_name] = aggregated
            logger.info(f"Aggregate {group_name}: {aggregated}")

        # Save aggregated results (overall + per-language)
        with open(output_path / f"{args.task}_results.json", "w") as f:
            json.dump(task_results, f, indent=2, ensure_ascii=False)

        # Save per-task sample-level JSONL files
        if args.log_samples and results.get("samples"):
            for task_name, task_samples in results["samples"].items():
                with open(output_path / f"samples_{task_name}.jsonl", "w") as f:
                    for sample in task_samples:
                        f.write(json.dumps(sample, default=str, ensure_ascii=False) + "\n")

        logger.info(f"Results and samples saved to {output_path}/")


if __name__ == "__main__":
    main()
