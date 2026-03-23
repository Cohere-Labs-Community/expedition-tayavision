"""
Evaluate CohereLabs/tiny-aya-base (text-only, no vision encoder) on:
  1. XMMMU        (neulab/PangeaBench-xmmmu)      — MC, answers available
  2. Kaleidoscope (CohereLabs/kaleidoscope)        — MC, answers available
  3. MaXM         (floschne/maxm)                  — Open-ended QA
  4. MTVQA        (ByteDance/MTVQA)                — Open-ended QA

Uses batched inference for throughput on multi-GPU setups.
MC tasks use log-likelihood scoring; open-ended QA uses greedy generation.
Images ignored (text-only model).

Usage:
    python eval_tiny_aya_base.py [--benchmarks xmmmu kaleidoscope maxm mtvqa]
                                  [--max-samples N]
                                  [--batch-size B]
                                  [--output-dir DIR]
"""

import argparse
import ast
import json
import re
import string
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

OPTION_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace/punctuation, normalize unicode."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def relaxed_match(prediction: str, references: list[str]) -> bool:
    """Check if normalized prediction matches any normalized reference."""
    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return False
    for ref in references:
        ref_norm = normalize_text(ref)
        if not ref_norm:
            continue
        if pred_norm == ref_norm or ref_norm in pred_norm or pred_norm in ref_norm:
            return True
    return False


# ─────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str = "CohereLabs/tiny-aya-base"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    # Left-padding for batched generation
    tokenizer.padding_side = "left"
    print(f"Model loaded (device_map=auto)")
    return model, tokenizer


def _get_device(model) -> torch.device:
    return next(model.parameters()).device


# ─────────────────────────────────────────────────────────
# Batched log-likelihood scoring for MC tasks
# ─────────────────────────────────────────────────────────

def build_mc_context(question: str, options: list[str]) -> str:
    ctx = f"Question: {question}\n"
    for i, opt in enumerate(options):
        ctx += f"{OPTION_LETTERS[i]}. {opt}\n"
    ctx += "Answer:"
    return ctx


@torch.inference_mode()
def score_mc_batch(model, tokenizer, batch_contexts: list[str],
                   batch_options: list[list[str]]) -> list[int]:
    """
    Batched MC scoring. For each question, create all (context + option)
    sequences, pad them into one big batch, forward once, then score
    each option's tokens by their average log-likelihood.

    Memory-efficient: computes log_softmax only at needed positions
    instead of over the full (batch, seq_len, vocab) tensor.
    """
    device = _get_device(model)
    pad_id = tokenizer.pad_token_id

    # Flatten all (question, option) pairs
    all_seqs = []  # (q_idx, o_idx, full_ids, n_ctx)
    for q_idx, (context, options) in enumerate(zip(batch_contexts, batch_options)):
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        n_ctx = len(context_ids)
        for o_idx, option_text in enumerate(options):
            full_ids = tokenizer.encode(
                context + " " + option_text, add_special_tokens=False
            )
            all_seqs.append((q_idx, o_idx, full_ids, n_ctx))

    if not all_seqs:
        return [0] * len(batch_contexts)

    # Pad and create tensors (left-padded)
    max_len = min(max(len(s[2]) for s in all_seqs), 2048)

    padded_ids = []
    attention_masks = []
    for _, _, full_ids, _ in all_seqs:
        ids = full_ids[:max_len]
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(padded_ids, device=device)
    attention_mask = torch.tensor(attention_masks, device=device)

    # Single batched forward pass — keep logits in bf16
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Score each option — compute log_softmax only at positions we need
    # This avoids materializing the full (batch, seq_len, vocab) float32 tensor
    scores = defaultdict(list)
    for seq_idx, (q_idx, o_idx, full_ids, n_ctx) in enumerate(all_seqs):
        ids = full_ids[:max_len]
        pad_len = max_len - len(ids)
        option_len = len(ids) - n_ctx
        if option_len <= 0:
            scores[q_idx].append((o_idx, float("-inf")))
            continue

        total_lp = 0.0
        for j in range(option_len):
            pos = pad_len + n_ctx - 1 + j
            token_id = ids[n_ctx + j]
            if pos < max_len:
                # log_softmax on a single (vocab,) vector — cheap
                lp = F.log_softmax(logits[seq_idx, pos].float(), dim=-1)
                total_lp += lp[token_id].item()

        avg_lp = total_lp / option_len
        scores[q_idx].append((o_idx, avg_lp))

    # Free logits memory
    del logits
    torch.cuda.empty_cache()

    # Pick best per question
    results = []
    for q_idx in range(len(batch_contexts)):
        if q_idx in scores and scores[q_idx]:
            best = max(scores[q_idx], key=lambda x: x[1])
            results.append(best[0])
        else:
            results.append(0)

    return results


@torch.inference_mode()
def score_mc_single(model, tokenizer, context: str, options: list[str]) -> int:
    """Fallback: score one question at a time."""
    return score_mc_batch(model, tokenizer, [context], [options])[0]


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int = 32) -> list[str]:
    """Batched greedy generation."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(_get_device(model)) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    answers = []
    for i in range(len(prompts)):
        gen_ids = outputs[i][input_len:]
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ans = ans.split("\n")[0].strip()
        answers.append(ans)
    return answers


# ─────────────────────────────────────────────────────────
# Benchmark: XMMMU
# ─────────────────────────────────────────────────────────

def parse_xmmmu_options(options_raw) -> list[str]:
    if isinstance(options_raw, list):
        return options_raw
    if isinstance(options_raw, str):
        try:
            opts = ast.literal_eval(options_raw)
            if isinstance(opts, list):
                return [str(o) for o in opts]
        except (ValueError, SyntaxError):
            pass
        opts = [o.strip() for o in re.split(r"[\n]", options_raw) if o.strip()]
        return opts
    return [str(options_raw)]


def eval_xmmmu(model, tokenizer, max_samples=None, batch_size=8):
    print("\n" + "=" * 60)
    print("XMMMU (neulab/PangeaBench-xmmmu) — MC log-likelihood batched")
    print("=" * 60)

    languages = ["ar", "fr", "hi", "id", "ja", "pt", "en"]
    all_results = []
    lang_stats = {}

    for lang in languages:
        try:
            ds = load_dataset("neulab/PangeaBench-xmmmu", split=lang)
        except Exception as e:
            print(f"  Skipping lang={lang}: {e}")
            continue

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        correct = 0
        total = 0

        samples_data = []
        for sample in ds:
            question = sample["question"]
            options_raw = sample.get("options", "")
            answer_gt = str(sample.get("answer", "")).strip().upper()
            options = parse_xmmmu_options(options_raw)
            if not options:
                continue
            context = build_mc_context(question, options)
            samples_data.append((question, options, context, answer_gt))

        n_batches = (len(samples_data) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(samples_data), batch_size),
                                desc=f"XMMMU-{lang}", total=n_batches):
            batch = samples_data[batch_start:batch_start + batch_size]
            batch_contexts = [s[2] for s in batch]
            batch_options = [s[1] for s in batch]

            try:
                pred_indices = score_mc_batch(
                    model, tokenizer, batch_contexts, batch_options)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                pred_indices = [
                    score_mc_single(model, tokenizer, ctx, opts)
                    for ctx, opts in zip(batch_contexts, batch_options)
                ]

            for i, (question, options, context, answer_gt) in enumerate(batch):
                pred_idx = pred_indices[i]
                pred_letter = OPTION_LETTERS[pred_idx] if pred_idx < len(OPTION_LETTERS) else "?"
                is_correct = (pred_letter == answer_gt)
                total += 1
                if is_correct:
                    correct += 1
                all_results.append({
                    "lang": lang,
                    "question": question[:100],
                    "predicted": pred_letter,
                    "ground_truth": answer_gt,
                    "correct": is_correct,
                })

        acc = correct / total if total > 0 else 0.0
        lang_stats[lang] = {
            "correct": correct, "total": total,
            "accuracy": round(acc * 100, 2),
        }
        print(f"  {lang}: {correct}/{total} = {acc*100:.2f}%")

    total_correct = sum(s["correct"] for s in lang_stats.values())
    total_count = sum(s["total"] for s in lang_stats.values())
    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    summary = {
        "benchmark": "XMMMU",
        "overall_accuracy": round(overall_acc * 100, 2),
        "total_correct": total_correct,
        "total_samples": total_count,
        "per_language": lang_stats,
    }
    print(f"  Overall: {total_correct}/{total_count} = {overall_acc*100:.2f}%")
    return summary, all_results


# ─────────────────────────────────────────────────────────
# Benchmark: Kaleidoscope
# ─────────────────────────────────────────────────────────

def eval_kaleidoscope(model, tokenizer, max_samples=None, batch_size=8):
    print("\n" + "=" * 60)
    print("Kaleidoscope (CohereLabs/kaleidoscope) — MC log-likelihood batched")
    print("=" * 60)

    ds = load_dataset("CohereLabs/kaleidoscope", split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    all_results = []
    lang_stats = defaultdict(lambda: {
        "correct": 0, "total": 0,
        "text_only_correct": 0, "text_only_total": 0,
    })

    samples_data = []
    for sample in ds:
        question = sample["question"]
        options = sample["options"]
        answer_idx = sample["answer"]
        lang = sample.get("language", "unknown")
        needs_image = sample.get("image_information", "") == "essential"
        gt_letter = OPTION_LETTERS[answer_idx] if 0 <= answer_idx < len(options) else "?"
        context = build_mc_context(question, options)
        samples_data.append((question, options, context, gt_letter, lang, needs_image))

    n_batches = (len(samples_data) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(samples_data), batch_size),
                            desc="Kaleidoscope", total=n_batches):
        batch = samples_data[batch_start:batch_start + batch_size]
        batch_contexts = [s[2] for s in batch]
        batch_options = [s[1] for s in batch]

        try:
            pred_indices = score_mc_batch(
                model, tokenizer, batch_contexts, batch_options)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            pred_indices = [
                score_mc_single(model, tokenizer, ctx, opts)
                for ctx, opts in zip(batch_contexts, batch_options)
            ]

        for i, (question, options, context, gt_letter, lang, needs_image) in enumerate(batch):
            pred_idx = pred_indices[i]
            pred_letter = OPTION_LETTERS[pred_idx] if pred_idx < len(options) else "?"
            is_correct = (pred_letter == gt_letter)

            lang_stats[lang]["total"] += 1
            if is_correct:
                lang_stats[lang]["correct"] += 1
            if not needs_image:
                lang_stats[lang]["text_only_total"] += 1
                if is_correct:
                    lang_stats[lang]["text_only_correct"] += 1

            all_results.append({
                "lang": lang,
                "question": question[:100],
                "predicted": pred_letter,
                "ground_truth": gt_letter,
                "correct": is_correct,
                "needs_image": needs_image,
            })

    total_correct = sum(s["correct"] for s in lang_stats.values())
    total_count = sum(s["total"] for s in lang_stats.values())
    text_only_correct = sum(s["text_only_correct"] for s in lang_stats.values())
    text_only_total = sum(s["text_only_total"] for s in lang_stats.values())
    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    text_only_acc = text_only_correct / text_only_total if text_only_total > 0 else 0.0

    per_lang = {}
    for lang, s in sorted(lang_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        to_acc = s["text_only_correct"] / s["text_only_total"] if s["text_only_total"] > 0 else 0.0
        per_lang[lang] = {
            "correct": s["correct"], "total": s["total"],
            "accuracy": round(acc * 100, 2),
            "text_only_correct": s["text_only_correct"],
            "text_only_total": s["text_only_total"],
            "text_only_accuracy": round(to_acc * 100, 2),
        }
        print(f"  {lang}: {s['correct']}/{s['total']} = {acc*100:.2f}%  "
              f"(text-only: {s['text_only_correct']}/{s['text_only_total']} = {to_acc*100:.2f}%)")

    summary = {
        "benchmark": "Kaleidoscope",
        "overall_accuracy": round(overall_acc * 100, 2),
        "text_only_accuracy": round(text_only_acc * 100, 2),
        "total_correct": total_correct,
        "total_samples": total_count,
        "text_only_correct": text_only_correct,
        "text_only_samples": text_only_total,
        "per_language": per_lang,
    }
    print(f"  Overall: {total_correct}/{total_count} = {overall_acc*100:.2f}%")
    print(f"  Text-only: {text_only_correct}/{text_only_total} = {text_only_acc*100:.2f}%")
    return summary, all_results


# ─────────────────────────────────────────────────────────
# Benchmark: MaXM
# ─────────────────────────────────────────────────────────

def eval_maxm(model, tokenizer, max_samples=None, batch_size=16):
    print("\n" + "=" * 60)
    print("MaXM (floschne/maxm) — Open-ended QA batched")
    print("=" * 60)

    languages = ["en", "fr", "hi", "iw", "ro", "th", "zh"]
    all_results = []
    lang_stats = {}

    for lang in languages:
        try:
            ds = load_dataset("floschne/maxm", split=lang)
        except Exception as e:
            print(f"  Skipping lang={lang}: {e}")
            continue

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        correct = 0
        total = 0

        samples_data = []
        for sample in ds:
            question = sample["question"]
            references = sample.get("processed_answers") or sample.get("answers", [])
            if isinstance(references, str):
                references = [references]
            prompt = f"Q: {question}\nA:"
            samples_data.append((question, references, prompt))

        n_batches = (len(samples_data) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(samples_data), batch_size),
                                desc=f"MaXM-{lang}", total=n_batches):
            batch = samples_data[batch_start:batch_start + batch_size]
            prompts = [s[2] for s in batch]

            try:
                answers = generate_batch(model, tokenizer, prompts, max_new_tokens=32)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                answers = []
                for p in prompts:
                    ans = generate_batch(model, tokenizer, [p], max_new_tokens=32)[0]
                    answers.append(ans)

            for i, (question, references, prompt) in enumerate(batch):
                raw_answer = answers[i]
                is_correct = relaxed_match(raw_answer, references)
                total += 1
                if is_correct:
                    correct += 1
                all_results.append({
                    "lang": lang,
                    "question": question[:100],
                    "predicted": raw_answer,
                    "references": references[:5],
                    "correct": is_correct,
                })

        acc = correct / total if total > 0 else 0.0
        lang_stats[lang] = {
            "correct": correct, "total": total,
            "accuracy": round(acc * 100, 2),
        }
        print(f"  {lang}: {correct}/{total} = {acc*100:.2f}%")

    total_correct = sum(s["correct"] for s in lang_stats.values())
    total_count = sum(s["total"] for s in lang_stats.values())
    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    summary = {
        "benchmark": "MaXM",
        "overall_accuracy": round(overall_acc * 100, 2),
        "total_correct": total_correct,
        "total_samples": total_count,
        "per_language": lang_stats,
    }
    print(f"  Overall: {total_correct}/{total_count} = {overall_acc*100:.2f}%")
    return summary, all_results


# ─────────────────────────────────────────────────────────
# Benchmark: MTVQA
# ─────────────────────────────────────────────────────────

def eval_mtvqa(model, tokenizer, max_samples=None, batch_size=16):
    print("\n" + "=" * 60)
    print("MTVQA (ByteDance/MTVQA) — Open-ended QA batched")
    print("=" * 60)

    ds = load_dataset("ByteDance/MTVQA", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    all_results = []
    lang_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # Flatten QA pairs
    qa_items = []
    for sample in ds:
        lang = sample.get("lang", "unknown")
        qa_pairs_raw = sample.get("qa_pairs", "[]")
        try:
            qa_pairs = (ast.literal_eval(qa_pairs_raw)
                        if isinstance(qa_pairs_raw, str) else qa_pairs_raw)
        except (ValueError, SyntaxError):
            qa_pairs = []
        if not isinstance(qa_pairs, list):
            continue
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question", "")
            answer_gt = qa.get("answer", "")
            if not question or not answer_gt:
                continue
            prompt = f"Q: {question}\nA:"
            qa_items.append((lang, question, answer_gt, prompt))

    print(f"  Total QA pairs: {len(qa_items)}")

    n_batches = (len(qa_items) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(qa_items), batch_size),
                            desc="MTVQA", total=n_batches):
        batch = qa_items[batch_start:batch_start + batch_size]
        prompts = [item[3] for item in batch]

        try:
            answers = generate_batch(model, tokenizer, prompts, max_new_tokens=32)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            answers = []
            for p in prompts:
                ans = generate_batch(model, tokenizer, [p], max_new_tokens=32)[0]
                answers.append(ans)

        for i, (lang, question, answer_gt, prompt) in enumerate(batch):
            raw_answer = answers[i]
            is_correct = relaxed_match(raw_answer, [answer_gt])
            lang_stats[lang]["total"] += 1
            if is_correct:
                lang_stats[lang]["correct"] += 1
            all_results.append({
                "lang": lang,
                "question": question[:100],
                "predicted": raw_answer,
                "ground_truth": answer_gt,
                "correct": is_correct,
            })

    per_lang = {}
    for lang, s in sorted(lang_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        per_lang[lang] = {
            "correct": s["correct"], "total": s["total"],
            "accuracy": round(acc * 100, 2),
        }
        print(f"  {lang}: {s['correct']}/{s['total']} = {acc*100:.2f}%")

    total_correct = sum(s["correct"] for s in lang_stats.values())
    total_count = sum(s["total"] for s in lang_stats.values())
    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    summary = {
        "benchmark": "MTVQA",
        "overall_accuracy": round(overall_acc * 100, 2),
        "total_correct": total_correct,
        "total_samples": total_count,
        "per_language": per_lang,
    }
    print(f"  Overall: {total_correct}/{total_count} = {overall_acc*100:.2f}%")
    return summary, all_results


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

BENCHMARKS = {
    "xmmmu": eval_xmmmu,
    "kaleidoscope": eval_kaleidoscope,
    "maxm": eval_maxm,
    "mtvqa": eval_mtvqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tiny-aya-base on VLM benchmarks (text-only, batched)")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Which benchmarks to run (default: all four)",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split/language (for debugging). "
                             "If not provided, evaluates on the FULL dataset.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    parser.add_argument("--output-dir", type=str,
                        default="vision_stuff/evals/results",
                        help="Output directory for results")
    parser.add_argument("--model-name", type=str,
                        default="CohereLabs/tiny-aya-base",
                        help="HuggingFace model name")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    all_summaries = {}
    start_time = time.time()

    for bench_name in args.benchmarks:
        bench_start = time.time()
        eval_fn = BENCHMARKS[bench_name]
        summary, results = eval_fn(
            model, tokenizer, args.max_samples, args.batch_size)
        bench_time = time.time() - bench_start
        summary["time_seconds"] = round(bench_time, 1)
        all_summaries[bench_name] = summary

        results_path = output_dir / f"{bench_name}_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to {results_path}")

        # Save running summary after each benchmark
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total time: {total_time:.1f}s\n")

    for bench_name, summary in all_summaries.items():
        if "overall_accuracy" in summary:
            print(f"  {bench_name:20s}  Accuracy: {summary['overall_accuracy']:6.2f}%  "
                  f"({summary['total_samples']} samples, {summary['time_seconds']:.1f}s)")
        else:
            print(f"  {bench_name:20s}  {summary['total_samples']} predictions "
                  f"({summary['time_seconds']:.1f}s)")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
