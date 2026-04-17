"""Evaluate TinyAyaVision on xMMMU.

Loads TrishanuDas/tayavision-instruct-665k (pre-merged model) and runs
xMMMU across all 7 languages using the official MMMU parsing and scoring logic.

Usage:
    python evaluation/eval_xmmmu_checkpoint.py
    python evaluation/eval_xmmmu_checkpoint.py --model TrishanuDas/tayavision-instruct-665k
    python evaluation/eval_xmmmu_checkpoint.py --languages en --limit 100
    python evaluation/eval_xmmmu_checkpoint.py --output-dir evaluation/results/checkpoint
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from tqdm import tqdm

from config.model_config import TinyAyaVisionConfig
from evaluation.tasks.xmmmu.utils import (
    get_multi_choice_info,
    parse_multi_choice_response,
    parse_open_response,
    eval_multi_choice,
    eval_open,
    MULTI_CHOICE_PROMPT,
    OPEN_ENDED_PROMPT,
    _parse_options,
    _format_options,
)
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from src.processing import TinyAyaVisionProcessor


LANGUAGES = ["ar", "en", "fr", "hi", "id", "ja", "pt"]
HF_REPO = "TrishanuDas/tayavision-instruct-665k"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(repo: str, device: torch.device):
    """Load pre-merged TinyAyaVision model from HF Hub."""
    print(f"Loading model from {repo}...")
    config = TinyAyaVisionConfig.from_pretrained(repo)
    model = TinyAyaVisionForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Build processor from config — avoids tokenizer_config.json issues in the uploaded repo
    processor = TinyAyaVisionProcessor(config=config)
    model.setup_tokenizer(processor.tokenizer)
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_answer(model, processor, images, prompt_text: str, device: torch.device, max_new_tokens: int = 64) -> str:
    """Run a single forward + generate pass."""
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]
                if images else
                [{"type": "text", "text": prompt_text}]
            ),
        }
    ]
    text = processor.apply_chat_template(
        messages,
        images=images if images else None,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    prompt_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    ).strip()
    return response


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_xmmmu_language(model, processor, lang: str, device: torch.device, limit=None):
    """Evaluate xMMMU for one language. Returns per-sample results."""
    ds = load_dataset("neulab/PangeaBench-xmmmu", split=lang)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    results = []
    correct = 0

    for sample in tqdm(ds, desc=f"xMMMU-{lang}", total=len(ds)):
        # Extract images
        images = []
        for i in range(1, 8):
            img = sample.get(f"image_{i}")
            if img is not None:
                images.append(img.convert("RGB"))

        # Build prompt — strip <image N> markers since images are passed separately
        question = sample["question"]
        for i in range(1, 8):
            question = question.replace(f"<image {i}>", "").strip()
        options = _parse_options(sample)
        options_str = _format_options(options)
        if sample.get("question_type") == "multiple-choice":
            prompt_text = f"{question}\n{options_str}\n{MULTI_CHOICE_PROMPT}"
        else:
            prompt_text = f"{question}\n{OPEN_ENDED_PROMPT}"

        response = generate_answer(model, processor, images, prompt_text, device)

        gold = sample["answer"].strip()
        question_type = sample.get("question_type", "multiple-choice")

        if question_type == "multiple-choice":
            options = ast.literal_eval(sample["options"].replace("\n", " "))
            index2ans, all_choices = get_multi_choice_info(options)
            parsed_pred = parse_multi_choice_response(response, all_choices, index2ans)
            is_correct = eval_multi_choice(gold, parsed_pred)
        else:
            parsed_pred = parse_open_response(response)
            is_correct = eval_open(gold, parsed_pred)

        if is_correct:
            correct += 1

        results.append({
            "lang": lang,
            "id": sample.get("id", ""),
            "question_type": question_type,
            "question": sample["question"][:100],
            "gold": gold,
            "response": response,
            "parsed_pred": parsed_pred if isinstance(parsed_pred, str) else str(parsed_pred),
            "correct": is_correct,
        })

    acc = correct / len(results) if results else 0.0
    print(f"  {lang}: {correct}/{len(results)} = {acc*100:.2f}%")
    return results, {"correct": correct, "total": len(results), "accuracy": round(acc * 100, 2)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyAyaVision on xMMMU")
    parser.add_argument("--model", type=str, default=HF_REPO)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    parser.add_argument("--limit", type=int, default=None, help="Max samples per language")
    parser.add_argument("--output-dir", type=str, default="evaluation/results/checkpoint_xmmmu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, device)

    lang_stats = {}
    start = time.time()

    for lang in args.languages:
        results, stats = eval_xmmmu_language(model, processor, lang, device, args.limit)
        lang_stats[lang] = stats

        with open(output_dir / f"samples_xmmmu_{lang}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_correct = sum(s["correct"] for s in lang_stats.values())
    total_samples = sum(s["total"] for s in lang_stats.values())
    overall_acc = total_correct / total_samples if total_samples else 0.0
    elapsed = round(time.time() - start, 1)

    summary = {
        "benchmark": "xMMMU",
        "model": args.model,
        "overall_accuracy": round(overall_acc * 100, 2),
        "total_correct": total_correct,
        "total_samples": total_samples,
        "per_language": lang_stats,
        "time_seconds": elapsed,
    }

    print(f"\nOverall: {total_correct}/{total_samples} = {overall_acc*100:.2f}%  ({elapsed}s)")

    with open(output_dir / "xmmmu_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_dir}/xmmmu_results.json")


if __name__ == "__main__":
    main()