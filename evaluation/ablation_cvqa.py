"""
Ablation analysis: English-only IFT vs. multilingual IFT on CVQA.

Reads per-sample JSONL files produced by run_eval.py and generates:
  - Per-language accuracy bar chart (both models side by side)
  - Delta bar chart (multilingual - english_only), sorted by gap
  - Summary table printed to stdout

Usage:
    python evaluation/ablation_cvqa.py \
        --multilingual <path/to/samples.jsonl> \
        --english-only <path/to/samples.jsonl> \
        [--qwen <path/to/samples.jsonl>] \
        [--output-dir evaluation/ablation_figures]
        [--min-samples 10]

The samples JSONL is produced either at:
  - evaluation/results/<model>/samples_cvqa.jsonl          (single-pass)
  - evaluation/results/<model>/cvqa_chunks/samples.jsonl   (chunked)
"""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path


def load_samples(path: str) -> dict[str, list[float]]:
    """Return {language: [exact_match scores]} from a samples JSONL."""
    scores: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            subset_raw = sample["doc"].get("Subset", "")
            try:
                language, _country = ast.literal_eval(subset_raw)
            except Exception:
                language = subset_raw or "unknown"
            score = sample.get("exact_match")
            if score is not None:
                scores[language].append(float(score))
    return dict(scores)


def per_language_accuracy(scores: dict[str, list[float]]) -> dict[str, float]:
    return {lang: sum(vs) / len(vs) for lang, vs in scores.items()}


def print_table(
    multilingual: dict[str, float],
    english: dict[str, float],
    qwen: dict[str, float] | None,
    min_samples: int,
    ml_counts: dict[str, int],
    en_counts: dict[str, int],
    qwen_counts: dict[str, int] | None,
) -> list[tuple]:
    languages = sorted(
        set(multilingual) | set(english) | (set(qwen) if qwen else set()),
        key=lambda l: multilingual.get(l, 0),
        reverse=True,
    )

    rows = []
    for lang in languages:
        ml = multilingual.get(lang)
        en = english.get(lang)
        ml_n = ml_counts.get(lang, 0)
        en_n = en_counts.get(lang, 0)
        qwen_n = qwen_counts.get(lang, 0) if qwen_counts else 0
        if ml_n < min_samples or en_n < min_samples:
            continue
        if ml is None or en is None:
            continue
        if qwen:
            q = qwen.get(lang)
            if q is None or qwen_n < min_samples:
                continue
            rows.append((lang, ml, en, q, ml - en, q - en))
        else:
            rows.append((lang, ml, en, ml - en))

    col = 24
    if qwen:
        print(
            f"\n{'Language':<{col}}  {'Multilingual':>14}  {'English-only':>14}  {'Qwen3-VL-4B':>14}  "
            f"{'Δ(ML-EN)':>10}  {'Δ(Qwen-EN)':>12}  {'n (ML/EN/QW)':>14}"
        )
        print("-" * (col + 92))
        for lang, ml, en, q, delta_ml_en, delta_q_en in rows:
            ml_n = ml_counts[lang]
            en_n = en_counts[lang]
            q_n = qwen_counts[lang]
            print(
                f"{lang:<{col}}  {ml*100:>13.1f}%  {en*100:>13.1f}%  {q*100:>13.1f}%  "
                f"{delta_ml_en*100:>+9.1f}%  {delta_q_en*100:>+11.1f}%  {ml_n:>4}/{en_n:<4}/{q_n:<4}"
            )
    else:
        print(f"\n{'Language':<{col}}  {'Multilingual':>14}  {'English-only':>14}  {'Delta (ML-EN)':>14}  {'n (ML/EN)':>12}")
        print("-" * (col + 60))
        for lang, ml, en, delta in rows:
            sign = "+" if delta >= 0 else ""
            ml_n = ml_counts[lang]
            en_n = en_counts[lang]
            print(f"{lang:<{col}}  {ml*100:>13.1f}%  {en*100:>13.1f}%  {sign}{delta*100:>12.1f}%  {ml_n:>5}/{en_n:<5}")

    overall_ml = sum(multilingual[l] * ml_counts[l] for l in multilingual) / sum(ml_counts.values())
    overall_en = sum(english[l] * en_counts[l] for l in english) / sum(en_counts.values())
    if qwen and qwen_counts:
        overall_q = sum(qwen[l] * qwen_counts[l] for l in qwen) / sum(qwen_counts.values())
        print("-" * (col + 92))
        print(
            f"{'OVERALL':<{col}}  {overall_ml*100:>13.1f}%  {overall_en*100:>13.1f}%  {overall_q*100:>13.1f}%  "
            f"{(overall_ml-overall_en)*100:>+9.1f}%  {(overall_q-overall_en)*100:>+11.1f}%"
        )
    else:
        print("-" * (col + 60))
        print(f"{'OVERALL':<{col}}  {overall_ml*100:>13.1f}%  {overall_en*100:>13.1f}%  {(overall_ml-overall_en)*100:>+13.1f}%")
    return rows


def plot_side_by_side(
    rows: list[tuple],
    output_dir: Path,
    include_qwen: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    langs = [r[0] for r in rows_sorted]
    ml_scores = [r[1] * 100 for r in rows_sorted]
    en_scores = [r[2] * 100 for r in rows_sorted]
    q_scores = [r[3] * 100 for r in rows_sorted] if include_qwen else None

    x = np.arange(len(langs))
    width = 0.25 if include_qwen else 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(langs) * 0.55), 6))
    if include_qwen:
        ax.bar(x - width, ml_scores, width, label="Multilingual IFT", color="#4C72B0")
        ax.bar(x, en_scores, width, label="English-only IFT", color="#DD8452")
        ax.bar(x + width, q_scores, width, label="Qwen3-VL-4B-Instruct", color="#55A868")
    else:
        ax.bar(x - width / 2, ml_scores, width, label="Multilingual IFT", color="#4C72B0")
        ax.bar(x + width / 2, en_scores, width, label="English-only IFT", color="#DD8452")

    ax.set_ylabel("Accuracy (%)")
    if include_qwen:
        ax.set_title("CVQA Per-Language Accuracy: Multilingual vs. English-only vs. Qwen3-VL-4B")
    else:
        ax.set_title("CVQA Per-Language Accuracy: Multilingual vs. English-only IFT")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(25, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (25%)")
    ax.legend()

    fig.tight_layout()
    out = output_dir / "cvqa_per_language.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.close(fig)


def plot_delta(
    rows: list[tuple],
    output_dir: Path,
    include_qwen: bool,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows_sorted = sorted(rows, key=lambda r: r[3] if not include_qwen else r[4], reverse=True)
    langs = [r[0] for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(max(12, len(langs) * 0.55), 5))
    x = np.arange(len(langs))
    if include_qwen:
        deltas_ml = [r[4] * 100 for r in rows_sorted]
        deltas_qw = [r[5] * 100 for r in rows_sorted]
        width = 0.38
        ax.bar(x - width / 2, deltas_ml, width, label="Multilingual - English-only", color="#4C72B0")
        ax.bar(x + width / 2, deltas_qw, width, label="Qwen - English-only", color="#55A868")
        ax.legend()
    else:
        deltas = [r[3] * 100 for r in rows_sorted]
        colors = ["#4C72B0" if d >= 0 else "#C44E52" for d in deltas]
        ax.bar(x, deltas, color=colors)

    ax.axhline(0, color="black", linewidth=0.8)
    if include_qwen:
        ax.set_ylabel("Δ Accuracy vs English-only (%)")
        ax.set_title("CVQA Accuracy Delta by Language")
    else:
        ax.set_ylabel("Δ Accuracy (Multilingual − English-only) in %")
        ax.set_title("CVQA Accuracy Delta by Language (Multilingual IFT − English-only IFT)")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    out = output_dir / "cvqa_delta.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CVQA ablation: multilingual vs. english-only IFT")
    parser.add_argument("--multilingual", required=True, help="Path to multilingual model samples JSONL")
    parser.add_argument("--english-only", required=True, help="Path to english-only model samples JSONL")
    parser.add_argument("--qwen", default=None, help="Optional path to Qwen model samples JSONL")
    parser.add_argument("--output-dir", default="evaluation/ablation_figures")
    parser.add_argument("--min-samples", type=int, default=10, help="Min samples per language to include")
    args = parser.parse_args()

    print(f"Loading multilingual samples from: {args.multilingual}")
    ml_scores = load_samples(args.multilingual)
    print(f"Loading english-only samples from: {args.english_only}")
    en_scores = load_samples(args.english_only)
    qwen_scores = None
    if args.qwen:
        print(f"Loading qwen samples from: {args.qwen}")
        qwen_scores = load_samples(args.qwen)

    ml_acc = per_language_accuracy(ml_scores)
    en_acc = per_language_accuracy(en_scores)
    qwen_acc = per_language_accuracy(qwen_scores) if qwen_scores else None
    ml_counts = {l: len(v) for l, v in ml_scores.items()}
    en_counts = {l: len(v) for l, v in en_scores.items()}
    qwen_counts = {l: len(v) for l, v in qwen_scores.items()} if qwen_scores else None

    rows = print_table(ml_acc, en_acc, qwen_acc, args.min_samples, ml_counts, en_counts, qwen_counts)

    if not rows:
        print("No languages passed the min-samples threshold.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_side_by_side(rows, output_dir, include_qwen=bool(qwen_scores))
        plot_delta(rows, output_dir, include_qwen=bool(qwen_scores))
    except ImportError:
        print("\nmatplotlib not available — skipping figures. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
