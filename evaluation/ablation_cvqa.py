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
    min_samples: int,
    ml_counts: dict[str, int],
    en_counts: dict[str, int],
) -> list[tuple[str, float, float, float]]:
    languages = sorted(
        set(multilingual) | set(english),
        key=lambda l: multilingual.get(l, 0),
        reverse=True,
    )

    rows = []
    for lang in languages:
        ml = multilingual.get(lang)
        en = english.get(lang)
        ml_n = ml_counts.get(lang, 0)
        en_n = en_counts.get(lang, 0)
        if ml_n < min_samples or en_n < min_samples:
            continue
        if ml is None or en is None:
            continue
        delta = ml - en
        rows.append((lang, ml, en, delta))

    col = 24
    print(f"\n{'Language':<{col}}  {'Multilingual':>14}  {'English-only':>14}  {'Delta (ML-EN)':>14}  {'n (ML/EN)':>12}")
    print("-" * (col + 60))
    for lang, ml, en, delta in rows:
        sign = "+" if delta >= 0 else ""
        ml_n = ml_counts[lang]
        en_n = en_counts[lang]
        print(f"{lang:<{col}}  {ml*100:>13.1f}%  {en*100:>13.1f}%  {sign}{delta*100:>12.1f}%  {ml_n:>5}/{en_n:<5}")

    overall_ml = sum(multilingual[l] * ml_counts[l] for l in multilingual) / sum(ml_counts.values())
    overall_en = sum(english[l] * en_counts[l] for l in english) / sum(en_counts.values())
    print("-" * (col + 60))
    print(f"{'OVERALL':<{col}}  {overall_ml*100:>13.1f}%  {overall_en*100:>13.1f}%  {(overall_ml-overall_en)*100:>+13.1f}%")
    return rows


def plot_side_by_side(
    rows: list[tuple[str, float, float, float]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    langs = [r[0] for r in rows_sorted]
    ml_scores = [r[1] * 100 for r in rows_sorted]
    en_scores = [r[2] * 100 for r in rows_sorted]

    x = np.arange(len(langs))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(langs) * 0.55), 6))
    ax.bar(x - width / 2, ml_scores, width, label="Multilingual IFT", color="#4C72B0")
    ax.bar(x + width / 2, en_scores, width, label="English-only IFT", color="#DD8452")

    ax.set_ylabel("Accuracy (%)")
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
    rows: list[tuple[str, float, float, float]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
    langs = [r[0] for r in rows_sorted]
    deltas = [r[3] * 100 for r in rows_sorted]

    colors = ["#4C72B0" if d >= 0 else "#C44E52" for d in deltas]

    fig, ax = plt.subplots(figsize=(max(12, len(langs) * 0.55), 5))
    ax.bar(np.arange(len(langs)), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Accuracy (Multilingual − English-only) in %")
    ax.set_title("CVQA Accuracy Delta by Language (Multilingual IFT − English-only IFT)")
    ax.set_xticks(np.arange(len(langs)))
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
    parser.add_argument("--output-dir", default="evaluation/ablation_figures")
    parser.add_argument("--min-samples", type=int, default=10, help="Min samples per language to include")
    args = parser.parse_args()

    print(f"Loading multilingual samples from: {args.multilingual}")
    ml_scores = load_samples(args.multilingual)
    print(f"Loading english-only samples from: {args.english_only}")
    en_scores = load_samples(args.english_only)

    ml_acc = per_language_accuracy(ml_scores)
    en_acc = per_language_accuracy(en_scores)
    ml_counts = {l: len(v) for l, v in ml_scores.items()}
    en_counts = {l: len(v) for l, v in en_scores.items()}

    rows = print_table(ml_acc, en_acc, args.min_samples, ml_counts, en_counts)

    if not rows:
        print("No languages passed the min-samples threshold.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_side_by_side(rows, output_dir)
        plot_delta(rows, output_dir)
    except ImportError:
        print("\nmatplotlib not available — skipping figures. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
