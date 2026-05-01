"""
Ablation analysis: English-only IFT vs. multilingual IFT on CVQA.

Reads per-sample JSONL files produced by run_eval.py and generates:
  - Per-language accuracy bar chart across comparison models
  - Delta bar chart versus the English-only baseline, sorted by gap
  - Summary table printed to stdout

Usage:
    python evaluation/ablation_cvqa.py \
        --multilingual <path/to/samples.jsonl> \
        --english-only <path/to/samples.jsonl> \
        [--qwen <path/to/samples.jsonl>] \
        [--model "Label=<path/to/samples.jsonl>"] \
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
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LATEST_RESULTS = [
    (
        "TayaVision Instruct 665K",
        "evaluation/results/TrishanuDas__tayavision-instruct-665k/cvqa_chunks/samples.jsonl",
    ),
    (
        "SmolVLM2 2.2B",
        "evaluation/results/HuggingFaceTB__SmolVLM2-2.2B-Instruct/cvqa_chunks/samples.jsonl",
    ),
]


@dataclass(frozen=True)
class ModelScores:
    label: str
    scores: dict[str, list[float]]

    @property
    def accuracies(self) -> dict[str, float]:
        return per_language_accuracy(self.scores)

    @property
    def counts(self) -> dict[str, int]:
        return {language: len(values) for language, values in self.scores.items()}


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


def print_table(models: list[ModelScores], min_samples: int) -> list[dict]:
    accuracies = {model.label: model.accuracies for model in models}
    counts = {model.label: model.counts for model in models}
    baseline = models[1].label
    primary = models[0].label
    languages = sorted(
        set().union(*(model.scores for model in models)),
        key=lambda language: accuracies[primary].get(language, 0),
        reverse=True,
    )

    rows = []
    for language in languages:
        if any(counts[model.label].get(language, 0) < min_samples for model in models):
            continue
        if any(language not in accuracies[model.label] for model in models):
            continue
        rows.append(
            {
                "language": language,
                "scores": {model.label: accuracies[model.label][language] for model in models},
                "counts": {model.label: counts[model.label][language] for model in models},
            }
        )

    col = 24
    score_cols = "".join(f"  {model.label[:18]:>18}" for model in models)
    delta_cols = "".join(f"  {'Δ ' + model.label[:15]:>18}" for model in models if model.label != baseline)
    print(f"\n{'Language':<{col}}{score_cols}{delta_cols}  {'n':>8}")
    print("-" * (col + 2 + 20 * len(models) + 20 * (len(models) - 1) + 10))
    for row in rows:
        score_cells = "".join(f"  {row['scores'][model.label] * 100:>17.1f}%" for model in models)
        baseline_score = row["scores"][baseline]
        delta_cells = "".join(
            f"  {(row['scores'][model.label] - baseline_score) * 100:>+17.1f}%"
            for model in models
            if model.label != baseline
        )
        sample_counts = "/".join(str(row["counts"][model.label]) for model in models)
        print(f"{row['language']:<{col}}{score_cells}{delta_cells}  {sample_counts:>8}")

    print("-" * (col + 2 + 20 * len(models) + 20 * (len(models) - 1) + 10))
    overall = {}
    for model in models:
        total = sum(model.counts.values())
        overall[model.label] = sum(model.accuracies[l] * model.counts[l] for l in model.accuracies) / total
    score_cells = "".join(f"  {overall[model.label] * 100:>17.1f}%" for model in models)
    baseline_score = overall[baseline]
    delta_cells = "".join(
        f"  {(overall[model.label] - baseline_score) * 100:>+17.1f}%" for model in models if model.label != baseline
    )
    print(f"{'OVERALL':<{col}}{score_cells}{delta_cells}")
    return rows


def plot_side_by_side(rows: list[dict], models: list[ModelScores], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows_sorted = sorted(rows, key=lambda row: row["scores"][models[0].label], reverse=True)
    langs = [row["language"] for row in rows_sorted]

    x = np.arange(len(langs))
    width = min(0.8 / len(models), 0.28)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#937860"]
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(12, len(langs) * 0.55), 6))
    for i, model in enumerate(models):
        scores = [row["scores"][model.label] * 100 for row in rows_sorted]
        ax.bar(x + offsets[i], scores, width, label=model.label, color=colors[i % len(colors)])

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CVQA Per-Language Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.axhline(25, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (25%)")
    ax.legend()

    fig.tight_layout()
    out = output_dir / "cvqa_per_language.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.close(fig)


def plot_delta(rows: list[dict], models: list[ModelScores], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    baseline = models[1].label
    comparison_models = [model for model in models if model.label != baseline]
    rows_sorted = sorted(rows, key=lambda row: row["scores"][models[0].label] - row["scores"][baseline], reverse=True)
    langs = [row["language"] for row in rows_sorted]
    fig, ax = plt.subplots(figsize=(max(12, len(langs) * 0.55), 5))
    x = np.arange(len(langs))
    width = min(0.8 / len(comparison_models), 0.38)
    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52", "#937860"]
    offsets = (np.arange(len(comparison_models)) - (len(comparison_models) - 1) / 2) * width
    for i, model in enumerate(comparison_models):
        deltas = [(row["scores"][model.label] - row["scores"][baseline]) * 100 for row in rows_sorted]
        ax.bar(x + offsets[i], deltas, width, label=f"{model.label} - {baseline}", color=colors[i % len(colors)])
    ax.legend()

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Δ Accuracy vs {baseline} (%)")
    ax.set_title("CVQA Accuracy Delta by Language")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    out = output_dir / "cvqa_delta.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def parse_labeled_sample_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, path


def add_model(models: list[tuple[str, str]], label: str, path: str) -> None:
    resolved_paths = {str(Path(existing_path).expanduser().resolve()) for _label, existing_path in models}
    resolved_path = str(Path(path).expanduser().resolve())
    if resolved_path not in resolved_paths:
        models.append((label, path))


def main():
    parser = argparse.ArgumentParser(description="CVQA ablation: multilingual vs. english-only IFT")
    parser.add_argument("--multilingual", required=True, help="Path to multilingual model samples JSONL")
    parser.add_argument("--english-only", required=True, help="Path to english-only model samples JSONL")
    parser.add_argument("--qwen", default=None, help="Optional path to Qwen model samples JSONL")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        type=parse_labeled_sample_arg,
        help='Additional comparison model as "Label=path/to/samples.jsonl". Can be repeated.',
    )
    parser.add_argument(
        "--no-default-latest",
        action="store_true",
        help="Do not auto-include the latest local TayaVision Instruct and SmolVLM2 CVQA results when present.",
    )
    parser.add_argument("--output-dir", default="evaluation/ablation_figures")
    parser.add_argument("--min-samples", type=int, default=10, help="Min samples per language to include")
    args = parser.parse_args()

    model_paths = [
        ("Multilingual IFT", args.multilingual),
        ("English-only IFT", args.english_only),
    ]
    if args.qwen:
        add_model(model_paths, "Qwen3-VL-4B-Instruct", args.qwen)
    if not args.no_default_latest:
        for label, path in DEFAULT_LATEST_RESULTS:
            if Path(path).exists():
                add_model(model_paths, label, path)
    for label, path in args.model:
        add_model(model_paths, label, path)

    models = []
    for label, path in model_paths:
        print(f"Loading {label} samples from: {path}")
        models.append(ModelScores(label=label, scores=load_samples(path)))

    rows = print_table(models, args.min_samples)

    if not rows:
        print("No languages passed the min-samples threshold.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_side_by_side(rows, models, output_dir)
        plot_delta(rows, models, output_dir)
    except ImportError:
        print("\nmatplotlib not available — skipping figures. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
