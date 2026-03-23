"""
Compare evaluation results across all models in evaluation/results/.

Usage:
    python evaluation/compare_results.py
    python evaluation/compare_results.py --results-dir evaluation/results
    python evaluation/compare_results.py --tasks cvqa_blind xmmmu
"""

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path, tasks: list[str]) -> dict[str, dict[str, float | None]]:
    """Return {model_slug: {task: score}} for all models found in results_dir."""
    data: dict[str, dict[str, float | None]] = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.replace("__", "/")
        data[model_name] = {}
        for task in tasks:
            result_file = model_dir / f"{task}_results.json"
            if not result_file.exists():
                data[model_name][task] = None
                continue
            with open(result_file) as f:
                results = json.load(f)
            # Pull the primary metric from the task results
            task_results = results.get(task, {})
            score = _extract_score(task_results)
            data[model_name][task] = score

    return data


def _extract_score(task_results: dict) -> float | None:
    """Extract the primary metric score from a task result dict."""
    for key, value in task_results.items():
        if isinstance(value, (int, float)) and not key.startswith("alias"):
            return round(float(value) * 100, 2)
    return None


def print_table(data: dict[str, dict[str, float | None]], tasks: list[str]) -> None:
    col_width = max(len(m) for m in data) + 2
    task_width = 14

    header = f"{'Model':<{col_width}}" + "".join(f"{t:>{task_width}}" for t in tasks)
    print(header)
    print("-" * len(header))

    for model, scores in data.items():
        row = f"{model:<{col_width}}"
        for task in tasks:
            score = scores.get(task)
            cell = f"{score:.2f}%" if score is not None else "N/A"
            row += f"{cell:>{task_width}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across models.")
    parser.add_argument("--results-dir", type=str, default="evaluation/results")
    parser.add_argument("--tasks", type=str, nargs="+", default=["cvqa_blind", "xmmmu"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    data = load_results(results_dir, args.tasks)
    if not data:
        print("No results found.")
        return

    print_table(data, args.tasks)


if __name__ == "__main__":
    main()
