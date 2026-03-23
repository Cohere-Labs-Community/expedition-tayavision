"""
Run all baseline models on all tasks defined in config/evaluation/baselines.yaml.

Usage:
    python evaluation/run_baselines.py
    python evaluation/run_baselines.py --tasks cvqa_blind
    python evaluation/run_baselines.py --models google/gemma-3-4b-it
    python evaluation/run_baselines.py --limit 10
    python evaluation/run_baselines.py --output-dir evaluation/results
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description="Run all baseline models defined in baselines.yaml.")
    parser.add_argument("--config", type=str, default="config/evaluation/baselines.yaml")
    parser.add_argument("--tasks", type=str, nargs="+", help="Override tasks to run (default: all in config)")
    parser.add_argument("--models", type=str, nargs="+", help="Override models to run (default: all in config)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (for quick tests)")
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tasks = args.tasks or config["tasks"]
    models = config["models"]
    if args.models:
        models = [m for m in models if m["name"] in args.models]

    logger.info(f"Tasks: {tasks}")
    logger.info(f"Models: {[m['name'] for m in models]}")
    logger.info("=========================================")

    failures = []

    for model in models:
        model_name = model["name"]
        slug = model_slug(model_name)
        output_dir = str(Path(args.output_dir) / slug)

        for task in tasks:
            logger.info(f"Running {task} on {model_name}")

            cmd = [
                sys.executable, "evaluation/run_eval.py",
                "--task", task,
                "--model-name", model_name,
                "--backend", model.get("backend", "hf-multimodal"),
                "--dtype", model.get("dtype", "bfloat16"),
                "--output-dir", output_dir,
                "--skip-registration",
                "--log-samples",
            ]

            if not model.get("trust_remote_code", True):
                cmd.append("--no-trust-remote-code")

            if model.get("apply_chat_template", False):
                cmd.append("--apply-chat-template")

            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(f"FAILED: {task} on {model_name}")
                failures.append((model_name, task))
            else:
                logger.info(f"Done: {task} on {model_name}")

    logger.info("=========================================")
    if failures:
        logger.error(f"{len(failures)} run(s) failed:")
        for model_name, task in failures:
            logger.error(f"  {model_name} / {task}")
        sys.exit(1)
    else:
        logger.info("All baseline runs completed successfully.")


if __name__ == "__main__":
    main()
