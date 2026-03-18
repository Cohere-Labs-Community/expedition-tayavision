"""
uv run python evaluation/run_eval.py

Args:
    [--task]: cvqa_blind, ...
    [--backend]: vllm | hf
    [--batch-size]: auto | 1 (vllm | hf)
    [--limit]: int  (num samples for quick tests)
    [--output-dir]: str
    [--model-name]: str

"""

import argparse
import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="lm-eval runner for Tiny Aya Vision benchmarks.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    parser.add_argument("--log-samples", action="store_true", help="Log per-question results")
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template")
    args = parser.parse_args()

    logger.info(f"Starting evaluation for task: {args.task}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"=========================================")

    model_args = f"pretrained={args.model_name},dtype=bfloat16,trust_remote_code=True"
    if args.backend == "vllm":
        model_args += ",tensor_parallel_size=1"
    
    cmd = [
        "lm_eval",
        "--model", args.backend,
        "--model_args", model_args,
        "--task", args.task,
        "--batch_size", args.batch_size,
        "--include_path", "evaluation/tasks",
        "--output_path", args.output_dir
    ]

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    
    if args.log_samples:
        cmd.append("--log_samples")
    
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with exit code: {e.returncode}")
        exit(e.returncode)


if __name__ == "__main__":
    main()
