#!/bin/bash
# Evaluation entry-point for the Qwen3-backed taya-vision build.
# Usage:
#   bash eval_qwen3.sh                         # uses default model below
#   bash eval_qwen3.sh /path/to/checkpoint     # eval a local Qwen3 checkpoint
#   MODEL=org/name bash eval_qwen3.sh          # override model via env var
#   TASK=xmmmu bash eval_qwen3.sh              # override task via env var

MODEL="${1:-${MODEL:-Qwen/Qwen3-4B-Instruct-2507}}"
TASK="${TASK:-cvqa}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"

uv run evaluation/run_eval.py \
    --task "${TASK}" \
    --model-name "${MODEL}" \
    --backend taya-vision \
    --apply-chat-template \
    --chunk-size "${CHUNK_SIZE}" \
    --output-dir "${OUTPUT_DIR}"
