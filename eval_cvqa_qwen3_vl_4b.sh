#!/bin/bash


uv run evaluation/run_eval.py \
    --task cvqa \
    --model-name "Qwen/Qwen3-VL-4B-Instruct" \
    --backend hf-multimodal \
    --skip-registration \
    --apply-chat-template \
    --chunk-size 100 \
    --trust-remote-code \
    --output-dir "evaluation/results"
