#!/bin/bash


uv run evaluation/run_eval.py \
    --task cvqa \
    --model-name "TrishanuDas/tayavision-multilingual" \
    --backend tiny-aya-vision \
    --apply-chat-template \
    --chunk-size 100 \
    --output-dir "evaluation/results"
