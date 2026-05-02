MALLOC_TRIM_THRESHOLD_=0 python evaluation/run_eval.py \
    --task cvqa \
    --model-name "google/gemma-3-4b-it" \
    --backend hf-multimodal \
    --skip-registration \
    --apply-chat-template \
    --chunk-size 100 \
    --output-dir "evaluation/results"