# 131072 matches Ministral-3B's actual 128K context window. 
# This prevents the truncation that was corrupting the image token count.

MALLOC_TRIM_THRESHOLD_=0 python evaluation/run_eval.py \
    --task cvqa \
    --model-name "mistralai/Ministral-3-3B-Instruct-2512" \
    --backend hf-multimodal \
    --skip-registration \
    --apply-chat-template \
    --chunk-size 100 \
    --output-dir "evaluation/results" \
    --extra-model-args "max_length=131072"