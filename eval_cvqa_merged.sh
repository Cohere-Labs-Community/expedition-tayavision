#!/bin/bash
set -euo pipefail

# Evaluate merged models uploaded by merge_weights.sh.
# Expected subfolders in the repo: alpha_0p3, alpha_0p4, ...

REPO_ID="bkal01/tayavision-multilingual-merged"
ALPHAS=(0.3 0.4 0.5 0.6 0.7)

for alpha in "${ALPHAS[@]}"; do
  label="${alpha//./p}"
  subfolder="alpha_${label}"

  echo "Evaluating ${REPO_ID} (subfolder: ${subfolder})"

  uv run evaluation/run_eval.py \
    --task cvqa \
    --model-name "${REPO_ID}" \
    --model-subfolder "${subfolder}" \
    --backend tiny-aya-vision \
    --apply-chat-template \
    --chunk-size 100 \
    --output-dir "evaluation/results"
done
