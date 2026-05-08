#!/bin/bash
set -euo pipefail

uv run scripts/merge_weights.py \
--original CohereLabs/tiny-aya-global \
--finetuned TrishanuDas/tayavision-multilingual \
--alpha 0.3,0.4,0.5,0.6,0.7 \
--output outputs/merged/tayavision_multilingual \
--hub-repo-id bkal01/tayavision-multilingual-merged
