#!/usr/bin/env bash
# setup_gcp.sh -- one-time GCP bootstrap for Tiny Aya Vision TPU work.
#
# Enables the APIs, creates the ONE bucket, and grants the TPU service identity
# access to it. Idempotent: safe to re-run.
#
#   bash scripts/tpu/setup_gcp.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
load_env_file

BUCKET_LOCATION="${BUCKET_LOCATION:-europe-west4}"

log "project=$PROJECT_ID bucket=$BUCKET location=$BUCKET_LOCATION"
log ""
log "Tiny Aya Vision uses ONE bucket. Only the three europe-west4-* profiles are"
log "co-located with it; anything else pays egress. See"
log "docs/tpu/tpu-trc-allocation.md 'Bucket co-location'."
log ""

log "1/3 enabling APIs (tpu, storage, compute)"
gcloud services enable \
  tpu.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com \
  --project="$PROJECT_ID"

log "2/3 creating $BUCKET in $BUCKET_LOCATION (if absent)"
if gcloud storage buckets describe "$BUCKET" --project="$PROJECT_ID" >/dev/null 2>&1; then
  log "    already exists"
else
  gcloud storage buckets create "$BUCKET" \
    --project="$PROJECT_ID" \
    --location="$BUCKET_LOCATION" \
    --uniform-bucket-level-access
  log "    created"
fi

log "3/3 granting objectAdmin to the TPU service identity"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
TPU_SA="service-${PROJECT_NUMBER}@cloud-tpu.iam.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for sa in "$TPU_SA" "$COMPUTE_SA"; do
  gcloud storage buckets add-iam-policy-binding "$BUCKET" \
    --member="serviceAccount:${sa}" \
    --role=roles/storage.objectAdmin \
    --project="$PROJECT_ID" >/dev/null
  log "    granted: $sa"
done

log ""
log "done. Next:"
log "  cp .env.example .env    # then fill HF_TOKEN -- the model repos are GATED"
log "  bash scripts/tpu/ops.sh preflight"
log "  TRC_PROFILE=v6e-8-eu bash scripts/tpu/launch_spot.sh"
