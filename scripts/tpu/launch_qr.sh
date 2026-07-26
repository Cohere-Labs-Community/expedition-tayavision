#!/usr/bin/env bash
# launch_qr.sh -- create the Queued Resource that provisions a slice and runs
# startup_script.sh on every host.
#
# Normally invoked via launch_spot.sh (TRC v6e/v5e is spot-only). Direct use is
# for the on-demand v4 quota.
#
#   ZONE=... ACCEL_TYPE=... bash scripts/tpu/launch_qr.sh
#   DRY_RUN=1 TRC_PROFILE=v6e-8-eu bash scripts/tpu/launch_qr.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
load_env_file
resolve_profile

: "${SPOT:=0}"
: "${DRY_RUN:=0}"
: "${WANDB_RUN_NAME:=}"
: "${SAVE_CKPT_DIR:=}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
CODE_URI="${BUCKET}/${CODE_PREFIX}/${STAMP}.tar.gz"
LATEST_URI="${BUCKET}/${CODE_PREFIX}/latest.tar.gz"

log "profile=$TRC_PROFILE accel=$ACCEL_TYPE zone=$ZONE hosts=$NUM_HOSTS"
log "node=$NODE_ID qr=$QR_NAME runtime=$RUNTIME spot=$SPOT"
log "entrypoint=$TAYAVISION_ENTRYPOINT strategy=$TPU_STRATEGY"
warn_if_cross_region "$ZONE"

if [ "$NUM_HOSTS" -gt 1 ]; then
  log ""
  log "NOTE: $NUM_HOSTS-host slice -- needs $NUM_HOSTS external IPs and a PJRT rendezvous"
  log "      across every host. Check regional IN_USE_ADDRESSES headroom first:"
  log "      gcloud compute regions describe \${ZONE%-*} --format='value(quotas)'"
  log "      A sibling project hit a cap of 8 here once; europe-west4 measured"
  log "      16/64 on 2026-07-25, so it is headroom, not a wall. Verify, do not assume."
fi

if [ -z "${SAVE_CKPT_DIR}" ]; then
  log ""
  log "NOTE: SAVE_CKPT_DIR is unset. qr_watch.sh will NOT auto-recycle this slice"
  log "      on preemption -- recycling without durable checkpoints restarts from"
  log "      step 0. Set it to a gs:// path for anything long-running."
fi

# --- upload the code tarball ------------------------------------------------
if [ "$DRY_RUN" != "1" ]; then
  log "tarring working tree (including the gitignored .env)"
  TARBALL="$(make_code_tarball)"
  log "uploading -> $CODE_URI"
  gcloud storage cp "$TARBALL" "$CODE_URI" --project="$PROJECT_ID"
  gcloud storage cp "$CODE_URI" "$LATEST_URI" --project="$PROJECT_ID"
  rm -f "$TARBALL"
fi

# --- metadata carried by the QR, so a recycled node self-heals --------------
STARTUP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/startup_script.sh"
META="startup-script=${STARTUP}"
META="${META},tayavision-code-uri=${CODE_URI}"
META="${META},tayavision-entrypoint=${TAYAVISION_ENTRYPOINT}"
META="${META},tayavision-hydra-overrides=${TAYAVISION_HYDRA_OVERRIDES}"
# `auto` here, not off: this metadata is replayed verbatim by every recycle, so
# the overrides cannot drift and resuming the previous run is exactly the
# self-heal SPEC.md section 5 describes.
META="${META},tayavision-resume=${TAYAVISION_RESUME:-auto}"
META="${META},tayavision-tpu-strategy=${TPU_STRATEGY}"
META="${META},tayavision-save-ckpt-dir=${SAVE_CKPT_DIR}"
META="${META},tayavision-wandb-run-name=${WANDB_RUN_NAME}"

CMD=(gcloud compute tpus queued-resources create "$QR_NAME"
     --node-id="$NODE_ID"
     --project="$PROJECT_ID"
     --zone="$ZONE"
     --accelerator-type="$ACCEL_TYPE"
     --runtime-version="$RUNTIME"
     --metadata-from-file=startup-script="$STARTUP"
     --metadata="${META#startup-script=${STARTUP},}")
[ "$SPOT" = "1" ] && CMD+=(--spot)

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN -- would run:"
  printf '  %q ' "${CMD[@]}"; echo
  exit 0
fi

log "creating queued resource"
"${CMD[@]}"

notify "QR submitted: $QR_NAME ($TRC_PROFILE, $ZONE)" "tayavision"
log "submitted. Watch with: bash scripts/tpu/ops.sh status"
log "Keep it alive with:   tmux new -d -s qrwatch-tayavision 'bash scripts/tpu/qr_watch.sh'"
