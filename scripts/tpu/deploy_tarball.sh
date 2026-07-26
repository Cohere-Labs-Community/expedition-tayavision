#!/usr/bin/env bash
# deploy_tarball.sh -- the T2 action: push the working tree to a LIVE slice and
# relaunch, without recreating the queued resource.
#
#   bash scripts/tpu/deploy_tarball.sh
#   TAYAVISION_ENTRYPOINT=scripts/tpu/tpu_smoke.py bash scripts/tpu/deploy_tarball.sh
#
# A redeploy here costs a FULL XLA recompile (~14 min) because the persistent
# cache has nondeterministic keys on v6e SPMD and never warm-hits. Batch fixes.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
load_env_file
resolve_profile

: "${DRY_RUN:=0}"
: "${SAVE_CKPT_DIR:=}"
: "${WANDB_RUN_ID:=}"

STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_TAG="${TAYAVISION_RUN_TAG:-deploy-${STAMP}}"
CODE_URI="${BUCKET}/${CODE_PREFIX}/${STAMP}.tar.gz"
REMOTE_DIR="/root/expedition-tayavision"

log "deploying to $NODE_ID ($ZONE, $NUM_HOSTS host(s)) tag=$RUN_TAG"
log "entrypoint=$TAYAVISION_ENTRYPOINT strategy=$TPU_STRATEGY"
warn_if_cross_region "$ZONE"

# --- mid-run safety ---------------------------------------------------------
# A deploy kills `train` on every worker. Without durable checkpoints + resume
# that silently discards progress.
if [ -z "$SAVE_CKPT_DIR" ]; then
  log "WARNING: SAVE_CKPT_DIR unset -- this deploy DISCARDS in-flight progress,"
  log "         and qr_watch.sh will refuse to auto-recycle on the next preemption."
elif [ "${SAVE_CKPT_DIR#gs://}" = "$SAVE_CKPT_DIR" ]; then
  log "WARNING: SAVE_CKPT_DIR is not a gs:// URI ($SAVE_CKPT_DIR)."
  log "         Node-local checkpoints do not survive a recycle. See SPEC.md section 5."
fi
if [ -n "$SAVE_CKPT_DIR" ] && [ -z "$WANDB_RUN_ID" ]; then
  log "NOTE: WANDB_RUN_ID unset -- a resumed run will fork a second W&B entry"
  log "      instead of rejoining."
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN -- would tar, upload to $CODE_URI, and relaunch on --worker=all"
  exit 0
fi

# --- 1. tar + upload (including the gitignored .env) ------------------------
log "1/3 tarring + uploading -> $CODE_URI"
TARBALL="$(make_code_tarball)"
gcloud storage cp "$TARBALL" "$CODE_URI" --project="$PROJECT_ID"
gcloud storage cp "$CODE_URI" "${BUCKET}/${CODE_PREFIX}/latest.tar.gz" --project="$PROJECT_ID"
rm -f "$TARBALL"

# --- 2. build the remote script locally, scp it -----------------------------
# Generated as a file rather than inlined so there is exactly ONE level of shell
# quoting to reason about.
REMOTE_SH=/tmp/tayavision_remote_deploy.sh
cat > "$REMOTE_SH" <<REMOTE
#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/.local/bin:\$PATH"
UV_BIN=/root/.local/bin/uv

mkdir -p "$REMOTE_DIR"
gcloud storage cp "$CODE_URI" /tmp/code.tar.gz || { echo "code tarball not found"; exit 1; }
tar -xzf /tmp/code.tar.gz -C "$REMOTE_DIR"
cd "$REMOTE_DIR"
# See startup_script.sh step 4: torch_xla is not in pyproject; the named CUDA
# index is redirected to CPU wheels and torch_xla is layered on top.
# CPython 3.12 exactly -- pyproject floors at 3.12, libtpu 0.0.21 ceilings at cp312.
export UV_PYTHON=3.12
UV_INDEX="pytorch-cu124=https://download.pytorch.org/whl/cpu" \\
UV_INDEX_STRATEGY=unsafe-best-match "\$UV_BIN" sync || exit 1
# torch_xla <= 2.7 has no cp312/cp313 wheels; 2.8+ does. torchvision is ABI-locked
# to torch and must move in lockstep (2.9 -> 0.24). See startup_script.sh step 4.
"\$UV_BIN" pip install \\
  --extra-index-url https://download.pytorch.org/whl/cpu \\
  --extra-index-url https://storage.googleapis.com/libtpu-releases/index.html \\
  --index-strategy unsafe-best-match \\
  "torch==${TORCH_VERSION:-2.9.*}" \\
  "torchvision==${TORCHVISION_VERSION:-0.24.*}" \\
  "torch_xla[tpu]==${TORCH_XLA_VERSION:-2.9.*}" || exit 1
"\$UV_BIN" run --no-sync python -c "
import torch, torchvision
torch.ops.torchvision.nms
print(f'torch {torch.__version__} / torchvision {torchvision.__version__} ABI ok')" || exit 1

LIBPYTHON_DIR="\$(find /root/.local/share/uv/python -name 'libpython3.12.so.1.0' -printf '%h\n' 2>/dev/null | head -1)"
[ -n "\$LIBPYTHON_DIR" ] && export LD_LIBRARY_PATH="\${LIBPYTHON_DIR}:\${LD_LIBRARY_PATH:-}"

export TAYAVISION_ENTRYPOINT="$TAYAVISION_ENTRYPOINT"
export TAYAVISION_HYDRA_OVERRIDES="$TAYAVISION_HYDRA_OVERRIDES"
export TAYAVISION_RESUME="${TAYAVISION_RESUME:-off}"
export TAYAVISION_RUN_TAG="$RUN_TAG"
export TPU_STRATEGY="$TPU_STRATEGY"
export SAVE_CKPT_DIR="$SAVE_CKPT_DIR"
[ -n "$WANDB_RUN_ID" ] && export WANDB_RUN_ID="$WANDB_RUN_ID"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" \\
  "bash $REMOTE_DIR/scripts/tpu/train_launcher.sh 2>&1 | tee -a /tmp/train.log"
echo "relaunched tmux $TMUX_SESSION tag=$RUN_TAG"
REMOTE

log "2/3 scp remote script to all workers"
gcloud compute tpus tpu-vm scp "$REMOTE_SH" "${NODE_ID}:/tmp/_deploy.sh" \
  --zone="$ZONE" --project="$PROJECT_ID" --worker=all --quiet

log "3/3 running on --worker=all"
gcloud compute tpus tpu-vm ssh "$NODE_ID" \
  --zone="$ZONE" --project="$PROJECT_ID" --worker=all --quiet \
  --command="sudo bash /tmp/_deploy.sh"

rm -f "$REMOTE_SH"
notify "deployed tag=$RUN_TAG to $NUM_HOSTS worker(s)" "tayavision"
log "done. Expect no step line for up to 20 min -- that is XLA compiling, not a hang."
log "Watch: bash scripts/tpu/ops.sh tail-logs"
