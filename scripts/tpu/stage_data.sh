#!/usr/bin/env bash
# stage_data.sh -- download and extract a training corpus ON the TPU VM.
#
#   bash scripts/tpu/stage_data.sh                 # from the workstation, fans out
#   bash scripts/tpu/stage_data.sh --local         # ON a worker, does the work
#
# Runs detached in tmux `stage` on each worker. That is not decoration: a
# ~13 GB download takes long enough that a workstation `gcloud ssh` will be
# interrupted, and a killed ssh client does NOT kill the remote command -- it
# orphans it. Detached + a marker file means an interrupted invocation is
# resumable rather than corrupt.
#
# Idempotent: a completed stage writes `.staged`, and re-running is a no-op.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# --local: the worker-side half.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--local" ]; then
  DATA_DIR="${TAYAVISION_DATA_DIR:-/root/data/llava-pretrain}"
  REPO_DIR="${REPO_DIR:-/root/expedition-tayavision}"
  MARKER="$DATA_DIR/.staged"

  exec > >(tee -a /tmp/stage_data.log) 2>&1
  echo "[$(date -Is)] stage_data begin -> $DATA_DIR"

  if [ -f "$MARKER" ]; then
    echo "[$(date -Is)] already staged (marker present); nothing to do"
    exit 0
  fi

  # ~13 GB zip + ~13 GB extracted at peak. Refuse early rather than filling the
  # root disk and taking the training process down with it.
  avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if [ "${avail_gb:-0}" -lt 35 ]; then
    echo "[$(date -Is)] FATAL: only ${avail_gb}G free on /, need ~35G peak"
    exit 1
  fi
  echo "[$(date -Is)] ${avail_gb}G free, proceeding"

  mkdir -p "$DATA_DIR"
  cd "$REPO_DIR" || exit 1

  export PATH="/root/.local/bin:$PATH"
  # HF_TOKEN comes from the tarball's .env. LLaVA-Pretrain is public, but the
  # same loader is used for gated corpora, so keep the env consistent.
  set -a; [ -f "$REPO_DIR/.env" ] && . "$REPO_DIR/.env"; set +a

  # Validate against the contract the DATASET uses, not against a guessed
  # layout. `AlignmentDataset.__getitem__` does `self.data_dir / item["image"]`,
  # and the JSON's image field is shard-relative ("00207/002078413.jpg"), so the
  # zip's shard dirs land directly under DATA_DIR. There is no `images/`
  # subdirectory, despite what download_llava_pretrain.py prints.
  validate() {
    local json="$DATA_DIR/blip_laion_cc_sbu_558k.json"
    [ -s "$json" ] || { echo "  no JSON at $json"; return 1; }
    local first
    first=$(/root/.local/bin/uv run --no-sync python -c "
import json,sys
d=json.load(open('$json'))
print(d[0]['image'] if d and 'image' in d[0] else '')" 2>/dev/null)
    [ -n "$first" ] || { echo "  JSON has no image field"; return 1; }
    [ -f "$DATA_DIR/$first" ] || { echo "  first image missing: $DATA_DIR/$first"; return 1; }
    local n
    n=$(find "$DATA_DIR" -name '*.jpg' -type f 2>/dev/null | head -20000 | wc -l)
    [ "$n" -ge 10000 ] || { echo "  only $n jpgs found"; return 1; }
    echo "  ok: JSON + $first resolves + ${n}+ jpgs"
    return 0
  }

  # Check BEFORE downloading. download_llava_pretrain.py guards on an
  # `images/` dir that this layout never creates, so an unguarded rerun would
  # happily re-fetch 13 GB over a corpus that is already complete.
  echo "[$(date -Is)] checking for an existing corpus..."
  if validate; then
    echo "[$(date -Is)] corpus already present; skipping download"
  else
    echo "[$(date -Is)] not present, downloading"
    /root/.local/bin/uv run --no-sync python scripts/download_llava_pretrain.py \
      --output-dir "$DATA_DIR" || {
      echo "[$(date -Is)] download failed"
      exit 1
    }
    if ! validate; then
      echo "[$(date -Is)] FATAL: corpus still invalid after download"
      exit 1
    fi
  fi

  date -Is > "$MARKER"
  # `n_img` used to be set by the old inline check; validate() owns counting now.
  # Under `set -u` the stale reference aborted the script AFTER the marker was
  # written, so it looked like success with an error printed underneath.
  echo "[$(date -Is)] staged: $(du -sh "$DATA_DIR" | cut -f1)"
  echo "[$(date -Is)] stage_data complete"
  exit 0
fi

# ---------------------------------------------------------------------------
# Workstation side: fan out, detached.
# ---------------------------------------------------------------------------
source "$HERE/_lib.sh"
load_env_file
resolve_profile

: "${TAYAVISION_DATA_DIR:=/root/data/llava-pretrain}"

log "staging data on $NODE_ID ($NUM_HOSTS worker(s)) -> $TAYAVISION_DATA_DIR"
log "  ~13 GB per worker; runs detached in tmux 'stage'"
log "  follow with: ops.sh stage-logs"

gcloud compute tpus tpu-vm ssh "$NODE_ID" \
  --zone="$ZONE" --project="$PROJECT_ID" --worker=all --quiet \
  --command="sudo tmux kill-session -t stage 2>/dev/null; \
             sudo tmux new-session -d -s stage \
             'TAYAVISION_DATA_DIR=$TAYAVISION_DATA_DIR bash /root/expedition-tayavision/scripts/tpu/stage_data.sh --local'; \
             echo launched"

notify "data staging started on $NODE_ID" "tayavision"
log "launched. This takes a while; poll with:"
log "  bash scripts/tpu/ops.sh stage-logs"
