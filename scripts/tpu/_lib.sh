#!/usr/bin/env bash
# _lib.sh -- shared helpers for the TPU scripts. SOURCE THIS, do not execute it.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
#
# Adapted from llm-architectures/scripts/tpu/_lib.sh, itself ported from
# tinyaya-stage2-scale. See .claude/orchestration/README.md for provenance.

# ---------------------------------------------------------------------------
# Repo root, regardless of where the caller ran from.
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT

# ---------------------------------------------------------------------------
# Defaults. Precedence: shell env > repo-root .env > these.
# ---------------------------------------------------------------------------
: "${PROJECT_ID:=ml-pipelines-315702}"
: "${TRC_PROFILE:=v6e-8-eu}"
: "${BUCKET:=gs://tayavision-eu}"
: "${CODE_PREFIX:=code}"
: "${TMUX_SESSION:=train}"
: "${REPO_DIR:=/home/\$USER/expedition-tayavision}"
: "${TAYAVISION_ENTRYPOINT:=scripts/tpu/tpu_smoke.py}"
: "${TAYAVISION_HYDRA_OVERRIDES:=}"
# TAYAVISION_RESUME is deliberately NOT defaulted here. `auto` resumes whatever
# ran last, which is right on the boot/recycle path (same QR metadata, so the
# same overrides by construction) and wrong on a human redeploy, where the
# overrides change between invocations and resuming would load the previous
# run's optimizer state and step count into a different config -- silently.
# So each caller sets its own default: launch_qr.sh and startup_script.sh use
# `auto`, deploy_tarball.sh uses `off`. Every reference must use `${VAR:-...}`,
# since callers run under `set -u`.
: "${TPU_STRATEGY:=replicated}"
: "${MAX_RESUBMITS:=20}"
: "${POLL_SECONDS:=300}"
: "${COOLDOWN_SECONDS:=120}"
: "${HEARTBEAT_SECONDS:=7200}"

# ---------------------------------------------------------------------------
# load_env_file -- read KEY=VALUE without clobbering anything already set.
#
# Deliberately does NOT overwrite an exported shell var: that is what makes
# `FOO=bar bash script.sh` beat the .env file.
# ---------------------------------------------------------------------------
load_env_file() {
  local f="${1:-$REPO_ROOT/.env}"
  [ -f "$f" ] || return 0
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    val="${line#*=}"
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    # strip one layer of surrounding quotes
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    [ -n "$key" ] || continue
    if [ -z "${!key:-}" ]; then
      export "$key=$val"
    fi
  done < "$f"
}

# ---------------------------------------------------------------------------
# notify -- push an event to ntfy.
#
# A NO-OP when NTFY_TOPIC is unset, and it NEVER fails its caller. That is what
# lets every call site stay unconditional: a first run with no .env works, it is
# just deaf. See .claude/orchestration/playbook/event-taxonomy.md.
# ---------------------------------------------------------------------------
notify() {
  local msg="$1"
  local title="${2:-tayavision}"
  [ -n "${NTFY_TOPIC:-}" ] || return 0
  curl -fsS -m 10 \
    -H "Title: $title" \
    -d "$msg" \
    "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
  return 0
}

# ---------------------------------------------------------------------------
# log -- timestamped line to stdout.
# ---------------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

# ---------------------------------------------------------------------------
# bucket_for_profile -- Tiny Aya Vision uses ONE bucket, gs://tayavision-eu.
#
# So this does not map profile -> bucket; it reports whether the profile's zone
# is co-located with the single bucket, and warns when it is not. A cross-region
# profile still works -- it just pays egress on every checkpoint write and every
# code-tarball pull. That is a deliberate operator choice, not something a
# script should make silently or block outright.
# ---------------------------------------------------------------------------
bucket_region_ok() {
  local zone="$1"
  case "$zone" in
    europe-west4-*) return 0 ;;
    *) return 1 ;;
  esac
}

warn_if_cross_region() {
  local zone="$1"
  if ! bucket_region_ok "$zone"; then
    log "WARNING: zone '$zone' is NOT co-located with $BUCKET (europe-west4)."
    log "         Every checkpoint write and code pull crosses regions and pays egress."
    log "         See docs/tpu/tpu-trc-allocation.md 'Bucket co-location'."
    log "         Either accept it for a short run, or stand up a region-paired"
    log "         bucket and point SAVE_CKPT_DIR at it."
  fi
}

# ---------------------------------------------------------------------------
# profile_spec -- TRC_PROFILE -> "ACCEL_TYPE ZONE RUNTIME NODE_ID QR_NAME HOSTS"
#
# Table is authoritative in docs/tpu/tpu-trc-allocation.md; keep them in sync.
# ---------------------------------------------------------------------------
profile_spec() {
  case "${1:-$TRC_PROFILE}" in
    v6e-8-eu)     echo "v6e-8         europe-west4-a v2-alpha-tpuv6e      tayavision-v6e8  tayavision-v6e8-qr  1" ;;
    # v6e-16 host count MEASURED 2026-07-25: 4 network endpoints, i.e. 4 chips
    # per host, not the 8 the public topology tables imply. The others are
    # derived from that ratio and remain unverified -- NUM_HOSTS is cosmetic
    # (fan-out uses --worker=all), so a wrong value misreports but never breaks.
    v6e-16-ew4a)  echo "v6e-16        europe-west4-a v2-alpha-tpuv6e      tayavision-v6e16 tayavision-v6e16-qr 4" ;;
    v6e-32-ew4a)  echo "v6e-32        europe-west4-a v2-alpha-tpuv6e      tayavision-v6e32 tayavision-v6e32-qr 8" ;;
    v6e-64-ew4a)  echo "v6e-64        europe-west4-a v2-alpha-tpuv6e      tayavision-v6e64 tayavision-v6e64-qr 16" ;;
    v5e-64-ew4b)  echo "v5litepod-64  europe-west4-b v2-alpha-tpuv5-lite  tayavision-v5e64 tayavision-v5e64-qr 16" ;;
    v6e-64-ue1d)  echo "v6e-64        us-east1-d     v2-alpha-tpuv6e      tayavision-v6e64 tayavision-v6e64-qr 8" ;;
    v5e-64-uc1a)  echo "v5litepod-64  us-central1-a  v2-alpha-tpuv5-lite  tayavision-v5e64 tayavision-v5e64-qr 16" ;;
    v4-32-uc2b)   echo "v4-32         us-central2-b  tpu-ubuntu2204-base  tayavision-v4-32 tayavision-v4-32-qr 4" ;;
    *) return 1 ;;
  esac
}

# Populate ACCEL_TYPE/ZONE/RUNTIME/NODE_ID/QR_NAME/NUM_HOSTS from TRC_PROFILE,
# without clobbering anything the caller already set.
resolve_profile() {
  local spec
  if ! spec="$(profile_spec "$TRC_PROFILE")"; then
    log "ERROR: unknown TRC_PROFILE '$TRC_PROFILE'."
    log "       EU (co-located): v6e-8-eu v6e-16-ew4a v6e-32-ew4a v6e-64-ew4a v5e-64-ew4b"
    log "       US (cross-region, pays egress): v6e-64-ue1d v5e-64-uc1a v4-32-uc2b"
    return 1
  fi
  # shellcheck disable=SC2086
  set -- $spec
  : "${ACCEL_TYPE:=$1}"
  : "${ZONE:=$2}"
  : "${RUNTIME:=$3}"
  : "${NODE_ID:=$4}"
  : "${QR_NAME:=$5}"
  : "${NUM_HOSTS:=$6}"
  export ACCEL_TYPE ZONE RUNTIME NODE_ID QR_NAME NUM_HOSTS
}

# ---------------------------------------------------------------------------
# make_code_tarball -- tar the working tree INCLUDING the gitignored .env.
#
# The .env inclusion is the entire reason there is no git clone in this flow:
# CohereLabs/tiny-aya-* is a GATED repo, and HF_TOKEN has to reach the VM.
# ---------------------------------------------------------------------------
make_code_tarball() {
  local out="${1:-/tmp/tayavision-code.tar.gz}"
  tar -czf "$out" -C "$REPO_ROOT" \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='data' \
    --exclude='outputs' \
    --exclude='checkpoints' \
    --exclude='wandb' \
    --exclude='.modal' \
    --exclude='.ruff_cache' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.inductor_cache' \
    . 2>/dev/null
  printf '%s\n' "$out"
}
