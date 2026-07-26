#!/usr/bin/env bash
# train_launcher.sh -- the canonical on-worker launcher.
#
# Runs ON the TPU VM, inside tmux `train`. Expects the TAYAVISION_* env to be
# exported already (startup_script.sh and deploy_tarball.sh both do that).
#
# This script's ONLY job is to run the entrypoint and emit the two markers every
# watcher greps for:
#
#     launching <entrypoint> tag=<RUN_TAG>
#     <entrypoint> exited with status <rc>
#
# Change those strings and you break qr_watch, the watchdog agent, and the
# diagnosis table simultaneously.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"
load_env_file

: "${TAYAVISION_RUN_TAG:=$(date -u +%Y%m%d-%H%M%S)-$$}"
ENTRYPOINT="${TAYAVISION_ENTRYPOINT:-scripts/tpu/tpu_smoke.py}"

cd "$REPO_ROOT"

# --- fail loudly and immediately on a typo'd entrypoint ---------------------
# Without this you discover the mistake 20 minutes into a spot acquisition, as a
# confusing python traceback buried in a log nobody is tailing yet.
if [ ! -f "$ENTRYPOINT" ]; then
  log "FATAL: entrypoint not found: $ENTRYPOINT"
  log "       TAYAVISION_ENTRYPOINT must be a repo-relative path."
  log "       Default is scripts/tpu/tpu_smoke.py."
  log "       NOTE: pipeline/train_*.py does NOT run on TPU yet --"
  log "             see .claude/orchestration/SPEC.md section 9."
  notify "entrypoint missing: $ENTRYPOINT" "tayavision FAILED"
  exit 2
fi

# --- torch_xla needs libpython on the loader path --------------------------
# _XLAC.so dynamically links libpython3.12.so.1.0, which uv keeps outside the
# default path. Without this, `import torch_xla` dies with ImportError.
if [ -z "${LD_LIBRARY_PATH:-}" ] || ! printf '%s' "$LD_LIBRARY_PATH" | grep -q libpython; then
  LIBPYTHON_DIR="$(find "${UV_PYTHON_ROOT:-$HOME/.local/share/uv/python}" \
                     -name 'libpython3.12.so.1.0' -printf '%h\n' 2>/dev/null | head -1)"
  if [ -n "$LIBPYTHON_DIR" ]; then
    export LD_LIBRARY_PATH="${LIBPYTHON_DIR}:${LD_LIBRARY_PATH:-}"
    log "LD_LIBRARY_PATH += $LIBPYTHON_DIR"
  fi
fi

export PJRT_DEVICE=TPU
export TAYAVISION_TPU=1
export TPU_STRATEGY="${TPU_STRATEGY:-replicated}"

# Route Hydra entrypoints to the TPU W&B project by DEFAULT, not by remembering.
#
# `pipeline/train_*.py` calls wandb.init(project=cfg.wandb.project, ...), and an
# explicit arg beats the WANDB_PROJECT env var. config/config.yaml says
# `tayavision-instruct-sweep`, so a TPU run that forgets the override lands its
# metrics in a GPU RESULTS project. Injecting the override here makes .env the
# single source of truth and removes the footgun.
#
# Only for pipeline/ entrypoints (tpu_smoke.py is argparse, not Hydra), and only
# when the caller has not already said something about wandb.
case "$ENTRYPOINT" in
  pipeline/*)
    if [ -n "${WANDB_PROJECT:-}" ] && \
       ! printf '%s' "${TAYAVISION_HYDRA_OVERRIDES:-}" | grep -q 'wandb\.project'; then
      TAYAVISION_HYDRA_OVERRIDES="${TAYAVISION_HYDRA_OVERRIDES:-} wandb.project=${WANDB_PROJECT}"
      log "  injected wandb.project=${WANDB_PROJECT} from .env"
    fi
    if [ -n "${WANDB_ENTITY:-}" ] && \
       ! printf '%s' "${TAYAVISION_HYDRA_OVERRIDES:-}" | grep -q 'wandb\.entity'; then
      TAYAVISION_HYDRA_OVERRIDES="${TAYAVISION_HYDRA_OVERRIDES:-} wandb.entity=${WANDB_ENTITY}"
      log "  injected wandb.entity=${WANDB_ENTITY} from .env"
    fi
    # train_multilingual.py hardcodes project="tayavision-multilingual" and
    # ignores the Hydra override entirely -- this injection cannot save it.
    # One-line fix, tracked as PLAN P6.
    case "$ENTRYPOINT" in
      *train_multilingual.py)
        log "  WARNING: train_multilingual.py ignores wandb.project and will write to"
        log "           'tayavision-multilingual' (a GPU results project). See PLAN P6." ;;
    esac
    ;;
esac
# W&B: export only when non-empty. An empty WANDB_* export makes wandb.init die
# with "Run ID cannot be empty" -- diagnoser row 18.
[ -n "${WANDB_RUN_NAME:-}" ] && export WANDB_RUN_NAME
[ -n "${WANDB_RUN_ID:-}" ] && export WANDB_RUN_ID

log "launching $ENTRYPOINT tag=$TAYAVISION_RUN_TAG"
# Print what is actually in the environment. This line used to read
# `${TAYAVISION_RESUME:-auto}`, which reported "resume=auto" whenever the
# variable was unset -- while nothing read the variable at all. A banner that
# invents a value it does not pass on is worse than no banner.
log "  strategy=$TPU_STRATEGY resume=${TAYAVISION_RESUME:-off}"
log "  overrides='${TAYAVISION_HYDRA_OVERRIDES:-}'"
log "  ckpt=${SAVE_CKPT_DIR:-<local, NOT durable>}"

UV_BIN="$(command -v uv || echo /root/.local/bin/uv)"

# shellcheck disable=SC2086
"$UV_BIN" run --no-sync python -u "$ENTRYPOINT" ${TAYAVISION_HYDRA_OVERRIDES:-}
rc=$?
# ^ rc MUST be captured on its own line. A $(date) or any command substitution
#   on the same line as $? resets it, and every exit status reads 0. That bug
#   made a sibling project's launcher report success for every fatal traceback.

log "$ENTRYPOINT exited with status $rc"

if [ "$rc" -eq 0 ]; then
  notify "run OK — $(tail -3 /tmp/train.log 2>/dev/null | tr '\n' ' ' | cut -c1-160)" "tayavision"
else
  notify "run FAILED rc=$rc — $(tail -3 /tmp/train.log 2>/dev/null | tr '\n' ' ' | cut -c1-160)" "tayavision FAILED"
fi

exit "$rc"
