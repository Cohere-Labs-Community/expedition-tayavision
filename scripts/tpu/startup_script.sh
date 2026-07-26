#!/usr/bin/env bash
# startup_script.sh -- runs as root on EVERY TPU host, on EVERY boot.
#
# Must be idempotent: a spot preemption reboots the host and re-runs this.
#
# Reads its knobs from GCE metadata, which is what makes boot self-heal work --
# a recycled node re-fetches the same code, the same entrypoint, and the same
# W&B identity with no human action.
set -uo pipefail

exec > >(tee -a /tmp/startup.log) 2>&1
echo "[$(date -Is)] startup_script.sh begin"

md() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}

CODE_URI="$(md tayavision-code-uri)"
ENTRYPOINT="$(md tayavision-entrypoint)"
HYDRA_OVERRIDES="$(md tayavision-hydra-overrides)"
RESUME="$(md tayavision-resume)"
STRATEGY="$(md tayavision-tpu-strategy)"
SAVE_CKPT_DIR="$(md tayavision-save-ckpt-dir)"
WANDB_RUN_NAME="$(md tayavision-wandb-run-name)"

REPO_DIR="${REPO_DIR:-/root/expedition-tayavision}"
TMUX_SESSION="${TMUX_SESSION:-train}"

echo "[$(date -Is)] code=$CODE_URI entrypoint=$ENTRYPOINT strategy=$STRATEGY"

# --- 1. apt, with a retry loop --------------------------------------------
# Two workers out of sixteen used to miss the PJRT rendezvous because dpkg was
# locked at boot by unattended-upgrades. Diagnoser row 4.
for i in $(seq 1 30); do
  if apt-get update -qq >/dev/null 2>&1 && \
     apt-get install -y -qq tmux curl tar >/dev/null 2>&1; then
    echo "[$(date -Is)] apt ok (attempt $i)"
    break
  fi
  echo "[$(date -Is)] apt locked, retry $i/30"
  sleep 10
done

# --- 2. uv -----------------------------------------------------------------
# `which uv` is empty under sudo; always use the absolute path in root contexts.
UV_BIN=/root/.local/bin/uv
if [ ! -x "$UV_BIN" ]; then
  echo "[$(date -Is)] installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
fi
export PATH="/root/.local/bin:$PATH"

# --- 3. code, from the GCS tarball. NEVER a git clone ----------------------
# The tarball carries the gitignored .env, and .env carries HF_TOKEN. The model
# repos (CohereLabs/tiny-aya-*) are GATED: without it the run cannot start.
if [ -n "$CODE_URI" ]; then
  echo "[$(date -Is)] fetching $CODE_URI"
  mkdir -p "$REPO_DIR"
  if gcloud storage cp "$CODE_URI" /tmp/code.tar.gz 2>/dev/null; then
    tar -xzf /tmp/code.tar.gz -C "$REPO_DIR"
    echo "[$(date -Is)] code extracted to $REPO_DIR"
  else
    echo "[$(date -Is)] ERROR: code tarball not found: $CODE_URI"
    exit 1
  fi
else
  echo "[$(date -Is)] ERROR: code tarball not found (no tayavision-code-uri metadata)"
  exit 1
fi

cd "$REPO_DIR"

# --- 4. deps ---------------------------------------------------------------
# torch_xla is deliberately NOT in pyproject.toml. pyproject pins torch to the
# CUDA 12.4 index for all Linux (that is correct for the Modal/GPU path), and a
# TPU VM needs CPU torch + torch_xla + libtpu instead. Rather than fight that
# pin -- or fork the lockfile and risk the Modal path -- we redirect the named
# index at sync time and add torch_xla afterwards. pyproject and uv.lock stay
# untouched, so the GPU path is bit-identical.
# Version choice is forced, not preferred. torch_xla <= 2.7 ships only
# cp39/cp310/cp311 wheels; pyproject's `requires-python = ">=3.12,<3.14"` makes uv
# pick CPython 3.13, so those versions cannot resolve at all. 2.8.0 is the first
# release with cp312/cp313. torch is upgraded to match on the VM only -- the
# lockfile still says 2.6.0 and the Modal path is untouched.
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.9.*}"
TORCH_VERSION="${TORCH_VERSION:-2.9.*}"
# torchvision is ABI-locked to torch and MUST be upgraded in lockstep. The lock
# installs the 2.6-matched build; leaving it there against torch 2.9 fails with
# "operator torchvision::nms does not exist", which then surfaces far from its
# cause as `ModuleNotFoundError: Could not import module 'AutoProcessor'` --
# transformers' lazy importer swallowing the real error. Pairing: torch 2.6->tv
# 0.21, 2.7->0.22, 2.8->0.23, 2.9->0.24.
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.*}"
LIBTPU_INDEX=https://storage.googleapis.com/libtpu-releases/index.html

# CPython 3.12 EXACTLY, and it is pinned by intersection, not preference:
#   pyproject requires-python  >= 3.12   -> floor 3.12
#   libtpu 0.0.21 wheels       <= cp312  -> ceiling 3.12   (torch_xla 2.9 dep)
# Left to itself uv picks 3.13 (the newest the floor allows) and torch_xla
# cannot resolve. Do not "modernise" this to 3.13 until libtpu ships cp313.
export UV_PYTHON="${UV_PYTHON:-3.12}"

# UV_INDEX_STRATEGY is required here, not optional. Overriding a named index via
# UV_INDEX drops its `explicit = true` flag from pyproject, so the CPU wheel
# index becomes a GENERAL index. uv's default first-index strategy then refuses
# to look at PyPI for anything that index also carries, and resolution dies on
# `requests==2.28.1` (which is all download.pytorch.org publishes).
# unsafe-best-match lets uv pick the best version across both. The "unsafe" name
# is about dependency confusion between a private and a public index; here both
# are public and trusted (pypi.org, download.pytorch.org).
echo "[$(date -Is)] uv sync (torch redirected to CPU wheels)"
UV_INDEX="pytorch-cu124=https://download.pytorch.org/whl/cpu" \
UV_INDEX_STRATEGY=unsafe-best-match \
  "$UV_BIN" sync || {
  echo "[$(date -Is)] uv sync failed"
  exit 1
}

echo "[$(date -Is)] installing torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torch_xla[tpu]==$TORCH_XLA_VERSION"
"$UV_BIN" pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url "$LIBTPU_INDEX" \
  --index-strategy unsafe-best-match \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torch_xla[tpu]==${TORCH_XLA_VERSION}" || {
  echo "[$(date -Is)] torch_xla install failed"
  exit 1
}
# Fail loudly here rather than 200 lines into a traceback somewhere else.
"$UV_BIN" run --no-sync python -c "
import torch, torchvision
torch.ops.torchvision.nms  # the op that is missing on an ABI mismatch
print(f'torch {torch.__version__} / torchvision {torchvision.__version__} ABI ok')" || {
  echo "[$(date -Is)] torch/torchvision ABI mismatch"
  exit 1
}

# --- 5. the torch_xla / libpython link -------------------------------------
# torch_xla's _XLAC.so dynamically links libpython3.12.so.1.0, which uv keeps
# outside the default loader path. Without this every `import torch_xla` fails.
# Diagnoser row 1. This is the single most likely first-boot failure.
LIBPYTHON_DIR="$(find /root/.local/share/uv/python -name 'libpython3.12.so.1.0' -printf '%h\n' 2>/dev/null | head -1)"
if [ -n "$LIBPYTHON_DIR" ]; then
  export LD_LIBRARY_PATH="${LIBPYTHON_DIR}:${LD_LIBRARY_PATH:-}"
  echo "[$(date -Is)] LD_LIBRARY_PATH += $LIBPYTHON_DIR"
else
  echo "[$(date -Is)] WARNING: libpython3.12.so.1.0 not found; torch_xla import may fail"
fi

# --- 6. launch in tmux -----------------------------------------------------
export TAYAVISION_ENTRYPOINT="${ENTRYPOINT:-scripts/tpu/tpu_smoke.py}"
export TAYAVISION_HYDRA_OVERRIDES="$HYDRA_OVERRIDES"
export TAYAVISION_RESUME="${RESUME:-auto}"
export TPU_STRATEGY="${STRATEGY:-replicated}"
export SAVE_CKPT_DIR="$SAVE_CKPT_DIR"
export TAYAVISION_RUN_TAG="boot-$(date -u +%Y%m%d-%H%M%S)"
[ -n "$WANDB_RUN_NAME" ] && export WANDB_RUN_NAME

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" \
  "bash $REPO_DIR/scripts/tpu/train_launcher.sh 2>&1 | tee -a /tmp/train.log"

echo "[$(date -Is)] startup_script.sh complete"
