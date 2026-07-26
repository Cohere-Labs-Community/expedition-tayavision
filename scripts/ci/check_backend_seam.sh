#!/usr/bin/env bash
# check_backend_seam.sh -- enforce the TPU/GPU separation seam.
#
# WHY THIS EXISTS
# ---------------
# The repo is dual-backend. Shared code in src/, models/, pipeline/, and config/
# must import cleanly on a CPU or GPU box that has no torch_xla installed. The
# ONLY module allowed to `import torch_xla` at module scope is
# src/backend/tpu_backend.py.
#
# A *module-level* torch_xla import anywhere else drags libtpu into the import
# graph. That breaks every Modal/GPU run and all 129 tests -- and it breaks them
# at import time, so the failure lands nowhere near its cause.
# models/__init__.py matters most: it registers the HF Auto classes at import and
# is pulled in by every eval script.
#
# Lazy (in-function, indented) `import torch_xla` inside a TPU-only code path is
# fine -- it only fires when actually running on TPU. This check flags column-0
# (module-level) imports only.
#
# Usage:  bash scripts/ci/check_backend_seam.sh
# Exit 0 = seam holds; exit 1 = a module-level torch_xla import leaked.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TREES=(src models pipeline config evaluation)
ALLOWED="src/backend/tpu_backend.py"

hits="$(grep -rnE '^(import torch_xla|from torch_xla)' "${TREES[@]}" 2>/dev/null \
        | grep -v "^${ALLOWED}:" || true)"

if [ -n "$hits" ]; then
  echo "FAIL: module-level torch_xla import outside ${ALLOWED} --" >&2
  echo "      move it into src/backend/tpu_backend.py, or make it a lazy" >&2
  echo "      in-function import guarded by the TPU code path." >&2
  echo "$hits" >&2
  exit 1
fi

# The allowed module must actually exist, or this check silently passes forever
# on a repo where the seam was deleted.
if [ ! -f "$ALLOWED" ]; then
  echo "FAIL: $ALLOWED is missing -- the seam has no TPU implementation." >&2
  exit 1
fi

echo "backend-seam-ok (torch_xla confined to ${ALLOWED}; trees: ${TREES[*]})"
