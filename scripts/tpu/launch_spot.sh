#!/usr/bin/env bash
# launch_spot.sh -- TRC_PROFILE wrapper around launch_qr.sh with SPOT=1 pinned.
#
# TRC v6e and v5e capacity is spot-only, so this is the normal entry point.
#
#   TRC_PROFILE=v6e-8-eu bash scripts/tpu/launch_spot.sh
#   DRY_RUN=1 TRC_PROFILE=v6e-64-ue1d bash scripts/tpu/launch_spot.sh
#
# Profiles (authoritative table: docs/tpu/tpu-trc-allocation.md):
#
#   v6e-8-eu      v6e-8         europe-west4-a   1 host   co-located   <- DEFAULT
#   v6e-64-ew4a   v6e-64        europe-west4-a   8 hosts  co-located
#   v5e-64-ew4b   v5litepod-64  europe-west4-b  16 hosts  co-located
#   v6e-64-ue1d   v6e-64        us-east1-d       8 hosts  CROSS-REGION
#   v5e-64-uc1a   v5litepod-64  us-central1-a   16 hosts  CROSS-REGION
#   v4-32-uc2b    v4-32         us-central2-b    4 hosts  CROSS-REGION
#
# One bucket, gs://tayavision-eu. The three CROSS-REGION profiles still work;
# they pay egress on every checkpoint write. This script warns, it does not
# block -- see docs/tpu/tpu-capacity-log.md 1 for when that is the right call.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"
load_env_file

if ! profile_spec "$TRC_PROFILE" >/dev/null; then
  log "ERROR: unknown TRC_PROFILE '$TRC_PROFILE'"
  log "Valid: v6e-8-eu v6e-64-ew4a v5e-64-ew4b v6e-64-ue1d v5e-64-uc1a v4-32-uc2b"
  exit 1
fi

export SPOT=1
exec bash "$HERE/launch_qr.sh"
