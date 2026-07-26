#!/usr/bin/env bash
# qr_watch.sh -- the QR babysitter (tier T3).
#
# Runs on the WORKSTATION, in detached tmux:
#   tmux new -d -s qrwatch-tayavision 'bash scripts/tpu/qr_watch.sh'
#
# The session name is suffixed on purpose: this workstation also runs a plain
# `qrwatch` for llm-architectures, and `tmux new -s qrwatch` against an existing
# session attaches to the wrong one, leaving this project's QR unwatched.
#
# Polls the queued resource; on SUSPENDING/SUSPENDED/FAILED it captures
# forensics, checks for a quota-class abort, CHECKS CHECKPOINT DURABILITY,
# deletes the QR, and resubmits the identical launch env.
#
# ONE RESUBMITTER PER QR, EVER. If the QR is absent this exits rather than
# racing a human who is mid-recreate -- two resubmitters produce duplicate nodes
# and split-brain rendezvous.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"
load_env_file
resolve_profile

LOG=/tmp/qr_watch_tayavision.log
exec > >(tee -a "$LOG") 2>&1

: "${SAVE_CKPT_DIR:=}"

resubmits=0
last_state=""
last_heartbeat=$(date +%s)

log "qr_watch start: qr=$QR_NAME zone=$ZONE profile=$TRC_PROFILE"
log "  budget=$MAX_RESUBMITS poll=${POLL_SECONDS}s cooldown=${COOLDOWN_SECONDS}s"

# --- the durability gate ----------------------------------------------------
# This is the divergence from BOTH source repos. tinyaya never auto-recycled;
# llm-architectures always did. Here: only when checkpoints survive the node.
#
# Recycling with node-local checkpoints restarts from step 0 AND re-pays ~14 min
# of cold-boot XLA compile, up to 20 times, silently. That is not self-healing,
# it is a capacity incinerator.
DURABLE=0
if [ -n "$SAVE_CKPT_DIR" ] && [ "${SAVE_CKPT_DIR#gs://}" != "$SAVE_CKPT_DIR" ]; then
  DURABLE=1
  log "  checkpoints: $SAVE_CKPT_DIR (durable) -- auto-recycle ENABLED"
else
  log "  checkpoints: ${SAVE_CKPT_DIR:-<unset>} (NOT durable) -- auto-recycle DISABLED"
  log "  Preemption will be reported, not repaired. Set SAVE_CKPT_DIR to a gs:// path."
fi

while true; do
  state="$(gcloud compute tpus queued-resources describe "$QR_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" \
            --format='value(state.state)' 2>/dev/null || echo MISSING)"

  if [ "$state" != "$last_state" ]; then
    log "QR state: ${last_state:-<start>} -> $state"
    [ -n "$last_state" ] && notify "QR state changed: $last_state -> $state" "tayavision"
    last_state="$state"
  fi

  case "$state" in
    MISSING)
      log "QR is absent. Exiting rather than racing a human mid-recreate."
      notify "qr_watch exiting: QR $QR_NAME is absent" "tayavision"
      exit 0
      ;;

    SUSPENDING|SUSPENDED|FAILED)
      log "QR in terminal state $state -- capturing forensics"
      forensics="/tmp/qr_forensics_${QR_NAME}_$(date -u +%Y%m%d-%H%M%S).json"
      gcloud compute tpus queued-resources describe "$QR_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" --format=json > "$forensics" 2>/dev/null || true
      log "  forensics -> $forensics"

      # quota-class abort: resubmitting into a wall cannot succeed
      if grep -qiE 'quota|IN_USE_ADDRESSES|exhausted|PerProjectPerZone' "$forensics" 2>/dev/null; then
        log "QUOTA-CLASS FAILURE -- resubmitting cannot help. Stopping."
        log "  On this grant the usual wall is IN_USE_ADDRESSES (regional cap 8),"
        log "  not TPU chips. See docs/tpu/tpu-capacity-log.md 7.1."
        notify "quota abort on $QR_NAME -- needs a human" "tayavision STOPPED"
        exit 1
      fi

      if [ "$DURABLE" -ne 1 ]; then
        log "RECYCLE BLOCKED: SAVE_CKPT_DIR is not a gs:// URI."
        log "  Recycling would restart from step 0 and re-pay ~14 min of compile."
        notify "recycle BLOCKED — SAVE_CKPT_DIR not gs:// — $QR_NAME preempted, progress lost" \
               "tayavision STOPPED"
        exit 1
      fi

      if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
        log "budget exhausted ($resubmits/$MAX_RESUBMITS). Stopping."
        notify "budget exhausted ($MAX_RESUBMITS) on $QR_NAME" "tayavision STOPPED"
        exit 1
      fi

      log "deleting dead QR"
      gcloud compute tpus queued-resources delete "$QR_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" --quiet >/dev/null 2>&1 || true

      sleep "$COOLDOWN_SECONDS"

      resubmits=$((resubmits + 1))
      log "resubmitting ($resubmits/$MAX_RESUBMITS)"
      if bash "$HERE/launch_spot.sh"; then
        notify "resubmitted ($resubmits/$MAX_RESUBMITS): $QR_NAME" "tayavision"
        last_state=""
      else
        log "resubmit failed"
        notify "resubmit FAILED ($resubmits/$MAX_RESUBMITS): $QR_NAME" "tayavision STOPPED"
        exit 1
      fi
      ;;
  esac

  # heartbeat -- a silent watcher is indistinguishable from a dead one
  now=$(date +%s)
  if [ $((now - last_heartbeat)) -ge "$HEARTBEAT_SECONDS" ]; then
    notify "heartbeat: $QR_NAME is $state, $resubmits/$MAX_RESUBMITS resubmits used" "tayavision"
    last_heartbeat=$now
  fi

  sleep "$POLL_SECONDS"
done
