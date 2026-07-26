#!/usr/bin/env bash
# ops.sh -- daily operations against the current slice.
#
#   bash scripts/tpu/ops.sh preflight    # check before spending capacity
#   bash scripts/tpu/ops.sh status       # QR + node state
#   bash scripts/tpu/ops.sh tail-logs    # follow /tmp/train.log on worker 0
#   bash scripts/tpu/ops.sh attach       # tmux attach on worker 0
#   bash scripts/tpu/ops.sh ssh          # shell on worker 0
#   bash scripts/tpu/ops.sh delete       # tear down the QR (stops billing)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_lib.sh"
load_env_file
resolve_profile

CMD="${1:-status}"

case "$CMD" in

  preflight)
    rc=0
    echo "== preflight: $TRC_PROFILE ($ACCEL_TYPE, $ZONE, $NUM_HOSTS host(s)) =="
    echo

    echo "-- gcloud --"
    acct="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)"
    if [ -n "$acct" ]; then echo "   authed: $acct"; else echo "   NOT AUTHED  <- gcloud auth login"; rc=1; fi
    proj="$(gcloud config get-value project 2>/dev/null)"
    echo "   project: ${proj:-<unset>} (scripts use $PROJECT_ID)"

    echo
    echo "-- .env (presence only, never values) --"
    if [ -f "$REPO_ROOT/.env" ]; then
      for k in HF_TOKEN WANDB_API_KEY NTFY_TOPIC WANDB_PROJECT WANDB_ENTITY PROJECT_ID; do
        # `KEY=` with an empty value must NOT report as set. A key that is
        # present-but-blank is the worst case: it looks configured and behaves
        # as absent. That shape already cost a sibling project a day of silently
        # dropped alerts.
        val="$(sed -n "s/^${k}=//p" "$REPO_ROOT/.env" 2>/dev/null | head -1 | tr -d '"'"'"' \t\r')"
        if [ -n "$val" ]; then
          case "$k" in
            WANDB_PROJECT|WANDB_ENTITY|PROJECT_ID) echo "   $k: $val" ;;
            *)                                     echo "   $k: set" ;;
          esac
        else
          case "$k" in
            HF_TOKEN)
              echo "   $k: MISSING/EMPTY  <- FATAL: CohereLabs/tiny-aya-* is a GATED repo"; rc=1 ;;
            WANDB_API_KEY)
              echo "   $k: missing/empty  (run trains but is unobservable)" ;;
            NTFY_TOPIC)
              echo "   $k: missing/empty  (notify() is a silent no-op: fine for a"
              echo "                       foreground smoke, NOT for an overnight run)" ;;
            WANDB_PROJECT)
              echo "   $k: missing/empty  (Hydra runs fall back to config/config.yaml,"
              echo "                       which names a GPU results project)" ;;
            WANDB_ENTITY)
              echo "   $k: missing/empty  (wandb picks your default entity)" ;;
            PROJECT_ID)
              echo "   $k: missing/empty  (defaults to $PROJECT_ID)" ;;
          esac
        fi
      done
    else
      echo "   NO .env AT REPO ROOT  <- cp .env.example .env, then fill HF_TOKEN"
      echo "   The tarball ships .env to the VM; there is no git clone in this flow."
      rc=1
    fi

    echo
    echo "-- entrypoint --"
    if [ -f "$REPO_ROOT/$TAYAVISION_ENTRYPOINT" ]; then
      echo "   $TAYAVISION_ENTRYPOINT: exists"
      case "$TAYAVISION_ENTRYPOINT" in
        pipeline/train_*)
          echo "   WARNING: pipeline/train_*.py is DDP + CUDA and does NOT run on TPU."
          echo "            See .claude/orchestration/SPEC.md section 9."
          rc=1 ;;
      esac
    else
      echo "   $TAYAVISION_ENTRYPOINT: MISSING"; rc=1
    fi

    echo
    echo "-- checkpoints --"
    if [ -z "${SAVE_CKPT_DIR:-}" ]; then
      echo "   SAVE_CKPT_DIR unset -- qr_watch will NOT auto-recycle on preemption"
    elif [ "${SAVE_CKPT_DIR#gs://}" != "$SAVE_CKPT_DIR" ]; then
      echo "   $SAVE_CKPT_DIR (durable, auto-recycle enabled)"
    else
      echo "   $SAVE_CKPT_DIR is NOT gs:// -- auto-recycle disabled"
    fi

    echo
    echo "-- bucket --"
    if gcloud storage ls "$BUCKET" >/dev/null 2>&1; then
      echo "   $BUCKET: reachable"
    else
      echo "   $BUCKET: NOT REACHABLE  <- bash scripts/tpu/setup_gcp.sh"; rc=1
    fi
    warn_if_cross_region "$ZONE"

    echo
    [ "$rc" -eq 0 ] && echo "preflight OK" || echo "preflight FAILED -- fix the above before launching"
    exit "$rc"
    ;;

  status)
    echo "== queued resource: $QR_NAME ($ZONE) =="
    gcloud compute tpus queued-resources describe "$QR_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" \
      --format='table(name.basename(),state.state)' 2>/dev/null \
      || echo "  (no QR named $QR_NAME)"
    echo
    echo "== all queued resources in $ZONE =="
    echo "   (SUSPENDED/FAILED husks still book quota -- delete them)"
    gcloud compute tpus queued-resources list \
      --zone="$ZONE" --project="$PROJECT_ID" \
      --format='table(name.basename(),state.state)' 2>/dev/null || true
    echo
    echo "== node: $NODE_ID =="
    gcloud compute tpus tpu-vm describe "$NODE_ID" \
      --zone="$ZONE" --project="$PROJECT_ID" \
      --format='table(name.basename(),state,health)' 2>/dev/null \
      || echo "  (no node named $NODE_ID)"
    ;;

  stage-logs)
    gcloud compute tpus tpu-vm ssh "$NODE_ID" \
      --zone="$ZONE" --project="$PROJECT_ID" --worker=0 --quiet \
      --command="sudo tail -n 25 /tmp/stage_data.log 2>/dev/null || echo '(no staging log yet)'; \
                 echo '--- marker ---'; \
                 sudo ls -la ${TAYAVISION_DATA_DIR:-/root/data/llava-pretrain}/.staged 2>/dev/null \
                   || echo 'not yet staged'"
    ;;

  tail-logs)
    # tail, never head: a head -N over repeating step lines pushes the exit
    # marker out of range. Diagnoser row 25.
    gcloud compute tpus tpu-vm ssh "$NODE_ID" \
      --zone="$ZONE" --project="$PROJECT_ID" --worker=0 --quiet \
      --command="tail -f -n 60 /tmp/train.log"
    ;;

  attach)
    echo "tmux session '$TMUX_SESSION' is root-owned; using sudo."
    gcloud compute tpus tpu-vm ssh "$NODE_ID" \
      --zone="$ZONE" --project="$PROJECT_ID" --worker=0 \
      -- -t "sudo tmux attach -t $TMUX_SESSION"
    ;;

  ssh)
    gcloud compute tpus tpu-vm ssh "$NODE_ID" \
      --zone="$ZONE" --project="$PROJECT_ID" --worker=0
    ;;

  delete)
    echo "This deletes QR '$QR_NAME' in $ZONE and stops billing."
    read -r -p "Type the profile name ($TRC_PROFILE) to confirm: " confirm
    if [ "$confirm" != "$TRC_PROFILE" ]; then echo "aborted"; exit 1; fi
    # --force is REQUIRED for the normal case. An ACTIVE queued resource owns a
    # node, and without --force the API rejects the delete outright:
    #   code 9: DeleteQueuedResource is not supported when state is ACTIVE
    #           (must be one of [ACCEPTED WAITING_FOR_RESOURCES SUSPENDED FAILED])
    # i.e. the un-forced form only ever worked on husks -- never on a slice you
    # had actually finished using, which is the only time you run this.
    gcloud compute tpus queued-resources delete "$QR_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" --force --quiet
    notify "QR deleted: $QR_NAME" "tayavision"
    ;;

  *)
    echo "usage: bash scripts/tpu/ops.sh {preflight|status|stage-logs|tail-logs|attach|ssh|delete}"
    exit 1
    ;;
esac
