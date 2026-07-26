---
name: tpu-orchestrate
description: Run-control playbook for Tiny Aya Vision TPU runs — how orchestration is layered, where truth lives, and the recovery ladder. Load when operating, debugging, or resuming a TPU run.
---

# TPU run-control (Tiny Aya Vision)

> **No entrypoint under `pipeline/` runs on TPU today.** `train_alignment.py`,
> `train_instruct.py`, and `train_multilingual.py` are DDP + CUDA. Read
> `.claude/orchestration/SPEC.md` §9 before launching anything. The default entrypoint is
> `scripts/tpu/tpu_smoke.py`, which proves provisioning — not training.
>
> The Modal/GPU path is unaffected and remains authoritative for every published result.

## Where to look

| Need | File |
|---|---|
| Surface ownership; which compute path owns a fact | `orchestration/CONTROL_PLANE.md` |
| Run loop + recovery policy + what does not work yet | `orchestration/SPEC.md` |
| What to do at each tier | `orchestration/playbook/tier-definitions.md` |
| Signature → classification | `.claude/agents/tpu-diagnoser.md` |
| Is the run healthy right now | `.claude/agents/tpu-watchdog.md` |
| Protected config + promotion gates | `orchestration/TPU_OPTIMIZATION_SPEC.md` |
| Known-good numbers | `orchestration/playbook/baseline-v6e8-siglip-cohere2.md` |
| Metric names + comparability rules | `orchestration/playbook/perf-metrics-schema.md` |
| ntfy event meanings | `orchestration/playbook/event-taxonomy.md` |
| The actual commands | `docs/tpu/tpu-runbook.md` |

## Layering

1. **VM tmux `train`**, every worker — launched by `startup_script.sh` at boot and by
   `deploy_tarball.sh` on redeploy, via `train_launcher.sh`. Log `/tmp/train.log`.
   Deploys kill and relaunch on all workers (`--worker=all`).
2. **Workstation tmux `qrwatch-tayavision`** — `qr_watch.sh`: forensics → quota check →
   durability gate → delete → identical resubmit. Log `/tmp/qr_watch_tayavision.log`.
   The suffix matters; a bare `qrwatch` on this workstation is `llm-architectures`'.
3. **Boot self-heal via QR metadata** — tarball URI, entrypoint, Hydra overrides,
   `resume=auto`, W&B identity. A preemption *reboot* (QR survives) recovers with zero
   action.
4. **Claude session** — never owns a loop that must outlive it.

## Recovery ladder (least → most invasive)

1. **Read.** `ops.sh tail-logs`, or the `tpu-watchdog` agent. Most apparent hangs are a
   compile; budget 20 minutes cold and check the compile-cause counter, not the clock.
2. **Redeploy (T2).** `deploy_tarball.sh`. Costs a full ~14 min recompile here — batch
   your fixes rather than making two one-line trips.
3. **Recycle the QR (T3).** `qr_watch.sh` does this automatically **only when
   `SAVE_CKPT_DIR` is a `gs://` URI**. Manually: delete the QR, then `launch_spot.sh`
   with the same env. One resubmitter per QR, ever.
4. **Zone rotation.** Follow the fallback tree in `docs/tpu/tpu-capacity-log.md` §1 —
   it leads with EU because there is one bucket, `gs://tayavision-eu`, and a US zone pays
   egress on every write.

## Invariants

- **196 image tokens per image** on the SigLIP path. MoonViT's variable count means a new
  HLO program per shape and is not TPU-eligible unbucketed.
- **`DistributedSampler` off under SPMD.** One process holding a 1/N shard trains on 1/N
  of the data and reports nothing wrong.
- **Never wrap `Cohere2DecoderLayer` in an FSDPv2 auto-wrap policy** — 36 wraps, 36 bf16
  reduce-scatters, known NaN. `replicated` is the default and probably sufficient.
- **Tarball deploys including `.env`, never a git clone.** It is the only way `HF_TOKEN`
  reaches the VM, and `CohereLabs/tiny-aya-*` is gated.
- **One resubmitter per QR.**
- **Long loops live in tmux**, never in a Claude session.
- **No TPU number enters a `docs/*.md` results table** until the same entrypoint runs on
  both backends.

## Optimization mode

Only once a run is stable. `TPU_OPTIMIZATION_SPEC.md` owns the six gates and the
nine-phase program; `playbook/optimization-experiment-matrix.md` owns the candidates and
the 20 → 300 → 1000 → full run ladder.

Start with `opt-0-metrics` (you cannot tune what you cannot see) and `opt-7-strategy`
(cheapest way to find out whether the inherited FSDPv2 NaN applies to this model at all).
