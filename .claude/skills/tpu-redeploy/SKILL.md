---
name: tpu-redeploy
description: Push the local working tree to a live TPU slice and relaunch, without recreating the queued resource. Covers multi-host fan-out and mid-run safety rules.
---

# Redeploy to a live TPU slice

This is the **T2** action in the recovery ladder. It never touches the queued resource —
a lost node is T3 and belongs to `qr_watch.sh`.

> **A redeploy here costs a full XLA recompile (~14 min)**, unlike the ~1 minute it costs
> on the pure-JAX sibling: the persistent cache has nondeterministic keys on v6e SPMD and
> never warm-hits. Batch your fixes. Two T2s for two one-line patches is half an hour of
> silicon spent compiling.

## What it does

1. Tars the working tree **including the gitignored `.env`** — the only way `HF_TOKEN`
   (for the gated `CohereLabs/tiny-aya-*` repos), `WANDB_API_KEY`, and `NTFY_TOPIC` reach
   the VM. There is no git clone anywhere in this flow.
2. Uploads to `gs://tayavision-eu/code/` (stamped + `latest.tar.gz`).
3. Pulls and extracts over `$REPO_DIR` on every worker, preserving `.venv` and any staged
   datasets.
4. `uv sync` (CPU torch) + `uv pip install torch_xla[tpu]`, and re-resolves `LD_LIBRARY_PATH` for
   `torch_xla`'s `_XLAC.so`.
5. Kills and relaunches tmux `train` via `train_launcher.sh`, with a unique
   `TAYAVISION_RUN_TAG` baked into the launch line so later log queries can be scoped.

## Invocation

```bash
TAYAVISION_ENTRYPOINT=scripts/tpu/tpu_smoke.py \
TAYAVISION_HYDRA_OVERRIDES="vision=siglip llm=base wandb.project=tayavision-tpu" \
TPU_STRATEGY=replicated \
SAVE_CKPT_DIR=gs://tayavision-eu/checkpoints/smoke-01 \
ZONE=europe-west4-a NODE_ID=tayavision-v6e8 \
bash scripts/tpu/deploy_tarball.sh
```

Precedence: shell env > repo-root `.env` > script defaults. Defaults target `v6e-8-eu`.

## Mid-run safety

Three things must hold before redeploying a run you want to keep:

- **`SAVE_CKPT_DIR` set, and a `gs://` URI.** A redeploy kills `train` on every worker.
  Without durable checkpoints that discards progress — and `qr_watch.sh` will also refuse
  to auto-recycle on the next preemption (`SPEC.md` §5).
- **`TAYAVISION_RESUME=auto`**, or the relaunch starts from step 0.
- **`WANDB_RUN_ID` fixed**, or the resumed run forks a second W&B entry instead of
  rejoining.

## Multi-host rules

Only relevant on `v6e-64` (8 hosts) or `v5e-64` (16 hosts). `v6e-8-eu`, the default, is
single-host and skips all of this.

- Fan out with `--worker=all`. Every host must relaunch or the PJRT rendezvous hangs the
  survivors until `DEADLINE_EXCEEDED`.
- Gate on every worker reporting startup complete in `/tmp/startup.log` before launching.
- Re-run a missing worker's startup **detached** (`nohup … &` or tmux) — never a
  foreground ssh with a local `timeout`.
- Every multi-host profile is currently blocked on the `IN_USE_ADDRESSES` regional cap
  (8, and an 8-host slice wants exactly 8). See `docs/tpu/tpu-capacity-log.md` §7.1.

## Gotchas

- **`ImportError: libpython3.12.so.1.0`.** `torch_xla`'s `_XLAC.so` dynamically links
  libpython, which uv keeps outside the default loader path. The redeploy must re-export
  `LD_LIBRARY_PATH` from the uv Python root, exactly as `startup_script.sh` does — this
  is the torch_xla-specific step the JAX sibling has no equivalent of.
- **A killed `gcloud ssh` does not kill the remote command.** A local `timeout` around it
  orphans the remote process. Long remote work goes in a detached session with an
  explicit liveness check.
- **Never pipe a critical deploy step through `head`.** SIGPIPE kills it mid-work.
- **`which uv` is empty under sudo.** Use `/root/.local/bin/uv` in root contexts.
- **Empty `WANDB_*` exports break `wandb.init`** with `Run ID cannot be empty`. Export
  only when non-empty.

## After deploying

Watch the tag-scoped segment of worker 0's `/tmp/train.log`. Read status markers from the
**tail**; fetch the repeating per-step line separately. Expect no step line for up to 20
minutes on a cold boot — that is compilation, not a hang. Confirm with the compile-cause
counter rather than the clock.
