# `scripts/tpu/`

Provisioning and supervision for Tiny Aya Vision on TRC Cloud TPU.

> **These scripts cannot train this model yet.** `pipeline/train_*.py` is DDP + CUDA.
> `TAYAVISION_ENTRYPOINT` defaults to `tpu_smoke.py`, which proves the control plane on
> real silicon. See `.claude/orchestration/SPEC.md` §9. Design docs live in
> `.claude/orchestration/`; the operator guide is `docs/tpu/tpu-runbook.md`.

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `_lib.sh` | everywhere (sourced) | `load_env_file`, `notify`, `make_code_tarball`, `profile_spec`, cross-region warning |
| `setup_gcp.sh` | workstation, once | APIs, `gs://tayavision-eu`, IAM |
| `launch_qr.sh` | workstation | Create the queued resource; carries all knobs as VM metadata |
| `launch_spot.sh` | workstation | `TRC_PROFILE` wrapper with `SPOT=1`. **The normal entry point.** |
| `startup_script.sh` | TPU VM, every boot | apt → uv → tarball → `uv sync` (CPU torch) + `uv pip install torch_xla[tpu]` → libpython link → tmux `train` |
| `train_launcher.sh` | TPU VM, in tmux | Runs the entrypoint; emits the markers every watcher greps |
| `deploy_tarball.sh` | workstation | **T2**: push code to a live slice and relaunch. No QR change. |
| `qr_watch.sh` | workstation, tmux | **T3**: recycle on preemption, gated on `gs://` checkpoints |
| `ops.sh` | workstation | `preflight`, `status`, `tail-logs`, `attach`, `ssh`, `delete` |
| `tpu_smoke.py` | TPU VM | The default entrypoint. Not training. |

Not ported from the sibling repos: sweep runners, VM coordinators, provisioning races, and
supervisor VMs. There is nothing to sweep until the backend seam lands.

## Environment

Precedence: **shell env > repo-root `.env` > script defaults.**

| Variable | Default | Notes |
|---|---|---|
| `TRC_PROFILE` | `v6e-8-eu` | The only profile not blocked by the external-IP cap |
| `PROJECT_ID` | `ml-pipelines-315702` | The TRC grant project |
| `BUCKET` | `gs://tayavision-eu` | **One bucket.** Non-EU zones pay egress. |
| `TAYAVISION_ENTRYPOINT` | `scripts/tpu/tpu_smoke.py` | Repo-relative, passed to `uv run python -u` |
| `TAYAVISION_HYDRA_OVERRIDES` | `""` | Appended verbatim, e.g. `vision=siglip llm=base` |
| `TAYAVISION_RESUME` | **per caller** — `auto` for `launch_qr.sh`/boot, `off` for `deploy_tarball.sh` | See below |
| `TPU_STRATEGY` | `replicated` | Not FSDPv2 — see `TPU_OPTIMIZATION_SPEC.md` §4 |
| `SAVE_CKPT_DIR` | unset | **Must be `gs://`** for T3 auto-recycle *and* for `TAYAVISION_RESUME=auto` |

### `TAYAVISION_RESUME`

| Value | Effect |
|---|---|
| `off` / `none` / `no` / `0` / `false` / unset | Start fresh |
| `auto` | Resume the run with the **most recently mirrored** checkpoint under `SAVE_CKPT_DIR` |
| `<run-id>` | Resume that run specifically |

An explicit Hydra `resume=<run-id>` on the command line beats this variable.

**Why the default differs by caller.** `auto` resumes whatever ran last. On the
boot/recycle path that is exactly right: the QR metadata is replayed verbatim, so the
overrides cannot have drifted. On a human `deploy_tarball.sh` the overrides *do* change
between invocations, and resuming would load the previous run's optimizer state, LR
schedule position, and step count into a different config — silently. So `deploy_tarball.sh`
defaults to `off`; pass `TAYAVISION_RESUME=auto` explicitly when you mean it.

`auto` needs a `gs://` `SAVE_CKPT_DIR`: node-local checkpoints are exactly what a recycle
destroys, so there would be nothing to find.
| `MAX_RESUBMITS` | `20` | T3 budget |
| `HF_TOKEN` | — | **Required.** The model repos are gated. |
| `NTFY_TOPIC` | — | Absent → `notify()` is a silent no-op |

## Typical flow

```bash
cp .env.example .env              # fill HF_TOKEN at minimum
bash scripts/tpu/setup_gcp.sh     # once
bash scripts/tpu/ops.sh preflight

TRC_PROFILE=v6e-8-eu \
SAVE_CKPT_DIR=gs://tayavision-eu/checkpoints/smoke-01 \
bash scripts/tpu/launch_spot.sh

tmux new -d -s qrwatch-tayavision 'bash scripts/tpu/qr_watch.sh'
bash scripts/tpu/ops.sh status
bash scripts/tpu/ops.sh tail-logs
# ... iterate with deploy_tarball.sh ...
bash scripts/tpu/ops.sh delete    # a live slice bills whether or not it is busy
```

## What these scripts do NOT do

- **Train the model.** The entrypoint is the last mile and it is not built.
- **Decide cross-region tradeoffs.** `launch_spot.sh` warns on a non-EU profile against
  the single EU bucket; it does not block. That call is the operator's.
- **Recycle without durable checkpoints.** `qr_watch.sh` refuses and says why.
- **Sweep, race zones, or run a supervisor VM.**
- **Touch the Modal path.** That is separate, working, and authoritative.

## Dry runs

`launch_qr.sh` and `deploy_tarball.sh` both honour `DRY_RUN=1` and print what they would
do without spending capacity:

```bash
DRY_RUN=1 TRC_PROFILE=v6e-8-eu    bash scripts/tpu/launch_spot.sh
DRY_RUN=1 TRC_PROFILE=v6e-64-ue1d bash scripts/tpu/launch_spot.sh   # prints the egress warning
```
