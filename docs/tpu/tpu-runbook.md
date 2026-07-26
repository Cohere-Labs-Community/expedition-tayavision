# TPU Runbook — Tiny Aya Vision on Cloud TPU (torch_xla)

Operational guide for provisioning and driving a TRC Cloud TPU slice for Tiny Aya Vision.
Companions: [`tpu-trc-allocation.md`](tpu-trc-allocation.md) (the grant),
[`tpu-capacity-log.md`](tpu-capacity-log.md) (observed capacity + the fallback tree), and
`.claude/orchestration/` (why any of this is shaped the way it is).

> ## Read this first
>
> **No training entrypoint runs on TPU yet.** `pipeline/train_alignment.py`,
> `train_instruct.py`, and `train_multilingual.py` are DDP + CUDA — `torch.autocast("cuda", …)`,
> `torch.device(f"cuda:{local_rank}")`, `DistributedDataParallel`, `DistributedSampler`.
> None of that runs under torch_xla.
>
> What this runbook gives you today is **provisioning and supervision**: a slice, code
> delivery, dependency install, a launched-and-watched process, and automatic recovery
> from spot preemption. The default entrypoint is `scripts/tpu/tpu_smoke.py`, which
> exercises every one of those layers and reports device count, mesh, HBM, and XLA
> compile counters on real silicon.
>
> The remaining work is a `src/backend/` seam. See `.claude/orchestration/SPEC.md` §9.
> **Do not edit `pipeline/train_*.py` to "just add TPU support" inline** — that breaks the
> Modal/GPU path, which is where every published result comes from.

## Hardware & zones

The grant is shared with sibling projects on the same GCP project
(`ml-pipelines-315702`). Full table in [`tpu-trc-allocation.md`](tpu-trc-allocation.md).

| Profile | Topology | Zone | Runtime | Bucket |
|---|---|---|---|---|
| **`v6e-8-eu`** (default) | 1 host × 8 chips | `europe-west4-a` | `v2-alpha-tpuv6e` | co-located |
| `v6e-64-ew4a` | 8 hosts × 8 chips | `europe-west4-a` | `v2-alpha-tpuv6e` | co-located |
| `v5e-64-ew4b` | 16 hosts × 4 chips | `europe-west4-b` | `v2-alpha-tpuv5-lite` | co-located |
| `v6e-64-ue1d`, `v5e-64-uc1a`, `v4-32-uc2b` | — | US zones | — | **cross-region** |

- TRC v6e is **spot-only**. 31.25 GiB usable HBM per chip; the model is ~7.5 GB in bf16.
- **One bucket: `gs://tayavision-eu` (europe-west4).** Only the three EU profiles are
  co-located; anything else pays egress on every checkpoint write.
- **Multi-host external-IP headroom: verify, do not assume.** A sibling project hit an
  `IN_USE_ADDRESSES` cap of 8 in this grant (capacity log §7.1), which would block any
  8-host slice. **Measured in `europe-west4` on 2026-07-25: usage 16, limit 64** — 48
  free, so an 8-host v6e-64 fits. The cap was either raised or was never 8 in this
  region. Check before blaming it:
  `gcloud compute regions describe europe-west4 --format='json(quotas)'`.
  `v6e-8-eu` is single-host and sidesteps both the IP question and PJRT rendezvous —
  still the right default while the TPU path is being brought up.
- **SUSPENDED/FAILED QR husks still book quota.** Delete them before blaming capacity.

## Prerequisites

```bash
bash scripts/tpu/ops.sh preflight
```

Checks gcloud auth, the four `.env` keys, entrypoint existence, and bucket reachability.
It prints presence, never values.

`.env` at the repo root is **required** and is gitignored. Copy `.env.example` and fill:

| Key | Absent → |
|---|---|
| `HF_TOKEN` | **The run cannot start.** `CohereLabs/tiny-aya-{base,global}` is a gated repo. |
| `WANDB_API_KEY` | Trains, but is unobservable — `no WANDB_API_KEY found` in the log. |
| `NTFY_TOPIC` | Self-healing still works, silently. Tolerable for a foreground smoke, not overnight. |
| `PROJECT_ID` | Defaults to `ml-pipelines-315702`. |

The tarball ships `.env` to the VM. **There is no git clone anywhere in this flow** —
that is the only way a gated-repo token reaches the worker.

> Append to `.env` with `printf '\n%s\n'`, never `echo X >>`. A file with no trailing
> newline silently concatenates the new key onto the previous line, and the variable
> then does not exist. That cost a full day of dropped alerts on a sibling project.

## One-time bootstrap

```bash
bash scripts/tpu/setup_gcp.sh    # enable APIs, create gs://tayavision-eu, grant IAM
```

Idempotent. Creates exactly one bucket, in `europe-west4`.

## Provision and launch

```bash
TRC_PROFILE=v6e-8-eu bash scripts/tpu/launch_spot.sh
```

`launch_spot.sh` → `launch_qr.sh` tars the working tree (**including the gitignored
`.env`**), uploads it to `gs://tayavision-eu/code/`, and creates the queued resource with
`startup_script.sh` as metadata. On boot each host installs uv, pulls and extracts the
pinned tarball, runs `uv sync` (CPU torch) + `uv pip install torch_xla[tpu]`, resolves the `libpython3.12.so.1.0` link that
`torch_xla`'s `_XLAC.so` needs, and launches the entrypoint in a `tmux` session teed to
`/tmp/train.log`.

Knobs travel as VM metadata:

| Variable | Default | Meaning |
|---|---|---|
| `TAYAVISION_ENTRYPOINT` | `scripts/tpu/tpu_smoke.py` | Repo-relative path passed to `uv run python -u` |
| `TAYAVISION_HYDRA_OVERRIDES` | `""` | Appended verbatim, e.g. `vision=siglip llm=base wandb.project=tayavision-tpu` |
| `TAYAVISION_RESUME` | `auto` | Resume from the latest checkpoint under `SAVE_CKPT_DIR` |
| `TPU_STRATEGY` | `replicated` | `replicated` \| `fsdpv2_lora` \| `fsdpv2` \| `auto` |
| `SAVE_CKPT_DIR` | unset | **Set this to a `gs://` path** for anything long-running — see below |

This repo is Hydra-configured, so the orchestration does not invent a dozen env
overrides: it carries one override string and appends it to the launch line.

### Keep the QR alive across preemption

```bash
tmux new -d -s qrwatch-tayavision 'bash scripts/tpu/qr_watch.sh'
```

Polls the queued resource every 300s; on `SUSPENDED`/`FAILED` it captures forensics,
checks for a quota-class abort, deletes the QR, and resubmits the identical launch env.
Bounded by `MAX_RESUBMITS=20` plus a cooldown.

The session name is **suffixed** deliberately — the same workstation runs a plain
`qrwatch` for `llm-architectures`, and `tmux new -s qrwatch` against an existing session
attaches to the wrong one, leaving this project's QR silently unwatched.

> **`qr_watch.sh` refuses to auto-recycle unless `SAVE_CKPT_DIR` is a `gs://` URI.**
> A recycled node whose checkpoints went to node-local disk resumes from step 0 and
> re-pays ~14 minutes of cold-boot XLA compile to redo work it already did — up to twenty
> times, silently. Without durable checkpoints it degrades to notify-only and says so in
> the push event.

## Observe and control

```bash
bash scripts/tpu/ops.sh status       # QR + node state
bash scripts/tpu/ops.sh tail-logs    # follow /tmp/train.log on worker 0
bash scripts/tpu/ops.sh attach       # sudo tmux attach -t train (root-owned)
bash scripts/tpu/ops.sh ssh          # shell on worker 0
bash scripts/tpu/ops.sh delete       # tear down the QR (stops billing)
```

**Compile is not a hang.** The persistent XLA cache has nondeterministic keys on v6e SPMD
and effectively never warm-hits, so **~14 minutes of recompile on a cold boot is normal**.
Judge a stall by the `met.metrics_report()` compile-cause counter, not by elapsed time.
The `tpu-watchdog` agent uses a 20-minute threshold for exactly this reason.

When reading logs, take status markers from the **tail** and fetch repeating per-step
lines in a separate query. Piping both through one `head -N` truncates the rare marker
behind N repeats of the noisy one; that bug hid a completed run for an hour on a sibling
project.

## Redeploy without reprovisioning

```bash
bash scripts/tpu/deploy_tarball.sh
```

Tars, uploads, pulls on `--worker=all`, and relaunches tmux `train` everywhere. This is
the T2 action in the recovery ladder; it never touches the queued resource.

**A redeploy costs a full recompile here** (~14 min), unlike the ~1 minute it costs on a
JAX sibling. Batch your fixes — two T2s in a row for two one-line patches is half an hour
of silicon spent compiling.

## Gotchas

- **`torch_xla` import fails with `ImportError: libpython3.12.so.1.0`.** `_XLAC.so`
  dynamically links libpython, which uv keeps outside the default loader path.
  `startup_script.sh` exports `LD_LIBRARY_PATH` from the uv Python root; if this fires,
  that block was dropped.
- **SPMD is a single process.** `DistributedSampler` must be **off**. Left on, one process
  holds a 1/N shard, trains on 1/N of the dataset, and reports nothing wrong.
- **Never wrap `Cohere2DecoderLayer` in an FSDPv2 auto-wrap policy.** 36 per-layer wraps
  produce 36 bf16 reduce-scatters and hit a known NaN (pytorch/xla #8591 / #8778); FSDPv2
  has no `fp32_reduce_scatter`. Observed on a sibling project with **this same backbone**.
  On this model `replicated` fits comfortably and sidesteps it — Phase 1 trains ~11.5M
  params of ~3.75B.
- **MoonViT is not TPU-eligible unbucketed.** It emits a variable token count per image,
  so XLA compiles a new program per distinct shape. SigLIP's fixed 196 tokens is the TPU
  path.
- **A killed `gcloud ssh` does not kill the remote command.** A local `timeout` orphans
  it. Long remote work goes in a detached session with an explicit liveness check.
- **`which uv` is empty under sudo.** Use `/root/.local/bin/uv` in root contexts.
- **TRC is a free grant.** Never delete or reprovision a slice without intent, and delete
  husks you are done with — they book quota for everyone on the project.

## Where to look when something breaks

| Symptom | First thing to open |
|---|---|
| Anything at all | `.claude/agents/tpu-diagnoser.md` — signature → tier → action |
| What to do at a tier | `.claude/orchestration/playbook/tier-definitions.md` |
| QR state / preemption | `/tmp/qr_watch_tayavision.log`, then `ops.sh status` |
| Run behaviour | worker 0 `/tmp/train.log`, tail-first |
| Capacity refusals | [`tpu-capacity-log.md`](tpu-capacity-log.md) §7.1 |
| "Should this even work yet?" | `.claude/orchestration/SPEC.md` §9 |
