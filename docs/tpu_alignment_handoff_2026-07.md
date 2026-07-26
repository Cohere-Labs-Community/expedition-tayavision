# Tiny Aya Vision on TPU — run record and resume guide

**Written 2026-07-26, the last day of the TRC grant.** Everything below was measured on
real hardware, not estimated. Where something is unverified it says so.

The headline: **Phase 1 alignment completed a full epoch on a Cloud TPU v6e-16.** The
`src/backend/` seam now lets the same `pipeline/train_alignment.py` run on either CUDA/DDP
or torch_xla/SPMD.

---

## 0. TL;DR

| | |
|---|---|
| **Result** | Full 1-epoch alignment, 2,180 optimizer steps, loss **9.053 → 2.367** |
| **Hardware** | v6e-16 (4 hosts x 4 chips), `europe-west4-a`, spot |
| **Wall clock** | **59 min 08 s** |
| **Trained weights** | `outputs/tpu-align-v6e16-2026-07-25/checkpoint_4360.pt` — **local only, and gitignored.** See section 9. |
| **W&B** | https://wandb.ai/cataluna84/tayavision-tpu/runs/e93fe1cd684c4c0f925bad9eeb38ddf1 |
| **Effective batch** | **256 — deliberately identical to the Modal GPU baseline** |
| **Code** | Branch `tpu`, PR #76, CI green |
| **GCP** | **Fully decommissioned** — slice and `gs://tayavision-eu` both deleted |

The checkpoint is **backend-portable**: all tensors are CPU-resident bf16, so it loads on
CUDA with no conversion. Verified by loading it (section 1).

> **The weights exist in exactly one place: `outputs/` on one machine.** That directory is
> gitignored, so it is not on GitHub and not in any bucket. Back it up.

---

## 1. Artifacts — what exists and where

### The trained projector

```
outputs/tpu-align-v6e16-2026-07-25/         <- gitignored, LOCAL ONLY
    checkpoint_1000.pt   checkpoint_2000.pt   checkpoint_3000.pt
    checkpoint_4000.pt   checkpoint_4360.pt   <- final, end of epoch
    config.json
```

Filenames count **micro-steps**; 4360 micro-steps = 2,180 optimizer steps at
`grad_acc_steps=2`.

These were mirrored to `gs://tayavision-eu` during the run, pulled down at wrap-up, and
**MD5-verified against their GCS objects before that bucket was deleted** — so they are
confirmed faithful copies, not assumed ones. Hashes are in section 9. There is no cloud
copy any more.

Verified contents of `checkpoint_4360.pt`:

| Field | Value |
|---|---|
| `step` | 4360 |
| `lr_scheduler.last_epoch` | 2180 (consistent with the above) |
| `projector` | 6 tensors, **11,547,648 params** — matches the documented ~11.5 M |
| dtype / device | **bf16 / cpu** — portable to CUDA as-is |
| Shapes | `layernorm.{weight,bias}` (4608,), `linear_1` (2048,4608), `linear_2` (2048,1024) |
| Sanity | mean 0.00040, std 0.03096, absmax 1.0, **0 NaNs** |
| Also contains | full `optimizer` (AdamW) and `lr_scheduler` state, so a resume is exact |

### Runs in W&B (`cataluna84/tayavision-tpu`)

| Run id | Steps | Final loss | What it is |
|---|---|---|---|
| `e93fe1cd684c4c0f925bad9eeb38ddf1` | 2180 | **2.3665** | **The full epoch. This is the run that matters.** |
| `2726cdffdb4d4e97b83265c2e1dcafab` | 19 | 5.7143 | Bounded smoke that gated the full run |
| `a2d64c8e…`, `a0d4bea2…`, `9b0c8a97…` | <20 | — | Checkpoint-durability and resume self-tests |

Eight further runs are in `failed`/`crashed` state; they are the debugging history in
section 6 and can be deleted.

---

## 2. The result, measured

Full epoch over LLaVA-Pretrain (558,128 caption pairs), SigLIP2-so400m frozen,
`CohereLabs/tiny-aya-base` frozen, only the projector training.

| Metric | Value |
|---|---|
| `train/loss` | 9.053 → **2.367** |
| `perf/step_time` p50 | **1.508 s** per optimizer step |
| p90 / max | 1.516 s / 2.014 s |
| `perf/images_per_sec` | **169.2** |
| `mem/used_gib` | 23.13 – **27.86** of 31.246 |
| `xla/compile_cause_count` | **`[1]` — one single distinct value across all 2,180 steps** |
| `train/max_image_tokens` | **196** every step (the SigLIP invariant) |
| Step 1 / step 2 | 162.5 s / 56.5 s (compile); warm from step 4 |

The flat compile counter is the most important line in that table. On XLA every distinct
tensor shape is a new HLO compile, so with dynamic padding this number climbs forever.
`[1]` across a whole epoch proves the fixed-sequence-length collate works.

### Batch configuration, and why

| Quantity | Value |
|---|---|
| Global batch | **128** |
| Per host (of 4) | 32 |
| Per chip (of 16) | **8** |
| `grad_acc_steps` | **2** |
| **Effective batch** | **256** |
| Sequence length | **640**, fixed |
| LR / warmup / schedule | **1e-3 / 0.03 / cosine — unchanged from the GPU baseline** |

`B_eff = 256` matches the Modal GPU baseline exactly (there `batch_size=8 x grad_acc=32`).
That is invariant #2 of `.claude/orchestration/playbook/perf-metrics-schema.md` and is what
makes this loss curve comparable to the GPU one. **Do not change LR or warmup without
changing `B_eff`, and vice versa.**

**Global batch 256 does not fit.** It OOMs at step 2: *"Attempting to reserve 16.25G ...
There are 15.69G free"*. 128x2 was chosen to halve activation memory while holding `B_eff`.

### Sequence length 640 is measured, not guessed

`tiny-aya-global` has a chat template with a ~370-token preamble, so a LLaVA-Pretrain
sample is far longer than the caption suggests. Measured over 400 samples: **p99 = 599,
max = 601**, image span ending at most at 569. 640 keeps 100% of samples intact.

An earlier value of 256 silently truncated **every image token away**, which surfaced as
`image_hidden_states is None` rather than as anything mentioning truncation.
`collate_fn` now raises if `fixed_seq_len` would cut into image tokens.

### Do not trust the analytical activation model

It was wrong **four times in this project, always low**:

1. Used `hidden_size` 2048 for the dominant term. Cohere2's FFN `intermediate_size` is
   **11008** — 5.38x larger, and the MLP intermediates dominate.
2. Assumed per-chip batch was `B_global / chips`. Nothing sharded the input.
3. Corrected, still predicted 10.2 GiB against a measured 17.1.
4. Predicted ~19.4 GiB for `B_global=128` by halving the 256 figure; actual peak **27.8**.
   Halving the batch does not halve total memory — the static term and XLA's live fused
   temporaries do not scale with batch.

**Size batches by bisection and read `mem/used_gib`.** Treat any formula as an
order-of-magnitude check only.

---

## 3. Resuming on GPU / Modal

The Modal path is untouched and remains authoritative. The GPU code path is
byte-equivalent to before this work: `uv run pytest tests/ -q` gives the same
**121 passed / 8 errors** it did at the start (the 8 need gated HF repos), plus 31 new
tests for the checkpointing added here.

### Continue to Phase 2 (instruct) from the TPU-trained projector

`config/training/instruct.yaml:22` already has the hook:

```yaml
alignment_checkpoint: "/data/model_ckpts/alignment_ckpts/4d93b48c-.../checkpoint_4361.pt"
```

Point it at this run's checkpoint instead — as a CLI override, per repo convention:

```bash
python pipeline/train_instruct.py training=instruct \
  training.alignment_checkpoint=/path/to/checkpoint_4360.pt
```

The tensors are CPU bf16, so nothing about the TPU origin needs undoing.

> Note the existing default there is a **UUID-bearing absolute path**, which is one of the
> still-open Definition-of-Done items in `.claude/PLAN.md`.

### Re-run Phase 1 on GPU for comparison

```bash
python pipeline/train_alignment.py training=alignment
```

`B_eff` is already 256 on that path (`batch_size=8 x grad_acc_steps=32`), so the loss curve
should be directly comparable to the TPU run above. **Always pass `training=alignment`** —
`config/config.yaml` defaults to `training: instruct`, which would silently run Phase 1
with LR 2e-5 (50x too low) against the wrong dataset.

---

## 4. Resuming on TPU

### Provision (see section 8 first — size matters)

`gs://tayavision-eu` was deleted at wrap-up, so recreate it before anything else.
`setup_gcp.sh` is idempotent and creates the bucket if absent:

```bash
bash scripts/tpu/setup_gcp.sh

TRC_PROFILE=v6e-16-ew4a \
SAVE_CKPT_DIR=gs://tayavision-eu/checkpoints/<new-prefix> \
bash scripts/tpu/launch_spot.sh

TRC_PROFILE=v6e-16-ew4a bash scripts/tpu/ops.sh status   # wait for ACTIVE (~4-6 min)
```

### Stage the corpus (~3.5 min, 28 G, idempotent)

```bash
TRC_PROFILE=v6e-16-ew4a bash scripts/tpu/stage_data.sh
TRC_PROFILE=v6e-16-ew4a bash scripts/tpu/ops.sh stage-logs
```

Runs detached in tmux `stage` on every worker and writes a `.staged` marker, so an
interrupted invocation is resumable rather than corrupt.

### Launch training (this is the exact command that produced the result)

```bash
TRC_PROFILE=v6e-16-ew4a \
SAVE_CKPT_DIR=gs://tayavision-eu/checkpoints/<new-prefix> \
TAYAVISION_ENTRYPOINT=pipeline/train_alignment.py \
TAYAVISION_HYDRA_OVERRIDES='training=alignment training.batch_size=128 training.grad_acc_steps=2 training.fixed_seq_len=640 training.data_dir=/root/data/llava-pretrain training.num_workers=16' \
bash scripts/tpu/deploy_tarball.sh
```

Add `training.max_steps=20` for a bounded smoke first. Expect **no step line for ~3.5 min**
— that is XLA compiling.

### Resume after a preemption

```bash
TAYAVISION_RESUME=auto ...   # resolves the newest mirrored checkpoint, refetches, resumes
TAYAVISION_RESUME=<run-id>   # a specific run
```

`deploy_tarball.sh` defaults to `off`; `launch_qr.sh` and the boot path default to `auto`.
That split is deliberate — see section 6.

### Watch a run

```bash
TRC_PROFILE=v6e-16-ew4a bash scripts/tpu/ops.sh tail-logs
```

**Rank 0 is not necessarily gcloud worker 0.** In this run `process_index=0` lived on
worker 1, and that is where tqdm and W&B write. `ops.sh tail-logs` follows worker 0. Grep
for `[shard] process_index=` to find rank 0 before assuming a host is idle.

---

## 5. Where the code is

**Branch `tpu`, PR #76, CI green.** 90 files, +8,845 / -181.

https://github.com/Cohere-Labs-Community/expedition-tayavision/pull/76

Ten commits, ordered so each stands alone and the branch bisects:

| Commit | |
|---|---|
| `74b9422` | ruff/pytest config, 76 → 0 lint findings, CI workflow |
| `d681c9c` | `CLAUDE.md` + the `.claude/` memory system |
| `1694ba6` | **`src/backend/` seam** + `check_backend_seam.sh` |
| `11daf17` | TPU run control (`scripts/tpu/`, `docs/tpu/`) |
| `5581944` | durable `gs://` checkpointing + 31 tests |
| `a0ee236` | **trainer conversion + the six silent XLA bugs** |
| `5f9e1e1` | W&B project routing fix |
| `fd9eff6` | this document |
| `860831a` | decommission plan |
| `268f210` | `requires_gpu` fix — the CI failure described below |

The `.claude/` memory files (`PLAN.md`, `PROGRESS.md`, `memories.md`) are **gitignored as
per-contributor state**, so they are not on the branch. That is deliberate — `PROGRESS.md`
is an append-only machine log that would conflict on every merge — but it does mean this
document is the durable record, not those.

### The CI failure worth knowing about

The workflow added in `74b9422` failed on its first run and caught a bug in that same
commit. `requires_gpu` was written as

```python
requires_gpu = pytest.mark.requires_gpu(pytest.mark.skipif(...))   # WRONG
```

which does **not** compose two marks — calling a `MarkDecorator` with a non-callable
argument stores it as a *parameter of the mark*, so the skipif is never applied. Six GPU
tests then ran on the CPU-only runner and errored with `Found no NVIDIA driver`.

**The broken version is invisible on any machine with a GPU**, because the skip is never
needed there. Only a CPU-only runner distinguishes them. Fixed in `268f210` with a function
decorator; verified by reproducing the runner with `CUDA_VISIBLE_DEVICES=""`, which turns
the exact CI command from `135 passed, 6 errors` into `135 passed, 6 skipped`.

`.env` is gitignored and must stay that way. It is shipped to the VM inside the code
tarball on purpose, because `CohereLabs/tiny-aya-*` are gated and there is no git clone in
that flow.

### What changed, by area

| Path | Change |
|---|---|
| `src/backend/{base,gpu_backend,tpu_backend}.py` | New accelerator seam. `torch_xla` is confined to `tpu_backend.py`, enforced by `scripts/ci/check_backend_seam.sh` |
| `pipeline/train_alignment.py` | Converted to the seam; collective-symmetry fixes; heartbeats; resume ordering |
| `pipeline/data.py` | `fixed_seq_len` in `collate_fn`, plus a guard against truncating image tokens |
| `pipeline/utils.py` | `gcs_checkpoint_root`, `mirror_checkpoint_to_gcs`, `fetch_checkpoint_from_gcs`, `resolve_resume_run_id` |
| `scripts/tpu/` | 12 files: provisioning, deploy, staging, watchdog, ops |
| `config/training/alignment.yaml` | `max_steps` and `fixed_seq_len` keys (Hydra struct mode rejects keys absent from the YAML) |
| `tests/test_checkpoint_gcs.py` | 31 tests, all mocking `gcloud`; no bucket or network needed |
| `.claude/orchestration/` | Run-control design; `playbook/baseline-v6e8-siglip-cohere2.md` holds the measured numbers |

State at handoff: **`ruff check .` = 0**, backend-seam guard passes, **152 passed / 8
errors** (the 8 are the pre-existing gated-repo tests).

---

## 6. Bugs found, and how to recognise them again

Every one of these failed **silently**. None produced a traceback at the point of the bug.

### 6.1 `if is_main:` around device work deadlocks SPMD

**Signature:** all hosts idle in `futex_wait_queue`, host CPU ~1 jiffy/5 s, HBM resident,
TPU duty cycle 0%, no traceback. Survived 2 h 39 min before being noticed, because "idle
with memory allocated" is indistinguishable from "compiling" without a stack dump.

**Cause:** under SPMD every host runs one program, so any device→host read (`.item()`,
`.cpu()`, `state_dict()`) is effectively collective — it forces a graph execution all hosts
must join. Rank 0 sat in `pn.std().item()` inside the W&B block while ranks 1-3 had already
raced into the next step's forward.

**Rule:** compute every scalar on **all** ranks; gate only `wandb.log`, `tqdm`, and disk
writes. Checkpointing goes through `backend.save()`, which every rank calls.

**Do not use `xm.save`** — it does `_maybe_convert_to_cpu(data, convert=should_write_data)`
(`xla_model.py:1246`), i.e. it transfers **only on the writer**, the same asymmetry. Its
docstring compensates by telling you to follow it with `xm.rendezvous`, but a barrier
cannot repair a mismatch in the collective work itself.

**Diagnosis that worked:** `py-spy dump --pid <main>` on every host, then compare where the
four MainThreads sit. The divergence is obvious in one shot.

### 6.2 Host-sharded input needs `ShardingSpec(minibatch=True)`

**Signature:** identical to 6.1 — silent stall, no error.

**Cause:** with each host loading a disjoint 1/N shard but `minibatch=False`, every host
asserts that *its* 64 rows are the global tensor to split across all 16 chips. Nothing
raises, because 64 divides 16 and every shape check passes.

With `minibatch=True` torch_xla instead requires the per-host batch to divide by the
**local** device count (`xla_model.py:1279`). The mesh stays **global** —
`ShardingSpec.__post_init__` derives its type from `xr.global_runtime_device_count()`.

**Confirmation it is working:** the logged batch shape becomes the global `(256, 640)`
rather than the per-host `(64, 640)`.

### 6.3 `MpDeviceLoader` does not shard at all

`torch_xla/distributed/parallel_loader.py` contains no `process_index`, `process_count`, or
any `xr.*` call. It wraps, prefetches, and applies a `ShardingSpec` **if you give it one**.
Without `input_sharding`, every chip processes the whole per-host batch.

### 6.4 `xr.world_size()` and `xr.global_ordinal()` lie under SPMD

They return **1** and **0** on every host. Use `xr.process_count()` and
`xr.process_index()`. Using the former makes host sharding a silent no-op.

### 6.5 `drop_last=False` is fatal on TPU, not merely wasteful

The sampler's tail-drop leaves 139,532 per host, but 139,532/32 leaves a short final batch.
That is (a) a new shape, i.e. a recompile every epoch, and (b) not divisible by the local
chip count, so `minibatch=True` **raises** — at the very end of an otherwise complete epoch.

### 6.6 `torch.load(map_location=<xla device>)` raises

```
RuntimeError: don't know how to restore data location of
torch.storage.UntypedStorage (tagged with xla:0)
```

The tag names the **target**, not the file. Load to host and let `load_state_dict` place it.

### 6.7 Resumed step counter double-counted

`step = step_offset + _i` is correct only on the GPU **subset** path, where the loader
yields only the remainder so `_i` restarts at 0. On the TPU **skip** path the loader still
yields the whole epoch, so `_i` is already absolute. Resuming from step 8 reported step 16,
tripped `max_steps` immediately, and exited `Training complete` having done **no work**.

### 6.8 Two env knobs read as wired and were not

`SAVE_CKPT_DIR` reached only `scripts/tpu/*.sh` (deploy warnings, QR metadata, the
`qr_watch.sh` durability gate); no Python read it, and `/models` is plain local disk, not a
GCS mount. `TAYAVISION_RESUME` was only echoed in a launcher banner.

Both are now live. **When auditing this repo's env knobs, grep for a *consumer*, not for
the assignment.**

`TAYAVISION_RESUME`'s default had to move out of `_lib.sh`: `auto` resumes whatever ran
last, which is right on the boot/recycle path (QR metadata is replayed verbatim, so
overrides cannot drift) and wrong on a human redeploy, where overrides change between
invocations and resuming would load run A's optimizer state into run B's config.

### 6.9 `ops.sh delete` could never delete a slice you had finished using

It omitted `--force`, and the API rejects deleting an `ACTIVE` queued resource
(`code 9`, must be one of `ACCEPTED / WAITING_FOR_RESOURCES / SUSPENDED / FAILED`). The
un-forced form only ever worked on husks. Fixed.

### 6.10 Environment pinning

`torch 2.9.1+cpu` / `torchvision 0.24.1+cpu` / `torch_xla 2.9.0` / `libtpu 0.0.21`, on
**Python 3.12 exactly** — floored by `pyproject.toml`, ceilinged by libtpu's cp312 wheels.
A torch/torchvision ABI mismatch surfaces misleadingly as
`Could not import module 'AutoProcessor'`; the real error is `torchvision::nms does not
exist`. `startup_script.sh` asserts the ABI immediately after install.

---

## 7. Known gaps

| Gap | Detail |
|---|---|
| **Weights exist in one place only** | `outputs/` is gitignored and the bucket is deleted, so the trained projector lives on a single machine with no backup. Highest priority. Section 9. |
| Only Phase 1 ran on TPU | `train_instruct.py` and `train_multilingual.py` are **not** converted to the seam. They are still DDP+CUDA and will not run on TPU. |
| No eval of the TPU checkpoint | The projector trained and the loss fell, but no CVQA/benchmark number was produced from it. |
| ~~`train_multilingual.py` wandb project~~ | **Fixed.** It hardcoded `project="tayavision-multilingual"`, so a TPU bring-up run would have written into a GPU *results* project. Now `os.environ.get("WANDB_PROJECT") or "tayavision-multilingual"`; `cfg` is not in scope in `main()`, so the env var is the override channel — which is what `scripts/tpu/train_launcher.sh` already sets from `.env`. The literal stays as fallback, so Modal is byte-for-byte unchanged. |
| Generation disabled on TPU | `_prepare_cache_for_generation` forces a `DynamicCache` whose shape grows per token, so each token is a new HLO. In-training sample logging is off on TPU by design. |
| MoonViT untested on TPU | It emits a *variable* token count per image, which is unbounded recompilation on XLA. PLAN P1 belongs on the Modal path. |
| `NTFY_TOPIC` empty | `qr_watch.sh` self-heals silently; fill it before any unattended run. |
| `index_put`/`nonzero` | `models/tiny_aya_vision.py:173` uses `nonzero` because `masked_scatter_backward` is broken under inductor on CUDA. `nonzero` is a dynamic-shape op XLA dislikes. It works, but it forces a host sync per forward. Belongs behind the seam. |
| Analytical memory model | Wrong four times. Bisect, do not compute. |

---

## 8. TRC capacity — read this before planning a bigger run

Five provisioning attempts across ~10 hours in `europe-west4-a`, zone clean before each:

| Profile | Attempts | Outcome |
|---|---|---|
| `v6e-64-ew4a` | 3 | **all FAILED** |
| `v6e-32-ew4a` | 2 | **all FAILED** |
| `v6e-16-ew4a` | 3 | **all ACTIVE**, ~4-6 min |

Every failure identical: `WAITING_FOR_RESOURCES → PROVISIONING → SUSPENDING → FAILED`,
`{"code": 13, "message": "an internal error has occurred"}`.

**Read `code 13` as "no capacity at this slice size", not as a bug.** The service reaches
`PROVISIONING` — it accepts and begins allocating — then withdraws. Quota is *not* the
constraint: `IN_USE_ADDRESSES` measured 16/64. A failed QR **terminates rather than staying
queued**, so waiting is not a strategy; each attempt is a fresh ~5-minute attempt.

**16 chips was the practical ceiling on this grant in this zone.**

Untried alternatives, with their real costs:

- **`v6e-64-ue1d`** (us-east1-d) — cross-region: every checkpoint write pays egress against
  the EU bucket, and staging pulls across regions.
- **`v5e-64-ew4b`** (europe-west4-b) — co-located, and podslice quota demonstrably exists
  (`TPU_LITE_PODSLICE_V5` limit 16, usage 0). But v5e has **16 GiB HBM/chip vs v6e's
  31.25**, and this run already needs 27.8 GiB at per-chip batch 8, so the batch config
  needs reworking and re-measuring, not just relaunching.

If you do get 64 chips: use **per-chip batch 4** so global stays 256 and `B_eff` stays 256
with `grad_acc_steps=1`. Raising `B_eff` means re-tuning LR — and the trap is not the LR but
**warmup**, which is a *fraction* of total steps, so a 4x larger `B_eff` yields 4x fewer
warmup steps to reach a higher LR on a randomly-initialised projector. That is a plausible
NaN that would look like a hardware bug.

---

## 9. Decommission — DONE

**Fully torn down 2026-07-26. Nothing of this project remains in GCP.**

| Resource | State |
|---|---|
| v6e-16 slice + queued resource | **deleted** — `europe-west4-a` verified clean, 0 QRs, 0 nodes |
| `gs://tayavision-eu` | **deleted**, bucket and all 963 MiB of contents |
| Trained weights | **preserved locally**, all five checkpoints, each MD5-verified against the remote before deletion |

### Where the weights are now

`outputs/tpu-align-v6e16-2026-07-25/` — 331 MB, **gitignored, so this directory is the only
copy that exists anywhere.** Back it up if it matters to you.

| File | MD5 (base64) |
|---|---|
| `checkpoint_1000.pt` | `mtB7CLQSRYeJlYqGwUCurQ==` |
| `checkpoint_2000.pt` | `l+/JVOTVksEkCBB3Ikm5vQ==` |
| `checkpoint_3000.pt` | `U6YFJxb22teEwhvTuzhBng==` |
| `checkpoint_4000.pt` | `EqvkeKHW/o1AJSWhLKI+vw==` |
| **`checkpoint_4360.pt`** | `Uw0Cd5zogH3iBH8Ew9gcMw==` — **the final, end-of-epoch projector** |
| `config.json` | — |

Each was byte-compared to its GCS object before the bucket was removed, so these are
faithful copies rather than assumed ones.

### Rotate the tokens

`gs://tayavision-eu/code/` held 45 deploy tarballs, each containing a live `.env`. Verified
at the time, not assumed:

```console
$ gcloud storage cat gs://tayavision-eu/code/latest.tar.gz | tar tzf - | grep '\.env$'
./.env
```

They are gone now, but **deleting objects does not undo a read that may already have
happened**. Rotate `HF_TOKEN` and `WANDB_API_KEY`.

If the tarball-ships-`.env` mechanism is used again, give it a lifecycle rule or fetch
secrets at boot instead of leaving snapshots at rest.

### Before the next TPU run

The bucket no longer exists, and `scripts/tpu/_lib.sh:20` still defaults `BUCKET` to
`gs://tayavision-eu`. Recreate it first — `setup_gcp.sh` is idempotent and creates it if
absent:

```bash
bash scripts/tpu/setup_gcp.sh          # recreates gs://tayavision-eu in europe-west4
TRC_PROFILE=v6e-16-ew4a bash scripts/tpu/launch_spot.sh
```

Everything else in section 4 then applies unchanged.

### Not ours — left alone

`gs://tinyaya-stage2-eu` (**391 GiB**, `europe-west4`) belongs to the sibling project
`tinyaya-stage2-scale`, which shares this GCP project and the same TRC grant. It was **not**
touched. At europe-west4 Standard pricing that is roughly **$8/month**, so it is worth
confirming it is still wanted — but that is a decision for that project, not this one.

## 10. Full provenance

| | |
|---|---|
| Slice | `v6e-16`, `europe-west4-a`, runtime `v2-alpha-tpuv6e`, spot |
| Topology | **4 hosts x 4 chips** — measured, not the 8 chips/host the public tables imply |
| Chips visible to SPMD | 16 (`xr.global_runtime_device_count()`) |
| Mesh | 1-D over 16 chips, `("data", "model")` = (16, 1) |
| `TPU_STRATEGY` | `replicated` — model is 8.06 GiB of 31.25, FSDPv2 buys nothing and costs the reduce-scatter NaN hazard |
| Bucket | `gs://tayavision-eu` (co-located) |
| Model | SigLIP2-so400m-p14-384 frozen + `MultiModalProjector` (11.5 M trainable) + `CohereLabs/tiny-aya-base` (3.35 B, frozen) |
| Params | 4.33 B total; 8.058 GiB resident per chip in bf16 |
| Dataset | LLaVA-Pretrain, 558,128 samples, sharded 139,532 per host |
| Boot | QR submit → ACTIVE ~4-6 min; deps ~2 min; redeploy → first step ~90 s + 3.5 min compile |

Related documents:

- `.claude/orchestration/playbook/baseline-v6e8-siglip-cohere2.md` — the protected baseline, all measured
- `.claude/orchestration/SPEC.md` — run loop, and §9 on what does not work yet
- `.claude/agents/tpu-diagnoser.md` — canonical log-signature → diagnosis table
- `docs/tpu/tpu-capacity-log.md` — full provisioning history
- `scripts/tpu/README.md` — every environment variable and its default
- `CLAUDE.md` — project guide, including the `training=alignment` trap
