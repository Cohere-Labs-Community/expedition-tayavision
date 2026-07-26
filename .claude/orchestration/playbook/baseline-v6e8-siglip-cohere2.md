# Protected baseline — v6e-8, SigLIP + Cohere2 (Tiny Aya Vision)

> **The first real Tiny Aya Vision TPU run happened 2026-07-25** (v6e-16, W&B run
> `2726cdff`, 20 optimizer steps). Slice, boot, batch, throughput, memory, and loss rows
> are now **measured**. What remains `TBD` is the long-run behaviour: full-epoch loss,
> checkpoint/resume, and the regression thresholds.
>
> Inherited figures stay confined to the clearly-marked non-authoritative section at the
> bottom; they belong to a different model. Nothing measured here came from there.
>
> Add `baseline-<slice>-<arch>.md` for a new slice rather than editing this one — a
> protected baseline that gets edited in place stops being a baseline.

## Slice

> **Measured on `v6e-16-ew4a`, not `v6e-8-eu`, on 2026-07-25.** v6e-64 and v6e-32 both
> failed to provision (see `docs/tpu/tpu-capacity-log.md`); v6e-16 came up clean. Slice
> rows below are real; **model rows are still TBD** — nothing but a synthetic matmul has
> run. Rename or split this file when a v6e-8 baseline exists.

| Quantity | Value | Source |
|---|---|---|
| `TRC_PROFILE` | `v6e-16-ew4a` | measured |
| Accelerator | `v6e-16`, **4 workers** | measured — 4 network endpoints, i.e. 4 chips/host, not the 8 public topology tables imply |
| Chips visible to SPMD | **16** (`global_runtime_device_count`) | measured |
| Mesh | 1-D over 16 chips, single process | measured |
| Zone / runtime | `europe-west4-a` / `v2-alpha-tpuv6e` | — |
| Bucket | `gs://tayavision-eu` (co-located) | created 2026-07-25 |
| `torch` / `torch_xla` | **2.9.1+cpu / 2.9.0**, `libtpu` 0.0.21 | measured |
| Python | **3.12 exactly** — floored by pyproject, ceilinged by libtpu cp312 | measured |
| `TPU_STRATEGY` | `replicated` | default; `opt-7-strategy` untested |
| HBM per chip | 31.25 GiB usable; **29 GiB working ceiling** | inherited, unverified here |

## Boot and provisioning (measured 2026-07-25)

| Phase | Duration |
|---|---|
| QR `WAITING_FOR_RESOURCES` → `PROVISIONING` | ~2 min |
| `PROVISIONING` → node `READY/HEALTHY` → QR `ACTIVE` | ~3.5 min |
| **Submit → ACTIVE, total** | **~5.5 min** |
| apt + uv + tarball fetch + extract | ~25 s |
| `uv sync` + torch/torch_xla install | ~2 min |
| T2 redeploy (deps cached) → first step line | **~90 s** |

**The inherited "~14 min cold-boot recompile" did not reproduce**, but this workload is a
single trivial matmul. Do not treat ~90 s as the model's compile time — that number is
still TBD and is expected to be far larger.

## Synthetic step timing (NOT a model number)

One `(8,196,2048) @ (2048,2048)` bf16 matmul, i.e. the projector output shape only.

| Metric | Value |
|---|---|
| step 1 (compile) | **1.80 s** |
| p50 warm (steps 4-20) | **0.0005 s** |
| `xla/compile_cause` markers | **1** — flat after step 1, which is the correct signal |
| `aten::nonzero` | **absent** (expected — no model code ran) |

Compile is isolated to the first step and the counter stays flat, which is exactly the
gate-1 behaviour `TPU_OPTIMIZATION_SPEC.md` requires. The 0.0005 s is a ~13 GFLOP op on
~14 PFLOP/s of silicon; it is overhead-dominated and says nothing about model throughput.

## Model materialized on TPU — MEASURED 2026-07-25

`tpu_smoke.py --load-backbone` on the live v6e-16. Gated `CohereLabs/tiny-aya-base`
downloaded and materialized; **gate 3 is comfortably satisfied.**

| Quantity | Value |
|---|---|
| Params total | **4.33 B** |
| Params trainable (bare constructor) | **3.90 B (90%)** — see note below |
| **HBM resident per chip** | **8.058 GiB** |
| HBM total per chip | **31.246 GiB** (confirms the inherited 31.25) |
| Headroom under the 29 GiB working ceiling | **~21 GiB** |
| Chip | `TPU v6e chip`, 4 local chips/host |
| First step with model resident | 2.62 s (vs 1.79 s without) |
| Compile-cause markers | 1 — flat after warmup |

8.058 GiB matches `4.33e9 x 2 bytes` exactly, which is the check that the figure is real
rather than an artifact.

**`replicated` fits with room to spare, and that largely answers `opt-7-strategy`.** At
8.06 GiB of a 29 GiB working ceiling, the whole model sits on every chip with ~21 GiB
left for activations, gradients, and optimizer state. The inherited FSDPv2 per-layer
reduce-scatter NaN (`../SPEC.md` §7) is therefore **avoidable rather than survivable** on
this model — do not reach for FSDPv2 until a real memory pressure appears.

### Measured with a real training step (2026-07-25)

| Config | Per-chip HBM | Outcome |
|---|---|---|
| B_global 256, seq 640, **no input sharding** | — | **OOM.** HLO showed ~20 live `bf16[64,640,11008]` temps at 860 MB each |
| B_global 256, seq 640, **input sharding on** | **17.14 / 31.2 GiB** | fits, under the 29 ceiling |

### Do not trust the analytical activation model

It has now been wrong **three times in one session**, always low:

1. Used `hidden_size` 2048 for the dominant term. Cohere2's FFN `intermediate_size` is
   **11008** — 5.38x larger, and it is the MLP intermediates that actually dominate.
2. Assumed the per-chip batch was `B_global / chips`. Nothing sharded the input, so every
   chip held the whole per-**host** batch of 64. `MpDeviceLoader` does not shard by
   itself; it needs `input_sharding=ShardingSpec(mesh, ("data", None))`.
3. Even corrected, it predicted 10.2 GiB against a measured 17.1 — XLA keeps far more
   fused temporaries alive concurrently than "boundaries plus a working set" suggests.

**Gate 3 of `../TPU_OPTIMIZATION_SPEC.md` is enforced by outcome, not by arithmetic.**
Size batches by bisection and read `mem/used_gib`; treat the formula as an order-of-
magnitude sanity check only, and never as a reason to skip the measurement.

> **Trainable-parameter caveat.** 3.90 B trainable is the *bare constructor*: only
> `SigLIPVisionEncoder.__init__` freezes anything. Phase 1's ~11.5 M-trainable figure
> comes from the training pipeline freezing the LLM, not from the model class. Do not
> read 90% as a bug, and do not read it as the Phase-1 configuration either.

### How the HBM figure is obtained, and one correction

`xm.get_memory_info()` is unusable under SPMD: it defaults to `xla_device()`, which is
the *virtual* device `SPMD:0`, and `_xla_memory_info` rejects it. `torch_xla.devices()`
returns that same virtual device. The raw strings from
`_XLAC._xla_get_all_runtime_devices()` fail differently — `get_memory_info` calls
`str(device)` and `torch.device` cannot parse the format.

`tpu_info.metrics.get_chip_usage()` reads libtpu's gRPC metrics server and **works**.

An earlier entry here claimed its `MEMORY_USAGE` gauge "does not track program
allocations", based on pinning 8 GiB of `torch.zeros` and seeing ~0. **That conclusion
was wrong and is retracted.** The gauge is fine; the test was not — a buffer that no
computation consumes is dead-code-eliminated by XLA before it is ever allocated. Loaded
weights cannot be elided, which is why they register. The lesson generalises: on a lazy,
optimising backend, *allocating* something is not the same as it existing.

## Model

| Quantity | Value |
|---|---|
| Vision encoder | `google/siglip2-so400m-patch14-384`, frozen, ~400M |
| Image tokens | **196** (384/14 → 27×27=729 → pad 28 → pixel-shuffle 2 → 196) |
| Connector | `MultiModalProjector`, ~11.5M trainable |
| LLM | `CohereLabs/tiny-aya-base` — Cohere2, 3.35B, **36 layers**, d=2048 |
| Total / trainable (Phase 1) | ~3.75B / ~11.5M |
| Precision | bf16 |

## Batch configuration — MEASURED 2026-07-25 (v6e-16, run `2726cdff`)

| Quantity | Value |
|---|---|
| Images per **chip** per micro-step | **8** |
| Per-host DataLoader batch | **32** (= 128 / 4 hosts) |
| Global images per micro-step | **128** |
| Grad accumulation | **2** |
| **Effective batch** | **256 — identical to the Modal GPU baseline** |
| Sequence length | **640**, fixed (`training.fixed_seq_len=640`) |
| Micro-steps per epoch | **4,360** (139,532 per host / 32, `drop_last`) |
| Optimizer steps per epoch | **2,180** |

`B_eff` matching the Modal baseline is deliberate: it is invariant #2 of
`perf-metrics-schema.md`, and it is what makes this run's loss curve comparable to the GPU
one. LR stays 1e-3 and `warmup_ratio` 0.03 for the same reason.

**Global batch 256 does NOT fit** — see the memory section. 128x2 was chosen to preserve
`B_eff` while halving activation memory.

## Throughput — MEASURED 2026-07-25

| Metric | Value |
|---|---|
| `perf/step_time` p50 (per **optimizer** step, i.e. 2 micro-steps) | **1.494 s** |
| Spread, steps 4-19 | **1.491-1.503 s** — essentially zero variance |
| `perf/images_per_sec` | **171.4** |
| Step 1 (compile) | **160.8 s** |
| Step 2 (residual compile) | **54.8 s** |
| First warm step | step 4 |
| `xla/compile_cause_count` at steady state | **1, FLAT for all 19 steps** |
| Projected 1-epoch wall clock | **~54 min** (2,180 x 1.494 s) |

The flat compile counter is the gate-1 signal from `TPU_OPTIMIZATION_SPEC.md` and it
confirms the fixed-`S` collate: with dynamic padding every batch is a new shape and this
number climbs forever.

**The inherited "~14 min cold-boot recompile" still did not reproduce** — 160.8 s for a
real model step, not 840 s.

## Memory — MEASURED 2026-07-25

| Metric | Value |
|---|---|
| `mem/used_gib` peak, B_global=128 | **27.80** of 31.25 |
| `mem/used_gib` typical | 25.2-25.3 |
| `mem/total_gib` | **31.246** |
| B_global=256 | **OOM at step 2** — "reserve 16.25G ... 15.69G free" |

At B_global=256 the run completed step 1 (loss 9.0502) and died loading step 2's program.
The margin at 128 is real but not generous: **27.80 of a 29 GiB working ceiling.**

### The analytical activation model has now been wrong FOUR times, always low

The fourth: predicted ~19.4 GiB for B_global=128 by halving the measured 256 figure;
actual peak **27.8**. Halving the batch did **not** halve total memory, because the static
term and XLA's live fused temporaries do not scale with batch.

**Gate 3 is enforced by outcome, not arithmetic** — bisect the batch and read
`mem/used_gib`. Treat any formula here as an order-of-magnitude check only.

## Loss reference — MEASURED 2026-07-25 (20 optimizer steps, B_eff 256, LR 1e-3)

| Opt step | `train/loss` | `train/grad_norm` | `train/lr` |
|---:|---|---|---|
| 1 | 9.053 | — | 1.54e-05 |
| 5 | 8.519 | — | 7.7e-05 |
| 10 | 7.365 | — | 1.54e-04 |
| 15 | 6.727 | — | 2.31e-04 |
| 19 | **5.714** | 19.50 | 2.92e-04 |

Monotone apart from small wiggles at steps 9, 11-12. Still inside warmup at step 19
(`warmup_ratio` 0.03 of 2,180 = 65 steps), so the LR has not yet reached 1e-3.

`train/max_image_tokens` = **196** every step, which is the SigLIP token invariant holding.
`norms/projector_token_mean` 8.64, `norms/emb_matrix_mean` 1.23.

## Checkpoint + resume

| Check | Status |
|---|---|
| `SAVE_CKPT_DIR` reaches the training code | **yes** — `pipeline/utils.gcs_checkpoint_root()` |
| Exactly one rank writes | **yes, measured** — rank 0 logged `Saved checkpoint`, the other three silent |
| Objects written per save | **1** — `<root>/<run_id>/checkpoint_<micro_step>.pt`, 69 MB |
| Checkpoint mirrored to `gs://` | **verified end-to-end 2026-07-26** |
| Resume with an EMPTY local `/models` | **verified end-to-end 2026-07-26** |
| Optimizer + LR-scheduler state restored | **yes** — `train_alignment.py:694-696` |
| Resumed run rejoins the same W&B run id | **yes** — `wandb: Resuming run a2d64c8e...` |
| Resume from `TAYAVISION_RESUME=auto` | **verified end-to-end 2026-07-26** — resolves the run id, fetches, resumes, rejoins the same W&B run, with no Hydra `resume=` override |
| Default deploy does NOT silently resume | **verified** — `deploy_tarball.sh` defaults to `off`, so a redeploy against a prefix full of checkpoints still starts a fresh run id |

### How durability was verified (not asserted)

1. Ran with `save_steps=2`, saw `Saved checkpoint ...` immediately followed by
   `Mirrored checkpoint to gs://.../<run_id>/checkpoint_N.pt`.
2. Confirmed the objects independently with `gcloud storage ls -l` — 2 objects, 132 MiB.
3. **Deleted `/models/<run_id>` on all four hosts** to simulate a recycle.
4. Relaunched with `resume=<run_id>`; the log showed
   `Fetched gs://.../checkpoint_17.pt -> /models/...` then `Resuming from step 17`,
   and training continued and mirrored again.

Two real bugs surfaced only because step 3 was actually performed:

- **`torch.load(map_location=<xla device>)` raises.** "don't know how to restore data
  location of torch.storage.UntypedStorage (tagged with xla:0)" — and the tag names the
  *target*, not the file. TPU now loads to host and lets `load_state_dict` place it.
- **The resumed step counter double-counted.** `step = step_offset + _i` is right only on
  the GPU *subset* path. On the TPU *skip* path the loader still yields the whole epoch,
  so `_i` is already absolute. Resuming from step 8 reported step 16, tripped `max_steps`
  instantly, and exited "Training complete" having done no work at all.

Neither is visible without a real resume, which is why "it writes to `gs://`" is not the
same claim as "it survives a preemption."

## Regression triggers

Fill the thresholds once the baseline exists. The shape:

- p50 step time > ~2× baseline, sustained, after ruling out a recompile.
- `xla/compile_cause_delta` non-zero mid-run — a shape is varying; see diagnoser rows on
  `nonzero`, MoonViT, and generation.
- Peak HBM > 29 GiB.
- MFU below TBD%.
- Any change in images per step or the 196-token invariant — that makes the number
  incomparable rather than merely worse.

---

## Non-authoritative inherited envelope

**Different model. Do not cite these as Tiny Aya Vision numbers.** From
`tinyaya-stage2-scale` v6e-16 hardening probes (2026-07-14), recorded in
`docs/tpu/tpu-capacity-log.md`. They are here because they bound what is plausible on
first boot, not because they predict this model.

| Quantity | Observed (speech model, v6e-16) | Transfers? |
|---|---|---|
| Step time | 1.79-1.85 s | **No** — different model, different batch |
| HBM peak | 25.7-27.8 GiB of 31.25 | **Partly** — the 31.25 GiB ceiling is silicon, the usage is not |
| Cold-boot recompile | ~14 min | **Yes** — a property of torch_xla on v6e SPMD |
| XLA persistent cache hit rate | ~0, nondeterministic keys | **Yes** — toolchain behaviour |

The two "yes" rows are why `tier-definitions.md` budgets 20 minutes for T1 and why the
watchdog's `compiling` threshold is 20 min rather than the JAX sibling's 10.
