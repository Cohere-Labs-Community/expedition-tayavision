# Performance metrics schema

Fields emitted to W&B `cataluna84/tayavision-tpu`. Deliberately a separate project from
`tayavision-instruct-sweep` and `tayavision-multilingual` so TPU bring-up noise never
lands in a results project.

Merged from `llm-architectures` (identity / training / throughput shape) and
`tinyaya-stage2-scale` (the XLA and XProf namespaces, which JAX had no use for and this
stack needs most).

## Identity

| Field | Source | Notes |
|---|---|---|
| `WANDB_RUN_NAME` | launch env | Human-readable, e.g. `v6e8-smoke-01` |
| `WANDB_RUN_ID` | launch env | **Fixed** for resumable runs — a preemption must rejoin, not fork |
| `TAYAVISION_RUN_TAG` | `deploy_tarball.sh` | Unique per deploy; baked into the launch log line so log segments can be scoped |
| `TRC_PROFILE` | launch env | Which slice produced the number. A step time without a slice is meaningless. |
| `TPU_STRATEGY` | launch env | `replicated` \| `fsdpv2_lora` \| `fsdpv2` |

Empty `WANDB_*` exports break `wandb.init` with `Run ID cannot be empty`. Launchers must
export only when set **and** the init path must scrub empties — either half alone has
been observed to fail.

## Training

| Field | Unit | Meaning |
|---|---|---|
| `train/loss` | nats | Per-step training loss |
| `train/lr` | — | Current LR |
| `train/grad_norm` | — | Pre-clip gradient norm |
| `val/loss` | nats | Validation loss |

## Throughput

| Field | Unit | Formula |
|---|---|---|
| `perf/step_time` | s | Measured wall time per optimizer step |
| `perf/p50_step_time`, `p90`, `p99` | s | Percentiles. **A mean hides tail stalls**, and on XLA the tail is where recompiles live. |
| `perf/images_per_sec` | img/s | images per step / step_time |
| `perf/tokens_per_sec` | tok/s | Text tokens; report alongside images, never instead |
| `perf/mfu` | fraction | achieved FLOP/s ÷ (peak × chips) |

## XLA and profiling

The namespace that matters most on this stack, and the one the JAX sibling has no
analogue for. Source: `torch_xla.debug.metrics.metrics_report()`.

| Field | Meaning |
|---|---|
| `xla/compile_cause_count` | Cumulative compilations. **Flat = steady state. Rising = a shape is varying.** This single counter is what distinguishes a normal 14-minute cold boot from a genuine hang, and it is what four of the VLM diagnosis rows key on. |
| `xla/compile_cause_delta` | Compilations since the last log. Should be 0 after warmup. Non-zero mid-run means dynamic shapes — see diagnoser rows on `nonzero`, MoonViT, and generation. |
| `xla/first_visible_step_compile_s` | Seconds from launch to the first step line. Expect ~840s (~14 min) cold. |
| `xla/aten_fallback_count` | Ops that fell back to CPU. `aten::nonzero` and `aten::_local_scalar_dense` here are the dynamic-shape smoking gun. |
| `xprof/profile_path` | GCS path of the captured trace |
| `xprof/trace_window` | Step range the trace covers |

Phase 0 of `../TPU_OPTIMIZATION_SPEC.md` instruments these seven XProf labels:
`data_fetch`, `device_transfer`, `forward_loss`, `backward`, `mark_step`,
`optimizer_step`, `logging_materialize`.

## Memory and host

| Field | Meaning |
|---|---|
| `mem/hbm_peak_gb` | Peak HBM per chip. Ceiling is **29 GiB** of 31.25 usable on v6e. |
| `mem/model_gb` | Materialized model size — expect ~7.5 GB bf16 |
| `host/input_gap_ms` | Time the device waited on the input pipeline |
| `host/device_transfer_ms` | Host→device copy time |

## Invariants that make numbers comparable

1. **196 image tokens per image** on the SigLIP path. A run at a different token count is
   not comparable and must not share a chart.
2. **Report the effective batch, not the per-device batch.** Per-device batch × chips ×
   grad-accum is the only quantity that compares across slices.
3. **Images per step is the memory-relevant axis**, not sequence length. A VLM's
   activation peak scales with images × 196, so an OOM is fixed by cutting images first.
4. **`TRC_PROFILE` and `TPU_STRATEGY` are part of every number.** Step time from a v6e-8
   and a v5e-64 are different measurements, not two samples of one.
5. **Phase 2' only**: the 40% English / 60% multilingual mix and T=5 temperature sampling
   must match the GPU run, or loss curves are not comparable across backends.

## Leaderboard fields

`run.summary` should carry final/best `val/loss`, total images seen, wall-clock, MFU,
peak HBM, and `TRC_PROFILE`. Verify all six before promoting anything into
`playbook/baseline-*.md` — and note that a TPU number never enters a `docs/*.md` results
table until the same entrypoint has run on both backends (`../SPEC.md` §7).
