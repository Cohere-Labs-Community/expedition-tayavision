# Optimization experiment matrix

Ported from `tinyaya-stage2-scale` and re-derived for a vision-language model. Governed
by `../TPU_OPTIMIZATION_SPEC.md`, which owns the gates; this file owns the candidate list.

**Status: nothing has been run.** Every row is a hypothesis. The `Result` column stays
empty until a real run fills it — an invented result here would propagate into
`baseline-*.md` and then into `docs/`.

## Standard run ladder

Kept verbatim from the source, because the shape is what makes results comparable:

| Rung | Steps | Purpose | Gate to advance |
|---|---:|---|---|
| **Smoke** | 20 | Does it compile and produce a step? | No crash; `xla/compile_cause_delta` → 0 |
| **Failure boundary** | 300 | Does it survive the window where NaN and OOM live? | Finite loss; HBM ≤ 29 GiB |
| **Promotion** | 1000 | Is it actually faster, and still correct? | p50 step time improves ≥5%; loss curve within noise of baseline |
| **Production** | full | The real run | — |

A candidate that fails at rung N is never promoted past it. The 300-step rung exists
because the inherited FSDPv2 NaN appeared at step 24-130 — a 20-step smoke would have
called it green.

## Candidates

| ID | Phase | Hypothesis | Change | Ladder | Promote if | Roll back if |
|---|---|---|---|---|---|---|
| `opt-0-metrics` | 0 | We cannot tune what we cannot see | Emit the `xla/*` + `xprof/*` fields in `perf-metrics-schema.md`; XProf labels on, tracing default-off | 20 | Counters appear and are plausible | Overhead >2% step time |
| `opt-1-log25` | 1 | Per-step logging forces a device sync | `log_every: 25` | 300 | Step time improves, loss curve unchanged | Loss becomes unreadable |
| `opt-2-warmup` | 2 | A zero-LR warmup step absorbs compile before real data | One compile-warmup macro-step | 300 | `first_visible_step_compile_s` unchanged, first real step faster | No measurable change |
| `opt-3-imgbatch` | 3 | Images/step is the memory axis on a VLM, not sequence length | Sweep images/step at fixed effective batch via grad-accum | 20→300 | HBM headroom without step-time loss | OOM or NaN |
| `opt-4-nockpt` | 4 | Gradient checkpointing may be unnecessary at `replicated` | `xla_grad_checkpoint: false` | 20→300 | Faster and HBM ≤ 29 GiB | OOM |
| `opt-5-mpdl` | 5 | Host input pipeline starves the device | `MpDeviceLoader` + prefetch; `num_workers` sweep | 300 | `host/input_gap_ms` drops | No change — then the bottleneck is elsewhere |
| `opt-6-nonzero` | 6 | **The highest-value row.** `nonzero()` in the image-token scatter is a dynamic-shape op that recompiles per distinct image count | Replace `nonzero` + `index_put` in `models/tiny_aya_vision.py` with a static-shape `torch.where`, **behind the backend seam** so CUDA keeps its own path | 20→300 | `compile_cause_delta` → 0 mid-run; `aten_fallback_count` drops | GPU path changes numerically — it must not |
| `opt-6-moonvit-buckets` | 6 | MoonViT's variable token count can be bucketed | Fixed token buckets, every bucket graph prewarmed | 300 | Compile count bounded by bucket count | Unbounded compiles — then MoonViT stays GPU-only |
| `opt-7-strategy` | 7 | `replicated` may beat FSDPv2 outright here | A/B `replicated` vs `fsdpv2_lora` on the Cohere2 backbone | 300 | `replicated` fits and is faster | HBM > 29 GiB at target batch |
| `opt-7-flash` | 7 | Flash attention on the SigLIP encoder | Attention backend A/B | 300 | Faster, loss unchanged | Any loss delta |
| `opt-7-scan` | 7 | `scan_layers` reduces compile time | Isolated probe only | synth→300 | Compile time drops materially | Any of the three scan signatures fires — see the diagnoser |
| `opt-8-promote` | 8 | The selected config holds at length | Full selected config | 1000→full | All gates in `../TPU_OPTIMIZATION_SPEC.md` §5 pass | Any gate fails |

### Why `opt-7-scan` is last and isolated

`scan_layers` requires homogeneous, pure layers. Phase 2 puts LoRA on layers **18-35
only**, so layers 0-17 have a different parameter structure by construction — this repo
is *more* likely to hit `Layer N has mismatched keys` than the source project was. It is
a probe, never a default. `use_scan_layers: false` is in the protected config.

### Why `opt-7-strategy` might close several rows at once

Phase 1 trains ~11.5M params of ~3.75B; the model is ~7.5 GB bf16 against 31.25 GiB/chip.
If `replicated` fits comfortably — and the arithmetic says it should — then the inherited
FSDPv2 per-layer reduce-scatter NaN is **sidestepped rather than worked around**, and the
wrap-policy hazard in `../SPEC.md` §7 stops being a live risk for Phase 1 entirely. That
would be the single most valuable result in this table, and it is cheap to get.

## Required record fields

Every promoted or rejected candidate records: `ID` · date · `TRC_PROFILE` ·
`TPU_STRATEGY` · git SHA · W&B run id · steps run · p50/p90 step time · peak HBM ·
final loss · `compile_cause_count` at end · verdict (`promote` / `reject` / `rollback` /
`needs-more-data`) · one-line rationale.

A candidate without a W&B run id and a git SHA is not a result, it is a recollection.
