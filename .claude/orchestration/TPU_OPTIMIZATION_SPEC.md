# TPU Optimization Spec — Tiny Aya Vision (torch_xla)

**Version:** v1 (2026-07-25)
**Ported from:** `tinyaya-stage2-scale` v2. `llm-architectures` deliberately dropped this
file as "torch_xla-specific with no analogue in pure JAX" — which was correct there and
is exactly why it comes back here.
**Owns:** the protected TPU config, the promotion gates, and the phase program.
**Does not own:** failure recovery (`SPEC.md`), signatures (`agents/tpu-diagnoser.md`),
or the candidate list (`playbook/optimization-experiment-matrix.md`).

> **Nothing in this file has been measured on Tiny Aya Vision.** The protected config is
> a *starting position* derived from a sibling project on the same backbone, not a
> validated baseline. Phase 0 exists to replace assumption with measurement.

## 1. Goal

Get a training step onto TPU that is **correct first, then fast**. On XLA those are not
the same problem and the order matters: a config that is fast because it silently
recompiles per batch, or that fits because it trains on 1/N of the data, will look great
on a dashboard.

## 2. Protected config

The starting position. Changing any line requires a candidate row in the experiment
matrix and a passed gate.

```yaml
# Sharding
tpu_strategy: replicated        # NOT fsdpv2 -- see §4 and SPEC.md §7
                                # Phase 1 trains ~11.5M of ~3.75B params;
                                # ~7.5 GB bf16 fits 31.25 GiB/chip comfortably.

# Precision + memory
precision: bfloat16
xla_grad_checkpoint: true       # conservative default; opt-4-nockpt tests removing it
hbm_ceiling_gib: 29             # of 31.25 usable on v6e

# Compilation
use_scan_layers: false          # LoRA on layers 18-35 only => heterogeneous layers
                                # => scan's _ensure_same_structure will fail. Probe only.
compile_warmup_steps: 0         # opt-2-warmup tests 1
persistent_xla_cache: false     # nondeterministic keys on v6e SPMD; never warm-hits,
                                # and costs the UNIMPLEMENTED deserialize failure

# Vision
vision_encoder: siglip          # fixed 196 tokens/image.
                                # MoonViT is variable-token => a new HLO per shape.
image_tokens: 196               # invariant, enforced by VERIFY config-contracts

# Data
distributed_sampler: false      # SPMD is ONE process. Leaving this on trains
                                # on 1/N of the dataset and reports nothing wrong.
log_every: 1                    # opt-1-log25 tests raising it
```

## 3. Research basis

Stack-level findings, not model-level. These carry across projects because they are
properties of torch_xla.

| Finding | Consequence here |
|---|---|
| XProf `xp.Trace()` labels are the only way to attribute step time on XLA | Phase 0 instruments seven labels before any tuning decision |
| `.item()` / `.cpu()` force a device sync and break the lazy graph | `models/tiny_aya_vision.py` calls `.item()` on the image-token count every forward — see `opt-0-metrics` |
| `MPDeviceLoader` overlaps host input with device compute | `opt-5-mpdl`; only worth it once `host/input_gap_ms` proves the pipeline starves the device |
| FSDPv2 SPMD is the right memory path at scale, but `scan_layers` requires homogeneous **and pure** layers | Both are Phase 7 probes here, not defaults |
| Dynamic-shape ops (`nonzero`, boolean masking) either fall back to CPU or recompile per shape | The single biggest VLM-specific risk — `opt-6-nonzero` |

## 4. The inherited NaN, and why `replicated` is the default

`tinyaya-stage2-scale`, running **the same Cohere2 backbone with the same 36 decoder
layers**, hit NaN loss at step 24-130 under FSDPv2. Root cause: the auto-wrap policy
wrapped each `Cohere2DecoderLayer`, producing 36 separate bf16 reduce-scatters.
FSDPv2 has no `fp32_reduce_scatter` (FSDPv1 does — pytorch/xla #3588 / #8056); the
accumulated bf16 error goes non-finite. Refs: pytorch/xla #8591, #8778.

Their fix was a custom wrap policy with one outer reduce-scatter.

**This repo's position is that the question may not arise.** Phase 1 trains ~11.5M
trainable params; Phase 2 adds LoRA r=256 on 18 layers, roughly 75M more. Model ~7.5 GB
bf16, optimizer state ~1 GB, against 31.25 GiB per chip. `replicated` fits with room.

So `opt-7-strategy` is not a micro-optimization — it is the cheapest way to find out
whether an entire class of failure applies to this model at all. Run it early.

If FSDPv2 does turn out to be needed: **the wrap policy matches on
`type(module).__name__`.** A wrong class-name string wraps nothing, silently falls back
to replicated, and looks like a successful FSDPv2 run. Print the wrap count at startup
and verify the name against the installed `transformers`.

## 5. Global gates

A candidate must pass **all six** to be promoted. Any single failure is a rollback.

| # | Gate | Threshold |
|---|---|---|
| 1 | **Compile** | `xla/compile_cause_delta` reaches 0 after warmup and stays there. A rising counter mid-run is a hard fail regardless of step time. |
| 2 | **Stability** | Finite loss through the 300-step rung. NaN is never "retry and see" — see §4. |
| 3 | **Memory** | `mem/hbm_peak_gb` ≤ **29 GiB** on v6e. |
| 4 | **Quality** | Loss curve within noise of the baseline at the same step count. A speedup that moves the loss curve is not a speedup. |
| 5 | **Throughput** | p50 step time improves ≥5%, and p90 does not regress. A better mean with a worse tail is a recompile hiding. |
| 6 | **Rollback** | The change is revertible by one config line or one commit. Anything requiring a coordinated multi-file revert does not get promoted. |

## 6. Phases

Ordered by dependency, not by expected payoff. Phase 0 is mandatory; without it every
later phase is guesswork.

| Phase | Purpose | Promotion |
|---|---|---|
| **0 — Instrument** | Emit the `xla/*`, `xprof/*`, `mem/*`, `host/*` fields. XProf labels: `data_fetch`, `device_transfer`, `forward_loss`, `backward`, `mark_step`, `optimizer_step`, `logging_materialize`. Tracing default-off. | Counters present, overhead <2% |
| **1 — Logging cadence** | Per-step logging forces syncs | Step time improves, loss still readable |
| **2 — Compile warmup** | Absorb first-step compile with a zero-LR macro-step | First real step faster |
| **3 — Batch shape** | Find the images/step the HBM ceiling allows | Gate 3 holds at target effective batch |
| **4 — Recompute** | Test whether grad checkpointing is needed at `replicated` | Faster, gate 3 holds |
| **5 — Input pipeline** | Only if `host/input_gap_ms` shows starvation | Gap drops |
| **6 — Dynamic shapes** | **The VLM-specific phase.** Static-shape image-token scatter; MoonViT bucketing | Gate 1 holds mid-run |
| **7 — Sharding + kernels** | `replicated` vs FSDPv2; flash attention; `scan_layers` probe | Per-candidate |
| **8 — Promote** | Full selected config at length | All six gates |

### Phase 6 is where the real work is

Three of this repo's own code paths produce dynamic shapes under XLA:

1. **`models/tiny_aya_vision.py` image-token scatter** — `special_image_mask.nonzero()`
   plus advanced indexing. This exists because `masked_scatter_backward` is broken under
   inductor on **CUDA**; on XLA `nonzero` is a dynamic-shape op. The two backends want
   opposite code, which is precisely what the `src/backend/` seam is for. Do **not** fix
   this by reverting to `masked_scatter` — that breaks the GPU path.
2. **`n_image_tokens.item()`** in the same function — a `torch.compiler.is_compiling()`
   bypass guards it, but that predicate is a **Dynamo** check and is False under
   torch_xla LazyTensor tracing, so the guard syncs every forward. Confirm with
   `met.metrics_report()` before acting.
3. **MoonViT variable tokens** — `num_tokens_after_shuffle: -1` means a new HLO program
   per distinct image resolution. Unbucketed, MoonViT is not TPU-eligible.

### Phase 7 scan hazards

`scan_layers` has three known failure signatures, all inherited and all in the diagnoser
as Phase-7-only hazards rather than live rows: `Layer N has mismatched keys`,
`FakeTensor.*aten.index_select`, and `unexpected keyword attention_mask`. This repo is
*more* exposed than the source project because Phase 2 makes layers 18-35 structurally
different from 0-17.

## 7. Experiment record

Every candidate records: `ID` · date · `TRC_PROFILE` · `TPU_STRATEGY` · git SHA ·
W&B run id · steps · p50/p90 step time · peak HBM · final loss ·
`compile_cause_count` at end · verdict · one-line rationale.

Full field list and the candidate table: `playbook/optimization-experiment-matrix.md`.

## 8. Stop conditions

Stop the program and escalate when:

1. Two consecutive candidates fail the same gate — the model is telling you the gate is
   wrong or the bottleneck is elsewhere.
2. NaN appears at all. Go to §4, not to the next candidate.
3. `compile_cause_delta` never reaches 0 — a shape is varying and no tuning matters until
   Phase 6 closes it.
4. Peak HBM exceeds 29 GiB at the minimum useful batch.
5. A candidate needs a change to `pipeline/train_*.py` that is not behind the backend
   seam. That is a `SPEC.md` §9 problem, not an optimization problem.
6. Wall-clock spent compiling exceeds wall-clock spent training over a full session.
7. The slice is preempted more than twice during a single candidate — the measurement is
   contaminated; rerun rather than reporting it.
8. Any result would need to be written into a `docs/*.md` results table. TPU numbers do
   not go there until the same entrypoint runs on both backends.
