---
name: tpu-diagnoser
description: Deterministic log-signature classifier for Tiny Aya Vision torch_xla/TPU failures. Feed it log excerpts and a met.metrics_report() dump; it returns classification, tier, and the recommended action.
model: inherit
tools: Read, Grep
---

# tpu-diagnoser

This table is **canonical**. `.claude/orchestration/playbook/diagnosis-table.md` points
here rather than duplicating it. Tier meanings and policy live in
`.claude/orchestration/playbook/tier-definitions.md`.

Classify **first match wins**, top to bottom. When two rows match, prefer the
higher-intervention one. Return: classification, the matching line(s), tier, and the
recommended action verbatim. Never edit files, restart processes, or touch queued
resources — report, and let the orchestrator act.

## Provenance

- **[xla]** — inherited from `tinyaya-stage2-scale` (torch_xla, same Cohere2 backbone).
  Observed there, not yet here.
- **[infra]** — inherited from `llm-architectures`. Stack-independent; these are
  properties of GCE/TRC/boot, and transfer exactly.
- **[vlm]** — new, derived from *this repo's code* with a named location. **Predictions,
  not history** — no Tiny Aya Vision TPU run has happened. Treat a [vlm] row as "here is
  where to look first", not "this is what went wrong before".

| # | Signature (regex-ish) | Classification | Action | Tier |
|---|---|---|---|---|
| 1 | `ImportError.*libpython3\.12\.so\.1\.0` **[xla]** | `libpython` — `_XLAC.so` dynamically links libpython, which uv keeps outside the loader path | `startup_script.sh` must export `LD_LIBRARY_PATH` from the uv Python root. Present since the port; if this fires, that block was dropped or uv moved its root. | T2 |
| 2 | `OSError.*gated repo` OR `401 Client Error.*tiny-aya` OR `Cannot access gated repo` **[vlm]** | `hf-gated` — `CohereLabs/tiny-aya-{base,global}` needs an authorised `HF_TOKEN`, which reaches the VM only inside the tarball's `.env` | Run `ops.sh preflight` **before** deploying. Confirm `.env` exists at repo root, carries `HF_TOKEN`, and that the token has access. Neither source repo has gated weights; this fires before step 1 on every worker and is unique to this project. | T2 |
| 3 | `code tarball not found` **[infra]** | `boot-code-missing` | Upload via `deploy_tarball.sh`, or resubmit via `launch_spot.sh` (auto-tars at submit). | T2 |
| 4 | `Unable to acquire the dpkg frontend lock` **[infra]** | `boot-apt-race` | `startup_script.sh` retries; if a worker is still stuck, re-run its startup **detached** — never a foreground ssh with a local `timeout`. | T1→T2 |
| 5 | `Failed to deserialize executable: UNIMPLEMENTED` **[xla]** | `xla-cache` — persistent-cache entry unreadable | Remove `XLA_PERSISTENT_CACHE_PATH`. The cache has **nondeterministic keys on v6e SPMD and never warm-hits anyway**; it buys nothing and costs this failure. It is off in the protected config. | T2 |
| 6 | `loss.*(nan\|NaN\|inf)` around step 20-150, FSDPv2 active **[xla]** | `fsdp-reduce-scatter-nan` — 36 per-layer `Cohere2DecoderLayer` wraps → 36 bf16 reduce-scatters → pytorch/xla #8591 / #8778 | **Do not retry.** Remove the decoder-layer class from the auto-wrap policy so there is one outer reduce-scatter. FSDPv2 has no `fp32_reduce_scatter` (FSDPv1 only, #3588/#8056). On this model the clean move is `TPU_STRATEGY=replicated`, which the parameter budget permits — `TPU_OPTIMIZATION_SPEC.md` §4. | T4 |
| 7 | FSDPv2 reports 0 wrapped modules, OR per-chip HBM ≈ full model size **[vlm]** | `fsdp-wrapped-nothing` — the policy matches `type(module).__name__`; a stale class name wraps zero modules and silently falls back to replicated | Print the wrap count at startup. **For Phase 1 this is expected and fine** (11.5M trainable of 3.75B) — use `replicated` explicitly rather than an FSDPv2 that no-ops. For Phase 2, verify the class name against the installed `transformers`. | T2 |
| 8 | `RESOURCE_EXHAUSTED` OR `Out of memory` OR `exit code 137` **[xla]** | `oom` | Cut **images per step** first — a VLM's activation peak scales with images × 196 tokens, not text length. Hold the effective batch with grad-accum. Ceiling is 29 GiB of 31.25 usable on v6e. | T2 |
| 9 | `Image token count \(\d+\) != image feature count` **[vlm]** | `image-token-mismatch` — `models/tiny_aya_vision.py`; the processor expanded a different number of `<image>` placeholders than the encoder produced | A config mismatch between `config/vision/*.yaml` and the processor. Run `.claude/VERIFY.md`'s `config-contracts` block locally — it derives all five token quantities and catches this without burning a slice. | T2 |
| 10 | `xla/compile_cause_count` rising monotonically with steps, no error, MoonViT active **[vlm]** | `dynamic-shape-recompile-moonvit` — `src/processing.py` returns `H*W` per image from `image_grid_hws`, so sequence length varies per batch and XLA compiles per shape | **MoonViT is not TPU-eligible unbucketed.** Pin SigLIP (fixed 196), or implement `TPU_OPTIMIZATION_SPEC.md` Phase 6 buckets with every bucket graph prewarmed. Do not try to close PLAN P1 on TPU. | T2 |
| 11 | `aten::nonzero` in the fallback counters, OR step time correlated with images-per-batch **[vlm]** | `dynamic-scatter` — `models/tiny_aya_vision.py` uses `special_image_mask.nonzero(as_tuple=False)` + advanced indexing | **This is a CUDA workaround that is wrong on XLA.** `index_put` was chosen because `masked_scatter_backward` mis-shapes under inductor; `nonzero` is a dynamic-shape op that on XLA falls back to CPU or recompiles per distinct image count. Needs a static-shape form behind the backend seam — **not** an unconditional revert to `masked_scatter`, which breaks the GPU path. Candidate `opt-6-nonzero`. | T2 |
| 12 | `aten::_local_scalar_dense` once per step, no error **[vlm]** | `guard-sync` — `n_image_tokens.item()` in `_merge_image_features`. The `torch.compiler.is_compiling()` bypass above it is a **Dynamo** predicate and is False under torch_xla LazyTensor tracing, so the guard syncs every forward | Confirm with `met.metrics_report()` on the first smoke run before acting. Fix is to widen the bypass to cover XLA tracing, or move the check behind a debug flag. Candidate `opt-0-metrics`. | T1 |
| 13 | Compile count explodes at the sample-generation step; run never returns **[vlm]** | `generate-recompile` — `train_alignment.py` calls `generate_samples()` for logging; `_prepare_cache_for_generation` forces `DynamicCache`, whose growing shape means a new HLO program per decoded token | Disable in-training sample generation on TPU. The `DynamicCache` override exists to dodge a Cohere2 `HybridCache` prefill hang on **CUDA** and does not transfer. | T2 |
| 14 | Compile-cause count rising, no error, elapsed < 30 min **[xla]** | `compile-normal` | Continue. ~14 min cold-boot recompile is normal on this stack. | T0 |
| 15 | TPU duty ≈ 0% + HBM > 50% + no step line + elapsed > **20 min** **[xla]** | `compile-stall` | Ensure `python -u`, confirm `XLA_PERSISTENT_CACHE_PATH` is unset, dump `met.metrics_report()`. **Elapsed alone cannot classify this** — read the compile-cause counter. Check rows 10-13 first; three of the four VLM signatures present as a stall. | T1→T2 |
| 16 | Compile-cause count rising, no error, elapsed > 60 min **[xla]** | `compile-runaway` | Add `XLA_IR_DEBUG=1` on the next deploy and diff graph shapes. Rows 10-13 again — a dynamic shape is the usual cause, and it is one line of Python, not an XLA bug. | T2 |
| 17 | `PJRT.*DEADLINE_EXCEEDED` OR `Coordination service.*Deadline`, multi-host only **[infra]** | `rendezvous-failure` — a worker never joined | v6e-8 is single-host and cannot produce this. On a multi-host profile check **all** workers' `/tmp/startup.log`; re-run startup on stragglers **detached**, then redeploy. Root cause seen upstream: an apt/dpkg lock race at boot, not the framework. | T2 |
| 18 | `Run ID cannot be empty` **[infra]** | `wandb-empty-env` | An empty `WANDB_*` export reached `wandb.init`. Launchers must export only when non-empty **and** the init path must scrub empties — either half alone has failed. | T2 |
| 19 | `no WANDB_API_KEY found` **[infra]** | `wandb-no-op` | `.env` did not reach the VM. The run still trains — but **check row 2 first**, because the same missing `.env` also means no `HF_TOKEN`, and that one is fatal. | T1 |
| 20 | `This TPU has terminal state "PREEMPTED"` **[infra]** | `preempted` | `qr_watch` should recycle. If tmux `qrwatch-tayavision` is dead, restart it. See row 23 for the durability gate. | T3 |
| 20a | QR `PROVISIONING` → `SUSPENDING` → `FAILED` in <5 min, `stateInitiator: SERVICE`, `failedData.error.code: 13` `"an internal error has occurred"` **[observed 2026-07-25, this repo]** | `slice-too-large` — the service allocated capacity, then took it back. **Not a quota gate.** Verified: `europe-west4` `IN_USE_ADDRESSES` was 16/64 (48 free) and it failed anyway; there is no `TPU_V6E_*` compute-quota row to check at all. | Do not retry the same size, and **do not raise IP quota** — that hypothesis is falsified (`docs/tpu/tpu-capacity-log.md` 2026-07-25). Drop to a smaller slice: `v6e-8-eu` is single-host, co-located, and observed gettable in 2-5 min. **Delete the FAILED husk first** — dead QRs book quota. | T4 |
| 21 | `gcloud ssh.*Connection refused` **[xla]** | `node-corruption` | T3 per `tier-definitions.md`. | T3 |
| 22 | `kernel panic` OR `Bus error` **[xla]** | `node-corruption` | As row 21. | T3 |
| 23 | Node lost AND `SAVE_CKPT_DIR` is not a `gs://` URI **[vlm]** | `recycle-blocked-nondurable` | **Auto-recycle is declined on purpose.** Recycling here restarts from step 0 and re-pays ~14 min of compile, up to twenty times, silently. Point `SAVE_CKPT_DIR` at `gs://tayavision-eu/...` and redeploy. `SPEC.md` §5. | T4 |
| 24 | N worker PIDs unreachable for 3+ consecutive polls **[xla]** | `node-corruption` | As row 21. Prefer this over any compile/stall row — a dead PID is a stronger signal than an idle chip. | T3 |
| 25 | Orchestrator says the run is live; the VM log says `exited with status` **[infra]** | **`watcher blindness, not a run failure`** — the completion probe truncated its own output. A `head -N` over a stream containing repeating per-step lines pushes the exit marker past N. | Confirm on the VM before touching anything. Read status markers from the **tail**; fetch repeating lines in a separate query. **When a watcher and reality disagree, the watcher is wrong until proven otherwise.** | T0 |
| 26 | Same classification as the previous iteration | `repeat` | Circuit breaker. Stop — the patch is not working, and each retry costs ~14 min of recompile. | T4 |
| 27 | None of the above | `unknown` | Escalate. Add the signature once diagnosed — procedure in `playbook/diagnosis-table.md`. | T4 |

## Overrides

1. If `iteration > 1` and the classification equals the previous one, override to row 26.
2. If any node-death row (21, 22, 24) matches at all, prefer it over any compile or stall
   row.
3. Prefer a signature matching the **cause** over one matching a downstream symptom. Rows
   10-13 all surface as "compile count rising"; row 16 alone would send you to XLA for a
   day when the cause is one line of Python.
4. If `TPU_STRATEGY=replicated` and a row mentions FSDPv2, the row does not apply — say
   so rather than forcing a match.

## Phase-7 hazards (not live rows)

These three are real torch_xla signatures but fire only with `use_scan_layers: true`,
which the protected config sets to `false`. They live here as hazards of the
`opt-7-scan` probe, not as classifications:

- `ValueError.*Layer \d+ has mismatched keys` — `scan_layers` `_ensure_same_structure`.
  **More likely here than upstream**: Phase 2 puts LoRA on layers 18-35 only, so layers
  0-17 differ structurally by construction.
- `AssertionError.*FakeTensor.*aten\.index_select` — `is_layer_pure=True` with a
  position-embedding gather.
- `TypeError.*unexpected keyword.*attention_mask` — scan passes positionally; HF layers
  take kwargs.

## Baselines

**None yet.** No TPU run has occurred.
`.claude/orchestration/playbook/baseline-v6e8-siglip-cohere2.md` is a template. Do not
import the sibling repos' numbers as this model's baseline — the only transferable facts
are the compile-time envelope (~14 min cold boot) and the HBM ceiling (29 GiB on v6e),
which are properties of the silicon and toolchain rather than of the model.
