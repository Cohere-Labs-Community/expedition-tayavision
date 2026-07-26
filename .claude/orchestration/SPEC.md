# Tiny Aya Vision TPU Run Control — SPEC

**Version:** v1 (2026-07-25)
**Stack:** torch_xla SPMD (PyTorch), *not* JAX. Adapted from `llm-architectures` v1 (the
loop, the ladder, the dedup rules) and `tinyaya-stage2-scale` v4 (the torch_xla content,
same Cohere2 backbone).
**Status:** control plane designed and testable; **no training entrypoint runs on TPU
yet** — read §9 before launching anything.
**Topology:** default `v6e-8` = 1 host × 8 chips, single Python process, SPMD. Multi-host
profiles exist in `launch_spot.sh` but are blocked on `IN_USE_ADDRESSES` — see
`docs/tpu/tpu-capacity-log.md` §7.1.

Read `CONTROL_PLANE.md` first for surface ownership.

## 1. Goal

Drive a TPU stage to completion with **bounded autonomous recovery**, on spot capacity
that will be preempted mid-run, without a human in the loop and without a Claude session
being load-bearing.

## 2. Decisions locked

| Decision | Value | Why |
|---|---|---|
| Stack | torch_xla SPMD | Proven on `tinyaya-stage2-scale` against the same Cohere2 + LoRA backbone. A JAX rewrite of an HF `PreTrainedModel` is not a port, it is a second model. |
| Orchestration home | workstation tmux `qrwatch-tayavision` | Claude sessions die; three did on the sibling repo. Suffixed because that repo's `qrwatch` runs on the same workstation. |
| Training home | VM tmux `train`, every worker | Multi-host PJRT rendezvous needs every host |
| Code delivery | GCS tarball incl. `.env` | `CohereLabs/tiny-aya-*` is **gated**; without `HF_TOKEN` on the VM the run cannot start. A git clone can never ship it. |
| Config transport | one Hydra override string | This repo is Hydra, not a wall of env vars. `TAYAVISION_HYDRA_OVERRIDES` is appended verbatim to the launch line. |
| Bucket | **one**, `gs://tayavision-eu` | User decision. Makes the three `europe-west4-*` profiles co-located and everything else an explicit egress choice. |
| Default `TPU_STRATEGY` | `replicated` | Phase 1 trains ~11.5M of ~3.75B params; the model is ~7.5 GB bf16 against 31.25 GiB/chip. FSDPv2 buys nothing here and costs the reduce-scatter hazard in §7. |
| T3 (node lost) | **auto-recycle, gated on `gs://` checkpoints** | See §5 — a considered split between the two source policies |
| Check-ins | **push-only** (ntfy) + 2h heartbeat | Nothing blocks on a human |
| Resume | `TAYAVISION_RESUME=auto` | Preemption reboot self-heals |
| Iteration cap | `MAX_RESUBMITS=20`, then stop | Anti-flap without a hard wall-clock cap |
| Modal | **unchanged and authoritative** | Nothing here replaces it. See `CONTROL_PLANE.md` §0. |

## 3. Architecture (4 layers)

See `diagrams/01-architecture.mmd`.

```
Claude session      reads logs, decides, edits code        (disposable)
      |
Workstation tmux    qrwatch-tayavision (QR lifecycle)      (survives Claude)
      |
QR metadata         tarball URI, entrypoint, resume=auto,
                    hydra overrides, W&B identity          (survives reboot)
      |
VM tmux `train`     the actual process, x N workers        (survives workstation)
      |
GCS + W&B + ntfy    checkpoints, metrics, events           (survives everything)
```

`llm-architectures` has a fifth layer, a `sweep` tmux running an ordered runs-file. It is
deliberately not ported: there is nothing to sweep until §9 is closed.

## 4. The run loop

See `diagrams/02-state-machine.mmd`.

```
DEPLOY -> WATCH -> [ok]           -> WATCH
                -> [exit clean]   -> DONE
                -> [exit dirty]   -> CLASSIFY -> RECOVER -> DEPLOY
                -> [node gone]    -> qr_watch recycles -> boot self-heal
                -> [budget spent] -> STOP + notify
```

**DEPLOY** — `deploy_tarball.sh` tars the working tree (incl. `.env`), uploads to GCS,
pulls on `--worker=all`, and relaunches VM tmux `train` everywhere via
`train_launcher.sh`. Each deploy bakes a unique `TAYAVISION_RUN_TAG` into the launch line.

**WATCH** — poll worker 0's `/tmp/train.log`, scoped to the segment *after* this run's
`RUN_TAG`. Never match a stale segment from an earlier deploy or boot; a 7-hour-old
segment once triggered a spurious abort on the sibling repo.

Completion detection has two independent queries, and this is load-bearing:

- rare status markers (`exited with status`, `Traceback`, `DEADLINE_EXCEEDED`,
  `RESOURCE_EXHAUSTED`, `nan`) — take the **tail**
- the repeating per-step line — fetched separately, **tail -1**

Piping both through one `head -N` truncates the status marker behind N repeats of the
noisy line and the run looks hung forever. That exact bug hid a completed 1000-step run
for an hour, and it is already recorded in this repo's `memories.md`.

**Compile is not a hang here.** On torch_xla the persistent XLA cache has
nondeterministic keys on v6e SPMD and effectively never hits: **~14 min of recompile on
every cold boot is normal** (`docs/tpu/tpu-capacity-log.md` §8). `llm-architectures`'s
"no step line for 10 min = hang" rule is a JAX rule and produces a false positive on
every single boot here. The threshold is **20 min**, and the discriminator is
`met.metrics_report()` compile-cause counters, not elapsed time.

**CLASSIFY** — match the log against the canonical table in
`.claude/agents/tpu-diagnoser.md`; it returns a tier.

**RECOVER** — act per `playbook/tier-definitions.md`.

## 5. Preemption and T3 (a split decision)

The two source repos disagree. `tinyaya-stage2-scale` locked T3 to "always escalate,
never auto-recreate the QR". `llm-architectures` inverted it and auto-recycles, on the
grounds that TRC quota is free, preemption is routine (~25 in one day on v6e), and
overnight runs must survive.

**This repo takes the auto-recycle policy with one brake neither source has.**

`qr_watch.sh` polls the queued resource every 300s and, on `SUSPENDING` / `SUSPENDED` /
`FAILED`:

1. captures `describe` JSON forensics,
2. checks for a quota-class abort (no point resubmitting into a wall),
3. **checks that `SAVE_CKPT_DIR` starts with `gs://`** — see below,
4. deletes the QR,
5. resubmits the identical `launch_spot.sh` env file,
6. pushes an ntfy event.

Bounded by `MAX_RESUBMITS=20` plus a cooldown. If the QR is absent entirely it exits
rather than competing with a human — **one resubmitter per QR, ever.**

**The `gs://` gate.** A recycled node whose checkpoints went to node-local disk resumes
from **step 0** — and then pays a cold-boot XLA recompile to redo work it already did,
twenty times, silently. Auto-recycle without durable checkpoints is not self-healing, it
is a capacity incinerator. So when `SAVE_CKPT_DIR` is not a `gs://` URI, `qr_watch.sh`
degrades to **notify-only (T4)** and says so in the push event.

> **Implemented 2026-07-26, verified on the v6e-16.** This paragraph used to describe the
> gate as the *only* protection, because `save_checkpoint` genuinely did write node-local
> and `SAVE_CKPT_DIR` reached no Python at all. Both halves now exist and were verified by
> deleting `/models/<run_id>` on all four hosts and resuming from the bucket alone:
>
> - `save_checkpoint` mirrors every write to `<SAVE_CKPT_DIR>/<run_id>/`.
> - The resume path calls `fetch_checkpoint_from_gcs` on **every rank** before concluding
>   there is nothing to resume.
> - `TAYAVISION_RESUME` is read by `resolve_resume_run_id()`; `auto` resolves to the run
>   with the most recently mirrored checkpoint and rejoins the same W&B run.
>
> The gate still matters — `auto` has nothing to find without a `gs://` root — but it is
> no longer the only thing standing between a recycle and lost work.

Boot self-heal closes the loop when the gate passes: QR metadata carries the tarball URI,
`TAYAVISION_ENTRYPOINT`, `TAYAVISION_HYDRA_OVERRIDES`, `TAYAVISION_RESUME=auto`, and the
W&B run identity — so a recycled node re-fetches code, resumes from the latest GCS
checkpoint, and keeps logging to the *same* W&B run with no human action.

## 6. Notifications

Nothing blocks on a human. Events push to `https://ntfy.sh/$NTFY_TOPIC` and a 2-hour
heartbeat proves the watcher is alive — a silent watcher is indistinguishable from a dead
one. See `playbook/event-taxonomy.md`.

`notify()` is a no-op when `NTFY_TOPIC` is unset, so a first run with no `.env` works; it
is just deaf. Run one test event end to end before trusting the pipe: this repo's
`memories.md` already records a full day of alerts lost to a `.env` with no trailing
newline.

## 7. Invariants (do not break)

- **196 image tokens per image** on the SigLIP path. `config-contracts` in
  `.claude/VERIFY.md` enforces the arithmetic; XLA enforces it harder, because a variable
  token count means a new HLO program per shape.
- **Never wrap `Cohere2DecoderLayer` in an FSDPv2 auto-wrap policy.** 36 per-layer wraps
  produce 36 bf16 reduce-scatters and hit pytorch/xla #8591 / #8778: NaN loss at step
  ~24-130. FSDPv2 has no `fp32_reduce_scatter` (only FSDPv1 does, #3588 / #8056). Use one
  outer reduce-scatter at the composite level. Observed on `tinyaya-stage2-scale` with
  **this same backbone and the same 36 layers**; assume it applies until disproved.
- **The wrap policy matches on `type(module).__name__`.** A wrong class-name string wraps
  nothing and looks like it worked. Verify the name against the installed `transformers`
  before trusting any FSDPv2 run.
- **SPMD is one process.** `DistributedSampler` must be OFF; a single process holding a
  1/N sampler shard trains on 1/N of the dataset and never says so.
- **Deploys ship the working tree as a GCS tarball incl. `.env`.** Never clone.
- **One resubmitter per QR** — `qr_watch` or a human, never both.
- **Long loops live in tmux**, never in a Claude session.
- **No TPU number enters a `docs/*.md` results table** until the same
  `pipeline/train_*.py` runs on both backends.

## 8. Known-good numbers

**There are none.** `playbook/baseline-v6e8-siglip-cohere2.md` is a template with every
cell marked TBD, plus one clearly-labelled non-authoritative envelope inherited from a
*different model* on the same backbone family. Fill it from the first real run; do not
fill it from the sibling repos.

## 9. What does not work yet

`pipeline/train_alignment.py`, `train_instruct.py`, and `train_multilingual.py` are DDP +
CUDA and **cannot run on TPU today**. In `train_alignment.py` alone:
`torch.autocast("cuda", ...)`, `torch.device(f"cuda:{local_rank}")`,
`torch.device("cuda" if torch.cuda.is_available() ...)`, `torch.cuda.manual_seed_all`,
`DistributedDataParallel`, and `DistributedSampler`.

Everything in this folder and in `scripts/tpu/` is therefore **provisioning and
supervision**, which is genuinely useful and genuinely testable on its own:

- provision a TRC slice by profile, on spot, with QR-metadata self-heal;
- deliver the working tree (incl. `.env`, and therefore `HF_TOKEN`) to every worker with
  no git clone;
- install `uv` + `torch_xla` and resolve the `libpython3.12.so.1.0` / `_XLAC.so` link;
- launch in tmux `train` with the exact log markers every watcher matches;
- watch, classify, redeploy, recycle the QR, and push events.

`TAYAVISION_ENTRYPOINT` defaults to `scripts/tpu/tpu_smoke.py`, which exercises every one
of those layers end to end and proves libtpu/mesh/HBM on real silicon — without
pretending training works. `--load-backbone` extends it to download the gated repos and
materialize the real model, which proves the `HF_TOKEN` path and the memory envelope: the
two things most likely to be wrong.

**The ladder:** `tpu_smoke.py` → `tpu_smoke.py --load-backbone` → `train_alignment.py`.

### The seam that closes it

Mirroring `tinyaya-stage2-scale`'s proven structure:

- New `src/backend/`: `base.py` (ABC — `device()`, `wrap_model()`, `optimizer_step()`,
  `sync()`, `memory_info()`, `is_main()`, `autocast()`), `gpu_backend.py` (today's
  DDP+CUDA extracted **verbatim**, so the Modal path stays byte-equivalent), and
  `tpu_backend.py` — the **only** module allowed a module-level `import torch_xla`.
- `scripts/ci/check_backend_seam.sh` greps `^(import torch_xla|from torch_xla)` under
  `src/`, `models/`, `pipeline/`, `config/` and fails if one leaks. A module-level import
  drags `libtpu` into the graph and breaks every CPU/GPU run and the 129-test suite.
  `models/__init__.py` matters most: it registers HF Auto classes at import time and is
  imported by every eval script.
- Lazy in-function imports on TPU-only paths stay legal; the check flags column-0 imports
  only.
- **The subtle call-site change**: SPMD is a single process, so `DistributedSampler` must
  be off and the loader wrapped in `MpDeviceLoader`. Leaving `DistributedSampler` on
  under SPMD silently trains on 1/N of the dataset and looks completely healthy.
- When `src/backend/` lands, `check_backend_seam.sh` becomes a `# verify: backend-seam`
  block. **That** is the moment to raise `MAX_BLOCKS` to 6 in `stop.py` and drop
  `PER_BLOCK_TIMEOUT` to 12 (6 × 12 = 72s, still inside the 90s hook timeout). Not before
  — a sixth block today would be silently dropped.

Anything that reads this file and then edits `pipeline/train_*.py` to "just add TPU
support" inline is about to break the Modal path for everyone. The seam exists so that
does not happen.
