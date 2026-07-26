# Orchestration Control Plane

**Version:** v1 (2026-07-25)
**Adapted from:** `llm-architectures` v1 (shape, dedup rules) + `tinyaya-stage2-scale` v2
(torch_xla content). Both are sibling repos on the same TRC grant and GCP project.
**Scope:** repo-level. Covers **both** compute paths — Modal/GPU and TPU/torch_xla.
**Purpose:** one map of skills, agents, hooks, memory files, scripts, and external
surfaces — so every fact is written once.

## 0. Two compute paths

Tiny Aya Vision has two ways to run heavy work. They are not competitors, and one is not
replacing the other.

| Path | Status | Entry | Owns |
|---|---|---|---|
| **Modal / GPU** (DDP + CUDA) | **Authoritative.** Everything published came from here. | `scripts/modal_*.py`, `torchrun` | Every number in `docs/*.md`, every checkpoint, every eval, CI |
| **TPU / torch_xla** (SPMD) | **Experimental.** No training step has run. | `scripts/tpu/*.sh` | Slice provisioning, run supervision, the throughput/HBM envelope |

The boundary is a single rule: **no TPU number may be written into a `docs/*.md` results
table until the same `pipeline/train_*.py` produces it on both backends.** Until the
backend seam lands (`SPEC.md` §9) the TPU path cannot produce a comparable number even in
principle, and pretending otherwise is how two sets of books start.

## 1. Entry points

| Intent | Entry point | Supporting files |
|---|---|---|
| Load state before non-trivial work | SessionStart hook (automatic) or `/recall` | `PLAN.md`, `PROGRESS.md`, `VERIFY.md`, `memories.md`, this file |
| Run or supervise a TPU stage | `tpu-orchestrate` skill | `SPEC.md`, `playbook/*`, `tpu-watchdog`, `tpu-diagnoser` |
| Tune TPU throughput | `tpu-orchestrate` in optimization mode | `TPU_OPTIMIZATION_SPEC.md`, `playbook/optimization-experiment-matrix.md` |
| Push code to a live slice | `tpu-redeploy` skill | `scripts/tpu/deploy_tarball.sh` |
| Classify a TPU failure | `tpu-diagnoser` agent | `playbook/tier-definitions.md` |
| Run training / eval / tests **today** | `modal run scripts/modal_<name>.py` | `CLAUDE.md` "Commands" — **not** this folder |
| Record goal / phase changes | `/plan` | `.claude/PLAN.md` |
| Record events | PostToolUse hook (automatic) or `/progress` | `.claude/PROGRESS.md` |
| Record a durable decision | `/remember` or `#decision …` | `.claude/memories.md` |
| Prove done | `/verify` or Stop hook | `.claude/VERIFY.md` |

`tpu-orchestrate` is the master run-control skill **for TPU work only**. It has no
authority over the Modal path, which is driven directly from `CLAUDE.md`.

## 2. Source-of-truth boundaries

| Surface | Owns | Does NOT own |
|---|---|---|
| `CLAUDE.md` | Repo architecture, commands, conventions, **which path is authoritative** | Run state, operational history, TPU procedure |
| `AGENTS.md` | Modal SDK usage — a vendor cheat-sheet | Anything about this project |
| `.claude/PLAN.md` | Current goal + phase checklist | Run history, raw logs |
| `.claude/PROGRESS.md` | Append-only event log | Durable rationale |
| `.claude/memories.md` | Durable decisions + gotchas | Step-by-step task lists |
| `.claude/VERIFY.md` | Commands that prove the repo is sane | Experiment hypotheses |
| `.claude/orchestration/CONTROL_PLANE.md` | This map: which surface owns which fact | Any fact itself |
| `.claude/orchestration/SPEC.md` | TPU run loop + recovery policy | Detection signatures; optimization phases |
| `.claude/orchestration/TPU_OPTIMIZATION_SPEC.md` | Protected TPU config + promotion gates | Failure recovery |
| `.claude/agents/tpu-diagnoser.md` | **The diagnosis table** (canonical) | Recovery policy — that is `SPEC.md` |
| `.claude/orchestration/playbook/*` | Tier policy, event meanings, metric names, baselines | Raw runtime data |
| `scripts/modal_*.py` | **Every published result**: training, eval, tests, checkpoints | TPU provisioning |
| `scripts/tpu/*.sh` | Slice provisioning, code delivery, run supervision | What the model computes |
| `pipeline/train_*.py` | The training step, for **both** paths | Which accelerator runs it — that is `src/backend/`, once it exists |
| `config/` (Hydra YAML + dataclasses) | Hyperparameters, token arithmetic | Anything topology-shaped (mesh, chips, hosts) |
| `docs/tpu/tpu-trc-allocation.md` | TRC quota + zone grants | Live capacity, zone selection |
| `docs/tpu/tpu-capacity-log.md` | Observed queue times + **the fallback tree** | The grant itself |
| `docs/tpu/tpu-runbook.md` | The operator command sequence | Why any of it is shaped that way |
| `docs/baselines.md`, `docs/instruction_tuning_results.md` | Published numbers — **Modal/GPU only** | TPU throughput or HBM |
| W&B `tayavision-instruct-sweep`, `tayavision-multilingual` | GPU/Modal metrics | TPU metrics |
| W&B `tayavision-tpu` | TPU metrics, step time, HBM envelope | Any published quality number |
| Modal volumes `tayavision-{data,models}`, `multilingual-data` | Datasets + GPU checkpoints | Anything TPU-side |
| `gs://tayavision-eu` | TPU checkpoints + code tarballs | Modal volume contents |
| `/tmp/*.log` (VM + workstation) | Ephemeral runtime detail | Anything durable |

Rule: **write a fact once, at the most specific durable layer, then link to it.**

Two places where this is load-bearing rather than decorative:

- The diagnosis table lives in `tpu-diagnoser.md` **only**. `playbook/diagnosis-table.md`
  is a pointer, not a second copy. `tinyaya-stage2-scale` carried it in five places and
  they drifted.
- The zone fallback tree lives in `docs/tpu/tpu-capacity-log.md` **only**. The allocation
  doc points at it. A fallback tree that drifts from observed capacity sends you to a
  zone that is blocked or costs egress.

## 3. Pipeline stages

Every stage runs through the same deploy/supervise machinery, selected by
`TAYAVISION_ENTRYPOINT`. A new stage plugs in by adding a row here, not by touching
`scripts/tpu/`.

| Stage | `TAYAVISION_ENTRYPOINT` | Inputs | Produces | TPU today? |
|---|---|---|---|---|
| Control-plane smoke | `scripts/tpu/tpu_smoke.py` | none | device count, mesh, HBM, compile counters | **yes** |
| Backbone smoke | `scripts/tpu/tpu_smoke.py --load-backbone` | `HF_TOKEN` | materialized model + real HBM peak | **yes** |
| Alignment (Phase 1) | `pipeline/train_alignment.py` | LLaVA-Pretrain 558K | projector weights | **no** — DDP+CUDA |
| Instruct (Phase 2) | `pipeline/train_instruct.py` | LLaVA-mix665k + Phase 1 ckpt | projector + LoRA r=256 | **no** — DDP+CUDA |
| Multilingual (Phase 2') | `pipeline/train_multilingual.py` | 9-source, 67-language mix | multilingual checkpoint | **no** — DDP+CUDA |
| Eval | `evaluation/run_eval.py` | a checkpoint | lm-eval scores | **no** — stays on Modal |
| Weight merge | `scripts/merge_weights.py` | 2 checkpoints | LERP'd backbone | **no** — CPU, stays on Modal |

The three "no — DDP+CUDA" rows are the whole of the remaining work. See `SPEC.md` §9.

## 4. Layers (who survives what)

| Layer | Dies when | Survives |
|---|---|---|
| VM tmux `train` (per worker) | node preempted/rebooted | workstation off, Claude session ends |
| Workstation tmux `qrwatch-tayavision` | workstation off | Claude session ends, node preemption |
| Boot self-heal (QR metadata) | QR deleted | node reboot, preemption |
| Claude session | anything | nothing — **never** owns a long loop |

The rule this encodes: **a loop that must outlive the conversation never runs inside the
conversation.** Session-bound monitors died three times on `llm-architectures` before
this was enforced.

The workstation session name is suffixed because the same workstation runs a plain
`qrwatch` for `llm-architectures`. `tmux new -s qrwatch` against an existing session
attaches to the wrong one, which leaves this project's queued resource silently
unwatched — precisely the failure this layer exists to prevent.

## 5. Agents

| Agent | Mode | Responsibility |
|---|---|---|
| `tpu-watchdog` | read-only | Report live run state across every worker |
| `tpu-diagnoser` | read-only | Classify a failure signature → tier + action |

Agents never edit files, restart processes, or touch queued resources. They report; the
orchestrator acts.

## 6. External surfaces

- TPU metrics: `https://wandb.ai/cataluna84/tayavision-tpu` — deliberately separate from
  the two GPU projects so bring-up noise never lands in a results project.
- GPU/Modal metrics: `tayavision-instruct-sweep`, `tayavision-multilingual`.
- Push events: `https://ntfy.sh/$NTFY_TOPIC` (topic in the gitignored `.env`; see
  `.env.example`). `notify()` is a no-op when unset, so call sites are unconditional.
- TPU checkpoints + code tarballs: **`gs://tayavision-eu` (europe-west4), one bucket.**
  Only the three `europe-west4-*` profiles are co-located; anything else pays egress on
  every write. See `docs/tpu/tpu-trc-allocation.md` "Bucket co-location".
- Modal: apps `tayavision-train-alignment`, `tayavision-eval`, `tayavision-pytest`;
  volumes `tayavision-data`, `tayavision-models`, `multilingual-data`.
- Code delivery to TPU: GCS tarball only — **never** a git clone. The tarball includes
  the gitignored `.env`, which is how `HF_TOKEN` (the gated `CohereLabs/tiny-aya-*`
  repos), `WANDB_API_KEY`, and `NTFY_TOPIC` reach the VM.
