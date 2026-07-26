# `.claude/orchestration/`

Source-of-truth design artifacts for TPU run control on Tiny Aya Vision. Documents live
here; implementations live in their natural homes and are indexed below.

**Status (2026-07-25):** imported and adapted. **No Tiny Aya Vision TPU run has
happened.** `pipeline/train_*.py` is DDP + CUDA and does not run under torch_xla — see
`SPEC.md` §9 before launching anything. Everything here describes provisioning and
supervision, which is real and testable today via `scripts/tpu/tpu_smoke.py`.

## Layout

```
orchestration/
├── README.md                   this file — index + provenance
├── CONTROL_PLANE.md            who owns which fact; BOTH compute paths
├── SPEC.md                     the run loop, recovery policy, invariants, §9 gap
├── TPU_OPTIMIZATION_SPEC.md    protected config, 6 gates, 9-phase program
├── playbook/
│   ├── tier-definitions.md              T0-T4 ladder + escalation triggers
│   ├── diagnosis-table.md               POINTER to agents/tpu-diagnoser.md
│   ├── event-taxonomy.md                ntfy events + design rules
│   ├── perf-metrics-schema.md           W&B field names + comparability invariants
│   ├── optimization-experiment-matrix.md  12 candidates + the run ladder
│   └── baseline-v6e8-siglip-cohere2.md  protected numbers (all TBD today)
└── diagrams/
    ├── 01-architecture.mmd     layers and what each survives
    ├── 02-state-machine.mmd    DEPLOY/WATCH/CLASSIFY/RECOVER
    ├── 03-memory-lifecycle.mmd ephemeral runtime data → durable knowledge
    └── render.sh               *.mmd → svg/ + png/
```

## Implementation files (NOT here)

| Artifact | Location | Purpose |
|---|---|---|
| Shared helpers | `scripts/tpu/_lib.sh` | `load_env_file`, `notify`, `make_code_tarball` |
| One-time bootstrap | `scripts/tpu/setup_gcp.sh` | APIs, `gs://tayavision-eu`, IAM |
| QR submit | `scripts/tpu/launch_qr.sh`, `launch_spot.sh` | Provision a slice; profile → zone/accelerator |
| Boot | `scripts/tpu/startup_script.sh` | Fetch tarball, install deps, resolve `_XLAC.so`, launch |
| Launcher | `scripts/tpu/train_launcher.sh` | Emits the markers every watcher greps |
| Deploy (T2) | `scripts/tpu/deploy_tarball.sh` | tar → GCS → all workers → relaunch |
| QR babysitter (T3) | `scripts/tpu/qr_watch.sh` | Recycle on preemption, gated on `gs://` checkpoints |
| Daily ops | `scripts/tpu/ops.sh` | `preflight`, `status`, `tail-logs`, `attach`, `ssh`, `delete` |
| Smoke entrypoint | `scripts/tpu/tpu_smoke.py` | The default `TAYAVISION_ENTRYPOINT` |
| Skill | `.claude/skills/tpu-orchestrate/SKILL.md` | Run-control entry point |
| Skill | `.claude/skills/tpu-redeploy/SKILL.md` | The T2 procedure |
| Agent | `.claude/agents/tpu-watchdog.md` | Read-only live state |
| Agent | `.claude/agents/tpu-diagnoser.md` | **Canonical diagnosis table** |
| Grant + capacity | `docs/tpu/*.md` | TRC quota, observed capacity, operator runbook |

## Provenance

Merged from two sibling repos on the same TRC grant and GCP project.

**Shape from `llm-architectures`** (pure JAX, 2026-07-20) — a deduplicated refactor of
the tinyaya original: the DEPLOY/WATCH/CLASSIFY/RECOVER loop, the T0-T4 ladder,
non-blocking ntfy events, the surface-ownership table, and the discipline of one
canonical diagnosis table. Its `playbook/diagnosis-table.md` is a pointer, not a copy,
and that choice is preserved here.

**torch_xla content from `tinyaya-stage2-scale`** (2026-05-13) — `TPU_OPTIMIZATION_SPEC.md`
and `optimization-experiment-matrix.md`, which `llm-architectures` dropped as having no
JAX analogue. They come back because this project is PyTorch, and because that project
runs **the same Cohere2 backbone with the same 36 decoder layers** — its FSDPv2
reduce-scatter NaN is a hazard this repo inherits rather than merely resembles.

### Divergences from both sources

1. **T3 is conditionally automatic.** tinyaya locked T3 to "always escalate";
   `llm-architectures` inverted it to unconditional auto-recycle. Here `qr_watch.sh`
   auto-recycles **only when `SAVE_CKPT_DIR` is a `gs://` URI**, and otherwise degrades
   to notify-only. Recycling without durable checkpoints restarts from step 0 and re-pays
   ~14 min of XLA compile, twenty times, silently.
2. **The compile threshold is 20 minutes, not 10.** `llm-architectures`'s "no step line
   for 10 min = hang" is a JAX rule. torch_xla on v6e SPMD takes ~14 min to compile on a
   cold boot, so that rule would fire on every single launch here. Classification uses
   the `met.metrics_report()` compile-cause counter, not elapsed time.
3. **No sweep layer.** `llm-architectures` has a fifth layer — `sweep` tmux running an
   ordered runs-file — plus `vm_coordinator`, `race_provision`, and a supervisor VM. None
   are ported: there is nothing to sweep until `SPEC.md` §9 is closed.
4. **`replicated` is the default strategy, not FSDPv2.** Phase 1 trains ~11.5M of ~3.75B
   params against 31.25 GiB/chip. Neither source could default this way; both were memory
   constrained.
5. **Config travels as one Hydra override string.** nanoGPT reads ~15 env overrides;
   tinyaya edits YAML. This repo is Hydra, so `TAYAVISION_HYDRA_OVERRIDES` is appended
   verbatim to the launch line rather than inventing a parallel env surface.
6. **The control plane covers two compute paths.** Neither source had to say which of
   Modal and TPU owns a given fact. `CONTROL_PLANE.md` §0 and §2 do.
7. **One bucket.** `gs://tayavision-eu`. Both sources kept region-paired siblings; here
   every non-EU profile is an explicit egress decision, and the zone fallback tree in
   `docs/tpu/tpu-capacity-log.md` was reordered to match.

## Read order

1. `CONTROL_PLANE.md` — surface ownership, and which compute path is authoritative
2. `SPEC.md` — the run loop; **§9 first if you are about to launch something**
3. `playbook/tier-definitions.md` — what to do at each tier
4. `.claude/agents/tpu-diagnoser.md` — signature → tier
5. `TPU_OPTIMIZATION_SPEC.md` + `playbook/optimization-experiment-matrix.md` — only once
   a run is stable
6. `docs/tpu/tpu-runbook.md` — the actual commands

## Rendering diagrams

```bash
bash .claude/orchestration/diagrams/render.sh
```

Requires `mmdc` (`npm i -g @mermaid-js/mermaid-cli`), falling back to `npx`. **Neither is
installed on this workstation**, so rendering is currently unavailable — the `.mmd`
sources are the checked-in artifact and `svg/`/`png/` are gitignored. GitHub renders
mermaid natively, so the sources are readable without local tooling.

## Versioning

An edit that changes behaviour bumps the version banner in `SPEC.md` and updates the
affected diagram **in the same commit**. A spec that disagrees with its diagram is how
five diagnosis tables happened.
