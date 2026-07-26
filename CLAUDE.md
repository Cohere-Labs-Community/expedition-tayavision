# CLAUDE.md

Tiny Aya Vision — a multilingual vision-language model built for the Tiny Aya
Expedition (Cohere Labs community project). A frozen SigLIP 2 vision encoder is
bolted onto the Tiny Aya LLM through a small trainable connector.

> `AGENTS.md` at the root is a **Modal SDK cheat-sheet**, not project
> instructions. It is there because the production compute path runs on Modal.
> This file is the project guide.

## Compute paths

| Path | Status | Where | Notes |
|---|---|---|---|
| **Modal / GPU** (DDP + CUDA) | **Authoritative** | `scripts/modal_*.py`, apps `tayavision-train-alignment` / `-eval` / `-pytest`, volumes `tayavision-{data,models}` / `multilingual-data` | Every published number, checkpoint, and eval came from here. Unchanged. |
| **TPU / torch_xla** (SPMD) | **Experimental** | `scripts/tpu/*.sh`, TRC grant in `docs/tpu/`, design in `.claude/orchestration/` | Provisioning and supervision only. **No training entrypoint runs on TPU yet.** |

The Modal path is not being replaced. The TPU work adds a second way to run the
*same* `pipeline/train_*.py`, behind a `src/backend/` seam **that does not exist
yet** — `pipeline/train_*.py` is DDP + CUDA today. Until that lands, nothing
under `scripts/tpu/` can train this model, and no TPU number belongs in a
results table. See `.claude/orchestration/SPEC.md` §9.

Do **not** edit `pipeline/train_*.py` to "just add TPU support" inline; that
breaks the Modal path for everyone. The seam exists to prevent it.

## Architecture

```
SigLIP2-so400m-p14-384 (frozen, ~400M)
  -> (B, 729, 1152) patch embeddings          # 384/14 = 27, 27x27 = 729
  -> MultiModalProjector                       # ~11.5M trainable
       pad 27x27 -> 28x28, PixelShuffle 2x2, LayerNorm,
       Linear(4608 -> 2048), SwiGLU chunk-gate, Linear(1024 -> 2048)
  -> (B, 196, 2048)                            # (28//2)^2 = 196 tokens/image
  -> scattered into <image> placeholder positions
  -> CohereLabs/tiny-aya-{base,global}         # Cohere2, 3.35B, 36 layers, d=2048
```

~7.5 GB in bf16 — fits a 24 GB GPU. The token arithmetic above is enforced by
the `config-contracts` check in `.claude/VERIFY.md`; changing any of
`image_size`, `patch_size`, `downsample_factor`, or the derived counts in
`config/vision/siglip.yaml` will fail it.

MoonViT (`moonshotai/MoonViT-SO-400M`) is wired as an alternative encoder with a
`linear_mlp` connector and native-resolution tiling, so it emits a *variable*
token count per image. It has produced no results yet — see `.claude/PLAN.md` P1.

## Layout

| Path | What |
|---|---|
| `src/vision_encoders/` | `base.py` (ABC), `siglip.py`, `moonvit.py`, `create_vision_encoder` factory keyed on `config.vision_encoder_type` |
| `src/vision_encoder.py` | 10-line back-compat shim; `VisionEncoder = SigLIPVisionEncoder` |
| `src/connector.py` | `MultiModalProjector` (pixel shuffle, SigLIP), `LinearMLPProjector` (MoonViT), `create_projector` factory |
| `src/processing.py` | `TinyAyaVisionProcessor` — patches the chat template, expands each `<image>` marker into placeholder tokens (196 fixed for SigLIP, `H*W` from `image_grid_hws` for MoonViT) |
| `models/tiny_aya_vision.py` | `TinyAyaVisionForConditionalGeneration` (HF `PreTrainedModel` + `GenerationMixin`) |
| `models/__init__.py` | Registers with `AutoConfig`/`AutoModel`/`AutoModelForCausalLM`/`AutoProcessor` **at import time** |
| `pipeline/` | `train_alignment.py` (Phase 1), `train_instruct.py` (Phase 2), `train_multilingual.py` (Phase 2'), `data.py`, `multilingual_data.py`, `apply_lora.py`, `utils.py` |
| `evaluation/` | `run_eval.py` (wraps lm-eval), `tiny_aya_vision_lm_eval.py` (custom backend), `tasks/` (123 YAMLs across cvqa, xmmmu, mtvqa, maxm, kaleidoscope, global_mgsm) |
| `config/` | Hydra YAML (`config.yaml`, `vision/`, `training/`, `evaluation/`) **plus** Python dataclasses (`model_config.py`, `training_config.py`, `multilingual_config.py`, `lora_config.py`) |
| `scripts/` | ~18 Modal apps, data downloaders, `merge_weights.py` |

## Training phases

1. **Alignment** — LLaVA-Pretrain (558K caption pairs). Only the projector
   trains. LR 1e-3, bsz 8 x grad-acc 32.
2. **Instruct** — LLaVA-v1.5-mix665k. Projector + LoRA (r=256, alpha=512) on the
   **upper half** of the LLM (layers 18-35). LR 2e-5, global bsz 128.
3. **Multilingual instruct** — a 9-source mix targeting ~15K samples/language
   across all 67 Tiny Aya languages. mT5-style temperature sampling (T=5),
   40% English / 60% multilingual (the Pangea finding).
4. **Weight merging** (post-hoc) — LERP the fine-tuned LLM backbone toward the
   original text-only Tiny Aya at alpha in {0.3..0.7} to recover text ability.
   Only `language_model.*` keys interpolate; projector and encoder are copied
   verbatim from the fine-tuned checkpoint.

## Commands

```bash
uv sync --group dev                  # setup; uv is authoritative, not requirements.txt

# Training (Hydra; config/config.yaml is the root)
# ALWAYS pass training=alignment for Phase 1. config.yaml defaults to
# `training: instruct`, so a bare `python pipeline/train_alignment.py` silently
# runs with instruct's lr=2e-5 (50x too low), batch 128/grad-acc 8, a lora block,
# data_dir=/data/llava-instruct (the WRONG dataset), and logs to
# wandb project tayavision-instruct-sweep.
python pipeline/train_alignment.py training=alignment
python pipeline/train_alignment.py training=alignment vision=siglip llm=global training.batch_size=16
python pipeline/train_alignment.py resume="<uuid>"
python pipeline/train_instruct.py training=instruct
python pipeline/train_multilingual.py training=multilingual_instruct
modal run scripts/modal_train_alignment.py vision=siglip              # remote

# Evaluation
uv run evaluation/run_eval.py --task cvqa --model-name <repo> \
  --backend tiny-aya-vision --apply-chat-template --chunk-size 100
./eval_cvqa.sh            # CVQA on the multilingual model
./eval_cvqa_merged.sh     # alpha sweep over merged checkpoints
./merge_weights.sh        # LERP merge + push to hub

# Checks
uv run ruff check .                  # must be 0 — also a Stop-hook gate
uv run pytest tests/ -q              # 129 tests
uv run pytest tests/ -q -m "not requires_gpu"
```

Tests run remotely via `modal run scripts/modal_pytest.py` (A10G). On a PR,
commenting `/run scripts/modal_<name>.py` dispatches that Modal app — see
`docs/modal_computes_on_pr_comment.md`.

## Gotchas

- **`models` is a side-effect import.** `import models` registers the HF Auto
  classes. Eval scripts rely on a bare `import models`; do not "clean up" those
  imports (they carry `# noqa: F401`).
- **Generation forces `DynamicCache`.** `_prepare_cache_for_generation` overrides
  Cohere2's default `HybridCache`, which triggers a static-cache compile path
  that hangs during prefill (`models/tiny_aya_vision.py:235`).
- **Image merging uses `index_put`, not `masked_scatter`.**
  `masked_scatter_backward` produces incorrect shapes under `torch.compile` /
  inductor (`models/tiny_aya_vision.py:170`).
- **`prepare_inputs_for_generation` merges image features early**, before the LM
  trims `input_ids` — afterwards `forward()` can no longer locate `<image>`
  positions.
- **`pipeline/train_{instruct,multilingual}.py` set `TORCHINDUCTOR_CACHE_DIR`
  and extend `sys.path` before importing torch.** That ordering is load-bearing;
  both files carry a `per-file-ignores = ["E402"]` entry in `pyproject.toml`.
- **`requirements.txt` is stale.** It lists `hydra-core`, `orjson`, and
  `unsloth` (removed from `pyproject.toml` in c5826fd to unbreak `uv sync`) and
  omits `hydra-zen`, `matplotlib`, `modal`. Treat `pyproject.toml` + `uv.lock`
  as authoritative.
- **17 tests need gated HF repos.** `tests/test_processing.py` and
  `tests/test_vlm_assembly.py` load `CohereLabs/tiny-aya-{base,global}`. Without
  an `HF_TOKEN` that has access, they error with `OSError: gated repo`. The
  other 112 pass offline.
- **`config/config.yaml` defaults to `training: instruct`.** So
  `python pipeline/train_alignment.py` with no override runs Phase 1 using
  *Phase 2* settings: `lr=2e-5` instead of `1e-3`, `batch=128/grad_acc=8`
  instead of `8/32`, a `lora` block that means nothing for alignment,
  `data_dir=/data/llava-instruct` instead of `/data/llava-pretrain`, and
  `wandb.project=tayavision-instruct-sweep`. **Always pass `training=alignment`.**
  Verified 2026-07-25 by composing the config directly.
- **`.gitignore` has a blanket `*.json`.** Committed JSON needs `git add -f` or
  a negation — that is why `!.claude/settings.json` exists.
- **`docs/architecture.md` predates the `src/vision_encoders/` refactor** and
  still describes `src/vision_encoder.py` as the encoder.
- **The `index_put` workaround is backend-specific in *both* directions.**
  `models/tiny_aya_vision.py:173` uses `nonzero` + advanced indexing because
  `masked_scatter_backward` is broken under inductor on CUDA — and `nonzero` is
  a *dynamic-shape* op that XLA cannot trace without falling back to CPU or
  recompiling per image count. The two backends want opposite code. The fix
  belongs behind the `src/backend/` seam, not in either branch unconditionally.
- **MoonViT emits a variable token count per image** (`src/processing.py`),
  which is fine on CUDA and is unbounded recompilation on XLA. PLAN P1 (produce
  a MoonViT number) belongs on the Modal path.
- **`train_multilingual.py:146` hardcodes `project="tayavision-multilingual"`**
  and ignores the Hydra `wandb.project` override, so a TPU run would contaminate
  the GPU project. One-line fix; belongs with the backend-seam work.

## Conventions

- Python >= 3.12, `uv` for everything. `ruff check .` must stay at 0.
- Hydra overrides on the command line, never edited into the YAML for a one-off.
- No emojis in code, comments, or commit messages.
- Upstream is the shared `Cohere-Labs-Community` repo. Do not commit or push
  without being asked.

## Memory system

`.claude/` holds an external memory system: session context is auto-injected
from `PLAN.md` / `PROGRESS.md` / `memories.md` / `orchestration/`, work is
auto-logged, and `.claude/VERIFY.md` runs after every response. Read
`.claude/MEMORY-SYSTEM.md` before changing anything under `.claude/hooks/`.

`.claude/orchestration/` is the TPU run-control design: `CONTROL_PLANE.md`
(which surface owns which fact, across both compute paths), `SPEC.md` (the run
loop, and §9 on what does not work yet), `TPU_OPTIMIZATION_SPEC.md`, and a
`playbook/`. The canonical diagnosis table is `.claude/agents/tpu-diagnoser.md`
— `playbook/diagnosis-table.md` is a pointer to it and must never become a
second copy.

Quick capture: start a prompt with `#progress`, `#plan`, `#decision`, or
`#verify`. Slash commands: `/recall`, `/plan`, `/verify`, `/verify-full`,
`/progress`, `/remember`, `/curate`.
