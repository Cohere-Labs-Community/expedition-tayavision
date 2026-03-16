# Training Vision Encoders Experiment

This project trains a multimodal Vision-Language Model (VLM) called **Tiny Aya Vision**. It connects a frozen vision encoder to a frozen LLM backbone (CohereLabs/tiny-aya-base) through a trainable connector (projector). The goal of the alignment phase is to teach the connector to translate vision tokens into the LLM's embedding space using image-caption pairs from [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) (558k examples).

We support two vision encoders to compare:

| Encoder | Model | Connector | Tokens per image |
|---------|-------|-----------|------------------|
| **SigLIP** | `google/siglip2-so400m-patch14-384` | Pixel Shuffle + SwiGLU MLP | 196 (fixed) |
| **MoonViT** | `moonshotai/MoonViT-SO-400M` | Linear SwiGLU MLP | Dynamic (depends on image resolution) |

Both encoders output 1152-dim features. The connector projects them to the LLM's 2048-dim space.

---

## Key Tools: Hydra, YAML, and Modal

### Hydra & YAML (Configuration)

[Hydra](https://hydra.cc/) is a configuration framework that lets you compose and override YAML configs from the command line. Instead of hardcoding hyperparameters, everything lives in YAML files under `config/`:

```
config/
├── config.yaml              # Root config — sets defaults
├── vision/
│   ├── siglip.yaml           # SigLIP encoder settings
│   └── moonvit.yaml          # MoonViT encoder settings
└── training/
    └── alignment.yaml        # Training hyperparameters
```

The root `config/config.yaml` composes everything:

```yaml
defaults:
  - _self_
  - vision: moonvit    # loads config/vision/moonvit.yaml
  - training: alignment # loads config/training/alignment.yaml

llm: base               # "base" or "global"
resume: null

wandb:
  project: tayavision
  entity: null
  mode: online
```

**How to change the experiment:**

- **Switch vision encoder:** Change `vision: moonvit` to `vision: siglip` in `config.yaml`, or override from CLI: `--vision siglip`
- **Change LLM backbone:** Set `llm: global` for the instruction-tuned variant
- **Tweak training:** Edit values in `config/training/alignment.yaml` (batch size, learning rate, etc.)

Example YAML overrides from the CLI (when running locally with Hydra):

```bash
python pipeline/train_alignment.py vision=siglip llm=base training.batch_size=4
```

### Modal (Cloud GPU Execution)

[Modal](https://modal.com/) is a serverless cloud platform that lets you run GPU workloads without managing infrastructure. Each `scripts/modal_*.py` script defines:

1. **An `Image`** — a Docker-like container with all Python dependencies
2. **A `Volume`** — persistent cloud storage for datasets and model checkpoints
3. **A `Function` or `Class`** — the code to run on a remote GPU

Modal handles provisioning, scaling, and teardown automatically. You pay per second of GPU time.

Key concepts:
- **`modal.Volume`** — persistent disk attached to your function (survives across runs)
- **`modal.Secret`** — securely injects env vars like `HF_TOKEN` and `WANDB_API_KEY`
- **`--detach`** — runs the job in the background so you can close your terminal
- **GPU selection** — pass `--gpu A10G` or `--gpu A100` to the training script

---

## Running Experiments on Modal

### Prerequisites

1. Install Modal: `pip install modal`
2. Authenticate: `modal setup`
3. Set up secrets in the [Modal dashboard](https://modal.com/secrets):
   - `huggingface` — with `HF_TOKEN`
   - `wandb` — with `WANDB_API_KEY`

### Step 1: Create Modal Volumes (v2)

Create the volumes that store data and checkpoints. Modal now defaults to v2 volumes:

```bash
modal volume create tayavision-data
modal volume create tayavision-models
```

### Step 2: Download the Dataset

First, download LLaVA-Pretrain locally (optional, for inspection):

```bash
python scripts/download_llava_pretrain.py --output-dir ./data/llava-pretrain
```

Then upload it to the Modal volume:

```bash
modal run scripts/modal_download.py
```

This downloads the 558k conversation JSON and ~13GB of images directly onto the `tayavision-data` volume at `/data/llava-pretrain/`.

### Step 3: Run Alignment Training

Train with **SigLIP** on an A10G:

```bash
modal run --detach scripts/modal_train_alignment.py --vision siglip --gpu A10G
```

Train with **MoonViT** on an A100:

```bash
modal run --detach scripts/modal_train_alignment.py --vision moonvit --gpu A100
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--vision` | `moonvit` | Vision encoder: `siglip` or `moonvit` |
| `--llm` | `base` | LLM variant: `base` or `global` |
| `--gpu` | `A100` | GPU type: `A10G`, `A100`, `H100`, etc. |
| `--resume-run-id` | `None` | Resume from a previous run's checkpoint |

Resume a previous run:

```bash
modal run --detach scripts/modal_train_alignment.py --resume-run-id <uuid>
```

Checkpoints are saved to the `tayavision-models` volume every 500 optimizer steps (configurable via `save_steps` in `config/training/alignment.yaml`). Training logs go to Weights & Biases.

### Step 4: Run Tests

Run the test suite on a cloud GPU:

```bash
modal run scripts/modal_pytest.py
```

This runs `tests/test_vision_encoder.py` on an A10G.

### Step 5: Run Evaluation

**LM Evaluation Harness** (language benchmarks):

```bash
modal run --detach scripts/modal_eval.py --task <task_name> --model-name CohereLabs/tiny-aya-base
```

Results are synced back to `evaluation/results/` locally.

**Alignment quality** (vision token analysis):

```bash
modal run scripts/modal_eval_aligned_tokens.py --top-k 10
modal run scripts/modal_eval_mlp_l2_norm.py --num-per-class 1
```

---

## Config Reference

### Training Hyperparameters (`config/training/alignment.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Per-GPU batch size |
| `grad_acc_steps` | 32 | Gradient accumulation steps (effective batch = 256) |
| `learning_rate` | 1e-3 | Peak learning rate |
| `warmup_ratio` | 0.03 | Fraction of total steps for linear warmup |
| `lr_scheduler_type` | cosine | Learning rate schedule |
| `num_epochs` | 1 | Number of passes over the dataset |
| `max_seq_len` | 2048 | Max token sequence length |
| `torch_dtype` | bfloat16 | Compute precision |
| `save_steps` | 500 | Save checkpoint every N optimizer steps |

### Vision Encoder Configs

**SigLIP** (`config/vision/siglip.yaml`): Fixed 384x384 input, pixel shuffle compresses 729 patches down to 196 tokens.

**MoonViT** (`config/vision/moonvit.yaml`): Native-resolution tiling, produces a variable number of tokens per image (4 per tile). No pixel shuffle needed.
