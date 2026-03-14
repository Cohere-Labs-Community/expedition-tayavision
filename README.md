# Expedition Tiny Aya - Tiny Aya Vision

Part of Tiny Aya Expedition - Tiny Aya Vision

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For development (includes pytest, ruff):

```bash
uv sync --group dev
```

### PyTorch CUDA/CPU override

The default configuration pulls PyTorch wheels for CUDA 12.4. To use a different CUDA version or CPU-only:

```bash
# CPU-only
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu uv sync

# CUDA 12.1
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 uv sync
```

## Get Started

Download the dataset (~13 GB)

```bash
python scripts/download_llava_pretrain.py --output-dir data/llava-pretrain
```

Train Alignment

```bash
python pipeline/train_alignment.py --vision-encoder siglip --llm global --models-dir outputs/checkpoints --data-dir data/llava-pretrain
```
## Running Alignment Training

We use [Hydra](https://hydra.cc/) for configuration management. You can run training locally or on Modal.

### Local Execution

```bash
# Run with defaults
python pipeline/train_alignment.py

# Switch vision encoder to siglip and customize parameters inline
python pipeline/train_alignment.py vision=siglip training.batch_size=16 llm=global

# Resume an existing run
python pipeline/train_alignment.py resume="my-previous-uuid"
```

### Remote Execution on Modal

Run the alignment training seamlessly on Modal without touching code. Overrides are passed directly:

```bash
# Run on Modal with defaults
modal run scripts/modal_train_alignment.py

# Or with Hydra overrides
modal run scripts/modal_train_alignment.py vision=siglip training.batch_size=32
```
