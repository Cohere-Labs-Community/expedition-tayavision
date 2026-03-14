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