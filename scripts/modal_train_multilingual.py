"""
Run multilingual multimodal instruction finetuning on Modal.

Trains Tiny Aya Vision with multilingual data from the `multilingual-data` volume:
- PangeaInstruct (39 languages, primary multimodal)
- WIT/Wikipedia Image Text (105 languages)
- ALM-Bench (100 languages)
- MVL-SIB (197 languages)
- Aya Dataset (70+ languages, text-only)
- bloom-lm (364 languages, text-only)

Supports both single-GPU and multi-GPU DDP training via torchrun.
Configuration uses Hydra with `training=multilingual_instruct` base config.

Prerequisites:
    1. Download data: modal run scripts/modal_download_multilingual_data.py
    2. Create Modal secrets:
        modal secret create huggingface HF_TOKEN=hf_xxx...
        modal secret create wandb WANDB_API_KEY=xxx...

Usage:
    # Single-GPU training (default)
    modal run --detach scripts/modal_train_multilingual.py

    # Multi-GPU DDP training (2, 4, or 8 GPUs)
    modal run --detach scripts/modal_train_multilingual.py --num-gpus 4
    modal run --detach scripts/modal_train_multilingual.py --num-gpus 8

    # Resume from checkpoint
    modal run --detach scripts/modal_train_multilingual.py --resume-run-id <run_id>

    # Start from instruct checkpoint (Phase 2 continuation)
    modal run --detach scripts/modal_train_multilingual.py \\
        --instruct-checkpoint /models/<run_id>/checkpoint_<step>.pt

    # Custom training parameters
    modal run --detach scripts/modal_train_multilingual.py \\
        --total-samples 500000 \\
        --english-ratio 0.4 \\
        --temperature 5.0 \\
        --batch-size 64 \\
        --learning-rate 2e-5

    # Multi-GPU with custom batch size (auto-scales by default)
    modal run --detach scripts/modal_train_multilingual.py \\
        --num-gpus 4 \\
        --batch-size 256

    # Use different GPU type (default: A100-80GB)
    MODAL_GPU=H100 modal run --detach scripts/modal_train_multilingual.py --num-gpus 4

GPU Scaling:
    --num-gpus 1: Single GPU, batch_size=64 (default)
    --num-gpus 2: 2x GPU DDP via torchrun, batch_size auto-scales to 128
    --num-gpus 4: 4x GPU DDP via torchrun, batch_size auto-scales to 256
    --num-gpus 8: 8x GPU DDP via torchrun, batch_size auto-scales to 512

    Batch size auto-scaling keeps per-GPU batch size constant (64).
    Override with --batch-size to use a custom value.
"""

import os
import subprocess

import modal

GPU_TYPE = os.environ.get("MODAL_GPU", "A100-80GB")

app = modal.App("tayavision-train-multilingual")

# Data volume with multilingual datasets (populated by modal_download_multilingual_data.py)
data_volume = modal.Volume.from_name("multilingual-data")
models_volume = modal.Volume.from_name("tayavision-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        # Core ML
        "torch==2.9.1",
        "torchvision",
        "transformers==4.56.2",
        "accelerate",
        "huggingface_hub",
        "tokenizers",
        "sentencepiece",
        "protobuf",
        # Dataset loading
        "datasets>=2.16.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "orjson>=3.9.0",
        # Config
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyyaml",
        # Training
        "peft",
        "wandb",
        "einops",
        "tqdm",
        # Utilities
        "Pillow",
        "numpy",
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("models", remote_path="/root/project/models")
)


# ---------------------------------------------------------------------------
# Helper: Build Hydra overrides from training parameters
# ---------------------------------------------------------------------------


def build_hydra_overrides(
    total_samples: int,
    english_ratio: float,
    temperature: float,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    resume_run_id: str | None = None,
    instruct_checkpoint: str | None = None,
    alignment_checkpoint: str | None = None,
) -> list[str]:
    """Build Hydra override list from training parameters."""
    overrides = [
        "training=multilingual_instruct",
        # Override paths for Modal volume structure
        "training.models_dir=/models",
        "training.hf_cache_dir=/data/.hf_cache",
        # Training hyperparameters
        f"training.total_samples={total_samples}",
        f"training.english_ratio={english_ratio}",
        f"training.temperature={temperature}",
        f"training.batch_size={batch_size}",
        f"training.learning_rate={learning_rate}",
        f"training.num_epochs={num_epochs}",
    ]

    # Add checkpoint paths if provided
    if instruct_checkpoint:
        overrides.append(f"training.instruct_checkpoint={instruct_checkpoint}")
    if alignment_checkpoint:
        overrides.append(f"training.alignment_checkpoint={alignment_checkpoint}")
    if resume_run_id:
        overrides.append(f"resume={resume_run_id}")

    return overrides


def fix_source_data_dirs(multilingual_sources: list) -> None:
    """Fix data_dir paths for Modal volume structure (in-place)."""
    source_to_path = {
        "pangea_ins": "/data/pangea-instruct",
        "palo": "/data/palo",
        "llava_instruct": "/data/llava-instruct",
        "mvl_sib": "/data/mvl-sib",
        "wit": "/data/wit",
        "bloom_lm": "/data/bloom-lm",
        "alm_bench": "/data/alm-bench",
        "bloom_captioning": "/data/bloom-captioning",
        "aya_text": "/data/aya-dataset",
    }
    for source in multilingual_sources:
        name = source.get("name", "")
        if name in source_to_path and not source.get("data_dir"):
            source["data_dir"] = source_to_path[name]


def print_training_banner(
    num_gpus: int,
    total_samples: int,
    english_ratio: float,
    temperature: float,
    batch_size: int,
    learning_rate: float,
    resume_run_id: str | None,
    instruct_checkpoint: str | None,
) -> None:
    """Print training configuration banner."""
    print("=" * 60)
    print("Multilingual Multimodal Instruction Finetuning on Modal")
    print("=" * 60)
    print(f"GPU: {GPU_TYPE} x {num_gpus}")
    print(f"Mode: {'DDP (torchrun)' if num_gpus > 1 else 'Single GPU'}")
    print(f"Total samples: {total_samples:,}")
    print(f"English ratio: {english_ratio:.0%}")
    print(f"Temperature: {temperature}")
    print(f"Batch size: {batch_size} (global)")
    if num_gpus > 1:
        print(f"Per-GPU batch size: {batch_size // num_gpus}")
    print(f"Learning rate: {learning_rate}")
    print(f"Resume run: {resume_run_id or 'None'}")
    print(f"Instruct checkpoint: {instruct_checkpoint or 'None'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Single-GPU training function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 48,  # 48 hours for large multilingual training
)
def train(
    resume_run_id: str | None = None,
    instruct_checkpoint: str | None = None,
    alignment_checkpoint: str | None = None,
    total_samples: int = 1_000_000,
    english_ratio: float = 0.4,
    temperature: float = 5.0,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
):
    """Run single-GPU multilingual multimodal instruction finetuning.

    Args:
        resume_run_id: Resume training from a previous run (loads latest checkpoint).
        instruct_checkpoint: Path to Phase 2 instruct checkpoint (projector + LoRA).
        alignment_checkpoint: Path to Phase 1 alignment checkpoint (projector only).
        total_samples: Total dataset size after mixing (default: 1M).
        english_ratio: Fraction of English data (default: 0.4 = 40%).
        temperature: Temperature for low-resource upsampling (default: 5.0).
        batch_size: Global batch size (default: 64).
        learning_rate: Learning rate for LoRA adapters (default: 2e-5).
        num_epochs: Number of training epochs (default: 1).
    """
    import sys

    sys.path.insert(0, "/root/project")

    # Set environment variables for Modal paths
    os.environ["HF_HOME"] = "/data/.hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/.hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/data/.hf_cache"

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from config.lora_config import LoraAdapterConfig
    from config.model_config import TinyAyaVisionConfig
    from config.multilingual_config import MultilingualInstructConfig
    from pipeline.train_multilingual import main as train_main

    print_training_banner(
        num_gpus=1,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
    )

    overrides = build_hydra_overrides(
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
        alignment_checkpoint=alignment_checkpoint,
    )

    # Initialize Hydra and compose config
    with initialize_config_dir(config_dir="/root/project/config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

        # Extract config components (mirrors train_multilingual.run())
        training_dict = OmegaConf.to_container(cfg.training, resolve=True)
        assert isinstance(training_dict, dict), "Expected training config to be a dict"
        lora_dict = training_dict.pop("lora", {})
        multilingual_sources = training_dict.pop("multilingual_sources", [])

        # Fix data_dir paths for Modal volume structure
        fix_source_data_dirs(multilingual_sources)

        training_config = MultilingualInstructConfig(
            **training_dict, multilingual_sources=multilingual_sources
        )

        # Build model config (Tiny Aya Vision with global LLM)
        model_config = TinyAyaVisionConfig.for_encoder(
            cfg.vision.vision_encoder_type, llm=cfg.llm
        )
        lora_config = LoraAdapterConfig(**lora_dict)

        # Run training
        train_main(
            training_config=training_config,
            model_config=model_config,
            lora_config=lora_config,
            resume_run_id=cfg.get("resume"),
        )


# ---------------------------------------------------------------------------
# Multi-GPU DDP training function
# ---------------------------------------------------------------------------


def _get_gpu_spec(num_gpus: int) -> str:
    """Get Modal GPU spec string for multi-GPU allocation."""
    # Modal uses format like "A100-80GB:4" for 4x A100-80GB
    return f"{GPU_TYPE}:{num_gpus}"


@app.function(
    image=image,
    gpu=_get_gpu_spec(2),  # Default to 2 GPUs, overridden dynamically
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 48,
)
def train_ddp_2gpu(
    resume_run_id: str | None = None,
    instruct_checkpoint: str | None = None,
    alignment_checkpoint: str | None = None,
    total_samples: int = 1_000_000,
    english_ratio: float = 0.4,
    temperature: float = 5.0,
    batch_size: int = 128,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
):
    """Run 2-GPU DDP multilingual training via torchrun."""
    _run_ddp_training(
        num_gpus=2,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
        alignment_checkpoint=alignment_checkpoint,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )


@app.function(
    image=image,
    gpu=_get_gpu_spec(4),
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 48,
)
def train_ddp_4gpu(
    resume_run_id: str | None = None,
    instruct_checkpoint: str | None = None,
    alignment_checkpoint: str | None = None,
    total_samples: int = 1_000_000,
    english_ratio: float = 0.4,
    temperature: float = 5.0,
    batch_size: int = 256,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
):
    """Run 4-GPU DDP multilingual training via torchrun."""
    _run_ddp_training(
        num_gpus=4,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
        alignment_checkpoint=alignment_checkpoint,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )


@app.function(
    image=image,
    gpu=_get_gpu_spec(8),
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 48,
)
def train_ddp_8gpu(
    resume_run_id: str | None = None,
    instruct_checkpoint: str | None = None,
    alignment_checkpoint: str | None = None,
    total_samples: int = 1_000_000,
    english_ratio: float = 0.4,
    temperature: float = 5.0,
    batch_size: int = 512,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
):
    """Run 8-GPU DDP multilingual training via torchrun."""
    _run_ddp_training(
        num_gpus=8,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
        alignment_checkpoint=alignment_checkpoint,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )


def _run_ddp_training(
    num_gpus: int,
    resume_run_id: str | None,
    instruct_checkpoint: str | None,
    alignment_checkpoint: str | None,
    total_samples: int,
    english_ratio: float,
    temperature: float,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
):
    """Internal function to run DDP training via torchrun subprocess.

    This uses torchrun to launch the training script with proper DDP setup.
    The train_multilingual.py script auto-detects DDP via LOCAL_RANK env var.
    """
    import sys

    sys.path.insert(0, "/root/project")

    # Set environment variables for Modal paths
    os.environ["HF_HOME"] = "/data/.hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/.hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/data/.hf_cache"

    print_training_banner(
        num_gpus=num_gpus,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
    )

    # Build Hydra overrides as command-line arguments
    overrides = build_hydra_overrides(
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
        alignment_checkpoint=alignment_checkpoint,
    )

    # Add data_dir overrides for Modal volume paths
    # These need to be added as Hydra list overrides
    data_dir_overrides = [
        "training.multilingual_sources.0.data_dir=/data/pangea-instruct",
        # Note: Other sources will be auto-fixed by multilingual_data.py
        # if data_dir is empty, but we can add explicit overrides here if needed
    ]
    overrides.extend(data_dir_overrides)

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        "/root/project/pipeline/train_multilingual.py",
    ] + overrides

    print(f"\nLaunching DDP training with {num_gpus} GPUs...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run torchrun subprocess
    result = subprocess.run(
        cmd,
        cwd="/root/project",
        env=os.environ.copy(),
        check=False,  # Don't raise on non-zero exit
    )

    if result.returncode != 0:
        raise RuntimeError(f"DDP training failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    resume_run_id: str = None,
    instruct_checkpoint: str = None,
    alignment_checkpoint: str = None,
    total_samples: int = 1_000_000,
    english_ratio: float = 0.4,
    temperature: float = 5.0,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
    num_gpus: int = 1,
):
    """Local entrypoint for Modal CLI.

    Args:
        num_gpus: Number of GPUs for training (1, 2, 4, or 8).
            - 1: Single GPU (default)
            - 2/4/8: Multi-GPU DDP via torchrun

    All other arguments are forwarded to the training function.
    """
    # Scale default batch size with number of GPUs
    if batch_size == 64 and num_gpus > 1:
        batch_size = 64 * num_gpus
        print(f"Auto-scaling batch size to {batch_size} for {num_gpus} GPUs")

    kwargs = dict(
        resume_run_id=resume_run_id,
        instruct_checkpoint=instruct_checkpoint,
        alignment_checkpoint=alignment_checkpoint,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    if num_gpus == 1:
        train.remote(**kwargs)
    elif num_gpus == 2:
        train_ddp_2gpu.remote(**kwargs)
    elif num_gpus == 4:
        train_ddp_4gpu.remote(**kwargs)
    elif num_gpus == 8:
        train_ddp_8gpu.remote(**kwargs)
    else:
        raise ValueError(
            f"num_gpus must be 1, 2, 4, or 8 (got {num_gpus}). "
            "For other GPU counts, create a custom function with the desired GPU spec."
        )
