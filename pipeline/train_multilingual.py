"""Multilingual multimodal instruction-finetuning pipeline for Tiny Aya Vision.

Phase 2 training with balanced multilingual data mixing:
  - Vision encoder: frozen
  - Multi-modal projector: trainable (initialised from Phase 1 alignment checkpoint)
  - LLM backbone: LoRA adapters on upper layers (base weights frozen)

Data mixing follows Pangea optimal recipe:
  - 40% English / 60% multilingual
  - Temperature-based sampling (T=5) for low-resource upsampling
  - Multiple HF Hub sources (PangeaIns, PALO, Aya Dataset)

Launch:
  Single GPU:  python pipeline/train_multilingual.py  training=multilingual_instruct
  Multi GPU:   torchrun --nproc_per_node=NUM_GPUS pipeline/train_multilingual.py  training=multilingual_instruct
"""

import json
import os
import sys
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

# Persist torch.compile cache across runs to avoid 12+ min cold starts
_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".inductor_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _CACHE_DIR)
def setup_environment() -> None:
    """Set HF cache environment variables based on config."""
    base = "/data/v-tridas/"
    os.environ["HF_HOME"]            = f"{base}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{base}/hf_cache"
    os.environ["HF_DATASETS_CACHE"]  = f"{base}/datasets_cache"

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb

from config.lora_config import LoraAdapterConfig
from config.model_config import TinyAyaVisionConfig
from config.multilingual_config import MultilingualInstructConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from pipeline.data import collate_fn
from pipeline.multilingual_data import MultilingualInstructDataset
from pipeline.apply_lora import apply_lora, get_lora_optimizer_groups
from pipeline.train_instruct import (
    is_torchrun,
    setup_ddp,
    cleanup_ddp,
    _unwrap_model,
    save_checkpoint,
    find_latest_checkpoint,
    generate_samples,
    train,
)
from src.processing import TinyAyaVisionProcessor


def main(
    training_config: MultilingualInstructConfig,
    model_config: TinyAyaVisionConfig,
    lora_config: LoraAdapterConfig,
    resume_run_id: str | None = None,
):
    setup_environment()
    use_ddp = is_torchrun()
    if use_ddp:
        # Set NCCL timeout before init_process_group — multilingual training
        # has variable batch times across ranks due to mixed data sources
        import datetime
        nccl_timeout_min = getattr(training_config, "nccl_timeout_minutes", 30)
        nccl_timeout = datetime.timedelta(minutes=nccl_timeout_min)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=nccl_timeout,
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)

    # Enable TF32 for matmul and cuDNN for ~10-20% speedup on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Compute per-GPU batch size from global batch size
    assert training_config.batch_size % world_size == 0, (
        f"batch_size ({training_config.batch_size}) must be "
        f"divisible by world_size ({world_size})"
    )
    per_gpu_batch_size = training_config.batch_size // world_size

    if is_main:
        print(f"{'DDP' if use_ddp else 'Single-GPU'}: world_size={world_size}, "
              f"global_batch_size={training_config.batch_size}, "
              f"per_gpu_batch_size={per_gpu_batch_size}")
        print(f"\nMultilingual mixing config:")
        print(f"  Total samples: {training_config.total_samples}")
        print(f"  English ratio: {training_config.english_ratio}")
        print(f"  Temperature: {training_config.temperature}")
        print(f"  Sources: {len(training_config.multilingual_sources)}")

    if resume_run_id:
        run_id = resume_run_id
    else:
        run_id = str(uuid.uuid4())

    checkpoint_dir = Path(training_config.models_dir) / run_id
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Checkpoint dir: {checkpoint_dir}")
    if use_ddp:
        dist.barrier()

    config_path = checkpoint_dir / "config.json"
    if is_main and not config_path.exists():
        with open(config_path, "w") as f:
            json.dump({
                "training_config": asdict(training_config),
                "model_config": model_config.to_dict(),
                "lora_config": asdict(lora_config),
            }, f, indent=2)

    if is_main:
        wandb.init(
            project="tayavision-multilingual",
            name=run_id,
            id=run_id.replace("-", ""),
            resume="allow",
            config={
                **asdict(training_config),
                **asdict(lora_config),
                "multilingual": True,
            },
        )

    # Build model with LoRA adapters
    model = apply_lora(vlm_config=model_config, lora_config=lora_config)

    # Load Phase 2 instruct checkpoint (projector + LoRA) if provided
    if training_config.instruct_checkpoint:
        ckpt = torch.load(training_config.instruct_checkpoint, map_location="cpu", weights_only=True)
        projector_state = ckpt["projector"] if "projector" in ckpt else ckpt
        model.multi_modal_projector.load_state_dict(projector_state)
        lora_state = ckpt.get("lora_adapter", {})
        if lora_state:
            model.language_model.load_state_dict(lora_state, strict=False)
        if is_main:
            print(f"Loaded projector + LoRA from instruct checkpoint: {training_config.instruct_checkpoint}")
    # Fall back to Phase 1 alignment checkpoint (projector only)
    elif training_config.alignment_checkpoint:
        ckpt = torch.load(training_config.alignment_checkpoint, map_location="cpu", weights_only=True)
        projector_state = ckpt["projector"] if "projector" in ckpt else ckpt
        model.multi_modal_projector.load_state_dict(projector_state)
        if is_main:
            print(f"Loaded projector from {training_config.alignment_checkpoint}")

    model.to(device, non_blocking=True)

    processor = TinyAyaVisionProcessor(config=model_config)
    compute_dtype = getattr(torch, training_config.torch_dtype)
    model.vision_encoder.to(dtype=compute_dtype, non_blocking=True)
    model.language_model.base_model.enable_input_require_grads()

    if use_ddp:
        # find_unused_parameters=True because text-only batches (Aya) skip
        # the vision encoder, leaving its params without gradients.
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True,)
    # torch._dynamo needs to handle the varying DDP sync state from no_sync()
    torch._dynamo.config.optimize_ddp = "python_reducer"
    model = torch.compile(model)

    # Enable gradient checkpointing AFTER torch.compile + DDP wrapping so
    # dynamo doesn't try to trace through the checkpointing hooks.
    raw_for_gc = _unwrap_model(model)
    raw_for_gc.language_model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    resume_step = 0
    if resume_run_id:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            if is_main:
                print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_model = _unwrap_model(model)
            raw_model.multi_modal_projector.load_state_dict(ckpt["projector"])
            lora_state = ckpt.get("lora_adapter", {})
            if lora_state:
                raw_model.language_model.load_state_dict(lora_state, strict=False)
            resume_step = ckpt["step"]
            if is_main:
                print(f"Resuming from step {resume_step}")
        else:
            if is_main:
                print(f"No checkpoints found in {checkpoint_dir}, starting from scratch")

    # Build multilingual dataset — rank 0 does the heavy JSON→Parquet conversion,
    # other ranks wait then load from the fast Parquet cache.
    _ds_kwargs = dict(
        config=model_config,
        sources=training_config.multilingual_sources,
        total_samples=training_config.total_samples,
        english_ratio=training_config.english_ratio,
        temperature=training_config.temperature,
        max_seq_len=training_config.max_seq_len,
        cache_dir=training_config.hf_cache_dir or None,
        seed=training_config.seed,
        allow_upsampling=training_config.allow_upsampling,
        max_upsample_factor=training_config.max_upsample_factor,
    )
    if is_main:
        dataset = MultilingualInstructDataset(**_ds_kwargs)
    if use_ddp:
        dist.barrier()
    if not is_main:
        dataset = MultilingualInstructDataset(**_ds_kwargs)

    full_dataset_len = len(dataset)
    samples_to_skip = resume_step * per_gpu_batch_size
    if samples_to_skip > 0 and samples_to_skip < len(dataset):
        remaining_indices = list(range(samples_to_skip, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, remaining_indices)
        if is_main:
            print(f"Skipped {samples_to_skip} samples, {len(dataset)} remaining")

    if use_ddp:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=training_config.seed,
        )
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=training_config.num_workers > 0,
        prefetch_factor=2 if training_config.num_workers > 0 else None,
        drop_last=False,
    )

    # Optimizer
    param_groups = get_lora_optimizer_groups(
        model, training_config.learning_rate, lora_config,
    )
    opt = torch.optim.AdamW(
        param_groups,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    full_loader_len = full_dataset_len // (per_gpu_batch_size * world_size)
    total_steps = training_config.num_epochs * full_loader_len // training_config.grad_acc_steps
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    if training_config.lr_scheduler_type == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-8 / training_config.learning_rate, total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps, eta_min=training_config.learning_rate * 0.01,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
        )
    else:
        raise ValueError(f"Unsupported LR scheduler: {training_config.lr_scheduler_type}")

    if resume_step > 0:
        opt.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    # Reuse the existing training loop from train_instruct
    train(
        model=model,
        dataloader=loader,
        sampler=sampler,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        compute_dtype=compute_dtype,
        device=device,
        image_token_id=processor.image_token_id,
        processor=processor,
        step_offset=resume_step,
    )

    if is_main:
        wandb.finish()
    if use_ddp:
        cleanup_ddp()


def run(cfg: DictConfig):
    """Hydra entry — convert DictConfig to typed configs and call main()."""
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    lora_dict = training_dict.pop("lora", {})

    # Extract multilingual-specific fields
    multilingual_sources = training_dict.pop("multilingual_sources", [])
    training_config = MultilingualInstructConfig(
        **training_dict,
        multilingual_sources=multilingual_sources,
    )

    model_config = TinyAyaVisionConfig.for_encoder(
        cfg.vision.vision_encoder_type, llm=cfg.llm,
    )

    if "layers_to_transform" not in lora_dict:
        n = model_config.num_llm_layers
        lora_dict["layers_to_transform"] = list(range(n // 2, n))
    lora_config = LoraAdapterConfig(**lora_dict)

    main(
        training_config=training_config,
        model_config=model_config,
        lora_config=lora_config,
        resume_run_id=cfg.get("resume", None),
    )


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def hydra_main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    hydra_main()
