"""Instruction-finetuning pipeline for Tiny Aya Vision (DDP).

Uses LLaVA-Instruct-150K with the instruction-tuned tiny-aya-global backbone,
LoRA adapters on the LLM, and a chat_template for proper conversation formatting.

Phase 2 training:
  - Vision encoder: frozen
  - Multi-modal projector: trainable (initialised from Phase 1 alignment checkpoint)
  - LLM backbone: LoRA adapters on upper layers (base weights frozen)

Launch:
  Single GPU:  python pipeline/train_instruct.py
  Multi GPU:   torchrun --nproc_per_node=NUM_GPUS pipeline/train_instruct.py
"""

import json
import os
import re
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

from config.lora_config import LoraAdapterConfig
from config.model_config import TinyAyaVisionConfig
from config.training_config import InstructConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from pipeline.data import InstructDataset, collate_fn
from scripts.apply_lora import apply_lora, get_lora_optimizer_groups
from src.processing import TinyAyaVisionProcessor


def is_torchrun() -> bool:
    """True when launched via torchrun / torch.distributed.launch."""
    return "LOCAL_RANK" in os.environ


def setup_ddp():
    """Initialize distributed process group and set CUDA device."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model):
    """Unwrap DDP and torch.compile wrappers to access the raw module."""
    raw = model
    if hasattr(raw, "module"):       # DDP
        raw = raw.module
    if hasattr(raw, "_orig_mod"):    # torch.compile
        raw = raw._orig_mod
    return raw


def save_checkpoint(checkpoint_dir, step, model, optimizer, lr_scheduler):
    save_path = checkpoint_dir / f"checkpoint_{step}.pt"
    raw_model = _unwrap_model(model)
    torch.save({
        "step": step,
        "projector": raw_model.multi_modal_projector.state_dict(),
        "lora_adapter": {
            k: v for k, v in raw_model.language_model.state_dict().items()
            if "lora_" in k
        },
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    def extract_step(p):
        m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    return max(checkpoints, key=extract_step)


def train(
    model,
    dataloader: torch.utils.data.DataLoader,
    sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: InstructConfig,
    checkpoint_dir: Path,
    compute_dtype: torch.dtype,
    device: torch.device,
    step_offset: int = 0,
):
    model.train()
    accumulated_loss = 0.0
    use_ddp = dist.is_initialized()
    is_main = (not use_ddp) or dist.get_rank() == 0

    for epoch in range(training_config.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{training_config.num_epochs}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for step, batch in enumerate(pbar, start=step_offset):
            input_ids, attention_mask, pixel_values, labels = (
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["pixel_values"].to(device, non_blocking=True),
                batch["labels"].to(device, non_blocking=True),
            )

            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / training_config.grad_acc_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (step + 1) % training_config.grad_acc_steps == 0:
                # Clip all trainable parameters (projector + LoRA)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step = (step + 1) // training_config.grad_acc_steps

                if is_main:
                    wandb.log({
                        "train/loss": accumulated_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    }, step=opt_step)

                    pbar.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}", gnorm=f"{grad_norm.item():.2f}")

                    if opt_step % training_config.logging_steps == 0:
                        tqdm.write(f"Epoch {epoch}, Opt Step {opt_step}, Loss {accumulated_loss:.4f}, LR {lr_scheduler.get_last_lr()[0]}")

                    if opt_step % training_config.save_steps == 0:
                        save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)

                if use_ddp:
                    dist.barrier()

                accumulated_loss = 0.0

    if is_main:
        save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)
    if use_ddp:
        dist.barrier()
    if is_main:
        print("Training complete")


def main(
    training_config: InstructConfig,
    model_config: TinyAyaVisionConfig,
    lora_config: LoraAdapterConfig,
    resume_run_id: str | None = None,
):
    use_ddp = is_torchrun()
    if use_ddp:
        local_rank = setup_ddp()
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
                "model_config": asdict(model_config),
                "lora_config": asdict(lora_config),
            }, f, indent=2)

    if is_main:
        wandb.init(
            project="tayavision-instruct",
            name=run_id,
            id=run_id.replace("-", ""),
            resume="allow",
            config={**asdict(training_config), **asdict(lora_config)},
        )

    # Build model with LoRA adapters
    model = apply_lora(vlm_config=model_config, lora_config=lora_config)

    # Load Phase 1 alignment checkpoint for the projector
    if training_config.alignment_checkpoint:
        ckpt = torch.load(training_config.alignment_checkpoint, map_location="cpu")
        projector_state = ckpt["projector"] if "projector" in ckpt else ckpt
        model.multi_modal_projector.load_state_dict(projector_state)
        if is_main:
            print(f"Loaded projector from {training_config.alignment_checkpoint}")

    model.to(device, non_blocking=True)

    processor = TinyAyaVisionProcessor(config=model_config)

    compute_dtype = getattr(torch, training_config.torch_dtype)
    model.vision_encoder.to(dtype=compute_dtype, non_blocking=True)

    model.language_model.base_model.enable_input_require_grads()
    # model.language_model.base_model.gradient_checkpointing_enable()

    model = torch.compile(model)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

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

    dataset = InstructDataset(
        config=model_config,
        data_dir=training_config.data_dir,
        max_seq_len=training_config.max_seq_len,
    )

    full_dataset_len = len(dataset)

    samples_to_skip = resume_step * per_gpu_batch_size
    if samples_to_skip > 0 and samples_to_skip < len(dataset):
        remaining_indices = list(range(samples_to_skip, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, remaining_indices)
        if is_main:
            print(f"Skipped {samples_to_skip} samples, {len(dataset)} remaining")

    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=training_config.seed,
        )
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            pad_token_id=processor.tokenizer.pad_token_id,
        ),
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=training_config.num_workers > 0,
        prefetch_factor=2 if training_config.num_workers > 0 else None,
        drop_last=False,
    )

    # Optimizer with differential LR for LoRA A/B matrices
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
        raise ValueError(f"Unsupported LR scheduler type: {training_config.lr_scheduler_type}")

    if resume_step > 0:
        opt.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

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
        step_offset=resume_step,
    )

    if is_main:
        wandb.finish()
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    model_config = TinyAyaVisionConfig.for_global()
    lora_config = LoraAdapterConfig.from_vlm_config(model_config)

    main(
        training_config=InstructConfig(),
        model_config=model_config,
        lora_config=lora_config,
    )
