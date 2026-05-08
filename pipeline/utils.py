"""Common utility functions for training pipelines."""

import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

from models import save_for_inference
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
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model):
    """Unwrap torch.compile and DDP wrappers to access the raw module."""
    raw = model
    if hasattr(raw, "_orig_mod"):    # torch.compile
        raw = raw._orig_mod
    if hasattr(raw, "module"):       # DDP
        raw = raw.module
    return raw


def save_checkpoint(checkpoint_dir, step, model, optimizer, lr_scheduler, save_lora=False):
    """Save a training checkpoint to disk.

    Always saves the projector state dict, optimizer, and LR scheduler.
    When ``save_lora=True``, also saves LoRA adapter weights from the
    language model (used by instruct / multilingual pipelines).
    """
    save_path = checkpoint_dir / f"checkpoint_{step}.pt"
    raw_model = _unwrap_model(model)
    state = {
        "step": step,
        "projector": raw_model.multi_modal_projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if save_lora:
        state["lora_adapter"] = {
            k: v for k, v in raw_model.language_model.state_dict().items()
            if "lora_" in k
        }
    torch.save(state, save_path)
    print(f"Saved checkpoint to {save_path}")


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the path to the highest-step checkpoint in a directory, or None."""
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    def extract_step(p):
        m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    return max(checkpoints, key=extract_step)


def save_hf_model(model, processor: TinyAyaVisionProcessor, checkpoint_dir: Path, training_config=None) -> Path:
    """Merge LoRA into the base model and save in HuggingFace format.

    If ``training_config.merge_with_base_llm`` is True, additionally performs
    linear interpolation (LERP) of the VLM's LLM backbone with the
    original base LLM weights.

    Returns the output directory path.
    """
    print("Merging LoRA and saving HF-compatible model...")
    raw_model = _unwrap_model(model)
    raw_model.language_model = raw_model.language_model.merge_and_unload()

    # Optionally merge with original base LLM via weight interpolation
    if (
        training_config is not None
        and getattr(training_config, "merge_with_base_llm", False)
        and training_config.base_llm_name
    ):
        from scripts.merge_weights import build_merged_vlm_state, _load_original_llm

        alpha = training_config.merge_alpha
        print(f"Merging VLM backbone with '{training_config.base_llm_name}' (α={alpha})...")

        finetuned_state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
        original_llm_state = _load_original_llm(
            training_config.base_llm_name, device="cpu", dtype=torch.bfloat16,
        )
        merged_state = build_merged_vlm_state(original_llm_state, finetuned_state, alpha)
        raw_model.load_state_dict(merged_state, strict=False)
        del original_llm_state, finetuned_state, merged_state
        print("Weight merge complete.")

    hf_output_dir = checkpoint_dir / "hf_model"
    save_for_inference(raw_model, processor, hf_output_dir)
    print(f"Saved HF-compatible model to {hf_output_dir}")
    return hf_output_dir


def build_lr_scheduler(optimizer, training_config, full_dataset_len, per_gpu_batch_size, world_size):
    """Build a cosine LR scheduler with linear warmup.

    Computes total optimisation steps from the dataset size, batch
    configuration and number of epochs, then constructs a sequential
    scheduler: linear warmup followed by cosine decay.
    """
    full_loader_len = full_dataset_len // (per_gpu_batch_size * world_size)
    total_steps = training_config.num_epochs * full_loader_len // training_config.grad_acc_steps
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    if training_config.lr_scheduler_type == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8 / training_config.learning_rate, total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=training_config.learning_rate * 0.01,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
        )
    else:
        raise ValueError(f"Unsupported LR scheduler type: {training_config.lr_scheduler_type}")
