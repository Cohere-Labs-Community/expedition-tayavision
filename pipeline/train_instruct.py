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

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

from config.lora_config import LoraAdapterConfig
from config.model_config import TinyAyaVisionConfig
from config.training_config import InstructConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from pipeline.data import InstructDataset, collate_fn
from pipeline.apply_lora import apply_lora, get_lora_optimizer_groups
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
    """Unwrap torch.compile and DDP wrappers to access the raw module."""
    raw = model
    if hasattr(raw, "_orig_mod"):    # torch.compile
        raw = raw._orig_mod
    if hasattr(raw, "module"):       # DDP
        raw = raw.module
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


@torch.no_grad()
def generate_samples(
    model,
    batch: dict,
    processor: TinyAyaVisionProcessor,
    compute_dtype: torch.dtype,
    device: torch.device,
    max_new_tokens: int = 256,
    num_samples: int = 2,
) -> list[dict[str, str]]:
    """Generate text from the first `num_samples` items in a batch.

    Extracts the prompt (tokens before the first assistant response, i.e.
    where labels == -100) and runs generation. Returns a list of dicts
    with 'prompt' and 'generation' decoded strings.

    To avoid image-token-count mismatches that can arise when
    ``GenerationMixin.generate()`` internally mutates ``input_ids``
    (e.g. via ``_cache_dependant_input_preparation`` under torch.compile /
    DDP), we pre-compute ``inputs_embeds`` with vision features already
    merged and pass those to ``generate()`` instead of raw ``input_ids`` +
    ``pixel_values``.
    """
    raw = _unwrap_model(model)
    was_training = raw.training
    raw.eval()

    results = []
    n = min(num_samples, batch["input_ids"].size(0))
    for i in range(n):
        labels_i = batch["labels"][i]
        # Find first non-masked label (start of first assistant response)
        response_mask = labels_i != -100
        if response_mask.any():
            prompt_len = response_mask.nonzero(as_tuple=False)[0].item()
        else:
            prompt_len = labels_i.size(0)

        prompt_ids = batch["input_ids"][i, :prompt_len].unsqueeze(0).to(device)
        prompt_mask = batch["attention_mask"][i, :prompt_len].unsqueeze(0).to(device)
        pixel_values = batch["pixel_values"][i].unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=compute_dtype):
            # Pre-compute inputs_embeds with vision features merged so that
            # generate() never needs to call _merge_image_features itself.
            inputs_embeds = raw.get_input_embeddings()(prompt_ids)
            image_features = raw.get_image_features(pixel_values)
            inputs_embeds = raw._merge_image_features(
                prompt_ids, inputs_embeds, image_features,
            )

            gen_ids = raw.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # gen_ids contains [bos_token, generated_tokens…] when using
        # inputs_embeds — the original prompt tokens are NOT present.
        # Skip the leading dummy token to get only generated output.
        new_ids = gen_ids[0, 1:]
        prompt_text = processor.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        gen_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Denormalize pixel values → [0, 255] PIL image for wandb
        img_tensor = batch["pixel_values"][i].float().cpu()
        mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
        img_tensor = (img_tensor * std + mean).clamp(0, 1)
        img_pil = Image.fromarray(
            (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        results.append({
            "image": wandb.Image(img_pil),
            "prompt": prompt_text,
            "generation": gen_text,
        })

    if was_training:
        raw.train()
    return results


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
    image_token_id: int,
    processor: TinyAyaVisionProcessor | None = None,
    step_offset: int = 0,
):
    model.train()
    accumulated_loss = 0.0
    max_image_tokens_in_window = 0
    use_ddp = dist.is_initialized()
    is_main = (not use_ddp) or dist.get_rank() == 0

    # Accumulate generation samples across save steps
    generation_rows = []

    # Forward hook to capture projector output norms
    norm_cache = {}
    raw = _unwrap_model(model)
    def _projector_hook(module, input, output):
        with torch.no_grad():
            norms = output.detach().float().norm(dim=-1)
            norm_cache["projector_token"] = norms
    hook_handle = raw.multi_modal_projector.register_forward_hook(_projector_hook)

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

            max_image_tokens_in_window = max(
                max_image_tokens_in_window,
                (input_ids == image_token_id).sum(dim=1).max().item(),
            )

            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss / training_config.grad_acc_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (step + 1) % training_config.grad_acc_steps == 0:
                # Compute grad norms separately for projector and LoRA
                projector_params = [p for p in raw.multi_modal_projector.parameters() if p.requires_grad and p.grad is not None]
                lora_params = [p for p in raw.language_model.parameters() if p.requires_grad and p.grad is not None]

                projector_grad_norm = torch.nn.utils.clip_grad_norm_(projector_params, float("inf"))
                lora_grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, float("inf"))

                # Clip all trainable parameters (projector + LoRA)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step = (step + 1) // training_config.grad_acc_steps

                # Token masking stats (computed before logging, outside is_main guard for clarity)
                num_total_tokens = labels.numel()
                num_masked_tokens = (labels == -100).sum().item()
                num_response_tokens = num_total_tokens - num_masked_tokens
                masked_pct = 100.0 * num_masked_tokens / num_total_tokens

                if is_main:
                    log_dict = {
                        "train/loss": accumulated_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/grad_norm_projector": projector_grad_norm.item(),
                        "train/grad_norm_lora": lora_grad_norm.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/response_tokens": num_response_tokens,
                        "train/masked_pct": masked_pct,
                        "train/max_image_tokens": max_image_tokens_in_window,
                    }

                    if "projector_token" in norm_cache:
                        pn = norm_cache["projector_token"]
                        log_dict["norms/projector_token_mean"] = pn.mean().item()
                        log_dict["norms/projector_token_std"] = pn.std().item()
                        log_dict["norms/projector_token_min"] = pn.min().item()
                        log_dict["norms/projector_token_max"] = pn.max().item()

                        emb_w = raw.get_input_embeddings().weight.detach().float()
                        emb_norms = emb_w.norm(dim=-1)
                        log_dict["norms/emb_matrix_mean"] = emb_norms.mean().item()
                        log_dict["norms/emb_matrix_std"] = emb_norms.std().item()
                        log_dict["norms/emb_matrix_min"] = emb_norms.min().item()
                        log_dict["norms/emb_matrix_max"] = emb_norms.max().item()

                    pbar.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}", gnorm=f"{grad_norm.item():.2f}")

                    if opt_step % training_config.logging_steps == 0:
                        tqdm.write(f"Epoch {epoch}, Opt Step {opt_step}, Loss {accumulated_loss:.4f}, LR {lr_scheduler.get_last_lr()[0]}")

                    if opt_step % training_config.save_steps == 0:
                        save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)

                        if processor is not None:
                            samples = generate_samples(
                                model, batch, processor,
                                compute_dtype, device,
                            )
                            for s in samples:
                                generation_rows.append([opt_step, s["image"], s["prompt"], s["generation"]])
                            table = wandb.Table(
                                columns=["step", "image", "prompt", "generation"],
                                data=generation_rows,
                            )
                            log_dict["generations"] = table

                    wandb.log(log_dict, step=opt_step)

                if use_ddp:
                    dist.barrier()

                accumulated_loss = 0.0
                max_image_tokens_in_window = 0

    hook_handle.remove()
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
    
    # if training_config.enable_gradient_checkpointing:
        # model.language_model.base_model.gradient_checkpointing_enable() # This is causing a problem wrt the current torch.compile and DDP setup, hence disabling it

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    model = torch.compile(model)

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
        image_token_id=processor.image_token_id,
        processor=processor,
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
