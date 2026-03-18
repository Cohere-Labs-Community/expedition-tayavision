import json
import re
import sys
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import wandb

from config.training_config import AlignmentConfig
from config.model_config import TinyAyaVisionConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from pipeline.data import AlignmentDataset, collate_fn
from src.processing import TinyAyaVisionProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_checkpoint(checkpoint_dir, step, model, optimizer, lr_scheduler):
    save_path = checkpoint_dir / f"checkpoint_{step}.pt"
    torch.save({
        "step": step,
        "projector": model.multi_modal_projector.state_dict(),
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
    model: TinyAyaVisionForConditionalGeneration,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: AlignmentConfig,
    checkpoint_dir: Path,
    compute_dtype: torch.dtype,
    projector_output_norms: list,
    step_offset: int = 0,
):
    model.train()
    accumulated_loss = 0.0
    accumulated_ce_loss = 0.0
    accumulated_align_reg_loss = 0.0

    for epoch in range(training_config.num_epochs):
        for step, batch in enumerate(dataloader, start=step_offset):
            input_ids, attention_mask, pixel_values, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["labels"].to(device),
            )
            image_grid_hws = batch.get("image_grid_hws")
            if image_grid_hws is not None:
                image_grid_hws = image_grid_hws.to(device)

            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_hws=image_grid_hws,
                    labels=labels,
                )
                ce_loss = outputs.loss / training_config.grad_acc_steps

                token_embeddings = model.language_model.get_input_embeddings().weight # (vocab size, D)
                image_hidden_states = outputs.image_hidden_states # (B, V, D) or (total_tokens, D)

                # Flatten to 2-D for both SigLIP (B, V, D) and MoonViT (total_tokens, D)
                ihs = image_hidden_states.reshape(-1, image_hidden_states.shape[-1])
                align_reg_loss = (token_embeddings.mean(dim=0) - ihs.mean(dim=0)).square().sum() \
                                + (token_embeddings.std(dim=0) - ihs.std(dim=0)).square().sum()
                align_reg_loss /= training_config.grad_acc_steps
                
            loss = ce_loss + training_config.embed_align_reg * align_reg_loss
            loss.backward()

            accumulated_loss += loss.item()
            accumulated_ce_loss += ce_loss.item()
            accumulated_align_reg_loss += align_reg_loss.item()

            if (step + 1) % training_config.grad_acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.multi_modal_projector.parameters(), training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                opt_step = (step + 1) // training_config.grad_acc_steps
                avg_projector_norm = sum(m for m, s in projector_output_norms) / len(projector_output_norms)
                avg_projector_norm_std = sum(s for m, s in projector_output_norms) / len(projector_output_norms)
                projector_output_norms.clear()
                wandb.log({
                    "train/loss": accumulated_loss,
                    "train/ce_loss": accumulated_ce_loss,
                    "train/align_reg_loss": accumulated_align_reg_loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/projector_output_norm": avg_projector_norm,
                    "train/projector_output_norm_std": avg_projector_norm_std,
                }, step=opt_step)

                if opt_step % training_config.logging_steps == 0:
                    print(f"Epoch {epoch}, Opt Step {opt_step}, Loss {accumulated_loss:.4f}, LR {lr_scheduler.get_last_lr()[0]}")

                if opt_step % training_config.save_steps == 0:
                    save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)

                accumulated_loss = 0.0
                accumulated_ce_loss = 0.0
                accumulated_align_reg_loss = 0.0

    save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)
    print("Training complete")


def run(cfg: DictConfig):
    """Core training logic. Can be called directly with a composed DictConfig."""
    # 1. Translate omegaconf nodes back to plain dictionaries
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    
    # 2. Re-instantiate your configurations to maintain types downwards
    training_config = AlignmentConfig(**training_dict)
    
    # Instantiate Model Config 
    model_config = TinyAyaVisionConfig.for_encoder(
        cfg.vision.vision_encoder_type, 
        llm=cfg.llm
    )
    
    # Optional logic: If vision params scale further inside the yaml than presets:
    # model_dict = OmegaConf.to_container(cfg.vision, resolve=True)
    # model_config = TinyAyaVisionConfig(**model_dict, llm=cfg.llm)

    # Allow CLI-based resuming (e.g. `python train.py resume=xyz123`)
    resume_run_id = cfg.get("resume", None)

    if resume_run_id:
        run_id = resume_run_id
    else:
        run_id = str(uuid.uuid4())

    checkpoint_dir = Path(training_config.models_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        import json
        with open(config_path, "w") as f:
            json.dump({
                "training_config": asdict(training_config),
                "model_config": asdict(model_config),
            }, f, indent=2)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_id,
        id=run_id.replace("-", ""),
        resume="allow",
        config=asdict(training_config),
    )

    model = TinyAyaVisionForConditionalGeneration(
        config=model_config,
    )
    model.to(device)

    processor = TinyAyaVisionProcessor(
        config=model_config,
    )

    model.setup_tokenizer(processor.tokenizer)

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False

    compute_dtype = getattr(torch, training_config.torch_dtype)
    model.vision_encoder.to(dtype=compute_dtype)
    model.language_model.to(dtype=compute_dtype)

    model.language_model.gradient_checkpointing_enable()

    projector_output_norms = []
    def _capture_norm(m, i, o):
        norms = o.detach().float().norm(dim=-1)
        projector_output_norms.append((norms.mean().item(), norms.std().item()))
    model.multi_modal_projector.register_forward_hook(_capture_norm)

    model = torch.compile(model)

    resume_step = 0
    if resume_run_id:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.multi_modal_projector.load_state_dict(ckpt["projector"])
            resume_step = ckpt["step"]
            print(f"Resuming from step {resume_step}")
        else:
            print(f"No checkpoints found in {checkpoint_dir}, starting from scratch")

    dataset = AlignmentDataset(
        config=model_config,
        dataset_name=training_config.dataset_name,
        data_dir=training_config.data_dir,
    )

    full_dataset_len = len(dataset)

    samples_to_skip = resume_step * training_config.batch_size
    if samples_to_skip > 0 and samples_to_skip < len(dataset):
        remaining_indices = list(range(samples_to_skip, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, remaining_indices)
        print(f"Skipped {samples_to_skip} samples, {len(dataset)} remaining")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token_id=processor.tokenizer.pad_token_id,
        ),
        num_workers=training_config.num_workers,
    )

    opt = torch.optim.AdamW(
        model.multi_modal_projector.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    full_loader_len = full_dataset_len // training_config.batch_size
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
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        compute_dtype=compute_dtype,
        projector_output_norms=projector_output_norms,
        step_offset=resume_step,
    )

    wandb.finish()


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
