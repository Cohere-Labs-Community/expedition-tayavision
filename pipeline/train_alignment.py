import json
import re
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path

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
    step_offset: int = 0,
):
    model.train()
    accumulated_loss = 0.0

    for epoch in range(training_config.num_epochs):
        for step, batch in enumerate(dataloader, start=step_offset):
            input_ids, attention_mask, pixel_values, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["labels"].to(device),
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.multi_modal_projector.parameters(), training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                opt_step = (step + 1) // training_config.grad_acc_steps
                wandb.log({
                    "train/loss": accumulated_loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }, step=opt_step)

                if opt_step % training_config.logging_steps == 0:
                    print(f"Epoch {epoch}, Opt Step {opt_step}, Loss {accumulated_loss:.4f}, LR {lr_scheduler.get_last_lr()[0]}")

                if opt_step % training_config.save_steps == 0:
                    save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)

                accumulated_loss = 0.0

    save_checkpoint(checkpoint_dir, step + 1, model, optimizer, lr_scheduler)
    print("Training complete")


def main(
    training_config: AlignmentConfig,
    model_config: TinyAyaVisionConfig,
    resume_run_id: str | None = None,
):
    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)

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
        with open(config_path, "w") as f:
            json.dump({
                "training_config": asdict(training_config),
                "model_config": asdict(model_config),
            }, f, indent=2)

    wandb.init(
        project="tayavision",
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
        step_offset=resume_step,
    )

    wandb.finish()

if __name__ == "__main__":
    main(
        training_config=AlignmentConfig(),
        model_config=TinyAyaVisionConfig(),
    )
