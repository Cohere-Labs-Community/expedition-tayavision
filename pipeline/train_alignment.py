import uuid

import torch

from functools import partial
from pathlib import Path

from config.training_config import AlignmentConfig
from config.model_config import TinyAyaVisionConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from pipeline.data import AlignmentDataset, collate_fn
from src.processing import TinyAyaVisionProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(
    model: TinyAyaVisionForConditionalGeneration,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: AlignmentConfig,
    checkpoint_dir: Path,
):
    model.train()

    for epoch in range(training_config.num_epochs):
        for step, batch in enumerate(dataloader):
            input_ids, attention_mask, pixel_values, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["labels"].to(device),
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss / training_config.grad_acc_steps
            loss.backward()

            if (step + 1) % training_config.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.multi_modal_projector.parameters(), training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % training_config.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step + 1}, Loss {loss.item()}, LR {lr_scheduler.get_last_lr()[0]}")

            if (step + 1) % training_config.save_steps == 0:
                save_path = checkpoint_dir / f"projector_{step + 1}.pt"
                torch.save(model.multi_modal_projector.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

    final_path = checkpoint_dir / "projector_final.pt"
    torch.save(model.multi_modal_projector.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")


def main(
    training_config: AlignmentConfig,
    model_config: TinyAyaVisionConfig,
):
    run_uuid = uuid.uuid4()
    checkpoint_dir = Path(training_config.models_dir) / str(run_uuid)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run UUID: {run_uuid}")
    print(f"Checkpoint dir: {checkpoint_dir}")

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

    model.language_model.gradient_checkpointing_enable()

    dataset = AlignmentDataset(
        config=model_config,
        dataset_name=training_config.dataset_name,
        data_dir=training_config.data_dir,
    )
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

    total_steps = training_config.num_epochs * len(loader)
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


    train(
        model=model,
        dataloader=loader,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main(
        training_config=AlignmentConfig(),
        model_config=TinyAyaVisionConfig(),
    )
