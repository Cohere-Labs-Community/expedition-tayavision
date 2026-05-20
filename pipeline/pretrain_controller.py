"""Stage-1 controller pre-training pipeline for Tiny Aya Vision.

Trains only ``script_controller`` (everything else frozen) on a balanced
multilingual subsample (~700 samples per language). Goal: give the
controller a non-trivial per-language policy before the joint multilingual
run, so it doesn't collapse to the dominant-class policy under the 40%
English training distribution.

The controller learns from a fertility-derived per-sample
``target_tokens``: high-fertility scripts get a larger token budget, low
fertility gets a smaller one. CE loss is logged but has zero gradient
w.r.t. the controller in soft mode (a known limitation of the current
forward path); the per-sample rate target is what drives learning.

Output: ``controller_<step>.pt`` files holding only
``script_controller.state_dict()``. Stage-2 ``train_multilingual.py``
reads these via the new ``controller_checkpoint`` flag.

Launch:
  Single GPU only — controller is 387 params, no DDP/torch.compile.

      python pipeline/pretrain_controller.py \
          backbone=qwen3 controller=b1 training=controller_pretrain
"""

import json
import os
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

from config.lora_config import LoraAdapterConfig
from config.model_config import TinyAyaVisionConfig
from config.multilingual_config import MultilingualInstructConfig
from pipeline.apply_lora import apply_lora
from pipeline.data import collate_fn
from pipeline.multilingual_data import MultilingualInstructDataset
from pipeline.utils import build_lr_scheduler, is_torchrun
from src.processing import TinyAyaVisionProcessor


def _build_balanced_dataset(
    training_config: MultilingualInstructConfig,
    model_config: TinyAyaVisionConfig,
) -> MultilingualInstructDataset:
    """Construct the Stage-1 balanced multilingual dataset.

    The standard ``MultilingualInstructDataset`` already supports near-uniform
    per-language sampling via ``temperature=1e6`` and a small ``english_ratio``;
    no subclassing required. We just call it with the Stage-1 yaml params.
    """
    return MultilingualInstructDataset(
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


def _preflight_checks(
    model,
    dataset: MultilingualInstructDataset,
    loader,
    processor: TinyAyaVisionProcessor,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Abort early if any prereq is broken."""
    # 1. Sample-level lang_code presence + fertility lookup distinctness.
    fertility = model.script_controller.fertility_table
    sample_langs, sample_tpws = [], []
    for i in range(min(32, len(dataset))):
        item = dataset[i]
        lang = item.get("lang_code")
        if lang is None:
            raise RuntimeError(
                f"Prereq broken: dataset[{i}] has no 'lang_code' key. "
                "Plumb it in MultilingualInstructDataset._get_example."
            )
        sample_langs.append(lang)
        sample_tpws.append(fertility.lookup_tpw(lang))

    distinct_tpw = len({round(x, 3) for x in sample_tpws})
    non_default = sum(1 for x in sample_tpws if x != 1.0)
    print(
        f"Pre-flight: {len(sample_langs)} samples, "
        f"{distinct_tpw} distinct tpw values, "
        f"{non_default}/{len(sample_langs)} non-default lookups"
    )
    if distinct_tpw < 10:
        raise RuntimeError(
            f"Prereq broken: only {distinct_tpw} distinct tpw values in 32 samples "
            f"(want >= 10). Likely missing fertility table or unmapped lang_codes. "
            f"Samples: {list(zip(sample_langs, sample_tpws))[:8]}"
        )

    # 2. Sample one batch through the model; assert per-sample target varies.
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    pixel_values = batch["pixel_values"]
    if pixel_values is not None:
        pixel_values = pixel_values.to(device, non_blocking=True)
    lang_codes = batch.get("lang_codes")
    with torch.no_grad(), torch.autocast("cuda", dtype=compute_dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,
            use_cache=False,
            lang_codes=lang_codes,
        )
    aux = outputs.controller_aux or {}
    et_std = float(aux["expected_tokens"].float().std().item()) if "expected_tokens" in aux else 0.0
    tt_std = float(aux["target_tokens"].float().std().item()) if "target_tokens" in aux else 0.0
    print(f"Pre-flight: expected_tokens.std={et_std:.3f}, target_tokens.std={tt_std:.3f}")
    if tt_std == 0.0:
        raise RuntimeError(
            "Prereq broken: target_tokens has zero std across a batch. "
            "Either the fertility table is empty, lang_codes is None, or the "
            "per-sample target_tokens change to ScriptController.forward "
            "didn't land."
        )


def _save_controller(model, checkpoint_dir: Path, step: int) -> Path:
    """Save *only* ``script_controller.state_dict()`` to disk."""
    save_path = checkpoint_dir / f"controller_{step}.pt"
    state = model.script_controller.state_dict()
    torch.save(state, save_path)
    print(f"Saved controller checkpoint to {save_path}")
    return save_path


def train_loop(
    model,
    loader,
    optimizer,
    lr_scheduler,
    training_config: MultilingualInstructConfig,
    checkpoint_dir: Path,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Slim Stage-1 training loop. ~80 lines, no DDP/compile gymnastics."""
    model.train()
    # Pre-cache identifiers for per-language logging.
    last_full_step = 0
    last_norm = float(
        sum(p.detach().float().norm() ** 2 for p in model.script_controller.parameters()).sqrt().item()
    )

    accumulated_loss = 0.0
    accumulated_ce = 0.0
    accumulated_rate = 0.0
    accumulated_et = 0.0
    accumulated_et_var = 0.0
    accumulated_tt_var = 0.0
    accumulated_tpw_resolved = 0.0
    accumulated_probs = torch.zeros(len(model.script_controller.compression_levels), device=device)
    micro_steps_in_window = 0

    # Per-language running mean of expected_tokens, logged at save_steps.
    per_lang_et_sum: dict[str, float] = {}
    per_lang_et_n: dict[str, int] = {}

    rate_lambda = float(getattr(model.script_controller.cfg, "rate_lambda", 0.1))

    for epoch in range(training_config.num_epochs):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"]
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lang_codes = batch.get("lang_codes")

            with torch.autocast("cuda", dtype=compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    use_cache=False,
                    lang_codes=lang_codes,
                )
                ce_loss = outputs.loss / training_config.grad_acc_steps
                aux = outputs.controller_aux or {}
                rate_term = aux.get("rate_loss")
                if rate_term is None:
                    raise RuntimeError(
                        "controller_aux missing 'rate_loss'. Is controller=b1 enabled?"
                    )
                rate_term = rate_term / training_config.grad_acc_steps
                # Stage 1 backward only on rate_term — ce_loss has zero
                # gradient w.r.t. controller in soft mode (the forward
                # pass-through). It's logged purely for monitoring.
                loss = rate_lambda * rate_term

            loss.backward()

            # Aggregate per-window stats (FP32, no autocast).
            with torch.no_grad():
                et = aux["expected_tokens"].float()
                tt = aux["target_tokens"].float()
                probs_mean = aux["compression_probs"].float().mean(dim=0)
                accumulated_loss += float(loss.detach().item())
                accumulated_ce += float(ce_loss.detach().item())
                accumulated_rate += float(rate_term.detach().item())
                accumulated_et += float(et.mean().item())
                accumulated_et_var += float(et.var(unbiased=False).item())
                accumulated_tt_var += float(tt.var(unbiased=False).item())
                accumulated_probs += probs_mean
                if lang_codes is not None:
                    fertility = model.script_controller.fertility_table
                    et_cpu = et.cpu().tolist()
                    n_resolved = 0
                    for lang, et_val in zip(lang_codes, et_cpu):
                        per_lang_et_sum[lang] = per_lang_et_sum.get(lang, 0.0) + et_val
                        per_lang_et_n[lang] = per_lang_et_n.get(lang, 0) + 1
                        if fertility.lookup_tpw(lang) != 1.0:
                            n_resolved += 1
                    accumulated_tpw_resolved += n_resolved / max(1, len(lang_codes))
                micro_steps_in_window += 1

            if (step + 1) % training_config.grad_acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.script_controller.parameters(), training_config.max_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step = (step + 1) // training_config.grad_acc_steps + epoch * (len(loader) // training_config.grad_acc_steps)
                last_full_step = opt_step

                if opt_step % training_config.logging_steps == 0:
                    current_norm = float(
                        sum(p.detach().float().norm() ** 2 for p in model.script_controller.parameters()).sqrt().item()
                    )
                    n = max(1, micro_steps_in_window)
                    log_dict = {
                        "train/loss": accumulated_loss,
                        "train/ce_loss": accumulated_ce,
                        "train/rate_loss": accumulated_rate,
                        "train/expected_tokens": accumulated_et / n,
                        "train/expected_tokens_std": (accumulated_et_var / n) ** 0.5,
                        "train/target_tokens_std": (accumulated_tt_var / n) ** 0.5,
                        "train/tpw_resolution_rate": accumulated_tpw_resolved / n,
                        "train/grad_norm": float(grad_norm.item()),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "controller/mlp_norm": current_norm,
                        "controller/mlp_norm_delta": current_norm - last_norm,
                    }
                    probs_mean_w = (accumulated_probs / n).tolist()
                    for level_idx, p in enumerate(probs_mean_w):
                        level = model.script_controller.compression_levels[level_idx]
                        log_dict[f"train/p_level_{level}"] = float(p)
                    wandb.log(log_dict, step=opt_step)
                    last_norm = current_norm
                    print(
                        f"step={opt_step}  rate={accumulated_rate:.3f}  "
                        f"et={accumulated_et/n:.2f}  et_std={(accumulated_et_var/n)**0.5:.2f}  "
                        f"tt_std={(accumulated_tt_var/n)**0.5:.2f}"
                    )

                if opt_step % training_config.save_steps == 0:
                    _save_controller(model, checkpoint_dir, opt_step)
                    if per_lang_et_n:
                        rows = []
                        fertility = model.script_controller.fertility_table
                        for lang in sorted(per_lang_et_n.keys()):
                            mean_et = per_lang_et_sum[lang] / per_lang_et_n[lang]
                            rows.append([lang, fertility.lookup_tpw(lang), mean_et, per_lang_et_n[lang]])
                        table = wandb.Table(
                            columns=["lang_code", "tpw", "mean_expected_tokens", "n"],
                            data=rows,
                        )
                        wandb.log({"per_language/expected_tokens": table}, step=opt_step)

                # Reset window accumulators
                accumulated_loss = 0.0
                accumulated_ce = 0.0
                accumulated_rate = 0.0
                accumulated_et = 0.0
                accumulated_et_var = 0.0
                accumulated_tt_var = 0.0
                accumulated_tpw_resolved = 0.0
                accumulated_probs.zero_()
                micro_steps_in_window = 0

    # Final save
    _save_controller(model, checkpoint_dir, last_full_step + 1)
    print("Stage-1 controller pre-training complete.")


def main(
    training_config: MultilingualInstructConfig,
    model_config: TinyAyaVisionConfig,
    lora_config: LoraAdapterConfig,
):
    if is_torchrun():
        raise RuntimeError(
            "pretrain_controller is single-GPU only. The controller has 387 params; "
            "DDP/torch.compile is pure overhead. Launch without torchrun."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Stage-1 controller pre-training: device={device}")
    print(f"  Total samples: {training_config.total_samples}")
    print(f"  English ratio: {training_config.english_ratio}")
    print(f"  Temperature:   {training_config.temperature}")

    run_id = str(uuid.uuid4())
    checkpoint_dir = Path(training_config.models_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "training_config": asdict(training_config),
                "model_config": model_config.to_dict(),
                "lora_config": asdict(lora_config),
            },
            f,
            indent=2,
        )

    wandb.init(
        project="tayavision-controller-pretrain",
        name=run_id,
        id=run_id.replace("-", ""),
        resume="allow",
        config={**asdict(training_config), **asdict(lora_config), "stage": "controller_pretrain"},
    )

    # Build model with LoRA adapters (same as Stage 2), then load instruct
    # checkpoint so the projector + LoRA match what Stage 2 will fork from.
    model = apply_lora(vlm_config=model_config, lora_config=lora_config)
    if training_config.instruct_checkpoint:
        ckpt = torch.load(
            training_config.instruct_checkpoint, map_location="cpu", weights_only=True,
        )
        projector_state = ckpt["projector"] if "projector" in ckpt else ckpt
        model.multi_modal_projector.load_state_dict(projector_state)
        lora_state = ckpt.get("lora_adapter", {})
        if lora_state:
            model.language_model.load_state_dict(lora_state, strict=False)
        print(f"Loaded projector + LoRA from {training_config.instruct_checkpoint}")
    else:
        raise RuntimeError(
            "Stage 1 requires training.instruct_checkpoint to be set so the "
            "downstream stack matches what Stage 2 will fork from."
        )

    if model.script_controller is None:
        raise RuntimeError(
            "Stage 1 requires controller=b1 (or b2/b3). Got controller=off."
        )

    # Freeze everything; unfreeze only the controller MLP.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.script_controller.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (controller only): {n_trainable}")
    assert n_trainable > 0 and n_trainable < 10_000, (
        f"Expected ~387 trainable controller params, got {n_trainable}. "
        "Check the freeze logic."
    )

    model.to(device, non_blocking=True)
    compute_dtype = getattr(torch, training_config.torch_dtype)
    model.vision_encoder.to(dtype=compute_dtype, non_blocking=True)
    model.language_model.to(dtype=compute_dtype, non_blocking=True)
    model.script_controller.to(dtype=compute_dtype, non_blocking=True)

    # Build balanced dataset + verify per-language balance.
    dataset = _build_balanced_dataset(training_config, model_config)
    processor = TinyAyaVisionProcessor(config=model_config)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=training_config.num_workers > 0,
        prefetch_factor=2 if training_config.num_workers > 0 else None,
        drop_last=False,
    )

    _preflight_checks(model, dataset, loader, processor, compute_dtype, device)

    # Optimizer over only the controller MLP.
    opt = torch.optim.AdamW(
        model.script_controller.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    lr_scheduler = build_lr_scheduler(
        opt,
        training_config,
        full_dataset_len=len(dataset),
        per_gpu_batch_size=training_config.batch_size,
        world_size=1,
    )

    train_loop(
        model=model,
        loader=loader,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        compute_dtype=compute_dtype,
        device=device,
    )

    wandb.finish()


def run(cfg: DictConfig):
    training_dict = OmegaConf.to_container(cfg.training, resolve=True)
    lora_dict = training_dict.pop("lora", {})
    multilingual_sources = training_dict.pop("multilingual_sources", [])
    training_config = MultilingualInstructConfig(
        **training_dict, multilingual_sources=multilingual_sources,
    )

    backbone_name = (
        cfg.get("backbone", {}).get("backbone_type", "tiny_aya")
        if "backbone" in cfg
        else "tiny_aya"
    )
    model_config = TinyAyaVisionConfig.for_backbone(
        backbone=backbone_name,
        encoder=cfg.vision.vision_encoder_type,
    )
    for group_name in ("vision", "backbone"):
        if group_name in cfg:
            for k, v in OmegaConf.to_container(cfg[group_name], resolve=True).items():
                if hasattr(model_config, k):
                    setattr(model_config, k, v)
    if model_config.backbone_type == "tiny_aya":
        model_config.llm_model_name = {
            "base": "CohereLabs/tiny-aya-base",
            "global": "CohereLabs/tiny-aya-global",
        }[cfg.llm]
    if "controller" in cfg:
        model_config.controller_config = OmegaConf.to_container(cfg.controller, resolve=True)

    if "layers_to_transform" not in lora_dict:
        n = model_config.num_llm_layers
        lora_dict["layers_to_transform"] = list(range(n // 2, n))
    if "target_modules" not in lora_dict and model_config.lora_target_modules:
        lora_dict["target_modules"] = list(model_config.lora_target_modules)
    lora_config = LoraAdapterConfig(**lora_dict)

    main(
        training_config=training_config,
        model_config=model_config,
        lora_config=lora_config,
    )


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def hydra_main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    hydra_main()
