#!/usr/bin/env python3
"""Apply LoRA adapters to the Tiny Aya Base LLM backbone.

Usage:
    uv run python scripts/apply_lora.py [options]

    --save-dir DIR    Save adapter weights to DIR after setup (optional).
    --rank INT        LoRA rank (default: 256).
    --alpha INT       LoRA alpha (default: 2 * rank).
    --layers-start N  First LLM layer to apply LoRA (default: 18).

This script:
  1. Loads TinyAyaVisionForConditionalGeneration.
  2. Freezes all parameters (vision encoder + LLM backbone).
  3. Keeps the MultiModalProjector trainable.
  4. Wraps model.language_model with PEFT LoRA using LoraAdapterConfig.
  5. Prints a detailed trainable-parameter summary.
  6. Optionally saves the adapter weights.

The returned model from apply_lora() is ready for Phase 2 SFT training.
Use get_lora_optimizer_groups() to build per-matrix optimizer param groups
if you want differential learning rates for A vs. B adapter matrices.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from peft import get_peft_model

from config.lora_config import LoraAdapterConfig
from config.model_config import TinyAyaVisionConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from src.processing import TinyAyaVisionProcessor


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def apply_lora(
    vlm_config: TinyAyaVisionConfig,
    lora_config: LoraAdapterConfig,
) -> TinyAyaVisionForConditionalGeneration:
    """Load the VLM and inject LoRA adapters into the LLM backbone.

    Trainable after this call:
      - model.multi_modal_projector  (~11.5M, unchanged from Phase 1)
      - model.language_model LoRA A/B matrices  (layers 18–35 by default)

    Frozen after this call:
      - model.vision_encoder           (~400M)
      - model.language_model base weights  (~3.35B)

    Args:
        vlm_config:  TinyAyaVisionConfig for model construction.
        lora_config: LoraAdapterConfig specifying rank, targets, and layers.

    Returns:
        TinyAyaVisionForConditionalGeneration with LoRA applied.
    """
    print("Loading TinyAyaVision model...")
    model = TinyAyaVisionForConditionalGeneration(vlm_config)
    processor = TinyAyaVisionProcessor(vlm_config)
    model.setup_tokenizer(processor.tokenizer)

    # Freeze everything first — vision encoder and LLM base weights.
    for param in model.parameters():
        param.requires_grad = False

    # The connector is always trainable (same as Phase 1).
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True

    # Wrap only the LLM backbone with PEFT LoRA.
    # This inserts trainable A/B matrices while leaving base weights frozen.
    peft_config = lora_config.to_peft_config()
    model.language_model = get_peft_model(model.language_model, peft_config)

    return model


def get_lora_optimizer_groups(
    model: nn.Module,
    base_lr: float,
    lora_config: LoraAdapterConfig,
) -> list[dict]:
    """Build optimizer parameter groups with optional differential LRs.

    Splits trainable parameters into three groups:
      - "lora_A"  LoRA down-projection (input → rank): lr = base_lr * lora_a_lr_multiplier
      - "lora_B"  LoRA up-projection   (rank → output): lr = base_lr * lora_b_lr_multiplier
      - "other"   Remaining trainable params (connector, enabled biases): lr = base_lr

    When lora_a_lr_multiplier == lora_b_lr_multiplier == 1.0, all groups share
    base_lr and behaviour is equivalent to a single optimizer param group.

    Args:
        model:       Model returned by apply_lora() (or any nn.Module).
        base_lr:     Base learning rate.
        lora_config: Used for lr multiplier values.

    Returns:
        List of dicts suitable for passing to torch.optim.AdamW(groups, ...).
    """
    lora_a_params: list[nn.Parameter] = []
    lora_b_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)
        else:
            other_params.append(param)

    return [
        {
            "params": lora_a_params,
            "lr": base_lr * lora_config.lora_a_lr_multiplier,
            "name": "lora_A",
        },
        {
            "params": lora_b_params,
            "lr": base_lr * lora_config.lora_b_lr_multiplier,
            "name": "lora_B",
        },
        {
            "params": other_params,
            "lr": base_lr,
            "name": "other",
        },
    ]


def print_param_summary(model: TinyAyaVisionForConditionalGeneration) -> None:
    """Print trainable vs. frozen parameter counts by component."""
    trainable, total = count_parameters(model)
    frozen = total - trainable

    print("\n── Parameter Summary ───────────────────────────────────")
    print(f"  Total:     {total:>15,}")
    print(f"  Trainable: {trainable:>15,}  ({100 * trainable / total:.3f}%)")
    print(f"  Frozen:    {frozen:>15,}")

    components = {
        "vision_encoder": model.vision_encoder,
        "multi_modal_projector": model.multi_modal_projector,
        "language_model (base+LoRA)": model.language_model,
    }
    print("\n  By component:")
    for name, mod in components.items():
        tr, tot = count_parameters(mod)
        print(f"    {name:<30}  trainable={tr:>12,}  total={tot:>12,}")
    print("────────────────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply LoRA adapters to the Tiny Aya Base LLM backbone."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save PEFT adapter weights (optional).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=256,
        help="LoRA rank (default: 256).",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: 2 × rank).",
    )
    parser.add_argument(
        "--layers-start",
        type=int,
        default=18,
        help="First LLM layer index to apply LoRA (default: 18).",
    )
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else args.rank * 2

    lora_config = LoraAdapterConfig(
        rank=args.rank,
        lora_alpha=alpha,
        layers_to_transform=list(range(args.layers_start, 36)),
    )

    print(f"LoRA config:")
    print(f"  rank={lora_config.rank}, alpha={lora_config.lora_alpha} "
          f"(scale={lora_config.lora_alpha / lora_config.rank:.1f})")
    print(f"  layers: {lora_config.layers_to_transform[0]}–"
          f"{lora_config.layers_to_transform[-1]} "
          f"({len(lora_config.layers_to_transform)} layers)")
    print(f"  target modules: {lora_config.target_modules}")

    vlm_config = TinyAyaVisionConfig()
    model = apply_lora(vlm_config, lora_config)
    print_param_summary(model)

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.language_model.save_pretrained(save_path)
        print(f"Adapter weights saved to: {save_path}")


if __name__ == "__main__":
    main()
