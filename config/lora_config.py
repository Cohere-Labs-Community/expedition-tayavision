from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from peft import LoraConfig, TaskType

if TYPE_CHECKING:
    from config.model_config import TinyAyaVisionConfig


@dataclass
class LoraAdapterConfig:
    """LoRA adapter configuration for the Tiny Aya (Cohere2) backbone.

    LoRA injects trainable rank-decomposition matrices A (down) and B (up) into
    frozen linear layers. Only the injected adapter weights are updated during
    fine-tuning; the original backbone weights stay frozen.

    Adapter effective scaling: lora_alpha / rank (default 512/256 = 2.0).

    Differential learning rates:
        lora_a_lr_multiplier / lora_b_lr_multiplier scale base_lr independently
        for A and B matrices. Pass these to get_lora_optimizer_groups() in
        scripts/apply_lora.py to construct per-matrix optimizer parameter groups.
        Both default to 1.0 (uniform LR).
    """

    # Core LoRA hyperparameters
    rank: int = 256
    lora_alpha: int = 512  # scaling = alpha / rank = 2.0
    lora_dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"

    # Target submodules within each transformer layer.
    # Covers all attention projections (Q/K/V/O) and SwiGLU MLP projections.
    target_modules: list[str] = field(
        default_factory=lambda: [
            # Attention
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP (SwiGLU gate + up + down)
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Layer indices to inject LoRA into (mid-to-top: upper half of total layers).
    # Lower layers encode general language/multilingual knowledge that we want
    # to preserve; upper layers are more task-specific and benefit from adaptation.
    # Default targets layers 18–35 of 36 total (Cohere2 with 36 layers).
    layers_to_transform: list[int] = field(
        default_factory=lambda: list(range(18, 36))
    )

    # Differential LR multipliers for A and B matrices.
    # Set both to 1.0 to use a uniform LR for all adapter parameters.
    lora_a_lr_multiplier: float = 1.0
    lora_b_lr_multiplier: float = 1.0

    @classmethod
    def from_vlm_config(
        cls, vlm_config: TinyAyaVisionConfig, **kwargs
    ) -> LoraAdapterConfig:
        """Build a LoraAdapterConfig targeting the upper half of the LLM's layers.

        Derives layer indices from vlm_config.num_llm_layers so the config
        stays correct regardless of which Tiny Aya variant is used.
        """
        num_layers = vlm_config.num_llm_layers
        first_mid = num_layers // 2
        layers = list(range(first_mid, num_layers))
        return cls(layers_to_transform=layers, **kwargs)

    def to_peft_config(self) -> LoraConfig:
        """Return a PEFT LoraConfig ready for use with get_peft_model()."""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            target_modules=self.target_modules,
            layers_to_transform=self.layers_to_transform,
        )
