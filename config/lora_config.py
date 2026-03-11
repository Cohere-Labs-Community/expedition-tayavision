from dataclasses import dataclass, field

from peft import LoraConfig, TaskType

# Tiny Aya Base has 36 transformer layers (indices 0–35).
# "Mid-to-top" targets the upper half: layers 18–35.
_NUM_LLM_LAYERS = 36
_FIRST_MID_LAYER = _NUM_LLM_LAYERS // 2  # 18


@dataclass
class LoraAdapterConfig:
    """LoRA adapter configuration for the Tiny Aya Base (Cohere2) backbone.

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

    # Layer indices to inject LoRA into (mid-to-top: layers 18–35 of 36 total).
    # Lower layers encode general language/multilingual knowledge that we want
    # to preserve; upper layers are more task-specific and benefit from adaptation.
    layers_to_transform: list[int] = field(
        default_factory=lambda: list(range(_FIRST_MID_LAYER, _NUM_LLM_LAYERS))
    )

    # Differential LR multipliers for A and B matrices.
    # Set both to 1.0 to use a uniform LR for all adapter parameters.
    lora_a_lr_multiplier: float = 1.0
    lora_b_lr_multiplier: float = 1.0

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
