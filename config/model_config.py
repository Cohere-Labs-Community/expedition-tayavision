from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TinyAyaVisionConfig:
    """Central configuration for the Tiny Aya Vision model."""

    # Vision encoder type: "siglip" | "moonvit"
    vision_encoder_type: str = "siglip"

    # Vision encoder (SigLIP2-so400m-patch14-384)
    vision_model_name: str = "google/siglip2-so400m-patch14-384"
    vision_hidden_size: int = 1152
    image_size: int = 384
    patch_size: int = 14
    vision_grid_size: int = 27  # 384 // 14
    num_vision_tokens: int = 729  # 27 * 27
    trust_remote_code: bool = False

    # Connector type: "pixel_shuffle" | "linear_mlp"
    connector_type: str = "pixel_shuffle"

    # Pixel Shuffle (4x token compression) — SigLIP only
    downsample_factor: int = 2
    padded_grid_size: int = 28  # ceil_to_even(27)
    num_tokens_after_shuffle: int = 196  # (28 // 2) ** 2
    pixel_shuffle_embed_dim: int = 4608  # 1152 * (2 ** 2)

    # MoonViT — tokens per tile (output structure: list of (N_tiles, tokens_per_tile, D))
    tokens_per_tile: int = 4

    # Vision-language connector (2-layer MLP with SwiGLU)
    connector_intermediate_size: int = 2048  # matches LLM hidden_size
    adapter_layer_norm_eps: float = 1e-6

    # LLM backbone
    llm_model_name: str = "CohereLabs/tiny-aya-base"
    llm_hidden_size: int = 2048
    llm_vocab_size: int = 262144
    num_llm_layers: int = 36  # Cohere2: 36 transformer layers

    # Special tokens
    image_token: str = "<image>"

    # Inference defaults
    torch_dtype: str = "bfloat16"

    # Vision feature extraction
    vision_feature_layer: int = -1
    # "full" = all patches, "default" = crop CLS
    vision_feature_select_strategy: str = "full"

    @classmethod
    def for_base(cls) -> TinyAyaVisionConfig:
        """Config for CohereLabs/tiny-aya-base (pretrained base model)."""
        return cls(llm_model_name="CohereLabs/tiny-aya-base")

    @classmethod
    def for_global(cls) -> TinyAyaVisionConfig:
        """Config for CohereLabs/tiny-aya-global (instruction-tuned, best multilingual balance)."""
        return cls(llm_model_name="CohereLabs/tiny-aya-global")

    @classmethod
    def for_encoder(cls, encoder: str, llm: str = "base") -> TinyAyaVisionConfig:
        """Load config from config/vision/<encoder>.yaml and merge with defaults.

        Args:
            encoder: Vision encoder name — "siglip" or "moonvit".
            llm: LLM variant — "base" or "global".

        Example:
            config = TinyAyaVisionConfig.for_encoder("moonvit")
            config = TinyAyaVisionConfig.for_encoder("siglip", llm="global")
        """
        yaml_path = Path(__file__).parent / "vision" / f"{encoder}.yaml"
        if not yaml_path.exists():
            available = [p.stem for p in yaml_path.parent.glob("*.yaml")]
            raise FileNotFoundError(
                f"No vision config for '{encoder}' at {yaml_path}. "
                f"Available: {available}"
            )

        with open(yaml_path) as f:
            overrides = yaml.safe_load(f)

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in valid_fields}

        llm_names = {
            "base": "CohereLabs/tiny-aya-base",
            "global": "CohereLabs/tiny-aya-global",
        }
        if llm not in llm_names:
            raise ValueError(f"llm must be 'base' or 'global', got '{llm}'")

        return cls(**filtered, llm_model_name=llm_names[llm])
