from __future__ import annotations

import inspect
from pathlib import Path

import yaml
from transformers import PretrainedConfig


class TinyAyaVisionConfig(PretrainedConfig):
    """Central configuration for the Tiny Aya Vision model."""

    model_type = "tiny_aya_vision"

    def __init__(
        self,
        vision_encoder_type: str = "siglip",
        vision_model_name: str = "google/siglip2-so400m-patch14-384",
        vision_hidden_size: int = 1152,
        image_size: int = 384,
        patch_size: int = 14,
        vision_grid_size: int = 27,
        num_vision_tokens: int = 729,
        trust_remote_code: bool = False,
        connector_type: str = "pixel_shuffle",
        downsample_factor: int = 2,
        padded_grid_size: int = 28,
        num_tokens_after_shuffle: int = 196,
        pixel_shuffle_embed_dim: int = 4608,
        tokens_per_tile: int = 4,
        in_token_limit: int = 1024,
        connector_intermediate_size: int = 2048,
        adapter_layer_norm_eps: float = 1e-6,
        post_projector_rms_norm: bool = False,
        llm_model_name: str = "CohereLabs/tiny-aya-base",
        llm_hidden_size: int = 2048,
        llm_vocab_size: int = 262144,
        num_llm_layers: int = 36,
        image_token: str = "<image>",
        image_token_id: int | None = None,
        torch_dtype: str = "bfloat16",
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "full",
        cache_dir: str | None = None,
        text_config: dict | None = None,
        vision_tower_config: dict | None = None,
        **kwargs,
    ):
        self.vision_encoder_type = vision_encoder_type
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_grid_size = vision_grid_size
        self.num_vision_tokens = num_vision_tokens
        self.trust_remote_code = trust_remote_code
        self.connector_type = connector_type
        self.downsample_factor = downsample_factor
        self.padded_grid_size = padded_grid_size
        self.num_tokens_after_shuffle = num_tokens_after_shuffle
        self.pixel_shuffle_embed_dim = pixel_shuffle_embed_dim
        self.tokens_per_tile = tokens_per_tile
        self.in_token_limit = in_token_limit
        self.connector_intermediate_size = connector_intermediate_size
        self.adapter_layer_norm_eps = adapter_layer_norm_eps
        self.post_projector_rms_norm = post_projector_rms_norm
        self.llm_model_name = llm_model_name
        self.llm_hidden_size = llm_hidden_size
        self.llm_vocab_size = llm_vocab_size
        self.num_llm_layers = num_llm_layers
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.cache_dir = cache_dir
        self.text_config = text_config
        self.vision_tower_config = vision_tower_config
        self._text_config_obj = None
        super().__init__(torch_dtype=torch_dtype, **kwargs)

    def get_text_config(self, decoder: bool = False) -> "PretrainedConfig":
        """Return a proper PretrainedConfig for the LLM sub-model.

        Required by transformers >=4.56 for DynamicCache initialization
        during generate().
        """
        if self._text_config_obj is None and self.text_config is not None:
            from transformers import CONFIG_MAPPING
            text_cls = CONFIG_MAPPING[self.text_config["model_type"]]
            self._text_config_obj = text_cls.from_dict(self.text_config)
        if self._text_config_obj is not None:
            return self._text_config_obj
        return super().get_text_config(decoder=decoder)

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

        sig = inspect.signature(cls.__init__)
        valid_fields = set(sig.parameters.keys()) - {"self", "kwargs"}
        filtered = {k: v for k, v in overrides.items() if k in valid_fields}

        llm_names = {
            "base": "CohereLabs/tiny-aya-base",
            "global": "CohereLabs/tiny-aya-global",
        }
        if llm not in llm_names:
            raise ValueError(f"llm must be 'base' or 'global', got '{llm}'")

        return cls(**filtered, llm_model_name=llm_names[llm])
