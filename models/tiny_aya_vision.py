from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import ModelOutput

from config.model_config import TinyAyaVisionConfig
from src.connector import create_projector
from src.vision_encoders import create_vision_encoder


@dataclass
class TinyAyaVisionOutput(ModelOutput):
    """Output type for TinyAyaVisionForConditionalGeneration."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: tuple | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class TinyAyaVisionForConditionalGeneration(nn.Module, GenerationMixin):
    """Tiny Aya Vision: multilingual VLM connecting a vision encoder to Tiny Aya Base.

    Architecture:
        VisionEncoder (frozen) -> Connector -> Cohere2 LLM

    Supports swappable vision encoders (SigLIP, MoonViT) via TinyAyaVisionConfig.
    The model replaces <image> placeholder tokens in the text sequence with
    projected vision features, then runs the combined sequence through the LLM.
    """

    main_input_name = "input_ids"
    _is_stateful = False  # required by GenerationMixin._supports_default_dynamic_cache

    def __init__(self, config: TinyAyaVisionConfig):
        super().__init__()
        self.vlm_config = config

        # Vision encoder (frozen) — selected by config.vision_encoder_type
        self.vision_encoder = create_vision_encoder(config)

        # Vision-language connector (trainable), cast to configured dtype so
        # it matches the bfloat16 outputs from the frozen vision encoder.
        self.multi_modal_projector = create_projector(config).to(
            getattr(torch, config.torch_dtype)
        )

        # Language model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=getattr(torch, config.torch_dtype),
        )

        # Expose the LLM's PretrainedConfig as self.config so GenerationMixin
        # can call self.config._get_generation_parameters() and related methods.
        self.config = self.language_model.config

        # Store the image token id (set during tokenizer setup)
        self._image_token_id: int | None = None

    @property
    def image_token_id(self) -> int:
        if self._image_token_id is None:
            raise ValueError(
                "image_token_id not set. Call setup_tokenizer() first."
            )
        return self._image_token_id

    @property
    def device(self) -> torch.device:
        return next(self.language_model.parameters()).device

    @property
    def generation_config(self):
        return self.language_model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.language_model.generation_config = value

    def setup_tokenizer(self, tokenizer) -> None:
        """Add <image> special token to tokenizer and resize embeddings.

        Must be called after initialization and before forward/generate.
        """
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.vlm_config.image_token]}
        )
        self._image_token_id = tokenizer.convert_tokens_to_ids(self.vlm_config.image_token)

        if num_added > 0:
            self.language_model.resize_token_embeddings(len(tokenizer))

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract and project image features.

        Args:
            pixel_values: Preprocessed image tensor(s).
            image_grid_hws: (B, 2) tile-grid dimensions, required for MoonViT.

        Returns:
            SigLIP: (B, num_tokens_after_shuffle, llm_hidden_size) tensor.
            MoonViT: list of (N_tiles_i * tokens_per_tile, llm_hidden_size) tensors,
                     one per image.
        """
        if self.vlm_config.vision_encoder_type == "moonvit":
            # Returns list of (N_tiles_i, tokens_per_tile, D) tensors
            raw_features = self.vision_encoder(pixel_values, image_grid_hws=image_grid_hws)
            projected = []
            for feat in raw_features:
                # (N_tiles, tokens_per_tile, D) -> (N_tiles * tokens_per_tile, D)
                feat = feat.view(-1, feat.shape[-1])
                proj = self.multi_modal_projector(feat)  # (T, llm_hidden_size)
                projected.append(proj)
            return projected
        else:
            vision_features = self.vision_encoder(pixel_values)
            return self.multi_modal_projector(vision_features)

    def _merge_image_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        """Replace <image> placeholder tokens with projected vision features.

        For SigLIP: image_features is (B, T, D) — uses masked_scatter directly.
        For MoonViT: image_features is a list of (T_i, D) tensors — concatenated
                     then scattered into the placeholder positions in order.
        """
        special_image_mask = input_ids == self.image_token_id

        if isinstance(image_features, list):
            # MoonViT: concatenate all per-image features, scatter in sequence order
            flat_features = torch.cat(image_features, dim=0)  # (sum(T_i), D)
        else:
            # SigLIP: flatten (B, T, D) -> (B*T, D)
            flat_features = image_features.reshape(-1, image_features.shape[-1])

        n_image_tokens = special_image_mask.sum().item()
        n_image_features = flat_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image token count ({n_image_tokens}) != image feature count "
                f"({n_image_features}). Ensure the text has the correct number of "
                f"<image> placeholder tokens for this encoder."
            )

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        flat_features = flat_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, flat_features)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> TinyAyaVisionOutput:
        """Forward pass for the VLM.

        For multimodal input: pass both input_ids (with <image> placeholders)
        and pixel_values. The <image> tokens are replaced with projected vision
        features before being passed to the LLM.

        For MoonViT: also pass image_grid_hws from the processor output.
        For text-only input: pass input_ids without pixel_values.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Merge image features into the text embedding sequence
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_hws)
            inputs_embeds = self._merge_image_features(
                input_ids, inputs_embeds, image_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            cache_position=cache_position,
            **kwargs,
        )

        return TinyAyaVisionOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            image_hidden_states=image_features if not isinstance(image_features, list) else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_hws=None,
        attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        """Prepare model inputs for autoregressive generation.

        Pixel values are only passed on the first generation step; after that,
        image features are already merged into the cached key-values.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        # Only pass pixel_values on the first iteration (no cache yet)
        is_first = past_key_values is None or (
            cache_position is not None and cache_position[0] == 0
        )
        if is_first:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_hws is not None:
                model_inputs["image_grid_hws"] = image_grid_hws

        return model_inputs
