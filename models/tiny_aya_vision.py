from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, GenerationMixin, PreTrainedModel
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


class TinyAyaVisionForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Tiny Aya Vision: multilingual VLM connecting a vision encoder to Tiny Aya Base.

    Architecture:
        VisionEncoder (frozen) -> Connector -> Cohere2 LLM

    Supports swappable vision encoders (SigLIP, MoonViT) via TinyAyaVisionConfig.
    The model replaces <image> placeholder tokens in the text sequence with
    projected vision features, then runs the combined sequence through the LLM.
    """

    config_class = TinyAyaVisionConfig
    main_input_name = "input_ids"
    _supports_flash_attn_2 = False
    _no_split_modules = ["SigLIPVisionEncoder", "MoonViTVisionEncoder"]
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config: TinyAyaVisionConfig):
        super().__init__(config)

        self.vision_encoder = create_vision_encoder(config)
        if config.vision_tower_config is None:
            config.vision_tower_config = self.vision_encoder.vision_model.config.to_dict()

        self.multi_modal_projector = create_projector(config).to(config.torch_dtype)

        if config.text_config is not None:
            from transformers import CONFIG_MAPPING
            text_config_cls = CONFIG_MAPPING[config.text_config["model_type"]]
            text_cfg = text_config_cls.from_dict(config.text_config)
            self.language_model = AutoModelForCausalLM.from_config(text_cfg)
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                torch_dtype=config.torch_dtype,
                cache_dir=config.cache_dir,
            )
            config.text_config = self.language_model.config.to_dict()
            config._text_config_obj = None  # invalidate cached config

        self.generation_config = self.language_model.generation_config
        self._image_token_id: int | None = config.image_token_id

        self.post_init()

    def _init_weights(self, module):
        pass

    @property
    def image_token_id(self) -> int:
        if self._image_token_id is None:
            raise ValueError(
                "image_token_id not set. Call setup_tokenizer() first."
            )
        return self._image_token_id

    def setup_tokenizer(self, tokenizer) -> None:
        """Add <image> special token to tokenizer and resize embeddings.

        Must be called after initialization and before forward/generate.
        """
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.config.image_token]}
        )
        self._image_token_id = tokenizer.convert_tokens_to_ids(self.config.image_token)
        self.config.image_token_id = self._image_token_id

        if num_added > 0:
            self.language_model.resize_token_embeddings(len(tokenizer))
            self.config.text_config = self.language_model.config.to_dict()
            self.config._text_config_obj = None  # invalidate cached config

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

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
        if self.config.vision_encoder_type == "moonvit":
            raw_features = self.vision_encoder(pixel_values, image_grid_hws=image_grid_hws)
            projected = []
            for feat in raw_features:
                feat = feat.view(-1, feat.shape[-1])
                proj = self.multi_modal_projector(feat)
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
            flat_features = torch.cat(image_features, dim=0)
        else:
            flat_features = image_features.reshape(-1, image_features.shape[-1])

        n_image_tokens = special_image_mask.sum()
        n_image_features = flat_features.shape[0]
        if torch.compiler.is_compiling():
            # Inside torch.compile — skip the .item() validation to avoid
            # graph breaks.  The masked_scatter below will raise if sizes
            # are actually mismatched.
            pass
        elif n_image_tokens.item() != n_image_features:
            raise ValueError(
                f"Image token count ({n_image_tokens.item()}) != image feature count "
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
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None and (input_ids == self.image_token_id).any():
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
            image_hidden_states=torch.cat(image_features, dim=0) if isinstance(image_features, list) else image_features,
        )

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        # Delegate to the language model so it uses the correct cache type (uses hybrid cache)
        return self.language_model._prepare_cache_for_generation(
            generation_config, model_kwargs, *args, **kwargs
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
        is_first_iteration=False,
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
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Merge image features into embeddings before the LM trims input_ids
        # forward() can't locate <image> positions after trimming so its done here
        
        # cache_position[0] == 0 identifies the first generation step.
        is_first_step = cache_position is None or int(cache_position[0]) == 0
        if (
            is_first_step
            and pixel_values is not None
            and input_ids is not None
            and (input_ids == self.image_token_id).any()
        ):
            word_embeds = self.get_input_embeddings()(input_ids)
            image_features = self.get_image_features(pixel_values, image_grid_hws)
            merged_embeds = self._merge_image_features(input_ids, word_embeds, image_features)
            model_inputs["inputs_embeds"] = merged_embeds
            model_inputs.pop("input_ids", None)
            # pixel_values already merged

        return model_inputs
