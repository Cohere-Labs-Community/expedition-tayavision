from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from config.model_config import TinyAyaVisionConfig

# Jinja2 snippet that replaces {{ message['content'] }} to handle both
# plain-string content and aya-vision-style list-of-dicts content.
# Images are rendered first, then text — matching aya-vision's ordering.
_MULTIMODAL_CONTENT_RENDER = (
    "{%- if message['content'] is string -%}"
    "{{ message['content'] }}"
    "{%- else -%}"
    "{%- for item in message['content'] | selectattr('type', 'equalto', 'image') -%}"
    "<image>"
    "{%- endfor -%}"
    "{%- for item in message['content'] | selectattr('type', 'equalto', 'text') -%}"
    "{{ item['text'] }}"
    "{%- endfor -%}"
    "{%- endif -%}"
)


class TinyAyaVisionProcessor(ProcessorMixin):
    """Combined processor for Tiny Aya Vision multimodal inputs.

    Handles both image preprocessing (via SiglipImageProcessor) and text
    tokenization (via CohereTokenizer), inserting the correct number of
    <image> placeholder tokens per image.

    The chat template is patched at init to support structured multimodal
    messages (list-of-dicts with ``type: "image"`` / ``type: "text"``),
    enabling a standard instruction-finetuning workflow via
    :meth:`apply_chat_template`.

    SigLIP: fixed 196 tokens per image (after pixel shuffle).
    MoonViT: variable tokens per image = N_tiles * tokens_per_tile (4),
             where N_tiles depends on image resolution. The image processor
             is run first to determine N_tiles via image_grid_hws.

    Usage (from config — for training)::

        config = TinyAyaVisionConfig.for_encoder("siglip", llm="global")
        processor = TinyAyaVisionProcessor(config=config)

    Usage (from pretrained — after uploading to HF)::

        processor = AutoProcessor.from_pretrained("your-hf-repo", trust_remote_code=True)
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    processor_type = "TinyAyaVisionProcessor"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        # Config-based construction (backward compat with training code)
        config: TinyAyaVisionConfig | None = None,
        # Extra kwargs stored in processor_config.json for from_pretrained
        vision_encoder_type: str = "siglip",
        image_token: str = "<image>",
        num_tokens_after_shuffle: int = 196,
        tokens_per_tile: int = 4,
        in_token_limit: int = 1024,
        patch_chat_template: bool = True,
        **kwargs,
    ):
        # Support positional: TinyAyaVisionProcessor(config) for backward compat
        if isinstance(image_processor, TinyAyaVisionConfig):
            config = image_processor
            image_processor = None

        # Config-based construction: build sub-processors from config
        if config is not None:
            image_processor_kwargs = {}
            if config.vision_encoder_type == "moonvit":
                image_processor_kwargs["in_token_limit"] = config.in_token_limit
            image_processor = AutoImageProcessor.from_pretrained(
                config.vision_model_name,
                cache_dir=config.cache_dir,
                trust_remote_code=config.trust_remote_code,
                **image_processor_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                config.llm_model_name, cache_dir=config.cache_dir,
            )
            vision_encoder_type = config.vision_encoder_type
            image_token = config.image_token
            num_tokens_after_shuffle = config.num_tokens_after_shuffle
            tokens_per_tile = config.tokens_per_tile
            in_token_limit = config.in_token_limit
            patch_chat_template = "base" not in config.llm_model_name

        super().__init__(image_processor=image_processor, tokenizer=tokenizer)

        self.vision_encoder_type = vision_encoder_type
        self.image_token = image_token
        self.num_tokens_after_shuffle = num_tokens_after_shuffle
        self.tokens_per_tile = tokens_per_tile
        self.in_token_limit = in_token_limit
        self.patch_chat_template = patch_chat_template

        # Add the <image> special token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_token]}
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        if self.patch_chat_template:
            self._patch_chat_template()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """Serialize only the extra processor kwargs (not sub-processors)."""
        output = {
            "vision_encoder_type": self.vision_encoder_type,
            "image_token": self.image_token,
            "num_tokens_after_shuffle": self.num_tokens_after_shuffle,
            "tokens_per_tile": self.tokens_per_tile,
            "in_token_limit": self.in_token_limit,
            "patch_chat_template": self.patch_chat_template,
            "processor_class": self.__class__.__name__,
        }
        if hasattr(self, "image_processor") and self.image_processor is not None:
            output["image_processor"] = self.image_processor.to_dict()
        return output

    # ------------------------------------------------------------------
    # Chat-template patching
    # ------------------------------------------------------------------

    def _patch_chat_template(self) -> None:
        """Replace ``{{ message['content'] }}`` in the tokenizer's chat
        template with a multimodal-aware renderer so that structured
        messages (list-of-dicts with type: image / text) are handled
        exactly the way aya-vision does it.
        """
        template = self.tokenizer.chat_template
        if isinstance(template, dict):
            template = template.get("default", "")
        if template is None:
            return

        patched = template.replace(
            "{{ message['content'] }}", _MULTIMODAL_CONTENT_RENDER
        )
        self.tokenizer.chat_template = patched

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def chat_template(self) -> str:
        """Return the (patched) chat template string."""
        return self.tokenizer.chat_template

    @chat_template.setter
    def chat_template(self, value):
        if value is not None:
            self.tokenizer.chat_template = value

    @property
    def image_placeholder(self) -> str:
        """The string of <image> tokens to insert per image (SigLIP only).

        For SigLIP: fixed num_tokens_after_shuffle (196) tokens.
        For MoonViT: token count is dynamic; use _tokens_per_image() instead.
        """
        return self.image_token * self.num_tokens_after_shuffle

    def apply_chat_template(
        self,
        messages: list[dict],
        images: "Image.Image | list[Image.Image] | None" = None,
        padding: "bool | str" = False,
        truncation: bool = False,
        max_length: "int | None" = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        return_tensors: str = "pt",
        **kwargs,
    ) -> "dict[str, torch.Tensor] | str":
        """Format structured chat messages into model inputs.

        Supports aya-vision-style multimodal messages::

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ]},
                {"role": "assistant", "content": "The image shows …"},
            ]

        Args:
            messages: Chat messages. ``content`` may be a plain string
                **or** a list of ``{"type": "image"}`` / ``{"type": "text",
                "text": "..."}`` dicts.
            images: PIL Image(s) matching the ``{"type": "image"}`` entries
                in *messages*, in order.
            padding: Padding strategy forwarded to the tokenizer.
            truncation: Whether to truncate.
            max_length: Maximum sequence length.
            add_generation_prompt: Append the assistant-turn prefix.
            tokenize: If ``False`` return the formatted string (with
                ``<image>`` markers) instead of token ids.
            return_tensors: Tensor format (default ``"pt"``).

        Returns:
            If *tokenize* is ``True``: dict with ``input_ids``,
            ``attention_mask``, and optionally ``pixel_values``.
            If *tokenize* is ``False``: the formatted text string.
        """
        # continue_final_message and add_generation_prompt are mutually exclusive
        if kwargs.get("continue_final_message"):
            add_generation_prompt = False
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if not tokenize:
            return text

        return self(
            text=text,
            images=images,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    def _tokens_per_image(self, image_grid_hws: torch.Tensor | None, n_images: int) -> list[int]:
        """Compute how many <image> tokens each image expands to.

        For SigLIP: always config.num_tokens_after_shuffle (196).
        For MoonViT: H * W per image, where H and W come from image_grid_hws
                     returned by the MoonViT image processor (already accounts
                     for internal tiling/compression).
        """
        if self.vision_encoder_type == "moonvit":
            if image_grid_hws is None:
                raise ValueError(
                    "image_grid_hws is required for MoonViT to determine token counts. "
                    "Run the image processor first and pass image_grid_hws here."
                )
            # image_grid_hws: (B, 2) — [H, W] grid per image; H * W = total visual tokens
            return (image_grid_hws[:, 0] * image_grid_hws[:, 1]).tolist()
        else:
            return [self.num_tokens_after_shuffle] * n_images

    def __call__(
        self,
        text: str | list[str],
        images: Image.Image | list[Image.Image] | None = None,
        image_grid_hws: torch.Tensor | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process text and optional images into model inputs.

        The text should contain `<image>` markers where images should be
        inserted. Each `<image>` marker is expanded to the appropriate number
        of placeholder tokens (fixed for SigLIP, dynamic for MoonViT).

        Args:
            text: Input text or list of texts. Use "<image>" as image placeholder.
            images: Optional PIL Image(s) corresponding to <image> markers.
            padding: Padding strategy.
            truncation: Whether to truncate.
            max_length: Maximum sequence length.
            return_tensors: Output tensor format.

        Returns:
            Dict with "input_ids", "attention_mask", and optionally
            "pixel_values" (and "image_grid_hws" for MoonViT).
        """
        if isinstance(text, str):
            text = [text]

        result: dict[str, torch.Tensor] = {}

        # Process images first to learn token counts (needed for MoonViT)
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            # lm-eval passes images as a list-of-lists [[img], [img], ...] -> flatten to [img, img, ...]
            elif images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
            result["pixel_values"] = image_inputs["pixel_values"]
            if "image_grid_hws" in image_inputs:
                image_grid_hws = image_inputs["image_grid_hws"]
                result["image_grid_hws"] = image_grid_hws

        # Determine per-image token counts
        n_images = len(images) if images is not None else len(text)
        tokens_per_img = self._tokens_per_image(image_grid_hws, n_images)

        # Expand each single <image> marker into the full placeholder sequence.
        # Assumes one <image> marker per text element, matching one image.
        expanded_text = []
        for i, t in enumerate(text):
            n = tokens_per_img[i] if i < len(tokens_per_img) else tokens_per_img[0]
            placeholder = self.image_token * n
            expanded_text.append(t.replace(self.image_token, placeholder))

        # Tokenize
        text_inputs = self.tokenizer(
            expanded_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        result.update(text_inputs)

        return BatchFeature(result)
