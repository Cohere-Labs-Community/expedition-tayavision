import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

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


class TinyAyaVisionProcessor:
    """Combined processor for Tiny Aya Vision multimodal inputs.

    Handles both image preprocessing (via SiglipImageProcessor) and text
    tokenization (via CohereTokenizer), inserting the correct number of
    <image> placeholder tokens per image.

    The chat template is patched at init to support structured multimodal
    messages (list-of-dicts with ``type: "image"`` / ``type: "text"``),
    enabling a standard instruction-finetuning workflow via
    :meth:`apply_chat_template`.
    Handles both image preprocessing and text tokenization, inserting the
    correct number of <image> placeholder tokens per image.

    SigLIP: fixed 196 tokens per image (after pixel shuffle).
    MoonViT: variable tokens per image = N_tiles * tokens_per_tile (4),
             where N_tiles depends on image resolution. The image processor
             is run first to determine N_tiles via image_grid_hws.
    """

    def __init__(self, config: TinyAyaVisionConfig):
        self.config = config
        image_processor_kwargs = {}
        if config.vision_encoder_type == "moonvit":
            image_processor_kwargs["in_token_limit"] = config.in_token_limit
        self.image_processor = AutoImageProcessor.from_pretrained(
            config.vision_model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
            **image_processor_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, cache_dir=config.cache_dir)

        # Add the <image> special token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.image_token]}
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(config.image_token)

        if "base" not in config.llm_model_name:
            # Patch the chat template so it can render multimodal content
            self._patch_chat_template()

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
            raise ValueError(
                f"Tokenizer for {self.config.llm_model_name!r} has no chat "
                "template. Use an instruction-tuned model like "
                "'CohereLabs/tiny-aya-global' via TinyAyaVisionConfig.for_global()."
            )

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

        if "base" not in config.llm_model_name:
            # Patch the chat template so it can render multimodal content
            self._patch_chat_template()

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
            raise ValueError(
                f"Tokenizer for {self.config.llm_model_name!r} has no chat "
                "template. Use an instruction-tuned model like "
                "'CohereLabs/tiny-aya-global' via TinyAyaVisionConfig.for_global()."
            )

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

    @property
    def image_placeholder(self) -> str:
        """The string of <image> tokens to insert per image (SigLIP only).

        For SigLIP: fixed num_tokens_after_shuffle (196) tokens.
        For MoonViT: token count is dynamic; use _tokens_per_image() instead.
        """
        return self.config.image_token * self.config.num_tokens_after_shuffle

    def apply_chat_template(
        self,
        messages: list[dict],
        images: "Image.Image | list[Image.Image] | None" = None,
        padding: "bool | str" = False,
        truncation: bool = False,
        max_length: "int | None" = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: str = "pt",
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
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
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
        if self.config.vision_encoder_type == "moonvit":
            if image_grid_hws is None:
                raise ValueError(
                    "image_grid_hws is required for MoonViT to determine token counts. "
                    "Run the image processor first and pass image_grid_hws here."
                )
            # image_grid_hws: (B, 2) — [H, W] grid per image; H * W = total visual tokens
            return (image_grid_hws[:, 0] * image_grid_hws[:, 1]).tolist()
        else:
            return [self.config.num_tokens_after_shuffle] * n_images

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
            placeholder = self.config.image_token * n
            expanded_text.append(t.replace(self.config.image_token, placeholder))

        # Tokenize
        text_inputs = self.tokenizer(
            expanded_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        result.update(text_inputs)

        return result
