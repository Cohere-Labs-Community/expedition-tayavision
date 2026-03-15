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


class TinyAyaVisionProcessor:
    """Combined processor for Tiny Aya Vision multimodal inputs.

    Handles both image preprocessing (via SiglipImageProcessor) and text
    tokenization (via CohereTokenizer), inserting the correct number of
    <image> placeholder tokens per image.

    The chat template is patched at init to support structured multimodal
    messages (list-of-dicts with ``type: "image"`` / ``type: "text"``),
    enabling a standard instruction-finetuning workflow via
    :meth:`apply_chat_template`.
    """

    def __init__(self, config: TinyAyaVisionConfig):
        self.config = config
        self.image_processor = AutoImageProcessor.from_pretrained(
            config.vision_model_name,
            cache_dir=config.cache_dir,
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

    @property
    def image_placeholder(self) -> str:
        """The string of <image> tokens to insert per image."""
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

    def __call__(
        self,
        text: str | list[str],
        images: Image.Image | list[Image.Image] | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Process text and optional images into model inputs.

        The text should contain `<image>` markers where images should be
        inserted. Each `<image>` marker is expanded to num_tokens_after_shuffle
        (196) placeholder tokens.

        Args:
            text: Input text or list of texts. Use "<image>" as image placeholder.
            images: Optional PIL Image(s) corresponding to <image> markers.
            padding: Padding strategy.
            truncation: Whether to truncate.
            max_length: Maximum sequence length.
            return_tensors: Output tensor format.

        Returns:
            Dict with "input_ids", "attention_mask", and optionally "pixel_values".
        """
        if isinstance(text, str):
            text = [text]

        # Expand each single <image> marker into the full placeholder sequence
        expanded_text = []
        for t in text:
            expanded = t.replace(self.config.image_token, self.image_placeholder)
            expanded_text.append(expanded)

        # Tokenize
        text_inputs = self.tokenizer(
            expanded_text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        result = dict(text_inputs)

        # Process images
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            image_inputs = self.image_processor(
                images=images, return_tensors=return_tensors
            )
            result["pixel_values"] = image_inputs["pixel_values"]

        return result
