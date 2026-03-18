import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from config.model_config import TinyAyaVisionConfig


class TinyAyaVisionProcessor:
    """Combined processor for Tiny Aya Vision multimodal inputs.

    Handles both image preprocessing and text tokenization, inserting the
    correct number of <image> placeholder tokens per image.

    SigLIP: fixed 196 tokens per image (after pixel shuffle).
    MoonViT: variable tokens per image = N_tiles * tokens_per_tile (4),
             where N_tiles depends on image resolution. The image processor
             is run first to determine N_tiles via image_grid_hws.
    """

    def __init__(self, config: TinyAyaVisionConfig):
        self.config = config
        self.image_processor = AutoImageProcessor.from_pretrained(
            config.vision_model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)

        # Add the <image> special token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.image_token]}
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(config.image_token)

    @property
    def image_placeholder(self) -> str:
        """The string of <image> tokens to insert per image (SigLIP only).

        For SigLIP: fixed num_tokens_after_shuffle (196) tokens.
        For MoonViT: token count is dynamic; use _tokens_per_image() instead.
        """
        return self.config.image_token * self.config.num_tokens_after_shuffle

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
