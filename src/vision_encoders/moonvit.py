import torch
from transformers import AutoModel

from config.model_config import TinyAyaVisionConfig
from src.vision_encoders.base import BaseVisionEncoder


class MoonViTVisionEncoder(BaseVisionEncoder):
    """Frozen MoonViT native-resolution vision encoder.

    MoonViT (moonshotai/MoonViT-SO-400M) is initialized from SigLIP-SO-400M
    and continually pre-trained for native-resolution image understanding.
    It tiles images at native resolution, producing a variable number of tokens.

    Forward returns a list (one tensor per image) rather than a batched tensor,
    because tile counts differ per image based on resolution.

    Output per image: (N_tiles, tokens_per_tile, vision_hidden_size)
    where N_tiles varies and tokens_per_tile == config.tokens_per_tile (4).
    """

    def __init__(self, config: TinyAyaVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = AutoModel.from_pretrained(
            config.vision_model_name,
            torch_dtype=getattr(torch, config.torch_dtype),
            trust_remote_code=True,
        )
        self.vision_model.requires_grad_(False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Extract tile features from images at native resolution.

        Args:
            pixel_values: Preprocessed image tensor from MoonViT's AutoImageProcessor.
            image_grid_hws: (B, 2) tensor of [H, W] tile-grid dimensions per image.

        Returns:
            List of (N_tiles_i, tokens_per_tile, vision_hidden_size) tensors,
            one per image in the batch.
        """
        with torch.no_grad():
            features = self.vision_model(pixel_values, image_grid_hws)
        return features
