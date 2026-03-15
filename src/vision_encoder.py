import torch
import torch.nn as nn
from transformers import SiglipVisionModel

from config.model_config import TinyAyaVisionConfig


class VisionEncoder(nn.Module):
    """Frozen SigLIP2 vision encoder that extracts patch features from images.

    Wraps SiglipVisionModel to produce (B, num_patches, hidden_size) features.
    All parameters are frozen on initialization.
    """

    def __init__(self, config: TinyAyaVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel.from_pretrained(
            config.vision_model_name,
            torch_dtype=getattr(torch, config.torch_dtype),
            cache_dir=config.cache_dir,
        )
        self.vision_model.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch features from images.

        Args:
            pixel_values: (B, C, H, W) preprocessed image tensor.

        Returns:
            (B, num_patches, vision_hidden_size) patch embeddings.
            With default config: (B, 729, 1152).
        """
        with torch.no_grad():
            outputs = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        if isinstance(self.config.vision_feature_layer, int):
            features = outputs.hidden_states[self.config.vision_feature_layer]
        else:
            features = outputs.last_hidden_state

        # SigLIP has no CLS token, so "full" strategy keeps all patches.
        # "default" strategy would crop CLS (index 0) if present.
        if self.config.vision_feature_select_strategy == "default":
            features = features[:, 1:]

        return features
