from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVisionEncoder(ABC, nn.Module):
    """Abstract base class for vision encoders.

    All encoders must freeze their parameters on init and implement forward().
    """

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor | list[torch.Tensor]:
        """Extract features from preprocessed image tensors.

        Args:
            pixel_values: Preprocessed image tensor(s).
            **kwargs: Encoder-specific extras (e.g. image_grid_hws for MoonViT).

        Returns:
            SigLIP: (B, N, D) tensor of patch embeddings.
            MoonViT: list of (N_tiles_i, tokens_per_tile, D) tensors, one per image.
        """
        ...
