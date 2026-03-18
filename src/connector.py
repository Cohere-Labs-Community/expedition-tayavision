import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import TinyAyaVisionConfig


class MultiModalProjector(nn.Module):
    """Vision-language connector using Pixel Shuffle + SwiGLU MLP.

    Adapted from Aya Vision's AyaVisionMultiModalProjector. Compresses vision
    tokens via Pixel Shuffle (4x reduction) then projects to the LLM's
    embedding space with a 2-layer MLP using SwiGLU activation.

    Data flow:
        (B, 729, 1152)  vision features
        -> pad 27x27 to 28x28
        (B, 784, 1152)
        -> pixel shuffle (2x2 block concat)
        (B, 196, 4608)
        -> LayerNorm
        -> Linear(4608, 2048)
        -> SwiGLU (chunk + SiLU gate)
        (B, 196, 1024)
        -> Linear(1024, 2048)
        (B, 196, 2048)  image embeddings in LLM space
    """

    def __init__(self, config: TinyAyaVisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor

        ps_dim = config.vision_hidden_size * (config.downsample_factor ** 2)

        self.layernorm = nn.LayerNorm(
            ps_dim, eps=config.adapter_layer_norm_eps
        )

        # First linear projects to connector_intermediate_size
        # Output is then split in half for SwiGLU gating
        self.linear_1 = nn.Linear(
            ps_dim, config.connector_intermediate_size, bias=True
        )

        self.act = nn.SiLU()

        # Second linear: gated half -> LLM hidden size
        self.linear_2 = nn.Linear(
            config.connector_intermediate_size // 2,
            config.llm_hidden_size,
            bias=True,
        )

        if config.post_projector_rms_norm:
            self.rms_norm = nn.RMSNorm(config.llm_hidden_size)
        else:
            self.rms_norm = None

    def pixel_shuffle(self, image_features: torch.Tensor) -> torch.Tensor:
        """Compress vision tokens by grouping 2x2 spatial patches.

        Pads the grid to even dimensions if needed, then concatenates
        2x2 block embeddings along the feature dimension, reducing
        token count by 4x.

        Args:
            image_features: (B, S, D) patch embeddings.

        Returns:
            (B, S/4, D*4) compressed features.
        """
        batch_size, seq_length, feature_dim = image_features.shape
        height = width = int(seq_length**0.5)

        # Reshape to spatial grid: (B, H, W, D)
        image_features = image_features.reshape(
            batch_size, height, width, feature_dim
        )

        # Pad to even dimensions if grid is odd
        dsf = self.downsample_factor
        pad_h = (dsf - height % dsf) % dsf
        pad_w = (dsf - width % dsf) % dsf
        if pad_h > 0 or pad_w > 0:
            image_features = F.pad(
                image_features, (0, 0, 0, pad_w, 0, pad_h)
            )

        height = height + pad_h
        width = width + pad_w
        channels = image_features.shape[-1]

        # Pixel shuffle: merge 2x2 blocks into channel dimension
        # Following Aya Vision's reshape-permute pattern
        image_features = image_features.reshape(
            batch_size,
            height,
            int(width / dsf),
            int(channels * dsf),
        )
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size,
            int(width / dsf),
            int(height / dsf),
            -1,
        )
        image_features = image_features.permute(0, 2, 1, 3)

        # Flatten spatial dims: (B, H', W', D') -> (B, H'*W', D')
        image_features = image_features.reshape(
            batch_size, -1, image_features.shape[-1]
        )

        return image_features

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project vision features into the LLM embedding space.

        Args:
            image_features: (B, num_patches, vision_hidden_size).

        Returns:
            (B, num_tokens_after_shuffle, llm_hidden_size).
        """
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        # SwiGLU: split into value and gate, apply SiLU to gate
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        if self.rms_norm is not None:
            hidden_states = self.rms_norm(hidden_states)
        return hidden_states


class LinearMLPProjector(nn.Module):
    """Vision-language connector using a SwiGLU MLP (no Pixel Shuffle).

    Used with MoonViT, which handles tiling and spatial compression internally.
    Projects each token from vision_hidden_size to llm_hidden_size without
    changing the token count.

    Data flow:
        (..., vision_hidden_size)
        -> LayerNorm
        -> Linear(vision_hidden_size, connector_intermediate_size)
        -> SwiGLU (chunk + SiLU gate)
        (..., connector_intermediate_size // 2)
        -> Linear(connector_intermediate_size // 2, llm_hidden_size)
        (..., llm_hidden_size)
    """

    def __init__(self, config: TinyAyaVisionConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(
            config.vision_hidden_size, eps=config.adapter_layer_norm_eps
        )
        self.linear_1 = nn.Linear(
            config.vision_hidden_size, config.connector_intermediate_size, bias=True
        )
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            config.connector_intermediate_size // 2,
            config.llm_hidden_size,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project vision features into the LLM embedding space.

        Args:
            image_features: (..., vision_hidden_size) — any leading dims.

        Returns:
            (..., llm_hidden_size).
        """
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        return self.linear_2(hidden_states)


def create_projector(config: TinyAyaVisionConfig) -> nn.Module:
    """Factory: instantiate the correct connector for the given config.

    Args:
        config: Model config with connector_type "pixel_shuffle" or "linear_mlp".

    Returns:
        An nn.Module projector.
    """
    projectors = {
        "pixel_shuffle": MultiModalProjector,
        "linear_mlp": LinearMLPProjector,
    }
    if config.connector_type not in projectors:
        raise ValueError(
            f"Unknown connector_type '{config.connector_type}'. "
            f"Choose from: {list(projectors)}"
        )
    return projectors[config.connector_type](config)
