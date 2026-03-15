from config.model_config import TinyAyaVisionConfig
from src.vision_encoders.base import BaseVisionEncoder
from src.vision_encoders.moonvit import MoonViTVisionEncoder
from src.vision_encoders.siglip import SigLIPVisionEncoder


def create_vision_encoder(config: TinyAyaVisionConfig) -> BaseVisionEncoder:
    """Factory: instantiate the correct vision encoder for the given config.

    Args:
        config: Model config with vision_encoder_type set to "siglip" or "moonvit".

    Returns:
        A frozen BaseVisionEncoder subclass instance.
    """
    encoders = {
        "siglip": SigLIPVisionEncoder,
        "moonvit": MoonViTVisionEncoder,
    }
    if config.vision_encoder_type not in encoders:
        raise ValueError(
            f"Unknown vision_encoder_type '{config.vision_encoder_type}'. "
            f"Choose from: {list(encoders)}"
        )
    return encoders[config.vision_encoder_type](config)


__all__ = [
    "BaseVisionEncoder",
    "SigLIPVisionEncoder",
    "MoonViTVisionEncoder",
    "create_vision_encoder",
]
