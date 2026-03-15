# Backwards-compatibility shim. Import from src.vision_encoders instead.
from src.vision_encoders import (  # noqa: F401
    BaseVisionEncoder,
    MoonViTVisionEncoder,
    SigLIPVisionEncoder,
    create_vision_encoder,
)

# Legacy alias used by existing code and tests
VisionEncoder = SigLIPVisionEncoder
