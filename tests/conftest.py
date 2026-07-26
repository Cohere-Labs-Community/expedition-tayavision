import pytest
import torch
from PIL import Image

from config.model_config import TinyAyaVisionConfig


@pytest.fixture
def config():
    return TinyAyaVisionConfig()


@pytest.fixture
def dummy_image():
    """Random 384x384 RGB PIL image."""
    return Image.fromarray(
        torch.randint(0, 256, (384, 384, 3), dtype=torch.uint8).numpy()
    )


@pytest.fixture
def dummy_image_batch(dummy_image):
    """Batch of 2 dummy images."""
    img2 = Image.fromarray(
        torch.randint(0, 256, (384, 384, 3), dtype=torch.uint8).numpy()
    )
    return [dummy_image, img2]


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Composed so the tests are both auto-skipped without CUDA *and* deselectable
# by name: `pytest -m "not requires_gpu"`. A bare skipif carries no marker, so
# the old version could not be deselected -- you paid full collection cost
# either way. `requires_gpu` is registered in pyproject.toml.
requires_gpu = pytest.mark.requires_gpu(
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
)
