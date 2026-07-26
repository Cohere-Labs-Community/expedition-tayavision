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


_skip_without_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def requires_gpu(obj):
    """Label a test GPU-only *and* skip it when there is no CUDA device.

    Both properties are wanted: the label makes `pytest -m "not requires_gpu"`
    able to deselect these before collection cost is paid, and the skip keeps a
    CPU-only machine green. `requires_gpu` is registered in pyproject.toml.

    This has to be a function. The obvious spelling,

        requires_gpu = pytest.mark.requires_gpu(pytest.mark.skipif(...))

    does NOT compose the two marks. Calling a MarkDecorator with a non-callable
    argument stores that argument as a *parameter of the mark*, so the result is
    a bare `requires_gpu` mark carrying the skipif in `mark.args`, and the skip
    never happens. The tests then run everywhere and error in the fixture on the
    first `.cuda()`.

    That is invisible on a CUDA box -- which is why it shipped and only CI, on a
    CPU-only runner, caught it: `RuntimeError: Found no NVIDIA driver`, six
    errors in test_vision_encoder.py.
    """
    return pytest.mark.requires_gpu(_skip_without_cuda(obj))
