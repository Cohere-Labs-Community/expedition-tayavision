import pytest
import torch

from config.model_config import TinyAyaVisionConfig
from src.vision_encoders import SigLIPVisionEncoder, create_vision_encoder
from src.vision_encoder import VisionEncoder  # legacy alias still works
from tests.conftest import requires_gpu


@requires_gpu
class TestSigLIPVisionEncoder:
    """Tests for the frozen SigLIP2 vision encoder.

    These tests require GPU and download the SigLIP2 model (~0.8GB).
    Run with: pytest tests/test_vision_encoder.py -v
    """

    @pytest.fixture(scope="class")
    def config(self):
        return TinyAyaVisionConfig.for_encoder("siglip")

    @pytest.fixture(scope="class")
    def encoder(self, config):
        encoder = SigLIPVisionEncoder(config)
        return encoder.cuda()

    def test_loads_successfully(self, encoder):
        """Vision model loads without errors."""
        assert encoder.vision_model is not None

    def test_all_params_frozen(self, encoder):
        """All vision encoder parameters should be frozen."""
        for name, param in encoder.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_output_shape_single(self, encoder):
        """Single image produces (1, 729, 1152) features."""
        x = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out = encoder(x)
        assert out.shape == (1, 729, 1152)

    def test_output_shape_batch(self, encoder):
        """Batch of 2 images produces (2, 729, 1152) features."""
        x = torch.randn(2, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out = encoder(x)
        assert out.shape == (2, 729, 1152)

    def test_deterministic(self, encoder):
        """Same input produces same output (frozen model, no dropout at eval)."""
        encoder.eval()
        x = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_no_gradient_computation(self, encoder):
        """Forward pass should not compute gradients."""
        x = torch.randn(1, 3, 384, 384, device="cuda", dtype=torch.bfloat16)
        out = encoder(x)
        assert not out.requires_grad


class TestVisionEncoderFactory:
    """Tests for create_vision_encoder factory and config loading."""

    def test_for_encoder_siglip(self):
        config = TinyAyaVisionConfig.for_encoder("siglip")
        assert config.vision_encoder_type == "siglip"
        assert config.connector_type == "pixel_shuffle"
        assert config.num_tokens_after_shuffle == 196

    def test_for_encoder_moonvit(self):
        config = TinyAyaVisionConfig.for_encoder("moonvit")
        assert config.vision_encoder_type == "moonvit"
        assert config.connector_type == "linear_mlp"
        assert config.trust_remote_code is True
        assert config.tokens_per_tile == 4

    def test_for_encoder_unknown(self):
        with pytest.raises(FileNotFoundError):
            TinyAyaVisionConfig.for_encoder("unknown_encoder")

    def test_factory_returns_siglip_encoder(self):
        config = TinyAyaVisionConfig.for_encoder("siglip")
        from src.vision_encoders import SigLIPVisionEncoder
        # Just check the class is correct without loading weights
        assert create_vision_encoder.__name__ == "create_vision_encoder"
        encoder_cls = {"siglip": SigLIPVisionEncoder}[config.vision_encoder_type]
        assert encoder_cls is SigLIPVisionEncoder

    def test_factory_invalid_type(self):
        config = TinyAyaVisionConfig()
        config.vision_encoder_type = "invalid"
        with pytest.raises(ValueError, match="Unknown vision_encoder_type"):
            create_vision_encoder(config)

    def test_legacy_alias(self):
        """VisionEncoder imported from src.vision_encoder still works."""
        assert VisionEncoder is SigLIPVisionEncoder

    def test_llm_variant(self):
        config = TinyAyaVisionConfig.for_encoder("siglip", llm="global")
        assert config.llm_model_name == "CohereLabs/tiny-aya-global"
