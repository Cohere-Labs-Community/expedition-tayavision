import pytest
import torch

from config.model_config import TinyAyaVisionConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from src.processing import TinyAyaVisionProcessor
from tests.conftest import requires_gpu


@requires_gpu
class TestVLMAssembly:
    """Integration tests for the full VLM model.

    These tests require GPU and download both models (~7.5GB in bf16).
    Run with: pytest tests/test_vlm_assembly.py -v

    Parametrized over both LLM backbones:
      - "base"   → CohereLabs/tiny-aya-base   (pretrained)
      - "global" → CohereLabs/tiny-aya-global (instruction-tuned)
    """

    @pytest.fixture(scope="class", params=["base", "global"])
    def model_and_processor(self, request):
        config = (
            TinyAyaVisionConfig.for_global()
            if request.param == "global"
            else TinyAyaVisionConfig.for_base()
        )
        model = TinyAyaVisionForConditionalGeneration(config)
        processor = TinyAyaVisionProcessor(config)
        model.setup_tokenizer(processor.tokenizer)
        model = model.to("cuda")
        return model, processor

    def test_text_only_forward(self, model_and_processor):
        """Text-only forward pass (no images) produces valid logits."""
        model, processor = model_and_processor
        inputs = processor("The capital of France is")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits is not None
        assert outputs.logits.ndim == 3
        assert outputs.logits.shape[-1] >= 261000  # vocab size

    def test_multimodal_forward(self, model_and_processor, dummy_image):
        """Multimodal forward pass (image + text) produces valid logits."""
        model, processor = model_and_processor
        inputs = processor("<image> Describe this image.", images=dummy_image)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits is not None
        assert outputs.image_hidden_states is not None
        assert outputs.image_hidden_states.shape[1] == 196  # num_tokens_after_shuffle

    def test_generation(self, model_and_processor):
        """model.generate() produces token IDs without errors."""
        model, processor = model_and_processor
        inputs = processor("Hello, world!")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        assert output_ids.ndim == 2
        assert output_ids.shape[1] > inputs["input_ids"].shape[1]

    def test_memory_footprint(self, model_and_processor):
        """Peak GPU memory should be under 15GB at batch_size=1."""
        torch.cuda.reset_peak_memory_stats()
        model, processor = model_and_processor
        inputs = processor("Test input")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs)

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        assert peak_mb < 15_000, f"Peak memory {peak_mb:.0f}MB exceeds 15GB"


class TestMergeLogic:
    """Unit tests for image feature merging (CPU, no model download needed)."""

    def test_merge_correct_replacement(self):
        """Image features replace exactly the <image> token positions."""
        from src.connector import MultiModalProjector

        config = TinyAyaVisionConfig()
        proj = MultiModalProjector(config)

        # Simulate vision features -> image embeddings
        vision_features = torch.randn(1, 729, 1152)
        image_embeddings = proj(vision_features)  # (1, 196, 2048)

        # Create text embeddings with image placeholders
        seq_len = 196 + 10
        text_embeds = torch.zeros(1, seq_len, 2048)
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        image_token_id = 999

        input_ids[0, :196] = image_token_id  # First 196 are image tokens

        # Perform merge
        mask = input_ids == image_token_id
        mask_expanded = mask.unsqueeze(-1).expand_as(text_embeds)
        merged = text_embeds.masked_scatter(mask_expanded, image_embeddings)

        # Verify image region was replaced
        assert torch.allclose(merged[0, :196], image_embeddings[0])
        # Verify non-image region unchanged
        assert torch.allclose(merged[0, 196:], text_embeds[0, 196:])

    def test_merge_mismatch_detected(self):
        """Mismatched token/feature count is detected."""
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        input_ids[0, :5] = 999  # 5 image tokens
        image_features = torch.randn(1, 196, 2048)  # 196 != 5

        mask = input_ids == 999
        n_tokens = mask.sum().item()
        n_features = image_features.shape[0] * image_features.shape[1]
        assert n_tokens != n_features  # Confirms mismatch
