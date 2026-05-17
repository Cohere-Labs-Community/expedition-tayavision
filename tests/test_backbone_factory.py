"""Test the backbone-swap abstraction in TinyAyaVisionConfig."""

from __future__ import annotations

import pytest

from config.model_config import TinyAyaVisionConfig


class TestForBackbone:
    def test_tiny_aya_backbone(self):
        cfg = TinyAyaVisionConfig.for_backbone("tiny_aya", "siglip")
        assert cfg.backbone_type == "tiny_aya"
        assert cfg.llm_hidden_size == 2048
        assert cfg.image_token == "<image>"
        assert cfg.connector_intermediate_size == 2048
        assert "q_proj" in cfg.lora_target_modules
        assert "gate_proj" in cfg.lora_target_modules

    def test_qwen3_backbone(self):
        cfg = TinyAyaVisionConfig.for_backbone("qwen3", "siglip")
        assert cfg.backbone_type == "qwen3"
        assert cfg.llm_hidden_size == 2560
        assert cfg.connector_intermediate_size == 2560
        assert cfg.image_token == "<|image_pad|>"
        assert cfg.llm_model_name.startswith("Qwen/Qwen3-")

    def test_vision_keys_merged(self):
        """Vision YAML fields propagate (siglip → vision_hidden_size=1152)."""
        cfg = TinyAyaVisionConfig.for_backbone("qwen3", "siglip")
        assert cfg.vision_hidden_size == 1152
        assert cfg.vision_grid_size == 27
        assert cfg.num_tokens_after_shuffle == 196

    def test_for_backbone_unknown_raises(self):
        with pytest.raises(FileNotFoundError):
            TinyAyaVisionConfig.for_backbone("does-not-exist", "siglip")

    def test_for_base_unchanged(self):
        """Tiny Aya legacy factories still produce a tiny_aya backbone config."""
        cfg = TinyAyaVisionConfig.for_base()
        assert cfg.backbone_type == "tiny_aya"
        assert cfg.llm_hidden_size == 2048
        assert cfg.image_token == "<image>"


class TestConnectorWithBackbones:
    @pytest.mark.parametrize(
        "backbone,expected_hidden",
        [("tiny_aya", 2048), ("qwen3", 2560)],
    )
    def test_connector_output_dim(self, backbone, expected_hidden):
        """MultiModalProjector picks up llm_hidden_size from the backbone config."""
        import torch

        from src.connector import MultiModalProjector

        cfg = TinyAyaVisionConfig.for_backbone(backbone, "siglip")
        connector = MultiModalProjector(cfg)
        x = torch.randn(1, 729, cfg.vision_hidden_size)
        out = connector(x)
        assert out.shape == (1, cfg.num_tokens_after_shuffle, expected_hidden)
