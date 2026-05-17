"""Tests for the Script-Conditioned Token Allocation Controller (SCTAC)."""

from __future__ import annotations

import json

import pytest
import torch

from src.script_controller import (
    FertilityTable,
    ScriptController,
    ScriptControllerConfig,
)


@pytest.fixture
def fertility_table(tmp_path):
    """Tiny fertility table with two scripts."""
    path = tmp_path / "fertility.json"
    payload = {
        "backbone": "test",
        "per_language": {
            "eng_Latn": {"script": "Latn", "tpw": 1.3, "tpc": 0.27},
            "hin_Deva": {"script": "Deva", "tpw": 2.5, "tpc": 0.6},
        },
        "per_script": {"Latn": 1.3, "Deva": 2.5},
    }
    path.write_text(json.dumps(payload))
    return FertilityTable.load(path)


@pytest.fixture
def cfg_b1():
    return ScriptControllerConfig(enabled=True, variant="B1")


@pytest.fixture
def cfg_b2():
    return ScriptControllerConfig(enabled=True, variant="B2")


class TestFertilityTable:
    def test_empty_table_falls_back_to_unit_fertility(self):
        table = FertilityTable.empty()
        assert table.lookup_tpw("eng_Latn") == 1.0
        assert table.lookup_script_index("eng_Latn") == -1

    def test_lookup_known_language(self, fertility_table):
        assert fertility_table.lookup_tpw("eng_Latn") == pytest.approx(1.3)
        assert fertility_table.lookup_tpw("hin_Deva") == pytest.approx(2.5)

    def test_lookup_unknown_language_uses_script(self, fertility_table):
        # "fra_Latn" not in per_language but its script is.
        assert fertility_table.lookup_tpw("fra_Latn") == pytest.approx(1.3)

    def test_script_index_consistent_with_scripts_list(self, fertility_table):
        idx = fertility_table.lookup_script_index("eng_Latn")
        assert 0 <= idx < len(fertility_table.scripts)
        assert fertility_table.scripts[idx] == "Latn"


class TestScriptController:
    def test_forward_b1(self, cfg_b1, fertility_table):
        controller = ScriptController(
            vision_hidden_size=1152, cfg=cfg_b1, fertility_table=fertility_table
        )
        patch = torch.randn(2, 729, 1152)
        post_shuffle = torch.randn(2, 196, 4608)
        out, mask, aux = controller(patch, post_shuffle, ["eng_Latn", "hin_Deva"])
        assert out.shape == (2, 196, 4608)
        assert mask.shape == (2, 196)
        # Soft mode → all tokens valid.
        assert mask.all()
        assert "rate_loss" in aux
        assert "expected_tokens" in aux
        assert aux["compression_probs"].shape == (2, 3)

    def test_forward_b2_uses_complexity(self, cfg_b2, fertility_table):
        controller = ScriptController(
            vision_hidden_size=1152, cfg=cfg_b2, fertility_table=fertility_table
        )
        patch = torch.randn(2, 729, 1152)
        post_shuffle = torch.randn(2, 196, 4608)
        _, _, aux = controller(patch, post_shuffle, ["eng_Latn", "hin_Deva"])
        assert aux["expected_tokens"].shape == (2,)

    def test_expected_tokens_within_bounds(self, cfg_b1, fertility_table):
        controller = ScriptController(
            vision_hidden_size=1152, cfg=cfg_b1, fertility_table=fertility_table
        )
        patch = torch.randn(2, 729, 1152)
        post_shuffle = torch.randn(2, 196, 4608)
        _, _, aux = controller(patch, post_shuffle, ["eng_Latn", "eng_Latn"])
        e_tokens = aux["expected_tokens"]
        # Levels {2,4,8} → token counts {196, 49, 16}; expectations must lie
        # in this convex hull.
        assert (e_tokens >= 15.9).all()
        assert (e_tokens <= 196.1).all()

    def test_hard_compression_zeroes_trailing_slots(
        self, fertility_table
    ):
        cfg = ScriptControllerConfig(enabled=True, variant="B1", use_gumbel=True)
        controller = ScriptController(
            vision_hidden_size=1152, cfg=cfg, fertility_table=fertility_table
        )
        controller.eval()
        patch = torch.randn(1, 729, 1152)
        # Make post_shuffle have a recognisable signature so we can check zero-padding.
        post_shuffle = torch.full((1, 196, 4608), 5.0)
        out, mask, aux = controller(patch, post_shuffle, ["eng_Latn"])
        n_kept = int(mask.sum().item())
        assert n_kept in {196, 49, 16}
        # Trailing slots (after n_kept) must be zero.
        if n_kept < 196:
            assert torch.allclose(out[0, n_kept:], torch.zeros_like(out[0, n_kept:]))


class TestRateLoss:
    def test_rate_loss_is_finite_and_gradient_attached(self, cfg_b1, fertility_table):
        controller = ScriptController(
            vision_hidden_size=1152, cfg=cfg_b1, fertility_table=fertility_table
        )
        patch = torch.randn(2, 729, 1152)
        post_shuffle = torch.randn(2, 196, 4608)
        _, _, aux = controller(patch, post_shuffle, ["eng_Latn", "hin_Deva"])
        loss = aux["rate_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in controller.parameters()
        )
        assert any_grad, "rate_loss should produce gradients in controller MLP"
