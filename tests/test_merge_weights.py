"""Unit tests for scripts/merge_weights.py.

All tests are CPU-only — no model downloads required.
Run with:
    pytest tests/test_merge_weights.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on the path so we can import the script
# regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent  # expedition-tayavision/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_weights import (  # noqa: E402
    LLM_PREFIX,
    PROJECTOR_PREFIX,
    build_merged_vlm_state,
    extract_llm_state_dict,
    extract_non_llm_state_dict,
    lerp_state_dicts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(keys: list[str], shape=(4, 4)) -> dict[str, torch.Tensor]:
    """Create a synthetic state dict of random float32 tensors."""
    return {k: torch.randn(*shape) for k in keys}


def _make_vlm_state(
    llm_keys: list[str] | None = None,
    projector_keys: list[str] | None = None,
    shape=(4, 4),
) -> dict[str, torch.Tensor]:
    """Build a synthetic VLM state dict with LLM and projector keys."""
    llm_keys = llm_keys or ["weight", "bias"]
    projector_keys = projector_keys or ["linear_1.weight", "linear_1.bias"]

    state: dict[str, torch.Tensor] = {}
    for k in llm_keys:
        state[f"{LLM_PREFIX}{k}"] = torch.randn(*shape)
    for k in projector_keys:
        state[f"{PROJECTOR_PREFIX}{k}"] = torch.randn(*shape)
    return state


# ---------------------------------------------------------------------------
# lerp_state_dicts — boundary / precision tests
# ---------------------------------------------------------------------------

class TestLerpStateDicts:
    def test_alpha_zero_returns_original(self):
        """α=0 → merged must equal original exactly."""
        orig = _make_state(["w"])
        ft = _make_state(["w"])
        merged = lerp_state_dicts(orig, ft, alpha=0.0)
        assert torch.allclose(merged["w"], orig["w"].float().to(orig["w"].dtype))

    def test_alpha_one_returns_finetuned(self):
        """α=1 → merged must equal finetuned exactly."""
        orig = _make_state(["w"])
        ft = _make_state(["w"])
        merged = lerp_state_dicts(orig, ft, alpha=1.0)
        assert torch.allclose(merged["w"], ft["w"].float().to(ft["w"].dtype))

    def test_alpha_half_is_midpoint(self):
        """α=0.5 → merged is the exact midpoint of original and finetuned."""
        orig = _make_state(["w"])
        ft = _make_state(["w"])
        merged = lerp_state_dicts(orig, ft, alpha=0.5)
        expected = (orig["w"].float() + ft["w"].float()) / 2.0
        assert torch.allclose(merged["w"], expected.to(orig["w"].dtype), atol=1e-5)

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.5, 0.6, 0.7])
    def test_sweep_range_produces_valid_tensors(self, alpha: float):
        """All α values in the recommended sweep range produce finite tensors."""
        keys = ["layer.weight", "layer.bias", "norm.weight"]
        orig = _make_state(keys, shape=(8, 8))
        ft = _make_state(keys, shape=(8, 8))
        merged = lerp_state_dicts(orig, ft, alpha=alpha)

        assert set(merged.keys()) == set(keys)
        for key in keys:
            assert merged[key].shape == orig[key].shape
            assert torch.isfinite(merged[key]).all(), f"Non-finite tensor at α={alpha}, key={key}"

    def test_shape_mismatch_raises(self):
        """Mismatched tensor shapes must raise ValueError."""
        orig = {"w": torch.randn(4, 4)}
        ft = {"w": torch.randn(8, 8)}
        with pytest.raises(ValueError, match="Shape mismatch"):
            lerp_state_dicts(orig, ft, alpha=0.5)

    def test_missing_key_raises(self):
        """Key present in orig but absent in ft must raise ValueError."""
        orig = {"w": torch.randn(4, 4), "b": torch.randn(4)}
        ft = {"w": torch.randn(4, 4)}  # 'b' missing
        with pytest.raises(ValueError, match="Key mismatch"):
            lerp_state_dicts(orig, ft, alpha=0.5)

    def test_invalid_alpha_raises(self):
        """α outside [0, 1] must raise ValueError."""
        orig = {"w": torch.randn(4)}
        ft = {"w": torch.randn(4)}
        with pytest.raises(ValueError, match="alpha must be"):
            lerp_state_dicts(orig, ft, alpha=1.5)


# ---------------------------------------------------------------------------
# extract_llm_state_dict / extract_non_llm_state_dict
# ---------------------------------------------------------------------------

class TestExtractStateDicts:
    def test_extract_llm_strips_prefix(self):
        vlm_state = _make_vlm_state(llm_keys=["model.weight"], projector_keys=["fc.weight"])
        llm = extract_llm_state_dict(vlm_state)
        assert "model.weight" in llm
        assert all(not k.startswith(LLM_PREFIX) for k in llm)

    def test_extract_non_llm_retains_full_keys(self):
        vlm_state = _make_vlm_state(llm_keys=["model.weight"], projector_keys=["fc.weight"])
        non_llm = extract_non_llm_state_dict(vlm_state)
        assert f"{PROJECTOR_PREFIX}fc.weight" in non_llm
        assert all(not k.startswith(LLM_PREFIX) for k in non_llm)

    def test_extract_llm_excludes_non_llm_keys(self):
        vlm_state = _make_vlm_state()
        llm = extract_llm_state_dict(vlm_state)
        assert all(not k.startswith(PROJECTOR_PREFIX) for k in llm)


# ---------------------------------------------------------------------------
# build_merged_vlm_state — integration-level (CPU, synthetic)
# ---------------------------------------------------------------------------

class TestBuildMergedVlmState:
    def _make_original_llm(self, llm_keys=None, shape=(4, 4)):
        """Original text LLM state dict (no prefix)."""
        llm_keys = llm_keys or ["weight", "bias"]
        return {k: torch.randn(*shape) for k in llm_keys}

    def _make_finetuned_vlm(self, llm_keys=None, shape=(4, 4)):
        """Fine-tuned VLM state dict (with LLM + projector prefixes)."""
        return _make_vlm_state(llm_keys=llm_keys, shape=shape)

    def test_projector_weights_unchanged(self):
        """Projector weights in the merged dict must be identical to fine-tuned."""
        orig_llm = self._make_original_llm()
        ft_vlm = self._make_finetuned_vlm()

        merged = build_merged_vlm_state(orig_llm, ft_vlm, alpha=0.5)

        proj_key = f"{PROJECTOR_PREFIX}linear_1.weight"
        assert torch.allclose(merged[proj_key], ft_vlm[proj_key])

    def test_merged_contains_llm_prefix_keys(self):
        """Merged dict must include language_model.* prefixed LLM keys."""
        orig_llm = self._make_original_llm(llm_keys=["model.weight"])
        ft_vlm = self._make_finetuned_vlm(llm_keys=["model.weight"])

        merged = build_merged_vlm_state(orig_llm, ft_vlm, alpha=0.4)

        assert f"{LLM_PREFIX}model.weight" in merged

    def test_alpha_zero_llm_equals_original(self):
        """With α=0, the LLM portion of the merged dict equals the original."""
        orig_llm = self._make_original_llm()
        ft_vlm = self._make_finetuned_vlm()

        merged = build_merged_vlm_state(orig_llm, ft_vlm, alpha=0.0)

        for k, v in orig_llm.items():
            merged_key = f"{LLM_PREFIX}{k}"
            assert torch.allclose(merged[merged_key], v.float().to(v.dtype)), (
                f"Mismatch at key '{merged_key}' with α=0"
            )

    def test_alpha_one_llm_equals_finetuned(self):
        """With α=1, the LLM portion of the merged dict equals the fine-tuned LLM."""
        orig_llm = self._make_original_llm()
        ft_vlm = self._make_finetuned_vlm()

        merged = build_merged_vlm_state(orig_llm, ft_vlm, alpha=1.0)

        ft_llm = extract_llm_state_dict(ft_vlm)
        for k, v in ft_llm.items():
            merged_key = f"{LLM_PREFIX}{k}"
            assert torch.allclose(merged[merged_key], v.float().to(v.dtype)), (
                f"Mismatch at key '{merged_key}' with α=1"
            )


# ---------------------------------------------------------------------------
# Output file written to disk
# ---------------------------------------------------------------------------

class TestOutputFile:
    def test_merged_state_pt_is_created(self):
        """The script's _save_outputs writes merged_state.pt to the output dir."""
        from merge_weights import _save_outputs  # noqa: PLC0415

        state = {"language_model.weight": torch.randn(4, 4)}
        with tempfile.TemporaryDirectory() as tmpdir:
            _save_outputs(
                merged_state=state,
                output_dir=Path(tmpdir),
                dtype=torch.float32,
                save_hf=False,
                original_llm_name="",  # not used when save_hf=False
            )
            saved = torch.load(Path(tmpdir) / "merged_state.pt", weights_only=True)

        assert "language_model.weight" in saved
        assert torch.allclose(saved["language_model.weight"], state["language_model.weight"])

    def test_saved_file_dtype(self):
        """Weights are cast to the requested dtype before saving."""
        from merge_weights import _save_outputs  # noqa: PLC0415

        state = {"language_model.w": torch.randn(4, 4, dtype=torch.float32)}
        with tempfile.TemporaryDirectory() as tmpdir:
            _save_outputs(
                merged_state=state,
                output_dir=Path(tmpdir),
                dtype=torch.float16,
                save_hf=False,
                original_llm_name="",
            )
            saved = torch.load(Path(tmpdir) / "merged_state.pt", weights_only=True)

        assert saved["language_model.w"].dtype == torch.float16
