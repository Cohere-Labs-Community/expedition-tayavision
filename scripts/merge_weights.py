"""merge_weights.py — Interpolate multimodal fine-tuned and text-only Tiny Aya weights.

Usage
-----
python scripts/merge_weights.py \\
    --original  CohereLabs/tiny-aya-base \\
    --finetuned ./checkpoints/tiny-aya-vision-ft \\
    --alpha     0.5 \\
    --output    ./merged/alpha_0.5 \\
    [--save-hf] \\
    [--dtype bfloat16] \\
    [--device cpu]

Merge strategy
--------------
Only the language-model backbone (all keys prefixed with ``language_model.``) is
interpolated via linear interpolation (LERP):

    merged_param = (1 - α) × original_param  +  α × finetuned_param

The multimodal projector (``multi_modal_projector.*``) and vision encoder
(``vision_encoder.*``) weights are **not** participants of this interpolation —
they are kept verbatim from the fine-tuned checkpoint because they contain no
text-only signal.

α = 0.0  →  identical to the original text-only Tiny Aya Base
α = 1.0  →  identical to the multimodal fine-tuned VLM
Recommended sweep range: {0.3, 0.4, 0.5, 0.6, 0.7}
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_PREFIX = "language_model."
PROJECTOR_PREFIX = "multi_modal_projector."
VISION_PREFIX = "vision_encoder."


# ---------------------------------------------------------------------------
# Core merge logic (importable for tests)
# ---------------------------------------------------------------------------

def lerp_state_dicts(
    original: Dict[str, torch.Tensor],
    finetuned: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Linear interpolation of two state dicts with matching keys.

    ``original`` and ``finetuned`` must have the same set of keys and
    identically-shaped tensors for every key.

    Args:
        original:  State dict of the text-only LLM (keys without any prefix).
        finetuned: State dict of the fine-tuned LLM (keys without any prefix).
        alpha:     Merge coefficient in [0, 1]. 0 → original; 1 → finetuned.

    Returns:
        A new state dict with merged tensors (detached, on CPU).

    Raises:
        ValueError: If keys or shapes do not match.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    orig_keys = set(original.keys())
    ft_keys = set(finetuned.keys())

    missing_in_ft = orig_keys - ft_keys
    missing_in_orig = ft_keys - orig_keys

    if missing_in_ft or missing_in_orig:
        raise ValueError(
            f"Key mismatch between original and finetuned state dicts.\n"
            f"  Missing in finetuned:  {sorted(missing_in_ft)[:5]!r}{'...' if len(missing_in_ft) > 5 else ''}\n"
            f"  Missing in original:   {sorted(missing_in_orig)[:5]!r}{'...' if len(missing_in_orig) > 5 else ''}"
        )

    merged: Dict[str, torch.Tensor] = {}

    for key in original:
        orig_t = original[key]
        ft_t = finetuned[key]

        if orig_t.shape != ft_t.shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': "
                f"original={tuple(orig_t.shape)}, finetuned={tuple(ft_t.shape)}"
            )

        # Cast to float32 for precision during interpolation, then back
        orig_f = orig_t.float()
        ft_f = ft_t.float()
        merged_f = (1.0 - alpha) * orig_f + alpha * ft_f

        # Preserve original dtype
        merged[key] = merged_f.to(orig_t.dtype).detach().cpu()

    return merged


def extract_llm_state_dict(full_vlm_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract LLM parameters from a full VLM state dict, stripping the prefix.

    Args:
        full_vlm_state: State dict from ``TinyAyaVisionForConditionalGeneration``.

    Returns:
        Dict with keys where ``language_model.`` prefix has been removed.
    """
    return {
        key[len(LLM_PREFIX):]: val
        for key, val in full_vlm_state.items()
        if key.startswith(LLM_PREFIX)
    }


def extract_non_llm_state_dict(full_vlm_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract non-LLM parameters (projector + vision encoder) from VLM state dict.

    These keys are kept verbatim from the fine-tuned checkpoint.
    """
    return {
        key: val
        for key, val in full_vlm_state.items()
        if not key.startswith(LLM_PREFIX)
    }


def build_merged_vlm_state(
    original_llm_state: Dict[str, torch.Tensor],
    finetuned_vlm_state: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Build a complete merged VLM state dict.

    LLM keys are linearly interpolated; projector and vision encoder keys
    are copied from the fine-tuned checkpoint untouched.

    Args:
        original_llm_state:  State dict of ``AutoModelForCausalLM``
                             (i.e. the text-only Tiny Aya Base).
        finetuned_vlm_state: State dict of
                             ``TinyAyaVisionForConditionalGeneration``.
        alpha:               Merge coefficient in [0, 1].

    Returns:
        Merged state dict ready to be loaded into a
        ``TinyAyaVisionForConditionalGeneration`` instance.
    """
    ft_llm_state = extract_llm_state_dict(finetuned_vlm_state)
    non_llm_state = extract_non_llm_state_dict(finetuned_vlm_state)

    log.info(
        "Merging %d LLM parameter tensors with α=%.2f …", len(original_llm_state), alpha
    )

    merged_llm = lerp_state_dicts(original_llm_state, ft_llm_state, alpha)

    # Re-attach the ``language_model.`` prefix and combine with non-LLM weights
    merged_vlm: Dict[str, torch.Tensor] = {}
    for key, val in merged_llm.items():
        merged_vlm[f"{LLM_PREFIX}{key}"] = val
    merged_vlm.update(non_llm_state)

    return merged_vlm


# ---------------------------------------------------------------------------
# Merge summary
# ---------------------------------------------------------------------------

def _print_merge_summary(
    original_llm: Dict[str, torch.Tensor],
    merged_llm: Dict[str, torch.Tensor],
    alpha: float,
    output_path: Path,
) -> None:
    """Print a human-readable summary of the merge operation."""
    total_params = sum(t.numel() for t in merged_llm.values())

    # Compute overall L2 norm delta between original and merged LLM weights
    delta_sq_sum = 0.0
    for key in original_llm:
        delta_sq_sum += (
            (merged_llm[key].float() - original_llm[key].float()) ** 2
        ).sum().item()
    norm_delta = delta_sq_sum ** 0.5

    print("\n" + "=" * 60)
    print("  Tiny Aya Vision — Weight Merge Summary")
    print("=" * 60)
    print(f"  α (merge ratio)   : {alpha:.2f}  (0=text-only, 1=full VLM)")
    print(f"  LLM param tensors : {len(original_llm):,}")
    print(f"  Total params (LLM): {total_params:,}")
    print(f"  ‖merged − orig‖₂  : {norm_delta:.4f}")
    print(f"  Output path       : {output_path}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_original_llm(model_name: str, device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Load the original text-only LLM state dict."""
    log.info("Loading original LLM from '%s' …", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def _load_finetuned_vlm(checkpoint_path: str, device: str) -> Dict[str, torch.Tensor]:
    """Load a fine-tuned VLM state dict from a ``.pt`` checkpoint or directory."""
    p = Path(checkpoint_path)

    # Try directory: look for a state dict file
    if p.is_dir():
        candidates = list(p.glob("*.pt")) + list(p.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No .pt / .pth checkpoint file found in '{checkpoint_path}'. "
                "Pass the path to a .pt file directly, or ensure the directory "
                "contains one."
            )
        checkpoint_path = str(candidates[0])
        log.info("Found checkpoint file: %s", checkpoint_path)

    log.info("Loading fine-tuned VLM state dict from '%s' …", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # unwrap common wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    return {k: v.detach().cpu() for k, v in state.items()}


def _save_outputs(
    merged_state: Dict[str, torch.Tensor],
    output_dir: Path,
    dtype: torch.dtype,
    save_hf: bool,
    original_llm_name: str,
) -> None:
    """Save merged state dict (and optionally a HuggingFace model dir)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always save a raw state dict
    pt_path = output_dir / "merged_state.pt"
    # Cast to target dtype before saving
    cast_state = {k: v.to(dtype) for k, v in merged_state.items()}
    torch.save(cast_state, pt_path)
    log.info("Saved merged state dict → %s", pt_path)

    if save_hf:
        hf_dir = output_dir / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)

        # Extract just LLM weights (stripped of prefix) to load into base model
        llm_state = {
            key[len(LLM_PREFIX):]: val
            for key, val in cast_state.items()
            if key.startswith(LLM_PREFIX)
        }

        log.info("Building HF model at '%s' …", hf_dir)
        model = AutoModelForCausalLM.from_pretrained(
            original_llm_name,
            torch_dtype=dtype,
        )
        missing, unexpected = model.load_state_dict(llm_state, strict=False)
        if missing:
            log.warning("HF save: %d missing keys (expected if vocab was resized): %s …",
                        len(missing), missing[:3])
        if unexpected:
            log.warning("HF save: %d unexpected keys: %s …", len(unexpected), unexpected[:3])

        model.save_pretrained(str(hf_dir))
        log.info("Saved HF model → %s", hf_dir)
        del model


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multimodal fine-tuned Tiny Aya weights with text-only base weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original",
        required=True,
        help="HuggingFace Hub ID or local path for the text-only Tiny Aya Base LLM.",
    )
    parser.add_argument(
        "--finetuned",
        required=True,
        help="Path to fine-tuned VLM checkpoint (.pt file or directory containing one).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Merge ratio α ∈ [0, 1]. 0 = pure original text model; 1 = pure fine-tuned VLM.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. merged_state.pt will be written here.",
    )
    parser.add_argument(
        "--save-hf",
        action="store_true",
        default=False,
        help="Also save the merged LLM backbone as a HuggingFace model directory.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype to cast weights to before saving.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load model weights onto (e.g. 'cpu', 'cuda:0').",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    alpha = args.alpha
    if not 0.0 <= alpha <= 1.0:
        log.error("--alpha must be in [0, 1], got %.3f", alpha)
        sys.exit(1)

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output)

    # ---- Load weights ----
    original_llm_state = _load_original_llm(args.original, args.device, dtype)
    finetuned_vlm_state = _load_finetuned_vlm(args.finetuned, args.device)

    # ---- Merge ----
    merged_vlm_state = build_merged_vlm_state(original_llm_state, finetuned_vlm_state, alpha)

    # ---- Summary ----
    merged_llm_state = {
        key[len(LLM_PREFIX):]: val
        for key, val in merged_vlm_state.items()
        if key.startswith(LLM_PREFIX)
    }
    _print_merge_summary(original_llm_state, merged_llm_state, alpha, output_dir)

    # ---- Save ----
    _save_outputs(
        merged_vlm_state,
        output_dir,
        dtype=dtype,
        save_hf=args.save_hf,
        original_llm_name=args.original,
    )

    log.info("Done. ✓")


if __name__ == "__main__":
    main()
