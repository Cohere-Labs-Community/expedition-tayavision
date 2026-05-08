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

`--alpha` accepts either one float (e.g. `0.5`) or a comma-separated sweep
(e.g. `0.3,0.4,0.5`). For sweeps, each alpha writes to a subdirectory under
`--output` named `alpha_<value>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is importable (consistent with apply_lora.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

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
OUTPUT_SHARD_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GiB

# Weight-tying keys: safetensors deduplication may drop one side of a tied
# pair.  We restore it so that both the original and fine-tuned state dicts
# always present the same key set to lerp_state_dicts.
_TIED_PAIRS = [
    # (source_key, tied_key) — source is kept by safetensors, tied is dropped
    ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight"),
]


class _ShardedSafetensorWriter:
    """Write safetensors shards while keeping only one shard in memory."""

    def __init__(self, output_dir: Path, shard_size_bytes: int = OUTPUT_SHARD_SIZE_BYTES):
        self.output_dir = output_dir
        self.shard_size_bytes = shard_size_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: Dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._parts: list[tuple[Path, list[str]]] = []

    @staticmethod
    def _tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def add(self, key: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().cpu().contiguous()
        t_bytes = self._tensor_nbytes(t)

        if self._buffer and self._buffer_bytes + t_bytes > self.shard_size_bytes:
            self.flush()

        self._buffer[key] = t
        self._buffer_bytes += t_bytes

    def flush(self) -> None:
        if not self._buffer:
            return

        part_idx = len(self._parts) + 1
        part_path = self.output_dir / f"model-part-{part_idx:05d}.safetensors"
        save_file(self._buffer, str(part_path))
        self._parts.append((part_path, list(self._buffer.keys())))

        self._buffer = {}
        self._buffer_bytes = 0

    def finalize(self) -> None:
        self.flush()
        total = len(self._parts)
        if total == 0:
            raise ValueError("No tensors were written to output shards")

        weight_map: Dict[str, str] = {}
        total_size = 0

        for idx, (part_path, keys) in enumerate(self._parts, start=1):
            final_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
            final_path = self.output_dir / final_name
            part_path.rename(final_path)

            for key in keys:
                weight_map[key] = final_name

            total_size += final_path.stat().st_size

        index_path = self.output_dir / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _restore_tied_weights(state: Dict[str, torch.Tensor]) -> None:
    """Restore weight-tied keys that safetensors deduplication may have dropped.

    Mutates *state* in-place.  Only acts when the source key is present and
    the tied key is absent — never overwrites an existing key.
    """
    for src_key, tied_key in _TIED_PAIRS:
        if src_key in state and tied_key not in state:
            state[tied_key] = state[src_key]


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

    tied_keys = set(original_llm_state) - set(ft_llm_state)
    if tied_keys:
        log.info(
            "Excluding %d key(s) from original absent in finetuned "
            "(weight-tied or not saved — will be restored on model load): %s",
            len(tied_keys), sorted(tied_keys),
        )
        original_llm_state = {k: v for k, v in original_llm_state.items() if k not in tied_keys}

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


def _alpha_label(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def _resolve_model_dir(model_ref: str) -> Path:
    p = Path(model_ref)
    if p.is_dir():
        return p
    return Path(snapshot_download(repo_id=model_ref))


def _get_weight_map(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return {k: v for k, v in payload.get("weight_map", {}).items()}

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: single.name for k in f.keys()}

    shard_files = sorted(model_dir.glob("*.safetensors"))
    if shard_files:
        out: Dict[str, str] = {}
        for shard in shard_files:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    out[k] = shard.name
        return out

    raise FileNotFoundError(f"No safetensors weights found under {model_dir}")


def _copy_hf_metadata_files(src_dir: Path, dst_dir: Path) -> None:
    keep_exts = {".json", ".txt", ".model", ".py", ".md"}
    skip_names = {"model.safetensors.index.json"}

    for item in src_dir.iterdir():
        if item.is_dir():
            continue
        if item.name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            continue
        if item.name in skip_names:
            continue
        if item.suffix.lower() in keep_exts or item.name in {"README", "LICENSE"}:
            shutil.copy2(item, dst_dir / item.name)


def _save_outputs(
    merged_state: Dict[str, torch.Tensor],
    output_dir: Path,
    dtype: torch.dtype,
    finetuned_vlm_name: str | None,
    device: str = "cpu",
    save_model: bool = True,
) -> None:
    """Compatibility helper retained for unit tests.

    The streaming merge path does not use this function.
    """
    del finetuned_vlm_name, device, save_model
    output_dir.mkdir(parents=True, exist_ok=True)
    cast_state = {k: v.to(dtype) for k, v in merged_state.items()}
    torch.save(cast_state, output_dir / "merged_state.pt")


def _stream_merge_one_alpha(
    *,
    original_dir: Path,
    finetuned_dir: Path,
    original_map: Dict[str, str],
    finetuned_map: Dict[str, str],
    alpha: float,
    output_dir: Path,
    dtype: torch.dtype,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_hf_metadata_files(finetuned_dir, output_dir)

    orig_embed = "model.embed_tokens.weight"
    ft_embed = f"{LLM_PREFIX}model.embed_tokens.weight"
    ft_lm_head = f"{LLM_PREFIX}lm_head.weight"
    has_ft_embed = ft_embed in finetuned_map
    has_ft_lm_head = ft_lm_head in finetuned_map

    llm_keys: list[str] = []
    for k in sorted(original_map):
        if k == "lm_head.weight" and has_ft_embed and not has_ft_lm_head:
            continue
        ft_key = f"{LLM_PREFIX}{k}"
        if ft_key not in finetuned_map:
            raise KeyError(f"Missing fine-tuned key for original key '{k}'")
        llm_keys.append(k)

    log.info("Merging %d LLM parameter tensors with α=%.2f …", len(llm_keys), alpha)

    writer = _ShardedSafetensorWriter(output_dir)
    delta_sq_sum = 0.0
    total_params = 0
    out_weight_map: Dict[str, str] = {}

    # Merge LLM tensors
    for key in llm_keys:
        orig_file = original_dir / original_map[key]
        ft_key = f"{LLM_PREFIX}{key}"
        ft_file = finetuned_dir / finetuned_map[ft_key]

        with safe_open(str(orig_file), framework="pt", device="cpu") as f_orig:
            orig_t = f_orig.get_tensor(key)
        with safe_open(str(ft_file), framework="pt", device="cpu") as f_ft:
            ft_t = f_ft.get_tensor(ft_key)

        if orig_t.shape != ft_t.shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': "
                f"original={tuple(orig_t.shape)}, finetuned={tuple(ft_t.shape)}"
            )

        orig_f = orig_t.float()
        ft_f = ft_t.float()
        merged_f = (1.0 - alpha) * orig_f + alpha * ft_f
        merged_out = merged_f.to(orig_t.dtype).to(dtype).detach().cpu().contiguous()

        delta_sq_sum += ((merged_out.float() - orig_t.float()) ** 2).sum().item()
        total_params += merged_out.numel()

        out_key = f"{LLM_PREFIX}{key}"
        writer.add(out_key, merged_out)
        del orig_t, ft_t, orig_f, ft_f, merged_f, merged_out

    # Copy non-LLM tensors from finetuned untouched (except output dtype cast)
    for ft_key, rel_file in sorted(finetuned_map.items()):
        if ft_key.startswith(LLM_PREFIX):
            continue
        with safe_open(str(finetuned_dir / rel_file), framework="pt", device="cpu") as f_ft:
            t = f_ft.get_tensor(ft_key)
        writer.add(ft_key, t.to(dtype).detach().cpu().contiguous())
        del t

    writer.finalize()

    # Also export a compatibility pt file; load shard-by-shard to avoid full spikes.
    index_payload = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for k, shard in index_payload["weight_map"].items():
        out_weight_map.setdefault(shard, []).append(k)
    merged_state: Dict[str, torch.Tensor] = {}
    for shard, keys in sorted(out_weight_map.items()):
        with safe_open(str(output_dir / shard), framework="pt", device="cpu") as f:
            for k in keys:
                merged_state[k] = f.get_tensor(k)
    torch.save(merged_state, output_dir / "merged_state.pt")

    norm_delta = delta_sq_sum ** 0.5
    print("\n" + "=" * 60)
    print("  Tiny Aya Vision — Weight Merge Summary")
    print("=" * 60)
    print(f"  α (merge ratio)   : {alpha:.2f}  (0=text-only, 1=full VLM)")
    print(f"  LLM param tensors : {len(llm_keys):,}")
    print(f"  Total params (LLM): {total_params:,}")
    print(f"  ‖merged − orig‖₂  : {norm_delta:.4f}")
    print(f"  Output path       : {output_dir}")
    print("=" * 60 + "\n")


def _push_to_hub(
    folder_path: Path,
    repo_id: str,
    path_in_repo: str | None,
    alpha: float,
    private: bool,
) -> None:
    """Upload an output folder to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    upload_kwargs = {
        "folder_path": str(folder_path),
        "repo_id": repo_url.repo_id,
        "commit_message": f"Upload merged weights (alpha={alpha})",
    }
    if path_in_repo:
        upload_kwargs["path_in_repo"] = path_in_repo
    api.upload_folder(**upload_kwargs)
    suffix = f"/tree/main/{path_in_repo}" if path_in_repo else ""
    log.info("Pushed merged model to %s%s", repo_url, suffix)


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
        required=True,
        help=(
            "Merge ratio(s) α ∈ [0, 1]. Accepts a single float (e.g. 0.5) "
            "or comma-separated list (e.g. 0.3,0.4,0.5)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory. This becomes a complete save_pretrained model "
            "directory and also includes merged_state.pt."
        ),
    )
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="Optional destination Hugging Face repo ID. If set, upload output there.",
    )
    parser.add_argument(
        "--hub-path-in-repo",
        default=None,
        help="Optional subdirectory inside the destination Hub repo, useful for alpha sweeps.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        default=False,
        help="Create the destination Hub repo as private if it does not exist.",
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
        help="Unused now; retained for CLI compatibility.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        alphas = [float(v.strip()) for v in str(args.alpha).split(",") if v.strip()]
    except ValueError:
        log.error("--alpha must be a float or comma-separated floats, got %r", args.alpha)
        sys.exit(1)
    if not alphas:
        log.error("--alpha cannot be empty")
        sys.exit(1)
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            log.error("--alpha must be in [0, 1], got %.3f", alpha)
            sys.exit(1)

    dtype = getattr(torch, args.dtype)
    output_root = Path(args.output)
    is_sweep = len(alphas) > 1

    original_dir = _resolve_model_dir(args.original)
    finetuned_dir = _resolve_model_dir(args.finetuned)
    original_map = _get_weight_map(original_dir)
    finetuned_map = _get_weight_map(finetuned_dir)

    for alpha in alphas:
        label = _alpha_label(alpha)
        output_dir = output_root / f"alpha_{label}" if is_sweep else output_root

        _stream_merge_one_alpha(
            original_dir=original_dir,
            finetuned_dir=finetuned_dir,
            original_map=original_map,
            finetuned_map=finetuned_map,
            alpha=alpha,
            output_dir=output_dir,
            dtype=dtype,
        )

        if args.hub_repo_id:
            path_in_repo = args.hub_path_in_repo
            if is_sweep:
                path_in_repo = f"{path_in_repo.rstrip('/')}/alpha_{label}" if path_in_repo else f"alpha_{label}"
            _push_to_hub(
                folder_path=output_dir,
                repo_id=args.hub_repo_id,
                path_in_repo=path_in_repo,
                alpha=alpha,
                private=args.hub_private,
            )

    log.info("Done. ✓")


if __name__ == "__main__":
    main()
