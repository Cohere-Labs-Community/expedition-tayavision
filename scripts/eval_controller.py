#!/usr/bin/env python3
"""Stage-1 → Stage-2 gate: probe a pre-trained controller checkpoint.

Loads a controller state_dict produced by ``pipeline/pretrain_controller.py``
and runs the MLP on 16 representative FLORES-200 languages with synthetic
post-shuffle features. Prints per-language ``compression_probs`` and
``expected_tokens``, plus the correlation between ``expected_tokens`` and
``tpw``. Used to decide whether Stage 2 is worth launching.

Pass criteria (see plan):
  - per-language ``expected_tokens.std() > 5.0``
  - ``|corr(expected_tokens, tpw)| > 0.3``

Usage:
  python scripts/eval_controller.py \\
      --controller-ckpt /data/.../controller_2450.pt \\
      --fertility-table data/fertility/qwen3.json \\
      --variant B1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.script_controller import (
    FertilityTable,
    ScriptController,
    ScriptControllerConfig,
)

# 16 languages spanning low-/mid-/high-fertility scripts for Qwen3.
# Chosen to maximize per-script diversity, not by sample frequency.
PROBE_LANGS: list[str] = [
    # Latin (low fertility)
    "en", "es", "fr", "de", "pt",
    # Cyrillic
    "ru", "bg",
    # Devanagari / Indic
    "hi", "bn", "ta",
    # CJK
    "zh", "ja", "ko",
    # Arabic / Hebrew / Thai / Burmese (high fertility)
    "ar", "th", "my",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller-ckpt", required=True, type=Path)
    p.add_argument("--fertility-table", required=True, type=Path)
    p.add_argument("--variant", default="B1", choices=["B1", "B2", "B3"])
    p.add_argument("--target-tokens", default=98, type=int)
    p.add_argument("--vision-hidden", default=1152, type=int,
                   help="SigLIP2-so400m hidden size; pre-shuffle features dim")
    args = p.parse_args()

    fertility = FertilityTable.load(args.fertility_table)
    if not fertility.per_language:
        raise SystemExit(
            f"Fertility table at {args.fertility_table} is empty. "
            "Generate it via scripts/compute_fertility_table.py first."
        )

    cfg = ScriptControllerConfig(
        enabled=True,
        variant=args.variant,
        target_tokens=args.target_tokens,
        rate_lambda=0.1,
        fertility_table_path=str(args.fertility_table),
    )
    controller = ScriptController(
        vision_hidden_size=args.vision_hidden,
        cfg=cfg,
        fertility_table=fertility,
    )
    state = torch.load(args.controller_ckpt, map_location="cpu", weights_only=True)
    controller.load_state_dict(state, strict=True)
    controller.eval()

    # Synthetic features — controller only needs language inputs for B1.
    # For B2/B3 we feed unit-variance dummies; per-language differences
    # will come only from the language signal.
    bsz = len(PROBE_LANGS)
    torch.manual_seed(0)
    patch_tokens = torch.randn(bsz, 729, args.vision_hidden) * 0.1
    post_shuffle = torch.randn(bsz, 196, args.vision_hidden * 4) * 0.1

    with torch.no_grad():
        _, _, aux = controller(patch_tokens, post_shuffle, lang_codes=PROBE_LANGS)
    probs = aux["compression_probs"]
    expected = aux["expected_tokens"]
    target = aux["target_tokens"]
    tpws = torch.tensor([fertility.lookup_tpw(l) for l in PROBE_LANGS])

    # Per-language breakdown
    print(f"\n{'lang':<6}{'tpw':>6}{'target':>8}{'E[tok]':>8}  "
          f"{'p(196)':>7}{'p(49)':>7}{'p(16)':>7}")
    for i, lang in enumerate(PROBE_LANGS):
        ps = probs[i].tolist()
        print(
            f"{lang:<6}{tpws[i].item():>6.2f}{target[i].item():>8.1f}"
            f"{expected[i].item():>8.1f}  "
            f"{ps[0]:>7.3f}{ps[1]:>7.3f}{ps[2]:>7.3f}"
        )

    # Aggregate diagnostics
    et_std = float(expected.std().item())
    et_centered = expected - expected.mean()
    tpw_centered = tpws - tpws.mean()
    denom = (et_centered.norm() * tpw_centered.norm()).clamp(min=1e-8)
    corr = float((et_centered * tpw_centered).sum() / denom)

    print()
    print(f"expected_tokens.std() = {et_std:.3f}    (gate: > 5.0)")
    print(f"corr(E[tok], tpw)     = {corr:+.3f}    (gate: |.| > 0.3)")
    et_ok = et_std > 5.0
    corr_ok = abs(corr) > 0.3
    if et_ok and corr_ok:
        print("\nPASS — controller has a non-trivial per-language policy. Stage 2 viable.")
        sys.exit(0)
    else:
        print("\nFAIL — controller did not learn a meaningful per-language policy.")
        if not et_ok:
            print("  - per-language expected_tokens spread is too narrow.")
        if not corr_ok:
            print("  - expected_tokens not correlated with tpw.")
        sys.exit(1)


if __name__ == "__main__":
    main()
