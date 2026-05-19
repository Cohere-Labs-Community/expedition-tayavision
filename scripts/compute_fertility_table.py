#!/usr/bin/env python
"""Compute per-language and per-script tokenizer fertility for a backbone.

Runs a backbone's tokenizer over a multilingual corpus (FLORES-200 dev by
default) and writes a JSON table that the Script-Conditioned Token
Allocation Controller loads at init time.

Output schema (matches ``FertilityTable.load``):

    {
      "backbone": "qwen3",
      "per_language": {
        "eng_Latn": {"script": "Latn", "tpw": 1.32, "tpc": 0.27},
        ...
      },
      "per_script": {"Latn": 1.45, "Deva": 2.51, ...}
    }

Usage::

    python scripts/compute_fertility_table.py --backbone qwen3 \\
        --output data/fertility/qwen3.json

The backbone-name to HF-id mapping mirrors ``config/backbone/<name>.yaml``.
A custom corpus path can be passed via ``--corpus-path`` (one line per
sentence, plain text); pair it with ``--lang-code`` to tag the output.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer


_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_backbone_model_name(backbone: str) -> str:
    """Read ``config/backbone/<backbone>.yaml`` and return ``llm_model_name``."""
    yaml_path = _REPO_ROOT / "config" / "backbone" / f"{backbone}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No backbone config at {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    name = cfg.get("llm_model_name")
    if not name:
        raise ValueError(f"{yaml_path} has no 'llm_model_name' field")
    return name


def iter_flores200_devsets(max_languages: int | None = None):
    """Yield ``(lang_code, sentences)`` from FLORES-200 dev split.

    Uses ``openlanguagedata/flores_plus`` (a parquet mirror that works with
    ``datasets>=3.0``). Older mirrors (``facebook/flores``,
    ``Muennighoff/flores200``, ``gsarti/flores_101``) ship a loader script
    and are unusable on datasets>=3.0, so we don't bother trying them.
    """
    from datasets import load_dataset

    hf_id = "openlanguagedata/flores_plus"
    try:
        ds = load_dataset(hf_id, split="dev", token=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "gated dataset" in msg or "is gated" in msg or "401" in msg:
            raise RuntimeError(
                f"{hf_id} is gated. Accept the dataset license at "
                f"https://huggingface.co/datasets/{hf_id} (one-time click), "
                "ensure `huggingface-cli login` (or HF_TOKEN) is set, then re-run."
            ) from e
        raise RuntimeError(
            f"Failed to load {hf_id}: {type(e).__name__}: {msg}"
        ) from e

    # flores_plus exposes ISO-639-3 (lang) and ISO-15924 (script) separately;
    # combine them into the FLORES-200 ``lang_Script`` code that the
    # downstream FertilityTable consumer expects.
    cols = ds.column_names
    text_col = next((c for c in ("text", "sentence") if c in cols), None)
    if text_col is None:
        raise RuntimeError(
            f"FLORES dataset has no text column (got {cols}) — update the iter helper"
        )
    has_split_codes = "iso_639_3" in cols and "iso_15924" in cols
    lang_col = next(
        (c for c in ("iso_639_3", "language", "lang", "id") if c in cols), None
    )
    if lang_col is None:
        raise RuntimeError(
            f"FLORES dataset has no language column (got {cols}) — update the iter helper"
        )

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        text = row[text_col]
        if not isinstance(text, str):
            continue
        if has_split_codes:
            iso_lang = row.get("iso_639_3")
            iso_script = row.get("iso_15924")
            if not (isinstance(iso_lang, str) and isinstance(iso_script, str)):
                continue
            code = f"{iso_lang}_{iso_script}"
        else:
            code = row.get(lang_col)
            if not isinstance(code, str):
                continue
        grouped[code].append(text)

    items = list(grouped.items())
    if max_languages is not None:
        items = items[:max_languages]
    yield from items


def fertility_for_sentences(tokenizer, sentences: list[str]) -> tuple[float, float]:
    """Return (tokens_per_word, tokens_per_char) means over ``sentences``."""
    tpw_values: list[float] = []
    tpc_values: list[float] = []
    for s in sentences:
        if not s.strip():
            continue
        n_tokens = len(tokenizer.encode(s, add_special_tokens=False))
        n_words = max(1, len(s.split()))
        n_chars = max(1, len(s))
        tpw_values.append(n_tokens / n_words)
        tpc_values.append(n_tokens / n_chars)
    if not tpw_values:
        return 0.0, 0.0
    return statistics.fmean(tpw_values), statistics.fmean(tpc_values)


def script_from_flores_code(lang_code: str) -> str:
    """Extract the ISO 15924 script tag from a FLORES code like ``eng_Latn``."""
    if "_" in lang_code:
        return lang_code.split("_", 1)[1]
    return ""


def build_table(
    tokenizer,
    sources: list[tuple[str, list[str]]],
) -> dict:
    """Compute per-language and per-script fertilities for ``sources``."""
    per_language: dict[str, dict] = {}
    by_script: dict[str, list[float]] = defaultdict(list)

    for lang_code, sentences in sources:
        tpw, tpc = fertility_for_sentences(tokenizer, sentences)
        script = script_from_flores_code(lang_code)
        per_language[lang_code] = {"script": script, "tpw": tpw, "tpc": tpc}
        if script:
            by_script[script].append(tpw)

    per_script = {s: statistics.fmean(vals) for s, vals in by_script.items() if vals}
    return {"per_language": per_language, "per_script": per_script}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, help="Backbone name in config/backbone/")
    parser.add_argument(
        "--corpus",
        default="flores200",
        choices=("flores200", "custom"),
        help="Source corpus. 'custom' requires --corpus-path and --lang-code.",
    )
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--lang-code", default=None, help="Tag for --corpus custom")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-languages",
        type=int,
        default=None,
        help="Limit number of languages (for fast smoke runs).",
    )
    args = parser.parse_args()

    model_name = resolve_backbone_model_name(args.backbone)
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if args.corpus == "flores200":
        sources = list(iter_flores200_devsets(args.max_languages))
    else:
        if not args.corpus_path or not args.lang_code:
            raise SystemExit("--corpus custom requires --corpus-path and --lang-code")
        sentences = args.corpus_path.read_text(encoding="utf-8").splitlines()
        sources = [(args.lang_code, sentences)]

    print(f"Computing fertility over {len(sources)} languages")
    table = build_table(tokenizer, sources)
    table["backbone"] = args.backbone

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
