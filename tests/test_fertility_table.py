"""Verify the fertility-table generation script's helpers on a tiny corpus."""

from __future__ import annotations

from transformers import AutoTokenizer

from scripts.compute_fertility_table import (
    build_table,
    fertility_for_sentences,
    script_from_flores_code,
)

# Small, deterministic tokenizer that ships with transformers and handles
# multiple scripts. Used as a smoke proxy for backbone tokenizers (Qwen3,
# Tiny Aya) so the test does not hit the network.
TOKENIZER_NAME = "bert-base-multilingual-cased"


_LATIN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz, judge my vow.",
]

_DEVANAGARI_SENTENCES = [
    "वह बहुत तेज़ दौड़ता है।",
    "मुझे हिंदी सीखनी है।",
    "यह एक परीक्षा वाक्य है।",
    "आज मौसम सुहावना है।",
    "वह हर रोज़ किताबें पढ़ती है।",
]


def test_script_extraction():
    assert script_from_flores_code("eng_Latn") == "Latn"
    assert script_from_flores_code("hin_Deva") == "Deva"
    assert script_from_flores_code("plain") == ""


def test_fertility_for_sentences_is_positive():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tpw, tpc = fertility_for_sentences(tokenizer, _LATIN_SENTENCES)
    assert tpw > 0
    assert tpc > 0


def test_build_table_schema():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    sources = [
        ("eng_Latn", _LATIN_SENTENCES),
        ("hin_Deva", _DEVANAGARI_SENTENCES),
    ]
    table = build_table(tokenizer, sources)
    assert set(table.keys()) == {"per_language", "per_script"}
    assert "eng_Latn" in table["per_language"]
    assert "hin_Deva" in table["per_language"]
    eng = table["per_language"]["eng_Latn"]
    assert eng["script"] == "Latn"
    assert eng["tpw"] > 0
    assert "Latn" in table["per_script"]
    assert "Deva" in table["per_script"]


def test_devanagari_fertility_higher_than_latin():
    """Sanity-check the core script-tax phenomenon on a tiny corpus."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tpw_latin, _ = fertility_for_sentences(tokenizer, _LATIN_SENTENCES)
    tpw_deva, _ = fertility_for_sentences(tokenizer, _DEVANAGARI_SENTENCES)
    assert tpw_deva > tpw_latin, (
        "Devanagari should fragment into more tokens-per-word than Latin under "
        f"mBERT — got latin={tpw_latin:.3f}, devanagari={tpw_deva:.3f}"
    )
