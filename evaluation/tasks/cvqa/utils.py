"""CVQA evaluation task utilities for lm-evaluation-harness.

CVQA is a culturally-diverse multilingual VQA benchmark with 31 languages.
Dataset: afaji/cvqa

Fields:
    image: PIL Image
    Question: str (in native language)
    Options: list[str] (4 answer options)
    Label: int (0-3, index of correct option)
"""

import re

from datasets import DatasetDict, load_dataset

OPTION_LETTERS = ["A", "B", "C", "D"]


def load_cvqa(cvqa_chunk_start=None, cvqa_chunk_end=None, **kwargs):
    """Load CVQA from Hugging Face datasets, optionally sliced for chunked eval."""
    has_chunk_bounds = cvqa_chunk_start is not None and cvqa_chunk_end is not None
    split = (
        f"test[{cvqa_chunk_start}:{cvqa_chunk_end}]"
        if has_chunk_bounds
        else "test"
    )
    return DatasetDict({"test": load_dataset("afaji/cvqa", split=split)})


def cvqa_doc_to_image(doc):
    """Return the image loaded by Hugging Face datasets."""
    return [doc["image"]]


def cvqa_doc_to_text(doc):
    """Format the question with options as a prompt."""
    question = doc["Question"]
    options = doc["Options"]

    options_str = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )

    return (
        f"<image>\n{question}\n{options_str}\n"
        "Respond with exactly one letter: A, B, C, or D."
    )


def cvqa_doc_to_target(doc):
    """Get the correct answer letter."""
    return OPTION_LETTERS[doc["Label"]]


def cvqa_process_results(doc, results):
    """Check if the model's answer matches the correct option letter."""
    pred = _extract_option_letter(results[0])
    gold = OPTION_LETTERS[doc["Label"]]

    return {"exact_match": float(pred == gold)}


def _extract_option_letter(response):
    """Extract explicit A/B/C/D answers without using arbitrary first letters."""
    text = str(response).strip()
    patterns = [
        r"^[\s*_`]*\(?([ABCD])\)?[\s*_`]*[\.:,-]?\s*$",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABCD])\)?\b",
        r"\bthe\s+answer\s+is\s*\(?([ABCD])\)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


# CVQA English translated task — uses English-translated questions and options for all samples
def cvqa_en_doc_to_text(doc):
    question = doc["Translated Question"]
    options = doc["Translated Options"]
    options_str = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"<image>\n{question}\n{options_str}\n"
        "Respond with exactly one letter: A, B, C, or D."
    )


# CVQA blind baseline utils
def cvqa_blind_doc_to_text(doc):
    return f"Question: {doc['Question']}\nAnswer:"

def cvqa_blind_doc_to_choice(doc):
    return [f" {opt}" for opt in doc["Options"]]

def cvqa_blind_doc_to_target(doc):
    return doc["Label"]
