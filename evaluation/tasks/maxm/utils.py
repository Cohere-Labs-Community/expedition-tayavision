"""MaXM evaluation task utilities for lm-evaluation-harness.

MaXM is a multilingual open-ended VQA benchmark with 7 languages.
Dataset: floschne/maxm

Splits are per-language: en, fr, hi, iw, ro, th, zh

Fields:
    question: str
    answers: list[str] (multiple valid answers)
    processed_answers: list[str] (expanded/normalized answers)
    image: dict (encoded image bytes)
    language: str
"""


def maxm_blind_doc_to_text(doc):
    """Format question as a text-only prompt."""
    return f"Question: {doc['question']}\nAnswer:"


def maxm_blind_doc_to_target(doc):
    """Return the first valid answer as target."""
    return doc["answers"][0]


def maxm_blind_process_results(doc, results):
    """Check if the model's answer matches any of the valid answers."""
    pred = results[0].strip().lower()
    valid_answers = [a.strip().lower() for a in doc["answers"]]

    return {"exact_match": float(pred in valid_answers)}
