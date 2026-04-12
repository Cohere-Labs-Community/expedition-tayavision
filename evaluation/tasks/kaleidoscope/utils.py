"""Kaleidoscope evaluation task utilities for lm-evaluation-harness.

Kaleidoscope is a multilingual MCQA benchmark with 18 languages
sourced from real-world exams.
Dataset: CohereLabs/kaleidoscope

Fields:
    question: str (in native language)
    options: list[str] (4 answer options)
    answer: int (0-3, index of correct option)
    image: str (path to image, may be empty)
    image_information: str ('essential', 'supplementary', etc.)
    language: str
"""

OPTION_LETTERS = ["A", "B", "C", "D"]


# ── Visual task utils ──────────────────────────────────────────────


def kaleidoscope_doc_to_text(doc):
    """Format the question with options as a prompt."""
    question = doc["question"]
    options = doc["options"]

    options_str = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )

    return (
        f"{question}\n{options_str}\n"
        "Answer with the option letter (A, B, C, or D)."
    )


def kaleidoscope_doc_to_target(doc):
    """Get the correct answer letter."""
    return OPTION_LETTERS[doc["answer"]]


def kaleidoscope_process_results(doc, results):
    """Check if the model's answer matches the correct option letter."""
    pred = results[0].strip().upper()
    gold = OPTION_LETTERS[doc["answer"]]

    if pred and pred[0] in OPTION_LETTERS:
        pred = pred[0]

    return {"exact_match": float(pred == gold)}


# ── Blind baseline utils ──────────────────────────────────────────


def kaleidoscope_blind_doc_to_text(doc):
    return f"Question: {doc['question']}\nAnswer:"


def kaleidoscope_blind_doc_to_choice(doc):
    return [f" {opt}" for opt in doc["options"]]


def kaleidoscope_blind_doc_to_target(doc):
    return doc["answer"]


# ── Per-language filtering ────────────────────────────────────────


def _kaleidoscope_filter_lang(dataset, lang):
    """Filter dataset to keep only the specified language."""
    return dataset.filter(lambda x: x["language"] == lang)


def kaleidoscope_process_docs_ar(dataset):
    return _kaleidoscope_filter_lang(dataset, "ar")


def kaleidoscope_process_docs_bn(dataset):
    return _kaleidoscope_filter_lang(dataset, "bn")


def kaleidoscope_process_docs_de(dataset):
    return _kaleidoscope_filter_lang(dataset, "de")


def kaleidoscope_process_docs_en(dataset):
    return _kaleidoscope_filter_lang(dataset, "en")


def kaleidoscope_process_docs_es(dataset):
    return _kaleidoscope_filter_lang(dataset, "es")


def kaleidoscope_process_docs_fa(dataset):
    return _kaleidoscope_filter_lang(dataset, "fa")


def kaleidoscope_process_docs_fr(dataset):
    return _kaleidoscope_filter_lang(dataset, "fr")


def kaleidoscope_process_docs_hi(dataset):
    return _kaleidoscope_filter_lang(dataset, "hi")


def kaleidoscope_process_docs_hr(dataset):
    return _kaleidoscope_filter_lang(dataset, "hr")


def kaleidoscope_process_docs_hu(dataset):
    return _kaleidoscope_filter_lang(dataset, "hu")


def kaleidoscope_process_docs_lt(dataset):
    return _kaleidoscope_filter_lang(dataset, "lt")


def kaleidoscope_process_docs_ne(dataset):
    return _kaleidoscope_filter_lang(dataset, "ne")


def kaleidoscope_process_docs_nl(dataset):
    return _kaleidoscope_filter_lang(dataset, "nl")


def kaleidoscope_process_docs_pt(dataset):
    return _kaleidoscope_filter_lang(dataset, "pt")


def kaleidoscope_process_docs_ru(dataset):
    return _kaleidoscope_filter_lang(dataset, "ru")


def kaleidoscope_process_docs_sr(dataset):
    return _kaleidoscope_filter_lang(dataset, "sr")


def kaleidoscope_process_docs_te(dataset):
    return _kaleidoscope_filter_lang(dataset, "te")


def kaleidoscope_process_docs_uk(dataset):
    return _kaleidoscope_filter_lang(dataset, "uk")
