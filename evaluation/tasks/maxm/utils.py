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

Used standard VQA scoring
- Normalized prediction and each ground truth
- Count the number of exact matches
- score = min(matches / 3, 1)
"""

import re
import string


# Normalization + helpers

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[" + re.escape(string.punctuation) + r"]")
_WHITESPACE = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    words = s.split()
    words = [_NUMBER_WORDS.get(w, w) for w in words]
    return _WHITESPACE.sub(" ", " ".join(words)).strip()

def vqa_score(prediction: str, processed_answers: list[str]) -> float:
    """Standard VQA score: min(matches / 3, 1.0)."""
    norm_pred = normalize_answer(prediction)
    matches = sum(1 for gt in processed_answers if normalize_answer(gt) == norm_pred)
    return min(matches / 3, 1.0)


# lm-eval task functions

def maxm_doc_to_image(doc):
    image = doc["image"]
    if isinstance(image, dict):
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image["bytes"]))
    return [image]

# Used to nudge the model towards short answers, and in the appropriate language
_SUFFIX = {
    "en": "Answer in 1-3 words.\nAnswer:",
    "fr": "Répondez en 1 à 3 mots.\nRéponse:",
    "hi": "1-3 शब्दों में उत्तर दें।\nउत्तर:",
    "th": "ตอบใน 1-3 คำ\nคำตอบ:",
    "zh": "用1-3个词回答。\n答案:",
    "iw": "ענה ב-1-3 מילים.\nתשובה:",
    "ro": "Răspundeți în 1-3 cuvinte.\nRăspuns:",
}
_DEFAULT_SUFFIX = "Answer in 1-3 words.\nAnswer:"


def maxm_doc_to_text(doc):
    lang = doc.get("language", "en")
    suffix = _SUFFIX.get(lang, _DEFAULT_SUFFIX)
    return f"<image>\n{doc['question']}\n{suffix}"

def maxm_doc_to_target(doc):
    return doc["answers"][0]

def maxm_process_results(doc, results):
    pred = results[0].strip()
    score = vqa_score(pred, doc.get("processed_answers", doc.get("answers", [])))
    return {"vqa_score": score}

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
