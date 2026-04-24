"""CVQA evaluation task utilities for lm-evaluation-harness.

CVQA is a culturally-diverse multilingual VQA benchmark with 31 languages.
Dataset: afaji/cvqa

Fields:
    image: PIL Image
    Question: str (in native language)
    Options: list[str] (4 answer options)
    Label: int (0-3, index of correct option)
"""

OPTION_LETTERS = ["A", "B", "C", "D"]


def cvqa_doc_to_image(doc):
    """Extract image(s) from dataset sample."""
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
        "Answer with the option letter (A, B, C, or D)."
    )


def cvqa_doc_to_target(doc):
    """Get the correct answer letter."""
    return OPTION_LETTERS[doc["Label"]]


def cvqa_process_results(doc, results):
    """Check if the model's answer matches the correct option letter."""
    pred = results[0].strip().upper()
    gold = OPTION_LETTERS[doc["Label"]]

    # Extract just the first letter if the model outputs more
    if pred and pred[0] in OPTION_LETTERS:
        pred = pred[0]

    return {"exact_match": float(pred == gold)}


# CVQA English translated task — uses English-translated questions and options for all samples
def cvqa_en_doc_to_text(doc):
    question = doc["Translated Question"]
    options = doc["Translated Options"]
    options_str = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"<image>\n{question}\n{options_str}\n"
        "Answer with the option letter (A, B, C, or D)."
    )


# CVQA blind baseline utils
def cvqa_blind_doc_to_text(doc):
    return f"Question: {doc['Question']}\nAnswer:"

def cvqa_blind_doc_to_choice(doc):
    return [f" {opt}" for opt in doc["Options"]]

def cvqa_blind_doc_to_target(doc):
    return doc["Label"]
