"""MTVQA evaluation task utilities for lm-eval

MTVQA (Multilingual Text-Centric Visual Question Answering)
ByteDance/MTVQA

Exact matching:
- Normalized prediction and each ground truth
- Count the number of exact matches
- score = min(matches / n, 1) where n is the number of ground truths
"""

import ast
from datasets import Dataset

# QA pair parsing
def parse_qa_pairs(s):
    """Parse qa_pairs string (single-quoted Python literal) into a list of dicts."""
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []


def make_process_docs(target_lang: str):
    """
    Factory that returns a process_docs function for a specific language.
        - lm-eval's process_docs hook only accepts (dataset)
    """
    def process_docs(dataset):
        """
        Filter to target_lang and flatten each image's qa_pairs into one doc per question.
            Stores n pairs on each doc for per-image scoring.
        """
        rows = []
        for doc in dataset:
            if str(doc.get("lang", "")).upper() != target_lang.upper():
                continue
            qa_list = parse_qa_pairs(doc["qa_pairs"])
            n = len(qa_list)
            if n == 0:
                continue
            for qa in qa_list:
                rows.append({
                    "image": doc["image"],
                    "id": doc["id"],
                    "lang": doc["lang"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "n": n,
                })
        return Dataset.from_list(rows)
    return process_docs


process_docs_ar = make_process_docs("AR")
process_docs_de = make_process_docs("DE")
process_docs_fr = make_process_docs("FR")
process_docs_it = make_process_docs("IT")
process_docs_ja = make_process_docs("JA")
process_docs_kr = make_process_docs("KR")
process_docs_ru = make_process_docs("RU")
process_docs_th = make_process_docs("TH")
process_docs_vi = make_process_docs("VI")


# doc_to_image
def mtvqa_doc_to_image(doc):
    image = doc["image"]
    if isinstance(image, dict):
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image["bytes"]))
    return [image]


# suffixes for nudge towards short answers in the appropriate language
DEFAULT_SUFFIX = "Answer in 1-3 words.\nAnswer:"
SUFFIX = {
    "AR": "أجب في 1-3 كلمات.\nالإجابة:",
    "DE": "Antworte in 1-3 Wörtern.\nAntwort:",
    "FR": "Répondez en 1 à 3 mots.\nRéponse:",
    "IT": "Rispondi in 1-3 parole.\nRisposta:",
    "JA": "1〜3語で答えてください。\n答え:",
    "KR": "1-3 단어로 답하세요.\n답:",
    "RU": "Ответьте в 1-3 словах.\nОтвет:",
    "TH": "ตอบใน 1-3 คำ\nคำตอบ:",
    "VI": "Trả lời trong 1-3 từ.\nTrả lời:",
}

def mtvqa_doc_to_text(doc):
    lang = str(doc.get("lang", "")).upper()
    suffix = SUFFIX.get(lang, DEFAULT_SUFFIX)
    return f"<image>\n{doc['question']}\n{suffix}"


def mtvqa_doc_to_target(doc):
    return doc["answer"]


# scoring - per-image exact match
def mtvqa_process_results(doc, results):
    """Return (image_id, is_correct, n) tuple for custom aggregation"""
    pred = results[0].strip().lower()
    answer = doc["answer"].strip().lower()
    is_correct = int(pred == answer)
    return {"mtvqa_score": (doc["id"], is_correct, doc["n"])}


def mtvqa_aggregation(items):
    """Aggregate per-QA results into per-image scores"""
    from collections import defaultdict
    image_data = defaultdict(lambda: {"correct": 0, "n": 0})
    for image_id, is_correct, n in items:
        image_data[image_id]["correct"] += is_correct
        image_data[image_id]["n"] = n  # constant per image
    if not image_data:
        return 0.0
    scores = [min(v["correct"] / v["n"], 1.0) for v in image_data.values()]
    return sum(scores) / len(scores)


def mtvqa_process_docs(dataset):
    """Flatten qa_pairs so each QA pair becomes its own evaluation doc."""
    flattened = []
    for doc in dataset:
        qa_pairs = ast.literal_eval(doc["qa_pairs"])
        for i, pair in enumerate(qa_pairs):
            flattened.append(
                {
                    "id": f"{doc['id']}_{i}",
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "lang": doc["lang"],
                }
            )
    return Dataset.from_list(flattened)


def mtvqa_blind_doc_to_text(doc):
    """Format question as a text-only prompt."""
    return f"Question: {doc['question']}\nAnswer:"


def mtvqa_blind_doc_to_target(doc):
    """Return the ground-truth answer."""
    return doc["answer"]


def mtvqa_blind_process_results(doc, results):
    """Check if the model's answer matches the ground truth (normalized)."""
    pred = results[0].strip().lower()
    gold = doc["answer"].strip().lower()

    return {"exact_match": float(pred == gold)}
