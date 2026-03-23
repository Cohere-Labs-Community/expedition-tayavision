"""MTVQA evaluation task utilities for lm-evaluation-harness.

MTVQA is a multilingual text-centric VQA benchmark with 9 languages:
AR, DE, FR, IT, JA, KO, RU, TH, VI.
Dataset: ByteDance/MTVQA

Fields:
    image: PIL Image
    id: str
    qa_pairs: str (stringified list of dicts with 'question' and 'answer')
    lang: str

Each sample can have multiple QA pairs. We use process_docs to flatten
them into individual evaluation instances.
"""

import ast

from datasets import Dataset


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


def _mtvqa_process_docs_lang(dataset, lang):
    """Flatten qa_pairs and keep only the specified language."""
    flattened = []
    for doc in dataset:
        if doc["lang"] != lang:
            continue
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


def mtvqa_process_docs_ar(dataset):
    return _mtvqa_process_docs_lang(dataset, "AR")


def mtvqa_process_docs_de(dataset):
    return _mtvqa_process_docs_lang(dataset, "DE")


def mtvqa_process_docs_fr(dataset):
    return _mtvqa_process_docs_lang(dataset, "FR")


def mtvqa_process_docs_it(dataset):
    return _mtvqa_process_docs_lang(dataset, "IT")


def mtvqa_process_docs_ja(dataset):
    return _mtvqa_process_docs_lang(dataset, "JA")


def mtvqa_process_docs_kr(dataset):
    return _mtvqa_process_docs_lang(dataset, "KR")


def mtvqa_process_docs_ru(dataset):
    return _mtvqa_process_docs_lang(dataset, "RU")


def mtvqa_process_docs_th(dataset):
    return _mtvqa_process_docs_lang(dataset, "TH")


def mtvqa_process_docs_vi(dataset):
    return _mtvqa_process_docs_lang(dataset, "VI")


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
