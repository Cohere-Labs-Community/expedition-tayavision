"""xMMMU evaluation task utilities for lm-evaluation-harness.

xMMMU (PangeaBench-xmmmu) is a cross-lingual multimodal understanding
benchmark with 7 languages: ar, en, fr, hi, id, ja, pt.
Dataset: neulab/PangeaBench-xmmmu

Parsing and evaluation logic adapted from the official MMMU repo:
https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/eval_utils.py

Fields:
    question: str (may contain <image N> placeholders)
    options: str (JSON-like list of option strings)
    answer: str (letter A-D for MCQ, free-form for open-ended)
    question_type: str ("multiple-choice" or "open")
    image_1 through image_7: PIL Image or None
"""

import ast
import random
import re


MULTI_CHOICE_PROMPT = "Answer with the option letter from the given choices directly."
OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."


# ---------------------------------------------------------------------------
# Option / prompt helpers
# ---------------------------------------------------------------------------

def _parse_options(doc):
    """Parse options field from string representation to list."""
    try:
        options = ast.literal_eval(doc["options"].replace("\n", " "))
    except (ValueError, SyntaxError):
        options = doc["options"]
    if isinstance(options, list):
        return options
    return []


def _format_options(options):
    """Format options as A. opt1\\nB. opt2\\n..."""
    if not options:
        return ""
    return "\n".join(
        f"{chr(ord('A') + i)}. {opt}"
        for i, opt in enumerate(options)
    )


def get_multi_choice_info(options):
    """Return index2ans dict and all_choices list from options."""
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        letter = chr(ord("A") + i)
        index2ans[letter] = option
        all_choices.append(letter)
    return index2ans, all_choices


# ---------------------------------------------------------------------------
# doc_to_image / doc_to_text
# ---------------------------------------------------------------------------

def xmmmu_doc_to_image(doc):
    """Extract all non-None images from the sample."""
    images = []
    for i in range(1, 8):
        img = doc.get(f"image_{i}")
        if img is not None:
            images.append(img)
    return images


def xmmmu_doc_to_text(doc):
    """Format the question with options as a prompt."""
    question = doc["question"]

    for i in range(1, 8):
        question = question.replace(f"<image {i}>", "<image>")

    options = _parse_options(doc)
    options_str = _format_options(options)

    if doc.get("question_type") == "multiple-choice":
        return f"{question}\n{options_str}\n{MULTI_CHOICE_PROMPT}"
    return f"{question}\n{OPEN_ENDED_PROMPT}"


# ---------------------------------------------------------------------------
# MCQ response parsing (from official MMMU eval)
# ---------------------------------------------------------------------------

def parse_multi_choice_response(response, all_choices, index2ans):
    """Parse predicted option letter from generated response."""
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []

    # Check for (A) style
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # Check for "A " style
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    # Check for "A." style
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # Content matching fallback
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(
                    response.lower().rfind(index2ans[can].lower())
                )
        pred_index = candidates[start_indexes.index(max(start_indexes))]
    else:
        pred_index = candidates[0]

    return pred_index


# ---------------------------------------------------------------------------
# Open-ended response parsing (from official MMMU eval)
# ---------------------------------------------------------------------------

def check_is_number(string):
    """Check if the given string is a number."""
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    """Normalize string to lower case; convert to float if numeric."""
    string = string.strip()
    if check_is_number(string):
        string = string.replace(",", "")
        string = round(float(string), 2)
        return [string]
    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]
    return [string]


def extract_numbers(string):
    """Extract all forms of numbers from a string."""
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)

    return numbers_with_commas + numbers_scientific + numbers_simple


def parse_open_response(response):
    """Parse prediction from generated response for open-ended questions."""
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ", "so ", "is ", "thus ", "therefore ",
            "final ", "answer ", "result ",
        ]
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.append("=")
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    tail = resp.split(indicator)[-1].strip()
                    if not shortest_key_response or len(tail) < len(shortest_key_response):
                        shortest_key_response = tail
            if shortest_key_response and shortest_key_response.strip() not in [
                ":", ",", ".", "!", "?", ";", ":", "'",
            ]:
                key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for item in pred_list:
        tmp_pred_list.extend(normalize_str(item))
    pred_list = list(set(tmp_pred_list))

    return pred_list


# ---------------------------------------------------------------------------
# Evaluation helpers (from official MMMU eval)
# ---------------------------------------------------------------------------

def eval_multi_choice(gold_i, pred_i):
    """Evaluate a multiple choice instance."""
    if isinstance(gold_i, list):
        return any(answer == pred_i for answer in gold_i)
    return gold_i == pred_i


def eval_open(gold_i, pred_i):
    """Evaluate an open question instance."""
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)

    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        else:
            if pred in norm_answers:
                return True
    return False


# ---------------------------------------------------------------------------
# lm-eval process_results
# ---------------------------------------------------------------------------

def xmmmu_process_results(doc, results):
    """Score model output using official MMMU parsing and evaluation."""
    pred = results[0].strip()
    gold = doc["answer"].strip()

    if doc.get("question_type") == "multiple-choice":
        options = _parse_options(doc)
        index2ans, all_choices = get_multi_choice_info(options)
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
        correct = eval_multi_choice(gold, parsed_pred)
    else:
        parsed_pred = parse_open_response(pred)
        correct = eval_open(gold, parsed_pred)

    return {"exact_match": float(correct)}


# ---------------------------------------------------------------------------
# xMMMU blind baseline
# ---------------------------------------------------------------------------

def xmmmu_blind_doc_to_text(doc):
    """Format question with options but without images for blind baseline."""
    question = doc["question"]

    for i in range(1, 8):
        question = question.replace(f"<image {i}>", "").strip()

    options = _parse_options(doc)
    options_str = _format_options(options)

    if doc.get("question_type") == "multiple-choice":
        return f"{question}\n{options_str}\n{MULTI_CHOICE_PROMPT}"
    return f"{question}\n{OPEN_ENDED_PROMPT}"
