"""m-ArenaHard LLM-as-judge evaluation pipeline.

Pairwise judging for m-ArenaHard following the methodology from:
- Üstün et al. (2024) — Aya model evaluation
- Arena-Hard-Auto (lmarena/arena-hard-auto)
- Aya Expanse paper (arXiv:2412.04261)

The paper uses GPT-4o (gpt-4o-2024-08-06) as judge. This implementation
uses Cohere Command A as judge via the Cohere API.

Dataset: CohereLabs/m-ArenaHard
Languages: ar, cs, de, el, en, es, fr, he, hi, id, it, ja, ko, nl,
           fa, pl, pt, ro, ru, tr, uk, vi, zh

Usage:
    # Step 1: Generate responses from both models
    python evaluation/m_arena_hard_judge.py generate \
        --model-name CohereLabs/tiny-aya-vision-global \
        --output-dir evaluation/results/m_arena_hard/tiny-aya-vision-global \
        --apply-chat-template

    python evaluation/m_arena_hard_judge.py generate \
        --model-name CohereLabs/tiny-aya-global \
        --output-dir evaluation/results/m_arena_hard/tiny-aya-global \
        --apply-chat-template

    # Step 2: Run pairwise judging
    export CO_API_KEY=your-cohere-api-key
    python evaluation/m_arena_hard_judge.py judge \
        --responses-a evaluation/results/m_arena_hard/tiny-aya-vision-global \
        --responses-b evaluation/results/m_arena_hard/tiny-aya-global \
        --model-a-name tiny-aya-vision-global \
        --model-b-name tiny-aya-global \
        --output-dir evaluation/results/m_arena_hard/vision-vs-text
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cohere
from datasets import load_dataset
from tqdm import tqdm


LANGUAGES = [
    "ar", "cs", "de", "el", "en", "es", "fr", "he", "hi",
    "id", "it", "ja", "ko", "nl", "fa", "pl", "pt", "ro",
    "ru", "tr", "uk", "vi", "zh",
]

# Exact judge prompt from lmarena/arena-hard-auto (utils/judge_utils.py)
# Used for all categories except creative_writing.
JUDGE_PROMPT = """\
Please act as an impartial judge and evaluate the quality of the responses \
provided by two AI assistants to the user prompt displayed below. You will be \
given assistant A's answer and assistant B's answer. Your job is to evaluate \
which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must \
provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with \
your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. \
Helpful means the answer correctly responds to the prompt or follows the \
instructions. Note when user prompt has any ambiguity or more than one \
interpretation, it is more helpful and appropriate to ask for clarifications \
or more information from the user than providing an answer based on assumptions. \
Relevant means all parts of the response closely connect or are appropriate to \
what is being asked. Concise means the response is clear and not verbose or \
excessive.

Then consider the creativity and novelty of the assistant's answers when needed. \
Finally, identify any missing important information in the assistants' answers \
that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following \
choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

JUDGE_USER_TEMPLATE = """\
<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""


def load_arena_hard(lang: str):
    """Load m-ArenaHard dataset for a specific language."""
    if lang not in LANGUAGES:
        raise ValueError(
            f"Language '{lang}' not in m-ArenaHard. "
            f"Available: {LANGUAGES}"
        )
    return load_dataset("CohereLabs/m-ArenaHard", lang, split="test")


def _parse_verdict(judge_output: str) -> str:
    """Extract verdict from judge output.

    Returns one of: 'A>>B', 'A>B', 'A=B', 'B>A', 'B>>A', or 'error'.
    """
    for verdict in ["A>>B", "A>B", "A=B", "B>A", "B>>A"]:
        if f"[[{verdict}]]" in judge_output:
            return verdict
    return "error"


def _flip_verdict(verdict: str) -> str:
    """Flip a verdict to account for position swap."""
    flip_map = {
        "A>>B": "B>>A",
        "A>B": "B>A",
        "A=B": "A=B",
        "B>A": "A>B",
        "B>>A": "A>>B",
        "error": "error",
    }
    return flip_map[verdict]


def _verdict_to_winner(verdict: str) -> str:
    """Map a 5-point verdict to a winner: 'A', 'B', or 'tie'."""
    if verdict in ("A>>B", "A>B"):
        return "A"
    elif verdict in ("B>A", "B>>A"):
        return "B"
    elif verdict == "A=B":
        return "tie"
    return "error"


def judge_pair(
    client: cohere.ClientV2,
    question: str,
    answer_a: str,
    answer_b: str,
    model: str = "command-a-03-2025",
) -> dict:
    """Send a pairwise comparison to the judge LLM.

    Returns:
        Dict with verdict and raw output.
    """
    user_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw_output = response.message.content[0].text
    verdict = _parse_verdict(raw_output)

    return {"verdict": verdict, "raw_output": raw_output}


def judge_with_swap(
    client: cohere.ClientV2,
    question: str,
    response_a: str,
    response_b: str,
    model: str = "command-a-03-2025",
) -> dict:
    """Two-game judging with position swap to reduce positional bias.

    Game 1: A=model_a, B=model_b
    Game 2: A=model_b, B=model_a (swapped)

    Returns:
        Dict with final_verdict, game1, and game2 results.
    """
    # Game 1: original order
    game1 = judge_pair(client, question, response_a, response_b, model)

    # Game 2: swapped order
    game2 = judge_pair(client, question, response_b, response_a, model)

    # Map game2 verdict back to original model perspective
    g1_verdict = game1["verdict"]
    g2_verdict = _flip_verdict(game2["verdict"])

    g1_winner = _verdict_to_winner(g1_verdict)
    g2_winner = _verdict_to_winner(g2_verdict)

    # Determine final verdict
    if g1_winner == g2_winner:
        final = g1_winner
    elif "error" in (g1_winner, g2_winner):
        final = g1_winner if g2_winner == "error" else g2_winner
    else:
        # Disagreement -> tie
        final = "tie"

    return {
        "final_verdict": final,
        "game1_verdict": g1_verdict,
        "game2_verdict": game2["verdict"],
        "game2_verdict_flipped": g2_verdict,
        "game1_raw": game1["raw_output"],
        "game2_raw": game2["raw_output"],
    }


def compute_win_rates(judgments: list[dict]) -> dict:
    """Compute win-rates from a list of judgment results.

    Args:
        judgments: List of dicts with 'final_verdict' key.

    Returns:
        Dict with win counts, rates, and error count.
    """
    counts = defaultdict(int)
    for j in judgments:
        counts[j["final_verdict"]] += 1

    total = counts["A"] + counts["B"] + counts["tie"]
    if total == 0:
        return {"win_a": 0, "win_b": 0, "tie": 0, "total": 0,
                "win_rate_a": 0.0, "win_rate_b": 0.0, "tie_rate": 0.0,
                "errors": counts["error"]}

    return {
        "win_a": counts["A"],
        "win_b": counts["B"],
        "tie": counts["tie"],
        "total": total,
        "win_rate_a": round(counts["A"] / total * 100, 2),
        "win_rate_b": round(counts["B"] / total * 100, 2),
        "tie_rate": round(counts["tie"] / total * 100, 2),
        "errors": counts["error"],
    }


def load_responses(path: str) -> dict:
    """Load pre-generated responses from a JSON file.

    Expected format: list of dicts with 'question_id' and 'model_response'.
    Returns a dict mapping question_id -> model_response.
    """
    with open(path) as f:
        data = json.load(f)
    return {item["question_id"]: item["model_response"] for item in data}


def run_judging(
    responses_a_path: str,
    responses_b_path: str,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    languages: list[str] | None = None,
    judge_model: str = "command-a-03-2025",
    output_dir: str = "evaluation/results/m_arena_hard",
    swap: bool = True,
):
    """Run the full m-ArenaHard judging pipeline.

    Args:
        responses_a_path: Directory with model A responses.
            Expected: {dir}/{lang}_responses.json
        responses_b_path: Directory with model B responses.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        languages: Languages to evaluate. Defaults to all 23.
        judge_model: Cohere model to use as judge.
        output_dir: Where to save results.
        swap: Whether to use two-game position swap debiasing.
    """
    client = cohere.ClientV2()
    languages = languages or LANGUAGES
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_judgments = {}
    all_stats = {}

    for lang in languages:
        resp_a_file = Path(responses_a_path) / f"{lang}_responses.json"
        resp_b_file = Path(responses_b_path) / f"{lang}_responses.json"

        if not resp_a_file.exists() or not resp_b_file.exists():
            print(f"Skipping {lang}: response files not found")
            continue

        responses_a = load_responses(str(resp_a_file))
        responses_b = load_responses(str(resp_b_file))

        dataset = load_arena_hard(lang)
        judgments = []

        for sample in tqdm(dataset, desc=f"Judging {lang}"):
            qid = sample["question_id"]
            if qid not in responses_a or qid not in responses_b:
                continue

            try:
                if swap:
                    result = judge_with_swap(
                        client, sample["prompt"],
                        responses_a[qid], responses_b[qid],
                        judge_model,
                    )
                else:
                    j = judge_pair(
                        client, sample["prompt"],
                        responses_a[qid], responses_b[qid],
                        judge_model,
                    )
                    result = {
                        "final_verdict": _verdict_to_winner(j["verdict"]),
                        "game1_verdict": j["verdict"],
                        "game1_raw": j["raw_output"],
                    }
            except Exception as e:
                print(f"  Error judging {qid}: {e}")
                result = {"final_verdict": "error", "error": str(e)}

            result["question_id"] = qid
            result["lang"] = lang
            judgments.append(result)

        stats = compute_win_rates(judgments)
        all_judgments[lang] = judgments
        all_stats[lang] = stats

        print(f"  {lang}: {model_a_name}={stats['win_rate_a']}% "
              f"{model_b_name}={stats['win_rate_b']}% "
              f"Tie={stats['tie_rate']}% "
              f"(n={stats['total']}, errors={stats['errors']})")

        # Save per-language results incrementally
        lang_file = output_path / f"{lang}_judgments.json"
        with open(lang_file, "w") as f:
            json.dump(judgments, f, ensure_ascii=False, indent=2)

    # Compute overall stats
    all_j = [j for lang_j in all_judgments.values() for j in lang_j]
    overall = compute_win_rates(all_j)

    summary = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "judge_model": judge_model,
        "swap_debiasing": swap,
        "overall": overall,
        "per_language": all_stats,
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nOverall: {model_a_name}={overall['win_rate_a']}% "
          f"{model_b_name}={overall['win_rate_b']}% "
          f"Tie={overall['tie_rate']}%")
    print(f"Results saved to {output_path}")

    return summary


def run_generation(
    model_name: str,
    output_dir: str,
    languages: list[str] | None = None,
    max_new_tokens: int = 512,
    apply_chat_template: bool = False,
):
    """Generate responses for all languages and save to disk.

    Args:
        model_name: HuggingFace model name.
        output_dir: Directory to save response files.
        languages: Languages to generate for. Defaults to all 23.
        max_new_tokens: Max tokens per response.
        apply_chat_template: Whether to wrap prompts in chat template.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    languages = languages or LANGUAGES
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for lang in languages:
        out_file = output_path / f"{lang}_responses.json"
        if out_file.exists():
            print(f"Skipping {lang}: {out_file} already exists")
            continue

        print(f"\nGenerating responses for {lang}...")
        dataset = load_arena_hard(lang)
        results = []

        for sample in tqdm(dataset, desc=f"Gen {lang}"):
            prompt = sample["prompt"]

            if apply_chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            response = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            results.append({
                "question_id": sample["question_id"],
                "prompt": sample["prompt"],
                "model_response": response,
            })

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} responses to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="m-ArenaHard: generate responses or run LLM-as-judge evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate model responses")
    gen_parser.add_argument("--model-name", required=True,
                            help="HuggingFace model name")
    gen_parser.add_argument("--output-dir", required=True,
                            help="Directory to save responses")
    gen_parser.add_argument("--languages", nargs="+", default=None,
                            help="Languages to evaluate (default: all 23)")
    gen_parser.add_argument("--max-new-tokens", type=int, default=512)
    gen_parser.add_argument("--apply-chat-template", action="store_true")

    # Judge subcommand
    judge_parser = subparsers.add_parser("judge", help="Run pairwise judging")
    judge_parser.add_argument("--responses-a", required=True,
                              help="Directory with model A responses")
    judge_parser.add_argument("--responses-b", required=True,
                              help="Directory with model B responses")
    judge_parser.add_argument("--model-a-name", default="model_a",
                              help="Name for model A (for reporting)")
    judge_parser.add_argument("--model-b-name", default="model_b",
                              help="Name for model B (for reporting)")
    judge_parser.add_argument("--languages", nargs="+", default=None)
    judge_parser.add_argument("--judge-model", default="command-a-03-2025")
    judge_parser.add_argument("--output-dir",
                              default="evaluation/results/m_arena_hard")
    judge_parser.add_argument("--no-swap", action="store_true",
                              help="Disable position swap debiasing")

    args = parser.parse_args()

    if args.command == "generate":
        run_generation(
            model_name=args.model_name,
            output_dir=args.output_dir,
            languages=args.languages,
            max_new_tokens=args.max_new_tokens,
            apply_chat_template=args.apply_chat_template,
        )
    elif args.command == "judge":
        run_judging(
            responses_a_path=args.responses_a,
            responses_b_path=args.responses_b,
            model_a_name=args.model_a_name,
            model_b_name=args.model_b_name,
            languages=args.languages,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
            swap=not args.no_swap,
        )


if __name__ == "__main__":
    main()