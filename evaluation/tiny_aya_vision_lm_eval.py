"""Custom lm-eval backend for TinyAyaVision

Extends on HFMultimodalLM but overrides _create_tokenizer to load
TinyAyaVisionProcessor directly, bypassing AutoProcessor (which requires ProcessorMixin registration)

Registered as the "tiny-aya-vision" for run_eval.py
"""

import ast
import hashlib
import json
import logging
import os
import time

import transformers
from lm_eval.api.registry import register_model
from lm_eval.models.hf_vlms import HFMultimodalLM

import models  # registers TinyAyaVision with HF Auto classes
from config.model_config import TinyAyaVisionConfig
from src.processing import TinyAyaVisionProcessor
from evaluation.tasks.maxm.utils import vqa_score

logger = logging.getLogger(__name__)

CHECKPOINT_BATCH_SIZE = 100  # save to disk / log every N requests

def request_key(request) -> str:
    """Stable hash key for a request based on context text + gen_kwargs."""
    context = request.args[0] if request.args else ""
    gen_kwargs = str(request.args[1]) if len(request.args) > 1 else ""
    return hashlib.md5((context + gen_kwargs).encode()).hexdigest()


# Subclasses HFMultimodalLM (lm-eval's multimodal backend) to override
@register_model("tiny-aya-vision")
class TinyAyaVisionLM(HFMultimodalLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def _create_tokenizer(
        self,
        pretrained,
        tokenizer,
        revision="main",
        trust_remote_code=False,
        **kwargs,
    ):
        # Load TinyAyaVisionProcessor directly
        model_path = pretrained if isinstance(pretrained, str) else self.model.name_or_path
        vlm_config = TinyAyaVisionConfig.from_pretrained(model_path)
        self.processor = TinyAyaVisionProcessor(vlm_config)
        self.tokenizer = self.processor.tokenizer

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Patch the processor's image token ID from the saved model config (261010)
        instead of processor's default (261008) so image tokens are correctly located during generation."""
        processor_token_id = self.processor.image_token_id
        if self.model.image_token_id != processor_token_id:
            logger.info(
                f"Patching model.image_token_id: {self.model.image_token_id} → "
                f"{processor_token_id} (to match processor tokenizer)"
            )
            self.model._image_token_id = processor_token_id

        # Resolve checkpoint file path from env var (set by modal script).
        checkpoint_dir = os.environ.get("TAYA_CHECKPOINT_DIR")
        checkpoint_file = os.path.join(checkpoint_dir, "responses.jsonl") if checkpoint_dir else None

        # Load any previously completed responses so we can skip them.
        completed: dict[str, str] = {}
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            completed[item["key"]] = item["response"]
                        except json.JSONDecodeError:
                            pass
            logger.info(f"Resumed from checkpoint: {len(completed)} responses already done")

        # Split requests into already-done (from checkpoint) and pending.
        responses: list[str | None] = [None] * len(requests)
        pending_indices: list[int] = []
        for i, req in enumerate(requests):
            key = request_key(req)
            if key in completed:
                responses[i] = completed[key]
            else:
                pending_indices.append(i)

        n_skip = len(requests) - len(pending_indices)
        logger.info(
            f"Requests: {len(requests)} total | {n_skip} from checkpoint | {len(pending_indices)} pending"
        )

        if not pending_indices:
            logger.info("All requests already completed — returning checkpoint results")
            return responses

        # Run pending requests in batches, writing to checkpoint after each batch.
        start_time = time.time()
        n_done = 0

        for batch_start in range(0, len(pending_indices), CHECKPOINT_BATCH_SIZE):
            batch_end = min(batch_start + CHECKPOINT_BATCH_SIZE, len(pending_indices))
            batch_orig_indices = pending_indices[batch_start:batch_end]
            batch_requests = [requests[i] for i in batch_orig_indices]

            batch_responses = super().generate_until(batch_requests, disable_tqdm=True)

            # Write each response to the checkpoint file with task-specific fields.
            if checkpoint_file:
                with open(checkpoint_file, "a") as f:
                    for req, resp in zip(batch_requests, batch_responses):
                        doc = req.doc
                        if "processed_answers" in doc:
                            
                            # MaXM format
                            entry = {
                                "key": request_key(req),
                                "question_id": doc.get("question_id", ""),
                                "question": doc.get("question", ""),
                                "answers": doc.get("answers", []),
                                "language": doc.get("language", ""),
                                "response": resp,
                                "vqa_score": vqa_score(resp, doc.get("processed_answers", [])),
                            }
                        elif "topic_difficulty" in doc:
                            # xMMMU format
                            try:
                                options_raw = ast.literal_eval(doc.get("options", "[]"))
                            except Exception:
                                options_raw = []
                            entry = {
                                "key": request_key(req),
                                "id": doc.get("id", ""),
                                "question": doc.get("question", ""),
                                "options": {k: v for k, v in zip(["A", "B", "C", "D"], options_raw)},
                                "correct": doc.get("answer", ""),
                                "response": resp,
                                "topic_difficulty": doc.get("topic_difficulty", ""),
                                "subfield": doc.get("subfield", ""),
                            }
                        elif "lang" in doc and "answer" in doc:
                            # MTVQA format
                            answer = doc.get("answer", "")
                            is_correct = int(resp.strip().lower() == answer.strip().lower())
                            entry = {
                                "key": request_key(req),
                                "id": doc.get("id", ""),
                                "lang": doc.get("lang", ""),
                                "n": doc.get("n", 1),
                                "question": doc.get("question", ""),
                                "answer": answer,
                                "response": resp,
                                "exact_match": is_correct,
                            }
                        else:
                            # CVQA format
                            label = doc.get("Label", 0)
                            options = doc.get("Translated Options") or doc.get("Options", [])
                            entry = {
                                "key": request_key(req),
                                "language": str(doc.get("Subset", "unknown")),
                                "question": doc.get("Translated Question") or doc.get("Question", ""),
                                "options": {k: v for k, v in zip(["A", "B", "C", "D"], options)},
                                "response": resp,
                                "correct": ["A", "B", "C", "D"][label],
                            }
                        f.write(json.dumps(entry) + "\n")

            for orig_idx, resp in zip(batch_orig_indices, batch_responses):
                responses[orig_idx] = resp

            n_done += len(batch_responses)
            elapsed = time.time() - start_time
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = len(pending_indices) - n_done
            eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
            total_done = n_skip + n_done
            logger.info(
                f"Progress: {total_done}/{len(requests)} ({100 * total_done / len(requests):.1f}%)"
                f" | {rate:.2f} it/s | ETA: {eta_min:.1f} min"
            )

        return responses
