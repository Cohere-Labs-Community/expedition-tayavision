# External Model Baselines

This document describes the external open-weight VLMs used as baselines for comparing against Tiny Aya Vision on CVQA and xMMMU.

The baseline registry is defined in [`config/evaluation/baselines.yaml`](../config/evaluation/baselines.yaml).

---

## Baseline Models

### Gemma 3 4B (`google/gemma-3-4b-it`)

- **Author:** Google DeepMind
- **Params:** 4B
- **License:** [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Context window:** 128K tokens
- **Multilingual:** 140+ languages
- **Vision input:** Images normalized to 896×896, encoded to 256 tokens each
- **Dtype:** `bfloat16`

The smallest multimodal variant in the Gemma 3 family. The 1B variant is text-only; vision support starts at 4B.

---

### Ministral 3 3B (`mistralai/Ministral-3-3B-Instruct-2512`)

- **Author:** Mistral AI
- **Params:** 3.4B LM + 0.4B vision encoder
- **License:** Apache 2.0
- **Context window:** 256K tokens
- **Multilingual:** English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic
- **Weights format:** FP8 — loaded with `dtype: auto` to avoid double memory from dequantization
- **VRAM:** Fits in 8GB (FP8), less with further quantization

Edge-optimized multimodal model from the Ministral 3 family. Instruct variant is post-trained for instruction following, chat, and agentic tasks with native function calling.

> **Note:** Weights are stored in FP8. Use `dtype=auto` (not `bfloat16`) when loading — passing `bfloat16` forces dequantization and doubles memory usage.

---

### SmolVLM (`HuggingFaceTB/SmolVLM-Instruct`)

- **Author:** Hugging Face
- **Params:** ~2B (SmolLM2 1.7B backbone + SigLIP-so400m vision encoder)
- **License:** Apache 2.0
- **Language:** **English only**
- **Vision input:** 81 visual tokens per 384×384 patch; larger images split into patches
- **Dtype:** `bfloat16`

Compact multimodal model built on SmolLM2 + SigLIP. Uses aggressive image compression for fast inference with low memory footprint (5GB VRAM).

> **Limitation:** SmolVLM is English-only. Multilingual results on CVQA and xMMMU should be interpreted with this in mind — scores on non-English splits reflect the model's cross-lingual zero-shot transfer, not native multilingual support.

---

### Qwen3-VL-2B (`Qwen/Qwen3-VL-2B-Instruct`)

- **Author:** Alibaba Qwen Team
- **Params:** 2B
- **License:** Apache 2.0
- **Context window:** 256K native, expandable to 1M
- **Multilingual OCR:** 32 languages
- **Architecture:** Interleaved-MRoPE + DeepStack multi-level ViT feature fusion
- **Dtype:** `bfloat16`
- **trust_remote_code:** required (`true`)

The smallest model in the Qwen3-VL family. Strong multimodal reasoning, video understanding, and spatial perception. Requires `trust_remote_code=True` to load.

---

## Summary

| Model | HuggingFace ID | Params | Multilingual | Notes |
|-------|---------------|--------|:------------:|-------|
| Gemma 3 4B | `google/gemma-3-4b-it` | 4B | 140+ langs | Smallest multimodal Gemma 3 |
| Ministral 3 3B | `mistralai/Ministral-3-3B-Instruct-2512` | 3.4B+0.4B | 11 langs | FP8 weights, use `dtype=auto` |
| SmolVLM | `HuggingFaceTB/SmolVLM-Instruct` | ~2B | English only | Low VRAM (5GB) |
| Qwen3-VL-2B | `Qwen/Qwen3-VL-2B-Instruct` | 2B | 32 langs (OCR) | Requires `trust_remote_code` |

---

## Running Baselines

### Run all baselines on all tasks

```bash
python evaluation/run_baselines.py
```

### Quick smoke test (10 samples per task)

```bash
python evaluation/run_baselines.py --limit 10
```

### Run a specific model only

```bash
python evaluation/run_baselines.py --models google/gemma-3-4b-it
```

### Run a specific task only

```bash
python evaluation/run_baselines.py --tasks cvqa_blind
```

Results are saved to `evaluation/results/<model_slug>/` (e.g. `evaluation/results/google__gemma-3-4b-it/cvqa_blind_results.json`).

---

## Comparing Results

Once results are available, generate a comparison table across all models:

```bash
python evaluation/compare_results.py
```

To compare on specific tasks only:

```bash
python evaluation/compare_results.py --tasks cvqa_blind xmmmu
```

---

## Running a Single Baseline Manually

```bash
python evaluation/run_eval.py \
  --task cvqa_blind \
  --model-name google/gemma-3-4b-it \
  --backend hf-multimodal \
  --dtype bfloat16 \
  --skip-registration \
  --apply-chat-template \
  --no-trust-remote-code \
  --log-samples \
  --output-dir evaluation/results/google__gemma-3-4b-it
```

Key flags for external models:

| Flag | Purpose |
|------|---------|
| `--skip-registration` | Skips TinyAyaVision Auto class import (not needed for external models) |
| `--dtype` | Set to `auto` for Ministral (FP8), `bfloat16` for others |
| `--no-trust-remote-code` | Disable for models that don't require it (Gemma, Ministral, SmolVLM) |
| `--trust-remote-code` | Required for Qwen3-VL (default: `true`) |
