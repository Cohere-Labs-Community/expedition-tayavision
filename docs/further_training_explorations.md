# Further Training Explorations for Tiny Aya Vision

## Current State

| Stage | What's trained | Data | Status |
|-------|---------------|------|--------|
| **Phase 1 — Alignment** | MLP projector only (vision encoder + LLM frozen) | LLaVA-Pretrain 558K image-caption pairs | Done |
| **Phase 2 — Instruction Tuning** | MLP projector + LoRA on upper LLM layers (18–35) | LLaVA-Instruct-150K | Done |

**Architecture:** SigLIP2-so400m (frozen, 400M) → Pixel Shuffle + SwiGLU MLP (~11.5M) → Tiny Aya Base / Global (Cohere2, 3.35B)

---

## 1. Data-Centric Improvements

### 1.1 Richer Pre-training Captions (Replace or Augment LLaVA-Pretrain)

The LLaVA-Pretrain dataset (558K BLIP-generated captions) is short (~11 words avg) and noisy. Multiple SoTA works show that **higher-quality, denser captions** in stage 1 dramatically improve downstream performance.

**Explorations:**
- **Re-caption with a stronger model.** Use LLaVA-NeXT-34B, InternVL2, or GPT-4o to re-caption the existing 558K images with detailed descriptions (100–200 words). LLaVA-OneVision's "Re-Captioned Detailed Description Data" re-captions COCO118K + BLIP558K + CC3M and gets 3.5M high-quality samples. Even re-captioning just the 558K images yields major gains.
- **Human-annotated dense captions (Molmo/PixMo-Cap approach).** Ai2's Molmo paper shows that training on 712K images with 200+ word human-spoken captions (PixMo-Cap) produces stronger models than distilled captions from GPT-4V. Their key insight: asking annotators to *speak* descriptions for 60–90 seconds produces far more detailed descriptions than writing. If budget permits, even 50–100K such captions on diverse images could help.
- **Length-conditioned captioning.** Molmo trains the model to generate captions of a specified length using length hints. This improves both captioning quality *and* downstream benchmark scores, serving as a better pre-training task than unconditional captioning.
- **ShareGPT4V / ShareGPT4o.** Use the publicly available ShareGPT4V (100K detailed captions by GPT-4V) or ShareGPT4o datasets. Caveat: Molmo found ShareGPT4V underperforms their own PixMo-Cap data at equal scale due to less image diversity.

**Priority: HIGH** — Molmo ablations show scaling PixMo-Cap from 0→712K images yields continuous improvement on both captioning F1 and 11-benchmark average.

### 1.2 Expanded Instruction Tuning Data Mix

LLaVA-Instruct-150K is a good starting point but is now considered small. SoTA models use 1–4M instruction samples across diverse categories.

**Explorations:**
- **Academic VQA datasets.** Add VQAv2 (83K), TextVQA (35K), OK-VQA (9K), A-OKVQA (17K), TallyQA (250K), ScienceQA (5K), GQA (72K), Visual Genome (86K). Molmo uses all of these in their fine-tuning mix.
- **Document/Chart/OCR data.** Add DocVQA (39K), ChartQA (28K), InfographicVQA (24K), AI2D (15K), DVQA (20K), ST-VQA (25K), FigureQA, PlotQA. These are critical for structured-data understanding. Molmo and LLaVA-OneVision both emphasize this category.
- **Synthetic document data (PixMo-Docs approach).** Generate charts, tables, diagrams, and documents programmatically using Matplotlib, Plotly, LaTeX, HTML, Vega-Lite, Mermaid, Graphviz. Use an LLM to generate code that renders images, then generate QA pairs from the code (not the image). Molmo generated 255K such images with 2.3M QA pairs.
- **Synthetic skill-specific data.**
  - Clock reading (PixMo-Clocks): render synthetic clocks with diverse watch faces (826K). Molmo achieved clock-reading accuracy 30× higher than GPT-4V.
  - Counting (PixMo-Count): use object detectors on web images to create counting QA with point annotations (36K).
- **Multilingual data.** Since Tiny Aya is a multilingual LLM, add multilingual VQA/caption data. LLaVA-OneVision adds 92K Chinese captions. Consider xGQA, MaXM, MTVQA, or captioning data in target languages.
- **Conversation/chat data.** Add LLaVA-Wild, ShareGPT4V/4o, ALLaVA-Instruct for free-form conversational ability.
- **Math/reasoning data.** Add Geo170K, MAVIS, MathV360K, TabMWP, UniGeo, CLEVR-Math to improve mathematical visual reasoning — an area where even Molmo-72B lags.

**Data mixing strategy (from Molmo):**
- Sample datasets at rates proportional to **sqrt(dataset_size)**.
- Manually down-weight very large synthetic datasets (PlotQA, FigureQA, DVQA).
- Up-weight tasks that learn slowly (e.g., pointing/grounding tasks).
- Use **style tags** (e.g., prefix questions with "vqa2:" or "docvqa:") to prevent short-answer academic styles from contaminating conversational responses.

**Priority: HIGH** — This is the single biggest lever. Molmo shows academic data alone gets 72.5% on their 11-avg; adding PixMo data pushes it to 76.9%.

### 1.3 Multi-Annotated Image Training

Many datasets have multiple QA pairs per image (e.g., VQAv2 has ~5 questions per image). Molmo's key efficiency trick:

- Concatenate all text annotations for one image into a single long sequence.
- Mask attention so each annotation attends to image tokens and its own tokens, but not to other annotations.
- This avoids redundant image encoding, reducing processed images by 2/3 and training time by over half, with only 25% increase in sequence length.

**Priority: MEDIUM** — Significant training efficiency gain for multi-annotation datasets.

### 1.4 Text-Only Data Mixing

Training exclusively on multimodal data can degrade text-only capabilities. Molmo showed that their Qwen2-7B LLM lost knowledge across text benchmarks after VLM training.

**Explorations:**
- **Mix in Tulu 3 SFT data** during instruction tuning. Molmo found that 10% down-sampled Tulu 3 improved both text-only benchmarks *and* the multimodal 11-avg (76.9 → 77.1).
- **Magpie Pro data.** LLaVA-OneVision uses 450K Magpie Pro samples (from Llama-3 and Qwen2) as pure text data mixed into single-image training — 14.3% of their 3.2M mix.
- Consider **Aya Dataset** or **Aya Collection** text-only data to preserve the model's multilingual text capabilities.

**Priority: MEDIUM** — especially important given Tiny Aya's multilingual strength that you don't want to lose.

---

## 2. Training Pipeline Improvements

### 2.1 Eliminate the Separate Alignment Stage

Molmo's key finding: **the separate connector-only pre-training stage is unnecessary** when pre-training on high-quality dense captions. Instead:

- Train all parameters (vision encoder, connector, LLM) in a single pre-training stage on dense captions.
- Give the connector a **higher learning rate** (e.g., 2e-4 vs 2e-5 for LLM) with a **shorter warmup** (200 steps vs 2000 steps) so it can adapt quickly.
- This eliminates one stage, reduces training time, and avoids the noisy web-scale data typically used in alignment.

**Priority: MEDIUM** — simplifies pipeline if you switch to dense caption pre-training data.

### 2.2 Unfreeze the Vision Encoder

In both your current stages, SigLIP2 is frozen. SoTA works unfreeze it during instruction tuning:

- Molmo trains vision encoder at a **lower LR** (6e-6 in pre-training, 5e-6 in fine-tuning) vs the LLM (2e-5 / 1e-5).
- LLaVA-OneVision uses LR for vision encoder = LR_LLM / 5.
- Unfreezing lets the vision encoder adapt its features to the specific tasks, especially for OCR, document understanding, and fine-grained tasks.
- Keep the vision encoder LR 3–5× smaller than LLM LR to prevent catastrophic forgetting of pre-trained visual features.

**Priority: HIGH** — almost all top VLMs now fine-tune the vision encoder. SigLIP2 features are good but can be improved for your specific downstream task distribution.

### 2.3 Stage 1.5: High-Quality Knowledge Learning

LLaVA-OneVision introduces a **Stage 1.5** between alignment and instruction tuning:

- Train on high-quality knowledge data using the same config as Stage 2 (full model trainable).
- Data: re-captioned detailed descriptions (3.5M), document/OCR data (1.1M), multilingual captions (92K Chinese), language understanding data (143K).
- This injects new knowledge before the model sees instruction-following formats.

**Priority: MEDIUM** — useful if you have quality knowledge data that doesn't fit neatly into instruction format.

### 2.4 Curriculum Learning / Multi-Stage Instruction Tuning

LLaVA-OneVision's 4-stage curriculum:

| Stage | Data | Resolution | Trainable |
|-------|------|-----------|-----------|
| 1 — Alignment | 558K captions | Base (384px) | Projector only |
| 1.5 — Knowledge | 4M knowledge | AnyRes (up to 5×) | Full model |
| 2 — Single-Image SFT | 3.2M instructions | AnyRes (up to 10×) | Full model |
| 2 — OneVision SFT | 1.6M mixed (image + video + multi-image) | AnyRes (up to 10×) | Full model |

Key insights:
- **Progressive resolution scaling**: start at base resolution, gradually increase. This is cheaper and more stable.
- **Single-image first, then multi-modal**: build strong single-image capabilities first, then extend to multi-image and video. This enables cross-scenario transfer.
- **Max sequence length increases across stages**: 729 tokens → 3645 → 7290.

**Priority: MEDIUM** — the progressive resolution and curriculum approach is especially useful if you want to support higher resolutions or video.

### 2.5 Full Fine-Tuning vs LoRA

Your Phase 2 uses LoRA (rank 256, upper half only). Consider:

- **Full fine-tuning of the LLM** if compute allows. LLaVA-OneVision and Molmo both fully fine-tune the LLM during instruction tuning. This generally yields better results than LoRA at the cost of more memory.
- **If sticking with LoRA**: consider applying LoRA to **all layers** (not just upper half). The lower layers encode multilingual knowledge which may also need adaptation for visual grounding.
- **QLoRA** (4-bit quantized base + LoRA) can enable full-layer LoRA within the same memory budget as upper-half LoRA on the full-precision model.
- **Differential LR for LoRA A and B matrices.** Your config already supports `lora_a_lr_multiplier` / `lora_b_lr_multiplier`. Some works find asymmetric LRs (e.g., 1.0 for B, 0.1 for A) can help.

**Priority: MEDIUM** — if benchmark scores plateau, full fine-tuning is the natural next step.

---

## 3. Architecture Improvements

### 3.1 Overlapping Multi-Crop Strategy

Currently your model processes images at 384×384 fixed resolution (single crop). Molmo's **overlapping multi-crop** is one of their most impactful innovations:

- Divide high-res images into multiple square crops that tile the image, plus a full-image low-res overview.
- Allow crops to **overlap** by a fixed margin (4 patches = 56 pixels) so border patches have context from neighboring patches.
- Strip overlapping patch features before passing to the LLM (so features exactly tile the image).
- Use 12 crops for training, 36 for evaluation on document tasks.

Ablation results:
| Config | Cap F1 | 11-avg |
|--------|--------|--------|
| Single crop | 46.7 | 62.8 |
| Multi-crop, no overlap | 53.4 | 75.7 |
| Multi-crop, with overlap | 54.1 | 76.9 |

**Priority: HIGH** — single-crop to multi-crop is the largest single architectural improvement (+13 points on 11-avg).

### 3.2 Multi-Layer Vision Feature Extraction

Instead of extracting features from just the last ViT layer, Molmo concatenates features from the **3rd-to-last and 10th-from-last** layers:

| Config | Cap F1 | 11-avg |
|--------|--------|--------|
| Only 3rd-to-last | 53.7 | 76.6 |
| Only 10th-from-last | 52.5 | 76.3 |
| Both concatenated | 54.1 | 76.9 |

LLaVA-OneVision similarly uses grid features "before and after the last Transformer layer."

For your model: you currently use `vision_feature_layer=-1`. Consider concatenating features from two layers (e.g., layers -3 and -10). This requires updating the connector input dimension (1152 → 2304) but your Pixel Shuffle already handles the dimension through the SwiGLU MLP.

**Priority: LOW-MEDIUM** — small but consistent improvement.

### 3.3 Attention Pooling Instead of Feature Stacking

Molmo replaces simple 2×2 feature concatenation (similar to your Pixel Shuffle) with **multi-headed attention pooling**:

- For each 2×2 patch window, pool into a single vector using attention where the mean of the 4 patches serves as the query.
- This outperforms simple concatenation:

| Pooling | Cap F1 | 11-avg |
|---------|--------|--------|
| Stacking (like Pixel Shuffle) | 53.7 | 76.1 |
| Attention pooling | 54.1 | 76.9 |

**Priority: LOW** — marginal gain, but worth trying if you're already modifying the connector.

### 3.4 Padding Embedding for Crop Boundaries

Molmo adds learned embeddings to patch features depending on whether a patch contains (a) no padding, (b) some padding, or (c) all padding. This lets the model distinguish padding from naturally dark image borders.

**Priority: LOW** — minor detail but helps with multi-crop.

---

## 4. Post-Training / Preference Optimization

### 4.1 DPO (Direct Preference Optimization)

After supervised fine-tuning, apply DPO to align the model with human preferences and reduce hallucinations.

**Approaches:**
- **LLaVA-NeXT Video DPO:** Train with AI-generated preference data. Sample multiple responses to the same image-question, use a strong judge (GPT-4V or a reward model) to rank them, then apply DPO. This yields significant improvements in conversational quality.
- **RLHF-V / Silkie:** Generate candidate responses from the model, use human (or AI) feedback to create preference pairs, then DPO. Silkie (2023) demonstrates this on LLaVA-1.6 with strong hallucination reduction.
- **Self-play DPO (SPPO):** Generate both chosen and rejected responses from the model itself at different temperatures or decoding strategies, rank them with a reward model.

**Data for DPO:**
- **RLHF-V dataset** (human-annotated preference pairs for VLMs).
- **VLFeedback / Silkie dataset** (AI-judged preference data from multiple VLMs).
- **Self-generate** preference pairs from your own model.

**Priority: HIGH** — DPO is the standard post-training step for all modern VLMs. Expect 3–8 point improvements on chat/conversation benchmarks and meaningful hallucination reduction.

### 4.2 Multi-Modal Reward Models

InternLM-XComposer2.5-Reward (IXC-2.5-Reward) demonstrates three applications of a multi-modal reward model:

1. **RL training signal.** Use the reward model with PPO to train IXC-2.5-Chat, showing consistent improvements in instruction following and open-ended dialogue.
2. **Best-of-N sampling (test-time scaling).** At inference, generate N candidate responses and select the highest-scoring one. This is a simple but effective way to improve output quality without retraining.
3. **Training data filtering.** Use the reward model to filter outlier/noisy samples from instruction tuning data before training.

**Explorations:**
- Train your own reward model by fine-tuning a copy of Tiny Aya Vision on preference data.
- Or use an existing open reward model (IXC-2.5-Reward, LLaVA-RLHF) as a judge.
- Apply Best-of-N at inference time as a zero-cost quality improvement.

**Priority: MEDIUM** — reward models are powerful but require additional training infrastructure.

### 4.3 Iterative Self-Improvement

Use the model to improve its own training data:

1. Generate detailed captions on a diverse image set using your current model.
2. Filter/refine using a reward model or heuristic quality scores.
3. Re-train on the improved data.
4. Repeat.

LLaVA-OneVision does a version of this: they use LLaVA-NeXT-34B to re-caption images, then train LLaVA-OneVision on those captions. This is "self-improvement AI" where training data is generated by earlier model versions.

**Priority: LOW-MEDIUM** — depends on model quality being good enough for useful self-captioning.

---

## 5. Grounding and Pointing

### 5.1 Point-Based Grounding (Molmo/PixMo-Points)

Molmo's unique capability: the model can **point to objects** in images using (x,y) coordinates normalized to 0–100 in plain text.

**Benefits:**
- Enables natural counting via chain-of-thought: "point to each instance, then count." Molmo's counting accuracy is 20+ points above GPT-4V.
- Enables pointing-as-explanation: the model supports its answers by pointing to relevant parts of the image.
- Opens up **agentic capabilities**: the model can point to UI elements, navigation waypoints, etc.

**Implementation:**
- Collect or generate pointing data: annotators point at objects and label them. Molmo collected 2.3M pointing annotations on 223K images.
- Train with a mix of pointing tasks: single-point, multi-point, counting, pointing-as-explanation.
- Use plain-text coordinates (not special tokens — Molmo ablation shows plain text works significantly better).
- Points should be ordered top-down, left-to-right with numbering.

**Priority: MEDIUM** — a unique differentiator, especially valuable for agentic applications.

### 5.2 Referring Expression / Bounding Box Grounding

Alternatively, train on referring expression datasets with bounding box outputs:

- **RefCOCO / RefCOCO+ / RefCOCOg** for referring expression comprehension.
- **Visual Genome** regions and relationships.
- Florence-2 style tasks: object detection, region captioning, OCR with spatial awareness.

**Priority: LOW-MEDIUM** — pointing is simpler and faster to annotate than boxes but boxes are more established.

---

## 6. Multi-Image and Video Extension

### 6.1 Multi-Image Understanding

LLaVA-OneVision's key insight: **a strong single-image model already has surprising multi-image capability** through the AnyRes design. To fully unlock it:

- Add multi-image instruction data: Spot-the-Diff, visual storytelling, multi-image VQA, interleaved image-text, image editing instructions.
- LLaVA-NeXT-Interleave's M4-Instruct dataset contains 560K multi-image samples across diverse tasks.
- Use simpler visual representations per image in multi-image mode (just base resolution, no multi-crop) to keep total token count manageable.

**Priority: MEDIUM** — extends capability significantly for practical applications.

### 6.2 Video Understanding via Image Transfer

LLaVA-OneVision demonstrates that **image-trained models transfer surprisingly well to video** with zero-shot modality transfer. To improve:

- Sample video frames at 1 FPS, resize each to base resolution.
- Use bilinear interpolation to reduce tokens per frame (e.g., 729 → 196).
- Fine-tune with video QA data: ActivityNet-QA, NextQA, EgoSchema, ShareGPT4Video.
- Balance total visual tokens across modalities (~7K tokens max regardless of whether input is image, multi-image, or video).

**Priority: LOW** — video is a separate modality milestone; focus on single-image first.

---

## 7. Training Recipe Refinements

### 7.1 Text-Only Dropout in Pre-Training

Molmo applies dropout **only to text tokens** during pre-training (not to vision tokens or prompt tokens). This:
- Encourages the model to rely more on vision tokens rather than guessing from language priors.
- Improves captioning precision (reduces hallucination).
- Improves downstream benchmarks.

| Dropout (pre-train, fine-tune) | Cap F1 | 11-avg |
|-------------------------------|--------|--------|
| off, off | 53.1 | 74.6 |
| on, on | 53.7 | 77.0 |
| on (text only), on | 54.1 | 76.9 |

**Priority: MEDIUM** — simple to implement, meaningful captioning improvement.

### 7.2 Component-Wise Gradient Clipping

Molmo clips gradients **separately** for the LLM, image encoder, and connector rather than using a single global norm. This prevents one component's large gradients from dominating updates to others.

**Priority: LOW** — small refinement but ensures training stability.

### 7.3 Correct Loss Normalization

A subtle but important bug exists in many codebases: when computing per-device gradients in DDP, divide total loss by the **average number of loss-tokens across all devices** (not the device-local count). Using device-local counts up-weights examples with short responses, since they are paired with a smaller divisor. This can drop captioning performance by 0.5–1 points.

**Priority: LOW** — but worth verifying in your codebase.

### 7.4 Learning Rate Configuration

Based on Molmo (7B scale):

| Component | Pre-Train LR | Fine-Tune LR |
|-----------|-------------|-------------|
| Connector | 2e-4 | 5e-6 |
| Vision Encoder | 6e-6 | 5e-6 |
| LLM | 2e-5 | 1e-5 |

Key: the connector needs a much higher LR during pre-training to quickly learn the alignment, then a lower LR during fine-tuning to maintain stability.

**Priority: MEDIUM** — tuning LRs per component can make or break VLM training.

### 7.5 High-Resolution Post-Training

Molmo shows that a brief high-resolution fine-tuning stage (3000 steps, 10% of fine-tuning) after the main training restores counting/pointing performance when using more crops at inference:

- Increase crops from 12 → 36 for 3000 additional steps.
- Halve learning rates across all components.
- This lets you use more crops at inference time for all tasks without degrading counting/pointing.

**Priority: LOW-MEDIUM** — only relevant once multi-crop is implemented.

---

## 8. Evaluation Improvements

### 8.1 Captioning F1 as Development Metric

Molmo introduces a **cap F1** metric (precision and recall of atomic statements in generated captions vs. ground truth). They found:
- Cap F1 correlates with downstream benchmark performance (Pearson ρ = 0.82).
- Most modeling decisions can be made based on captioning quality alone, without running costly fine-tuning + benchmark evaluation.
- This allows rapid iteration on pre-training design choices.

**Priority: MEDIUM** — saves significant compute during development.

### 8.2 Human Evaluation (Elo)

Molmo and LLaVA-OneVision both conduct human evaluations using Elo rankings:
- Collect diverse image-question pairs across 10 categories.
- Present annotators with two model outputs (blind).
- Compute Elo using Bradley-Terry model.
- This is more reliable than academic benchmarks for measuring real-world quality.

**Priority: LOW** — important for final assessment but expensive to run.

---

## 9. Interleaved Pre-Training (VILA Approach)

NVIDIA's VILA paper provides three key findings directly applicable to your model:

### 9.1 Interleaved Image-Text Corpus for Pre-Training

VILA shows that **interleaved image-text data** (images embedded within long text documents, like web pages) is far superior to image-caption pairs for pre-training:

| Pre-train corpus | 0-shot avg | 4-shot avg | MMLU (text-only) |
|-----------------|------------|------------|------------------|
| COYO (image-text pairs) | 51.1% | 50.3% | 28.8% (-17.2%) |
| MMC4 (interleaved) | 68.7% | 70.9% | 40.7% (-5.3%) |
| MMC4 + COYO (blend) | 69.0% | 71.3% | 40.2% (-5.8%) |

**Key insights:**
- Image-text pairs (like LLaVA-Pretrain captions) cause **catastrophic forgetting** of text capabilities (MMLU drops 17.2%).
- Interleaved data preserves text capabilities while teaching vision-language alignment.
- The **interleave structure** matters, not just longer text. Breaking interleaved data into pairs removes the benefit.
- Blending interleaved + caption data gives best diversity.

**Datasets to consider:**
- **MMC4-core** (25M images in interleaved web documents) — publicly available.
- **OBELICS** (141M web pages with interleaved images) — from HuggingFace.
- **Multimodal C4** — the original interleaved corpus.

**Priority: HIGH** — if you add a visual-language pre-training stage (beyond just connector alignment), interleaved data is essential.

### 9.2 Update the LLM During Pre-Training

VILA conclusively shows that **freezing the LLM** during pre-training preserves zero-shot performance but **destroys in-context learning (ICL)**:

- Freezing LLM: 0-shot OK, 4-shot accuracy degrades significantly.
- Updating LLM: enables deep embedding alignment between vision and text tokens, which is critical for ICL.
- A **simple linear projector** (not Transformer blocks) forces the LLM to learn more, leading to better generalization.

This has direct implications for your Tiny Aya Vision. Your Phase 1 freezes everything except the projector. If you want ICL capability, you need to unfreeze the LLM at some point during pre-training (not just during instruction tuning).

**Priority: MEDIUM-HIGH** — critical for in-context learning and few-shot capabilities.

### 9.3 Joint SFT with Text-Only Data (Recovers Forgetting Completely)

VILA's most actionable finding: blending text-only instruction data during SFT **both recovers text degradation AND improves visual accuracy**:

| Pre-train | SFT type | 0-shot avg | 4-shot avg | MMLU |
|-----------|----------|------------|------------|------|
| MMC4+COYO | Visual only | 69.0% | 71.3% | 40.2% (-5.8%) |
| MMC4+COYO | Visual + Text | **72.3%** | **73.6%** | **50.9%** (-0.3%) |

The text-only instruction data (1M samples from FLAN) improves instruction-following capability, which transfers to visual tasks. Text capabilities are "temporarily hidden, but not forgotten" — a small amount of text SFT recovers them.

**Priority: HIGH** — easy to implement and nearly eliminates the multimodal-training tax on text capabilities.

---

## 10. Reinforcement Learning with Verifiable Rewards (RLVR)

Ai2's Tulu 3 introduces **RLVR** as a post-DPO training stage. The idea: for tasks with verifiable correct answers (math, code, precise instruction following), use RL where the reward is binary (correct / incorrect) determined by a verifier, not a learned reward model.

### 10.1 Three-Stage Post-Training Pipeline (Tulu 3 Recipe)

```
Base Model → SFT → DPO → RLVR → Final Model
```

1. **SFT**: Train on diverse instruction data targeting core skills (math, code, safety, instruction-following, knowledge recall).
2. **DPO**: Train on **on-policy** preference data — generate completions from the SFT model, compare against outputs from other models, have a judge rank them.
3. **RLVR**: Final RL stage using PPO where rewards are **verified correctness** (not model-judged preferences). Only applied to tasks where answers can be programmatically verified.

### 10.2 Applying to VLMs

For Tiny Aya Vision, RLVR could be applied to:
- **Math-visual reasoning**: Verify if the model's answer to a math/geometry problem is numerically correct.
- **Counting**: Verify if the model correctly counts objects in the image using a detector as ground truth.
- **OCR**: Verify if extracted text matches ground truth.
- **Clock reading**: Verify if the read time is correct.

This is simpler than training a full reward model — you just need a verifier function.

**Priority: MEDIUM** — powerful but requires infrastructure for RL training and verifier functions for your specific tasks.

### 10.3 On-Policy Preference Data Generation

Tulu 3 finds that **on-policy** preference data (generated by the model being trained) significantly outperforms off-policy data (from other models). For DPO:

1. Take your SFT model.
2. Generate multiple responses per prompt.
3. Use a strong judge (GPT-4o, or an open reward model) to rank responses.
4. Create chosen/rejected pairs.
5. Train DPO on these pairs.

This is more effective than using pre-existing preference datasets because the model learns from its own failure modes.

**Priority: HIGH** — the difference between on-policy and off-policy DPO data is substantial.

---

## 11. Synthetic Data Generation at Scale

### 11.1 Persona-Driven Data Diversity

Molmo's PixMo-Docs uses **personas** (from Scaling Synthetic Data Creation with 1B Personas, arXiv:2406.20094) to control content and style of synthetic data. Example:
- Input query: "restaurant menu"
- Persona: "A barbecue enthusiast known for their amazing grilled food at every Tennessee Vols game"
- Output: A "Southern fusion menu combining traditional BBQ with international flavors, presented on a wooden board background"

This approach can generate massively diverse data from a single template.

**Explorations:**
- Generate diverse document images (invoices, receipts, menus, scientific papers, social media posts) using personas.
- Generate diverse chart types (heatmaps, violin plots, chord diagrams, geographic maps, treemaps) beyond basic bar/line charts.
- Generate diverse diagram types via Mermaid, Graphviz, TikZ.
- Use 7 rendering tools: Matplotlib, Plotly, LaTeX, HTML, Vega-Lite, Mermaid, Graphviz.

**Priority: MEDIUM** — high-impact if you want to improve document/chart understanding without human annotation.

### 11.2 Self-Captioning / Self-Play Data Augmentation

Use your trained model to generate new training data:

1. **Dense re-captioning**: Run your model on diverse web images to generate detailed captions. Filter with a reward model or heuristic quality scores. Use these as pre-training data for the next iteration.
2. **Question generation**: Given an image and its caption, use a text-only LLM to generate QA pairs from the caption (this is exactly PixMo-CapQA — 214K QA pairs generated from PixMo-Cap captions).
3. **Error-targeted data generation**: Analyze failure cases on your evaluation benchmarks. Generate synthetic training data targeting those specific weaknesses.
4. **Translation augmentation**: Since Tiny Aya is multilingual, take English instruction data and translate the text portions to other languages using a translation model, keeping the images the same.

**Priority: MEDIUM** — increasingly effective as your base model gets stronger.

### 11.3 Negative / Hard Example Mining

Train the model to say "I don't know" or refuse when appropriate:

- **"Not present" pointing data**: Molmo collects "not present" annotations where annotators ask the model to point to something that isn't in the image. This teaches the model to respond "This isn't in the image" instead of hallucinating.
- **Adversarial VQA**: Generate questions that are designed to elicit hallucinations (e.g., "What color is the dog?" when there's no dog). Train the model to correctly refuse.
- **Contrastive image pairs**: Show two similar images and ask about differences. This requires fine-grained visual discrimination.

**Priority: MEDIUM** — directly reduces hallucination, a key weakness of smaller VLMs.

---

## 12. Vision Encoder Improvements

### 12.1 Dual Vision Encoder Strategy

Cambrian-1 and other works show that combining **multiple vision encoders** can improve performance:

- **CLIP/SigLIP** (language-aligned): good at semantic understanding, object recognition.
- **DINOv2** (self-supervised): better at fine-grained spatial features, textures, counting.
- **Combination approach**: Concatenate features from both encoders, let the connector learn to fuse them.

Surprisingly, Molmo found that **DINOv2 alone** (with no text-based pre-training!) performs only slightly worse than CLIP — 45% win rate vs standard Molmo-7B-D in human evaluation. This suggests spatial features from SSL encoders provide complementary information.

**Explorations:**
- Add DINOv2 features alongside SigLIP2, concatenating before the connector.
- Use the `vision_feature_select_strategy` to extract features from different ViT layers of each encoder.
- This increases vision tokens but can be offset by more aggressive pooling.

**Priority: LOW-MEDIUM** — significant engineering effort for moderate gains. Best explored after other optimizations.

### 12.2 Higher Resolution Via Native ViT Scaling

Instead of multi-crop (which processes crops independently), some approaches scale the ViT natively:

- **Qwen2-VL**: Uses "Naive Dynamic Resolution" — resizes positional embeddings to support arbitrary image sizes directly in the ViT, without cropping. This preserves global context.
- **InternVL2**: Uses "dynamic high-resolution" with progressive scaling.

For SigLIP2, you could:
- Interpolate positional embeddings from 384×384 to 768×768 or higher.
- Fine-tune the vision encoder at the higher resolution for a few thousand steps.
- This gives genuinely higher resolution understanding without the border-context problem of cropping.

**Priority: LOW** — requires significant vision encoder modification.

### 12.3 Unfreezing Vision Encoder with Layerwise LR Decay

When unfreezing SigLIP2, use **layerwise learning rate decay** (LLRD):
- Top ViT layers get a higher LR (they adapt more).
- Bottom ViT layers get exponentially lower LR (they preserve pre-trained features).
- Typical decay: γ = 0.9, so layer n gets LR × γ^(N-n).

This is more nuanced than a single lower LR for the whole encoder and can prevent forgetting of low-level features while allowing high-level adaptation.

**Priority: LOW-MEDIUM** — a refinement when unfreezing the vision encoder.

---

## 13. Efficiency & Inference Optimizations

### 13.1 Knowledge Distillation from Larger Models

If you have access to a larger VLM (e.g., 7B or 13B), distill its capabilities into Tiny Aya Vision:

- **Response distillation**: Generate high-quality responses from the teacher model on your training images. Train the student on these responses.
- **Logit distillation**: Match the student's output logits to the teacher's during training (requires running both models).
- **Feature distillation**: Match intermediate representations between teacher and student connectors.

LLaVA-OneVision's re-captioning approach (using LLaVA-NeXT-34B to caption images) is a form of response distillation.

**Priority: MEDIUM** — particularly effective for smaller models like the 3.35B Tiny Aya.

### 13.2 Token Compression / Visual Token Pruning

Reduce the number of visual tokens to speed up training and inference:

- **VILA finding**: "Image resolution matters, not #tokens." Compressing 576 tokens to 144 (4× reduction) with 2×2 concatenation + linear at 336px resolution still outperforms 256 tokens at 224px.
- **FastV / LLaVA-PruMerge**: Prune redundant visual tokens after the first few LLM layers (the LLM attends heavily to only a subset of visual tokens).
- **Your Pixel Shuffle** already does 4× reduction (729→196). You could add a second stage of compression for inference efficiency.

**Priority: LOW** — useful for deployment but not for model quality.

### 13.3 Quantization-Aware Training (QAT)

If you plan to deploy on edge devices:

- Train with **QAT** (quantize weights during forward pass, use straight-through estimator for gradients).
- AWQ (Activation-aware Weight Quantization) can compress the 3.35B model to INT4 with minimal quality loss.
- VILA demonstrated deployment on **Jetson Orin** via AWQ + TinyCLIP, showing edge VLMs are viable.

**Priority: LOW** — deployment optimization, not accuracy improvement.

---

## 14. Multilingual-Specific Strategies

Since Tiny Aya is specifically designed for multilingual support (70+ languages), these explorations are particularly relevant:

### 14.1 Multilingual Visual Instruction Data

- **Translate-train**: Translate existing English instruction data (LLaVA-Instruct-150K, VQAv2, etc.) into target languages using a strong translation model (NLLB-200, Aya-101). Keep images the same. This is the cheapest way to get multilingual vision-language data.
- **xGQA**: Multilingual visual QA across 7 languages — use for evaluation and training.
- **MaXM / MTVQA**: Multilingual text-in-image VQA benchmarks — critical for evaluating OCR in non-Latin scripts.
- **Crossmodal-3600 (XM3600)**: Multilingual image captioning evaluation set covering 36 languages with human-written captions.
- **Multilingual dense captions**: Re-caption images in diverse languages using GPT-4o or Aya-Expanse. LLaVA-OneVision generated 92K Chinese captions and saw meaningful improvements.

### 14.2 Script-Specific OCR Data

Many languages use non-Latin scripts with unique OCR challenges:
- Generate synthetic text images in Arabic, Devanagari, CJK, Cyrillic, etc.
- Create reading/OCR QA pairs for each script family.
- The current training data (LLaVA) is almost entirely English — this is a huge gap for a multilingual model.

### 14.3 Culturally Diverse Image Sourcing

Web images are heavily biased toward Western cultures. For a multilingual model:
- Source images from region-specific databases (e.g., food from different cuisines, street scenes from different countries, signs/documents in local languages).
- Use geographically diverse image datasets like Dollar Street, GeoDE, or Casual Conversations.
- This ensures the model can handle visual concepts from the cultures it serves linguistically.

**Priority: HIGH** — this is the unique value proposition of Tiny Aya Vision and currently an underserved area.

---

## 15. Advanced Connector Architectures

### 15.1 C-Abstractor / Perceiver-Based Connector

Instead of Pixel Shuffle + MLP, use a learnable query-based connector:

- **C-Abstractor** (Honeybee, CVPR 2024): Uses convolutional layers + learnable queries to compress and project vision features. Outperforms linear/MLP projectors with fewer visual tokens.
- **Perceiver Resampler** (Flamingo/IDEFICS): Uses cross-attention with a fixed number of learnable queries to compress variable-length visual features into a fixed number of tokens. Good for multi-image/video since token count doesn't scale with #images.
- **Q-Former** (BLIP-2/InstructBLIP): Uses cross-attention queries but also interleaves self-attention with text. Adds capacity to the bridge.

However, VILA found that a **simpler projector forces the LLM to learn more**, leading to better generalization. Molmo's attention pooling is a middle ground.

**Priority: LOW** — the current Pixel Shuffle + SwiGLU MLP is already a strong design.

### 15.2 Post-Projector RMS Norm

Your config already includes `post_projector_rms_norm: bool = False`. Enabling this adds an RMS normalization layer after the projector output, which can stabilize the scale of visual embeddings entering the LLM. Some works (Aya Vision) use this.

**Priority: LOW** — trivial to try, could help with training stability.

---

## 16. Test-Time Compute Scaling

### 16.1 Best-of-N Sampling

At inference time, generate N responses and select the best one:
- Use a reward model to score each response.
- Or use self-consistency: generate N answers and take the majority vote (for deterministic answers like VQA/math).
- IXC-2.5-Reward showed this significantly improves output quality.
- Molmo uses 36 crops at test time (vs 12 during training) for document tasks — a form of test-time compute scaling on the vision side.

### 16.2 Chain-of-Thought Prompting

Molmo demonstrates chain-of-thought for counting (point-then-count), and VILA shows CoT is inherited from text-only SFT:
- Add "Think step-by-step" to prompts for complex reasoning.
- Train with explicit CoT data for math and reasoning tasks.
- Include spatial reasoning chains: "First I see X in the top-left, then Y in the center, therefore..."

### 16.3 Visual Self-Correction

After generating an initial answer:
1. Ask the model: "Look at the image again. Is your answer correct?"
2. The model re-examines the image and potentially revises its answer.
3. This can be trained via DPO (prefer corrected answers over incorrect first attempts).

**Priority: MEDIUM** — free quality improvement at inference time.

---

## 17. Recommended Exploration Roadmap (Updated)

### Phase A — Quick Wins (days)
1. **Expand instruction data**: Add VQAv2, TextVQA, DocVQA, ChartQA, AI2D, A-OKVQA to your Phase 2 mix.
2. **Mix in text-only data**: Add 10% Aya Collection or Tulu-3 text data to preserve multilingual LLM capabilities.
3. **Unfreeze vision encoder** in Phase 2 with 5× lower LR.
4. **Multi-layer vision features**: Concatenate features from ViT layers -3 and -10 instead of just -1.

### Phase B — Medium-Term (1–2 weeks)
5. **Replace LLaVA-Pretrain** with denser captions (re-caption with strong VLM or use ShareGPT4V).
6. **Implement multi-crop** (even 4 crops + 1 base is a big improvement over single-crop).
7. **DPO training** with on-policy preference data from your SFT model.
8. **Add synthetic document data** (charts, tables, diagrams via code generation).
9. **Text-only dropout** during pre-training (apply dropout only to text tokens).
10. **Multilingual instruction data** via translate-train on target languages.

### Phase C — Longer-Term Explorations (weeks)
11. **Full fine-tuning** of the LLM (replace LoRA) with proper data mix.
12. **Interleaved pre-training** on MMC4/OBELICS before SFT (VILA recipe).
13. **Stage 1.5 knowledge learning** with 1–4M high-quality samples.
14. **Pointing / grounding** capability via point-annotation training.
15. **RLVR** on math/counting/OCR tasks with programmatic verifiers.
16. **Multi-image / video extension** via OneVision-style training.
17. **Knowledge distillation** from a larger open VLM.
18. **Reward model** training + Best-of-N at inference.
19. **Dual encoder** (SigLIP2 + DINOv2) for stronger visual features.
20. **Culturally diverse** image + multilingual caption data collection.

---

## 18. Key References

| Paper | Key Contribution | Relevance |
|-------|-----------------|-----------|
| **Molmo & PixMo** (Ai2, 2024) [arXiv:2409.17146](https://arxiv.org/abs/2409.17146) | Open data/weights VLM, PixMo datasets, no distillation, overlapping multi-crop, pointing, detailed ablations | Training pipeline, data mixing, architecture choices |
| **LLaVA-OneVision** (2024) [arXiv:2408.03326](https://arxiv.org/abs/2408.03326) | 4-stage curriculum, 3.2M single-image + 1.6M OneVision data, cross-scenario transfer | Data curation, multi-stage training, multi-modal extension |
| **VILA** (NVIDIA, 2024) [arXiv:2312.07533](https://arxiv.org/abs/2312.07533) | Interleaved pre-training > caption pairs, unfreezing LLM essential for ICL, joint text+visual SFT recovers forgetting | Pre-training recipe, data mixing, text capability preservation |
| **Tulu 3** (Ai2, 2024) [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) | Three-stage post-training (SFT→DPO→RLVR), on-policy preference data, verifiable rewards | Post-training recipe, RL with verifiers |
| **Cambrian-1** (2024) [arXiv:2406.16860](https://arxiv.org/abs/2406.16860) | Vision-centric VLM exploration, multiple vision encoders, large instruction data mix | Data mixing, dual encoder, vision-centric design |
| **IXC-2.5-Reward** (2025) [arXiv:2501.12368](https://arxiv.org/abs/2501.12368) | Multi-modal reward model for PPO, Best-of-N, data filtering | Post-training with RL, test-time scaling |
| **MM1 / MM1.5** (Apple, 2024) [arXiv:2403.09611](https://arxiv.org/abs/2403.09611) | Systematic analysis of pre-training data mix, architecture, training recipe | Pre-training data composition insights |
| **InternVL 2** (2024) [arXiv:2312.14238](https://arxiv.org/abs/2312.14238) | Progressive scaling, dynamic resolution, strong multilingual VLM | Architecture, training strategy |
| **Qwen2-VL** (2024) [arXiv:2409.12191](https://arxiv.org/abs/2409.12191) | Naive dynamic resolution ViT, any-resolution support, strong document understanding | Resolution handling, native ViT scaling |
| **RLHF-V** (2024) [arXiv:2312.00849](https://arxiv.org/abs/2312.00849) | Human preference data for VLMs, DPO for hallucination reduction | Post-training preference optimization |
| **Silkie / VLFeedback** (2023) [arXiv:2312.10665](https://arxiv.org/abs/2312.10665) | AI-judged preference data from multiple VLMs, DPO on LLaVA | Post-training with AI feedback |
| **LLaVA-RLHF** (2023) [arXiv:2309.14525](https://arxiv.org/abs/2309.14525) | Factually augmented RLHF for VLMs, reducing hallucinations | RL-based post-training |
| **Honeybee/C-Abstractor** (2024) [arXiv:2312.06742](https://arxiv.org/abs/2312.06742) | Locality-enhanced projector with convolutional abstractor | Connector architecture alternatives |
| **Scaling Synthetic Data with 1B Personas** (2024) [arXiv:2406.20094](https://arxiv.org/abs/2406.20094) | Persona-driven synthetic data generation for diversity | Synthetic data at scale |
| **ShareGPT4V** (2023) [arXiv:2311.12793](https://arxiv.org/abs/2311.12793) | GPT-4V generated detailed captions for VLM training | Dense caption distillation |
| **OBELICS** (2023) [arXiv:2306.16527](https://arxiv.org/abs/2306.16527) | 141M web pages with interleaved images, open dataset for multimodal pre-training | Interleaved pre-training data |
| **Aya Expanse** (Cohere, 2024) [arXiv:2405.15032](https://arxiv.org/abs/2405.15032) | Multilingual LLM covering 100+ languages | Multilingual data generation, translation |
