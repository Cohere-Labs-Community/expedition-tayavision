# Multilingual Multimodal Training for Tiny Aya Vision

## 1. Language Coverage of Tiny Aya Global

**Model**: `CohereLabs/tiny-aya-global` (3.35B parameters, 262K vocab, 8K context)

Tiny Aya Global supports **70+ languages** with focus on:

| Region | Languages |
|--------|-----------|
| **European** | English, Dutch, French, Italian, Portuguese, Romanian, Spanish, Czech, Polish, Ukrainian, Russian, Greek, German, Danish, Swedish, Norwegian, Catalan, Galician, Welsh, Irish, Basque, Croatian, Latvian, Lithuanian, Slovak, Slovenian, Estonian, Finnish, Hungarian, Serbian, Bulgarian, Maltese |
| **Middle Eastern** | Arabic, Persian, Turkish, Hebrew |
| **South Asian** | Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu, Nepali, Urdu |
| **East/Southeast Asian** | Chinese, Japanese, Korean, Vietnamese, Thai, Lao, Burmese, Khmer, Tagalog, Malay, Indonesian, Javanese |
| **African** | Amharic, Hausa, Igbo, Malagasy, Shona, Swahili, Wolof, Xhosa, Yoruba, Zulu |

**Current gap**: Instruction tuning uses English-only LLaVA data — zero multilingual multimodal coverage.

---

## 2. Recommended Multilingual Multimodal Datasets

### 2.1 Tier 1: Large-Scale Multilingual Instruction Data

| Dataset | Size | Languages | Method | Citation |
|---------|------|-----------|--------|----------|
| **PangeaIns** | 6M, 39 langs | ar, bn, zh, en, fr, de, hi, id, ig, it, ja, jv, ko, ms, nl, no, pl, pt, ro, ru, es, sw, ta, te, th, tr, uk, ur, vi + more | Machine-translated + multicultural LAION-Multi images | Yue et al., 2024 [1] |
| **PALO** | ~2.1M (150K/lang × 10 + 665K EN) | en, zh, hi, es, fr, ar, bn, ru, ur, ja | Semi-automated translation with human-corrected LLM fine-tuning | Rasheed et al., 2025 [2] |
| **M3IT** | 2.4M, 80 tasks | Multilingual (400 instruction templates) | Multi-task multilingual multimodal instruction tuning | Li et al., 2023 [3] |
| **The Cauldron** | ~1.9M, 50 sub-datasets | English (but diverse tasks) | Massive multi-task VQA/chart/doc/OCR collection | Laurençon et al., 2024 [4] |

### 2.2 Tier 2: Language-Specific Datasets

| Dataset | Language(s) | Size | Citation / Source |
|---------|-------------|------|-------------------|
| **ALLaVA-4V Chinese** | Chinese | ~700K | Chen et al., 2024 [5] |
| **LLaVA-Japanese-Instruct** | Japanese | 108K | Toshi456, 2023 |
| **GQA-ru** | Russian | ~80K | `deepvk/GQA-ru` |
| **French Doc-VQA** | French | ~10K | `cmarkea/doc-vqa` |
| **French Table-VQA** | French | ~7K | `cmarkea/table-vqa` |
| **Viet Document/OCR QA** | Vietnamese | ~50K | Doan et al., 2024 [6] |
| **STAIR Captions** | Japanese | 820K | Yoshikawa et al., 2017 [7] |
| **MTVQA** | ar, zh, ja, ko, vi, th + more | ~30K | Tang et al., 2024 [8] |
| **Chinese-LLaVA** | Chinese | ~150K | LinkSoul-AI, 2023 |
| **LLaVA-Med Chinese** | Chinese (medical) | — | BUAA, 2023 |

### 2.3 Tier 3: Culturally Diverse Image Sources

| Source | Description | Citation |
|--------|-------------|----------|
| **LAION-Multi** | Web images with multilingual alt-text from diverse regions | Schuhmann et al., 2022 [9] |
| **Dollar Street** | Photos of everyday objects across income levels worldwide | — |
| **GeoDE** | Geographically diverse images with location labels | Ramaswamy et al., 2024 [10] |
| **CVQA** | Culturally diverse multilingual VQA | Romero et al., 2024 [11] |
| **MaRVL** | Multicultural reasoning over vision and language | Liu et al., 2021 [12] |
| **XM3600** | Multilingual captioning in 36 languages | Thapliyal et al., 2022 [13] |

### 2.4 Tier 4: African Language Coverage (Critical Gap)

Almost no multimodal data exists for Tiny Aya's 12 African languages. Options:

- Apply PangeaIns' LAION-Multi filtering + LLM-scoring pipeline to create culturally relevant data for sw, yo, ha, ig, am, etc.
- Translate a high-quality English subset (10–20K) into target African languages using Gemini 1.5 Pro (Pangea's validated translation model [1])
- Use CVQA [11] African-language subsets as both training and evaluation data

---

## 3. Data Mixing Techniques

### 3.1 The English Ratio Problem

Pangea [1] ran the most rigorous ablation on English-to-multilingual balance. They fixed total samples at 500K and varied the English proportion:

| English % | English Score | Multilingual Score |
|-----------|--------------|-------------------|
| 0% | Low | 37.2 |
| 20% | Medium | 37.8 |
| **40%** | **Good** | **38.7 (peak)** |
| 60% | Higher | 37.5 |
| 100% | Highest | 31.0 |

**Key finding**: English acts as a *cross-lingual transfer catalyst*. Zero English gives poor multilingual performance (no transfer signal). 100% English drowns out language-specific adaptation. **40% English / 60% multilingual is optimal** for instruction tuning.

> "Using only multilingual data results in relatively lower multilingual performance. As we introduce more English data, multilingual performance improves, peaking at 38.7% with 40% English." — Yue et al. [1]

### 3.2 Temperature-Based Language Sampling

Standard approach for multilingual models (mT5 [14], BLOOM [15]):

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

where $n_i$ is the number of samples in language $i$ and $T$ controls the distribution:

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| $T = 1$ | Proportional (high-resource dominates) | Default |
| $T = 2$–$3$ | Moderate upsampling of low-resource | Balanced multilingual |
| $T = 5$ | Strong upsampling (typical for multilingual LLMs) | mT5's choice |
| $T \to \infty$ | Uniform (all languages equal) | Maximum diversity |

**Temperature annealing curriculum** (advanced variant):

$$T(t) = T_{\max} - (T_{\max} - T_{\min}) \cdot \frac{t}{t_{\max}}$$

Start with high $T$ (uniform), anneal to low $T$ (proportional). Forces early learning of all languages, then refines on data-rich languages.

### 3.3 Sqrt-Proportional Sampling (Molmo)

Molmo [16] uses a simpler formula with manual adjustments:

$$\text{rate}(d) = \sqrt{|d|}$$

Then manually:
- Down-weight very large synthetic datasets
- Up-weight slow-learning tasks (pointing, counting)
- Up-weight human-annotated over synthetic

### 3.4 DoReMi: Learned Domain Weights

DoReMi [17] automates mixing via a proxy model:

1. Train a small proxy model (e.g., 280M) with **Group Distributionally Robust Optimization (Group DRO)** over domains — the model upweights domains where it struggles most
2. Extract the learned domain weights (mixing proportions)
3. Resample the full dataset according to these weights
4. Train the target model on the resampled data

**Results**: +6.5 points average few-shot accuracy over default weights; matches hand-tuned domain weights with zero downstream task knowledge; reaches baseline performance in **2.6× fewer steps**.

> "DoReMi improves perplexity across all domains, even when it downweights a domain." — Xie et al. [17]

For multilingual mixing, each language becomes a "domain" and DoReMi automatically learns optimal per-language sampling rates.

### 3.5 Source Balancing vs. Instance Balancing (Cambrian-1)

Cambrian-1 [18] found that **data source balancing** (equalizing contribution from each dataset source) generally outperforms **instance balancing** (equal weight per example). Too much of any single dataset category degrades other capabilities, and optimal ratios must be found through empirical ablation.

### 3.6 Cross-Lingual Transfer Exploitation

Pangea's analysis [1] revealed that typologically similar languages transfer strongly — you don't need equal data for every language:

| Language Family | Anchor (invest most) | Transfer Recipients |
|----------------|---------------------|---------------------|
| Romance | Spanish, French | Portuguese, Italian, Romanian, Catalan, Galician |
| Slavic | Russian | Polish, Ukrainian, Czech, Slovak, Croatian, Serbian, Bulgarian |
| Indic | Hindi | Marathi, Bengali, Gujarati, Punjabi, Nepali, Urdu |
| Dravidian | Tamil | Telugu |
| Southeast Asian | Indonesian | Malay, Javanese |
| East Asian | Chinese | (partial to Japanese, Korean) |
| Semitic/ME | Arabic | Persian, Hebrew |
| Bantu/African | Swahili | Shona, Xhosa, Zulu |

> "For low-resource languages, even a small increase in proportion yielded disproportionately large performance gains. Interestingly, we also noted instances of positive transfer between typologically similar languages." — Yue et al. [1]

Target **15–20 anchor languages** with real multimodal data, and the rest benefit from transfer.

---

## 4. Advanced Multilingual Training Strategies

### 4.1 Data Arbitrage (Aya Expanse)

No single teacher model excels across all languages for synthetic data generation. Aya Expanse [19] introduces **multilingual arbitrage**:

1. Maintain a pool of teacher models (GPT-4o, Gemini 1.5 Pro, Qwen, etc.)
2. For each prompt in each language, generate completions from all models
3. Use a **reward model** to score and select the best completion per prompt per language
4. The winning completions become SFT training data

**Result**: +9.1% win rate over prior Aya 23 at SFT stage alone.

> "Data arbitrage leverages performance variations among a pool of models... multilingual arbitrage proves valuable by utilizing a diverse pool of models to strategically sample different parts of the data distribution." — Dang et al. / Odumakinde et al. [19, 20]

### 4.2 Translate → Rephrase Pipeline (Aya Vision)

Aya Vision's key innovation over naive translation [21]:

```
English (Q,A) → Machine Translate → Rephrase (match with original synthetic sample)
```

The rephrase step eliminates translation artifacts and improves linguistic fluency. This gave **+17.2% win rate improvement** (40.9% → 58.1%) over academic-data-only on AyaVisionBench across 23 languages.

> "Synthetic annotations and scaling up the multilingual data lead to a 58.1% win rate with a gain of 17.2%. This significant improvement showcases the impact of significant investment in multilingual data coverage." — Aya Vision Blog [21]

### 4.3 Language-Cluster Model Merging (Aya Expanse)

Instead of finding one perfect data mix for all languages, train specialized models and merge [19, 22]:

1. **Group languages into families** by typological similarity
2. **Include bridge languages** (English, Spanish, French) in every cluster
3. **Train separate SFT models** per cluster
4. **Merge with weighted linear averaging** (SLERP, TIES, DARE were less consistent)
5. Repeat merging at each preference training iteration

Gains were **3× larger at 32B scale** than 8B.

> "We maximize diversity between checkpoints by training models for different language families. This takes advantage of cross-lingual transfer... naively splitting per language does not achieve the same benefits." — Dang et al. [19]

For TayaVision at 3.35B: train 2–3 LoRA adapters for language clusters, merge weights.

### 4.4 Multimodal Model Merging (Aya Vision)

After multimodal instruction tuning, merge the VLM's LLM weights with the original base LLM:

$$\theta_{\text{final}} = \alpha \cdot \theta_{\text{VLM}} + (1 - \alpha) \cdot \theta_{\text{base LLM}}$$

This recovered text-only capabilities degraded during multimodal training, yielding **+11.9% multilingual win rate** [21].

### 4.5 Iterative Online DPO (Aya Expanse)

After SFT, Aya Expanse runs multi-stage preference training [19]:

1. **Offline DPO**: Use highest/lowest reward responses from arbitrage as chosen/rejected
2. **Online iterative DPO** (3 iterations): Generate multiple responses from current model → rank with reward model → create preference pairs → train DPO

Offline + Online DPO gave **+7.1% win rate** over SFT alone. Beyond 3 iterations, gains were minimal and reward hacking appeared.

### 4.6 Text-Only Data Preservation

Training exclusively on multimodal data degrades text capabilities. Key findings:

- **VILA** [23]: Adding 1M text-only FLAN samples during SFT **fully recovers** MMLU degradation (-5.3% → -0.3%) and improves visual accuracy (+3.3%)
- **Molmo** [16]: 10% down-sampled Tulu 3 text data improved both text and multimodal benchmarks (76.9 → 77.1)
- Use **Aya Dataset** or **Aya Collection** multilingual text data to preserve multilingual LLM capabilities specifically

---

## 5. Recommended Recipe for TayaVision Multilingual

### 5.1 Phase 1 — Alignment (Keep English-only)

Connector training doesn't need multilingual data. Use dense English captions (PixMo-Cap or ShareGPT4V).

### 5.2 Phase 2 — Multilingual Instruction Tuning (~1.5–2M samples)

| Category | % | Samples | Composition |
|----------|---|---------|-------------|
| **English Multimodal** | 35% | 700K | LLaVA-665K + DocVQA/ChartQA/TextVQA (78K) |
| **Multilingual Multimodal (translated)** | 25% | 500K | Translate top 200K EN instruct → 10 anchor langs via Gemini 1.5 Pro + rephrase |
| **Multilingual Multimodal (native)** | 10% | 200K | PangeaIns cultural subset (150K) + ALLaVA-Chinese (50K) |
| **Multilingual Text-only** | 10% | 200K | Aya Dataset multilingual subset |
| **English Text-only** | 5% | 100K | Magpie Pro or Tulu 3 sample |
| **Document/Chart/OCR** | 10% | 200K | DocVQA, ChartQA, TextVQA, AI2D, MTVQA |
| **Math/Reasoning** | 5% | 100K | Geo170K, MAVIS, NuminaMath (translated subset) |

**Anchor languages** (highest translation priority): Chinese, Hindi, Arabic, Spanish, French, Russian, Japanese, Korean, Indonesian, Vietnamese, Turkish, Portuguese, Swahili, Bengali

### 5.3 Post-Training

1. **Merge LoRA** with original Tiny Aya Global weights (0.7 trained : 0.3 original)
2. Optionally run 1–2 rounds of **online DPO** with multilingual arbitrage-scored preference pairs
3. Evaluate on PangeaBench [1], AyaVisionBench [21], and m-ArenaHard [19]

### 5.4 Translation Pipeline

For creating multilingual instruction data:

1. Select 10–20K highest-quality English image-instruction pairs
2. Translate with **Gemini 1.5 Pro** (Pangea's validated choice [1])
3. **Post-process**: fix mismatched conversation turns, dropped MC candidates
4. **Rephrase**: match translated pairs with originals for fluency (Aya Vision approach [21])
5. **Quality filter**: back-translate a 1K sample; remove pairs where back-translation BLEU < threshold

---

## 6. Evaluation Benchmarks

| Benchmark | Languages | Task | Citation |
|-----------|-----------|------|----------|
| **PangeaBench** | 47 langs, 14 datasets | Multimodal chat, captioning, cultural understanding, VQA, reasoning | [1] |
| **AyaVisionBench** | 23 langs | 9 task categories, 135 image-question pairs/lang | [21] |
| **m-WildVision** | 23 langs | Multilingual Wild Vision (translated) | [21] |
| **xGQA** | 8 langs | Cross-lingual VQA | Pfeiffer et al., 2022 [24] |
| **MaXM** | 7 langs | Multilingual VQA | Changpinyo et al., 2022 [25] |
| **XM3600** | 36 langs | Multilingual captioning | [13] |
| **MTVQA** | 9 langs | Multilingual text-centric VQA | [8] |
| **CVQA** | 30+ langs | Culturally diverse VQA | [11] |
| **MaRVL** | 5 langs | Multicultural visual reasoning | [12] |
| **m-ArenaHard** | 23 langs | Open-ended generation (LLM-as-judge) | [19] |

---

## 7. Additional Data Mixing Strategies for English Retention + Multilingual Gains

The strategies in Sections 3–4 cover the established approaches. This section documents **further ideas** — beyond what Pangea, Aya, VILA, and Molmo have published — aimed specifically at increasing multilingual performance while retaining English capabilities.

### 7.1 Dynamic Temperature Annealing (Curriculum over T)

Section 3.2 mentions temperature annealing as an advanced variant but it has not been implemented. The schedule:

$$T(t) = T_{\max} - (T_{\max} - T_{\min}) \cdot \frac{t}{t_{\max}}$$

- Start with $T_{\max} = 100$ (near-uniform): forces the model to see all languages early.
- Anneal to $T_{\min} = 3$ by end: refines on data-rich languages.

**Why it helps**: A fixed $T=5$ over-samples low-resource from the beginning, when the model hasn't yet learned basic vision-language alignment. Early uniform exposure prevents the model from "locking in" to high-resource patterns before it has had a chance to see low-resource data. Late proportional weighting fine-tunes on clean, data-rich languages.

**Implementation**: Make `temperature` in `compute_sampling_indices()` a function of the current training epoch/step. Re-compute `target_per_lang` at the start of each epoch.

### 7.2 Per-Capability English Multimodal Replay

VILA [23] and Molmo [16] add **text-only** English data to prevent MMLU degradation. But this does not protect **English visual** capabilities (VQA, captioning, chart reading). During multilingual training, English multimodal skills also degrade.

**Strategy**: Reserve 10–15% of each training batch as an explicit English multimodal replay buffer:

- Sample from a curated English instruct set (LLaVA-Instruct, ShareGPT4V, DocVQA) alongside the multilingual mix.
- This functions as **experience replay** from the continual learning literature — it directly prevents catastrophic forgetting of English VQA, captioning, and document understanding skills.
- Unlike the text-only data preservation approach (Section 4.6), this also preserves the cross-modal alignment for English.

**Implementation**: In the `MultilingualMixedDataset`, maintain a separate `replay_dataset` loaded from English Phase 2 data. At each `__getitem__`, with probability `p_replay=0.12`, return a sample from the replay buffer instead of the multilingual mix.

### 7.3 Language-Adaptive LoRA Routing

Instead of training one global LoRA adapter for all 70+ languages, train **language-family-specific LoRA adapters** that are routed based on input language:

| Component | Treatment |
|-----------|-----------|
| Base LLM weights | Frozen |
| Vision encoder + connector | Shared (language-agnostic) |
| LoRA adapter bank | One per language family: {germanic, romance, slavic, indic, east_asian, southeast_asian, semitic, african} |

At training time, each batch is routed to the appropriate LoRA based on the `language` field. At inference time, detect language from the input and route accordingly.

**Advantages**:
- Each adapter specialises without interfering with other language families.
- English performance is protected in the `germanic` adapter without compromise.
- Each LoRA is ~1–3% of model parameters — total overhead is <20%.
- For deployment, merge a subset (e.g. 3 most-needed language families) using the cluster-merge strategy from Section 4.3.

**Trade-off**: More complex training and inference. Requires language detection at inference time. Best suited when you need strong performance in specific language clusters rather than uniform performance everywhere.

### 7.4 Difficulty-Aware Dynamic Sampling (Online DoReMi-Lite)

Instead of static temperature-based mixing, **dynamically adjust per-language weights based on validation loss**:

1. Hold out a small validation set (~500 samples per language).
2. Every $N$ steps (e.g. every 500), evaluate per-language validation loss.
3. Compute a learning signal: $\Delta_l = \text{loss}_{l}^{(t-N)} - \text{loss}_{l}^{(t)}$ (loss decrease for language $l$).
4. **Upweight** languages with the steepest loss decrease — these are in their steepest learning phase and have the most transferable signal.
5. **Downweight** languages that are plateaued (negligible $\Delta$) or diverging.

This is a lightweight online approximation of DoReMi [17] that doesn't require training a separate proxy model. The intuition: if Chinese VQA loss is dropping fast, the model is actively learning cross-modal alignment for CJK — give it more Chinese data. If Swahili loss has plateaued, additional Swahili data won't help; cross-lingual transfer from other Bantu languages may be more efficient.

**Implementation**: Track per-language EMA of validation loss. Recompute `target_per_lang` in `compute_sampling_indices()` using loss-derived weights blended with temperature weights:

$$w_l = \alpha \cdot w_l^{\text{temp}} + (1 - \alpha) \cdot w_l^{\text{loss-based}}$$

### 7.5 Staged Multilingual Introduction (Curriculum over Languages)

Instead of mixing all languages from step 0, introduce them in stages:

| Stage | Training % | Data Composition |
|-------|-----------|-----------------|
| **2a** | First 30% | English multimodal (80%) + multilingual text-only (20%) |
| **2b** | Next 40% | English multimodal (40%) + high-resource multilingual multimodal (40%) + multilingual text-only (10%) + English text-only (10%) |
| **2c** | Final 30% | Full Pangea-style mix: 40% EN / 60% multilingual across all languages |

**Why it helps**: The vision-language connector and LoRA adapt to basic visual understanding on clean English data first. Once that alignment is stable, multilingual multimodal data (which is noisier — machine-translated, fewer images) is introduced without destabilising the foundation. Low-resource languages are added last, when the model already has strong cross-lingual transfer capabilities from high-resource languages.

**Priority languages for Stage 2b**: Chinese, Spanish, French, Hindi, Arabic, Russian, Japanese, Korean (highest data quality and volume in PangeaIns).

### 7.6 EWC / Fisher-Weighted Regularisation for English Protection

During multilingual instruction tuning, add an **Elastic Weight Consolidation** (EWC) penalty [31] that explicitly protects the parameters most important for English:

$$\mathcal{L} = \mathcal{L}_{\text{multilingual}} + \frac{\lambda}{2} \sum_i F_i \left(\theta_i - \theta_i^{\text{EN-checkpoint}}\right)^2$$

where:
- $\theta_i^{\text{EN-checkpoint}}$ are the weights after English-only Phase 2 training.
- $F_i$ is the diagonal Fisher information for parameter $i$, computed on a sample of English multimodal data.
- $\lambda$ controls the regularisation strength (start with $\lambda = 0.1$, ablate).

The Fisher information identifies *which parameters matter most for English*. High-Fisher parameters are strongly penalised for drifting, while low-Fisher parameters are free to adapt for multilingual capabilities. This is strictly more targeted than the naive approach of weight-merging back to the base model (Section 4.4), because it selectively protects only what matters.

**Practical notes**:
- Computing the full Fisher is too expensive. Use a diagonal approximation over a 5-10K sample of English data.
- Only apply EWC to the LLM LoRA parameters, not the vision encoder or connector (those should be free to adapt).
- This can be combined with the weight-merging approach: use EWC during training, then merge post-training as a safety net.

### 7.7 Back-Translation Quality Gating

For machine-translated multilingual data (Section 5.4), add a hard quality gate before the data enters the mix:

1. Translate: EN → target language (forward pass)
2. Back-translate: target → EN (reverse pass)
3. Score: compute chrF++ between original English and back-translated English
4. **Reject** samples below threshold (e.g. chrF++ < 0.5)

**Why**: Poorly translated samples teach the model broken language patterns and degrade both the target language AND English (because the model learns to associate garbled text with correct images). Filtering these out is strictly beneficial.

**Additional quality signals** to combine with back-translation:
- **Language ID confidence**: Run fastText language ID on the translation. If the predicted language doesn't match the target language, reject.
- **Length ratio**: If the translation is <40% or >250% the length of the original, it's likely a bad translation.
- **Perplexity**: Score the translation with a language-specific model (e.g. Aya-101). High-perplexity translations are likely garbled.

### 7.8 Code-Switching Data Augmentation

Generate training samples that mix languages within a single conversation:

```
Human: この画像には何がありますか？ (What is in this image?)
Assistant: The image shows a mountain landscape with snow-capped peaks...
```

```
Human: Describe this image in detail.
Assistant: La imagen muestra un mercado con frutas tropicales. There are mangoes,
papayas, and dragonfruit arranged on wooden tables...
```

**Why**: Code-switching forces the model to maintain both language representations simultaneously in its hidden states. Research in multilingual NLP shows this improves cross-lingual transfer without degrading any single language [32]. It is especially useful for:
- **Bilingual users** who naturally code-switch
- **Cross-lingual retrieval**: understanding a query in one language about content in another
- Preventing the model from "forgetting" English when it sees a non-English prompt

**Implementation**: For 5–10% of translated training samples, randomly keep the human turn in the target language but the assistant response in English (or vice versa). This is zero-cost — no new data needed, just selective mixing of existing parallel data.

### 7.9 Contrastive Cross-Lingual Alignment Loss

Add an auxiliary loss that encourages the model to produce similar internal representations for semantically identical inputs in different languages:

$$\mathcal{L}_{\text{align}} = -\log \frac{\exp\left(\text{sim}(h_{\text{en}}, h_{\text{target}}) / \tau\right)}{\sum_j \exp\left(\text{sim}(h_{\text{en}}, h_j) / \tau\right)}$$

Using parallel data (same image, same question in English and a target language), extract the hidden representation at the last token of the question, and push the English and target-language representations closer together via contrastive learning.

**Requirements**: A set of parallel image-question pairs (readily available from the translated data pipeline). A projection head on top of the LLM hidden states (2-layer MLP, discarded after training).

**Trade-off**: Adds training complexity and requires parallel batches. Best combined with Stage 2b (Section 7.5) when parallel translated data is available. May be overkill for languages that already benefit from strong cross-lingual transfer.

### 7.10 Synthetic Native Multilingual Captions (Non-Translation)

Instead of only translating English instructions, generate **native multilingual captions** from scratch:

1. Select 10–20K culturally diverse images (GeoDE [10], Dollar Street, LAION-Multi [9]).
2. Prompt GPT-4o or Gemini 1.5 Pro: *"Describe this image in detail in [Hindi/Arabic/etc.]. Use natural, idiomatic language."*
3. Filter with language ID + perplexity scoring.
4. Pair with follow-up QA generated from the caption (Section 5.1 of data_curation_and_mixing.md).

**Why**: Translated captions carry "translationese" — unnatural phrasing patterns that are artefacts of the source language structure. A natively generated Hindi caption of a Delhi street market will use different vocabulary, idioms, and cultural references than a translated English caption of the same image. This produces more natural training signal for non-English languages.

**Cultural relevance**: An image of a Japanese shrine described natively in Japanese is genuinely more useful training data than a translated COCO caption about the same shrine. The native description will reference the torii gate, ema boards, and temizuya in culturally appropriate terms.

**Cost**: ~$0.01–0.03 per sample with GPT-4o-mini. 20K images × 10 languages = 200K samples at ~$2K–6K total.

### 7.11 Priority Ranking

| Priority | Strategy | Implementation Effort | Expected Impact on Multilingual | English Retention |
|----------|----------|----------------------|--------------------------------|-------------------|
| **1** | Staged multilingual introduction (7.5) | Low | High | High |
| **2** | Per-capability English replay (7.2) | Low | Neutral | High |
| **3** | Dynamic temperature annealing (7.1) | Low | Medium | Neutral |
| **4** | Back-translation quality gating (7.7) | Medium | Medium | Medium |
| **5** | Difficulty-aware sampling (7.4) | Medium | Medium-High | Medium |
| **6** | Language-adaptive LoRA routing (7.3) | Medium-High | High | High |
| **7** | EWC regularisation (7.6) | Medium | Neutral | High |
| **8** | Synthetic native captions (7.10) | High (API cost) | High | Neutral |
| **9** | Code-switching augmentation (7.8) | Low | Low-Medium | Medium |
| **10** | Contrastive alignment loss (7.9) | High | Medium | Medium |

**Recommended first batch** (low-effort, high-impact): Strategies 7.5, 7.2, and 7.1 can all be implemented in the existing `multilingual_data.py` pipeline with minimal changes. Together they address the two core failure modes: (a) multilingual data destabilising early training, and (b) English multimodal skills degrading over time.

---

## 8. Lessons from SoTA VLM Training Recipes

This section distils actionable training insights from the strongest open-source VLMs (Qwen2-VL [33], InternVL 2.5 [34], LLaVA-OneVision [26]) that are **directly applicable** to TayaVision's multilingual multimodal training.

### 8.1 Three-Stage Training Protocol (Qwen2-VL)

Qwen2-VL trains in three stages across **1.4 trillion tokens**:

| Stage | Focus | What's Trained | Data |
|-------|-------|---------------|------|
| **Stage 1** | ViT pre-training | ViT only (LLM frozen) | ~600B tokens: image-caption pairs, OCR, classification |
| **Stage 2** | Full multi-task pre-train | All parameters unfrozen | ~800B tokens: mixed image-text, VQA, multitask |
| **Stage 3** | Instruction fine-tuning | ViT frozen, LLM fine-tuned | Instruction data (ChatML format) |

**Key takeaway for TayaVision**: Stage 2 unfreezes everything *including the ViT*. This is critical — most LLaVA-style models freeze the ViT during instruction tuning, but Qwen2-VL shows that further ViT adaptation on diverse multimodal data during Stage 2 is essential for domain-specific visual understanding (OCR, charts, multilingual text in images). For TayaVision, unfreezing the ViT during multilingual training could significantly improve non-Latin script text recognition.

**Multilingual OCR**: Qwen2-VL explicitly trains with multilingual OCR data and achieves SoTA on MTVQA (multilingual text-centric VQA), outperforming GPT-4o across Korean, Japanese, French, German, Italian, Russian, and Vietnamese. This confirms that **multilingual OCR data is a high-ROI investment**.

### 8.2 Progressive Scaling Strategy (InternVL 2.5)

InternVL 2.5 matches Qwen2-VL's performance with **only ~120B tokens** (1/10th of Qwen2-VL) through a progressive scaling strategy:

| Stage | Focus | What's Trained | Purpose |
|-------|-------|---------------|---------|
| **Stage 1** | MLP warmup | MLP projector only | Cross-modal alignment |
| **Stage 1.5** | ViT incremental learning | ViT + MLP | Enhance visual features for rare domains (multilingual OCR, math charts) |
| **Stage 2** | Full model instruction tuning | All (ViT + MLP + LLM) | Final instruction-following capability |

**Key insights**:

1. **Train ViT once with a small LLM, reuse with larger ones.** InternVL trains InternViT-6B with a 20B LLM in Stage 1.5, then plugs the same ViT into 72B without retraining. For TayaVision, this means the vision encoder can be refined on multilingual image data once and reused.

2. **Large ViTs reduce data dependency.** InternVL2.5-78B (6B ViT) achieves better results than Qwen2-VL-72B (600M ViT) with 10× fewer training tokens. A stronger vision encoder compensates for less training data — relevant for resource-constrained settings.

3. **Stage 1.5 is specifically for rare visual domains.** InternVL uses this stage to inject multilingual OCR, mathematical charts, and other data types that are underrepresented in web-scale pretraining. **This is exactly what TayaVision needs** — a dedicated stage to train the ViT on non-Latin scripts before full multilingual instruction tuning.

### 8.3 Data Quality > Data Quantity (InternVL 2.5)

InternVL 2.5's most important finding: a few thousand noisy samples can cause catastrophic degradation.

**Repetitive pattern filtering**: InternVL found that "repetitive patterns" in training data — even just a few thousand samples out of 16M — cause the model to get stuck in repetitive loops during CoT reasoning. They implemented a three-pronged filtering pipeline:

1. **LLM-based quality scoring**: Score each sample 0–10 with a domain-specific prompt using a strong LLM. Remove samples below threshold (e.g., 7).
2. **Repetition detection**: Use an LLM with a specialised prompt to flag repetitive outputs. Manual review, remove below threshold (e.g., 3).
3. **Heuristic rule-based filtering**: Flag abnormal sentence lengths, excessive duplicate lines, long zero sequences. Manual review.

**For TayaVision**: Machine-translated multilingual data is especially prone to repetitive and degenerate patterns. Apply aggressive repetition detection on all translated data before mixing.

### 8.4 Random JPEG Compression (InternVL 2.5)

A simple but effective data augmentation: apply random JPEG compression (quality 75–100) to all image training data. This simulates degradation in real-world internet-sourced images and improves robustness. Enabled for all image data, disabled for video.

**For TayaVision**: Especially useful for multilingual OCR tasks where real-world text images are often low-quality compressed JPEGs.

### 8.5 Square Averaging Loss (InternVL 2.5)

Standard NTP loss weighting creates bias:
- **Token averaging**: Gradients biased toward long responses → models produce verbose outputs.
- **Sample averaging**: Each sample contributes equally → models favor short responses.

InternVL's solution: **square averaging** with $w_i = 1/x^{0.5}$, where $x$ is the number of response tokens. This balances contribution between short benchmark-style answers and long conversational responses.

**For TayaVision**: Different languages produce different response lengths for the same content (e.g., Chinese responses are typically shorter in tokens than English). Square averaging prevents the loss from being dominated by the verbosity of particular languages.

### 8.6 Multimodal Data Packing (InternVL 2.5)

To avoid wasting GPU compute on padding tokens, InternVL packs multiple samples into a single sequence:

1. **Select**: Sample data, truncate to fit within max sequence length and max tile count.
2. **Search**: Find a complementary sample from a buffer that fits the remaining space.
3. **Pack**: Concatenate with attention masking so tokens only attend within their own sample.
4. **Maintain**: Yield packed sequence when full, or buffer for future packing.

This dual constraint (sequence length **and** image tile count) is unique to multimodal packing. For TayaVision, this can be implemented via a custom data collator that packs multilingual samples together, significantly improving GPU utilization.

### 8.7 Dataset Repeat Factors (InternVL 2.5)

InternVL uses a **repeat factor** $r \in (0, 4]$ per dataset to control sampling frequency:
- $r < 1$: down-sample (reduce dataset weight)
- $r > 1$: up-sample (train multiple epochs on that dataset)

This is combined with per-dataset `n_max` (max image tiles) to control resolution:
- High-res docs/infographics: $n_{max} = 24\text{--}36$
- Standard images: $n_{max} = 6\text{--}12$
- Video frames: $n_{max} = 1$

**For TayaVision**: Use repeat factors to fine-tune the balance between language families without changing the underlying sampling temperature. E.g., set $r=2$ for African language data (critical gap) and $r=0.5$ for overrepresented English VQA.

### 8.8 Bilingual Data as Baseline (InternVL 2.5 + Qwen2-VL)

Both InternVL 2.5 and Qwen2-VL are primarily trained on **English + Chinese bilingual data** with smaller amounts of other languages. Their multilingual results confirm that:

> "The multilingual capabilities of MLLMs are largely inherited from the underlying language model." — InternVL 2.5 [34]

InternVL2.5-78B and Qwen2-VL-72B share the same LLM (Qwen2.5-72B) and show near-identical multilingual performance despite different training data. This means **TayaVision's multilingual ceiling is primarily set by Tiny Aya Global's multilingual backbone**, not by the multimodal training data.

**Implication**: Invest in high-quality bilingual (EN + one high-resource target) multimodal data first. Expanding to more languages yields diminishing returns unless the base LLM strongly supports those languages.

### 8.9 Test-Time Scaling with CoT (InternVL 2.5)

InternVL 2.5 is the first open-source MLLM to show that CoT prompting at test time significantly improves performance (+3.7 points on MMMU). Combined with majority voting, gains increase further.

**CoT prompt template**: "Let's think step by step." at the end of the question, followed by "Therefore, the answer is" to extract the final answer.

**For TayaVision**: Build CoT into your evaluation pipeline. Especially important for multilingual mathematical reasoning and document understanding benchmarks where chain-of-thought reasoning in the target language improves accuracy.

### 8.10 Unified Image + Video Training (Qwen2-VL)

Qwen2-VL treats images as 2-frame videos and trains on both modalities simultaneously. This eliminates the need for separate image and video models. The 3D convolution in the ViT processes video inputs as 3D tubes (depth=2) rather than 2D patches, so more video frames can be processed without increasing sequence length.

**For TayaVision**: Not immediately relevant for multilingual performance, but if you plan to add video understanding later, designing the architecture to support both from the start avoids costly retraining.

### 8.11 Preserving Language Capabilities Through Data (InternVL 2.5)

InternVL 2.0 showed a 2.1–2.3 point decline in pure language benchmarks vs. the base LLM. InternVL 2.5 **fully recovered and even surpassed** the base LLM by:

1. Collecting a large, high-quality open-source **pure text instruction dataset** covering: general QA (UltraFeedback, LIMA, SlimOrca, FLAN, FLANv2), code (Code-Feedback, Evol-Instruct-Code), math (GSM8K-Socratic, NuminaMath, Orca-Math), long context (LongCite, LongAlpaca), and multilingual data (Korean, Chinese, English).
2. Applying the same strict data filtering pipeline (Section 8.3) to text data.

**For TayaVision**: Include **Aya Dataset + Aya Collection** (multilingual text-only) and **Magpie-Pro** (English text-only) in the mix. The text-only data should be ~10–15% of the total training mix. Filter aggressively for repetitive patterns.

### 8.12 Practical Data Composition for TayaVision (Updated)

Based on the SoTA recipes above, here's an updated recommended composition for a **~2–3M sample training mix**:

| Category | % | Samples | Source & Rationale |
|----------|---|---------|-------------------|
| **English Multimodal** | 30% | 750K | LLaVA-665K + ShareGPT4V + DocVQA/ChartQA/TextVQA/AI2D (high-quality English foundation) |
| **Multilingual Multimodal (translated + native)** | 25% | 625K | PangeaIns (300K) + Translate top EN instruct → 10 anchor langs (200K, quality-gated) + ALLaVA-Chinese (50K) + native captions (75K) |
| **Multilingual OCR** | 10% | 250K | SynthDoG multi-script (JP, KO, RU, ZH, AR) + MTVQA training data + synthetic non-Latin OCR + VCR training data (22K, per InternVL) |
| **Document / Chart / Science** | 10% | 250K | DocVQA + ChartQA + InfographicVQA + AI2D + Docmatix sample + Synthetic Chart2Markdown |
| **Math / Reasoning** | 7% | 175K | Geo170K-QA + MAVIS + NuminaMath-CoT sample + TabMWP + CLEVR-Math |
| **Text-only Multilingual** | 8% | 200K | Aya Dataset (100K) + Aya Collection sample (50K) + Korean/Chinese text instruction data (50K) |
| **Text-only English** | 5% | 125K | Magpie-Pro + Tulu 3 sample + UltraFeedback sample |
| **English Multimodal Replay** | 5% | 125K | Replay buffer from Phase 2 EN checkpoint (LLaVA-Instruct, VQAv2, ShareGPT4V) |

**Key differences from Section 5.2 recipe**:
- Explicit **multilingual OCR category** (10%) — the single biggest gap identified from Qwen2-VL and InternVL results.
- Explicit **English multimodal replay buffer** (5%) to preserve English VQA/captioning skills.
- **Repeat factors** per dataset rather than fixed sampling: $r=2$ for African language data, $r=0.5$ for VQAv2 (already well-represented), $r=1.5$ for multilingual OCR.
- All data passes through **quality gating**: repetition detection + back-translation quality filter for translated data.

### 8.13 Training Configuration Recommendations

Based on the combined findings of Qwen2-VL, InternVL 2.5, and LLaVA-OneVision:

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| **ViT training** | Unfreeze ViT in Stage 1.5 (with low LR: 1e-5) | InternVL shows this enables rare-domain visual adaptation |
| **Loss weighting** | Square averaging ($w = 1/x^{0.5}$) | Prevents bias toward verbose/terse languages |
| **Data augmentation** | Random JPEG compression (quality 75–100) on images | Robust OCR in real-world conditions |
| **Data packing** | Pack multiple samples per sequence | 30–50% GPU utilization improvement |
| **Repetition filter** | LLM-based + heuristic on all fine-tuning data | Prevents CoT degradation; critical for translated data |
| **Context length** | 2048 → 4096 tokens during Stage 2 | Enables longer multilingual conversations |
| **Eval protocol** | Test both direct-answer and CoT modes | CoT gives +3–5 points on reasoning benchmarks |
| **Text-only data** | 13% of total mix (8% multilingual + 5% English) | Fully recovers language capabilities per InternVL 2.5 |

---

## References

[1] Yue, X., Song, Y., Asai, A., Kim, S., et al. "Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages." arXiv:2410.16153, 2024.

[2] Rasheed, H., Maaz, M., Shaker, A., Khan, S., et al. "PALO: A Polyglot Large Multimodal Model for 5B People." WACV 2025. arXiv:2402.14818, 2024.

[3] Li, L., Yin, Y., Li, S., Chen, L., et al. "M3IT: A Large-Scale Dataset Towards Multi-Modal Multilingual Instruction Tuning." arXiv:2306.04387, 2023.

[4] Laurençon, H., Tronchon, L., Cord, M., Sanh, V. "What Matters When Building Vision-Language Models?" arXiv:2405.02246, 2024.

[5] Chen, G.H., Chen, S., Zhang, R., et al. "ALLaVA: Harnessing GPT4V-Synthesized Data for a Lite Vision-Language Model." arXiv:2402.11684, 2024.

[6] Doan, K.T., Huynh, B.G., et al. "Vintern-1B: An Efficient Multimodal Large Language Model for Vietnamese." arXiv:2408.12480, 2024.

[7] Yoshikawa, Y., Shigeto, Y., Takeuchi, A. "STAIR Captions: Constructing a Large-Scale Japanese Image Caption Dataset." ACL 2017.

[8] Tang, J., Liu, Q., Ye, Y., et al. "MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering." arXiv:2405.11985, 2024.

[9] Schuhmann, C., Beaumont, R., Vencu, R., et al. "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models." NeurIPS 2022. arXiv:2210.08402.

[10] Ramaswamy, V.V., Lin, S.Y., Zhao, D., et al. "GeoDE: A Geographically Diverse Evaluation Dataset for Object Recognition." NeurIPS 2023.

[11] Romero, D., Lyu, C., Wibowo, H.A., et al. "CVQA: Culturally-Diverse Multilingual Visual Question Answering Benchmark." arXiv:2406.05967, 2024.

[12] Liu, F., Bugliarello, E., Ponti, E.M., et al. "Visually Grounded Reasoning Across Languages and Cultures." EMNLP 2021.

[13] Thapliyal, A.V., Pont Tuset, J., Chen, X., Soricut, R. "Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset." EMNLP 2022.

[14] Xue, L., Constant, N., Roberts, A., et al. "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer." NAACL 2021.

[15] Le Scao, T., Fan, A., Akiki, C., et al. "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model." arXiv:2211.05100, 2022.

[16] Deitke, M., Clark, C., Lee, S., et al. "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models." arXiv:2409.17146, 2024.

[17] Xie, S.M., Pham, H., Dong, X., et al. "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining." NeurIPS 2023. arXiv:2305.10429.

[18] Tong, S., Brown, E., Wu, P., et al. "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs." arXiv:2406.16860, 2024.

[19] Dang, J., Singh, S., D'souza, D., Ahmadian, A., et al. "Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier." arXiv:2412.04261, 2024.

[20] Odumakinde, A., D'souza, D., Verga, P., Ermis, B., Hooker, S. "Multilingual Arbitrage: Optimizing Data Pools to Accelerate Multilingual Progress." arXiv:2408.14960, 2024.

[21] Dash, S., Nan, O., Ahmadian, A., et al. "A Deepdive into Aya Vision: Advancing the Frontier of Multilingual Multimodality." Cohere For AI / HuggingFace Blog, March 2025.

[22] Aakanksha, Ahmadian, A., Goldfarb-Tarrant, S., Ermis, B., Fadaee, M., Hooker, S. "Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning." arXiv, 2024.

[23] Lin, J., Yin, H., Ping, W., et al. "VILA: On Pre-training for Visual Language Models." arXiv:2312.07533, 2023.

[24] Pfeiffer, J., Geigle, G., Kamath, A., et al. "xGQA: Cross-Lingual Visual Question Answering." Findings of ACL 2022.

[25] Changpinyo, S., Xue, L., Yarom, M., et al. "MaXM: Towards Multilingual Visual Question Answering." arXiv:2209.05401, 2022.

[26] Li, B., Zhang, Y., Guo, D., et al. "LLaVA-OneVision: Easy Visual Task Transfer." arXiv:2408.03326, 2024.

[27] McKinzie, B., Gan, Z., Fauber, J., et al. "MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training." arXiv:2403.09611, 2024.

[28] Üstün, A., Aryabumi, V., et al. "Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model." arXiv, 2024.

[29] Geigle, G., Jain, A., Timofte, R., Glavaš, G. "mBLIP: Efficient Bootstrapping of Multilingual Vision-LLMs." ALVR Workshop, ACL 2024.

[30] Cohere Labs. "Tiny Aya Global Model Card." HuggingFace, 2026. arXiv:2603.11510.

[31] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. "Overcoming Catastrophic Forgetting in Neural Networks." PNAS 2017. arXiv:1612.00796.

[32] Winata, G.I., Cahyawijaya, S., Liu, Z., et al. "Are You Sure You're Sure? Effects of Visual Data on Language Model Representations of Code-Switching." ACL Findings, 2023.

[33] Wang, P., Bai, S., Tan, S., et al. "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv:2409.12191, 2024.

[34] Chen, Z., Wang, W., Cao, Y., et al. "Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling." (InternVL 2.5) arXiv:2412.05271, 2024.

[35] Zhang, H., Gao, M., Gan, Z., et al. "MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning." arXiv:2409.20566, 2024.
