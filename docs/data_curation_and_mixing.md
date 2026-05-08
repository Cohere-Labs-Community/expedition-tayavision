# Data Curation & Mixing Strategies for Tiny Aya Vision

## Current Data Situation

| Stage | Dataset | Size | Description |
|-------|---------|------|-------------|
| Phase 1 — Alignment | LLaVA-Pretrain (BLIP-558K) | 558K | Short BLIP-generated captions (~11 words avg), CC+SBU images |
| Phase 2 — Instruct | LLaVA-Instruct-150K | 150K | GPT-4 generated multi-turn visual QA on COCO images |

**Problem**: Both datasets are small by modern standards and English-only. SoTA models use 1–4M samples for instruction tuning and 50M+ for pre-training. The caption data is low-quality. There is zero multilingual data despite Tiny Aya being a multilingual LLM.

---

## 1. Pre-Training Data (Phase 1 Replacement / Enhancement)

### 1.1 The Case Against LLaVA-Pretrain

LLaVA-Pretrain (BLIP-558K) was the standard Phase 1 dataset in 2023, but it is now well-understood to be suboptimal:

- **Short captions** (~11 words avg) provide weak learning signal for the projector.
- **Machine-generated** by BLIP, not human-written — factually noisy.
- **Low diversity** — images are only from CC3M + SBU Captions.
- **Molmo finding**: Pre-training solely on noisy web-scale caption data "does not improve results." Their ablation shows a separate connector-only pre-training stage can be **skipped entirely** when using high-quality dense captions.

### 1.2 Dense Caption Datasets (Recommended Replacements)

| Dataset | Size | Avg Caption Length | Source | License |
|---------|------|-------------------|--------|---------|
| **PixMo-Cap** | 712K images, 1.3M transcripts | ~196 words | Human-spoken descriptions (Ai2/Molmo) | Open |
| **ShareGPT4V** | 100K | ~100 words | GPT-4V distilled captions | Open |
| **ShareGPT4o** | 57K | ~120 words | GPT-4o distilled captions | Open |
| **ALLaVA-Caption** | 700K | ~80 words | GPT-4V captions on LAION | Open |
| **LLaVA-ReCap (COCO118K)** | 118K | ~80 words | LLaVA-NeXT-34B re-captions | Open |
| **LLaVA-ReCap (BLIP558K)** | 558K | ~80 words | LLaVA-NeXT-34B re-captions | Open |
| **LLaVA-ReCap (CC3M)** | 3M | ~80 words | LLaVA-NeXT-34B re-captions | Open |
| **DetailCaps-4870** | 4.9K | ~150 words | GPT-4o + images as eval set | Open |

**Recommendations (in priority order):**

1. **Best quality (no distillation)**: Use **PixMo-Cap** (712K). Molmo shows human-spoken dense captions outperform GPT-4V-distilled ones on captioning F1 and perform comparably on downstream benchmarks.
2. **Pragmatic option**: Use **ShareGPT4V (100K) + LLaVA-ReCap (558K)** — re-caption the same images with a much stronger model. Zero cost if using existing open datasets.
3. **Maximum scale**: Use **LLaVA-ReCap (CC3M)** (3M samples). LLaVA-OneVision uses 3.5M re-captioned samples as their Stage 1.5 knowledge data.

### 1.3 Interleaved Image-Text Data (for Mid-Training)

MM1 (Apple) and VILA (NVIDIA) both demonstrate that a multimodal pre-training stage using interleaved image-text data is critical for in-context learning and preserving text capabilities:

| Dataset | Size | Description | License |
|---------|------|-------------|---------|
| **OBELICS** | 141M web pages | Interleaved images+text from Common Crawl | Open (HuggingFace) |
| **MMC4-core** | 25M images | Interleaved images+text from C4 | Open |
| **Multimodal-C4** | 103M images | Full MMC4 corpus | Open |

**MM1's optimal pre-training mix** (their most critical finding):
```
45% Interleaved image-text (e.g., OBELICS/MMC4)
45% Image-caption pairs (e.g., LAION, COYO, CC12M)
10% Text-only data (to prevent catastrophic forgetting)
```

MM1 found this ratio outperforms any single data type. The interleaved data is essential for few-shot in-context learning. The text-only data prevents the MMLU degradation that VILA observed (17.2% drop without it).

**VILA's finding**: Even a 50-50 blend of interleaved (MMC4) + caption pairs (COYO) at 50M images is enough to significantly outperform LLaVA-1.5 across all benchmarks.

### 1.4 Image-Caption Pair Datasets (Higher Quality than BLIP-558K)

| Dataset | Size | Description | License |
|---------|------|-------------|---------|
| **CC3M** | 3.3M | Conceptual Captions — curated alt-text | Open |
| **CC12M** | 12M | Larger Conceptual Captions, noisier | Open |
| **COYO-700M** (subsampled) | 25M+ | Korean institute, high CLIP similarity filtering | Open |
| **LAION-2B** (subsampled) | 25M+ | Massive web-crawled pairs | Open (metadata) |
| **SBU Captions** | 1M | Flickr images with user captions | Open |
| **DataComp** | 1.4B pool | Curated pool for CLIP training, can be filtered for VLM pre-training | Open |

**Key insight from Molmo**: Adding noisy web-scale data (like LAION) to a connector-only pre-training stage provides **no improvement** if you already have high-quality dense captions. Skip the noisy data stage and go straight to dense captions.

---

## 2. Instruction-Tuning Data (Phase 2 Enhancement)

### 2.1 The Core Data Categories

Every SoTA VLM organizes instruction data into skill categories. Here is the taxonomy used by LLaVA-OneVision and Cambrian-1, with concrete datasets for each:

#### Category A: General Visual QA (~30-35% of mix)

| Dataset | Size | Task | HuggingFace ID |
|---------|------|------|----------------|
| **VQAv2** | 83K train | Open-ended VQA on COCO | `HuggingFaceFW/VQAv2` |
| **GQA** | 72K | Scene graph-based VQA | `lmms-lab/GQA` |
| **A-OKVQA** | 17K | VQA requiring world knowledge | `HuggingFaceFW/a-okvqa` |
| **OK-VQA** | 9K | VQA requiring external knowledge | `Multimodal-Fatima/OK-VQA_train` |
| **Visual Genome** | 86K | Region descriptions + QA | — |
| **RefCOCO/+/g** | 51K | Referring expression comprehension | — |
| **LLaVA-Instruct-158K** | 158K | GPT-4 generated conversations | `liuhaotian/LLaVA-Instruct-150K` |
| **LLaVA-Wild** | 55K | Real user queries from LLaVA demo | — |
| **ShareGPT4V** | 91K | GPT-4V detailed visual conversations | `Lin-Chen/ShareGPT4V` |
| **ShareGPT4o** | 57K | GPT-4o detailed visual conversations | — |
| **ALLaVA-Instruct** | 70K | GPT-4V synthesized instruction data | — |
| **Vision-FLAN** | 186K | Human-labeled tasks, diverse prompts | `Vision-FLAN/vision-flan_191-task_1k` |
| **Cambrian-filtered (GPT-4o)** | 83K | Re-annotated by GPT-4o | — |
| **TallyQA** | 250K | Counting questions (simple + complex) | — |
| **ScienceQA** | 5K (image) | High-school science MC questions | `derek-thomas/ScienceQA` |
| **COCO Captions** | 20K | Brief captioning tasks | — |
| **IconQA** | 2.5K | Abstract diagram reasoning | — |
| **VizWiz** | 6.6K | VQA from blind users' photos | — |

#### Category B: Document / Chart / Screen (~20-25% of mix)

This category is critical for structured visual understanding and is the single biggest gap in your current data.

| Dataset | Size | Task | Priority |
|---------|------|------|----------|
| **DocVQA** | 39K | QA on document images | HIGH |
| **ChartQA** | 28K | QA on charts/graphs | HIGH |
| **InfographicVQA** | 24K | QA on infographics | HIGH |
| **AI2D** | 15K | Science diagram MC questions | HIGH |
| **TextVQA** | 35K | VQA requiring reading text in images | HIGH |
| **ST-VQA** | 25K | Scene-text VQA | MEDIUM |
| **OCR-VQA** | 80K | VQA on book covers (OCR) | MEDIUM |
| **DVQA** | 20K | VQA on synthetic bar charts | MEDIUM |
| **FigureQA** | 1K | Yes/No QA on scientific figures | LOW |
| **PlotQA** | 160K (down-sample) | QA on scientific plots | LOW |
| **UReader (Caption/IE/KG/QA)** | 399K | OCR-free document understanding suite | MEDIUM |
| **Screen2Words** | 16K | Mobile UI summarization | MEDIUM |
| **VisualMRC** | 3K | Machine reading on web screenshots | LOW |
| **HiTab** | 2.5K | Hierarchical table QA | LOW |
| **RoBUT (SQA/WikiSQL/WTQ)** | 122K | Table understanding tasks | MEDIUM |

**Synthetic document data (from Molmo/PixMo-Docs):**

Generate programmatically using 7 libraries:
- **Matplotlib** / **Plotly**: Charts (bar, line, scatter, heatmap, violin, geographic, treemap, etc.)
- **LaTeX**: Scientific documents, equations, papers
- **HTML**: Web pages, menus, receipts, emails
- **Vega-Lite**: Interactive data visualizations
- **Mermaid**: Flowcharts, sequence diagrams, Gantt charts
- **Graphviz**: Network graphs, dependency trees

Use persona-driven diversity (see Section 5.2 below). Molmo generated 255K images with 2.3M QA pairs this way.

#### Category C: Math / Reasoning (~15-20% of mix)

| Dataset | Size | Task |
|---------|------|------|
| **Geo170K (QA + Align)** | 128K | Geometry problem solving |
| **MAVIS (Manual + Data Engine)** | 187K | Mathematical visual instruction |
| **MathV360K subsets** | ~42K | Geometry3K, GeoMVerse, GeoQA+, MapQA |
| **TabMWP** | 45K | Math word problems with tables |
| **CLEVR-Math** | 5K | Math on synthetic CLEVR scenes |
| **UniGeo** | 12K | Unified geometry datasets |
| **MathQA** | 30K | (Text-only but beneficial) Operation-based math |
| **Super-CLEVR** | 9K | Diagnostic visual reasoning |
| **InterGPS** | 1.3K | Geometry problem solving |

#### Category D: General OCR (~8-10% of mix)

| Dataset | Size | Task |
|---------|------|------|
| **HME100K** | 75K | Handwritten math expression recognition |
| **OCR-VQA** | 80K | Book cover text VQA |
| **TextCaps** | 22K | Captions requiring reading text |
| **SynthDog-EN** | 40K | Synthetic document OCR |
| **TextOCR-GPT4V** | 25K | GPT-4V annotated text reading |
| **Rendered Text** | 10K | Synthetic text rendering |
| **K12 Printing** | 13K | Educational document reading |
| **ChromeWriting** | 9K | Chrome-style text recognition |
| **IAM** | 6K | Handwriting recognition |
| **IIIT5K** | 2K | Scene text recognition |

#### Category E: Pure Language (~10-15% of mix)

Critical for preserving LLM text capabilities:

| Dataset | Size | Description |
|---------|------|-------------|
| **Magpie Pro (Llama-3 ST)** | 150K | High-quality synthetic text instruction |
| **Magpie Pro (Llama-3 MT)** | 150K | Multi-turn synthetic conversations |
| **Magpie Pro (Qwen2 ST)** | 150K | Qwen2-generated text instruction |
| **Tulu 3 SFT Mix** (10% subsample) | ~30K | Diverse text SFT from Ai2 |
| **Aya Dataset / Aya Collection** | 50-100K sample | Multilingual text instruction data |
| **FLAN** | 1M sample | Canonical instruction-following data |
| **Evo-Instruct** | 143K | Language understanding data |

**VILA finding**: Adding 1M text-only FLAN data during SFT **fully recovers** MMLU degradation from pre-training (from -5.3% to -0.3%) AND improves visual accuracy (+3.3% zero-shot, +2.3% four-shot).

**Molmo finding**: 10% down-sampled Tulu 3 text data improves both text benchmarks AND multimodal 11-avg (76.9 → 77.1).

---

## 3. Data Mixing Strategies

### 3.1 Sampling Rate Formulas

**Molmo approach — sqrt-proportional + manual adjustment:**
```
base_rate(dataset) = sqrt(dataset_size)
```
Then manually:
- **Down-weight** very large synthetic datasets (PlotQA, FigureQA, DVQA, PixMo-Clocks) to avoid overwhelming the mix.
- **Up-weight** tasks that learn slowly (pointing/grounding, counting).
- **Up-weight** high-quality human-annotated data over synthetic.

**Cambrian-1 approach — distribution balancing:**
Cambrian emphasizes that data source balancing and distribution ratio is the critical factor. They find:
- Too much of any single dataset category degrades other capabilities.
- The optimal ratio must be found empirically through ablation.
- "Data source balancing" (equalizing contribution from each source) generally works better than "instance balancing" (equal weight per example).

**LLaVA-OneVision approach — category budgets:**
Allocate fixed budgets per category, then fill within each:
```
General:         36.1% (~1.14M samples)
Doc/Chart/Screen: 20.6% (~647K)
Math/Reasoning:   20.1% (~632K)
General OCR:       8.9% (~281K)
Language:         14.3% (~450K)
Total:            3.2M samples
```

### 3.2 Practical Mix for Tiny Aya Vision

Here is a recommended concrete mix, sized for compute constraints similar to your current setup:

#### Starter Mix (~500K samples, minimal effort)

| Category | Datasets | Samples | % |
|----------|----------|---------|---|
| General VQA | VQAv2 (83K) + A-OKVQA (17K) + GQA (72K) + LLaVA-Instruct (150K) | 322K | 64% |
| Doc/Chart | DocVQA (10K) + ChartQA (18K) + TextVQA (35K) + AI2D (15K) | 78K | 16% |
| OCR | OCR-VQA (20K) + TextCaps (22K) | 42K | 8% |
| Language | Aya Dataset sample (50K) | 50K | 10% |
| Math | TabMWP (10K) | 10K | 2% |

#### Full Mix (~2M samples, recommended)

| Category | Datasets | Samples | % |
|----------|----------|---------|---|
| General VQA | VQAv2 (83K) + A-OKVQA (17K) + GQA (72K) + OK-VQA (9K) + LLaVA-158K + ShareGPT4V (91K) + ShareGPT4o (57K) + ALLaVA (70K) + Vision-FLAN (186K) + TallyQA (10K) + RefCOCO (51K) | 804K | 40% |
| Doc/Chart | DocVQA (39K) + ChartQA (28K) + InfographicVQA (24K) + AI2D (15K) + TextVQA (35K) + ST-VQA (25K) + DVQA (20K) + UReader-QA (50K) + Screen2Words (16K) | 252K | 13% |
| Synthetic Docs | Generated charts/tables/diagrams (100K) | 100K | 5% |
| OCR | OCR-VQA (80K) + TextCaps (22K) + SynthDog (40K) + TextOCR-GPT4V (25K) + HME100K (30K) | 197K | 10% |
| Math | Geo170K-QA (68K) + MAVIS (100K) + TabMWP (45K) + CLEVR-Math (5K) + UniGeo (12K) + MathQA (30K) | 260K | 13% |
| Language (text-only) | Magpie-Pro (150K) + Aya Dataset (100K) + Tulu-3 sample (50K) | 300K | 15% |
| Multilingual VQA | Translated VQAv2 (40K) + Translated LLaVA (40K) | 80K | 4% |

### 3.3 Style Tags vs. Natural Language Prompts

**Molmo's style tag system** (recommended for academic datasets):

Prefix academic dataset questions with dataset-specific tags to prevent their short-answer styles from contaminating conversational responses:
```
"vqa2: What is the person holding?"        → expects "tennis racket"
"chartqa: What is the value for 2020?"      → expects "42.5"
"docvqa: Who signed the document?"          → expects "John Smith"
```

For your custom data (PixMo-style, ShareGPT4V, conversational), use **no style tag** so the model defaults to natural, conversational responses.

For captioning, use diverse prompt templates (~30 variations):
```
"Describe this image in detail."
"Provide a comprehensive description of what you see."
"What's happening in this image? Be thorough."
...
```

### 3.4 Handling Different Answer Formats

Different datasets have conflicting answer formats. You need explicit formatting prompts:

| Format | Prompt Suffix | Datasets |
|--------|--------------|----------|
| Short answer | "Answer the question with a single word (or phrase)." | VQAv2, OK-VQA, TextVQA |
| Multiple choice (letter) | "Answer with the option letter from the given choices directly." | AI2D, A-OKVQA, ScienceQA |
| Free-form | (no suffix) | LLaVA-Instruct, ShareGPT4V |
| Caption | "Provide a one-sentence caption." OR "Describe this image in detail." | COCO Caps, re-captioning |
| Yes/No | "Answer the question with Yes or No." | FigureQA, POPE |
| Math | "Hint: Please provide the final answer at the end." | Geo, MAVIS, TabMWP |
| LaTeX OCR | "Please write out the expression in LaTeX format." | HME100K |

---

## 4. Data Quality Filtering & Cleaning

### 4.1 Pre-Filtering Strategies

**Perplexity filtering**: Run a language model on the text portion. Remove samples where the text perplexity is extremely high (gibberish) or extremely low (templatic boilerplate).

**CLIP score filtering**: For image-caption pairs, compute CLIP similarity. Remove pairs below a threshold (e.g., < 0.25). VILA subsampled COYO-700M to 25M by keeping the highest CLIP-similarity samples.

**Image quality filtering**:
- Remove images smaller than 100×100 pixels.
- Remove images with aspect ratios more extreme than 5:1.
- Remove images that are mostly blank/white/black (histogram check).
- Remove duplicate/near-duplicate images (perceptual hashing).

**Text quality filtering**:
- Remove samples where the answer is empty or just punctuation.
- Remove samples where the question and answer are identical.
- Remove samples with known toxic content (use a toxicity classifier).
- For machine-translated data: remove samples where back-translation diverges significantly (a sign of poor translation quality).

### 4.2 Decontamination

**Critical**: Remove training samples that overlap with evaluation benchmark test sets.

Following Tulu 3's decontamination protocol:
1. Extract all questions/answers from your evaluation benchmarks (MTVQA, AI2D test, ChartQA test, VQAv2 testdev, etc.).
2. Compute 13-gram overlaps between training data text and benchmark text.
3. Remove any training sample with a significant overlap.
4. Also decontaminate against image IDs — if a benchmark uses specific COCO images, ensure those images don't appear in your training mix with different annotations that could leak answers.

### 4.3 Reward Model-Based Filtering

IXC-2.5-Reward demonstrates using a **multi-modal reward model** to filter noisy training data:
1. Score each training sample with the reward model.
2. Remove the bottom N% (outliers/noise).
3. This improved both instruction following and open-ended dialogue quality.

You can approximate this without a dedicated reward model:
- Use an existing open VLM (InternVL2, LLaVA-OneVision) to answer the same question given the image.
- If the existing model's answer disagrees significantly with the "ground truth" in your training data, flag the sample for review.
- This catches mislabeled, ambiguous, or poor-quality samples.

### 4.4 ChartQA Rebalancing (Molmo's Trick)

ChartQA contains 21K synthetic + 7K human-annotated questions. The synthetic portion can be noisy. Molmo reweights so that the total weight of synthetic and human subsets are **equal**. This better matches the test distribution (50/50 split) and improves benchmark scores.

Apply the same principle to any dataset with mixed quality subsets — down-weight the noisy portions.

---

## 5. Synthetic Data Generation Pipelines

### 5.1 Caption-to-QA Pipeline (PixMo-CapQA)

The simplest synthetic pipeline — no image processing needed:

```
Image → Dense caption (from human or strong VLM)
      → Feed caption to text-only LLM
      → LLM generates diverse QA pairs from the caption
      → QA pairs become training data (paired with original image)
```

Molmo generated 214K QA pairs from 165K images this way. The QA pairs cover diverse topics and question styles because the LLM varies its output.

**Prompt for LLM**:
```
Given this image description, generate 2-3 diverse question-answer pairs.
Questions should vary between factual, inferential, and descriptive types.

Description: [dense caption]
```

### 5.2 Code-to-Image-to-QA Pipeline (PixMo-Docs)

More sophisticated — generates both the image and the QA:

```
Persona + Topic → LLM generates rendering code (Matplotlib/HTML/LaTeX/etc.)
              → Execute code to render image
              → Feed code (not image!) to LLM
              → LLM generates QA pairs from the code (privileged access)
```

The QA generation uses the **code** as ground truth, not the rendered image, so answers are guaranteed correct.

**Example topics**:
- "Generate a stacked bar chart showing quarterly revenue for 3 companies"
- "Generate a restaurant menu with Southern BBQ items"
- "Generate a LaTeX document showing a physics derivation"
- "Generate a Mermaid flowchart for a CI/CD pipeline"

Use **1B+ personas** (from the Scaling Synthetic Data paper) to control style/content:
- "A marine biologist studying coral reef ecosystems"
- "A Tokyo-based financial analyst focusing on Asian markets"
- "A kindergarten teacher in rural India"

This generates enormous diversity from a single template.

### 5.3 Object-Detection-to-Counting Pipeline (PixMo-Count)

Generate counting QA using an off-the-shelf detector:

```
Image → Run DETIC (or OWLv2, Grounding DINO) detector
      → Select class with most detections (confidence > threshold)
      → Generate QA: "How many [class] are in the image?" → "[count]"
      → Also include point annotations (object centers) for grounding
```

Manually verify a subset to create clean validation/test splits.

### 5.4 Clock/Gauge/Meter Rendering (PixMo-Clocks)

Specialized synthetic data for time/measurement reading:

```
Random time → Render clock face with diverse watch body (~50 styles) + face (~160K)
           → Generate QA: "What time does the clock show?" → "HH:MM"
```

Molmo's 826K synthetic clocks gave them **30× better** clock-reading accuracy than GPT-4V.

This generalizes: you can render **gauges, speedometers, thermometers, rulers** with random values and generate reading-comprehension QA.

### 5.5 SFT Data Augmentation Using Your Own Model (VILA² Approach)

VILA² proposes using the trained VLM to augment its own SFT data:

1. Take a fine-tuned VLM (after Phase 2).
2. Run it on **new images** (not in the training set) to generate detailed captions.
3. Use these captions as new pre-training data for the next training iteration.
4. Alternatively, generate QA pairs from these captions using the caption-to-QA pipeline.

This is a form of self-improvement / data flywheel — each iteration produces more training data for the next.

---

## 6. Multilingual Data Strategies

### 6.1 Translate-Train (Cheapest Approach)

Take existing English instruction data and translate it:

```
English (question, answer) + Image → Translate(question, answer) → Target Language (question, answer) + Image
```

**Translation models**:
- **NLLB-200** (Meta): 200 languages, open-source. Best for low-resource languages.
- **Aya-101** (Cohere): 101 languages, specifically tuned for instruction-following quality.
- **Tower** (Unbabel): High quality for European languages.
- **GPT-4o / Claude**: Highest quality, expensive. Use for smaller validation sets.

**What to translate (priority order)**:
1. LLaVA-Instruct-150K → 5-10 target languages × 150K = 750K-1.5M
2. ShareGPT4V answers → 5-10 target languages (rich, varied text)
3. VQAv2 short answers → good for benchmark-style training

**Quality check**: Back-translate a 1K sample and measure BLEU/chrF against the original English. Remove samples where back-translation quality is below threshold.

### 6.2 Multilingual Evaluation & Training Datasets

| Dataset | Languages | Task | Size |
|---------|-----------|------|------|
| **MTVQA** | 9 languages | Text-in-image VQA | Small (eval) |
| **xGQA** | 7 languages | Cross-lingual VQA | 12K+ |
| **MaXM** | 7 languages | Multilingual VQA | 22K |
| **XM3600** | 36 languages | Multilingual captioning | 3.6K (eval) |
| **Multi30K** | EN/DE/FR/CZ | Multilingual image descriptions | 31K |
| **Crossmodal3600** | 36 languages | Human-written captions | 3.6K (eval) |
| **Belebele** | 122 languages | Reading comprehension (text-only but useful) | 900/lang |
| **Aya Eval** | 100+ languages | Open-ended multilingual evaluation | — |

### 6.3 Multilingual Caption Generation

Generate captions in target languages directly:

1. Take diverse images (not just COCO — include culturally diverse sources).
2. Prompt GPT-4o: "Describe this image in detail in [language]."
3. Or prompt your own Tiny Aya model (since it's multilingual): "Describe this image in [language]."
4. Even 20K–50K per language makes a significant difference.

LLaVA-OneVision did this for Chinese only (92K samples with GPT-4V on ShareGPT4V images), and it improved Chinese VLM capabilities substantially.

### 6.4 Script-Specific OCR Training Data

For non-Latin scripts, generate synthetic text-in-image data:

```
Sample text → Render using fonts for target script(s)
           → Apply augmentations (rotation, blur, noise, background texture)
           → Generate OCR QA: "What text is shown?" → "[original text]"
```

**Scripts to prioritize** (based on Aya's language coverage):
- Arabic (Arabic script)
- Hindi/Marathi (Devanagari)
- Chinese/Japanese (CJK)
- Korean (Hangul)
- Thai (Thai script)
- Russian (Cyrillic)
- Bengali (Bengali script)

SynthDog (from Donut) already exists for English — generate analogues for other scripts.

### 6.5 Culturally Diverse Image Sources

| Source | Description | Why It Matters |
|--------|-------------|----------------|
| **Dollar Street** | Photos of everyday objects across income levels worldwide | Cultural diversity in household items, food, etc. |
| **GeoDE** | Geographically diverse images with location labels | Reduces Western bias in visual understanding |
| **Casual Conversations** | Meta's diverse face dataset | Diverse demographics |
| **WikiCommons** | CC-licensed images from Wikipedia | Diverse cultural/geographic content |
| **Flickr30K Entities** | User-uploaded photos with annotations | Diverse real-world scenes |
| **Open Images** | Google's 9M+ image dataset | Massive diversity, includes classes rare in COCO |

---

## 7. Multi-Image and Video Data

### 7.1 Multi-Image Instruction Data

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| **Spot-the-Diff** | 11K | Find differences between image pairs | LLaVA-NeXT-Interleave |
| **Birds-to-Words** | 14K | Describe differences between bird images | LLaVA-NeXT-Interleave |
| **NLVR2** | 86K | Natural language visual reasoning on pairs | LLaVA-NeXT-Interleave |
| **VIST** | 26K | Visual storytelling from image sequences | — |
| **ContrastiveCaption** | 25K | Caption what makes an image unique vs. a set | — |
| **ImageCode** | 17K | Select correct image from text description | — |
| **DreamSim** | 16K | Judge perceptual similarity between images | — |
| **CLEVR-Change** | 4K | Describe changes between CLEVR scene variations | — |
| **HQ-Edit-Diff / MagicBrush-Diff** | 14K | Describe image edit transformations | — |
| **Co-Instruct** | 50K | Compare/contrast image pairs | — |

### 7.2 Video Instruction Data

| Dataset | Size | Task |
|---------|------|------|
| **ShareGPT4Video** | 255K | GPT-4o captioned video QA (high quality) |
| **ActivityNet-QA** | 6.5K | Human activity QA |
| **Charades** | 24K | Daily activity recognition |
| **NextQA** | 10K | Temporal/causal reasoning |
| **Youcook2** | 42K | Cooking procedure understanding |
| **Ego4D** | 0.8K | Egocentric video understanding |

LLaVA-OneVision's OneVision stage uses 350K video samples, 560K multi-image, and 800K single-image (sampled from the 3.2M) = 1.6M total.

---

## 8. Data Processing & Formatting Best Practices

### 8.1 Multi-Annotation Training (Molmo)

When multiple QA pairs share the same image (common in VQAv2, GQA, PlotQA):

```python
# Instead of: N separate forward passes with same image
# Do: 1 forward pass with all N annotations concatenated

# Mask attention so each annotation only sees:
# 1. Image tokens (shared)
# 2. Its own text tokens
# NOT other annotations' tokens
```

This reduces processed images by 2/3 and training time by over half.

### 8.2 Multiple Choice Formatting

For MC datasets (AI2D, A-OKVQA, ScienceQA), use consistent formatting:

```
Question: [question text]
Choices:
A) [option 1]
B) [option 2]
C) [option 3]
D) [option 4]

→ Model predicts letter only: "B"
```

### 8.3 Handling Multiple Valid Answers

VQAv2 has multiple answers per question (e.g., ["tennis racket", "racket", "bat"]). Following Molmo: use only the **most common answer**. If tied, randomly select among tied answers each epoch.

### 8.4 Sequence Length Management

Set a maximum sequence length (e.g., 2048 tokens) and truncate longer examples. In practice, truncation mainly affects:
- DVQA/PlotQA (many annotations per image with multi-annotation batching)
- Very long ShareGPT4V/4o conversations
- Long document OCR outputs

### 8.5 Image Deduplication Across Datasets

Many datasets share the same underlying images (especially COCO). When mixing:
- VQAv2, OK-VQA, A-OKVQA, RefCOCO, COCO Captions all use COCO images.
- GQA uses Visual Genome images.
- DocVQA, InfographicVQA have unique document images.

Deduplication matters for multi-annotation training (merge all annotations for the same image) and for preventing data leakage.

---

## 9. Data-Centric Ablation Framework

### 9.1 Fast Ablation Protocol

Adapted from Molmo and LLaVA-OneVision:

1. **Proxy model**: Use a small model (e.g., 0.5B LLM + SigLIP) for data ablations. LLaVA-OneVision uses SO400M-Qwen-1.5-0.5B for all data validation.
2. **Proxy metric**: Use captioning F1 (Molmo's approach) or a small subset of benchmarks (AI2D, ChartQA, DocVQA, MME) — evaluatable in minutes.
3. **Incremental addition**: Start with a strong baseline mix. Add one dataset at a time and measure impact. If adding a dataset degrades performance, investigate formatting/quality issues.
4. **Category budgets**: Fix category percentages, then vary datasets within each category to find the best composition.

### 9.2 What to Ablate (Ranked by Expected Impact)

1. **Pre-training data**: Dense captions vs. BLIP captions (largest single impact).
2. **SFT mix size**: 150K vs. 500K vs. 1M vs. 2M (more is generally better up to ~3M).
3. **Document/chart data**: With vs. without DocVQA/ChartQA/TextVQA (critical for those benchmarks).
4. **Text-only data ratio**: 0% vs. 5% vs. 10% vs. 15% (recover text capabilities).
5. **Multilingual data**: 0% vs. 5% vs. 10% (measure impact on MTVQA and language-specific benchmarks).
6. **Synthetic data quality**: Filtered vs. unfiltered synthetic docs/charts.
7. **Math data**: With vs. without geometry/math data (impacts MathVista, MMMU).

---

## 10. Concrete Dataset Download & Implementation Plan

### Step 1: Download Core Datasets

```bash
# HuggingFace datasets (use huggingface_hub)
pip install datasets huggingface_hub

# Core VQA datasets
python -c "
from datasets import load_dataset
load_dataset('HuggingFaceM4/VQAv2', split='train')
load_dataset('lmms-lab/DocVQA', split='train')
load_dataset('lmms-lab/ChartQA', split='train')
"

# LLaVA data (already in your setup)
# liuhaotian/LLaVA-Instruct-150K — already have
# liuhaotian/LLaVA-Pretrain — already have

# Dense captions
# huggingface.co/datasets/Lin-Chen/ShareGPT4V
# huggingface.co/datasets/allenai/pixmo-cap
```

### Step 2: Format Converter

Build a unified dataset loader that:
1. Reads from multiple formats (JSON, Parquet, JSONL).
2. Normalizes to your LLaVA conversation format: `[{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]`.
3. Applies formatting prompts per dataset.
4. Applies style tags per dataset.
5. Handles image path resolution across different dataset structures.

### Step 3: Mixing & Sampling

```python
# Pseudocode for the mixing strategy
import math

def compute_sampling_rates(datasets):
    """Sqrt-proportional sampling with manual adjustments."""
    raw_rates = {name: math.sqrt(len(ds)) for name, ds in datasets.items()}
    total = sum(raw_rates.values())
    rates = {name: r / total for name, r in raw_rates.items()}
    
    # Manual overrides
    rates['plotqa'] *= 0.3      # down-weight large synthetic
    rates['figureqa'] *= 0.3
    rates['dvqa'] *= 0.5
    rates['pointing'] *= 2.0    # up-weight slow-learning tasks
    
    # Renormalize
    total = sum(rates.values())
    return {name: r / total for name, r in rates.items()}
```

### Step 4: Validation

Before full training:
1. Sample 100 examples from each dataset category.
2. Visually inspect image-question-answer triplets.
3. Verify format conversions are correct.
4. Check that style tags are properly applied.
5. Run a quick 1K-step training to verify the data pipeline doesn't crash.

---

## References

| Paper | Key Data Insight |
|-------|-----------------|
| **Molmo/PixMo** (Ai2) [arXiv:2409.17146](https://arxiv.org/abs/2409.17146) | Sqrt-proportional sampling, style tags, PixMo-Cap > ShareGPT4V, PixMo-Docs via code generation, multi-annotation training |
| **LLaVA-OneVision** [arXiv:2408.03326](https://arxiv.org/abs/2408.03326) | 3.2M single-image mix across 5 categories, Stage 1.5 knowledge learning with 4M re-captions, progressive data scaling |
| **MM1** (Apple) [arXiv:2403.09611](https://arxiv.org/abs/2403.09611) | Optimal pre-training ratio: 45% interleaved, 45% caption, 10% text. Data mix matters more than model architecture. |
| **MM1.5** (Apple) [arXiv:2409.20566](https://arxiv.org/abs/2409.20566) | High-quality OCR data + synthetic captions for continual pre-training. Optimized visual instruction-tuning data mixture. |
| **Cambrian-1** [arXiv:2406.16860](https://arxiv.org/abs/2406.16860) | Data source balancing > instance balancing. Publicly available data can match proprietary with careful curation. |
| **VILA** (NVIDIA) [arXiv:2312.07533](https://arxiv.org/abs/2312.07533) | Interleaved data essential, text-only SFT data recovers LLM degradation AND boosts VLM accuracy. |
| **Tulu 3** (Ai2) [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) | Aggressive decontamination, on-policy preference data, skill-specific synthetic data at each stage. |
| **IXC-2.5-Reward** [arXiv:2501.12368](https://arxiv.org/abs/2501.12368) | Reward model for data filtering — remove outlier/noisy samples from instruction tuning data. |
| **Scaling Synthetic Data with 1B Personas** [arXiv:2406.20094](https://arxiv.org/abs/2406.20094) | Persona-driven diversity for synthetic data generation at scale. |
| **ShareGPT4V** [arXiv:2311.12793](https://arxiv.org/abs/2311.12793) | GPT-4V distilled captions as high-quality pre-training data. |
| **OBELICS** [arXiv:2306.16527](https://arxiv.org/abs/2306.16527) | 141M interleaved web documents for multimodal pre-training. |
| **Vision-FLAN** [arXiv:2402.11690](https://arxiv.org/abs/2402.11690) | 191 tasks with human-labeled instructions, no synthetic annotation. |
