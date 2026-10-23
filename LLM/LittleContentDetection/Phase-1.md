# Phase 1 — Problem Formulation and Model Selection

**Project:** Little Content Detection using Qwen2.5-VL  
**Status:** Design specification (not measured production results)  
**Goal:** Build a screenshot-based model that identifies webpages containing too little meaningful content.

## 1. Problem statement

A search engine processes many webpages. Some pages load successfully but contain very little useful information: mostly empty areas, placeholder text, or missing main content. We want to identify these pages to support search-quality assessment.

**Important:** A visually simple webpage is not always a bad webpage. A calculator, search homepage, login screen, or short answer page can still serve its purpose. Our labeling rules must reflect usefulness, not just the number of words.

**Business objective:** Find as many truly low-content pages as possible without incorrectly flagging too many useful pages. Model predictions are signals for downstream quality systems, **not automatic decisions to remove or penalize pages**.

## 2. Selected project configuration

| Item | Initial choice | Why |
| --- | --- | --- |
| Model | Qwen2.5-VL-3B-Instruct | Can understand both visual layout and text in screenshots; relatively manageable model size |
| Task | Binary classification | Clear starting point for the quality signal |
| Input | Webpage screenshot | Captures rendered content and layout |
| Output | `LITTLE_CONTENT` or `NOT_LITTLE_CONTENT` | Simple, consistent labels |
| Training | LoRA-based supervised fine-tuning (SFT) | Adapts a pretrained model with fewer trainable parameters |
| Inference | Batch | Appropriate starting architecture for large collections of webpages |
| Primary metric | Recall | We want to miss fewer truly low-content pages |
| Guardrails | Precision and false-positive rate | We must avoid incorrectly flagging useful pages |

These are **design choices for the reconstructed project**, not verified details of the original Microsoft implementation.

## 3. What exactly are we predicting?

Given screenshot `X`, predict one of two classes:

- **1 — LITTLE_CONTENT:** The page has insufficient meaningful visible content under our labeling policy.
- **0 — NOT_LITTLE_CONTENT:** The page provides sufficient meaningful content for its intended purpose.

In notation:

`f(X) → score for LITTLE_CONTENT → threshold → final label`

The score is not automatically a calibrated probability just because it comes from a generative VLM. We must define how to obtain a classification score (for example, from constrained label-token likelihoods) and check calibration on validation data if probability interpretation is required.

### Example model instruction

```text
Look at this webpage screenshot.
Decide whether the page contains too little meaningful content
for its apparent purpose.
Return exactly one label:
LITTLE_CONTENT
NOT_LITTLE_CONTENT
```

A consistent output format makes batch processing and evaluation easier.

## 4. Why choose a Vision-Language Model (VLM)?

A screenshot contains both **visual** and **textual** information. The model may need to understand:

- Empty spaces versus actual content
- Headlines, paragraphs, and readable text
- Images and other meaningful visual elements
- Loading indicators and placeholders
- Overall page structure and purpose

A VLM can consider these signals together. However, a simpler image classifier might be cheaper and good enough. We must compare the VLM against simpler baselines rather than assume it will win.

## 5. Why Qwen2.5-VL?

We chose Qwen2.5-VL as our **starting candidate**, for four reasons:

1. **Image + text understanding:** Useful when classifying webpage screenshots with mixed visual and written content.
2. **Flexible image resolution:** We can trade off fine detail against the number of visual tokens and compute cost.
3. **Fine-tuning ecosystem:** PyTorch, Transformers, and PEFT/LoRA support practical experimentation.
4. **Open-weight deployment:** We can adapt and self-host the model, rather than depend entirely on an external API.

These are reasons to **test** Qwen, not proof that it is the best model for our dataset.

### Why the 3B size?

The Qwen2.5-VL family includes **3B, 7B, 32B, and 72B Instruct** models. Larger models may perform better on difficult examples but generally need more compute.

The 3B model is a practical starting point for repeated fine-tuning experiments and batch inference. We can evaluate the 7B model if the 3B model fails to meet our quality targets.

Approximate **weight-only** storage for a 3-billion-parameter model:

| Precision | Approximate weight storage |
| --- | --- |
| FP32 | 12 GB |
| FP16 / BF16 | 6 GB |
| 8-bit | 3 GB |
| 4-bit | 1.5 GB |

Actual GPU memory needs are higher due to activations, image processing, caches, and runtime overhead.

## 6. Why the Instruct version instead of a Base model?

Think of the difference simply:

- **Base model:** Has learned patterns from large amounts of training data, but is not necessarily trained to reliably follow instructions.
- **Instruct model:** Has received additional training to understand and respond to requests.

Our task is: **“Look at this screenshot and answer with one of two labels.”** An Instruct model is a natural starting point for that format.

We still fine-tune it because **following instructions** and **knowing our specific definition of Little Content** are different skills.

- Earlier instruction tuning helps the model follow the classification request.
- Our task-specific SFT teaches the model to match our human labels.
- LoRA updates small adapter matrices while keeping the pretrained model weights frozen in the standard setup.

**Caution:** Instruct is not always superior to Base for every classifier. A Base model with a dedicated classification head may work well. We should not assume an official, matching 3B Base checkpoint is available without checking.

## 7. Alternative models and baselines

| Candidate | Why test it? | Main trade-off |
| --- | --- | --- |
| Simple layout/heuristic rules | Very fast, inexpensive baseline | Can confuse sparse-but-useful pages with low-content pages |
| CLIP embeddings + classifier | Efficient discriminative baseline | May miss some fine-grained text and purpose-related cues |
| PaliGemma 2 | Another fine-tunable VLM | Needs head-to-head task evaluation |
| Qwen2.5-VL-3B-Instruct | Our initial VLM | More costly than small classifiers |
| Qwen2.5-VL-7B-Instruct | Possible quality improvement | Higher memory and inference cost |

**Model selection rule:** Prefer the lowest-cost candidate that meets recall **and** the required precision/false-positive-rate guardrails on representative held-out data.

## 8. Why recall is our priority

**Recall** answers: *Of all pages that really have Little Content, how many did we catch?*

`Recall = True Positives / (True Positives + False Negatives)`

**Precision** answers: *Of all pages we flagged, how many truly have Little Content?*

`Precision = True Positives / (True Positives + False Positives)`

**False-positive rate** answers: *Of all good pages, how many did we incorrectly flag?*

`FPR = False Positives / (False Positives + True Negatives)`

### Worked example (hypothetical)

Suppose 1,000 pages contain 200 true low-content pages and 800 other pages.

| | Predicted Little Content | Predicted Not Little Content |
| --- | ---: | ---: |
| Actually Little Content | 180 (TP) | 20 (FN) |
| Actually Not Little Content | 80 (FP) | 720 (TN) |

- Recall = 180 / 200 = **90%**
- Precision = 180 / 260 ≈ **69.2%**
- Accuracy = (180 + 720) / 1000 = **90%**
- FPR = 80 / 800 = **10%**

Despite 90% accuracy, 80 useful pages were incorrectly flagged. That is why accuracy alone is insufficient.

### Initial, illustrative targets

- Recall: **at least 95%**
- Precision: **at least 85%**
- FPR: **at most 3%**
- Batch processing success rate: **at least 99.9%**

These are **proposed targets, not achieved results**. They may not all be feasible simultaneously. We will revise them using baseline experiments and downstream business costs.

### Threshold selection

A lower positive-class threshold often improves recall but can increase false positives. We will choose the threshold on a **validation set**, subject to our guardrails, and report final performance only on an untouched test set.

## 9. Important assumptions and unresolved decisions

| Area | Current assumption / open question |
| --- | --- |
| Screenshot coverage | **Open:** initial viewport, full page, or multiple screenshots |
| Screenshot consistency | Standardize viewport, rendering, and capture timing where practical |
| Model input | Screenshot only; no URL or HTML metadata supplied to the model |
| Labeling | Human-annotated examples with clear guidelines |
| Sparse useful pages | Should not automatically be labeled low-content |
| Loading/error pages | Need explicit handling or exclusion rules |
| Ambiguous examples | Human adjudication and an uncertainty policy |
| Dataset splitting | Avoid leakage from near-duplicate pages, hosts, and templates |
| Inference | Batch processing with retries and error tracking |
| Downstream use | Quality signal, not automatic page removal |

The **screenshot coverage policy** is especially important. An initial viewport may miss content farther down the page; full-page screenshots may be harder to process efficiently. We must decide this before finalizing the dataset.

## 10. What does success look like?

A successful Phase 1 design defines:

1. A precise and testable meaning of Little Content.
2. A clear screenshot input contract.
3. Two unambiguous output labels, plus a policy for invalid/uncertain inputs.
4. Recall as the main optimization goal, with precision and FPR guardrails.
5. A model-selection plan comparing Qwen with simpler alternatives.
6. Constraints for batch throughput, GPU memory, reliability, and cost.
7. A human-labeled evaluation strategy.

## 11. End-to-end project roadmap

1. **Phase 1 — Problem formulation and model selection** (this document)
2. **Phase 2 — Dataset collection and annotation**
3. **Phase 3 — Preprocessing and leakage-safe splitting**
4. **Phase 4 — Baseline experiments**
5. **Phase 5 — Qwen2.5-VL architecture**
6. **Phase 6 — LoRA-based SFT**
7. **Phase 7 — Evaluation, calibration, and threshold selection**
8. **Phase 8 — Model and inference optimization**
9. **Phase 9 — Batch deployment**
10. **Phase 10 — Monitoring, feedback, and retraining**

## 12. Interview-ready explanation

**Why Qwen2.5-VL-3B-Instruct?**

> We chose Qwen2.5-VL because detecting low-content webpages from screenshots requires understanding both the visual layout and the text. We started with the 3B Instruct version because it can follow our classification prompt and is relatively manageable to fine-tune and deploy. We use LoRA-based SFT to teach it our specific definition of Little Content. We would validate the choice against simpler classifiers and larger VLMs, comparing recall, false positives, and inference cost.

**Why fine-tune an Instruct model?**

> The Instruct model already understands how to follow a request, but it does not automatically know our company's exact labeling rules. Fine-tuning teaches it to apply those rules consistently.

## Next step: Phase 2

Before collecting training examples, decide whether each sample is a **single initial viewport, a full-page screenshot, or multiple viewport screenshots**. Then define the annotation rubric, dataset sampling strategy, class balance, and train/validation/test splitting policy.
