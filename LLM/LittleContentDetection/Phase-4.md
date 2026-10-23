# Phase 4 — Baselines and Experimental Design

## Goal
Prove that a large VLM adds enough value compared with simpler, cheaper methods.

## 1. Baselines
1. **Majority-class:** always predict the most common class. Sanity check only.
2. **Heuristics:** visible text area, blank-area fraction, OCR text count, or layout signals.
3. **Image embeddings + linear classifier:** CLIP-like encoder plus logistic regression.
4. **Zero-shot Qwen2.5-VL-3B-Instruct:** prompt without task-specific training.
5. **Few-shot Qwen:** supply a small number of examples if context and cost allow.

OCR and heuristics are baseline features, not inputs to our final screenshot-only Qwen model.

## 2. Fixed evaluation protocol
Use the same frozen test samples and label definitions. Set model version, prompts, random seeds, image processor settings, and decoding parameters. Compare models at a **common precision or FPR constraint**, not at arbitrary default thresholds.

## 3. Key metrics
`Recall = TP/(TP+FN)`  
`Precision = TP/(TP+FP)`  
`FPR = FP/(FP+TN)`

Also report PR-AUC, confusion matrix, per-segment recall, throughput (images/sec), peak GPU memory, and estimated cost per 1,000 images. Include confidence intervals for key metrics where feasible.

## 4. Controlled prompt example
```text
You are evaluating visible webpage content.
Classify the screenshot using one label only:
LITTLE_CONTENT or NOT_LITTLE_CONTENT.
Do not explain your answer.
```

For generative models, enforce valid outputs and record invalid-label rate. A verbal answer is not a calibrated probability.

## 5. Error analysis
Review at least these categories: blank/loading pages, short-but-useful interfaces, image-heavy pages, non-English pages, mobile layouts, popups/overlays, and missing below-the-fold content. Count false positives and false negatives separately.

## 6. Experiment tracker schema
```text
experiment_id | model | prompt_version | image_budget |
dataset_version | split_version | recall | precision | FPR |
PR_AUC | images_per_sec | GPU_hours | notes
```

## 7. Decision gate
Advance to fine-tuning only if there is a meaningful gap between the zero-shot Qwen baseline and the desired quality targets, and a credible reason Qwen can outperform a cheaper baseline.

## Deliverables
Baseline leaderboard, fixed evaluation script, error taxonomy, and documented model-selection rationale.

## Interview recall
**Q: Why not start directly with LoRA?**  
A: Without a baseline, we cannot tell whether fine-tuning improved the system enough to justify its cost.

**Next:** Phase 5 — understand the Qwen architecture.
