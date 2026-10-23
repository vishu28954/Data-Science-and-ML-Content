# Phase 3 — Preprocessing and Data Splitting

## Goal
Turn raw screenshots and human labels into reliable train, validation, and test sets.

## 1. Validate before training
Verify that each image can be decoded, has reasonable dimensions, is a completed capture, and has a valid label. Remove or quarantine failed captures; log why each row was excluded.

## 2. Standardize carefully
Use consistent screenshot capture settings where possible. Keep aspect ratio; avoid blindly stretching images. Let the official Qwen processor handle model-specific image resizing and patching. Compare image-resolution budgets because small webpage text may disappear when downsampled.

Avoid preprocessing that destroys useful signals (e.g., cropping away blank regions, which might itself be evidence of little content).

## 3. Deduplicate
- **Exact duplicates:** hash raw image bytes.
- **Near duplicates:** perceptual hashing or image embeddings.
- **Template families:** group pages with similar layout or the same host/site family.
- **Temporal repeats:** track repeated captures of the same URL across days.

Review near-duplicate thresholds manually before removing examples.

## 4. Split without leakage
A random image-level split can leak near-identical pages across train and test. Prefer grouped splits by site/host and near-duplicate family; consider a time-based holdout to measure drift. Ensure no group appears in more than one split.

A starting allocation might be 80/10/10, but the final split depends on host diversity and the precision of the evaluation estimates. Do not tune thresholds on the test set.

## 5. Class balance
If positives are rare, try weighted sampling for training, but preserve a representative validation/test distribution (or use a separately labeled representative set). Never infer deployment precision from an artificially balanced test set.

## 6. Model-ready example
```python
example = {
    "image_path": "/private/captures/0001.png",
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": "/private/captures/0001.png"},
            {"type": "text", "text": "Classify this webpage as LITTLE_CONTENT or NOT_LITTLE_CONTENT."}
        ]},
        {"role": "assistant", "content": "LITTLE_CONTENT"}
    ]
}
```
Adapt the example to the installed Qwen processor and training library's expected chat template; do not assume this illustrative Python structure is a drop-in training record.

## 7. Checks to automate
- Split overlap count = 0 for grouping keys.
- Every image opens successfully.
- Label values are valid.
- Class counts and prevalence are reported per split.
- Resolution, language, and site distributions are compared.
- Preprocessing settings are version-controlled.

## Deliverables
Immutable dataset manifest, split IDs, deduplication report, preprocessing configuration, and data-quality tests.

## Interview recall
**Q: Why group by host?**  
A: Pages from one website often share templates; putting them in both training and test sets can inflate results.

**Next:** Phase 4 — build simple baselines before fine-tuning Qwen.
