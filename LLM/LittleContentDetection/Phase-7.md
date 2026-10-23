# Phase 7 — Evaluation, Calibration, and Threshold Selection

## Goal
Measure whether the fine-tuned model catches Little Content without flagging too many useful pages.

## 1. Confusion matrix
- **TP:** low-content page correctly flagged
- **FN:** low-content page missed
- **FP:** useful page incorrectly flagged
- **TN:** useful page correctly retained

## 2. Core metrics
`Recall = TP/(TP+FN)`  
`Precision = TP/(TP+FP)`  
`FPR = FP/(FP+TN)`  
`F1 = 2*Precision*Recall/(Precision+Recall)`

Prefer PR curves/PR-AUC when the positive class is uncommon. Accuracy can hide many missed positives.

## 3. Worked example (illustrative only)
Out of 1,000 pages: TP=180, FN=20, FP=80, TN=720.

- Recall = 90%
- Precision ≈ 69.2%
- FPR = 10%
- Accuracy = 90%

A 90% accuracy number alone would hide the cost of 80 false positives.

## 4. Getting a score
A generated label is not inherently a probability. Options:
1. Use constrained decoding and normalized log-likelihood of each allowed label sequence.
2. Train a dedicated classifier head.
3. Use another validated scoring mechanism.

For multi-token labels, calculate full sequence likelihood consistently. Validate score ranking and calibration on held-out validation data.

## 5. Choosing the threshold
For each threshold, compute recall, precision, and FPR on validation data. Choose the threshold with the highest recall **subject to required guardrails** (e.g., minimum precision and maximum FPR). Lock it before evaluating the test set.

Do not use the test set repeatedly to adjust threshold or prompts.

## 6. Calibration
If downstream consumers need actual probabilities, evaluate reliability diagrams, Brier score, and/or expected calibration error. Calibrate only with appropriate held-out data.

## 7. Slice analysis
Report metrics by language, device width, host/site family, page type, capture quality, and screenshot resolution. Inspect hard negatives and hard positives. Provide confidence intervals, especially for small slices.

## 8. Failure policy
If the image cannot be decoded or the model returns an invalid output, emit a distinct **operational error/abstain status** rather than silently labeling it negative.

## 9. Acceptance gate
Approve only when quality guardrails, operational reliability, and cost constraints all pass on representative held-out data. Numerical targets from Phase 1 are proposals, not achieved outcomes.

## Deliverables
Evaluation report, confusion matrices, PR curves, calibration analysis, selected threshold, and error taxonomy.

## Interview recall
**Q: Why maximize recall under a precision constraint?**  
A: We want to catch more bad pages without incorrectly flagging too many good pages.

**Next:** Phase 8 — optimize model serving.
