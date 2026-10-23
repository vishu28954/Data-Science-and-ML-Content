# Phase 10 — Monitoring, Feedback, and Retraining

## Goal
Keep the Little Content detector reliable after deployment as webpages and traffic change.

## 1. What to monitor
**Operational:** input volume, queue backlog, images/sec, GPU utilization, memory, failure/retry rate, job completion time, and cost.

**Prediction behavior:** predicted-positive rate, score distribution, invalid output rate, and changes by language, page type, device, and host.

**True quality:** precision, recall, and FPR on newly human-labeled samples. Prediction distributions alone cannot prove accuracy.

## 2. Drift
- **Data drift:** screenshots look different (new layouts, devices, languages).
- **Label/concept drift:** what counts as sufficient content changes.
- **Pipeline drift:** capture resolution, rendering, or preprocessing changes.

Use alerts as signals to investigate, not automatic evidence that the model has failed.

## 3. Human feedback loop
Sample predicted positives, predicted negatives, low-confidence cases, and high-impact segments for review. Keep a representative random sample so evaluation is not biased toward model-flagged pages.

## 4. Retraining triggers
Consider retraining when a representative labeled evaluation set shows sustained quality deterioration, a major new page type appears, or annotation policy changes. Do not retrain only because predicted-positive rate moved.

## 5. Safe retraining workflow
1. Collect new approved labeled examples.
2. Update dataset and annotation versions.
3. Recheck duplicates and leakage.
4. Fine-tune a candidate adapter.
5. Compare with the deployed model on frozen and recent holdouts.
6. Verify recall, precision/FPR, throughput, and cost.
7. Shadow-test or canary.
8. Promote with rollback available.

## 6. Dashboard example
| Metric | Purpose | Action if abnormal |
| --- | --- | --- |
| Batch success rate | Reliability | Inspect input failures, worker errors |
| Images/sec | Throughput | Review batching, resolution, GPU saturation |
| Predicted-positive rate | Distribution shift | Check traffic mix and capture changes |
| Recall on audited set | Missed low-content pages | Review false negatives and labels |
| FPR on audited set | Harm to useful pages | Review false positives, threshold |
| Cost per 1,000 images | Efficiency | Rebenchmark model/precision/batching |

## 7. Governance
Version models, adapters, thresholds, prompts, processors, datasets, and annotation rubrics. Limit screenshot access, define retention, and avoid logging sensitive page content.

## 8. Project completion checklist
- [ ] Dataset and labeling policy reviewed
- [ ] Leakage-safe test set frozen
- [ ] Baselines benchmarked
- [ ] Qwen SFT experiment reproducible
- [ ] Recall and guardrails validated
- [ ] Optimized inference re-evaluated
- [ ] Batch jobs are idempotent and recoverable
- [ ] Monitoring and rollback are operational
- [ ] Retraining criteria documented

## Interview recall
**Q: How do you know the model is still good in production?**  
A: Audit fresh human-labeled samples; monitor recall and false positives, not just prediction counts.

**Q: When do you retrain?**  
A: When evidence shows meaningful degradation or the definition of the task changes.

## Final perspective
The project is not complete when LoRA training finishes. It is complete when we can **measure, deploy, monitor, and safely improve** the model over time.
