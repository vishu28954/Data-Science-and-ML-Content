# Phase 9 — Production Batch Deployment

## Goal
Run the fine-tuned model reliably on large screenshot collections and store traceable predictions.

## 1. Proposed architecture
```text
Screenshot object store
       ↓
Input manifest / job scheduler
       ↓
Queue or distributed batch partitions
       ↓
GPU inference workers (Qwen + LoRA)
       ↓
Output validation + scoring + threshold
       ↓
Versioned predictions store
       ↓
Downstream search-quality consumers
```

## 2. Input contract
```json
{
  "sample_id": "sample_0001",
  "image_uri": "s3://private-bucket/captures/sample_0001.png",
  "capture_time": "2026-10-01T10:00:00Z"
}
```

## 3. Output contract
```json
{
  "sample_id": "sample_0001",
  "model_version": "qwen25vl3b-lora-v1",
  "prompt_version": "v1",
  "score_version": "v1",
  "little_content_score": 0.91,
  "threshold": 0.70,
  "label": "LITTLE_CONTENT",
  "status": "success"
}
```
The score and threshold are **illustrative** and should not be interpreted as calibrated probabilities unless validated.

## 4. Worker behavior
1. Read an input batch and validate image references.
2. Load images with access controls and size limits.
3. Apply the pinned Qwen processor.
4. Run inference with bounded batch size.
5. Validate labels and calculate scores.
6. Write idempotent outputs keyed by sample ID and model version.
7. Log operational metrics without leaking screenshot content.

## 5. Deployment choices
Containerize the inference worker. Run on a GPU batch platform (for example, Kubernetes Jobs or a managed batch service). Store models/adapters in a versioned registry and pin exact revisions. The exact cloud provider is an implementation decision.

## 6. Operational safeguards
- Separate malformed inputs from model uncertainty.
- Retry transient failures; quarantine permanent failures.
- Use checkpointed progress and dead-letter queues.
- Protect image access with least-privilege credentials.
- Encrypt storage and transport; set retention policies.
- Roll out with shadow evaluation/canary batches before replacing a previous model.
- Maintain a rollback path.

## 7. Acceptance tests
Test duplicate job replay, corrupted image, missing image, OOM recovery, partial batch failure, model rollback, output schema, and throughput under load.

## Deliverables
Deployment diagram, worker/container specification, job manifest, output schema, runbook, and rollout checklist.

## Interview recall
**Q: Why batch instead of real-time?**  
A: Large-scale offline page assessment often values throughput and cost over per-request latency.

**Next:** Phase 10 — monitoring and retraining.
