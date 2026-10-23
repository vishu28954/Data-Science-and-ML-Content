# Phase 8 — Inference and Model Optimization

## Goal
Process screenshots efficiently without losing the recall we gained during training.

## 1. Cost drivers
GPU memory, image token count, batch size, output token count, model precision, and input/output overhead determine cost. Since we output only one label, keep generation short.

## 2. Optimization options
- **Dynamic batching:** combine compatible requests to improve GPU use.
- **Image resolution budget:** benchmark readability against speed.
- **Quantization:** 8-bit or 4-bit weights can reduce memory, but may change predictions.
- **LoRA merge or adapter loading:** benchmark both supported serving paths.
- **Constrained output:** avoid long free-form explanations.
- **Caching/deduplication:** skip exact repeat screenshots where policy permits.
- **Multiple workers:** partition batches and retry failures safely.

Verify that the chosen serving engine supports the exact Qwen2.5-VL model, image input format, quantization scheme, and adapter configuration at the pinned version.

## 3. Benchmark protocol
Hold hardware, screenshot distribution, and model version constant. Test several batch sizes and resolution budgets. Report:
- images/sec
- p50/p95 processing latency per image or batch
- peak GPU memory
- invalid-output rate
- quality metrics after optimization
- estimated cost per 1,000 images

## 4. Trade-off example
Lower image resolution may double throughput but hide small text, causing false positives or negatives. A speed improvement is not acceptable if it violates quality guardrails.

## 5. Capacity planning
`required GPU-hours ≈ number_of_images / (measured_images_per_second * 3600)`

Example: 1 million screenshots at 10 images/sec ≈ 27.8 GPU-hours **before** retries, overhead, and utilization losses. This is an illustrative calculation, not a measured Qwen throughput.

## 6. Reliability
Use bounded queues, batch size limits, backpressure, retries with exponential backoff, checkpointing, and idempotent output writes. Treat out-of-memory errors separately from invalid images.

## Deliverables
Benchmark matrix, chosen image budget, precision setting, batch policy, GPU cost model, and rollback criteria.

## Interview recall
**Q: Why not always use 4-bit quantization?**  
A: It saves memory, but must be tested for accuracy and runtime compatibility.

**Next:** Phase 9 — batch deployment.
