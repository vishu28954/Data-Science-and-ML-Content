# Phase 6 — LoRA-Based Supervised Fine-Tuning

## Goal
Teach the Instruct model our human labeling rules without updating every pretrained parameter.

## 1. SFT in plain language
For each training example, show the model a screenshot, a classification instruction, and the correct label. It predicts output tokens; training penalizes incorrect target-token predictions. The optimizer adjusts trainable parameters to improve the next prediction.

## 2. What LoRA changes
For a selected weight matrix `W`, LoRA learns a low-rank update:

`W_effective = W_frozen + (alpha/r) * B @ A`

Here `A` and `B` are small trainable matrices; `r` is the LoRA rank. The original `W` remains frozen in standard LoRA.

LoRA reduces **trainable parameters and optimizer-state memory**, but the base model still needs to be loaded. QLoRA adds low-bit base-weight storage when supported by the hardware and software stack.

## 3. Training example
```text
USER: [webpage screenshot]
Classify this page. Return LITTLE_CONTENT or NOT_LITTLE_CONTENT.
ASSISTANT: LITTLE_CONTENT
```

Mask user/prompt tokens in the loss so training focuses on the desired assistant response, according to the chosen SFT implementation.

## 4. Initial hyperparameter search (not final settings)
| Parameter | Starting candidates |
| --- | --- |
| LoRA rank | 8, 16, 32 |
| LoRA alpha | 16, 32, 64 |
| LoRA dropout | 0.0, 0.05 |
| Learning rate | 1e-4, 2e-4 (validate stability) |
| Epochs | 1–3 with early stopping |
| Precision | BF16 when supported |
| Gradient accumulation | Adjust to effective batch size |
| Image resolution | Compare low and high token budgets |

Target modules and whether to adapt vision components must be verified for the exact checkpoint. Do not copy module names from another architecture without inspecting it.

## 5. Training loop
1. Load pinned model and processor.
2. Build training and validation datasets.
3. Apply LoRA adapters to selected modules.
4. Train on correctly formatted multimodal examples.
5. Evaluate recall and guardrails after checkpoints.
6. Save adapter, processor settings, prompt version, and metadata.
7. Choose a checkpoint using validation metrics, never the test set.

## 6. Common failures
- Overfitting to webpage templates
- Learning class frequency instead of visual evidence
- Label leakage through file names or prompts
- Too aggressive downsampling
- Unstable training due to memory limits
- Generating explanations instead of allowed labels
- Misleading metrics on artificially balanced validation data

## 7. Reproducibility
Record model revision, package versions, dataset/split hash, seed, LoRA settings, training steps, optimizer, learning-rate schedule, hardware, and artifact checksum.

## Deliverables
Versioned adapter, training configuration, reproducible script, loss curves, and validation report.

## Interview recall
**Q: If Instruct already follows directions, why SFT?**  
A: It follows the request, but it must still learn our specific labeling policy.

**Q: Does LoRA train the entire model?**  
A: Usually no; it trains small adapter matrices while the base weights stay frozen.

**Next:** Phase 7 — evaluation and thresholding.
