# Phase 5 — Understanding Qwen2.5-VL Architecture

## Goal
Explain what happens when we send a webpage screenshot and a text instruction to the model.

## 1. High-level path
```text
Screenshot → image processor → vision encoder → visual representations
                                                     ↓
Instruction → tokenizer → text tokens → multimodal language model
                                                     ↓
                                      generated classification label
```

The actual model has specific visual-token handling and multimodal position encoding; this diagram is conceptual, not a layer-by-layer implementation.

## 2. Image processing
The image processor prepares the screenshot for the vision backbone. Larger images generally require more visual tokens and computation. A higher image budget can help with tiny text but slows inference.

## 3. Vision encoder
A vision transformer turns image patches into feature representations. Those features encode visual structures such as text regions, whitespace, buttons, and page layout.

## 4. Vision-language connection
Visual representations are made compatible with the language model's hidden representations and included in the multimodal sequence. The language model can then use both screenshot information and the classification instruction.

## 5. Decoder and generated label
Qwen2.5-VL-Instruct is a generative model: it predicts output tokens. We ask it to generate exactly one of our two labels. This is **generative classification**, not a dedicated sigmoid classification head.

For threshold selection, investigate normalized likelihoods of the two allowed labels and calibrate them. Multi-token labels require careful sequence scoring; a naive comparison of a single token may be invalid.

## 6. Why Instruct?
A Base model learns general patterns; an Instruct checkpoint has also been optimized to follow requests. We choose Instruct because our task is framed as an instruction: “Look at the image and return a label.” Fine-tuning still teaches the specific meaning of Little Content.

## 7. Important design choices
- Image token budget versus small-text readability
- 3B versus 7B quality/cost
- One screenshot versus multiple views
- Generative label versus dedicated classifier head
- Short controlled output to reduce decoding cost

## 8. Suggested hands-on exercise
Load the exact pinned Qwen2.5-VL-3B-Instruct checkpoint and processor, run a single screenshot, inspect the formatted prompt, generated tokens, latency, and peak memory. Pin versions and follow the model's official image/chat processing instructions.

## Interview recall
**Q: Why does screenshot resolution affect cost?**  
A: More visual detail usually means more visual tokens and computation.

**Q: Why can a VLM detect content beyond word count?**  
A: It can combine page layout, images, and visible text.

**Next:** Phase 6 — LoRA-based supervised fine-tuning.
