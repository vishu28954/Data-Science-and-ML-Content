# Part 10 — Fine-Tuning and Alignment

# Part 2 — Interview Style: Supervised Fine-Tuning Foundations

## Based on Part 2, Questions 1–11

This file follows the existing Part 1 interview-style convention: **direct, speakable answer → intuitive mechanism → equation or example → Qwen connection**, plus practical follow-up questions and answers.

- **Detailed source:** [Part 2 — SFT Foundations](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202.md)
- **Next companion:** [Part 2B — PEFT and LoRA Interview Style](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20Interview%20Style.md)

Use the answer aloud **before** reading the technical expansion. Answers describe a proposed Qwen Little Content training design; specific LoRA configurations, measured performance, data sizes and implementation details are not verified historical facts.

~~~text
Direct answer (30–60 seconds)
       ↓
Mechanism + mathematics (on follow-up)
       ↓
Qwen application
       ↓
Unfamiliar practical scenario
       ↓
Interview-ready takeaway
~~~

---

# Question 1 — What is Supervised Fine-Tuning?

**Interview answer:** SFT adapts a pretrained model using examples consisting of an input and a desired response. For a generative VLM, the input includes an image and an instruction; the supervised output is a sequence of assistant tokens. We update selected trainable parameters to make correct responses more probable, rather than learning visual and language capabilities from scratch.

**Technical expansion:** With dataset $\mathcal D_{\mathrm{SFT}}=\{(x_i,y_i)\}_{i=1}^N$, the response likelihood is:

$$
P_\theta(y_i\mid x_i)=\prod_{t=1}^{T_i}P_\theta(y_{i,t}\mid x_i,y_{i,<t}^{\mathrm{true}})
$$

A common SFT objective minimizes the negative log-likelihood of supervised assistant tokens.

**Follow-up — Why not simply prompt the pretrained model?**

I would first evaluate prompt-only Qwen on representative held-out screenshots. I would fine-tune if recurring, task-specific errors justify additional complexity. Fine-tuning is a response to a demonstrated behavioral gap, not an automatic first choice.

**Qwen application:** The input is a SERP screenshot plus classification instruction; the canonical assistant label is Little Content or Not Little Content.

**Memory line:** SFT learns desired response behavior from labelled demonstrations.

---

# Question 2 — What does a single Qwen SFT example contain?

**Interview answer:** One example contains a model-compatible multimodal conversation: the system or task instruction, a user message with the screenshot, and a supervised assistant response. The screenshot and instruction provide context; the correct canonical label is the target.

~~~text
Instruction: Classify this SERP screenshot.
Image: [visual input processed by Qwen]
Assistant target: Little Content
~~~

**Technical expansion:** Denote input context by $x=(I,x_{\mathrm{text}})$ and the target token sequence by $y=(y_1,\ldots,y_T)$. The precise image-token representation and chat-template delimiters are checkpoint-specific; I would use the chosen checkpoint's tokenizer/processor rather than invent special tokens.

**Follow-up — What could go wrong before any model training?**

Wrong screenshot-label pairing, inconsistent label strings, duplicate layouts across train and validation sets, invalid image crops, or accidentally putting the ground-truth class inside the user prompt. Inspect decoded batches and sample images.

**Qwen application:** Canonical labels simplify both supervision and the future production interface.

**Memory line:** A training example is context plus an assistant target, not merely an image and a class integer.

---

# Question 3 — How can a VLM learn from an image if it predicts text tokens?

**Interview answer:** The image pathway produces visual representations that condition the language model's hidden states. Those states produce next-token logits. Assistant-token loss can therefore depend on the screenshot even though the supervised target is textual.

~~~text
Screenshot → visual features ─┐
                             ├→ multimodal hidden state
Instruction → text context ──┘            ↓
                                   next-token logits
~~~

**Technical expansion:**

$$
v=f_{\theta_v}(I),\qquad h_t=g_\theta(v,x_{\mathrm{text}},y_{<t}^{\mathrm{true}})
$$

The probability of each target token depends on $h_t$. Specific projector and vision-encoder arrangements vary by Qwen variant.

**Follow-up — Must the vision encoder be trainable for the model to adapt?**

No. Its features can still be used when frozen. A trainable connector or language adapter downstream can learn from them. If the original visual features omit essential evidence, vision-side adaptation may need to be tested.

**Practical scenario:** Qwen ignores whitespace and image layout. First inspect resolution, cropping, processor compatibility and labelled examples before assuming language LoRA must be expanded.

**Memory line:** The loss is on response tokens, but the prediction is conditioned on visual evidence.

---

# Question 4 — How are next-token logits and training labels aligned?

**Interview answer:** Each causal position predicts the *next* token. During training, the correct response is present, but causal attention stops a position from looking at future target tokens. Logits must be matched with the following token, not the current one.

~~~text
Context ending in assistant-start → predict Little
... assistant-start + Little     → predict Content
... Little + Content             → predict EOS
~~~

This is illustrative tokenization; check the real tokenizer.

**Technical expansion:**

$$
P_\theta(y_t\mid x,y_{<t}^{\mathrm{true}})
$$

Many causal-LM implementations perform the one-token logit/label shift internally. Double-shifting or incorrectly indexing the targets breaks supervision.

**Follow-up — Isn't including the whole correct response training-time leakage?**

Not if the causal mask and shifted loss are correct: each position uses only the prefix available at that position.

**Debugging scenario:** A nearly zero training loss with poor predictions calls for checking causal masking, label shifting, ignored positions, duplicated data and prompt leakage.

**Memory line:** The logit after a prefix predicts the next target token.

---

# Question 5 — What is teacher forcing during SFT?

**Interview answer:** Teacher forcing means predictions are conditioned on the *correct* prior target tokens during training rather than on what the model would generate. It enables efficient parallel computation of causal training losses. During generation, future tokens are unavailable, so the model conditions on its own previous outputs.

**Technical expansion:**

$$
P_\theta(y_t\mid x,y_{<t}^{\mathrm{true}})
\quad\text{versus}\quad
P_\theta(\hat y_t\mid x,\hat y_{<t})
$$

**Follow-up — If the model would predict Small instead of Little, what is fed into the next training position?**

The ground-truth token Little. During inference, if Small were actually produced, the next prediction would use Small.

**Practical scenario:** Why can mistakes compound in a long generated answer? At inference the model conditions on its own possibly incorrect tokens; teacher forcing does not eliminate this mismatch.

**Qwen application:** Short canonical labels and controlled candidate scoring reduce dependence on unreliable free-form label generation.

**Memory line:** Correct previous tokens in training; generated previous tokens at inference.

---

# Question 6 — What loss does generative SFT use?

**Interview answer:** Token-level cross-entropy, equivalent to negative log-likelihood for a one-hot correct token, measures how much probability the model places on the desired next token. Higher correct-token probability means smaller loss.

**Technical expansion:**

$$
p_{t,k}=\frac{e^{z_{t,k}}}{\sum_{j=1}^{V}e^{z_{t,j}}},
\qquad
\mathcal L_t=-\log p_{t,y_t}
$$

If the correct-token probability is $0.05$, loss is approximately $2.996$; at $0.9$, it is approximately $0.105$.

For three correct-token probabilities $0.8,0.4,0.9$:

$$
\mathcal L_{\mathrm{mean}}
=
-\frac{\ln0.8+\ln0.4+\ln0.9}{3}
\approx 0.415
$$

**Follow-up — Is a falling SFT loss proof of better business precision?**

No. The training objective is next-token likelihood. False-positive cost, class balance, class-scoring method and validation threshold determine business performance.

**Qwen application:** A model can fit canonical labels better yet remain poor on sparse-but-useful pages. Track held-out precision, recall and error slices.

**Memory line:** Cross-entropy penalizes low probability on the correct next token.

---

# Question 7 — Which tokens should contribute to assistant-only SFT loss?

**Interview answer:** The instruction and screenshot should condition the response, but the direct next-token training loss generally applies to the assistant target positions. We mask prompt positions from the supervised loss while keeping them in the model's input.

**Technical expansion:** Let $m_t=1$ for supervised response targets and $m_t=0$ otherwise:

$$
\mathcal L
=
-\frac{\sum_t m_t\log P_\theta(y_t\mid c_t)}{\sum_t m_t}
$$

In common implementations, the ignored-label sentinel is -100. That is a label value interpreted by the loss, not a tokenizer vocabulary item.

**Follow-up — Does masked mean invisible to attention?**

No. Loss masking, causal masking, and attention/padding masks have different roles. Prompt tokens remain available as input context.

**Debugging scenario:** If a model never learns to output class tokens, decode the non-ignored label positions and make sure the assistant response and intended end marker are actually supervised.

**Memory line:** Mask direct target loss, not conditioning context.

---

# Question 8 — If image and prompt positions are masked, can they still influence training?

**Interview answer:** Yes. The assistant-token probability is computed from hidden states that depend on the screenshot and prompt. The loss on assistant targets backpropagates through trainable components of that computation, even though context positions are not direct loss targets.

**Technical expansion:** A conceptual gradient path through a trainable visual encoder is:

$$
\frac{\partial\mathcal L}{\partial\theta_v}
=
\sum_t
\frac{\partial\mathcal L}{\partial h_t}
\frac{\partial h_t}{\partial v}
\frac{\partial v}{\partial\theta_v}
$$

Here $v=f_{\theta_v}(I)$ and $h_t$ is the visual-conditioned hidden state. If the vision encoder is frozen, it still produces features, but does not receive optimizer updates.

**Follow-up — If the connector alone is trainable, can its parameters change?**

Yes, as long as connector output influences supervised assistant logits and is in the differentiable computation graph.

**Qwen application:** Little Content labels supervise how trainable modules use screenshot features, not whether the model can reproduce image tokens.

**Memory line:** Supervised outputs determine loss; the graph determines gradient paths.

---

# Question 9 — What if Little Content is multiple tokenizer tokens?

**Interview answer:** Class names that appear to be two or three words are not necessarily the same number of model tokens. Generative SFT supervises *each* tokenizer token in the canonical response, including any selected ending token.

**Technical expansion:**

$$
\log P_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log P_\theta(y_t\mid x,y_{<t})
$$

Two candidate labels with different token lengths can have different raw sequence-score behavior.

**Follow-up — Does length normalization automatically fix classification bias?**

No. Mean log-likelihood per token is one possible comparison, but changes the scoring rule. Choose a consistent scheme and validate it empirically.

**Practical scenario:** The model outputs LC, Little Content and It seems empty. I would standardize canonical training responses and use constrained class scoring or strict output parsing for production.

**Memory line:** Class labels are token sequences; their exact tokenizer representation matters.

---

# Question 10 — Generative SFT or a classification head?

**Interview answer:** Both are viable. Generative SFT uses the model's native assistant-token interface and scores canonical label responses. A classification head maps a hidden representation into class logits and may simplify inference and calibration. The better choice depends on validation quality, serving cost and training complexity.

**Technical expansion:** For a two-class head:

$$
z=W_ch+b_c,\qquad W_c\in\mathbb R^{2\times d},\quad z\in\mathbb R^2
$$

$$
P(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{2}e^{z_j}}
$$

**Follow-up — Why prefer generative classification for the proposed Qwen path?**

It fits an instruction-following VLM without designing a separate pooling/classifier pathway. But the classification head remains a useful baseline to compare.

**Practical scenario:** If unconstrained generation returns paragraphs, do not rely on arbitrary textual variations. Compare fixed canonical-label likelihoods or test a dedicated head.

**Memory line:** Generative label prediction and discriminative heads solve the same business task via different interfaces.

---

# Question 11 — How can a generative VLM produce a stable classification score?

**Interview answer:** Score only the allowed canonical label sequences with conditional log-likelihood, compare the scores, and select a threshold using held-out data. Do not assume an open-ended generated paragraph is already a reliable class score.

**Technical expansion:** Let $y^{(+)}$ and $y^{(-)}$ be canonical positive and negative token sequences:

$$
S_+(x)=\sum_{t=1}^{T_+}\log P_\theta(y_t^{(+)}\mid x,y_{<t}^{(+)})
$$

$$
S_-(x)=\sum_{t=1}^{T_-}\log P_\theta(y_t^{(-)}\mid x,y_{<t}^{(-)})
$$

$$
s(x)=S_+(x)-S_-(x),\qquad
\hat y=\mathbb 1[s(x)\ge\tau]
$$

The threshold $\tau$ is selected on independent validation data, not on the final test set.

**Follow-up — Is a softmax across these two sequence scores calibrated?**

Not automatically. It gives scores normalized over the candidate set, but may be miscalibrated relative to true class frequencies and error costs.

**Practical scenario:** False positives are expensive because useful pages could be removed. Tune an operating point for the required precision or false-positive-rate target, and check recall and unseen-host behavior.

**Memory line:** Compare canonical response likelihoods, then validate thresholds and calibration.

---

# Integrated Case 1 — Design Qwen SFT End to End

**Interviewer:** We have 100,000 labelled SERP screenshots. Walk through the complete supervised training process.

**Strong interview answer:** I would first audit label quality, near-duplicate screenshots, class balance and host-level leakage, and establish prompt-only and simpler-model baselines. I would build model-compatible screenshot/instruction/assistant-label examples, apply the correct Qwen multimodal processor and chat template, and supervise the canonical assistant labels using teacher-forced causal next-token cross-entropy with prompt positions excluded from direct loss. I would then backpropagate to selected trainable weights or adapters, validate on unseen hosts/layouts, compare controlled class-likelihood scores, and choose a threshold based on precision/recall and false-positive cost.

**Follow-up:** Would you report token loss as the sole evaluation measure?

**Answer:** No. Report task precision, recall, confusion matrix, robustness across layouts/hosts and threshold-dependent business trade-offs. SFT loss tests optimization progress, not deployment readiness.

---

# Integrated Case 2 — Debugging Without Guessing

**Interviewer:** Training loss drops very quickly, but new screenshots are misclassified. What next?

**Strong interview answer:** Check the data and supervision pipeline before changing capacity: image/label pairing, exact chat template, image preprocessing, accidental label leakage, duplicate or host-overlapping splits, causal shift, ignore-index positions, class tokenization and EOS. Then inspect train-versus-validation curves, hard negatives, class imbalance and scoring thresholds. Only after verifying these would I consider tuning adapter placement, learning rate or trainable modules.

---

# Part 2 — Rapid Interview Memory Map

~~~text
Labelled screenshots + instruction
        ↓
Multimodal context + canonical assistant target
        ↓
Teacher forcing + causal shift
        ↓
Token logits → softmax → cross-entropy
        ↓
Assistant-only loss mask
        ↓
Gradients for selected trainable parameters
        ↓
Canonical response likelihood comparison
        ↓
Threshold + validation + error analysis
~~~

# What We Should Not Claim Yet

This file does not establish an exact Qwen variant, training data count, tokenizer split, data partition, LoRA configuration, optimizer, measured improvements or production serving design. Those remain open or separately evidenced.

# One-Line Summary

> **SFT teaches a pretrained VLM to make a canonical assistant response more probable given its screenshot and instruction, using teacher-forced token loss while preserving proper context and label masking.**
