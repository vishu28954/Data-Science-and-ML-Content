# Part 10 — Fine-Tuning and Alignment

# Part 2 — Interview Style: Supervised Fine-Tuning Foundations

## Based on Part 2, Questions 1–11

**How to use this file:** First say the simple interview answer in your own words. Next explain how it works and give the Qwen example. Only open the mathematical details when the interviewer asks for depth. Technical accuracy is preserved; advanced terms appear after a plain-language explanation.

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

### Simple interview answer — say this aloud

Supervised Fine-Tuning, or SFT, means taking a model that already knows a lot and teaching it one specific task using examples with correct answers. For our Qwen project, we show it a search-page screenshot and the correct label, such as Little Content. The model learns to make that answer more likely the next time it sees a similar page. We are improving an existing model rather than training a new one from the beginning.

### How it works — step by step

Start with Qwen's pretrained weights. Prepare labelled screenshots. Show each screenshot with a clear instruction and the correct assistant response. Calculate how unlikely Qwen considered the correct response, then use that error to update whichever parameters we have chosen to train.

### Qwen Little Content example

A screenshot of a mostly empty page is paired with the target response Little Content. Another screenshot with substantial useful information is paired with Not Little Content.

### How to understand the mathematics

The probability formula says that an entire answer is built one token at a time. At each step, Qwen uses the screenshot, instruction and correct earlier answer tokens to predict the next one.

### If the interviewer asks for technical details With dataset $\mathcal D_{\mathrm{SFT}}=\{(x_i,y_i)\}_{i=1}^N$, the response likelihood is:

$$
P_\theta(y_i\mid x_i)=\prod_{t=1}^{T_i}P_\theta(y_{i,t}\mid x_i,y_{i,<t}^{\mathrm{true}})
$$

A common SFT objective minimizes the negative log-likelihood of supervised assistant tokens.

**Follow-up — Why not simply prompt the pretrained model?**

I would first evaluate prompt-only Qwen on representative held-out screenshots. I would fine-tune if recurring, task-specific errors justify additional complexity. Fine-tuning is a response to a demonstrated behavioral gap, not an automatic first choice.

**Qwen application:** The input is a SERP screenshot plus classification instruction; the canonical assistant label is Little Content or Not Little Content.

**Memory line:** SFT learns desired response behavior from labelled demonstrations.

---

---

# Question 2 — What does a single Qwen SFT example contain?

### Simple interview answer — say this aloud

One training example has two main parts: what we give Qwen and what we want it to answer. We give it a screenshot and an instruction asking it to classify the page. We also provide the correct assistant answer during training. That correct answer becomes the training target. We must use the exact input format supported by the chosen Qwen model.

### How it works — step by step

First process the screenshot using Qwen's image processor. Next put the instruction into the appropriate chat message. Then add the canonical assistant response as the target. Check that the screenshot, prompt and label all belong to the same example.

### Qwen Little Content example

Input: screenshot plus Classify this page. Correct response: Little Content. The model should learn the connection between what it sees in the screenshot and that label.

### How to understand the mathematics

In the notation, I means the image, x_text means the written instruction, and y means the assistant answer. The model predicts y based on both parts of the input.

### If the interviewer asks for technical details Denote input context by $x=(I,x_{\mathrm{text}})$ and the target token sequence by $y=(y_1,\ldots,y_T)$. The precise image-token representation and chat-template delimiters are checkpoint-specific; I would use the chosen checkpoint's tokenizer/processor rather than invent special tokens.

**Follow-up — What could go wrong before any model training?**

Wrong screenshot-label pairing, inconsistent label strings, duplicate layouts across train and validation sets, invalid image crops, or accidentally putting the ground-truth class inside the user prompt. Inspect decoded batches and sample images.

**Qwen application:** Canonical labels simplify both supervision and the future production interface.

**Memory line:** A training example is context plus an assistant target, not merely an image and a class integer.

---

---

# Question 3 — How can a VLM learn from an image if it predicts text tokens?

### Simple interview answer — say this aloud

Even if Qwen answers using words, its answer can depend on the image. The model first converts the screenshot into numerical visual features. It combines those features with the instruction, and then uses the combined information to predict the answer tokens. So the label is text, but the information used to choose it comes from the screenshot.

### How it works — step by step

Think of three stages. First, the visual part reads the screenshot. Second, the model brings the visual and written information together. Third, the language part predicts which answer is most likely. A training error on the answer can help train the components that connect these stages.

### Qwen Little Content example

A page might contain very little readable text but a large navigation menu. Qwen must consider what the layout actually means instead of deciding only from the raw word count.

### How to understand the mathematics

v represents image features, h_t represents the model's internal information at a prediction position, and the final probability comes from the next-token scores. The specific internal wiring depends on the Qwen version.

### If the interviewer asks for technical details

$$
v=f_{\theta_v}(I),\qquad h_t=g_\theta(v,x_{\mathrm{text}},y_{<t}^{\mathrm{true}})
$$

The probability of each target token depends on $h_t$. Specific projector and vision-encoder arrangements vary by Qwen variant.

**Follow-up — Must the vision encoder be trainable for the model to adapt?**

No. Its features can still be used when frozen. A trainable connector or language adapter downstream can learn from them. If the original visual features omit essential evidence, vision-side adaptation may need to be tested.

**Practical scenario:** Qwen ignores whitespace and image layout. First inspect resolution, cropping, processor compatibility and labelled examples before assuming language LoRA must be expanded.

**Memory line:** The loss is on response tokens, but the prediction is conditioned on visual evidence.

---

---

# Question 4 — How are next-token logits and training labels aligned?

### Simple interview answer — say this aloud

Qwen predicts one next token at a time. During training, it already has the correct answer sequence, so we check whether each position predicts the token that comes immediately after it. For example, after the start of the assistant response, it should predict the first label token. After that first token, it should predict the next one. We must align each prediction with the following token, not the token already at that position.

### How it works — step by step

Imagine reading a sentence and guessing its next word. To judge your guess, we compare it with the actual next word. The training software does this for every supervised position. A causal mask also prevents a position from peeking at later words in the answer.

### Qwen Little Content example

If the tokenizer happened to split a label into Little and Content, the first prediction is for Little, the next is for Content, and the last may be for an end marker. This is an illustration, not a guaranteed tokenizer split.

### How to understand the mathematics

y_t is the token to predict, and y_<t are earlier correct tokens. This is why training labels and next-token logits need a one-position shift.

### If the interviewer asks for technical details

$$
P_\theta(y_t\mid x,y_{<t}^{\mathrm{true}})
$$

Many causal-LM implementations perform the one-token logit/label shift internally. Double-shifting or incorrectly indexing the targets breaks supervision.

**Follow-up — Isn't including the whole correct response training-time leakage?**

Not if the causal mask and shifted loss are correct: each position uses only the prefix available at that position.

**Debugging scenario:** A nearly zero training loss with poor predictions calls for checking causal masking, label shifting, ignored positions, duplicated data and prompt leakage.

**Memory line:** The logit after a prefix predicts the next target token.

---

---

# Question 5 — What is teacher forcing during SFT?

### Simple interview answer — say this aloud

Teacher forcing means that while training, Qwen uses the correct previous answer tokens to predict the next token. It does not use its own mistakes as the previous context during that training pass. At inference time, the correct answer is not available, so it must use the tokens it generated itself. This difference is why training can process many answer positions together while generation normally proceeds token by token.

### How it works — step by step

Suppose the desired answer starts with Little followed by Content. Even if Qwen would have guessed Small, training still supplies Little when predicting the next token. During actual generation, if it says Small, the next prediction depends on Small.

### Qwen Little Content example

Teacher forcing teaches Qwen the desired canonical labels using the correct response history. We can later compare allowed labels directly to reduce unreliable free-form output.

### How to understand the mathematics

The training expression conditions on true earlier target tokens. The inference expression conditions on earlier predicted tokens. The two expressions look similar but describe different sources of context.

### If the interviewer asks for technical details

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

---

# Question 6 — What loss does generative SFT use?

### Simple interview answer — say this aloud

Cross-entropy is the loss we use to tell Qwen how wrong a token prediction was. If the correct next token receives a high probability, the loss is small. If it receives a low probability, the loss is large. During supervised fine-tuning, we calculate this for the answer tokens and combine the losses so the model knows how to improve.

### How it works — step by step

Qwen first gives a score to every possible next token. A softmax turns those scores into probabilities. We take the probability assigned to the correct token and calculate its negative log. The optimizer then uses gradients from that loss to change trainable parameters.

### Qwen Little Content example

If Little is the correct token and Qwen assigns it only 5% probability, the loss is large. If it assigns 90%, the loss is much smaller. But lower loss does not automatically mean fewer costly false positives.

### How to understand the mathematics

z is the raw score or logit, V is vocabulary size, p is the probability after softmax, and negative log means that increasing the correct-token probability decreases the loss.

### If the interviewer asks for technical details

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

---

# Question 7 — Which tokens should contribute to assistant-only SFT loss?

### Simple interview answer — say this aloud

In assistant-only SFT, we usually calculate the loss only on the answer Qwen is supposed to produce. The screenshot and the user's instruction are still provided to the model, but we do not train it to recreate those input tokens. This focuses learning on the desired assistant response.

### How it works — step by step

The training sequence contains prompt tokens and response tokens. We mark the response targets as positions to learn from and mark the prompt targets as ignored for loss calculation. In many PyTorch training setups, ignored target labels use the value -100.

### Qwen Little Content example

Qwen should learn to answer Little Content after reading the screenshot and instruction. It should not spend supervised loss on predicting back the user's instruction Classify this page.

### How to understand the mathematics

m_t is a switch: 1 means this target token contributes to loss, and 0 means it does not. The denominator counts supervised positions so the loss is averaged over them.

### If the interviewer asks for technical details Let $m_t=1$ for supervised response targets and $m_t=0$ otherwise:

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

---

# Question 8 — If image and prompt positions are masked, can they still influence training?

### Simple interview answer — say this aloud

Yes. Even when the prompt and image do not have their own target loss, they still influence Qwen's answer. The model reads them to predict the class label. If that prediction is wrong, the loss can flow backward through any trainable parts that helped produce the answer. Loss masking does not make the screenshot invisible.

### How it works — step by step

The visual pathway extracts image features. Those features influence hidden states and answer probabilities. Cross-entropy compares the predicted answer with the correct one. Backpropagation then updates whichever relevant layers or adapters are trainable.

### Qwen Little Content example

If we freeze Qwen's image encoder but train a connector or LoRA adapter, the frozen encoder still supplies the screenshot features. Trainable downstream modules can learn how to use those features.

### How to understand the mathematics

The chain-rule expression follows the error backward: loss depends on hidden states, hidden states depend on visual features, and visual features may depend on trainable visual weights. Frozen weights are still used but not updated.

### If the interviewer asks for technical details A conceptual gradient path through a trainable visual encoder is:

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

---

# Question 9 — What if Little Content is multiple tokenizer tokens?

### Simple interview answer — say this aloud

A class name is not necessarily one model token. Qwen's tokenizer might split Little Content into several pieces, and the other label might have a different number of pieces. During SFT, Qwen learns to predict each piece of the correct answer in order. The exact tokenization matters when we later compare label probabilities.

### How it works — step by step

First check how the chosen tokenizer represents both labels. Use exactly the same canonical spellings during training and scoring. Remember that the probability of a whole label depends on the probabilities of every token in that label.

### Qwen Little Content example

If Qwen produces several variations such as LC or This page is empty, parsing becomes less reliable. Canonical class targets and controlled scoring keep the interface consistent.

### How to understand the mathematics

The whole-label log score is the sum of the next-token log probabilities. Longer labels contain more terms, so raw scores and length-normalized scores can behave differently. Check which method works on held-out data.

### If the interviewer asks for technical details

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

---

# Question 10 — Generative SFT or a classification head?

### Simple interview answer — say this aloud

There are two main ways to make Qwen a classifier. We can teach it to generate one of our allowed text labels, or we can add a small classification layer that directly produces scores for the two classes. Generating labels fits Qwen's existing language-model design; a classification layer may make the output easier to control. I would compare both approaches before deciding.

### How it works — step by step

In the generative approach, the next-token model learns to say Little Content or Not Little Content. In the classification-head approach, we take a model representation and pass it through a trainable layer that produces two scores. We measure both using the same held-out task examples.

### Qwen Little Content example

Our current theoretical project prefers canonical generative labels, while keeping a classification head as an experimental alternative. Neither choice is automatically superior.

### How to understand the mathematics

W_c converts the model representation h into two scores z. Softmax then turns those two scores into relative class probabilities. This is different from multiplying probabilities across an answer sequence.

### If the interviewer asks for technical details For a two-class head:

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

---

# Question 11 — How can a generative VLM produce a stable classification score?

### Simple interview answer — say this aloud

A generative model can still produce a useful classification score. Instead of asking it to write any answer, I give it the two allowed labels and measure how likely it considers each label for that screenshot. I compare those scores and choose a decision threshold using validation data. That gives me a more controlled decision than relying on whatever sentence the model generates.

### How it works — step by step

For the same screenshot, calculate the log-likelihood of Little Content and Not Little Content. Subtract one score from the other to measure which label is preferred. Choose a threshold that gives the precision and recall required by the application.

### Qwen Little Content example

For junk-page detection, false positives can be costly because useful pages might be wrongly flagged. I would check the selected threshold on unseen hosts and sparse-but-useful layouts.

### How to understand the mathematics

S_+ and S_- are scores for the two allowed label sequences. Their difference is s(x), and tau is the decision threshold. These scores are not guaranteed to be calibrated probabilities.

### If the interviewer asks for technical details Let $y^{(+)}$ and $y^{(-)}$ be canonical positive and negative token sequences:

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
