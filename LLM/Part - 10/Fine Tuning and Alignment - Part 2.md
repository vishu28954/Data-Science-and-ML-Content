# Part 10 — Fine-Tuning and Alignment

# Fine-Tuning and Alignment — Part 2

## 10.2 Supervised Fine-Tuning (SFT)

**Detailed Study Mode. Interview-style material remains separate.**

Part 1 established **why** fine-tuning is a reasonable tool for the Qwen Little Content project.

Part 2 now asks:

> **How do labelled examples actually change the model?**

This is where the project moves from a high-level idea into an actual training formulation.

Our fixed learning pattern remains:

~~~text
Story Bridge
    ↓
Natural Question
    ↓
Detailed Concept
    ↓
Mathematics / Example
    ↓
Qwen Project Application
    ↓
Design Decision
    ↓
Next Story Bridge
~~~

---

# Qwen Project — End-to-End Roadmap Snapshot

This roadmap stays visible so that we always know what is already designed and what is intentionally deferred.

| Project layer | Current status |
|---|---|
| Business problem | Defined in Part 1 |
| Why Qwen / why a VLM | Defined in Part 1 |
| Why fine-tuning | Defined in Part 1 |
| Prompt-only baseline | Defined in Part 1 |
| **SFT training formulation** | **Designed in this Part** |
| **Loss construction** | **Designed in this Part** |
| **Full FT vs PEFT / LoRA** | **Designed in this Part** |
| Multimodal chat template | Part 3 |
| Image preprocessing details | Part 3 |
| Preference tuning / RLHF / DPO decision | Part 4 |
| Safety / behavioral constraints | Part 5 |
| Dataset sampling + optimizer + LR + scheduler | Part 6 |
| Overfitting + leakage + generalization | Part 7 |
| Evaluation + thresholding + calibration | Part 8 |
| Deployment + 50K/day serving | Part 8 |
| Failure handling + monitoring + drift | Part 8 |

---

# Starting Project State From Part 1

~~~text
Task
Little Content detection

Input
Bing SERP screenshot
+
task instruction

Base model
Pretrained Qwen VLM

Desired output
Little Content
or
Not Little Content

Why fine-tune?
Prompting is the baseline,
but persistent task-specific errors
justify adaptation

Generalization goal
Unseen hosts / layouts / templates
~~~

The next question is:

> **What exactly happens during supervised training?**

---

### Story Bridge 1 — We Have Labels, but a Label Alone Does Not Train a Language Model

Part 1 ended with labelled screenshot pairs:

$$
(x_i,y_i)
$$

But a generative VLM does not consume the abstract phrase "binary classification dataset."

It consumes a multimodal context and learns to predict target tokens.

## Question 1 — What is Supervised Fine-Tuning?

Supervised Fine-Tuning, or **SFT**, adapts a pretrained model using examples containing:

1. An input or context.
2. A desired target response.

A generic SFT dataset is:

$$
\mathcal{D}_{\mathrm{SFT}}
=
\{
(x_i,y_i)
\}_{i=1}^{N}
$$

where:

- $x_i$ is the input.
- $y_i$ is the desired output.

For a generative model, we want to maximize:

$$
P_\theta(y_i\mid x_i)
$$

Equivalently, we minimize negative log-likelihood:

$$
\mathcal{L}_{\mathrm{SFT}}
=
-
\sum_{i=1}^{N}
\log
P_\theta
\left(
y_i\mid x_i
\right)
$$

If:

$$
y_i
=
\left(
y_{i,1},
y_{i,2},
\ldots,
y_{i,T_i}
\right)
$$

then:

$$
P_\theta(y_i\mid x_i)
=
\prod_{t=1}^{T_i}
P_\theta
\left(
y_{i,t}
\mid
x_i,
y_{i,<t}
\right)
$$

and:

$$
\mathcal{L}_{\mathrm{SFT}}
=
-
\sum_{i=1}^{N}
\sum_{t=1}^{T_i}
\log
P_\theta
\left(
y_{i,t}
\mid
x_i,
y_{i,<t}
\right)
$$

### Intuition

SFT is learning from demonstrations:

~~~text
Here is the input.
Here is the response I want.
        ↓
Repeat across many examples.
        ↓
Adjust parameters so desired responses
become more probable.
~~~

### Qwen Project Application

For one example:

~~~text
Input:
SERP screenshot
+
classification instruction

Desired response:
Little Content
~~~

For another:

~~~text
Input:
SERP screenshot
+
classification instruction

Desired response:
Not Little Content
~~~

### Design Decision

The Qwen project will use **generative SFT** as the primary training formulation.

The exact chat serialization is deferred to Part 3.

---

### Story Bridge 2 — SFT Needs an Actual Training Example, Not Just an Abstract Pair

For a multimodal chat model, one example contains several semantic pieces:

- Image.
- Instruction.
- Assistant response.
- Potential system context.

We need to separate what conditions the model from what we want it to predict.

## Question 2 — What does one Qwen SFT example contain?

Conceptually:

~~~text
SYSTEM
You are a page-quality classifier.

USER
<image>
Classify this SERP screenshot as
Little Content or Not Little Content.

ASSISTANT
Little Content
~~~

The conditioning context is:

$$
x
=
\{
\text{image},
\text{system instruction},
\text{user instruction}
\}
$$

The supervised target is:

$$
y
=
\text{"Little Content"}
$$

So the task is:

$$
P_\theta
\left(
\text{"Little Content"}
\mid
\text{image},
\text{instruction}
\right)
$$

### Qwen Project Application

A training row should eventually contain at least:

~~~text
Image reference
Task instruction
Ground-truth label
~~~

Metadata such as host, URL, annotation source, and template identity may be useful for splitting and analysis, but should not automatically become model input.

### Design Decision

Each project example will conceptually contain:

$$
(\text{SERP image},\text{task instruction},\text{canonical target label})
$$

The exact Qwen chat format is deferred to Part 3.

---

### Story Bridge 3 — The Model Is Multimodal, So the Screenshot Must Enter the Language Model Somehow

Text becomes tokens.

A screenshot is not naturally a text-token sequence.

So we need a conceptual picture of how visual evidence reaches the generative model.

## Question 3 — How does a VLM conceptually combine image and text during SFT?

A generic VLM pipeline is:

~~~text
Screenshot
    ↓
Vision encoder
    ↓
Visual features
    ↓
Projector / connector
    ↓
Language-model embedding space

Text instruction
    ↓
Tokenizer
    ↓
Text embeddings

Visual + text representations
    ↓
Language model
    ↓
Assistant-token predictions
~~~

Let the vision encoder produce:

$$
V
=
\left[
v_1,
v_2,
\ldots,
v_m
\right]
$$

where:

$$
v_j
\in
\mathbb{R}^{d_v}
$$

A projector can map a visual feature to the language-model hidden dimension:

$$
\tilde{v}_j
=
W_pv_j
$$

with:

$$
W_p
\in
\mathbb{R}^{d_{\mathrm{model}}\times d_v}
$$

Different VLMs vary in how they implement image-token compression, connectors, image placeholders, cross-attention, and positional representations.

### Qwen Project Application

The screenshot provides evidence.

The instruction defines the task.

The assistant response supplies the supervised target.

### Design Decision

Until we select the exact Qwen variant, we will reason using:

$$
\text{Vision encoder}
\rightarrow
\text{multimodal connector}
\rightarrow
\text{language model}
$$

as the conceptual architecture.

---

### Story Bridge 4 — Once the Context Is Inside the Model, We Must Align Inputs With Next-Token Targets

A causal language model predicts the next token.

That means the training target is shifted relative to the context.

## Question 4 — How are next-token predictions aligned during SFT?

Suppose the assistant target is:

$$
y_1,y_2,\ldots,y_T
$$

The model predicts:

$$
P(y_1\mid x)
$$

then:

$$
P(y_2\mid x,y_1)
$$

then:

$$
P(y_3\mid x,y_1,y_2)
$$

and so on.

Conceptually:

~~~text
Context | y1 | y2 | y3
          ↓    ↓    ↓
Predict   y1   y2   y3
~~~

The logit vector at a prediction position is:

$$
z_t
\in
\mathbb{R}^{V}
$$

where $V$ is vocabulary size.

Many causal-LM training APIs perform the one-position logits/labels shift internally.

### Qwen Project Application

If a label tokenizes into several tokens, each label token becomes a next-token prediction target.

### Design Decision

We retain the standard autoregressive causal-LM objective rather than creating a custom sequence-training rule.

---

### Story Bridge 5 — During Training the Correct Assistant Tokens Already Exist

Inference must generate future tokens sequentially.

SFT training is different because the complete correct response is already present in the dataset.

## Question 5 — What is teacher forcing during SFT?

Teacher forcing predicts each target token while conditioning on the **ground-truth previous target tokens**:

$$
P_\theta
\left(
y_t
\mid
x,
y_{<t}^{\mathrm{ground\ truth}}
\right)
$$

The entire target sequence is known during training, so token-position computations can be parallelized under a causal mask.

### Training

~~~text
Ground truth y1, y2, y3, y4 is already known.

The model predicts the next-token distribution
at all supervised positions under causal masking.
~~~

### Inference

~~~text
y1 does not exist yet.
Generate y1.
        ↓
Then generate y2 conditioned on generated y1.
~~~

### Connection to Part 8

This is exactly why training can exploit positional parallelism while autoregressive inference cannot generate unknown future output positions simultaneously.

### Qwen Project Application

Our class responses are short, but they still use the same teacher-forced causal objective.

### Design Decision

The project uses standard teacher-forced SFT.


---

### Story Bridge 6 — Teacher Forcing Gives Predictions, but We Still Need a Numeric Error Signal

The model produces a vocabulary distribution at every supervised position.

Training still needs a scalar objective that says how wrong those predictions were.

That brings us to token-level cross-entropy.

## Question 6 — What loss does generative SFT use?

Let the vocabulary logits at supervised position $t$ be:

$$
z_t
=
\left[
z_{t,1},
z_{t,2},
\ldots,
z_{t,V}
\right]
$$

Softmax converts those logits into a probability distribution:

$$
P_\theta(k\mid c_t)
=
\frac{
e^{z_{t,k}}
}{
\sum_{j=1}^{V}e^{z_{t,j}}
}
$$

where $c_t$ is the causal context.

If the correct target token is $y_t$, the token loss is:

$$
\mathcal{L}_t
=
-
\log
P_\theta
\left(
y_t\mid c_t
\right)
$$

For $T$ supervised response tokens:

$$
\mathcal{L}_{\mathrm{example}}
=
-
\frac{1}{T}
\sum_{t=1}^{T}
\log
P_\theta
\left(
y_t\mid c_t
\right)
$$

### Numerical intuition

If the correct token gets probability:

$$
0.8
$$

then:

$$
-\log(0.8)
\approx
0.223
$$

If it gets only:

$$
0.1
$$

then:

$$
-\log(0.1)
\approx
2.303
$$

So confident correct predictions produce small loss, while low probability on the correct token produces much larger loss.

### Qwen Project Application

The loss pushes Qwen to assign more probability to the correct Little Content class response given the screenshot and instruction.

### Design Decision

The primary SFT objective is **token-level negative log-likelihood / cross-entropy over the assistant response**.

---

### Story Bridge 7 — The Sequence Contains Context Tokens We Do Not Want to Train the Model to Reproduce

The SFT sequence includes:

- System text.
- User instruction.
- Image-related context.
- Assistant answer.

If we calculate loss everywhere, the model spends direct supervision on reproducing text that is only supposed to condition the task.

So we need a loss mask.

## Question 7 — Which tokens should contribute to the SFT loss?

A common instruction-SFT strategy is to compute loss only on assistant-response tokens.

Define:

$$
m_t
\in
\{0,1\}
$$

with:

$$
m_t
=
\begin{cases}
1 & \text{assistant target token}\\
0 & \text{context token}
\end{cases}
$$

Then:

$$
\mathcal{L}
=
-
\frac{
\sum_t
m_t
\log
P_\theta
\left(
y_t\mid c_t
\right)
}{
\sum_t m_t
}
$$

Conceptually:

~~~text
SYSTEM
attend to it
do not directly supervise it

USER
attend to it
do not directly supervise it

IMAGE
condition on it
not a text target

ASSISTANT TARGET
attend to it
supervise it
~~~

### Common implementation convention

Many causal-LM pipelines use an ignore value such as:

~~~text
-100
~~~

for positions that should not contribute to cross-entropy.

Conceptually:

~~~text
input_ids:
[system] [user] [image marker] [assistant marker] [Little] [Content]

labels:
[-100]  [-100] [-100]         [-100]             [Little] [Content]
~~~

The exact boundaries depend on the chosen Qwen chat template.

### Qwen Project Application

The screenshot and task instruction should condition the answer.

The assistant label supplies the direct supervision.

### Design Decision

Our default design is **assistant-only loss masking**.

We will not train the model to reconstruct the user's instruction.

---

### Story Bridge 8 — A Masked Context Can Still Affect the Gradient

Loss masking can sound as though the prompt and image are ignored during training.

That is not what happens.

The assistant prediction depends on those inputs.

We need to separate where loss is measured from where gradients can flow.

## Question 8 — If prompt and image positions are masked from the loss, can the model still learn from them?

Yes.

Suppose the supervised loss is:

$$
\mathcal{L}
=
-
\log
P_\theta
\left(
y
\mid
I,x
\right)
$$

where:

- $I$ is visual information.
- $x$ is textual context.

Even though the explicit target loss is evaluated on the assistant output, the prediction depends on the image and prompt.

For trainable visual-path parameters $\theta_v$:

$$
\frac{
\partial\mathcal{L}
}{
\partial\theta_v
}
=
\frac{
\partial\mathcal{L}
}{
\partial h
}
\frac{
\partial h
}{
\partial\theta_v
}
$$

if the visual path contributes to hidden representation $h$.

### Crucial distinction

~~~text
LOSS MASKING
Where is prediction error measured?

FREEZING
Which parameters are allowed to update?
~~~

These are different decisions.

### Example

The vision encoder may be frozen.

Then:

- Its output still conditions the assistant prediction.
- Gradients may mathematically reach its output.
- But its parameters are not updated.

A trainable multimodal connector can still receive gradient from the assistant loss.

### Qwen Project Application

The image can teach the downstream classifier behavior even though we do not define a separate "image-token loss."

### Design Decision

We will keep **loss masking** and **parameter freezing** conceptually separate throughout the project.

---

### Story Bridge 9 — The Human Class Name May Be Several Model Tokens

"Little Content" looks like one class label to us.

The tokenizer may represent it using multiple tokens.

That affects both training loss and later class scoring.

## Question 9 — What happens if a class label contains multiple tokens?

Suppose the positive label tokenizes as:

$$
y_1,y_2
$$

Then:

$$
P_\theta(y_1,y_2\mid x)
=
P_\theta(y_1\mid x)
P_\theta(y_2\mid x,y_1)
$$

and:

$$
\log
P_\theta(y_1,y_2\mid x)
=
\log
P_\theta(y_1\mid x)
+
\log
P_\theta(y_2\mid x,y_1)
$$

SFT therefore supervises every token in the canonical target string.

### Why this matters for classification

Suppose one class label takes one token and another takes three.

Raw sequence log-probability contains more additive terms for the longer label.

Potential strategies include:

- Use short canonical labels.
- Compare length-normalized sequence scores.
- Add dedicated special class tokens if justified.
- Use a classification head instead of text generation.

### Qwen Project Application

We should not allow arbitrary equivalent outputs such as:

~~~text
Little Content

This is Little Content

The page has little content

LC
~~~

That creates unnecessary output variation.

### Design Decision

The project will use a **small canonical target vocabulary**.

The exact target strings and tokenizer behavior will be finalized in Part 3.

---

### Story Bridge 10 — A Binary Business Task Does Not Necessarily Require Free-Form Generation

We are using a generative VLM, but the business output is binary.

That creates a real architectural choice:

> Generate a label as text, or attach a dedicated classifier?

## Question 10 — Generative SFT or a classification head: which should we use?

### Option A — Generative label prediction

The model produces a canonical class response.

The training objective remains ordinary causal-LM SFT.

Advantages:

- Preserves the native VLM/chat interface.
- Reuses standard SFT tooling.
- Easy to extend to richer responses later.
- Fits naturally with instruction tuning.

Limitations:

- Label tokenization must be handled carefully.
- Free generation can produce formatting variants.
- Generative probabilities are not automatically calibrated binary probabilities.

### Option B — Add a classification head

Suppose hidden representation $h$ feeds a two-class head:

$$
z
=
W_ch+b
$$

where:

$$
z
\in
\mathbb{R}^{2}
$$

Then:

$$
P(y=k\mid x)
=
\frac{
e^{z_k}
}{
\sum_{j=1}^{2}e^{z_j}
}
$$

Advantages:

- Direct class logits.
- Efficient for pure classification.
- Easier to threshold and calibrate.

Limitations:

- Modifies the native model interface.
- Requires choosing the representation used for classification.
- Gives up some flexibility of a generative chat formulation.

### Qwen Project Application

Because the project is framed as fine-tuning a Qwen VLM and later parts explicitly study instruction/chat tuning, our main theoretical design will use **generative SFT**.

A classifier head remains a valid experimental baseline.

### Design Decision

Primary:

$$
\text{Generative SFT with canonical class responses}
$$

Secondary ablation:

$$
\text{Dedicated classification head}
$$

if supported cleanly by the implementation.
