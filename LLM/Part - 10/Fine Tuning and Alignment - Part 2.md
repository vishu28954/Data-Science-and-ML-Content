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
