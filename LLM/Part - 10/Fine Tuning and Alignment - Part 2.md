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
\mathcal{D}_{\mathrm{SFT}} =
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
\mathcal{L}_{\mathrm{SFT}} = -
\sum_{i=1}^{N}
\log
P_\theta
\left(
y_i\mid x_i
\right)
$$

If:

$$
y_i =
\left(
y_{i,1},
y_{i,2},
\ldots,
y_{i,T_i}
\right)
$$

then:

$$
P_\theta(y_i \mid x_i) = \prod_{t=1}^{T_i} P_\theta \left( y_{i,t} \mid x_i, y_{i,\lt t} \right)
$$



and:

$$
\mathcal{L}_{\mathrm{SFT}} = - \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta \left( y_{i,t} \mid x_i, y_{i,<t} \right)
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


---

### Story Bridge 11 — A Generative Model Still Needs a Stable Production Score

Free-form generation is convenient for training, but production classification should not depend on wording variations.

A more controlled approach is to compare the likelihood of the allowed class responses.

## Question 11 — How can a generative model produce a classification score?

Let the positive canonical label be:

$$
y^{(+)}
$$

and the negative canonical label be:

$$
y^{(-)}
$$

We can compute:

$$
S_+(x)
=
\log
P_\theta
\left(
y^{(+)}
\mid x
\right)
$$

and:

$$
S_-(x)
=
\log
P_\theta
\left(
y^{(-)}
\mid x
\right)
$$

A relative class score is:

$$
s(x)
=
S_+(x)-S_-(x)
$$

Then a thresholded decision is:

$$
\hat{y}
=
\mathbb{1}
\left[
s(x)\ge\tau
\right]
$$

where $\tau$ is chosen using validation data.

If the two canonical labels have different token lengths, one possible comparison is the length-normalized sequence score:

$$
\bar{S}(y\mid x)
=
\frac{1}{|y|}
\log
P_\theta
\left(
y\mid x
\right)
$$

### Why this helps

Instead of allowing unconstrained generation such as:

~~~text
This seems like a Little Content page.
~~~

the production layer can compare only the two known candidate responses.

### Qwen Project Application

The model can remain a generative VLM while the business decision layer behaves like a controlled classifier.

### Design Decision

The theoretical production design will prefer **canonical label likelihood comparison** over unrestricted text generation.

The final threshold and calibration design are deferred to Part 8.

---

### Story Bridge 12 — SFT Defines the Loss, but It Does Not Decide Which Parameters Are Allowed to Move

We now know:

- What the model predicts.
- Which positions contribute to loss.
- How the classification score can be derived.

The next major decision is whether we update the entire model or only a small trainable subset.

## Question 12 — What is full fine-tuning?

In full fine-tuning, most or all pretrained model parameters remain trainable.

Let:

$$
\theta_0
$$

be the pretrained parameter state.

We optimize the model directly:

$$
\theta_0
\rightarrow
\theta^*
$$

through updates such as:

$$
\theta_{k+1}
=
\theta_k
-
\eta
\nabla_\theta
\mathcal{L}
$$

### Advantages

Full fine-tuning gives the model maximum adaptation capacity.

Every trainable representation can shift toward the downstream task.

This may help when:

- Domain shift is large.
- The downstream dataset is large and high quality.
- Compute is sufficient.
- The desired behavior differs substantially from the base model.

### Costs

It requires:

- Gradients for many parameters.
- Optimizer states for many parameters.
- More checkpoint storage.
- More training memory.
- More expensive distributed training.
- Greater risk of over-specializing a large model on a narrow dataset.

### Qwen Project Application

A full multimodal fine-tune could update:

~~~text
Vision encoder
+
multimodal connector
+
language backbone
~~~

That is more adaptation capacity than we may need for Little Content classification.

### Design Decision

Full fine-tuning is **not** our default first experiment.

It remains an escalation option if PEFT clearly underfits and sufficient data/compute are available.

---

### Story Bridge 13 — A Model That Fits for Inference May Still Be Too Expensive to Full-Fine-Tune

Inference mainly needs model weights, activations, and runtime state.

Training needs additional gradient and optimizer state.

So full fine-tuning can require far more memory than inference.

## Question 13 — Why is full fine-tuning so memory-intensive?

A simplified training-memory decomposition is:

$$
M_{\mathrm{train}}
\approx
M_{\mathrm{weights}}
+
M_{\mathrm{gradients}}
+
M_{\mathrm{optimizer}}
+
M_{\mathrm{activations}}
+
M_{\mathrm{runtime}}
$$

### Weight memory

For $P$ parameters stored using $b_w$ bytes each:

$$
M_{\mathrm{weights}}
\approx
Pb_w
$$

### Gradient memory

If every parameter is trainable:

$$
M_{\mathrm{gradients}}
\propto
P
$$

### Adam-like optimizer state

Adam maintains first and second moments:

$$
m_t
$$

and:

$$
v_t
$$

If both are FP32, those two tensors require roughly:

$$
8P
$$

bytes.

For a 7B-parameter model:

$$
7\times10^9\times8
=
56\times10^9
$$

bytes, or about 56 GB in decimal units, **only for the two FP32 moment tensors**.

This estimate still excludes:

- Model weights.
- Gradients.
- Activations.
- Possible FP32 master weights.
- Framework overhead.

### Important qualification

Modern techniques such as:

- ZeRO-style sharding.
- FSDP.
- Offloading.
- Lower-precision optimizer states.
- Quantization.

can change the actual memory footprint substantially.

The point is the scaling behavior, not a universal fixed number.

### Qwen Project Application

For a narrow VLM adaptation problem, paying optimizer-state cost for every pretrained parameter may be unnecessary.

### Design Decision

Training-memory efficiency is one of the reasons to prefer PEFT as our starting strategy.

---

### Story Bridge 14 — We Want Task Adaptation Without Paying to Train the Whole Model

Parameter-Efficient Fine-Tuning asks:

> Can we keep the pretrained model mostly fixed and learn only a small task-specific parameter set?

That is a natural match for our project goals.

## Question 14 — What is Parameter-Efficient Fine-Tuning?

Parameter-Efficient Fine-Tuning, or **PEFT**, keeps most pretrained parameters frozen and learns a much smaller trainable parameter set.

Let:

$$
\theta_0
=
\text{frozen base parameters}
$$

and:

$$
\phi
=
\text{trainable PEFT parameters}
$$

Then the model can be represented as:

$$
f(x;\theta_0,\phi)
$$

and the optimization becomes:

$$
\phi^*
=
\underset{\phi}{\operatorname{argmin}}
\;
\mathcal{L}
\left(
\theta_0,\phi
\right)
$$

while the base model is not directly updated.

### What PEFT can reduce

- Number of trainable parameters.
- Gradient storage for base weights.
- Optimizer-state memory for base weights.
- Task-specific checkpoint size.

### What PEFT does not remove

The frozen base model still participates in the forward computation.

So a model with only a small fraction of trainable parameters is not automatically proportionally cheaper in all forms of compute.

### Qwen Project Application

This aligns with our Part 1 design goal:

> Preserve broad Qwen capability while learning a narrow Little Content behavior.

### Design Decision

PEFT is the preferred adaptation family for the initial Qwen training design.

---

### Story Bridge 15 — LoRA Gives PEFT a Concrete Mathematical Form

Among PEFT methods, LoRA is especially attractive because it modifies the effect of a large linear layer through a small low-rank update.

We now need to derive that update.

## Question 15 — What is LoRA mathematically?

Consider a pretrained linear transformation:

$$
h
=
W_0x
$$

where:

$$
W_0
\in
\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

Full fine-tuning would directly update:

$$
W_0
$$

LoRA freezes $W_0$ and writes the task-specific update as:

$$
\Delta W
=
BA
$$

where:

$$
A
\in
\mathbb{R}^{r\times d_{\mathrm{in}}}
$$

and:

$$
B
\in
\mathbb{R}^{d_{\mathrm{out}}\times r}
$$

with:

$$
r
\ll
\min
\left(
d_{\mathrm{in}},
d_{\mathrm{out}}
\right)
$$

The effective transformation becomes:

$$
W
=
W_0+\Delta W
$$

and:

$$
h
=
W_0x
+
BAx
$$

A commonly used scaled form is:

$$
h
=
W_0x
+
\frac{\alpha}{r}
BAx
$$

where:

- $r$ is LoRA rank.
- $\alpha$ controls update scale.

### Intuition

Instead of learning an unrestricted update with:

$$
d_{\mathrm{out}}d_{\mathrm{in}}
$$

degrees of freedom, LoRA constrains the update to a lower-rank subspace.

### Qwen Project Application

The pretrained transformation remains intact.

The LoRA matrices learn a smaller task-specific correction for Little Content behavior.

### Design Decision

LoRA is our **primary PEFT method** for the theoretical project.

---

### Story Bridge 16 — Low Rank Is Useful Only If It Actually Saves Parameters

The factorization looks compact.

Now we should quantify how much smaller it is than a full matrix update.

## Question 16 — How many trainable parameters does LoRA add?

The original matrix has:

$$
N_{\mathrm{full}}
=
d_{\mathrm{out}}d_{\mathrm{in}}
$$

parameters.

LoRA adds:

$$
N_{\mathrm{LoRA}}
=
rd_{\mathrm{in}}
+
d_{\mathrm{out}}r
$$

or:

$$
N_{\mathrm{LoRA}}
=
r
\left(
d_{\mathrm{in}}
+
d_{\mathrm{out}}
\right)
$$

### Numerical example

Suppose:

$$
d_{\mathrm{in}}
=
d_{\mathrm{out}}
=
4096
$$

Then:

$$
N_{\mathrm{full}}
=
4096\times4096
=
16{,}777{,}216
$$

For:

$$
r=8
$$

LoRA uses:

$$
N_{\mathrm{LoRA}}
=
8\times4096
+
4096\times8
=
65{,}536
$$

The fraction is:

$$
\frac{
65{,}536
}{
16{,}777{,}216
}
\approx
0.003906
$$

or about:

$$
0.39\%
$$

of the original matrix size.

### Important qualification

This percentage is for **one matrix**.

The full model's trainable fraction depends on:

- Number of adapted layers.
- Number of adapted projections.
- Rank.
- Whether the connector is trainable.
- Whether vision modules are adapted.

### Qwen Project Application

This makes it possible to create a relatively small task-specific adapter while leaving the large Qwen backbone frozen.

### Design Decision

The final project will report the actual number and percentage of trainable parameters rather than merely saying "we used LoRA."


---

### Story Bridge 17 — LoRA Still Has Hyperparameters That Control Capacity and Update Strength

Saying "use LoRA" is not a complete design.

LoRA rank, scaling, and regularization determine how much task-specific capacity the adapter has and how strongly its update influences the frozen base model.

## Question 17 — What do LoRA rank, alpha, scaling, and dropout control?

### Rank

The rank is:

$$
r
$$

and controls the dimension of the low-rank update.

A larger rank gives:

- More trainable parameters.
- More adaptation capacity.
- More optimizer memory.
- Potentially greater overfitting risk.

A smaller rank gives:

- Fewer trainable parameters.
- Lower memory.
- Stronger capacity constraint.

### Alpha

A common LoRA scaling factor is:

$$
\frac{\alpha}{r}
$$

so the effective update is:

$$
\Delta W_{\mathrm{effective}}
=
\frac{\alpha}{r}BA
$$

Alpha controls the relative magnitude of the adapter contribution.

### LoRA dropout

A dropout operation can be applied on the LoRA path during training.

Conceptually:

~~~text
Input
   ↓
LoRA dropout
   ↓
A
   ↓
B
   ↓
Scaled adapter update
~~~

It can regularize the adapter, especially on smaller datasets.

### Initialization

A common design initializes the factors so that:

$$
BA
=
0
$$

at initialization.

Then:

$$
W
=
W_0
$$

at the start of training.

This lets the model begin from the pretrained behavior before the adapter learns a task-specific correction.

### Qwen Project Application

We should not choose:

~~~text
rank = 8
alpha = 16
dropout = 0.05
~~~

simply because those values are common online.

The appropriate values depend on:

- Model size.
- Dataset size.
- Adapted modules.
- Memory budget.
- Validation behavior.

### Design Decision

Part 2 defines the role of these hyperparameters.

The exact search values will be fixed in **Part 6**, where we design the complete training recipe.

---

### Story Bridge 18 — LoRA Has to Be Attached to Specific Matrices

A Transformer contains many linear transformations.

The adapter does not automatically know where to go.

So target-module selection is itself a model-capacity decision.

## Question 18 — Which Transformer matrices can receive LoRA adapters?

Self-attention commonly contains projection matrices:

$$
W_Q,\;
W_K,\;
W_V,\;
W_O
$$

corresponding to:

- Query.
- Key.
- Value.
- Attention output.

The feed-forward block also contains large linear transformations.

### Strategy A — Minimal attention adaptation

~~~text
Q projection
V projection
~~~

This is a relatively parameter-efficient starting point.

### Strategy B — Broader attention adaptation

~~~text
Q
K
V
O
~~~

This increases adaptation capacity.

### Strategy C — Attention + MLP adaptation

Adapters are placed on:

- Attention projections.
- Feed-forward linear layers.

This gives still more capacity.

### Tradeoff

~~~text
More target modules
        ↓
More trainable parameters
        ↓
More adaptation capacity
        ↓
More memory
+
potentially greater overfitting risk
~~~

### Qwen Project Application

The exact module names differ across Qwen variants and software implementations.

For example, generic names such as:

~~~text
q_proj
k_proj
v_proj
o_proj
~~~

should never be assumed without inspecting the selected model.

### Design Decision

Our **initial theoretical language-side LoRA baseline** will target:

> **Query and Value attention projections.**

If validation suggests under-capacity, we expand the target set systematically rather than immediately adapting every linear layer.

---

### Story Bridge 19 — A Vision-Language Model Has More Adaptation Choices Than a Text-Only LLM

Qwen receives the screenshot through a visual pathway before the language model makes the class decision.

That means we need to decide whether to adapt:

- Vision encoder.
- Multimodal connector.
- Language backbone.
- Some combination of them.

## Question 19 — Which VLM components should we freeze or adapt?

A simplified multimodal architecture is:

~~~text
Vision encoder
        ↓
Multimodal projector / connector
        ↓
Language model
~~~

There are several reasonable strategies.

### Strategy A — Freeze vision, adapt language only

Advantages:

- Very parameter-efficient.
- Preserves general visual representation.
- Good first test if pretrained visual features are already sufficient.

Risk:

- May not adapt enough to SERP-specific layout patterns.

### Strategy B — Freeze vision, train connector + language LoRA

Advantages:

- The connector can learn a more task-relevant mapping from visual features into language space.
- Much cheaper than full VLM fine-tuning.
- Preserves most pretrained visual capability.

### Strategy C — Add PEFT to upper vision layers

Advantages:

- Gives the visual representation some domain-specific flexibility.

Risks:

- More trainable parameters.
- More training complexity.
- Higher overfitting risk.

### Strategy D — Full multimodal fine-tuning

Advantages:

- Maximum adaptation capacity.

Risks:

- Highest memory and compute cost.
- Highest risk of over-specialization on a narrow dataset.

### Qwen Project Application

The task depends strongly on visual layout and content density.

Therefore, it would be risky to treat the screenshot pathway as irrelevant.

At the same time, the base Qwen model already has broad visual capability.

### Design Decision

Our initial theoretical configuration is:

~~~text
Vision encoder
Frozen initially

Multimodal connector / projector
Trainable if exposed separately

Language backbone
Frozen

Language attention
LoRA on Q and V projections
~~~

If validation shows systematic failure on visual-layout distinctions, we will test PEFT on upper vision layers.

---

### Story Bridge 20 — A Good Initial Adapter Placement Is a Hypothesis, Not a Truth

The previous configuration is intentionally a starting point.

A mature design should prove that additional adaptation capacity is useful before paying for it.

That calls for ablation experiments.

## Question 20 — How would we decide whether the initial LoRA placement is sufficient?

We can compare controlled variants while keeping the data and evaluation protocol fixed.

### Configuration A

~~~text
Language Q/V LoRA
Vision frozen
Connector trainable
~~~

### Configuration B

~~~text
Language Q/K/V/O LoRA
Vision frozen
Connector trainable
~~~

### Configuration C

~~~text
Language attention + MLP LoRA
Vision frozen
Connector trainable
~~~

### Configuration D

~~~text
Language LoRA
+
upper-vision PEFT
+
connector trainable
~~~

Compare:

- Validation precision.
- Validation recall.
- F1.
- Performance on unseen-host slices.
- Performance on hard visual-layout cases.
- Trainable parameter count.
- Peak training memory.
- Training time.

### How to interpret the result

If:

$$
\text{Performance}(A)
\approx
\text{Performance}(C)
$$

but A uses far fewer trainable parameters, A is preferable.

If A systematically fails on layout-specific examples and D fixes those failures without harming generalization, visual adaptation becomes justified.

### Qwen Project Application

This gives us a defensible interview answer:

> We started with the smallest plausible PEFT configuration and expanded capacity only when validation error analysis showed a need.

### Design Decision

Adapter placement will be selected through **ablation**, not habit.

---

### Story Bridge 21 — Fewer Trainable Parameters Do Not Mean the Whole Model Disappears During Training

LoRA can reduce trainable state dramatically.

But every training example still passes through the frozen base network.

So we need to understand what PEFT saves and what it does not.

## Question 21 — How does PEFT change training memory and compute?

With full fine-tuning, conceptually:

~~~text
Base weights
+
gradients for base weights
+
optimizer states for base weights
+
activations
~~~

With LoRA PEFT:

~~~text
Frozen base weights
+
LoRA gradients
+
optimizer states for LoRA
+
activations
~~~

Let:

$$
P
$$

be total base-model parameters and:

$$
p
$$

be trainable adapter parameters with:

$$
p\ll P
$$

Then gradient and optimizer-state storage associated with trainable parameters can shrink dramatically.

### But forward compute remains

The frozen model still performs:

- Attention.
- MLP computation.
- Vision processing.
- Multimodal fusion.

The base network is needed to compute the hidden states that the adapters modify.

### Important consequence

If only 1% of parameters are trainable, it does **not** mean training becomes 100 times faster.

The biggest PEFT savings are often:

- Optimizer memory.
- Gradient memory.
- Checkpoint size.

Compute savings can be much smaller than parameter-count savings.

### Qwen Project Application

LoRA makes adaptation practical without pretending that the underlying Qwen VLM is computationally tiny.

### Design Decision

When discussing LoRA, we will describe it as:

> **Parameter-efficient and optimizer-memory-efficient**, not computationally free.


---

### Story Bridge 22 — If the Frozen Base Model Is Still Too Large, We Can Compress the Base and Keep the Adapters Trainable

LoRA removes the need to optimize all base-model parameters.

But the frozen base weights still need to reside in memory.

If that memory requirement becomes the bottleneck, quantization can be combined with LoRA.

## Question 22 — What is QLoRA, and when would we use it?

QLoRA combines:

1. A quantized frozen base model.
2. Trainable LoRA adapters.

Conceptually:

~~~text
Base model weights
stored in quantized form
        ↓
Frozen

LoRA adapters
stored and trained at higher precision
        ↓
Updated during SFT
~~~

The goal is to reduce the memory required to hold the base model while retaining task-specific trainable adapters.

### Why this is different from ordinary LoRA

With ordinary LoRA:

~~~text
Base model
full training precision / normal inference precision
+
LoRA adapters
~~~

With QLoRA:

~~~text
Base model
quantized
+
LoRA adapters
trainable
~~~

### Important implementation details

Actual QLoRA behavior depends on:

- Quantization format.
- Compute dtype.
- Hardware support.
- Software stack.
- VLM implementation support.
- Which modules are quantized.

So QLoRA is not a purely mathematical replacement for LoRA; it is also an engineering choice.

### Qwen Project Application

If the selected Qwen VLM cannot fit comfortably for ordinary LoRA training on available hardware, QLoRA becomes an attractive option.

### Design Decision

Our preferred decision hierarchy is:

~~~text
LoRA
        ↓
If base-model memory is limiting:
QLoRA
        ↓
If PEFT underfits despite good data:
broader PEFT
        ↓
Full fine-tuning only if justified
~~~

QLoRA is primarily a **memory-efficiency strategy**, not automatically a better-performing training method.

---

### Story Bridge 23 — We Have All the Pieces; Now Follow One Example Through a Complete Training Step

At this point we understand:

- The input.
- The target.
- Teacher forcing.
- Token cross-entropy.
- Loss masking.
- Frozen versus trainable parameters.
- LoRA.

The best way to consolidate this is to follow one example from screenshot to optimizer update.

## Question 23 — What happens in one Qwen SFT training step?

Consider one labelled example.

### Step 1 — Load the screenshot

~~~text
SERP screenshot
~~~

### Step 2 — Apply the model's image processor

The exact processor depends on the selected Qwen variant.

It may perform operations such as:

- Resize.
- Normalize.
- Convert to tensor.
- Create model-specific image metadata.

We defer exact preprocessing details to Part 3.

### Step 3 — Construct the multimodal conversation

Conceptually:

~~~text
SYSTEM
You are a page-quality classifier.

USER
<image>
Classify the screenshot.

ASSISTANT
little_content
~~~

### Step 4 — Tokenize the textual portions

The system, user, and assistant text are converted into token IDs.

The image is represented through the model's visual pathway.

### Step 5 — Build attention and supervision masks

Context tokens:

~~~text
attention = active
loss = masked
~~~

Assistant target tokens:

~~~text
attention = active
loss = active
~~~

Padding:

~~~text
attention = inactive
loss = masked
~~~

### Step 6 — Forward pass

The model produces vocabulary logits:

$$
z_t
\in
\mathbb{R}^{V}
$$

for each relevant causal prediction position.

### Step 7 — Compute assistant-only cross-entropy

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

### Step 8 — Backpropagate

The gradient flows through the computation graph.

Frozen base parameters do not receive optimizer updates.

Trainable LoRA parameters and selected connector parameters do.

### Step 9 — Optimizer update

For trainable parameter set $\phi$:

$$
\phi_{k+1}
=
\phi_k
-
\eta
\widehat{g}_k
$$

where $\widehat{g}_k$ represents the optimizer-adjusted gradient.

### Step 10 — Repeat over batches

Across many labelled screenshots, the model gradually assigns more probability to the correct canonical class response.

### Qwen Project Application

This is the first complete theoretical training loop for the resume project.

### Design Decision

This SFT loop becomes the backbone of the final Qwen project file.

---

### Story Bridge 24 — Real Training Uses Batches, So Padding and Supervision Need Different Masks

Examples do not all have the same text length.

Images may also require model-specific batching behavior.

That means batching introduces multiple masking concepts that should not be confused.

## Question 24 — What is the difference between attention masking and loss masking?

These masks solve different problems.

### Attention mask

The attention mask identifies valid sequence positions rather than padding.

Conceptually:

$$
a_t
=
\begin{cases}
1 & \text{real token}\\
0 & \text{padding token}
\end{cases}
$$

Causal masking separately prevents access to future positions.

### Loss mask

The loss mask identifies which valid tokens should contribute to the supervised objective:

$$
m_t
=
\begin{cases}
1 & \text{assistant target token}\\
0 & \text{context or padding}
\end{cases}
$$

Therefore a user prompt token may have:

~~~text
attention mask = 1
loss mask = 0
~~~

because the model must read the prompt but should not be directly trained to reproduce it.

Padding may have:

~~~text
attention mask = 0
loss mask = 0
~~~

### Qwen Project Application

This distinction becomes important when we build the Qwen data collator in Part 3.

### Design Decision

The training pipeline will explicitly maintain:

> **attention validity** and **assistant supervision** as separate masks.

---

### Story Bridge 25 — Part 2 Should End With a Concrete Fine-Tuning Design, Not a List of LoRA Definitions

Part 1 told us why fine-tuning was needed.

Part 2 has now told us how the supervision signal is constructed and where the trainable capacity lives.

We should finish by reconstructing the whole training design in one flow.

## Question 25 — What is the Qwen fine-tuning design after Part 2?

The theoretical training pipeline is now:

~~~text
Labelled SERP screenshot
        ↓
Qwen multimodal processor
        ↓
Screenshot representation
+
task instruction
        ↓
Pretrained Qwen VLM
        ↓
Teacher-forced causal SFT
        ↓
Assistant-only token loss
        ↓
Cross-entropy / negative log-likelihood
        ↓
Backpropagation
        ↓
Frozen base weights

Trainable:
LoRA adapters
+
multimodal connector if separable
        ↓
Task-specific update
        ↓
Higher probability for the
correct Little Content label
~~~

### Primary adaptation strategy

~~~text
Vision encoder
Frozen initially

Multimodal connector / projector
Trainable if exposed separately

Language backbone
Frozen

Language attention
LoRA on Q and V projections
~~~

### Expansion strategy

If the starting setup underfits:

~~~text
Q/V LoRA
    ↓
Q/K/V/O LoRA
    ↓
Attention + MLP LoRA
    ↓
Upper-vision PEFT
    ↓
Full fine-tuning only if justified
~~~

### Production classification interface

The model remains generative.

The downstream decision layer can compare the likelihoods of the two canonical class responses rather than rely on unconstrained free-form generation.

### What remains intentionally unresolved

- Exact Qwen variant.
- Exact image processor.
- Exact chat template.
- Exact canonical label strings.
- LoRA rank.
- LoRA alpha.
- LoRA dropout.
- Batch size.
- Gradient accumulation.
- Learning rate.
- Optimizer.
- Scheduler.
- Number of epochs.
- Dataset split.
- Class balancing.
- Threshold.
- Calibration.
- Deployment configuration.

These belong to later parts rather than being guessed prematurely.

---

# Qwen Project Build Record — After Part 2

| Design element | Current state | Type |
|---|---|---|
| Business problem | Little Content detection | Resume Fact |
| Model family | Qwen Vision-Language Model | Resume Fact |
| Data modality | Labelled Bing SERP screenshots | Resume Fact |
| Production scale | About 50K URLs/day | Resume Fact |
| Core training paradigm | Generative Supervised Fine-Tuning | Design Decision |
| SFT context | Screenshot + task instruction | Design Decision |
| SFT target | Canonical Little Content class response | Design Decision |
| Training objective | Causal token-level negative log-likelihood | Design Decision |
| Teacher forcing | Used during SFT | Design Decision |
| Loss scope | Assistant target tokens only | Design Decision |
| Prompt tokens | Used as context; masked from direct loss | Design Decision |
| Image | Conditions the answer; not treated as a text target | Design Decision |
| Primary output formulation | Generative class label | Design Decision |
| Production class scoring | Compare canonical label likelihoods | Design Decision |
| Full fine-tuning | Not the default starting strategy | Design Decision |
| Preferred adaptation family | PEFT | Design Decision |
| Primary PEFT method | LoRA | Design Decision |
| Initial vision encoder | Frozen | Design Decision |
| Multimodal connector | Trainable if separable in chosen architecture | Design Decision |
| Language base weights | Frozen | Design Decision |
| Initial LoRA targets | Language attention Q and V projections | Design Decision |
| Expansion path | Q/K/V/O → MLP → upper-vision PEFT if validation requires | Design Decision |
| Adapter selection method | Controlled ablation | Design Decision |
| QLoRA | Memory-constrained alternative | Design Decision |
| Exact Qwen variant | Not established | Open Question |
| Exact target strings | Deferred to Part 3 | Open Question |
| Exact chat template | Deferred to Part 3 | Open Question |
| Exact image preprocessing | Deferred to Part 3 | Open Question |
| LoRA rank / alpha / dropout | Deferred to Part 6 | Open Question |
| Optimizer / LR / scheduler | Deferred to Part 6 | Open Question |
| Dataset split / leakage controls | Deferred to Part 7 | Open Question |
| Threshold / calibration | Deferred to Part 8 | Open Question |
| Deployment architecture | Deferred to Part 8 | Open Question |

---

# Part 2 — Key Equations

## SFT dataset

$$
\mathcal{D}_{\mathrm{SFT}}
=
\{
(x_i,y_i)
\}_{i=1}^{N}
$$

## Conditional response probability

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

## SFT negative log-likelihood

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

## Softmax

$$
P_\theta(k\mid c_t)
=
\frac{
e^{z_{t,k}}
}{
\sum_{j=1}^{V}e^{z_{t,j}}
}
$$

## Token cross-entropy

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

## Assistant-only masked loss

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

## PEFT objective

$$
\phi^*
=
\underset{\phi}{\operatorname{argmin}}
\;
\mathcal{L}
\left(
\theta_0,\phi
\right)
$$

## LoRA update

$$
\Delta W
=
BA
$$

## Scaled LoRA layer

$$
h
=
W_0x
+
\frac{\alpha}{r}
BAx
$$

## LoRA parameter count

$$
N_{\mathrm{LoRA}}
=
r
\left(
d_{\mathrm{in}}
+
d_{\mathrm{out}}
\right)
$$

## Approximate training-memory decomposition

$$
M_{\mathrm{train}}
\approx
M_{\mathrm{weights}}
+
M_{\mathrm{gradients}}
+
M_{\mathrm{optimizer}}
+
M_{\mathrm{activations}}
+
M_{\mathrm{runtime}}
$$

## Generative class score

$$
s(x)
=
\log
P_\theta
\left(
y^{(+)}
\mid x
\right)
-
\log
P_\theta
\left(
y^{(-)}
\mid x
\right)
$$

---

# Part 2 — Final Mental Model

~~~text
LABELLED EXAMPLE
Screenshot
+
Instruction
+
Correct response
        ↓

QWEN MULTIMODAL FORWARD PASS
Vision features
+
Text context
        ↓

TEACHER FORCING
Ground-truth response history is known
        ↓

TOKEN LOGITS
        ↓

ASSISTANT-ONLY CROSS-ENTROPY
Prompt conditions the answer
but does not receive direct target loss
        ↓

BACKPROPAGATION
        ↓

FROZEN BASE MODEL
+
TRAINABLE LoRA ADAPTERS
+
TRAINABLE CONNECTOR
        ↓

TASK-SPECIFIC UPDATE
        ↓

MODEL BECOMES MORE LIKELY TO PRODUCE
THE CORRECT LITTLE CONTENT LABEL
~~~

The central distinction is:

> **SFT defines the training signal; PEFT defines which parameters are allowed to respond to that signal.**

---

# Bridge to Part 3

Part 2 answered:

> **How do labelled examples produce gradients, and which parameters should we adapt?**

Part 3 will study:

- **10.3 Instruction Tuning**
- **10.4 Chat Format Training**

That is where the example becomes fully concrete:

~~~text
SERP screenshot
        ↓
Qwen image processor
        ↓
System message
        ↓
User instruction
        ↓
Assistant target
        ↓
Chat template
        ↓
Tokenization
        ↓
Loss labels
        ↓
Data collator
~~~

Part 3 will answer:

> **Exactly how should the Qwen Little Content examples be formatted and fed into the model?**
