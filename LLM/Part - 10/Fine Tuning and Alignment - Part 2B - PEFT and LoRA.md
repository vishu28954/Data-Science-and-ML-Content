# Part 10 — Fine-Tuning and Alignment

# Fine-Tuning and Alignment — Part 2B: Full Fine-Tuning, PEFT and LoRA

## 10.2 Supervised Fine-Tuning (SFT) — Questions 12–25

**Detailed Study Mode. Interview-style material remains separate.**

This companion file continues directly from [Part 2 — SFT Foundations, Questions 1–11](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202.md).

Part 2 established how labelled SERP screenshots become supervised assistant-token predictions and how token-level loss is calculated. Now we ask:

> **Once we have a training loss, which Qwen parameters should be allowed to change—and how can we adapt the model efficiently?**

Questions retain their original numbering **12–25** so the study sequence and references are unchanged.

The study pattern remains:

~~~text
Visible Story Bridge
    ↓
Natural Question
    ↓
Detailed Technical Explanation
    ↓
Mathematics / Example
    ↓
Qwen Project Application
    ↓
Design Decision
    ↓
Next Story Bridge
~~~

### Prerequisites from Part 2

- Input context includes screenshot and instructions.
- Target is a canonical assistant response such as `Little Content`.
- Teacher forcing uses the correct previous target tokens.
- Token cross-entropy measures how probable the correct target is.
- Assistant-only loss masking controls which prediction positions contribute to the loss.

**Scope of this file:** full fine-tuning, training memory, PEFT, LoRA, adapter placement, multimodal freezing, QLoRA and the end-to-end training loop. A later dedicated deep dive can explore LoRA derivations and hands-on implementations in more depth.

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
\theta_{k+1} = \theta_k - \eta \nabla_\theta \mathcal{L} \left( \theta_k \right)
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
M_{\mathrm{train}} \approx M_{\mathrm{weights}} + M_{\mathrm{gradients}} + M_{\mathrm{optimizer}} + M_{\mathrm{activations}} + M_{\mathrm{runtime}}
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

Adam maintains first- and second-moment estimates. To avoid confusing them with the loss mask $m_t$ from Question 7, write them as:

$$
\mu_k
$$

and:

$$
\nu_k
$$

where $k$ is the optimizer step.

If both moment tensors are stored in FP32, their combined memory is approximately:

$$
M_{\mathrm{Adam\ moments}}
\approx
8P
\text{ bytes}
$$

For a 7B-parameter model:

$$
7\times10^9\times8 =
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
\theta_0 =
\text{frozen base parameters}
$$

and:

$$
\phi =
\text{trainable PEFT parameters}
$$

Then the model can be represented as:

$$
f(x;\theta_0,\phi)
$$

and the optimization becomes:

$$
\phi^* = \arg\min_{\phi} \mathcal{L} \left( \theta_0, \phi \right)
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
h =
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

LoRA freezes $W_0$ and factorizes a low-rank update using:

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

The unscaled low-rank product is:

$$
BA
\in
\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

Using the common LoRA scaling factor $\alpha/r$, define the effective adapter update as:

$$
\Delta W_{\mathrm{LoRA}} =
\frac{\alpha}{r}
BA
$$

The effective weight matrix is:

$$
W =
W_0+\Delta W_{\mathrm{LoRA}}
$$

and the linear transformation becomes:

$$
h =
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
N_{\mathrm{full}} =
d_{\mathrm{out}}d_{\mathrm{in}}
$$

parameters.

LoRA adds:

$$
N_{\mathrm{LoRA}} =
rd_{\mathrm{in}}
+
d_{\mathrm{out}}r
$$

or:

$$
N_{\mathrm{LoRA}} =
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
d_{\mathrm{in}} = d_{\mathrm{out}} = 4096
$$

Then:

$$
N_{\mathrm{full}} = 4096\times4096 =
16{,}777{,}216
$$

For:

$$
r=8
$$

LoRA uses:

$$
N_{\mathrm{LoRA}} = 8\times4096 + 4096\times8 =
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
\Delta W_{\mathrm{LoRA}} =
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

A common initialization makes one LoRA factor zero so that initially:

$$
BA
=
0
$$

and therefore:

$$
\Delta W_{\mathrm{LoRA}} =
0
$$

so:

$$
W =
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

For the trainable parameter set $\phi$:

$$
g_k =
\nabla_\phi
\mathcal{L}
\left(
\theta_0,\phi_k
\right)
$$

An optimizer such as Adam transforms this raw gradient into an update direction $\widehat{g}_k$, after which:

$$
\phi_{k+1} = \phi_k -
\eta
\widehat{g}_k
$$

The frozen base parameters $\theta_0$ remain unchanged.

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
a_t =
\begin{cases}
1 & \text{real token}\\
0 & \text{padding token}
\end{cases}
$$

Causal masking separately prevents access to future positions.

### Loss mask

The loss mask identifies which valid tokens should contribute to the supervised objective:

$$
m_t =
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

# Qwen Project Build Record — After Part 2B (Questions 1–25)

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

# Part 2B — Key Equations

These equations build on the SFT loss and class-scoring equations covered in [Part 2](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202.md).

## Assistant-only SFT loss (recalled from Part 2)

$$
\mathcal{L}
=
-
\frac{\sum_t m_t\log P_\theta(y_t\mid c_t)}
{\sum_t m_t}
$$

## PEFT objective

$$
\phi^* = \arg\min_{\phi} \mathcal{L} \left( \theta_0, \phi \right)
$$

## LoRA update

$$
\Delta W_{\mathrm{LoRA}} = \frac{\alpha}{r} BA
$$

## Scaled LoRA layer

$$
h = W_0x + \frac{\alpha}{r} BAx
$$

## LoRA parameter count

$$
N_{\mathrm{LoRA}} = r \left( d_{\mathrm{in}} + d_{\mathrm{out}} \right)
$$

## Approximate training-memory decomposition

$$
M_{\mathrm{train}} \approx M_{\mathrm{weights}} + M_{\mathrm{gradients}} + M_{\mathrm{optimizer}} + M_{\mathrm{activations}} + M_{\mathrm{runtime}}
$$
# Part 2B — Final Mental Model

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

# Bridge to the Next Study Unit

Together, Parts 2 and 2B answered:

> **How do labelled examples produce gradients, and which parameters should we adapt?**

The upcoming instruction/chat-tuning unit will study:

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

That unit will answer:

> **Exactly how should the Qwen Little Content examples be formatted and fed into the model?**
