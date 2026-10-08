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

### Understand it using one small weight matrix

In Part 2, we already learned how assistant-token loss is calculated. Full fine-tuning answers a different question: **which original model weights should change because of that loss?**

Take a tiny linear layer:

$$
h=Wx
$$

Suppose its pretrained matrix is:

$$
W_0=
\begin{bmatrix}
0.2&0.5\\
-0.3&0.8
\end{bmatrix}
$$

After one training example, backpropagation might calculate:

$$
\nabla_W\mathcal{L}=
\begin{bmatrix}
0.1&-0.2\\
0.4&0.1
\end{bmatrix}
$$

With ordinary gradient descent and learning rate $\eta=0.1$:

$$
W_1=W_0-\eta\nabla_W\mathcal{L}
$$

so:

$$
W_1=
\begin{bmatrix}
0.19&0.52\\
-0.34&0.79
\end{bmatrix}
$$

**Notice what happened:** the entries of the original pretrained matrix changed. Real Qwen contains vastly larger matrices, but the logic is the same.

### What does one full fine-tuning step actually do?

~~~text
Screenshot + instruction + correct answer
        ↓
Forward pass with teacher forcing
        ↓
Assistant-only cross-entropy loss
        ↓
Backpropagation
        ↓
Gradients for all selected trainable weights
        ↓
Optimizer changes those pretrained weights
~~~

Full fine-tuning does not change the definition of SFT, teacher forcing or cross-entropy. It changes **how many pretrained parameters receive optimizer updates**.

### Why not always do this?

Qwen already knows a great deal about images and language. Our new task is narrow: identify visually low-content search pages. Updating billions of original weights can consume substantial training memory and compute, and it may over-specialize the model to a limited labelled dataset.

The next question is therefore unavoidable: **why does updating all those weights cost so much memory?**


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

### Build the memory cost one component at a time

A model with $P$ parameters does not require memory only for its weights during training. It may need gradients, optimizer states, saved activations and temporary buffers as well.

| Component | Purpose | Illustrative memory |
|---|---|---|
| Pretrained weights | Forward computation | $2P$ bytes if stored in FP16/BF16 |
| Gradients | Directions for trainable weight changes | $4P$ bytes if stored in FP32 |
| Adam moments | Optimizer tracks first and second moments | $8P$ bytes for two FP32 arrays |
| Activations | Needed to compute backward derivatives | Varies with batch, image resolution, context and checkpointing |
| Runtime buffers | Temporary and framework memory | Depends on implementation |

These precision choices are an **illustrative example**, not a universal training setup.

For seven billion trainable parameters, that example becomes:

$$
M_{\mathrm{weights}}
\approx 7\times10^9\times2
=14\ \mathrm{GB}
$$

$$
M_{\mathrm{gradients}}
\approx 7\times10^9\times4
=28\ \mathrm{GB}
$$

$$
M_{\mathrm{Adam\ moments}}
\approx 7\times10^9\times8
=56\ \mathrm{GB}
$$

Thus these three components alone would take:

$$
14+28+56=98\ \mathrm{GB}
$$

before activations, master weights or other overhead. Other implementations may use mixed precision, sharding, offloading or compressed states and produce substantially different totals.

### Why can inference fit while fine-tuning cannot?

~~~text
Inference:
weights + forward activations/cache

Full fine-tuning:
weights + backward activations
+ gradients + optimizer states + buffers
~~~

Inference does not need the optimizer moments used to train every weight.

### Why a VLM adds another challenge

Qwen receives images as well as text. Depending on its image processor, higher screenshot resolution can increase the amount of visual information flowing through the network. More tokens, longer sequences and larger batches can raise activation memory even when the parameter count stays fixed.

That is why Question 14 asks: **Can we avoid storing optimizer state and gradients for almost all pretrained parameters, without losing the pretrained model's useful representations?**


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

### The simple mental model: keep the old mapping, learn a small correction

We already know that a pretrained linear layer computes:

$$
h=W_0x
$$

Full fine-tuning changes the original matrix $W_0$. But a parameter-efficient method can instead keep $W_0$ frozen and add a trainable correction:

$$
h=W_0x+\Delta Wx
$$

Read this as:

~~~text
Original pretrained representation
                +
New task-specific correction
                ↓
Adapted representation
~~~

Our supervised assistant-token loss is unchanged; what changes is which parameters are trainable.

### What does frozen actually mean?

At successive optimizer steps, a frozen matrix stays unchanged:

$$
W_0^{(k+1)}=W_0^{(k)}
$$

But **frozen does not mean bypassed**. Qwen still executes the pretrained layer in its forward pass.

If a trainable adapter with parameters $\phi$ affects the output, then:

$$
h=f(x;W_0,\phi)
$$

The assistant loss can change $\phi$ even though it does not change $W_0$.

~~~text
Screenshot and instruction
        ↓
Frozen pretrained computation (still used)
        ↓
Trainable correction
        ↓
Assistant-token probabilities
        ↓
Masked cross-entropy
        ↓
Update only trainable correction
~~~

### An important catch

Simply making $\Delta W$ a second, full-size matrix would still require a large number of trainable parameters. We need the correction to be **small in trainable size**.

That leads directly to LoRA, which represents the correction as a product of two much smaller matrices.

### Interview check

If asked, "How can a frozen model learn?", the correct answer is: **The pretrained weights are fixed, but added trainable parameters modify the effective computation. Gradients reach those trainable parameters through the same assistant-token loss.**

PEFT is a family of methods. LoRA is one useful member of that family, rather than another name for all PEFT methods.


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

### Story continuation: a full-size correction is still expensive

Question 14 gave us:

$$
h=W_0x+\Delta Wx
$$

If $W_0$ has shape $4096\times4096$, an unrestricted correction $\Delta W$ also has $16{,}777{,}216$ entries. Freezing $W_0$ would not help enough if the new correction were just as expensive to train.

**LoRA's idea:** restrict the update itself so it can be built from two smaller trainable matrices.

### Step 1 — Understand the dimensions, not just the formula

Write the adapter as:

$$
\Delta W_{\mathrm{LoRA}}=\frac{\alpha}{r}BA
$$

where:

$$
A\in\mathbb{R}^{r\times d_{\mathrm{in}}}
$$

$$
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}
$$

Therefore:

$$
BA\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

The correction has the **same shape as the original weight matrix**, but its trainable parts are smaller.

~~~text
Input x: d_in dimensions
        ↓
Matrix A: maps d_in → r
        ↓
Small intermediate representation
        ↓
Matrix B: maps r → d_out
        ↓
Multiply by alpha/r
        ↓
Add to frozen W0 x
~~~

The adapter rank is at most $r$. LoRA does not require the **pretrained matrix** $W_0$ itself to be low-rank; only the learned **change** is constrained.

### Step 2 — A small example you can explain aloud

Suppose:

$$
W_0\in\mathbb{R}^{4\times4}
$$

Choose rank:

$$
r=2
$$

Then:

$$
A\in\mathbb{R}^{2\times4}
$$

and:

$$
B\in\mathbb{R}^{4\times2}
$$

For input $x\in\mathbb{R}^{4}$:

$$
Ax\in\mathbb{R}^{2}
$$

$$
B(Ax)\in\mathbb{R}^{4}
$$

So the result has exactly the shape needed to add it to $W_0x$.

The original $4\times4$ matrix has 16 weights; the two factor matrices also have 16 trainable values in this **tiny** example. Low rank is only parameter-efficient when $r$ is sufficiently small relative to the original layer dimensions. We will quantify that in Question 16.

### Step 3 — Why the adapter can learn

For one example, set:

$$
h=W_0x+sBAx,
\qquad
s=\frac{\alpha}{r}
$$

Let:

$$
g=\frac{\partial\mathcal{L}}{\partial h}
$$

denote the gradient arriving from later layers and assistant-token cross-entropy. With column-vector conventions:

$$
\frac{\partial\mathcal{L}}{\partial B}
=
s\,g\,(Ax)^{\mathsf T}
$$

$$
\frac{\partial\mathcal{L}}{\partial A}
=
s\,B^{\mathsf T}g\,x^{\mathsf T}
$$

The chain rule updates $A$ and $B$ while the original $W_0$ remains fixed.

You do **not** need to memorize these derivatives for every interview. You should understand the reason they exist: **the trainable adapter changes the output, so the final loss depends on its parameters.**

### Qwen connection

In the Qwen Little Content project, LoRA can change how a layer transforms its multimodal hidden representations. The pretrained mapping is retained, while a trainable correction can make the correct assistant label more probable.

This is a proposed fine-tuning design, not a claim that a particular Qwen variant has already been trained this way.


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

### Derive the parameter saving before looking at the answer

A full matrix update for:

$$
W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

would need:

$$
d_{\mathrm{out}}d_{\mathrm{in}}
$$

trainable values.

LoRA trains matrices $A$ and $B$ instead:

$$
N_{\mathrm{LoRA}}
=
rd_{\mathrm{in}}+rd_{\mathrm{out}}
$$

The fraction compared with the original matrix size is:

$$
\rho
=
\frac{
r(d_{\mathrm{in}}+d_{\mathrm{out}})
}{
d_{\mathrm{out}}d_{\mathrm{in}}
}
$$

For a square layer of width $d$:

$$
\rho=\frac{2r}{d}
$$

This makes the trade-off simple: with layer width fixed, doubling rank doubles LoRA parameter count.

### Rank comparison for one 4096 × 4096 projection

| Adaptation | Trainable values |
|---|---:|
| Original full matrix | 16,777,216 |
| LoRA rank 4 | 32,768 |
| LoRA rank 8 | 65,536 |
| LoRA rank 16 | 131,072 |
| LoRA rank 64 | 524,288 |

For rank 8:

$$
\frac{65{,}536}{16{,}777{,}216}
\approx0.003906
$$

or around $0.39\%$ **for this one projection**.

### Important practical qualification

The trainable fraction of a whole Qwen VLM depends on how many layers receive LoRA, which projections are selected, the rank of each adapter, and whether a connector or vision layers are also trainable.

A defensible experiment should report:

~~~text
Exact adapted modules
+
adapter rank for each module
+
total trainable parameter count
+
total model parameter count
        ↓
overall trainable percentage
~~~

**Interview follow-up:** Why isn't LoRA automatically small at any rank? Because if $r$ becomes large, $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ can approach or even exceed the number of entries in the original matrix.


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

### Think of four separate control knobs

The LoRA update is:

$$
\Delta W_{\mathrm{LoRA}}
=
\frac{\alpha}{r}BA
$$

But rank, alpha, dropout and initialization do **different jobs**.

| Knob | What it controls | What it does not guarantee |
|---|---|---|
| Rank $r$ | Maximum rank and parameter capacity of the update | Higher rank does not guarantee better validation |
| Alpha $\alpha$ | Scales the adapter contribution under standard $\alpha/r$ scaling | Larger alpha does not simply mean more learned knowledge |
| LoRA dropout | Regularizes the input to the adapter path during training | Does not replace an adequate validation split |
| Initialization | Determines the adapter's behavior before learning | Does not ensure the final adapter will generalize |

### Rank and alpha are related but not interchangeable

Suppose rank is 8 and alpha is 16:

$$
s=\frac{16}{8}=2
$$

If rank becomes 16 while alpha stays 16:

$$
s=\frac{16}{16}=1
$$

The second adapter has **more trainable parameters**, but a different scaling factor. Therefore, changing rank and alpha at the same time makes experiments hard to interpret.

Different LoRA variants may use different scaling rules. The expressions here describe the standard $\alpha/r$ convention used in this Part.

### Why zero initialization does not prevent learning

A common LoRA initialization makes $B$ zero while $A$ is nonzero. Then:

$$
BA=0
$$

so initially the model behaves like the pretrained model.

From the derivative discussed in Question 15:

$$
\frac{\partial\mathcal{L}}{\partial B}
=
s\,g\,(Ax)^{\mathsf T}
$$

This can be nonzero at the beginning, so $B$ starts learning. The initial derivative for $A$ is zero when $B=0$, but once $B$ moves away from zero, $A$ can learn too.

### How to choose these values for our task

Do not invent a final configuration yet. Treat the values as experimental choices. Hold the training data, evaluation slices and training budget fixed where possible, then compare a small set of ranks and scales. Check validation precision/recall, unseen-host performance, training stability and memory.

That is much stronger than saying the defaults worked elsewhere, so we used them here.


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

### Reconnect LoRA to the attention mechanism

From Transformer study, an attention layer forms query, key and value representations:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

Then, in simplified single-head notation:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)V
$$

Here:

- $Q$ and $K$ influence **which positions attend to which other positions**.
- $V$ influences **what information is carried forward** after attention weights are computed.
- $W_O$ mixes the attention outputs before they are passed onward.

### What changes if we attach LoRA to a projection?

Consider the query projection:

$$
W_Q^{\mathrm{eff}}
=
W_Q^{(0)}
+
\frac{\alpha}{r}B_QA_Q
$$

The projected queries become:

$$
Q
=
XW_Q^{\mathrm{eff}}
$$

The original query weights are frozen, but the new adapter term changes the resulting queries. The same principle can apply to keys, values and attention-output projections.

The exact tensor multiplication conventions depend on the implementation; what matters is that the effective linear transformation includes a frozen component plus a low-rank correction.

### Why begin with Q/V adapters?

Q/V is a **small starting hypothesis**, not a universal best practice. It alters how the model forms attention queries and carries values without training every projection.

Broader adaptation can increase capacity:

~~~text
Q + V adapters
        ↓
Q + K + V + O adapters
        ↓
Attention + feed-forward adapters
~~~

A better result from adding modules is meaningful only when the comparison uses the same dataset and evaluation protocol.

### Qwen screenshot intuition

If Qwen fails to use a visual feature that is already represented in its hidden states, adapting language-side attention could help. If the visual pathway fails to extract that feature at all, more language-side LoRA may not solve the problem. That distinction leads to Question 19.


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

### Follow the screenshot through the whole VLM

Our theoretical processing path is:

~~~text
SERP screenshot
        ↓
Vision encoder
        ↓
Visual features
        ↓
Multimodal connector / projector
        ↓
Language-model hidden representations
        ↓
Assistant logits
        ↓
Little Content / Not Little Content
~~~

Different Qwen VLM versions can arrange these parts differently. We must inspect the exact selected architecture rather than assume a separate connector always exists.

### Three different reasons an error may happen

Imagine Qwen misclassifies a page consisting mainly of a large empty area and a small login form.

**Possibility A — Vision representation problem:** the image pathway has not preserved a relevant layout detail.

**Possibility B — Vision-to-language alignment problem:** the necessary feature exists, but its integration into language-model hidden states is poor.

**Possibility C — Decision behavior problem:** the representation is available, but the language model applies the Little Content rule incorrectly.

These possibilities motivate different adaptation choices.

| Evidence from validation | Experiment to consider |
|---|---|
| Visual evidence looks useful but decision is wrong | Language-side LoRA |
| Visual features may be useful but poorly mapped | Trainable connector if accessible |
| Consistent failure on important visual patterns | PEFT on selected vision layers |
| Broader approaches still underfit and resources permit | Carefully controlled full fine-tuning |

This table gives hypotheses, not a guaranteed diagnosis.

### How can a frozen visual encoder still be useful?

Let:

$$
v=f_{\theta_v}(I)
$$

A trainable connector may form:

$$
u=g_{\phi_c}(v)
$$

Assistant cross-entropy can update $\phi_c$ through $u$ even if:

$$
\theta_v^{(k+1)}=\theta_v^{(k)}
$$

The frozen encoder still extracts visual features on every forward pass. Its weights simply do not update.

### Qwen design decision in plain language

Start with frozen vision features, train a compatible connector if separately exposed, and add a small amount of language-side LoRA. Escalate only if error analysis demonstrates the need.

Do not claim the chosen design has been empirically validated until experiments exist.


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

### Start with a specific failure, not a larger model

Suppose configuration A performs well on screenshots seen during training but often marks **sparse, yet useful** pages as Little Content. We need to decide whether this is caused by training data, thresholding or adapter capacity.

A controlled ablation changes one relevant design choice while holding other major variables steady.

### A practical experiment table

| Experiment | Main change | Why run it |
|---|---|---|
| A | Q/V LoRA, frozen vision | Small, inexpensive baseline |
| B | Q/K/V/O LoRA, frozen vision | Test broader attention adaptation |
| C | Attention + MLP LoRA, frozen vision | Test larger language-side capacity |
| D | Language LoRA + selected vision PEFT | Test whether visual adaptation is necessary |

If a separate connector is trainable in one configuration, keep its treatment consistent across the others when isolating the effects of language/vision modules.

### Control the experiment

Keep the following fixed as far as practical:

~~~text
Train/validation split
Annotation policy
Canonical class labels
Prompt and chat template
Input-image policy
Evaluation metric definitions
Random seeds or repeated-run reporting
Comparison training budget
~~~

Some configurations may need separate learning-rate tuning for a fair capability comparison. Report that explicitly instead of pretending every parameter choice was identical.

### What result would justify greater complexity?

Suppose A and C have similar unseen-host precision/recall, but C uses more trainable parameters and memory. Prefer A unless C offers another justified benefit.

If D fixes a reproducible pattern of visual-layout errors **on validation data from unseen hosts**, while maintaining other metrics, then adapting visual layers becomes defensible.

### Interview explanation

I would say: **Adapter placement is a hypothesis. I establish a small baseline, examine which errors it makes, change the adapted modules in controlled ablations and choose using generalization performance as well as compute cost.**

This is an engineering decision—not a popularity contest between LoRA configurations.


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

### Trace exactly what PEFT saves

With full fine-tuning, each selected pretrained parameter may need its weight, gradient and optimizer state.

With LoRA, the pretrained weights still exist but usually do not need optimizer state or trainable weight-gradient storage. Only the adapter and any other unfrozen modules do.

Let:

$$
P=\text{number of frozen base parameters}
$$

and:

$$
p=\text{number of trainable adapter parameters}
$$

with:

$$
p\ll P
$$

For an illustrative FP32 Adam setup, the two optimizer moments scale approximately as:

$$
M_{\mathrm{moments}}
\approx 8p
$$

instead of approximately:

$$
8P
$$

This can be a dramatic reduction when $p$ is very small.

### Why does the forward pass not disappear?

~~~text
Input screenshot
        ↓
Frozen vision/language layers
(still perform their computations)
        ↓
Trainable LoRA contribution
        ↓
Final prediction
~~~

The model must still calculate the frozen network's representations, because those representations are part of the prediction.

Backpropagation also needs some intermediate information to update adapters. Freezing the base does not mean training activations shrink in direct proportion to trainable parameter count.

### The interview trap

If someone says that training 1% of the parameters means training is 100 times faster, the explanation is:

> We may save most gradient and optimizer-state memory for the base weights, but forward compute, visual processing and many activation/backward costs remain. Parameter efficiency and end-to-end training speed are not the same quantity.

For our Qwen setup, memory efficiency may make a larger VLM trainable on limited hardware, but the actual throughput must still be measured.


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

### The problem left over after LoRA

LoRA reduces the number of trainable parameters, but the **frozen base model weights** still occupy memory.

That motivates quantizing the base model.

If a seven-billion-parameter base were stored in 16 bits per parameter:

$$
M_{\mathrm{base,16bit}}
\approx
7\times10^9\times2
=
14\ \mathrm{GB}
$$

At an ideal 4-bit packing cost:

$$
M_{\mathrm{base,4bit}}
\approx
7\times10^9\times0.5
=
3.5\ \mathrm{GB}
$$

These figures are **storage illustrations only**. Real QLoRA memory exceeds ideal packing because of quantization metadata, components stored at higher precision, adapter states, activations and runtime overhead.

### What actually remains trainable?

~~~text
Quantized base weights
        ↓
Frozen

LoRA matrices A and B
        ↓
Trainable

Optional separately exposed connector
        ↓
Trainable if deliberately selected
~~~

The quantized base participates in the forward computations, typically using supported higher-precision compute as needed. Its quantized stored values are not directly updated by the optimizer.

### Terms you should recognize

**NF4 (NormalFloat4)** is a 4-bit quantization format designed around the distribution of normally distributed weights. **Double quantization** further compresses quantization constants. **Paged optimizers** are an approach used to manage some memory spikes, although exact support depends on the software stack.

These are QLoRA engineering details, not changes to the teacher-forced SFT loss.

### LoRA versus QLoRA decision

LoRA is a good starting point if the model fits comfortably in training memory. QLoRA is worth evaluating if storing the frozen base is the bottleneck, provided the chosen Qwen VLM, hardware and training libraries support the approach.

QLoRA does not automatically improve model accuracy. We still need to validate quality, speed, numerical stability and component compatibility.


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

### Before the detailed steps: connect SFT and LoRA into one story

Our training example is a SERP screenshot, an instruction, and a canonical answer.

~~~text
Screenshot + "Classify this page"
        ↓
Qwen multimodal processor
        ↓
Visual + textual context
        ↓
Teacher-forced target: Little / Content / EOS
        ↓
Assistant-only token cross-entropy
        ↓
Backpropagation
        ↓
Trainable LoRA/connector weights update
~~~

The key point is that we do **not** run one independent training pass for each target token. Given the correct response tokens, the causal Transformer can produce the needed position-wise logits in one forward pass.

### A small, auditable example

Assume the model tokenizer splits the positive label as:

~~~text
Little | Content | EOS
~~~

This is an illustrative tokenization; the actual Qwen tokenizer must be checked.

Assume the following correct-token probabilities:

$$
p_{\mathrm{Little}}=0.8
$$

$$
p_{\mathrm{Content}}=0.6
$$

$$
p_{\mathrm{EOS}}=0.9
$$

Then the assistant-only mean loss is:

$$
\mathcal{L}
=
-\frac{1}{3}
\left(
\log0.8+\log0.6+\log0.9
\right)
\approx 0.280
$$

The system/user prompt positions are masked from **direct target loss**, but they still affect the three probabilities.

### What the optimizer is allowed to change

Suppose the selected PEFT configuration is:

~~~text
Vision encoder: frozen
Language base weights: frozen
Language Q/V LoRA: trainable
Connector: trainable if separately available
~~~

Only the selected trainable parameters are optimizer-updated. If we denote them jointly by $\phi$:

$$
g_k
=
\nabla_\phi
\mathcal{L}
\left(
\theta_0,\phi_k
\right)
$$

and an optimizer produces a step such as:

$$
\phi_{k+1}
=
\phi_k-\eta\widehat{g}_k
$$

where $\widehat{g}_k$ is the optimizer-adjusted update direction. The frozen base state remains:

$$
\theta_0^{(k+1)}
=
\theta_0^{(k)}
$$

### Common debugging checkpoints

Before claiming a training run is valid, verify:

- The exact multimodal processor and chat template match the selected Qwen checkpoint.
- Labels align with the **next-token logits** rather than the same-position logits.
- System and user prompt label positions have the appropriate ignore index.
- The assistant response and chosen EOS target are actually supervised.
- The total number of trainable parameters is what we intended.
- Frozen parameters remain unchanged after an optimizer step.
- Loss is finite and there is a nonzero gradient on trainable adapters for a nontrivial batch.

### Interview explanation

I would explain a training step as: **I construct a multimodal conversation, tokenize and process the screenshot, include the correct assistant response for teacher-forced training, mask prompt labels, run one causal forward pass, compute assistant-only cross-entropy, backpropagate, and update only the chosen LoRA and other unfrozen parameters.**


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

### Three masks or restrictions that are easy to mix up

Think about a padded training batch containing two examples:

~~~text
Example A:
[USER] [image] [ASSISTANT] Little Content EOS

Example B:
[USER] [image] [ASSISTANT] Not Little Content EOS PAD PAD
~~~

In a real Qwen VLM, image representation and padding can be more complex than this illustration.

We must keep three distinct concepts straight.

| Concept | Main question it answers |
|---|---|
| Attention validity / padding mask | Which input positions are real rather than padding? |
| Causal mask | Which future positions must each next-token prediction be unable to see? |
| Assistant loss mask | Which next-token targets count toward the supervised loss? |

### One token can be attended to but not directly supervised

A user instruction token may have:

~~~text
attention-validity mask = 1
assistant loss mask      = 0
~~~

The model reads it, but we do not score the model on recreating that instruction token.

An assistant target token may have:

~~~text
attention-validity mask = 1
assistant loss mask      = 1
~~~

A padding position may have:

~~~text
attention-validity mask = 0
assistant loss mask      = 0
~~~

The causal mask is applied separately so the model cannot look at future response tokens during training.

### Remember the shift from Question 4

~~~text
Input position:     ASSISTANT | Little | Content | EOS
Next-token target:     Little | Content | EOS
~~~

In common causal-LM APIs, ignored targets are represented by label value -100 and the library may apply the one-token shift internally. Always inspect the training API to avoid accidentally shifting twice or masking the wrong positions.

### Why this matters in practice

A mask error can produce a misleadingly low loss while the model never learns the intended assistant response. Inspect a few processed batches by decoding the non-ignored target labels and checking that they are exactly the intended canonical answers.


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

### Story Bridge 25 — Part 2B Should End With a Concrete Fine-Tuning Design, Not a List of LoRA Definitions

Part 1 told us why fine-tuning was needed.

Part 2 explained how the supervision signal is constructed; Part 2B has now shown where the trainable capacity lives.

We should finish by reconstructing the whole training design in one flow.

## Question 25 — What is the Qwen fine-tuning design after Part 2?

### Rebuild the design from the business task outward

Do not begin an interview by saying "I use LoRA because it is efficient." Begin with the prediction task and make each training choice follow from the previous one.

**Business requirement:** distinguish Little Content from Not Little Content using SERP screenshots, including unseen hosts and layouts.

**Training representation:** screenshot + instruction is conditioning context; canonical class text is the assistant target.

**Supervision:** teacher-forced causal next-token prediction with assistant-only cross-entropy.

**Adaptation:** test a small PEFT configuration first because full multimodal fine-tuning would cost considerably more memory and may not be necessary.

**Validation:** compare quality on the intended classification problem, including unseen hosts and visually difficult pages; track precision, recall and relevant error slices.

**Production implication:** prefer controlled canonical-label scoring and validate thresholds rather than relying on arbitrary free-form generations.

### The full decision chain

~~~text
Labelled screenshots
        ↓
Choose a canonical answer format
        ↓
Encode images and tokenize text
        ↓
Teacher-forced multimodal causal SFT
        ↓
Mask context labels; supervise assistant targets
        ↓
Compute mean next-token cross-entropy
        ↓
Backpropagate through trainable modules
        ↓
Q/V LoRA + optional trainable connector
        ↓
Compare with broader PEFT alternatives
        ↓
Select using unseen-host validation and cost
        ↓
Create controlled class score for serving
~~~

### Five questions a senior interviewer may ask

**Why not full fine-tuning?** Start with a cheaper, smaller trainable update; escalate if PEFT underfits and we have the data and resources to justify it.

**Why Q/V LoRA?** It is a narrow starting hypothesis. We test broader attention and vision-side adaptation if failure analysis points there.

**How can frozen vision help?** Frozen visual features still enter the forward pass; trainable downstream modules can use them and receive gradient from assistant loss.

**Why generate labels instead of using a classifier head?** Generative labels preserve the model's native multimodal instruction interface; a classifier head remains a legitimate baseline.

**How do you know the adapter improved the actual task?** Compare against a prompt-only baseline and other controlled configurations, with the same data protocol and independent validation slices.

### Reality check on the project build record

The project choices in this study are **theoretical design decisions**. We must not present selected LoRA target modules, exact hyperparameters or hypothetical validation improvements as historical implementation facts without evidence.

### Interview-ready conclusion

I would say: **I frame Little Content detection as supervised multimodal next-token training. The screenshot and instruction provide context; the canonical label provides assistant-only supervision. I then adapt Qwen with a small LoRA-based trainable parameter set while keeping the base model frozen, compare adapter placements using unseen-host validation, and select a controlled class-scoring approach for production.**


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

### Qwen Project Application

For the theoretical Little Content project, our current proposed configuration is:

~~~text
Pretrained Qwen VLM
        ↓
Frozen vision encoder initially
        ↓
Trainable connector if the architecture exposes one
        ↓
Frozen language backbone
with trainable Q/V LoRA adapters
        ↓
Assistant-only SFT for canonical class tokens
        ↓
Controlled evaluation on new hosts and layouts
~~~

Every configuration choice above is a **hypothesis to test**, not a measured production result. The actual Qwen variant, exact processor, training hyperparameters and validation results remain open.

### Design Decision

Our starting decision is to use **generative, teacher-forced assistant-only SFT** with a small LoRA-based trainable set. We retain broader language LoRA, upper-vision PEFT, QLoRA and full fine-tuning as alternatives justified by memory constraints or validation error analysis.

The next study unit will formalize the multimodal chat template, while later training and evaluation units will select exact implementation parameters.

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
