# Part 10 — Fine-Tuning and Alignment

# Part 2B — Interview Style: Full Fine-Tuning, PEFT, LoRA and QLoRA

## Based on detailed Part 2B, Questions 12–25

- **Detailed source:** [Part 2B — Full Fine-Tuning, PEFT and LoRA](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20PEFT%20and%20LoRA.md)
- **Previous interview companion:** [Part 2 — SFT Interview Style](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202%20-%20Interview%20Style.md)

**How to use this file:** Begin with the simple spoken answer. Then explain the steps, illustrate the idea with Qwen, and only move to equations or difficult follow-ups when requested. Technical terms are introduced in plain language before the detailed treatment.

~~~text
30–60 second answer
        ↓
Technical follow-up
        ↓
Practical calculation or design choice
        ↓
Trade-offs and failure diagnosis
        ↓
Clear interview takeaway
~~~

**Evidence boundary:** Qwen Little Content detection is used as an ongoing **theoretical design scenario**. LoRA placement, connector trainability, exact model size, rank, optimizer settings and validation outcomes are proposed or illustrative unless supported by verified implementation evidence. A 7B model or a 4096-dimensional matrix is a numeric example, not a claim about the selected Qwen VLM.

---

# Question 12 — What is full fine-tuning?

### Simple interview answer — say this aloud

Full fine-tuning means taking a pretrained model and allowing its original weights to change while training it on our new task. When Qwen makes a wrong prediction, we calculate a loss and use it to update the selected model weights. It can give the model a lot of flexibility, but updating a large vision-language model also requires substantial memory and computing power.

### How it works — step by step

Start with the pretrained Qwen model. Give it labelled screenshots. Calculate how wrong its answer was. Backpropagation works out which trainable weights contributed to that error. An optimizer makes small changes to those original weights. Repeating this over many examples gradually adapts the model.

### Qwen Little Content example

We want Qwen to recognize low-content pages. Full fine-tuning could update its image-related and language-related weights, but we would first test whether training a smaller set of parameters is sufficient.

### How to understand the mathematics

W is a matrix of model weights. The gradient says which direction would reduce the loss. The learning rate controls the size of the update. Subtracting learning-rate times gradient changes the original matrix itself.

### If the interviewer asks for technical details For one trainable weight matrix $W$ and learning rate $\eta$, basic gradient descent is:

$$
W_{k+1}=W_k-\eta\nabla_W\mathcal L
$$

For example, if $W=0.5$, gradient is $0.2$ and $\eta=0.1$, the updated weight is $0.48$. Real models use large matrices and typically advanced optimizers, but the underlying principle is the same.

**Follow-up — Does full fine-tuning change teacher forcing or cross-entropy?**

No. SFT defines the training objective. Full fine-tuning defines **which original weights are allowed to move** in response.

**Practical decision — Would you full-fine-tune Qwen for Little Content immediately?**

Not necessarily. First establish a prompt-only baseline, data quality and hardware budget. Start with a smaller trainable approach if it plausibly solves the task, and consider broader fine-tuning only if controlled validation shows a need.

**Memory line:** Full fine-tuning updates the pretrained weights themselves.

---

---

# Question 13 — Why is full fine-tuning memory-intensive?

### Simple interview answer — say this aloud

Training a large model uses more memory than simply running it to get an answer. For inference, we mainly need the model weights and temporary information used to generate an output. During full fine-tuning, we also need information about how to change the weights, such as gradients, optimizer states and saved intermediate results. That is why a model that runs on one GPU might not fit there for full training.

### How it works — step by step

Think of memory as several separate bills. First, store the weights. Second, store gradients showing how trainable weights should change. Third, store extra numbers used by the optimizer, for example Adam's running averages. Fourth, keep intermediate activations so backward calculations work. Screenshot resolution and batch size can raise that fourth bill.

### Qwen Little Content example

Higher-resolution SERP screenshots may create more visual tokens and intermediate activations. Even if LoRA shrinks the number of trainable weights, we still have to process the images.

### How to understand the mathematics

P means parameter count. The 2P, 4P and 8P terms use specific illustrative storage assumptions for weights, gradients and Adam moments. The numerical total excludes several other costs and is not a universal GPU requirement.

**Numerical follow-up — Give an illustrative 7B-parameter budget.**

Suppose weights are stored in 16-bit format, gradients in FP32 and two Adam moments in FP32:

$$
M_{\mathrm{weights}}\approx 2P,\qquad
M_{\mathrm{gradients}}\approx4P,\qquad
M_{\mathrm{moments}}\approx8P
$$

For $P=7\times10^9$:

$$
14\ \mathrm{GB}+28\ \mathrm{GB}+56\ \mathrm{GB} =
98\ \mathrm{GB}
$$

before activations, temporary buffers and any master-weight copies. **This is an illustrative assumption set, not a universal requirement:** mixed precision, optimizer choice, offloading, sharding and checkpointing affect the total.

**Follow-up — Why might screenshots be particularly expensive?**

Image resolution and visual-token count can increase activation memory. Long text contexts and larger batches add further cost, regardless of whether base weights are trainable.

**Practical decision — What would you optimize first?**

Inspect the real memory profile, sequence/image lengths, batch size, checkpointing and precision; then consider PEFT and QLoRA if trainable state or frozen base weight storage is the bottleneck.

**Memory line:** Weight storage is only one term in training memory.

---

---

# Question 14 — What is PEFT? What exactly is an adapter?

### Simple interview answer — say this aloud

PEFT means Parameter-Efficient Fine-Tuning. Instead of changing all the original weights of a large model, we keep most of them fixed and train a much smaller part. An adapter is that small trainable addition. It changes how the model uses its existing knowledge. In LoRA, an adapter is built from two trainable matrices that provide a correction to a frozen layer.

### How it works — step by step

Imagine Qwen already has a strong ability to understand screenshots. We do not want to rebuild all of that knowledge. We leave its original computation in place and add a small trainable path that learns how to adjust the result for our particular task. The original path and the added path work together.

### Qwen Little Content example

Qwen may already recognize empty areas, text blocks and menus. A LoRA adapter could help it use those clues to make the Little Content decision while most original Qwen weights remain unchanged.

### How to understand the mathematics

W_0 is the frozen pretrained matrix. Delta W is a trainable correction. The two outputs are added. A frozen matrix still runs during the forward pass; it simply does not receive an optimizer update.

**Technical intuition:**

$$
h=W_0x+\Delta Wx
$$

The pretrained mapping $W_0x$ stays intact, while $\Delta Wx$ is the learned correction. LoRA later makes that correction inexpensive.

~~~text
Input hidden state
     ├── Frozen pretrained projection ──┐
     └── Trainable adapter correction ──┤ Add
                                      ↓
                              Adapted hidden state
~~~

**Follow-up — If the base is frozen, how can the model learn?**

Frozen does not mean removed. The base network still executes. The adapter changes the output, so loss gradients can update the adapter while the original weights remain unchanged.

**Follow-up — Is PEFT identical to LoRA?**

No. PEFT is a family of approaches; LoRA is one method. Other methods use trainable prompts or small added modules.

**Qwen application:** We might freeze pretrained language/vision weights initially and train language adapters plus a connector if one exists and is accessible in the selected Qwen version.

**Memory line:** Adapter = *where task-specific parameters live*; frozen weights still provide the original capability.

---

---

# Question 15 — What is LoRA? Explain rank, the matrices and the forward pass.

### Simple interview answer — say this aloud

LoRA helps us fine-tune a large model by freezing its original weights and learning a smaller correction. It uses two small trainable matrices called A and B. The rank is the width of the small middle path between those matrices. For example, with rank 8, the adapter passes information through eight intermediate values before expanding it back to the original output size. A higher rank gives more freedom to learn changes, but also uses more parameters.

### How it works — step by step

A normal layer takes an input and transforms it using its weight matrix. LoRA keeps that original transformation. At the same time, matrix A reduces the input to a smaller intermediate size, and matrix B expands it back. We scale this correction and add it to the original output. Only A and B need to learn in the basic LoRA setup.

### Qwen Little Content example

Imagine a Qwen attention projection with 4,096 input and output values. A rank-8 adapter follows 4,096 → 8 → 4,096. It does not make the whole Qwen model eight-dimensional, and rank does not mean the number of output classes.

### How to understand the mathematics

W_0 is the frozen projection. A maps the input size down to rank r. B maps rank r back to output size. Alpha divided by rank scales the correction. The product BA has the same shape as the original projection, but its rank is at most r.

### If the interviewer asks for technical details For column-vector convention:

$$
h=W_0x+\frac{\alpha}{r}BAx
$$

with:

$$
A\in\mathbb R^{r\times d_{\mathrm{in}}},
\quad
B\in\mathbb R^{d_{\mathrm{out}}\times r},
\quad
BA\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

~~~text
Input 4096
    ├── W0: 4096 → 4096 (frozen) ────────────┐
    └── A: 4096 → 8 → B: 8 → 4096 (trainable) ┤ Add
                                              ↓
                                       Output 4096
~~~

**Follow-up — What does rank 8 mean here?**

The adapter has an **eight-dimensional intermediate bottleneck**. It does not mean there are eight Qwen layers, eight classes or an eight-dimensional final hidden state. The update matrix has rank at most eight.

**Follow-up — Does LoRA decompose the original pretrained matrix?**

No. It keeps the original $W_0$ and factorizes only the learned correction $\Delta W$.

**Mathematical follow-up — How can the two matrices receive gradients?**

For $s=\alpha/r$, $h=W_0x+sBAx$, and $g=\partial\mathcal L/\partial h$:

$$
\frac{\partial\mathcal L}{\partial B}=s\,g\,(Ax)^{\mathsf T}
$$

$$
\frac{\partial\mathcal L}{\partial A}=s\,B^{\mathsf T}g\,x^{\mathsf T}
$$

So regular backpropagation learns the correction without changing $W_0$.

**Memory line:** LoRA = frozen original weight + trainable low-rank change.

---

---

# Question 16 — How many trainable parameters does LoRA use?

### Simple interview answer — say this aloud

LoRA saves trainable parameters because it trains two smaller matrices instead of one enormous weight update. To count its parameters, multiply the rank by the input size, then add the rank multiplied by the output size. For a 4,096-by-4,096 layer with rank 8, LoRA needs 65,536 trainable values rather than more than 16 million for a full-size matrix.

### How it works — step by step

The original layer has input-size times output-size weights. Matrix A contains rank times input-size weights, and matrix B contains output-size times rank weights. Add those two small counts. The smaller the rank relative to the layer dimensions, the larger the saving.

### Qwen Little Content example

If we add LoRA to many Qwen attention projections, we must count the adapter parameters of every selected projection and any other trainable parts. The 65,536 figure describes one illustrative projection, not the entire VLM.

### How to understand the mathematics

The formula r(d_in+d_out) comes directly from the shapes of A and B. At a fixed layer size, increasing rank increases the adapter parameter count in direct proportion.

### If the interviewer asks for technical details

$$
N_{\mathrm{full}}=d_{\mathrm{out}}d_{\mathrm{in}}
$$

$$
N_{\mathrm{LoRA}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})
$$

For a $4096\times4096$ projection with rank 8:

$$
N_{\mathrm{LoRA}}=8(4096+4096)=65{,}536
$$

versus:

$$
N_{\mathrm{full}}=4096^2=16{,}777{,}216
$$

That is approximately **0.39% as many trainable parameters for this projection**, not for the whole model.

**Follow-up — What if rank becomes 64?**

$$
N_{\mathrm{LoRA},64}=64(4096+4096)=524{,}288
$$

Larger rank increases adaptation capacity and trainable state, but does not guarantee better validation accuracy.

**Trick follow-up — Is rank 2 necessarily efficient on a 4-by-4 matrix?**

No. A rank-2 LoRA factorization uses $2(4+4)=16$ trainable parameters, as many as the original matrix. Savings depend on $r$ relative to layer dimensions.

**Memory line:** LoRA count is $r(d_{\mathrm{in}}+d_{\mathrm{out}})$, not $r^2$.

---

---

# Question 17 — What do rank, alpha, scaling, dropout and initialization control?

### Simple interview answer — say this aloud

LoRA has a few settings that control different things. Rank tells us how much room the adapter has to learn a correction. Alpha controls how strongly we scale that correction. Dropout is a training trick that randomly leaves out some adapter-path information to reduce overfitting. Initialization decides what the adapter looks like before learning starts. We should tune these settings based on validation results, not guess them.

### How it works — step by step

First choose how wide the adapter's middle path should be. Then choose the scaling rule that controls how much its output contributes alongside the frozen layer. Use dropout if regularization is useful. A common starting setup makes the adapter's initial correction zero so that training starts from the pretrained model's behavior.

### Qwen Little Content example

For screenshot classification, we could compare several ranks and alpha values while keeping evaluation data and the broader experiment setup consistent. We should not claim any particular setting worked until it has been tested.

### How to understand the mathematics

In the standard formula, alpha/r multiplies BA. Increasing rank while leaving alpha fixed also changes this scaling value. Some other LoRA variants use different scaling rules, so always state which one you mean.

### If the interviewer asks for technical details Under standard scaling:

$$
\Delta W_{\mathrm{LoRA}}=\frac{\alpha}{r}BA
$$

If $r=8$ and $\alpha=16$, the scaling factor is $2$. If $r=16$ and $\alpha=16$, it is $1$. Changing rank without considering scaling changes more than parameter count.

A common initialization sets $B=0$ and $A$ nonzero so the initial LoRA update is zero, preserving the base-model computation. The gradient for $B$ can initially be nonzero; after $B$ updates, $A$ also receives learning signal.

**Follow-up — Why not always maximize rank and alpha?**

Because larger capacity or scaling may increase memory, overfitting or optimization instability. Compare configurations on held-out task metrics, ideally under comparable budgets.

**Follow-up — Does every LoRA variant scale by alpha/r?**

No. Some variants use different scaling, such as square-root rank scaling; always name the implementation convention.

**Qwen application:** Treat rank, alpha, dropout and target modules as experimental settings. Do not invent a deployed value for the hypothetical project.

**Memory line:** Rank is capacity; alpha is scale; dropout is regularization; initialization sets the starting update.

---

---

# Question 18 — Where are LoRA adapters attached inside a Transformer?

### Simple interview answer — say this aloud

A Transformer has several weight matrices inside each attention layer. LoRA can be attached to selected matrices, for example those used to create queries, keys, values or attention outputs. We call this adapter placement: it tells us where we are adding the trainable correction. Rank is a different choice: it tells us how wide the correction path is inside each selected adapter.

### How it works — step by step

When attention runs, it creates different versions of the hidden states called queries, keys and values. Queries and keys help decide which positions should pay attention to one another; values carry information forward. A LoRA adapter can slightly change one of these transformations without changing its original weight matrix.

### Qwen Little Content example

Our proposed starting point is to adapt Q and V projections on the language side of Qwen, then compare broader options if validation shows those adapters are not enough. Q/V is a starting hypothesis, not a universal rule.

### How to understand the mathematics

The attention equation combines queries, keys and values. The LoRA equation replaces one chosen weight with the sum of its frozen original matrix and a low-rank correction. The exact tensor multiplication order depends on the implementation.

### If the interviewer asks for technical details In simplified attention:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
$$

We can replace a selected projection with an effective weight:

$$
W_Q^{\mathrm{eff}}=W_Q^{(0)}+\frac{\alpha}{r}B_QA_Q
$$

Exact multiplication order depends on row- or column-vector implementation conventions.

**Follow-up — Why start with Q and V?**

They offer a relatively small adaptation hypothesis: queries affect attention behavior and values affect transmitted information. But Q/V is not universally optimal; broader Q/K/V/O or MLP adaptation may be better.

**Practical scenario:** Rank 8 on Q and V does **not** mean eight adapters or eight Transformer layers. The number of adapter instances depends on how many matrices in how many layers are selected.

**Memory line:** Placement answers *where*; rank answers *how wide*.

---

---

# Question 19 — Which parts of a Qwen VLM should be frozen or adapted?

### Simple interview answer — say this aloud

A vision-language model such as Qwen has components that read the image and components that generate language. Some versions also have a connector that helps pass image information to the language model. We can choose which parts stay frozen and which are trained. I would begin with a small training setup, study the errors, and train more of the vision or language parts only when the results show that it is necessary.

### How it works — step by step

First, the visual system extracts information from the screenshot. Next, the model connects that information with the text instruction. Finally, the language model predicts the class label. If the model has enough visual information but makes the wrong decision, language adapters might help. If the visual information itself is missing, the problem may be earlier in the pipeline.

### Qwen Little Content example

For Little Content detection, I would initially consider frozen vision weights with language LoRA, and a trainable connector only if the selected Qwen architecture exposes a suitable one. This is a design to test, not an already proven implementation.

### How to understand the mathematics

Freezing means keeping a parameter set unchanged between optimizer steps. Visual features can still flow through frozen layers and influence gradients on downstream trainable modules.

~~~text
Screenshot → vision encoder → image features
                                    ↓
Instruction ─────────────→ multimodal representations
                                    ↓
                             language backbone
                                    ↓
                              assistant label
~~~

**Technical follow-up — Can a frozen vision encoder still support learning?**

Yes. It still computes $v=f_{\theta_v}(I)$. If a connector $u=g_{\phi_c}(v)$ is trainable and affects logits, $\phi_c$ can update while $\theta_v$ is unchanged.

**Follow-up — How do you know whether errors are visual or language-side?**

Study failures. If the right visual evidence is absent after preprocessing or encoding, language-side adaptation may not help. If evidence appears represented but classification is wrong, trainable language attention could be sufficient. This is a hypothesis requiring experiments.

**Qwen application:** Proposed starting point: freeze vision and language base, adapt selected language projections with LoRA, and train the connector if architecture and resources allow. Exact components depend on the selected checkpoint.

**Memory line:** Decide *which pathway lacks adaptability* before unfrozen capacity is expanded.

---

---

# Question 20 — How would you determine whether your LoRA placement is sufficient?

### Simple interview answer — say this aloud

I would not assume that one LoRA configuration is best. I would test a few options, compare how well they classify new screenshots, and look at the extra cost of each option. A controlled ablation means changing one meaningful part of a setup while keeping the other important conditions the same. This helps us learn whether more adapters actually solve a problem.

### How it works — step by step

Begin with a small Q/V LoRA baseline. Then test broader attention adapters, additional feed-forward adapters, or selected vision-side training. Use comparable data splits, labels and metrics. Look beyond a single accuracy number: inspect false positives, unusual layouts and unseen hosts.

### Qwen Little Content example

If the simplest adapter already performs well on new search-page layouts, I would keep it. If adapting visual layers fixes a repeated type of image-layout error, I would consider the extra complexity.

### How to understand the mathematics

This question is mainly experimental reasoning rather than one mathematical formula. Compare results under the same evaluation definitions and avoid changing many variables at once.

**Example comparison:**

| Configuration | What it tests |
|---|---|
| A: Language Q/V LoRA | Small baseline |
| B: Language Q/K/V/O LoRA | Broader attention adaptation |
| C: Attention + MLP LoRA | Greater language-side capacity |
| D: Language LoRA + targeted vision PEFT | Visual adaptation necessity |

**Follow-up — What would make you choose A over C?**

If unseen-host precision, recall and failure slices are similar but A has lower training and serving cost, prefer the simpler configuration.

**Follow-up — What if D improves a rare visual failure category but overall accuracy changes little?**

Evaluate the importance and prevalence of that category, any degradation elsewhere, statistical uncertainty and implementation cost. Aggregate accuracy alone can conceal business-critical improvements.

**Qwen application:** Include sparse-but-useful pages, login-dominant pages, novel templates and unseen hosts as error slices.

**Memory line:** Adapter placement is decided by evidence, not convention.

---

---

# Question 21 — How does PEFT affect training memory and compute?

### Simple interview answer — say this aloud

PEFT makes training lighter mainly because we stop calculating and storing trainable updates for most original model weights. That reduces gradient storage and optimizer memory. But the large frozen model still exists and must run to process inputs, so PEFT does not remove all the computation or guarantee that training becomes dramatically faster.

### How it works — step by step

Think of two costs separately. One is remembering which weights to update and the optimizer's extra information about them. The other is actually running the model on screenshots. LoRA can shrink the first cost a lot. It usually does not shrink the second cost by the same amount.

### Qwen Little Content example

If image processing and activations dominate memory, lowering LoRA rank might barely change overall GPU use. We would measure peak memory and throughput rather than estimate speed from trainable parameter count.

### How to understand the mathematics

P is the number of frozen base parameters, and p is the number of trainable adapter parameters. Adam moment storage scales with p for LoRA, not with all P, but frozen base-weight memory remains.

### If the interviewer asks for technical details Let $P$ be base-model parameters and $p$ trainable adapter parameters, with $p\ll P$. Under FP32 Adam moments:

$$
M_{\mathrm{moments,PEFT}}\approx 8p
$$

instead of:

$$
M_{\mathrm{moments,full}}\approx 8P
$$

The base-model weight memory still scales with $P$.

**Trick follow-up — LoRA updates 1% of parameters. Is it 100 times faster?**

No. Frozen layers still run in the forward pass, gradients still propagate through trainable paths, and activations still need memory. We should benchmark wall-clock throughput and peak memory independently.

**Practical scenario:** If most memory is consumed by high-resolution image activations, reducing adapter rank may have limited effect; examine resolution, batch size and activation checkpointing.

**Memory line:** LoRA drastically reduces trainable states, not the cost of executing the whole model.

---

---

# Question 22 — What is QLoRA, and when would you choose it?

### Simple interview answer — say this aloud

QLoRA is useful when ordinary LoRA still uses too much GPU memory. LoRA already freezes the original weights, but those weights still take up space. QLoRA stores the frozen base model in a more compact form, commonly 4-bit quantization, while keeping the small LoRA matrices trainable. So it can reduce base-model memory without changing the basic idea of supervised fine-tuning.

### How it works — step by step

First load the pretrained model's frozen weights using a compact numerical representation. Add trainable LoRA adapters to the chosen layers. During training, use the quantized base for computation and update the adapters rather than the frozen base. Check whether the chosen Qwen checkpoint and libraries support this safely.

### Qwen Little Content example

If our chosen Qwen VLM is too large for available GPU memory with regular LoRA, QLoRA is an option to test. It does not automatically give higher classification accuracy or faster inference.

### How to understand the mathematics

Four bits use half a byte per weight in ideal packing, compared with two bytes for 16-bit weights. Real memory is higher because quantization needs extra metadata and the training run also needs adapters and activations.

**Illustrative calculation:** A 7B-parameter base at 16 bits uses approximately:

$$
7\times10^9\times2=14\ \mathrm{GB}
$$

An ideal 4-bit packed representation uses approximately:

$$
7\times10^9\times0.5=3.5\ \mathrm{GB}
$$

Actual QLoRA memory is higher than the ideal number because of quantization metadata, higher-precision components, activations and adapters.

**Follow-up — What are NF4 and double quantization?**

NF4 is a four-bit format designed for approximately normally distributed weights. Double quantization further compresses quantization constants. Paged optimizers are an additional memory-management technique associated with QLoRA implementations.

**Follow-up — Are the 4-bit base weights trained directly?**

Not in the standard QLoRA approach. Quantized pretrained weights remain frozen; the LoRA adapter weights are optimized.

**Practical decision:** Use QLoRA if base-weight memory is a real bottleneck and the selected Qwen VLM, hardware and library stack support it; measure any quality, speed and compatibility trade-offs.

**Memory line:** LoRA saves trainable state; QLoRA also compresses frozen base weights.

---

---

# Question 23 — Describe one end-to-end Qwen SFT training step with LoRA.

### Simple interview answer — say this aloud

In one training step, I take a labelled screenshot, prepare the image and instruction in the format Qwen expects, and include the correct assistant label. Qwen predicts the next answer tokens. We calculate the loss only on the intended assistant targets, then backpropagate that error. The optimizer updates our chosen LoRA adapters and other unfrozen parts, while the pretrained frozen weights stay unchanged.

### How it works — step by step

The flow is: prepare an image and its label; build the chat example; tokenize text and process the image; verify the attention and loss masks; run a forward pass; calculate next-token cross-entropy; backpropagate; update trainable parameters. Repeat over batches and validate on separate data.

### Qwen Little Content example

An example might contain a SERP screenshot with the target Little Content. Qwen reads the screenshot and text instruction, and the LoRA adapters learn from how likely the correct class tokens were.

### How to understand the mathematics

The gradient is calculated with respect to the trainable adapter parameters phi. The optimizer adjusts phi using the learning rate and its chosen update rule. The frozen base parameters theta_0 do not change.

~~~text
Screenshot + user instruction
        ↓
Qwen processor and chat template
        ↓
Correct assistant label in training sequence
        ↓
Causal forward pass: next-token logits
        ↓
Assistant-only cross-entropy
        ↓
Backpropagation
        ↓
LoRA / connector optimizer update
~~~

**Technical follow-up — Show the optimizer update.**

For trainable parameter set $\phi$, with pretrained base $\theta_0$ frozen:

$$
g_k=\nabla_\phi\mathcal L(\theta_0,\phi_k)
$$

$$
\phi_{k+1}=\phi_k-\eta\widehat g_k
$$

Here $\widehat g_k$ is the optimizer-adjusted direction; basic SGD uses $\widehat g_k=g_k$.

**Follow-up — Why can all target-token logits be computed in one forward pass?**

Teacher forcing provides the correct response prefix at every position, and the Transformer can compute the causal-masked positionwise activations in parallel during training.

**Debugging scenario:** Check assistant labels and their next-token shift, exact processor/template, EOS supervision, nonzero adapter gradients and that frozen weights do not change after an optimizer step.

**Memory line:** Context → target → causal logits → assistant loss → selected adapter updates.

---

---

# Question 24 — What is the difference between attention masking, causal masking and loss masking?

### Simple interview answer — say this aloud

These masks answer different questions. An attention or padding mask tells the model which input positions are real and which are just padding. A causal mask stops the model from looking ahead at future answer tokens. A loss mask tells the training process which token predictions should count toward the error. For assistant-only SFT, the instruction is still readable but is usually not scored as an answer token.

### How it works — step by step

Imagine one example is shorter than another, so extra padding is added to make a batch. Padding should not be treated like meaningful words. During training, future tokens should be hidden from earlier positions. And when calculating the loss, we want the model to learn to produce the assistant label, not reconstruct the input prompt.

### Qwen Little Content example

The screenshot and instruction are useful context even when they do not contribute direct target loss. The label Little Content is part of what we actually supervise.

### How to understand the mathematics

These masks act at different stages. Some control what positions can attend to, while the loss mask controls which labels are counted. A common ignored-label value in PyTorch is -100.

**Illustrative batch:**

~~~text
USER + IMAGE + ASSISTANT + Little + Content + EOS + PAD
context tokens: used to condition predictions
assistant tokens: used as supervised next-token targets
padding: excluded from meaningful processing/loss
~~~

| Mechanism | Purpose |
|---|---|
| Attention-validity / padding mask | Avoid treating padding as input content |
| Causal mask | Prevent future-token information leakage |
| Assistant loss mask | Do not directly train prompt-token reconstruction |

**Follow-up — Can a user-prompt token be visible to attention but excluded from direct loss?**

Yes. That is normal for assistant-only SFT. Masking its loss does not mean deleting its contextual contribution.

**Practical debugging:** Inspect a processed batch. Decode exactly which labels are unmasked; verify the intended class response is supervised and that the model API's shifting behavior is understood. Actual multimodal padding behavior is checkpoint- and processor-dependent.

**Memory line:** Attention mask controls visibility/validity; loss mask controls supervision.

---

---

# Question 25 — Give the complete Qwen fine-tuning design and defend it.

### Simple interview answer — say this aloud

I would first define the Little Content classification problem and measure how well a pretrained Qwen model already handles it using prompting. If fine-tuning is needed, I would create clean screenshot-and-label examples and train Qwen to predict the correct class using supervised next-token loss. To reduce training cost, I would begin by testing LoRA rather than changing every model weight. Then I would compare other training options only if validation reveals a need, and check accuracy, precision, recall and behavior on new hosts before choosing the production decision threshold.

### How it works — step by step

The design is built in order: establish the business metric, collect and split labelled screenshots, test baselines, create the model-compatible training examples, apply teacher-forced assistant-only SFT, update selected adapters, study mistakes, test alternative adapter placements if justified, and validate the scoring and threshold.

### Qwen Little Content example

Our example uses Bing SERP screenshots with canonical Little Content and Not Little Content labels. The LoRA targets and exact training setup are proposed design choices, not confirmed steps of a completed experiment.

### How to understand the mathematics

The main equations connect three ideas: assistant-only cross-entropy defines the learning signal; the LoRA update shows the trainable correction; and the parameter-count formula explains why this correction can be cheaper than training full matrices.

**Follow-up — Why choose LoRA instead of full fine-tuning?**

It reduces trainable parameter and optimizer-state memory while retaining pretrained capability; whether its smaller capacity is sufficient must be measured.

**Follow-up — What if the model overfits specific hosts?**

Audit data splits and duplicates, diversify hosts/templates, investigate annotation bias, compare train-versus-validation metrics and consider regularization or a smaller adapter if appropriate.

**Follow-up — Would RLHF or DPO automatically make this binary classifier better?**

No. Reliable supervised binary labels already specify desired outcomes. Preference optimization may be unnecessary unless a distinct preference-learning problem has been demonstrated.

**Evidence boundary:** These are defensible *proposed* choices, not claims that this exact LoRA training run or validation result has occurred.

**Memory line:** Start with the task and baseline, construct valid supervision, choose efficient adaptation, test generalization and control the production decision.

---

---

# Integrated Case A — Numerical Whiteboard Question

**Interviewer:** A projection is 4096 by 4096. Compare full fine-tuning to LoRA rank 8, and explain how rank 64 changes the answer.

**Strong answer:** A full projection has 16,777,216 weights. Rank-8 LoRA has $8(4096+4096)=65{,}536$ trainable values, about 0.39% of that projection. Rank 64 uses $64(4096+4096)=524{,}288$ adapter parameters. Higher rank permits a richer correction but must justify its cost on validation; the base projection is still present and frozen.

**Follow-up:** Would this make the whole Qwen model 0.39% trainable?

**Answer:** Not necessarily. Total trainable fraction depends on which layers/projections are adapted, their dimensions, and any unfrozen projector or head.

---

# Integrated Case B — Selecting LoRA, QLoRA or Full Fine-Tuning

**Interviewer:** You have limited VRAM, labelled screenshots and a pretrained VLM. Which method would you choose?

**Strong answer:** I would first characterize model compatibility and memory, then choose the simplest method that fits and meets quality requirements. If ordinary LoRA fits, it is a sensible baseline. If frozen model weights still dominate memory, consider QLoRA with compatible 4-bit support. If small adapters systematically underfit despite adequate data and tuning, compare broader adapters, vision PEFT or full fine-tuning with controlled validation. I would not assume LoRA is universally sufficient or QLoRA universally faster.

---

# Integrated Case C — Troubleshoot Validation Failure

**Interviewer:** Q/V LoRA has improving training loss, but Little Content false positives rise on previously unseen hosts. What do you do?

**Strong answer:** First check host-disjoint splits, duplicates, image preprocessing and annotation consistency, then inspect sparse-but-useful hard negatives and class balance. Evaluate the label scoring and threshold. Only after separating data/scoring problems from representation problems would I try larger rank, broader attention adapters, trainable connector or vision PEFT as controlled ablations. Evaluate improvements on independent hosts and track precision/recall, not training loss alone.

---

# Part 2B — One-Minute Memory Map

~~~text
Full fine-tuning
Changes pretrained model weights
        ↓
Training memory can be expensive
Weights + gradients + optimizer states + activations
        ↓
PEFT
Freeze most pretrained weights
        ↓
LoRA adapter
Train A and B: narrow rank-r path
        ↓
Choose placement
Q/V → broader attention → MLP / vision if necessary
        ↓
If frozen base memory is the problem
Consider QLoRA
        ↓
Teacher-forced SFT
Assistant-only token loss
        ↓
Only unfrozen adapters / connector update
        ↓
Unseen-host evaluation + thresholding
~~~

# Key Equations to Recall

**LoRA effective weight:**

$$
W_{\mathrm{eff}}=W_0+\frac{\alpha}{r}BA
$$

**Adapter parameter count:**

$$
N_{\mathrm{LoRA}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})
$$

**Assistant-only SFT objective:**

$$
\mathcal L
=
-\frac{\sum_t m_t\log P_\theta(y_t\mid c_t)}{\sum_t m_t}
$$

# What We Should Not Claim Yet

The specific Qwen checkpoint, production training setup, LoRA rank and alpha, connector design, image resolution, precise tokenizer splits and validation improvements remain to be verified. An illustrative 7B model, 4096-dimensional projection or proposed module configuration must not be narrated as a completed historical experiment.

# One-Line Summary

> **SFT defines what prediction to learn; PEFT and LoRA define how to learn a task-specific correction efficiently, while proper validation determines whether that correction is actually useful.**
