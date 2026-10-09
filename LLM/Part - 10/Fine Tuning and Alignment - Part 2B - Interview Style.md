# Part 10 — Fine-Tuning and Alignment

# Part 2B — Interview Style: Full Fine-Tuning, PEFT, LoRA and QLoRA

## Based on detailed Part 2B, Questions 12–25

- **Detailed source:** [Part 2B — Full Fine-Tuning, PEFT and LoRA](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20PEFT%20and%20LoRA.md)
- **Previous interview companion:** [Part 2 — SFT Interview Style](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202%20-%20Interview%20Style.md)

This follows the existing Part 1 interview style: **direct, speakable response → mechanism → one useful calculation/equation → Qwen example**, with additional applied and debugging follow-ups.

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

**Interview answer:** Full fine-tuning starts from pretrained model weights and allows the selected full model parameter set to be updated by the downstream loss. For a generative VLM, supervised response-token cross-entropy is backpropagated through the model, and its trainable pretrained matrices change.

**Technical expansion:** For one trainable weight matrix $W$ and learning rate $\eta$, basic gradient descent is:

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

# Question 13 — Why is full fine-tuning memory-intensive?

**Interview answer:** Training must store much more than the model weights. It may also keep gradients for trainable parameters, optimizer states such as Adam moments, backward activations and temporary buffers. That is why a model which fits for inference may not fit for full fine-tuning.

**Numerical follow-up — Give an illustrative 7B-parameter budget.**

Suppose weights are stored in 16-bit format, gradients in FP32 and two Adam moments in FP32:

$$
M_{\mathrm{weights}}\approx 2P,\qquad
M_{\mathrm{gradients}}\approx4P,\qquad
M_{\mathrm{moments}}\approx8P
$$

For $P=7\times10^9$:

$$
14\ \mathrm{GB}+28\ \mathrm{GB}+56\ \mathrm{GB}
=
98\ \mathrm{GB}
$$

before activations, temporary buffers and any master-weight copies. **This is an illustrative assumption set, not a universal requirement:** mixed precision, optimizer choice, offloading, sharding and checkpointing affect the total.

**Follow-up — Why might screenshots be particularly expensive?**

Image resolution and visual-token count can increase activation memory. Long text contexts and larger batches add further cost, regardless of whether base weights are trainable.

**Practical decision — What would you optimize first?**

Inspect the real memory profile, sequence/image lengths, batch size, checkpointing and precision; then consider PEFT and QLoRA if trainable state or frozen base weight storage is the bottleneck.

**Memory line:** Weight storage is only one term in training memory.

---

# Question 14 — What is PEFT? What exactly is an adapter?

**Interview answer:** Parameter-Efficient Fine-Tuning, or PEFT, adapts a pretrained model by optimizing a relatively small set of parameters while keeping most original weights frozen. An **adapter** is the added or selected trainable component that changes the effective computation. In LoRA, the adapter is a small pair of trainable matrices attached to a frozen linear projection.

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

# Question 15 — What is LoRA? Explain rank, the matrices and the forward pass.

**Interview answer:** LoRA, Low-Rank Adaptation, freezes an original linear weight matrix $W_0$ and learns a correction through two smaller trainable matrices $A$ and $B$. The **rank** $r$ is the dimension of the small intermediate path and limits the learned update to at most $r$ independent directions.

**Technical expansion:** For column-vector convention:

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

# Question 16 — How many trainable parameters does LoRA use?

**Interview answer:** A full matrix of shape $d_{\mathrm{out}}\times d_{\mathrm{in}}$ has $d_{\mathrm{out}}d_{\mathrm{in}}$ weights. LoRA trains $A$ and $B$, requiring $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ parameters.

**Technical expansion:**

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

# Question 17 — What do rank, alpha, scaling, dropout and initialization control?

**Interview answer:** Rank controls how expressive the low-rank correction can be; alpha controls its scaling under standard LoRA; adapter-path dropout regularizes training; initialization determines how the adapter behaves before updates.

**Technical expansion:** Under standard scaling:

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

# Question 18 — Where are LoRA adapters attached inside a Transformer?

**Interview answer:** LoRA attaches to selected **linear projection matrices**, commonly attention projections such as query, key, value and output, and sometimes MLP projections. Adapter placement means *which matrices receive the correction*; rank means *the width of each correction*.

**Technical expansion:** In simplified attention:

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

# Question 19 — Which parts of a Qwen VLM should be frozen or adapted?

**Interview answer:** The main candidates are the vision encoder, any separately exposed multimodal connector/projector, and the language backbone. I would start by minimizing trainable parameters and expand only when validation error analysis suggests where extra capacity is needed.

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

# Question 20 — How would you determine whether your LoRA placement is sufficient?

**Interview answer:** Treat the initial adapter placement as a hypothesis and perform **controlled ablations** with consistent data, labels, preprocessing, evaluation metrics and reasonable compute budgets.

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

# Question 21 — How does PEFT affect training memory and compute?

**Interview answer:** PEFT usually reduces **trainable gradient and optimizer state memory** because most pretrained weights are frozen. It does **not** eliminate the frozen base weights, forward computations, image processing or all activation-memory costs. Parameter-count reduction and throughput improvement are different measurements.

**Technical expansion:** Let $P$ be base-model parameters and $p$ trainable adapter parameters, with $p\ll P$. Under FP32 Adam moments:

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

# Question 22 — What is QLoRA, and when would you choose it?

**Interview answer:** QLoRA combines a **frozen quantized base model**, typically using 4-bit weight storage, with trainable LoRA adapters. It further reduces base-weight storage when regular LoRA still does not fit the available memory. The SFT loss remains the same.

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

# Question 23 — Describe one end-to-end Qwen SFT training step with LoRA.

**Interview answer:** I load and preprocess a labelled screenshot, construct the supported multimodal chat example, tokenize the instruction and canonical assistant response, create causal and supervision masks, run a teacher-forced forward pass, compute assistant-token cross-entropy, backpropagate, and update only chosen LoRA and other unfrozen parameters.

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

# Question 24 — What is the difference between attention masking, causal masking and loss masking?

**Interview answer:** A padding/attention-validity mask distinguishes real input positions from padding; a causal mask prevents attending to future token positions; a loss mask selects which next-token predictions contribute to supervised loss. They solve separate problems.

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

# Question 25 — Give the complete Qwen fine-tuning design and defend it.

**Interview answer (approximately 60 seconds):**

I would frame Little Content detection as supervised classification over SERP screenshots, first checking a prompt-only Qwen baseline and data quality. I would use the checkpoint-compatible multimodal processor to build screenshot-and-instruction inputs with canonical assistant class targets. I would train using teacher-forced causal next-token cross-entropy and mask prompt positions from direct loss.

Given the size of a VLM and a narrow classification objective, I would evaluate a PEFT baseline: frozen pretrained weights with LoRA on selected language projections, optionally a trainable multimodal connector if supported. I would compare this with broader adapters, vision-side adaptation or QLoRA only as justified by memory and validation. I would score allowed class labels consistently, tune the decision threshold for the business's false-positive cost and validate on unseen hosts and layouts.

**Follow-up — Why choose LoRA instead of full fine-tuning?**

It reduces trainable parameter and optimizer-state memory while retaining pretrained capability; whether its smaller capacity is sufficient must be measured.

**Follow-up — What if the model overfits specific hosts?**

Audit data splits and duplicates, diversify hosts/templates, investigate annotation bias, compare train-versus-validation metrics and consider regularization or a smaller adapter if appropriate.

**Follow-up — Would RLHF or DPO automatically make this binary classifier better?**

No. Reliable supervised binary labels already specify desired outcomes. Preference optimization may be unnecessary unless a distinct preference-learning problem has been demonstrated.

**Evidence boundary:** These are defensible *proposed* choices, not claims that this exact LoRA training run or validation result has occurred.

**Memory line:** Start with the task and baseline, construct valid supervision, choose efficient adaptation, test generalization and control the production decision.

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
