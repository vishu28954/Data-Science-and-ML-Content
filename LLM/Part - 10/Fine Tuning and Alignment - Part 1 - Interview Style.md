# Part 10 — Fine-Tuning and Alignment

# Part 1 — Interview Style

## 10.1 Pretraining vs Fine-Tuning

This file is the interview companion to the detailed Part 1 study notes.

The goal is not to reproduce the chapter word-for-word. The goal is to give **concise, speakable answers** that can be expanded when an interviewer probes deeper.

The preferred interview pattern is:

~~~text
Direct answer
    ↓
One key intuition
    ↓
One equation or example if useful
    ↓
Qwen project connection
~~~

---

# Question 1 — What problem are you solving in the Qwen project?

**Answer:**

The task is Little Content detection on Bing SERP screenshots. I use a pretrained Qwen Vision-Language Model because the decision depends on visual-layout and content-density information, not only text.

At a simplified level:

$$
f_\theta(x)\rightarrow y
$$

where $x$ is the SERP screenshot and:

$$
y\in
\{
\text{Little Content},
\text{Not Little Content}
\}
$$

The main challenge is not teaching the model basic vision. It is teaching a task-specific mapping from SERP evidence to the Little Content decision.

**Project fact:** The current project description uses labelled Bing SERP screenshots and a Qwen VLM.

---

# Question 2 — What is pretraining?

**Answer:**

Pretraining is the broad learning stage in which the model learns general reusable representations before seeing my specific downstream task.

For a causal language model, the basic objective is:

$$
\mathcal{L}_{\mathrm{pre}}
=
-\sum_{t=1}^{T}
\log
P_\theta
\left(
x_t\mid x_{<t}
\right)
$$

For a multimodal model, image information can also condition the generated text.

The main idea is that pretraining gives Qwen broad visual-language capability, but it does not automatically give it Bing's exact Little Content decision rule.

**Memory line:** Pretraining learns broad capability; fine-tuning specializes behavior.

---

# Question 3 — Why is pretraining not enough for the Little Content task?

**Answer:**

Because general understanding and task-specific decision-making are different objectives.

A pretrained Qwen model may already recognize sparse layouts, text regions, navigation, blank space, and visual hierarchy. But it was not necessarily optimized to reproduce our exact Little Content labels.

Conceptually:

$$
h=f_{\theta_0}(x)
$$

gives a useful pretrained representation, but we still need:

$$
h
\rightarrow
\text{Little Content decision}
$$

Fine-tuning adapts that mapping.

---

# Question 4 — What is fine-tuning mathematically?

**Answer:**

Fine-tuning starts from pretrained parameters rather than random initialization.

Let:

$$
\theta_0
$$

be the pretrained parameters and:

$$
\mathcal{D}_{\mathrm{task}}
=
\{
(x_i,y_i)
\}_{i=1}^{N}
$$

be the downstream dataset.

We initialize from $\theta_0$ and optimize the task loss:

$$
\theta^*
=
\underset{\theta}{\operatorname{argmin}}
\;
\mathcal{L}_{\mathrm{task}}
\left(
\theta;
\mathcal{D}_{\mathrm{task}}
\right)
$$

A gradient update is:

$$
\theta_{k+1}
=
\theta_k
-
\eta
\nabla_\theta
\mathcal{L}_{\mathrm{task}}
$$

For the Qwen project, the task examples are labelled SERP screenshots.

---

# Question 5 — How is fine-tuning different from training from scratch?

**Answer:**

Training from scratch starts from randomly initialized parameters and must learn both general representations and the downstream task.

Fine-tuning starts from a model that already has useful learned representations and adapts it to the target behavior.

~~~text
Training from scratch
Random model
→ learn representations
→ learn task

Fine-tuning
Pretrained model
→ adapt to task
~~~

For the Qwen project, it is much more sensible to reuse existing vision-language capability than to relearn multimodal understanding from the SERP dataset alone.

---

# Question 6 — What is transfer learning in this project?

**Answer:**

Transfer learning means reusing knowledge learned during pretraining for the downstream task.

The pretrained model may already encode information about:

- Text regions.
- Layout.
- Visual density.
- Blank space.
- Navigation.
- Semantic content.

Fine-tuning uses those existing capabilities and adapts them toward Little Content classification.

**Memory line:** Pretraining learns reusable features; downstream training teaches how those features should be used for this task.

---

# Question 7 — Prompting versus fine-tuning: what is the difference?

**Answer:**

Prompting changes the input while keeping the model parameters fixed:

$$
P_{\theta_0}
\left(
y\mid x,\text{instruction}
\right)
$$

Fine-tuning changes model parameters or a smaller trainable adapter set:

$$
\theta_0
\rightarrow
\theta^*
$$

I would first evaluate a prompt-only Qwen baseline. If prompting already meets the requirement, fine-tuning may not be necessary.

If there is a persistent task-specific error pattern, fine-tuning becomes more justified.

---

# Question 8 — When would you use RAG instead of fine-tuning?

**Answer:**

I separate the two by asking whether the problem is primarily about **knowledge** or **behavior**.

~~~text
RAG
→ provide external information at inference time

Fine-tuning
→ change or specialize model behavior
~~~

RAG is more suitable when the model needs changing or external knowledge.

Fine-tuning is more suitable when I want a persistent task behavior, such as a particular classification policy.

For Little Content detection, the evidence is primarily in the screenshot itself, so fine-tuning is more central than RAG.

---

# Question 9 — When should you not fine-tune?

**Answer:**

I would avoid fine-tuning as the first choice if:

- Prompting already solves the task.
- The real problem is missing fresh knowledge.
- I do not have reliable labelled data.
- The target behavior changes very frequently.
- A simpler model already performs well enough.
- The training and operational complexity is not justified.

The principle is:

> Fine-tuning should solve a demonstrated behavioral gap, not be used automatically.

---

# Question 10 — What exactly should the fine-tuned Qwen model learn?

**Answer:**

It should learn the relationship:

$$
P_\theta(y\mid x)
$$

where the output corresponds to Little Content or Not Little Content.

But more importantly, it should learn the **concept**, not shortcuts.

I want the model to rely on task-relevant signals such as content density, layout structure, meaningful primary content, and navigation dominance.

I do not want it to memorize specific hosts, templates, or screenshot artifacts.

---

# Question 11 — How can you think about fine-tuning as learning a decision boundary?

**Answer:**

A useful conceptual model is:

$$
h=f_\theta(x)
$$

followed by a task-relevant score:

$$
s=g(h)
$$

and a simplified binary decision:

$$
\hat{y}
=
\begin{cases}
1 & s\ge\tau\\
0 & s<\tau
\end{cases}
$$

A generative VLM may not literally use a single scalar classifier head, but this picture helps explain why examples near the boundary are important.

For example, a nearly empty page may be an easy positive, while a visually sparse page containing one strong answer block may be a hard negative.

---

# Question 12 — What is more important: memorization or generalization?

**Answer:**

Generalization.

The model is optimized on:

$$
\mathcal{D}_{\mathrm{train}}
$$

but what matters is performance on unseen examples from the target distribution:

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{target}}}
\left[
\mathcal{L}
\left(
f_\theta(x),
y
\right)
\right]
$$

For the Qwen project, a model that only performs well on known hosts or repeated templates is not useful.

The target is unseen SERP layouts, hosts, and templates.

---

# Question 13 — What is domain shift here?

**Answer:**

The pretraining distribution and the downstream task distribution are different:

$$
p_{\mathrm{pre}}
\neq
p_{\mathrm{task}}
$$

The Qwen project contains two kinds of specialization:

~~~text
Domain adaptation
General multimodal data
→ SERP screenshots

Task adaptation
General visual-language understanding
→ Little Content classification
~~~

Fine-tuning helps move the model toward this narrower domain and task.

---

# Question 14 — Can fine-tuning damage pretrained capabilities?

**Answer:**

Yes.

If the downstream data is narrow or updates are too aggressive, the model can over-specialize and lose useful pretrained behavior.

Conceptually:

$$
\theta_0
\rightarrow
\theta^*
$$

should improve the task while preserving useful capability as much as possible.

This connects to catastrophic forgetting, which is covered later in the syllabus.

---

# Question 15 — Does fine-tuning mean updating every model parameter?

**Answer:**

No.

There are two broad approaches.

### Full fine-tuning

Most or all parameters are trainable:

$$
\theta_0
\rightarrow
\theta^*
$$

### Parameter-Efficient Fine-Tuning

Most pretrained parameters remain frozen and a smaller trainable parameter set is optimized:

$$
f(x;\theta_0,\phi)
$$

where $\theta_0$ is frozen and $\phi$ is trainable.

LoRA is one important PEFT method.

For the Qwen project, the exact full-FT versus PEFT choice is intentionally deferred until we study LoRA properly in Part 2.

---

# Question 16 — What baselines would you compare against before fine-tuning?

**Answer:**

I would want at least:

1. A prompt-only pretrained Qwen baseline.
2. The fine-tuned Qwen model.
3. An appropriate existing, text-only, heuristic, or simpler supervised baseline if available.

The reason is simple:

> Fine-tuning should demonstrate incremental value over simpler alternatives.

For the project, a useful conceptual comparison is:

~~~text
Existing / text-only baseline
        vs
Prompt-only Qwen
        vs
Fine-tuned Qwen
~~~

---

# Question 17 — How would you define success for Little Content detection?

**Answer:**

I would not rely only on accuracy.

For the positive class Little Content:

$$
\mathrm{Precision}
=
\frac{TP}{TP+FP}
$$

and:

$$
\mathrm{Recall}
=
\frac{TP}{TP+FN}
$$

The project description reports improvement in Junk Metric precision, so precision is an important downstream measure.

I would still track recall, F1, confusion matrix, and slice-level performance because optimizing one aggregate metric can hide important failure modes.

---

# Question 18 — Why is data quality so important in fine-tuning?

**Answer:**

The model learns from the targets we provide.

If an observed label:

$$
\tilde{y}_i
$$

does not match the desired label:

$$
y_i
$$

then the gradient pushes the model toward the wrong behavior.

Other problems can also create misleading results:

~~~text
Noisy labels
→ noisy learning signal

Biased sampling
→ biased boundary

Near duplicates
→ inflated validation performance

Host/template leakage
→ fake generalization
~~~

For the Qwen project, annotation quality and leakage control are therefore core modeling problems, not just preprocessing details.

---

# Question 19 — When is fine-tuning justified for this project?

**Answer:**

My decision framework is:

~~~text
Pretrained Qwen already has useful VLM capability
        ↓
Little Content behavior is task-specific
        ↓
Labelled SERP screenshots are available
        ↓
Visual information matters
        ↓
Prompt-only / simpler baselines leave a persistent gap
        ↓
Fine-tuning is justified
~~~

So I would not say "we fine-tuned because fine-tuning is powerful."

I would say:

> We fine-tune because the base model already has the required underlying multimodal capability, while the remaining gap is a stable, task-specific screenshot-to-label behavior that can be learned from labelled examples.

---

# Qwen Project — 60-Second Interview Walkthrough After Part 1

A concise answer at this stage would be:

> The problem is to detect Little Content pages from Bing SERP screenshots using visual-layout and content-density information. I would start from a pretrained Qwen VLM rather than train from scratch because the pretrained model already has broad visual-language capability. The missing piece is the task-specific decision boundary for Little Content. Before fine-tuning, I would benchmark prompt-only Qwen and any existing simpler baseline to prove that a persistent gap exists. If fine-tuning is justified, the goal is to adapt the model to the screenshot-to-label mapping while preserving general multimodal capability and ensuring it generalizes to unseen hosts and layouts rather than memorizing templates. The exact SFT loss, LoRA configuration, dataset split, and training hyperparameters are design decisions that I would make in the next stages.

---

# What We Should Not Claim Yet

At the end of Part 1, the following are still intentionally unresolved:

- Exact Qwen model variant.
- Dataset size.
- Class balance.
- Train/validation/test split.
- Full fine-tuning versus PEFT.
- LoRA rank or target modules.
- Exact loss construction.
- Chat template.
- Optimizer.
- Learning rate.
- Batch size.
- Number of epochs.
- Thresholding and calibration.
- Deployment architecture.

These should not be invented in an interview before we have designed them in the later parts.

---

# Part 1 — Rapid Interview Memory Map

~~~text
PRETRAINING
Broad capability

        ↓

DOWNSTREAM GAP
Little Content policy is task-specific

        ↓

BASELINE FIRST
Prompt-only Qwen
+
simpler existing system

        ↓

FINE-TUNING
Adapt pretrained capability
toward task behavior

        ↓

DO NOT MEMORIZE SHORTCUTS
Generalize to unseen hosts / layouts

        ↓

DESIGN QUESTION FOR PART 2
How do supervised examples,
loss, and LoRA actually train the model?
~~~

# One-Line Summary

> **Pretraining gives Qwen broad visual-language capability; fine-tuning adapts that capability to the Little Content decision behavior, but only after simpler baselines show that adaptation is actually needed.**
