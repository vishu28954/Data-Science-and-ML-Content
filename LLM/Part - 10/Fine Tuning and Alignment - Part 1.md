# Part 10 — Fine-Tuning and Alignment

# Fine-Tuning and Alignment — Part 1

## 10.1 Pretraining vs Fine-Tuning

**Detailed Study Mode.** Interview-style material remains separate.

This chapter begins two things at the same time:

1. The formal Fine-Tuning and Alignment syllabus.
2. The theoretical design of the Qwen Little Content Detection project.

Our fixed learning pattern is:

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

# Project Ground Rules

Throughout Part 10, every project-specific statement belongs to one of three categories.

| Category | Meaning |
|---|---|
| **Resume Fact** | Explicitly stated in the current CV |
| **Theoretical Design Decision** | A design choice we make while learning |
| **Open Question** | A detail not established yet |

## Resume facts available at the start

The current resume states that:

- A Qwen Vision-Language Model was fine-tuned.
- The data consisted of labelled Bing SERP screenshots.
- The model detected Little Content pages using visual-layout and content-density signals.
- A multimodal inference pipeline evaluated roughly 50K URLs per day.
- Predictions fed into Bing's SERP Junk Metric.
- The resume reports roughly 10% improvement in Junk Metric precision.

## Details that are not established yet

The resume does not establish the exact:

- Qwen version or parameter count.
- Dataset size.
- Class balance.
- Train/validation/test split.
- Full fine-tuning versus LoRA configuration.
- Optimizer.
- Learning rate.
- Batch size.
- Number of epochs.
- Loss masking.
- Image resolution.
- Thresholding strategy.
- Deployment hardware.

We will derive these theoretically across Parts 1–8 rather than inventing them.

---

# 10.1 Pretraining vs Fine-Tuning

### Story Bridge 1 — Begin With the Task, Not the Training Technique

We already have a powerful pretrained Vision-Language Model. The project does not ask us to teach a model vision or language from zero.

It asks something much narrower:

> Given a Bing SERP screenshot, decide whether it should be classified as Little Content.

That means the first thing to understand is the exact behavior we want to learn.

## Question 1 — What problem are we actually trying to solve?

At the highest level, we want a model:

$$
f_\theta(x)\rightarrow y
$$

For the project:

$$
x=\text{SERP screenshot}
$$

and, in a simplified binary formulation:

$$
y\in
\{
\text{Little Content},
\text{Not Little Content}
\}
$$

The central problem is therefore not:

> Can Qwen understand an image?

The problem is:

> Can Qwen map visual and textual evidence in a SERP screenshot to our task-specific Little Content decision?

### Qwen Project Application

A theoretical training example could eventually look like:

~~~text
Input:
SERP screenshot
+
instruction asking for Little Content classification

Target:
Little Content
or
Not Little Content
~~~

The exact chat template is intentionally deferred to Part 3.

### Design Decision

We initially formulate the problem as a **multimodal supervised binary decision implemented through a generative VLM interface**.

---

### Story Bridge 2 — Qwen Already Understands Images, So What Did Pretraining Give Us?

A pretrained VLM may already recognize text blocks, buttons, whitespace, navigation, images, and page structure.

If those abilities already exist, we need to separate general capability from task-specific behavior.

## Question 2 — What does pretraining actually teach a model?

Pretraining exposes a model to broad data and optimizes a general learning objective.

For a causal language model, a simplified objective is:

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

The model learns to make the observed next token more probable given the previous context.

For a multimodal generative view, image information can be included conceptually as:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log
P_\theta
\left(
y_t
\mid
I,y_{<t}
\right)
$$

where:

- $I$ represents image-derived information.
- $y_t$ is the target text token.

This is a conceptual objective. The exact historical pretraining recipe depends on the specific Qwen model version and is not established by the resume.

### What pretraining gives us

A useful mental model is:

~~~text
Large broad dataset
        ↓
General visual features
        +
Language patterns
        +
Cross-modal associations
        +
Broad reasoning capability
        ↓
Reusable pretrained parameters
~~~

### Qwen Project Application

The pretrained model may already recognize:

- Sparse layouts.
- Dense layouts.
- Text regions.
- Empty space.
- Search-result structures.
- Navigation-heavy