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
- Navigation-heavy pages.

Those are useful ingredients for the downstream task.

### Design Decision

We treat pretraining as the source of **general reusable visual-language capability**.

---

### Story Bridge 3 — Recognizing a Page Is Not the Same as Knowing Our Label Policy

A model can describe a screenshot correctly and still classify it incorrectly.

For example, it may correctly say:

> The page contains a navigation bar, a short text block, and a large blank region.

But we still need a production decision.

That is where the gap between pretraining and fine-tuning appears.

## Question 3 — Why does pretraining not automatically solve the Little Content task?

Because the pretraining objective is not the same as the downstream objective.

The pretrained model may learn:

$$
h=f_{\theta_0}(x)
$$

where:

- $\theta_0$ is the pretrained parameter state.
- $h$ is a learned representation of the input.

That representation can contain useful signals such as:

~~~text
Text density
Layout sparsity
Visual hierarchy
Content blocks
Navigation structure
~~~

But our downstream problem still needs a mapping:

$$
h
\rightarrow
\text{Little Content decision}
$$

Pretraining may make that mapping easier to learn, but it does not define Bing's exact decision rule.

### Qwen Project Application

Consider two visually sparse screenshots.

~~~text
Page A:
Almost no meaningful primary content
→ Human label: Little Content

Page B:
Compact but genuinely useful answer block
→ Human label: Not Little Content
~~~

A generic VLM may describe both accurately yet fail to reproduce the task-specific label boundary.

### Design Decision

The main adaptation target is:

> **The mapping from SERP evidence to the Little Content decision boundary.**

---

### Story Bridge 4 — We Know What Is Missing, So Fine-Tuning Must Change Something

We do not want to discard everything learned during pretraining.

We want to start from the pretrained state and move it toward the downstream task.

## Question 4 — What is fine-tuning mathematically?

Let the pretrained parameters be:

$$
\theta_0
$$

and the downstream dataset be:

$$
\mathcal{D}_{\mathrm{task}}
=
\{
(x_i,y_i)
\}_{i=1}^{N}
$$

Fine-tuning starts from:

$$
\theta
\leftarrow
\theta_0
$$

and optimizes a downstream objective:

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

A gradient update can be written as:

$$
\theta_{k+1}
=
\theta_k
-
\eta
\nabla_\theta
\mathcal{L}_{\mathrm{task}}
$$

where:

- $\eta$ is the learning rate.
- $k$ is the optimization step.
- $\nabla_\theta\mathcal{L}$ is the gradient.

The key idea is:

> Fine-tuning begins from a useful pretrained solution instead of learning everything from zero.

### Qwen Project Application

For our simplified project:

$$
x_i=\text{SERP screenshot}_i
$$

and:

$$
y_i=\text{human Little Content label}_i
$$

The task examples provide gradients that make the desired output more likely for similar screenshots.

### Design Decision

The project follows:

~~~text
Pretrained Qwen VLM
        ↓
Labelled SERP screenshots
        ↓
Task-specific loss
        ↓
Gradient-based adaptation
        ↓
Adapted Qwen model
~~~

Exactly which parameters are trainable is deferred to Part 2.

---

### Story Bridge 5 — Fine-Tuning Starts From Knowledge; Training From Scratch Does Not

Now we can compare fine-tuning with the alternative of building a multimodal model from a random initialization.

## Question 5 — How is fine-tuning different from training from scratch?

### Training from scratch

We begin with parameters that do not contain useful learned representations:

$$
\theta_{\mathrm{init}}
\sim
\text{random initialization}
$$

The training data must teach both general representations and the downstream task.

### Fine-tuning

We begin from:

$$
\theta_0
=
\text{pretrained parameters}
$$

and adapt those parameters, or a small additional parameter set, toward the target task.

~~~text
TRAIN FROM SCRATCH
random parameters
    ↓
learn vision/language representations
    ↓
learn task

FINE-TUNE
pretrained representations
    ↓
adapt task behavior
~~~

### Qwen Project Application

A labelled SERP dataset should ideally teach:

> Which already-recognizable visual patterns correspond to Little Content?

It should not need to teach from zero:

- Basic vision.
- Language generation.
- Visual-text association.
- Layout understanding.
- General semantic understanding.

### Design Decision

We use **transfer learning from a pretrained VLM**, not training a multimodal model from scratch.

---

### Story Bridge 6 — If Knowledge Is Reused, What Exactly Is Being Transferred?

Saying that a pretrained model contains knowledge is still vague.

We need a more useful mental model for transfer learning.

## Question 6 — What does transfer learning mean here?

Suppose the pretrained network maps the input to a representation:

$$
h=f_{\theta_0}(x)
$$

If the representation already captures useful factors such as:

- Text regions.
- Visual density.
- Page structure.
- Semantic content.
- Empty regions.
- Navigation elements.

then the downstream task can reuse those factors.

Fine-tuning can then modify the model so those features become more useful for predicting:

$$
y=\text{Little Content label}
$$

A compact mental model is:

~~~text
Pretraining
→ learn useful general features

Fine-tuning
→ adapt how those features are used for the task
~~~

This is simplified because fine-tuning can also modify the representations themselves.

### Qwen Project Application

The desired transfer is:

~~~text
General VLM capability
        ↓
Understand screenshot structure
        ↓
Recognize content-density patterns
        ↓
Learn Little Content boundary
~~~

### Design Decision

Our design goal is:

> Preserve useful pretrained visual-language capability while adapting task behavior.

This will later connect directly to PEFT and catastrophic forgetting.

---

### Story Bridge 7 — Before Updating Weights, Try the Cheapest Baseline

A powerful pretrained model may already perform reasonably well if the task is explained clearly.

Therefore, fine-tuning should not be our automatic first move.

## Question 7 — How is prompting different from fine-tuning?

### Prompting

Prompting changes the input context while leaving the model parameters unchanged:

$$
P_{\theta_0}
\left(
y\mid x,\text{instruction}
\right)
$$

### Fine-tuning

Fine-tuning changes model parameters or a trainable adapter set:

$$
\theta_0
\rightarrow
\theta^*
$$

| Prompting | Fine-tuning |
|---|---|
| No parameter update | Parameters or adapters updated |
| Fast to test | Requires training |
| Easy to iterate | More expensive to iterate |
| Behavior depends heavily on prompt | Task behavior can be internalized |
| Good baseline | Useful for persistent adaptation |

### Qwen Project Application

A prompt-only baseline could theoretically ask:

~~~text
Inspect this SERP screenshot.

Classify it as:
1. Little Content
2. Not Little Content
~~~

If this baseline already satisfies the business requirement, fine-tuning may not be necessary.

### Design Decision

Before training, our theoretical project should include a **prompt-only Qwen baseline**.

---

### Story Bridge 8 — Sometimes the Problem Is Missing Information, Not Missing Behavior

Fine-tuning is often used when the real problem is that the model lacks fresh or proprietary information.

That leads to an important distinction.

## Question 8 — When should we think about RAG instead of fine-tuning?

A useful high-level distinction is:

~~~text
Fine-tuning
→ change model behavior

RAG
→ provide external information at inference time
~~~

Fine-tuning is useful for persistent task behavior, style, classification policy, or domain-specific response patterns.

RAG is useful when the main requirement is access to changing, traceable, or large external knowledge.

They can also be combined.

### Qwen Project Application

Little Content detection is primarily a **screenshot-to-label behavior problem**.

The model must inspect the screenshot and map the observed evidence to the desired label.

Therefore RAG is not the primary adaptation mechanism for the core task.

### Design Decision

Primary adaptation mechanism:

$$
\text{Fine-tuning}
$$

rather than retrieval.

---

### Story Bridge 9 — A Fine-Tuning Task Must Specify What the Model Should Actually Learn

Saying "fine-tune Qwen on screenshots" is still not a complete problem definition.

We need to define the target behavior and the evidence it should rely on.

## Question 9 — What exactly should the fine-tuned model learn?

At a simplified level, we want:

$$
P_\theta(y\mid x)
$$

with:

$$
y
\in
\{
\text{Little Content},
\text{Not Little Content}
\}
$$

We want:

$$
P_\theta
\left(
y_{\mathrm{correct}}
\mid x
\right)
$$

to be high.

But the model should learn the **concept**, not incidental correlations.

Useful evidence may include:

- Meaningful visible content.
- Content density.
- Layout structure.
- Empty regions.
- Primary versus secondary content.
- Navigation dominance.

Undesirable shortcuts could include:

- Specific host identity.
- Template identity.
- Screenshot artifacts.
- Watermarks.
- Collection-specific metadata.

### Qwen Project Application

The desired behavior is:

~~~text
Learn:
"What visual/content pattern constitutes Little Content?"

Not:
"Which screenshots or hosts appeared in training?"
~~~

### Design Decision

The project must be designed for **concept generalization**, not screenshot memorization.

---

### Story Bridge 10 — A Binary Label Implies a Decision Boundary

Once we know what concept should be learned, it is useful to visualize fine-tuning as moving the task decision boundary.

## Question 10 — How can fine-tuning be understood as learning a decision boundary?

Suppose the model produces an internal representation:

$$
h=f_\theta(x)
$$

and a task-relevant mechanism produces a score:

$$
s=g(h)
$$

A simplified binary decision can be written as:

$$
\hat{y}
=
\begin{cases}
1 & s\ge\tau\\
0 & s<\tau
\end{cases}
$$

where $1$ represents Little Content and $\tau$ is a decision threshold.

A generative VLM may not literally contain one scalar classification head. It can instead generate class tokens. The decision-boundary view is still useful conceptually.

### Qwen Project Application

Two screenshots can both look sparse yet belong to different classes.

Therefore the model should not learn:

$$
\text{Little Content}
=
\text{large blank area}
$$

It may need several interacting visual and semantic signals.

### Design Decision

Later dataset design should deliberately include:

- Clear positives.
- Clear negatives.
- Hard negatives.
- Ambiguous boundary examples.

---

### Story Bridge 11 — Good Training Accuracy Is Useless If the Model Memorizes the Dataset

Fine-tuning should improve performance on screenshots the model has never seen.

That is the real purpose of training.

## Question 11 — Is the goal memorization or generalization?

The goal is generalization.

Let the training data be:

$$
\mathcal{D}_{\mathrm{train}}
$$

but the real target is expected performance on the deployment dis