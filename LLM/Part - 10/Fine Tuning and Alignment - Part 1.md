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

but the real target is expected performance on the deployment distribution:

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{target}}}
\left[
\mathcal{L}
\left(
f_\theta(x),y
\right)
\right]
$$

Training examples are only a sample from the broader target distribution.

### Qwen Project Application

A model that performs well on known website templates but fails on previously unseen SERP structures is not a successful solution.

This will later make host/template leakage a major concern.

### Design Decision

Generalization target:

> Unseen SERP screenshots, including unseen host and layout variations.

---

### Story Bridge 12 — The Downstream Dataset Is Narrower Than the Pretraining World

The pretrained model saw broad data.

Our task data comes from one narrower environment: labelled SERP screenshots with a specific operational label.

## Question 12 — What is domain shift in fine-tuning?

Let:

$$
p_{\mathrm{pre}}(x)
$$

represent the broad pretraining distribution.

Let:

$$
p_{\mathrm{task}}(x,y)
$$

represent the downstream distribution.

In general:

$$
p_{\mathrm{pre}}
\neq
p_{\mathrm{task}}
$$

The downstream data can emphasize patterns that were rare or unimportant during pretraining.

### Qwen Project Application

Our project contains both:

~~~text
DOMAIN ADAPTATION
General visual-language data
→ SERP screenshots

TASK ADAPTATION
General multimodal understanding
→ Little Content classification
~~~

### Design Decision

We are theoretically performing:

$$
\text{Domain adaptation}
+
\text{Task adaptation}
$$

The relative importance of each will be explored later.

---

### Story Bridge 13 — Specialization Can Help, but It Can Also Damage Useful Capabilities

Fine-tuning changes the pretrained model.

If updates are too aggressive or the downstream data is too narrow, useful prior capability can deteriorate.

## Question 13 — Can fine-tuning damage pretrained capabilities?

Yes.

Conceptually:

$$
\theta_0
\rightarrow
\theta^*
$$

should improve the downstream task without unnecessarily damaging useful behavior encoded in $\theta_0$.

~~~text
Too little adaptation
→ task remains poorly learned

Too much narrow adaptation
→ over-specialization / forgetting
~~~

This connects to catastrophic forgetting, which we will study formally in Part 7.

### Qwen Project Application

We want Qwen to become better at Little Content classification while retaining the visual-language understanding that made the pretrained model useful.

### Design Decision

Training should balance:

$$
\text{Task adaptation}
\quad\text{and}\quad
\text{preservation of pretrained capability}
$$

---

### Story Bridge 14 — If Preserving the Base Model Matters, Maybe We Do Not Need to Update Every Parameter

So far, fine-tuning has sounded like every model weight must change.

Large models give us another option: adapt a smaller trainable parameter set while keeping most pretrained weights frozen.

## Question 14 — Does fine-tuning necessarily mean updating the whole model?

No.

### Full fine-tuning

Most or all model parameters are trainable:

$$
\theta_0
\rightarrow
\theta^*
$$

### Parameter-Efficient Fine-Tuning

Most pretrained parameters remain fixed:

$$
\theta_0
=
\text{frozen}
$$

while a smaller parameter set:

$$
\phi
=
\text{trainable}
$$

is optimized.

The adapted model can be written conceptually as:

$$
f(x;\theta_0,\phi)
$$

LoRA is one important PEFT method.

We will study its mathematics in Part 2.

### Qwen Project Application

The resume lists multimodal LoRA/PEFT among the skill set, but the exact historical adapter configuration is not established.

### Design Decision

The following remains an open question until Part 2:

~~~text
Full fine-tuning
vs
LoRA / PEFT
~~~

---

### Story Bridge 15 — A Fine-Tuned Model Is Meaningless Without a Baseline to Beat

Suppose we fine-tune Qwen and obtain a strong-looking metric.

That number alone does not tell us whether fine-tuning was actually useful.

If prompt-only Qwen already performs just as well, the training effort may not have added value.

## Question 15 — What baselines should exist before we fine-tune?

A strong experiment should compare the fine-tuned model against simpler alternatives.

### Baseline A — Prompt-only pretrained Qwen

~~~text
Pretrained Qwen
+
carefully written Little Content instruction
~~~

This tells us how much task performance is already available without changing any parameters.

### Baseline B — Existing text-only or heuristic system

If an existing production system already performs the task, it provides a practical reference point.

### Baseline C — Simpler supervised model

If suitable extracted features are available, a simpler classifier can tell us whether the complexity of a VLM is justified.

The principle is:

> **Fine-tuning should demonstrate incremental value over a meaningful baseline.**

### Qwen Project Application

The current CV states that the VLM extended quality classification beyond text-only features.

That makes the following conceptual comparison useful:

~~~text
Existing / text-only baseline
        ↓
Prompt-only Qwen VLM
        ↓
Fine-tuned Qwen VLM
~~~

The exact historical baseline implementation is not established by the resume.

### Design Decision

Our theoretical experiment plan will include at least:

1. Prompt-only Qwen.
2. Fine-tuned Qwen.
3. An appropriate existing or simpler baseline when available.

---

### Story Bridge 16 — We Need to Define Success Before We Start Training

A model can improve one metric while becoming worse on another.

For a classification task, false positives and false negatives can have very different business costs.

So the evaluation goal should be decided before fine-tuning.

## Question 16 — How should we define success for Little Content detection?

Let the positive class be Little Content.

Then:

$$
\mathrm{Precision}
=
\frac{TP}{TP+FP}
$$

Precision asks:

> Of the pages predicted as Little Content, how many truly were Little Content?

Recall is:

$$
\mathrm{Recall}
=
\frac{TP}{TP+FN}
$$

Recall asks:

> Of all true Little Content pages, how many did we detect?

These metrics can trade off against each other.

### Why accuracy can be misleading

Suppose only 5% of pages are Little Content.

A model that predicts Not Little Content for every page obtains:

$$
95\%
$$

accuracy while detecting no positive examples.

Therefore, class balance and business cost matter.

### Qwen Project Application

The current resume reports an improvement in Junk Metric precision, so precision is clearly one important downstream measure.

But a complete evaluation should also examine:

- Recall.
- F1.
- Confusion matrix.
- Performance across important slices.
- Error types.

The full evaluation framework belongs to Part 8.

### Design Decision

The Qwen project will use a **precision-sensitive evaluation**, while still tracking recall and broader error behavior.

---

### Story Bridge 17 — Fine-Tuning Can Only Learn the Signal We Put Into the Dataset

We now understand the model side of fine-tuning.

But the labelled examples determine what gradients the model receives.

If those labels are poor, the model can faithfully learn the wrong behavior.

## Question 17 — Why is data quality often more important than the fine-tuning algorithm?

Suppose the observed training label is:

$$
\tilde{y}_i
$$

while the desired true label is:

$$
y_i
$$

If:

$$
\tilde{y}_i
\neq
y_i
$$

the optimization process is pushed toward the wrong target.

More generally:

~~~text
Noisy labels
→ noisy learning signal

Biased sampling
→ biased decision boundary

Near duplicates
→ inflated validation performance

Host/template leakage
→ apparent generalization without true generalization
~~~

A sophisticated optimizer cannot rescue a fundamentally invalid target definition.

### Qwen Project Application

The labelled SERP screenshots do more than supply training examples.

They define the operational concept of Little Content.

Later we therefore need to answer:

- How were labels created?
- How consistent are annotators?
- How are ambiguous cases handled?
- Are hard negatives present?
- Are near-duplicate hosts or templates separated across splits?

### Design Decision

Dataset design is a first-class component of the project, not preprocessing trivia.

---

### Story Bridge 18 — We Can Finally State When Fine-Tuning Is Justified

We have now compared:

- Pretraining.
- Training from scratch.
- Prompting.
- RAG.
- Fine-tuning.

We have also established the need for baselines, meaningful evaluation, and trustworthy labelled data.

That gives us enough information to build a general decision rule.

## Question 18 — When should we fine-tune a pretrained model?

Fine-tuning is especially reasonable when:

1. The pretrained model already has useful underlying capability.
2. We need a persistent downstream behavior that prompting does not deliver reliably enough.
3. We have representative, high-quality task examples.
4. The desired behavior is sufficiently stable.
5. The expected gain justifies training and maintenance complexity.
6. We can evaluate the adapted model on genuinely unseen data.

Fine-tuning is less attractive when:

- The real problem is missing fresh knowledge.
- Prompting already solves the task.
- Training labels are too weak or inconsistent.
- Requirements change extremely quickly.
- A simpler model solves the problem reliably.
- Training and serving cost outweigh the benefit.

### Qwen Project Application

The theoretical justification becomes:

~~~text
General Qwen VLM capability already exists
        ↓
Little Content behavior is task-specific
        ↓
Labelled SERP screenshots exist
        ↓
Visual information is useful
        ↓
Prompt-only baseline leaves a persistent gap
        ↓
Fine-tuning becomes justified
~~~

### Design Decision

Fine-tuning is justified as a task-adaptation mechanism **only after the baseline gap is demonstrated**.

---

### Story Bridge 19 — Part 1 Should End With a Project State, Not a Collection of Definitions

Part 1 has answered the question:

> Why would we fine-tune Qwen at all?

Before moving to Supervised Fine-Tuning, we should be able to reconstruct the entire reasoning chain and clearly separate what is known from what still needs to be designed.

## Question 19 — What is our current theoretical design for the Qwen project?

At the end of Part 1:

~~~text
Business problem
        ↓
Detect Little Content pages

Why a VLM?
        ↓
Visual layout and content-density evidence matter

Why pretrained Qwen?
        ↓
Reuse broad multimodal capability

Why not train from scratch?
        ↓
Reuse existing representations

What comes before fine-tuning?
        ↓
Prompt-only and simpler baselines

Why fine-tune?
        ↓
Learn persistent task-specific behavior

What should the model learn?
        ↓
A screenshot-to-label decision boundary

What should it NOT learn?
        ↓
Host/template shortcuts

What matters most?
        ↓
Generalization to unseen pages

What is still unknown?
        ↓
Exact Qwen version
Dataset construction
Full FT vs PEFT
LoRA configuration
Loss construction
Chat format
Hyperparameters
Evaluation protocol
Deployment design
~~~

This is the project state that Part 2 will inherit.

---

# Qwen Project Build Record — After Part 1

| Design element | Current state | Type |
|---|---|---|
| Business problem | Little Content detection | Resume Fact |
| Model family | Qwen Vision-Language Model | Resume Fact |
| Training data modality | Labelled Bing SERP screenshots | Resume Fact |
| Core evidence | Visual-layout and content-density signals | Resume Fact |
| Production scale | About 50K URLs/day | Resume Fact |
| Downstream integration | Bing SERP Junk Metric | Resume Fact |
| Reported impact | About 10% improvement in Junk Metric precision | Resume Fact |
| Theoretical task formulation | Multimodal supervised binary decision | Design Decision |
| Why pretrained Qwen? | Reuse broad visual-language capability | Design Decision |
| Why fine-tuning? | Learn persistent task-specific decision behavior | Design Decision |
| Prompt-only Qwen | Required theoretical baseline | Design Decision |
| RAG | Not the primary mechanism for the core task | Design Decision |
| Generalization target | Unseen SERP layouts, hosts, and templates | Design Decision |
| Evaluation emphasis | Precision-sensitive while tracking recall and error slices | Design Decision |
| Dataset quality | First-class modeling concern | Design Decision |
| Exact Qwen version | Not established | Open Question |
| Dataset size | Not established | Open Question |
| Class balance | Not established | Open Question |
| Train/validation/test split | Not established | Open Question |
| Full FT vs PEFT | Deferred to Part 2 | Open Question |
| LoRA configuration | Deferred to Part 2 | Open Question |
| Exact loss construction | Deferred to Part 2 | Open Question |
| Chat template | Deferred to Part 3 | Open Question |
| Hyperparameters | Deferred to Part 6 | Open Question |
| Final evaluation protocol | Deferred to Part 8 | Open Question |
| Deployment architecture | Deferred to Part 8 | Open Question |

---

# Part 1 — Key Equations

## Pretraining-style autoregressive objective

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

## Multimodal conditional-generation view

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log
P_\theta
\left(
y_t\mid I,y_{<t}
\right)
$$

## Fine-tuning objective

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

## Gradient update

$$
\theta_{k+1}
=
\theta_k
-
\eta
\nabla_\theta
\mathcal{L}_{\mathrm{task}}
$$

## Reusable representation

$$
h=f_{\theta_0}(x)
$$

## Desired task behavior

$$
P_\theta(y\mid x)
$$

## Simplified binary decision

$$
\hat{y}
=
\begin{cases}
1 & s\ge\tau\\
0 & s<\tau
\end{cases}
$$

## Target-distribution generalization objective

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{target}}}
\left[
\mathcal{L}
\left(
f_\theta(x),y
\right)
\right]
$$

## Precision

$$
\mathrm{Precision}
=
\frac{TP}{TP+FP}
$$

## Recall

$$
\mathrm{Recall}
=
\frac{TP}{TP+FN}
$$

---

# Part 1 — Final Mental Model

~~~text
PRETRAINING
Broad multimodal data
        ↓
General visual-language capability
        ↓
Pretrained parameters θ0

DOWNSTREAM PROBLEM
SERP screenshot
        ↓
Need Little Content decision
        ↓
General capability alone is insufficient

BASELINES
Prompt-only Qwen
+
simpler existing system
        ↓
Measure the task gap

FINE-TUNING
Labelled SERP examples
        ↓
Task-specific objective
        ↓
Gradient-based adaptation
        ↓
Adapted model

GOAL
Preserve useful pretrained knowledge
+
learn Little Content behavior
+
generalize to unseen SERP layouts
~~~

The central distinction is:

> **Pretraining creates broad reusable capability. Fine-tuning adapts that capability toward a narrower target behavior.**

---

# Bridge to Part 2

Part 1 answered:

> **Why should we fine-tune Qwen?**

Part 2 will answer:

> **How do labelled examples actually train it?**

We will study **10.2 Supervised Fine-Tuning**, including:

- What one SFT example contains.
- Input tokens versus target tokens.
- Cross-entropy and next-token loss.
- Which tokens contribute to the loss.
- Full fine-tuning versus PEFT.
- LoRA intuition.
- LoRA mathematics.
- Which parameters remain frozen.
- Which parameters receive gradients.
- How parameter-efficient training changes memory and training cost.

This is where the Qwen project moves from high-level justification into an actual training formulation.
