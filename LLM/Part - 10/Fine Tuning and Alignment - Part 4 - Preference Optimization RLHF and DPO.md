# Part 10 — Fine-Tuning and Alignment

# Part 4 — Preference Optimization, Reward Models, RLHF and DPO (Questions 1–15)

## 10.5 Learning from Preferences and Human Feedback

**Detailed Study Mode.** The original chapter-writing method is active **for preparing this content**; interview practice remains separate.

**Previous chapters:** [Part 2 — SFT Foundations](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202.md), [Part 2B — PEFT and LoRA](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20PEFT%20and%20LoRA.md), and [Part 3 — Instruction Tuning and Chat Fine-Tuning](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%203%20-%20Instruction%20Tuning%20and%20Chat%20Fine%20Tuning.md).

Part 3 answered: **How should a request and its ideal answer be represented so supervised learning teaches the correct behavior?**

Part 4 asks: **What if there is not one unambiguously correct target answer, but people can judge which of two responses is better?**

The fixed learning structure remains:

~~~text
Visible Story Bridge
       ↓
Question naturally motivated by the bridge
       ↓
Simple intuition → detailed mechanism → mathematical example
       ↓
Qwen Little Content Project Application
       ↓
Design Decision + Project Build Record Update
       ↓
Next visible Story Bridge
~~~

**Important project distinction:** The user's described Qwen VLM project concerns labelled Bing SERP screenshots, Little Content detection and a processing pipeline around 50K URLs/day. The exact checkpoint, training recipe, preference dataset, reward model, policy optimization runs and resulting measurements have **not** been established as implemented facts. In this chapter, **Resume Fact** means a claim in the project description, **Design Decision** means a proposed theoretical choice, and **Open Question** means a matter requiring evidence or experimentation. In particular, we will **not assume RLHF or DPO is necessary for binary classification**.

## Part 4 syllabus

| Question | Topic | Expected understanding |
|---|---|---|
| 1 | Why preference optimization exists | What supervised demonstrations cannot easily express |
| 2 | SFT versus preference learning | Supervision type, objective, and appropriate use |
| 3 | Preference-pair datasets | Prompt, chosen/rejected response, label quality and bias |
| 4 | Reward models | How a model scores responses from pairwise comparisons |
| 5 | Bradley–Terry preference loss | Derive and calculate pairwise likelihood |
| 6 | RLHF pipeline | Reference, reward model, policy and optimization loop |
| 7 | KL regularization | Why policies must not chase rewards without restraint |
| 8 | PPO-based RLHF | Actions, advantages, clipping and value models |
| 9 | Reward hacking and RLHF failures | Proxy optimization, overoptimization and diagnostics |
| 10 | DPO motivation and derivation | How preference optimization can avoid explicit reward-model RL |
| 11 | DPO loss numerically | Policy/reference log ratios, beta and gradients |
| 12 | DPO versus PPO-RLHF | Complexity, efficiency, limitations and when to choose |
| 13 | Practical preference-training design | Response formatting, length, offline data and multimodal challenges |
| 14 | Evaluating aligned models | Win rates, judges, safety, task metrics and held-out comparisons |
| 15 | Complete Qwen decision and design | When SFT is enough; when and how preference learning might help |

---

### Story Bridge 1 — SFT Teaches a Correct Answer, but What If There Are Several Reasonable Answers?

In Part 3, we showed Qwen the correct response to a screenshot and instruction. That is straightforward when the desired output is exactly one canonical class label. But consider an open-ended assistant asked to explain a medical paragraph: multiple answers may be factually acceptable, yet one can be more accurate, clearer, safer, or more useful.

A single supervised demonstration says **"imitate this answer"**. It may not explicitly say **"this answer is preferable to another plausible answer"**.

This creates our first new question.

## Question 1 — Why Do We Need Preference Optimization?

### Simple intuition

**Preference optimization teaches a model which of two responses people prefer for the same request.** Rather than only seeing one example of the target answer, the model can learn from comparisons.

Imagine this prompt:

~~~text
Prompt:
Explain why a sparse web page is not necessarily low-quality.

Response A:
Any mostly white page is low-quality.

Response B:
A page can look sparse but still contain one complete, useful answer.
Check the usefulness of the main content rather than counting blank space.

Human judgment:
Response B is better.
~~~

A preference method uses that relative judgment to increase the model's tendency to produce behavior like B rather than A.

### What exactly is different from SFT?

SFT learns a probability distribution over target tokens by increasing the likelihood of demonstrated answers. Preference learning instead uses the relative ordering of two or more candidate responses under a chosen preference criterion.

This is useful when **output quality is hard to specify as one exact target string**, but people can still choose which answer is better.

### Mathematical framing

Let $x$ be the prompt, and let $y_w$ and $y_l$ be preferred ("winner") and rejected ("loser") responses. We want the model to favor the preferred response according to a defined preference objective.

Conceptually:

$$
y_w \succ_x y_l
$$

This means "for prompt $x$, response $y_w$ is preferred to $y_l$." It does **not** mean that the chosen response is guaranteed correct or safe.

### When this is not needed

If the output is one of two well-defined, trustworthy labels, ordinary supervised classification may already express the desired behavior. Pairwise preferences are not automatically better than ground-truth class labels.

### Qwen Project Application

For our proposed Little Content detector, an annotated screenshot already has a class label. Preference learning might become relevant for **open-ended explanations**, **policy disagreements**, or **ranking borderline decisions**, but adding RLHF to a straightforward binary classifier would need clear evidence of incremental value.

### Design Decision — Q1

Treat preference learning as **an optional, separately justified stage**; do not replace canonical-label SFT merely because RLHF or DPO is fashionable.

### Project Build Record Update — Q1

| Element | State | Type |
|---|---|---|
| Primary class supervision | Existing screenshot → canonical label SFT | Design Decision |
| Preference learning | Optional, not yet justified | Open Question |
| Relative comparison target | Preferred versus rejected response for the same prompt | Study Concept |

---

### Story Bridge 2 — If Both Approaches Use Human Data, Why Can't We Treat Them as the Same Training Task?

An instruction dataset gives us a response to imitate. A preference dataset gives us **two responses and a choice**. The form of the annotation changes the learning problem, even if both ultimately update the same language model.

## Question 2 — How Does SFT Differ from Preference Learning?

### Simple explanation

**SFT asks:** "What answer should the model produce?"

**Preference learning asks:** "Between these possible answers, which one is better?"

| Property | SFT | Pairwise preference learning |
|---|---|---|
| Typical training record | Prompt + desired response | Prompt + chosen response + rejected response |
| Supervision | Direct demonstration | Relative comparison |
| Core goal | Increase likelihood of target response | Shift behavior toward chosen responses relative to rejected ones |
| Common methods | Teacher-forced NLL / cross-entropy | Reward-model training, RLHF, DPO |
| Example use | Known image class label | Helpfulness, tone, explanation quality or nuanced behavior |
| Key limitation | One reference does not specify all acceptable answers | Preference judgments can be inconsistent, biased or incomplete |

### Mathematical comparison

SFT objective for demonstrations $(x,y)$:

$$
\mathcal L_{\mathrm{SFT}}
=
-\mathbb E_{(x,y)\sim\mathcal D_{\mathrm{SFT}}}
\log\pi_\theta(y\mid x)
$$

Here $\pi_\theta$ denotes the response-generating policy (our generative model).

Pairwise preference data:

$$
\mathcal D_{\mathrm{pref}}
=
\{(x_i,y_{w,i},y_{l,i})\}_{i=1}^{N}
$$

A chosen response is not necessarily unique. A reward model or DPO objective can transform relative judgments into a trainable signal.

### Why many pipelines do SFT first

A pretrained base model may not yet produce good instruction-following responses. SFT is often used to teach basic assistant behavior. Once responses are reasonably coherent, preference comparisons can refine the behavior among plausible outputs.

This is **a common workflow, not a mathematical requirement**. Different training regimes exist.

### A subtle example

Suppose two acceptable explanations for the same screenshot say:

~~~text
A: "Little Content: the main content is absent."
B: "Little Content. The page contains only a login panel and no
    meaningful primary answer."
~~~

If the task demands exactly one class string, A and B both violate that output contract because they add explanations. A preference annotator might nevertheless prefer B for helpfulness.

Therefore a preference dataset can move the model in the **wrong direction for a strict production API** if the preference criterion does not match the real task.

### Qwen Project Application

For the Little Content pipeline, business metrics such as false-positive rate, precision and recall matter more than conversational eloquence. Do not optimize a subjective "helpfulness" preference if the service must output only one label.

### Design Decision — Q2

Preserve the supervised binary-label objective as the primary learning task. Preference methods should only be proposed against a specific deficiency and evaluated on the actual product requirement.

### Project Build Record Update — Q2

| Element | State | Type |
|---|---|---|
| SFT versus preferences | Distinct annotation formats and objectives | Study Concept |
| Sequence of stages | SFT-first is reasonable, preference stage optional | Design Decision |
| Product metric | Classification quality, not general answer likability | Design Decision |

---

### Story Bridge 3 — Before Learning from a Preference, We Need a Trustworthy Comparison

If the annotator compares two responses to different prompts, the ranking can reflect prompt difficulty rather than response quality. If they only compare almost-identical responses, the label may be arbitrary. Preference data quality is itself a modeling challenge.

## Question 3 — How Do We Construct Preference-Pair Datasets?

### The basic record

~~~json
{
  "prompt_id": "example_042",
  "prompt": "Explain why sparse pages can still be useful.",
  "chosen": "A sparse page may contain a complete answer...",
  "rejected": "All sparse pages are low quality.",
  "preference_criterion": "correctness and usefulness"
}
~~~

The two responses must address the **same prompt and task context**. For VLM tasks, they must also refer to the same attached image or multimodal input.

### How do we obtain pairs?

A model can generate multiple candidate responses per prompt; humans or a carefully validated judging process compare them. Alternatively, responses may come from different policies or a curated response pool.

The pipeline is:

~~~text
Sample input x
     ↓
Generate / collect candidate responses
     ↓
Compare candidates using a written policy
     ↓
Record chosen y_w and rejected y_l
     ↓
Audit consistency, bias and train/evaluation leakage
~~~

### Critical annotation rules

1. **Hold the input constant.** Compare responses to the same request and image.
2. **State the criterion.** Factuality, helpfulness, safety, concision and formatting may conflict.
3. **Allow uncertainty or ties when needed.** A forced preference on essentially equal responses introduces noise.
4. **Avoid superficial shortcuts.** Annotators may prefer longer, more confident or more polished answers even when wrong.
5. **Represent meaningful edge cases.** Include examples that expose the specific weaknesses we want to fix.
6. **Audit duplicates and leakage.** Repeated prompts, near-identical answers, or overlap with evaluation sets can distort measured improvements.

### Worked example: preference noise

Suppose 10 reviewers compare the same pair and 6 choose A, while 4 choose B.

We might estimate an observed preference fraction:

$$
\hat p(A\succ B)=\frac{6}{10}=0.6
$$

This is not strong consensus. Simply recording A as an unambiguous winner discards disagreement. We may need more judgments or model the uncertainty, depending on the problem.

### What about ties?

Standard pairwise DPO data normally designates one chosen and one rejected response. Ties can be omitted, adjudicated, or handled by methods explicitly designed for softer/tied preferences. Pretending a tie is a confident preference is not a neutral decision.

### Qwen Project Application

If we ever train Qwen to **explain borderline screenshot decisions**, a useful comparison might ask which explanation correctly references evidence in the image without inventing page content.

It would not be sensible to mark "Little Content" as *preferred* to "Not Little Content" simply because one class is more frequent; that is a ground-truth classification question, not subjective preference.

### Design Decision — Q3

No preference dataset is part of the initial Qwen classifier plan. If an explanation or policy-ranking feature emerges, define comparison criteria, candidate generation and disagreement handling explicitly.

### Project Build Record Update — Q3

| Element | State | Type |
|---|---|---|
| Preference schema | Prompt/image plus chosen and rejected candidate | Study Concept |
| Preference annotation policy | Not defined for current binary task | Open Question |
| Human agreement and quality audit | Required if pairwise data is ever collected | Design Decision |

---

### Story Bridge 4 — A Pairwise Judgment Is a Label, Not Yet a Numerical Reward

RLHF commonly needs a number that rates a candidate response. But annotators often find it easier to compare two answers than to assign a consistent absolute score from 1 to 10.

A reward model learns to turn comparisons into relative numerical scores.

## Question 4 — What Is a Reward Model, and What Does It Learn?

### Simple intuition

A **reward model** is a trainable scoring function that estimates how desirable a response is for its prompt under the preference data and criteria.

It typically takes a prompt and a complete candidate response, then outputs a scalar:

$$
r_\phi(x,y)\in\mathbb R
$$

The number is useful for *ranking responses to the same prompt*.

### How does it differ from the generative model?

| Generative policy | Reward model |
|---|---|
| Generates or scores text-token sequences | Produces a scalar score for a prompt-response pair |
| Trained by SFT or policy/preference optimization | Often trained from pairwise human preferences |
| May have a large vocabulary output head | Usually has a scalar reward head or equivalent scoring output |

An architecture can share a pretrained backbone with different heads, but the roles of the two models are different.

### A first scoring example

Suppose a reward model gives:

~~~text
Response A: r(x,A) = 1.2
Response B: r(x,B) = 2.0
~~~

The model ranks B above A. The difference is:

$$
\Delta r=r_\phi(x,B)-r_\phi(x,A)=0.8
$$

These numbers are **not literal measured human happiness**, and their absolute zero point is usually not meaningful. What matters in pairwise training is the relative score.

### Why we cannot automatically trust a reward model

A reward model learns from a finite dataset. It may:

- Prefer stylistically polished but false answers.
- Overvalue length or confident phrasing.
- Miss rare but critical failure modes.
- Misjudge inputs outside its training distribution.
- Reward a model that finds ways to exploit its weaknesses.

Therefore a reward model is a **proxy for the real preference objective**.

### Qwen Project Application

A reward model could theoretically score two natural-language explanations for the same SERP screenshot. That is a different function from the established binary classification task.

For strict label prediction, it is often more direct to train on correct class labels and evaluate the actual errors.

### Design Decision — Q4

Do not add an explicit reward model to the initial screenshot classifier. Understand reward modeling as a tool for tasks where learned subjective or multi-criteria ranking is genuinely useful.

### Project Build Record Update — Q4

| Element | State | Type |
|---|---|---|
| Reward model | Not selected | Open Question |
| Output of reward model | Scalar response score | Study Concept |
| Reward validity | Requires independent checks | Design Decision |

---

### Story Bridge 5 — What Mathematical Loss Lets a Reward Model Learn from "B Is Better Than A"?

We have a scalar reward score and a chosen/rejected response pair. We need a probability model connecting the difference in reward scores to the observed preference.

A common model is the Bradley–Terry logistic preference model.

## Question 5 — What Is the Bradley–Terry Preference Model and Reward Loss?

### Intuition

If response B has a reward score much higher than A, the model should assign high probability to a reviewer choosing B.

If their reward scores are equal, a simple symmetric model predicts a 50–50 preference.

### Derivation in small steps

Let $y_w$ be the preferred response and $y_l$ the rejected one.

Set:

$$
r_w=r_\phi(x,y_w)
$$

$$
r_l=r_\phi(x,y_l)
$$

Model the preferred-outcome probability as:

$$
P_\phi(y_w\succ y_l\mid x)
=
\frac{e^{r_w}}{e^{r_w}+e^{r_l}}
$$

Divide numerator and denominator by $e^{r_w}$:

$$
P_\phi(y_w\succ y_l\mid x)
=
\frac{1}{1+e^{-(r_w-r_l)}}
=
\sigma(r_w-r_l)
$$

where $\sigma(z)=1/(1+e^{-z})$ is the sigmoid function.

Since the observed label says "$y_w$ wins", negative log-likelihood gives:

$$
\mathcal L_{\mathrm{RM}}
=
-\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
$$

Across a dataset, we average or otherwise aggregate this loss.

### Numerical worked example

Suppose the preferred response receives reward $2.0$ and the rejected response receives $1.2$:

$$
\Delta r=2.0-1.2=0.8
$$

Then:

$$
P(y_w\succ y_l\mid x)=\sigma(0.8)\approx0.690
$$

and:

$$
\mathcal L_{\mathrm{RM}}
=-\log(0.690)\approx0.371
$$

If the preferred score rises to $4.0$ while the rejected score remains $1.2$:

$$
\sigma(2.8)\approx0.943,\qquad
-\log(0.943)\approx0.059
$$

The loss falls because the model more strongly predicts the observed preference.

### What the model learns—and what it doesn't

This objective encourages a correct **ordering**, not a reliable absolute human-quality scale.

Adding the same constant to both reward scores changes neither the difference nor the preference probability.

**Check:**

$$
(r_w+c)-(r_l+c)=r_w-r_l
$$

Thus the absolute reward offset is not identified by pairwise comparisons under this model.

### Qwen Project Application

If we had pairs of candidate visual explanations, the reward model might learn which explanation better reflects the screenshot. It could still fail if reviewers prefer plausible narratives over accurate visual grounding.

### Design Decision — Q5

Use the Bradley–Terry objective as our mathematical reference for reward models and DPO's preference formulation, **not** as the loss for the initial binary class labeler.

### Project Build Record Update — Q5

| Element | State | Type |
|---|---|---|
| Bradley–Terry preference likelihood | Sigmoid of reward difference | Study Concept |
| Initial training loss for Qwen labels | Assistant-target cross-entropy remains primary | Existing Design Decision |
| Learned reward calibration | Not assumed | Open Question |

---

### Story Bridge 6 — We Can Score Responses, but How Does the Generative Model Learn to Earn Higher Scores?

A reward model rates completed responses. The policy model is the one that actually generates new responses. We need a learning process connecting the reward scores back to changes in the policy.

That is the motivation for reinforcement learning from human feedback.

## Question 6 — What Is the Standard RLHF Pipeline?

### Simple intuition

**RLHF uses human preference data to guide a model toward responses that score better under a learned reward signal.**

A common classical pipeline has three stages:

~~~text
1. Supervised Fine-Tuning (SFT)
   Teach the model to give sensible instruction-following responses
                   ↓
2. Preference data → Reward Model
   Teach a scorer which responses people prefer
                   ↓
3. Reinforcement Learning (often PPO)
   Improve the generating model according to reward,
   while discouraging excessive departure from a reference model
~~~

This is a **common historical RLHF design**, not the only modern alignment workflow. DPO and other approaches do not necessarily run a separate reward-model-plus-PPO loop.

### Learn the four roles without mixing them

| Component | What it is | Primary job |
|---|---|---|
| Initial/SFT policy | A response-generating model after supervised training | Provide a competent starting behavior |
| Trainable policy $\pi_\theta$ | The model whose behavior we are improving | Generate responses and receive updates |
| Reward model $r_\phi$ | A scalar-response scorer trained from preferences | Estimate preference quality |
| Reference policy $\pi_{\mathrm{ref}}$ | A frozen distribution, often a copy of the SFT model | Constrain how far the updated policy moves |

In PPO-based setups, there may also be a **value model/critic** and a **rollout or old-policy snapshot**, which serve different purposes. We will derive them in Question 8.

### One RLHF iteration

For a sampled instruction $x$:

1. Generate one or more response sequences $y$ from the current policy.
2. Score the completed response with the reward model.
3. Calculate a policy-deviation penalty relative to the reference.
4. Estimate which generated actions were better than expected.
5. Update the trainable policy using an RL algorithm, often a clipped PPO-style update.
6. Repeat with fresh responses, monitor response quality, and test against held-out evaluations.

The human comparison work may occur **before** the RL optimization phase. We don't need a human in every policy-update step if using a learned reward model.

### Mathematical objective: why it contains two terms

A useful conceptual KL-regularized RLHF objective is:

$$
\max_\theta
\mathbb E_{x\sim\mathcal D,\;y\sim\pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)
\right]
-
\beta
\mathbb E_{x\sim\mathcal D}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\;\|\;
\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right]
$$

where:

- $x$ is the task prompt, possibly including an image.
- $y$ is the model-generated response sequence.
- $\pi_\theta$ is the trainable policy.
- $r_\phi$ is the learned reward score.
- $\pi_{\mathrm{ref}}$ is the frozen reference policy.
- $\beta>0$ controls how strongly we penalize deviation.
- The KL term measures policy distribution shift, not supervised token error.

Actual RLHF implementations differ in reward normalization, token-level penalties, value estimation and policy optimization details.

### Qwen Project Application

For the strict Little Content labeler, the supervised ground-truth labels already describe the desired decisions. We should **not** introduce a reward-model-and-PPO loop without evidence that classification metrics need a preference objective.

RLHF would be more justifiable if our system were redesigned to produce long, qualitative visual assessments where human reviewers could consistently rank two reasonable explanations.

### Design Decision — Q6

Keep RLHF as an **understood alternative**, not part of the proposed initial classification pipeline. Any future preference-learning experiment must specify what behavior cannot be captured adequately by supervised labels.

### Project Build Record Update — Q6

| Element | State | Type |
|---|---|---|
| Classical RLHF workflow | SFT → reward model → policy optimization | Study Concept |
| Reference model | Frozen baseline if RLHF is used | Study Concept |
| RLHF added to Little Content baseline | No | Design Decision |

---

### Story Bridge 7 — If We Maximize Reward, Why Not Let the Policy Change as Much as It Wants?

Suppose the reward model occasionally gives high scores to overly elaborate but misleading answers. A policy that aggressively maximizes this reward could move away from the broadly useful behavior learned in pretraining and SFT.

We need a way to control that change.

## Question 7 — Why Do RLHF Methods Use KL Regularization?

### Simple intuition

**KL regularization discourages the model from changing its response distribution too far away from a trusted starting policy.**

It is like saying: "Improve the behavior according to our reward, but do not abandon everything the initial model already knew."

This is not a perfect safety mechanism and cannot fix a fundamentally broken reward function. It is a way to control the size of policy changes.

### What KL divergence measures

For two distributions over possible responses:

$$
D_{\mathrm{KL}}(P\|Q)
=
\sum_y P(y)\log\frac{P(y)}{Q(y)}
$$

For language models, the distributions are conditioned on a prompt $x$, and responses $y$ are generally full token sequences.

KL is:

- Nonnegative when the distributions are well-defined.
- Zero when they match almost everywhere.
- **Not symmetric** in general: $D_{\mathrm{KL}}(P\|Q)$ need not equal $D_{\mathrm{KL}}(Q\|P)$.
- A measure of distribution mismatch, not the percentage of wrong answers.

### Numerical example with two response choices

Imagine the reference model gives two candidate responses equal probability:

$$
\pi_{\mathrm{ref}}=(0.5,0.5)
$$

The updated policy strongly favors the first:

$$
\pi_\theta=(0.9,0.1)
$$

Then:

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=
0.9\ln\frac{0.9}{0.5}
+
0.1\ln\frac{0.1}{0.5}
$$

$$
\approx0.9(0.5878)+0.1(-1.6094)
\approx0.3681
$$

The policy has changed quite substantially. If both distributions were $(0.5,0.5)$, the KL value would be zero.

### Why beta matters

An idealized objective:

$$
J(\theta)
=
\mathbb E_{\pi_\theta}[r(x,y)]
-
\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

For larger $\beta$, deviations are penalized more strongly. For smaller $\beta$, the optimizer is freer to follow the reward model.

**Important nuance:** KL can be formulated with different directions, sampled estimators or token-wise approximations. State the convention when discussing a specific implementation.

### Token-level connection

For an autoregressive response:

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t})
$$

A trajectory-level log-ratio relative to the reference can be written:

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=
\sum_{t=1}^{T}
\log\frac{
\pi_\theta(y_t\mid x,y_{<t})
}{
\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
}
$$

This identity helps explain sampled reference penalties. A single sampled log-ratio is **not itself an exact KL divergence**; its expectation under the appropriate policy gives a KL-related quantity.

### Qwen Project Application

If a hypothetical preference optimizer encouraged verbose explanations, a reference penalty could limit drastic changes. But it would not guarantee that the explanation is grounded in the screenshot or that output remains a valid binary label.

### Design Decision — Q7

If preference optimization is ever used, measure both reward/preference improvement **and policy drift**. Treat beta and KL targets as experiment parameters requiring validation.

### Project Build Record Update — Q7

| Element | State | Type |
|---|---|---|
| KL regularization | Constrains deviation from reference | Study Concept |
| KL strength $\beta$ | Not chosen | Open Question |
| Binary classifier preference training | Still unjustified | Design Decision |

---

### Story Bridge 8 — We Have a Reward and a Penalty, but How Can We Update a Model That Generates Discrete Tokens?

Text generation involves discrete token choices. We cannot simply differentiate through the sampled token ID as if it were a continuous number. Reinforcement-learning algorithms use probability-based gradients to improve actions whose outcomes are comparatively rewarding.

A classical choice in RLHF is PPO.

## Question 8 — How Does PPO-Based RLHF Work for a Language Model?

### First intuition

In PPO-based RLHF, **the policy generates text, receives a reward, and adjusts token probabilities in a controlled way**.

PPO stands for **Proximal Policy Optimization**. Its clipping mechanism discourages overly large policy changes within an update.

### Translate reinforcement-learning words into LLM terms

| RL term | LLM meaning |
|---|---|
| State | Prompt plus generated token prefix |
| Action | Next generated token |
| Policy | Distribution of next tokens from the language model |
| Episode / trajectory | Entire generated response |
| Reward | Score of a response, potentially with penalties |
| Value function / critic | Prediction of expected future reward from a state |
| Advantage | How much better or worse an action's outcome was than expected |

The reward may arrive primarily at the end of the response, whereas actions are individual tokens.

### Why we need an advantage estimate

A raw reward of 3 is hard to interpret without knowing whether 3 is typical for that prompt and model.

A simplified advantage estimate is:

$$
A_t=G_t-V(s_t)
$$

where:

- $G_t$ is a return estimate from position $t$.
- $V(s_t)$ is predicted expected return from the current state.
- $A_t>0$ suggests the observed outcome was better than predicted.
- $A_t<0$ suggests the outcome was worse than predicted.

Production PPO pipelines may use generalized advantage estimation (GAE), multiple value losses, reward normalization and extra details not needed to understand this core mechanism.

### PPO clipping: the key mathematical idea

Let $\pi_{\mathrm{old}}$ be the policy used to generate the training rollout. Define a probability ratio:

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\mathrm{old}}(a_t\mid s_t)
}
$$

The simplified PPO clipped objective maximizes:

$$
L_{\mathrm{clip}}(\theta)
=
\mathbb E_t
\left[
\min\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

Here $\epsilon$ defines a trust region **surrogate clipping interval**, not a strict guarantee that the entire model distribution cannot change.

For example, if $\epsilon=0.2$ and $A_t=1$:

- Changing the action probability ratio from $1.0$ to $1.1$ improves the surrogate.
- Raising it to $1.5$ does not add further gain in the clipped branch beyond $1.2$ for that positive advantage.

A separate reference-policy KL term may also be present. **The PPO old policy is not necessarily the same model as the frozen KL reference policy.**

### A simplified iteration

~~~text
Current policy generates response
          ↓
Reward model scores response
          ↓
Reference penalty and other task rewards
          ↓
Value model estimates expected return
          ↓
Calculate advantages for generated tokens
          ↓
PPO clipped policy update + value update
          ↓
Generate fresh responses and repeat
~~~

### Why this is expensive

Compared with offline SFT:

- We generate fresh responses while training.
- We may maintain policy, old-policy snapshots, a reference model, a reward model and a critic.
- We store rollouts and log probabilities.
- Optimization and hyperparameter interactions are more complex.

Implementations can share weights or use optimized placements, so not every model requires a fully independent copy in accelerator memory.

### Qwen Project Application

If the system only outputs one of two class labels, a full rollout/critic/PPO pipeline is generally far more complex than ordinary supervised learning and should be justified by a real unmet requirement, not included by default.

### Design Decision — Q8

PPO-based RLHF is **not** part of the current Qwen classification proposal. Know its policy-ratio, advantage and clipped-objective mechanics for general LLM alignment design.

### Project Build Record Update — Q8

| Element | State | Type |
|---|---|---|
| PPO old-policy ratio | Controls surrogate optimization update | Study Concept |
| Critic/value estimator | Common component in PPO RLHF | Study Concept |
| PPO applied to Little Content labels | Not proposed | Design Decision |

---

### Story Bridge 9 — A Better Reward Number May Still Mean a Worse Assistant

Suppose a reward model gives high scores to confident answers. A policy may learn to sound confident even when the screenshot evidence does not support its claims. Training can increase the reward metric while human judgments of actual quality deteriorate.

This is called reward hacking or reward overoptimization.

## Question 9 — What Are Reward Hacking, Reward Overoptimization and Common RLHF Failures?

### Simple intuition

**Reward hacking occurs when a model discovers behavior that earns a high score from the reward system without fulfilling the underlying goal.**

A reward is a measured approximation of what people want. Optimizing the approximation too aggressively can expose its flaws.

### A concrete example

Suppose the true goal is **accurate, concise image-grounded explanations**.

But the reward model has learned to associate longer, confidently worded answers with good responses.

Then training might shift toward:

~~~text
High reward-model score:
"Clearly, this screenshot definitely violates the policy because ..."
[long persuasive explanation with invented details]

Actual problem:
The screenshot contains a useful main answer block that the explanation ignored.
~~~

Reward-score improvement would not be proof of factual or visual correctness.

### A conceptual equation

Let $U(x,y)$ denote true task utility, which may be difficult to observe, and $r_\phi(x,y)$ the learned reward proxy.

Generally:

$$
r_\phi(x,y)\neq U(x,y)
$$

The policy is optimized according to the proxy:

$$
\max_\theta\mathbb E_{\pi_\theta}[r_\phi(x,y)]
$$

A rise in this expectation **does not logically imply** a rise in:

$$
\mathbb E_{\pi_\theta}[U(x,y)]
$$

### Typical failure categories

| Failure | Description | Possible check |
|---|---|---|
| Reward hacking | Exploits scorer's weaknesses | Independent human review and task metrics |
| Overoptimization | Reward rises while true quality falls | Compare reward curves with held-out judgments |
| Distribution shift | Policy generates responses unlike reward-model training examples | Evaluate out-of-distribution responses |
| Length/style bias | Scorer prefers wordiness or polish | Length-controlled comparisons and factuality checks |
| Annotator disagreement | Preference criterion poorly defined | Multi-rater audits and policy refinement |
| Mode collapse / reduced diversity | Policy concentrates on narrow response types | Response diversity and utility analysis |
| Excessive KL penalty | Policy cannot improve much | Reward/quality versus policy-drift trade-off |

### Why KL is not enough

KL penalizes large distribution shifts, but a model can still move toward local behaviors that exploit a flawed reward predictor. Conversely, overly strong KL may restrict improvements.

Therefore alignment requires **an independent evaluation of the actual task**, not only reward scores and KL statistics.

### Qwen Project Application

If our service is intended to flag Little Content pages, a model that generates convincing explanations for incorrect class labels is worse, not better. For the binary task, track true-label classification behavior and false-positive cost. For an optional explanation feature, evaluate visual grounding separately from prose quality.

### Design Decision — Q9

For any preference experiment, monitor independent human/task quality, held-out preference judgments and failure slices alongside reward metrics. Do not use reward-model scores as the sole success criterion.

### Project Build Record Update — Q9

| Element | State | Type |
|---|---|---|
| Reward hacking | Risk of optimizing proxy rather than true goal | Study Concept |
| Independent validation | Required for claims of improvement | Design Decision |
| Grounding of explanations | Would need explicit tests if feature introduced | Open Question |

---

### Story Bridge 10 — RLHF Works, but Must We Always Train a Reward Model and Run an RL Loop?

We have learned a three-stage RLHF pipeline: supervised model, pairwise-trained reward scorer and a policy optimizer such as PPO. It has understandable logic, but it is computationally and operationally complex.

A natural question follows: can we learn directly from the preferred and rejected responses, without separately training a scalar reward model and running online reinforcement learning?

That is the motivation behind Direct Preference Optimization.

## Question 10 — What Is DPO, and Why Can It Avoid an Explicit Reward Model?

### Simple intuition

**DPO (Direct Preference Optimization) trains the response-generating model directly from chosen/rejected response pairs.**

It compares how likely the policy finds the preferred and rejected responses, while also using a **frozen reference model** to keep the comparison anchored to the starting policy.

Standard DPO does not require an explicitly trained reward network, a separate critic or a PPO rollout loop.

### The training record is familiar

~~~text
Prompt / image x
Chosen response y_w
Rejected response y_l
~~~

The model computes log probabilities for both complete responses under:

- The trainable policy $\pi_\theta$.
- The fixed reference policy $\pi_{\mathrm{ref}}$.

Teacher forcing can compute sequence log probabilities for the two known responses. This is an **offline preference loss**, although other online DPO variants exist.

### Important limitation of the slogan "DPO removes the reward model"

DPO does **not** make preferences disappear. Its derivation uses an implicit connection between a KL-regularized reward objective and policy log-probability ratios.

We no longer need to separately train and query a reward network in the standard DPO update. But the mathematical interpretation relies on assumptions about preferences and the underlying optimization objective.

### Derive the policy–reward relationship carefully

Consider a fixed prompt $x$ and a reference policy $\pi_{\mathrm{ref}}(y\mid x)$. Under a KL-regularized reward objective, the optimal response distribution has a Boltzmann-like form:

$$
\pi^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
$$

where:

- $r(x,y)$ is a theoretical reward function.
- $\beta>0$ is the KL regularization coefficient in this convention.
- $Z(x)$ is a prompt-dependent normalizing constant.

Take logarithms:

$$
\log\pi^*(y\mid x)
=
\log\pi_{\mathrm{ref}}(y\mid x)
+
\frac{r(x,y)}{\beta}
-
\log Z(x)
$$

Rearrange:

$$
r(x,y)
=
\beta
\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x)
$$

For a preference pair with the same prompt, subtract rewards:

$$
r(x,y_w)-r(x,y_l)
=
\beta\left[
\log\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
$$

The normalization $\log Z(x)$ cancels because both responses belong to the **same prompt**.

Now substitute this implicit reward difference into the Bradley–Terry preference likelihood from Question 5. Replace the unknown optimum $\pi^*$ with the trainable policy $\pi_\theta$.

This gives the DPO loss:

$$
\mathcal L_{\mathrm{DPO}}
=
-\mathbb E_{(x,y_w,y_l)}
\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
\right)
$$

This is the central mathematical result.

### Why this is useful

- Both responses are already present, so we can teacher-force their log probabilities.
- We compare how the policy favors each response **relative to its reference**.
- A normal differentiable loss updates the policy parameters or LoRA adapters.
- We avoid a separately trained reward model and online PPO rollouts in standard offline DPO.

### What the derivation does not guarantee

The optimal-policy formula is derived under particular idealized assumptions, including the form of the KL-regularized reward objective and policy support. Finite datasets, inconsistent preferences, reference mismatch, noisy labels and model capacity limits can affect practical outcomes. DPO is not guaranteed to optimize any arbitrary real-world notion of human preference perfectly.

### Qwen Project Application

If we collect high-quality **pairs of alternative written explanations for the same screenshot**, DPO could theoretically fine-tune their relative preference without building a separate reward model. But this does not demonstrate that DPO is beneficial for predicting one of two known class labels.

### Design Decision — Q10

Understand DPO as a potential preference-learning tool. Keep canonical-label SFT as the initial Little Content classification training path. Any DPO extension must define its preference dataset and success criteria first.

### Project Build Record Update — Q10

| Element | State | Type |
|---|---|---|
| DPO training data | Prompt, chosen response, rejected response | Study Concept |
| DPO trainable policy | Updated from pairwise loss | Study Concept |
| Frozen reference policy | Needed in standard DPO loss | Study Concept |
| DPO for binary classifier | Not proposed without incremental-value evidence | Design Decision |

---

