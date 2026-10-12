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
\mathcal{L}_{\mathrm{SFT}} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\mathrm{SFT}}} \log \pi_\theta(y \mid x)
$$


Here $\pi_\theta$ denotes the response-generating policy (our generative model).

Pairwise preference data:

$$
\mathcal{D}_{\mathrm{pref}} = \{(x_i, y_{w,i}, y_{l,i})\}_{i=1}^{N}
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
P_\phi(y_w\succ y_l\mid x) =
\frac{e^{r_w}}{e^{r_w}+e^{r_l}}
$$

Divide numerator and denominator by $e^{r_w}$:

$$
P_\phi(y_w\succ y_l\mid x) = \frac{1}{1+e^{-(r_w-r_l)}} =
\sigma(r_w-r_l)
$$

where $\sigma(z)=1/(1+e^{-z})$ is the sigmoid function.

Since the observed label says "$y_w$ wins", negative log-likelihood gives:

$$
\mathcal L_{\mathrm{RM}} =
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
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, \; y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x,y) \right] - \beta \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\mathrm{KL}} \left( \pi_\theta(\cdot \mid x) \parallel \pi_{\mathrm{ref}}(\cdot \mid x) \right) \right]
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
D_{\mathrm{KL}}(P\|Q) =
\sum_y P(y)\log\frac{P(y)}{Q(y)}
$$

For language models, the distributions are conditioned on a prompt $x$, and responses $y$ are generally full token sequences.

KL is:

- Nonnegative when the distributions are well-defined.
- Zero when they match almost everywhere.
- **Not symmetric** in general: $D_{\mathrm{KL}}(P\|Q)$ need not equal $D_{\mathrm{KL}}(Q\|P)$.
- A measure of distribution mismatch, not the percentage of wrong answers.

### Deep Dive — KL Divergence From First Principles

The brief definition above can feel abstract. To understand it, first forget RLHF and imagine **two models distributing probability across three possible responses**.

#### Step 1 — What is a distribution over responses?

Imagine this prompt:

> Complete the sentence: "The capital of France is ..."

For a deliberately simplified example, suppose we have only three possible responses:

| Response $y$ | Model $P(y)$ | Model $Q(y)$ |
|---|---:|---:|
| Paris | $0.80$ | $0.50$ |
| London | $0.15$ | $0.30$ |
| Berlin | $0.05$ | $0.20$ |
| **Total** | **1.00** | **1.00** |

Both distributions are valid: their probabilities sum to 1. Model P puts more probability on Paris, whereas Q spreads more probability to London and Berlin. We want a numerical description of **how different these probability assignments are**.

This tiny table is *only a mathematical teaching example*. An actual language model can assign probability to a huge number of possible token sequences, not just three city names.

#### Step 2 — Read the equation one component at a time

$$
D_{\mathrm{KL}}(P\|Q) =
\sum_y P(y)\ln\frac{P(y)}{Q(y)}
$$

| Piece | Plain-language meaning |
|---|---|
| $y$ | One possible response |
| $P(y)$ | Probability that P assigns to that response |
| $Q(y)$ | Probability that Q assigns to that response |
| $P(y)/Q(y)$ | How large P's probability is relative to Q's |
| $\ln(P(y)/Q(y))$ | Logarithm of that probability ratio |
| $P(y)\times\ln(P(y)/Q(y))$ | Contribution weighted by how likely that response is under P |
| $\sum_y$ | Add contributions over all possible responses |

Why use a **ratio**? If P assigns Paris probability $0.80$ and Q assigns $0.50$, the ratio is:

$$
\frac{P(\mathrm{Paris})}{Q(\mathrm{Paris})} =
\frac{0.80}{0.50}=1.6
$$

P considers that response 1.6 times as likely. For Berlin:

$$
\frac{P(\mathrm{Berlin})}{Q(\mathrm{Berlin})} =
\frac{0.05}{0.20}=0.25
$$

P considers that response only one-quarter as likely.

The natural log turns **equal probabilities** into zero because $\ln(1)=0$, ratios above one into positive values, and ratios below one into negative values.

#### Step 3 — Calculate all three contributions

For our distributions:

$$
D_{\mathrm{KL}}(P\|Q) =
0.80\ln\frac{0.80}{0.50}
+
0.15\ln\frac{0.15}{0.30}
+
0.05\ln\frac{0.05}{0.20}
$$

Compute each response's contribution:

| Response | Calculation | Approximate contribution |
|---|---|---:|
| Paris | $0.80\ln(1.6)$ | $+0.376$ |
| London | $0.15\ln(0.5)$ | $-0.104$ |
| Berlin | $0.05\ln(0.25)$ | $-0.069$ |

Add them:

$$
D_{\mathrm{KL}}(P\|Q)
\approx0.376-0.104-0.069
\approx0.203
$$

The result is about **0.203 nats**. "Nats" is simply the unit used when the natural logarithm is chosen.

**Why were some contributions negative?** A response gets a negative contribution whenever P assigns it lower probability than Q. Nevertheless, the *complete sum* is nonnegative for valid distributions. Nonnegativity is a property of the whole expression, not each individual term.

For finite distributions, this standard formula assumes that Q assigns positive probability wherever P assigns positive probability. If $P(y)>0$ and $Q(y)=0$, then $D_{\mathrm{KL}}(P\|Q)$ is infinite under the usual convention. Terms where $P(y)=0$ contribute zero.

#### Step 4 — Why does KL become zero when P and Q match?

Suppose the two models assign exactly the same probabilities:

$$
P(y)=Q(y)
$$

For each response with nonzero probability:

$$
\frac{P(y)}{Q(y)}=1,
\qquad
\ln(1)=0
$$

So every contribution is zero, giving:

$$
D_{\mathrm{KL}}(P\|Q)=0
$$

More generally, KL divergence is **zero if and only if the distributions match** (up to probability-zero events under the usual conditions). The greater its value, the greater this particular kind of mismatch; it is **not** a percentage error or a probability that the model is wrong.

#### Step 5 — Why is KL not symmetric?

The two directions weight their responses differently.

$$
D_{\mathrm{KL}}(P\|Q) =
\sum_y P(y)\ln\frac{P(y)}{Q(y)}
$$

But reversing the order gives:

$$
D_{\mathrm{KL}}(Q\|P) =
\sum_y Q(y)\ln\frac{Q(y)}{P(y)}
$$

In the first expression, **P determines the weighting**; in the second, **Q determines it**.

With our numbers:

$$
D_{\mathrm{KL}}(P\|Q)\approx0.203
$$

whereas:

$$
D_{\mathrm{KL}}(Q\|P) =
0.50\ln\frac{0.50}{0.80}
+
0.30\ln\frac{0.30}{0.15}
+
0.20\ln\frac{0.20}{0.05}
\approx0.240
$$

The directions differ even though the same two models are involved.

**Mental model:** $D_{\mathrm{KL}}(P\|Q)$ evaluates the mismatch with responses weighted according to P. Reversing to $D_{\mathrm{KL}}(Q\|P)$ changes whose likely responses matter most.

This is why KL divergence is not an ordinary symmetric distance metric.

#### Step 6 — Translate P and Q back to RLHF

Now restore the two actual RLHF roles:

- $\pi_{\mathrm{ref}}(y\mid x)$: frozen reference model's response distribution for prompt $x$.
- $\pi_\theta(y\mid x)$: updated trainable policy's response distribution for the same prompt.

For example, suppose the reference and updated policy assign the following probabilities to **three illustrative response styles**:

| Response style | Frozen reference | Updated policy |
|---|---:|---:|
| Concise and accurate | $0.60$ | $0.85$ |
| Detailed and accurate | $0.30$ | $0.10$ |
| Incorrect or misleading | $0.10$ | $0.05$ |

The updated model has become more likely to give a concise answer and less likely to give a detailed one. KL can quantify that probability-distribution shift, **but it cannot tell us whether the shift is a good one**. We still need accuracy, helpfulness and task-specific evaluation.

Real LLMs do not directly have these three discrete style categories in their output vocabulary; this is a coarse illustration of full-response behavior.

#### Step 7 — Why are full-response probabilities related to token probabilities?

An autoregressive model constructs each response one token at a time:

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{\lt t})
$$


Taking logs turns that product into a sum:

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{\lt t})
$$


Consequently, the log ratio between updated and reference policies is:

$$
\log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} = \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\mathrm{ref}}(y_t \mid x, y_{\lt t})}
$$


An exact KL divergence over **all possible complete responses** is generally too expensive to enumerate, so practical RLHF systems use sampled or token-level calculations and estimators.

A **single sampled response log ratio is not itself the exact KL divergence**. For the forward KL convention, its expectation under the updated policy yields the sequence-level KL (when the distributions satisfy the required support conditions).

#### Step 8 — Why RLHF subtracts a KL penalty

A reward model might encourage an update to make replies extremely short or overly confident. Optimizing only the learned reward could push the policy far away from its original response distribution, potentially exploiting reward-model weaknesses.

A common idealized objective is:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}[r(x,y)] - \beta D_{\mathrm{KL}} \left( \pi_\theta(\cdot \mid x) \parallel \pi_{\mathrm{ref}}(\cdot \mid x) \right)
$$


The first term encourages higher-scoring responses. The second discourages large reference-relative distribution shifts.

- **Smaller $\beta$:** weaker penalty, more freedom to change.
- **Larger $\beta$:** stronger penalty, more pressure to remain close to the reference.

A KL constraint **cannot guarantee factual accuracy or safety**, and restricting drift is not always desirable if the reference behavior needs substantial improvement.

### Three Concepts That Must Stay Separate

| Quantity | Main question |
|---|---|
| SFT cross-entropy | How likely is the designated correct target? |
| Reward score | How desirable is a response according to the reward system? |
| KL divergence | How different is the new policy's response distribution from the reference? |

**Final intuition:** A reward model expresses a direction we would like to improve; KL regularization discourages changing the whole probability distribution too drastically while pursuing that direction.

---


### Token-level connection

For an autoregressive response:

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{\lt t})
$$


A trajectory-level log-ratio relative to the reference can be written:

$$
\log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} = \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\mathrm{ref}}(y_t \mid x, y_{\lt t})}
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

### Story Bridge 11 — The DPO Formula Looks Compact, but What Does It Actually Reward?

Question 10 derived a loss with four response probabilities: policy and reference probabilities for the preferred and rejected answer. If we cannot understand how those numbers combine in a real example, we may incorrectly describe DPO as "simply increasing the chosen response probability."

Let us calculate one preference update conceptually.

## Question 11 — How Do We Calculate and Interpret the DPO Loss?

### Step 1 — Define the two response log probabilities

For one input $x$, suppose two complete candidate responses are:

~~~text
Chosen:  "The page is sparse, but the visible answer card is useful."
Rejected: "Any page with white space is low-quality."
~~~

Let:

$$
\ell_\theta^w=\log\pi_\theta(y_w\mid x)
$$

$$
\ell_\theta^l=\log\pi_\theta(y_l\mid x)
$$

and define the corresponding reference values:

$$
\ell_{\mathrm{ref}}^w=\log\pi_{\mathrm{ref}}(y_w\mid x)
$$

$$
\ell_{\mathrm{ref}}^l=\log\pi_{\mathrm{ref}}(y_l\mid x)
$$

Each complete-response log probability is the **sum of conditional next-token log probabilities** along that response under teacher forcing:

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t})
$$

As in Part 2, the input prompt/image conditions the answer but does not need to be included as supervised response tokens. Be consistent about whether the completion includes the end-of-turn token.

### Step 2 — Calculate how much the policy favors the winner relative to the loser

Define the policy log-probability margin:

$$
m_\theta=\ell_\theta^w-\ell_\theta^l
$$

The reference model has its own margin:

$$
m_{\mathrm{ref}}=\ell_{\mathrm{ref}}^w-\ell_{\mathrm{ref}}^l
$$

The DPO logit is the **change in relative preference margin**, scaled by $\beta$:

$$
z_{\mathrm{DPO}}
=
\beta(m_\theta-m_{\mathrm{ref}})
$$

The loss is:

$$
\mathcal L_{\mathrm{DPO}}=-\log\sigma(z_{\mathrm{DPO}})
$$

### Step 3 — Numerical worked example

Take these illustrative **log probabilities**:

| Candidate | Trainable policy | Frozen reference |
|---|---:|---:|
| Chosen | $-2.0$ | $-2.5$ |
| Rejected | $-3.0$ | $-2.8$ |

The policy margin is:

$$
m_\theta=-2.0-(-3.0)=1.0
$$

The reference margin is:

$$
m_{\mathrm{ref}}=-2.5-(-2.8)=0.3
$$

So:

$$
m_\theta-m_{\mathrm{ref}}=0.7
$$

For an illustrative $\beta=0.5$:

$$
z_{\mathrm{DPO}}=0.5(0.7)=0.35
$$

Then:

$$
\sigma(0.35)\approx0.587
$$

and:

$$
\mathcal L_{\mathrm{DPO}}
=
-\ln(0.587)\approx0.533
$$

If the policy favored the chosen response less than the reference does, the margin change could be negative and the loss higher.

### Step 4 — What does the gradient encourage?

Define:

$$
\Delta=
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
$$

Then:

$$
\mathcal L=-\log\sigma(\beta\Delta)
$$

Differentiate with respect to $\Delta$:

$$
\frac{\partial\mathcal L}{\partial\Delta}
=
-\beta\sigma(-\beta\Delta)
$$

For $\beta>0$, this derivative is negative. Gradient descent therefore tends to increase $\Delta$: the policy becomes relatively more favorable to the chosen response compared with the rejected one, after adjusting for reference preference.

**Important subtlety:** This does **not** promise that the *absolute* probability of the chosen response increases at every update. Model normalization and parameter sharing can cause complex probability changes. The direct training pressure is on the **relative reference-adjusted comparison**.

### Step 5 — What does beta control?

Under this particular DPO convention, $\beta$ scales the preference margin inside the sigmoid, and also corresponds to the KL coefficient in the idealized derivation.

It affects the gradient scale and how strongly preference log-ratio changes are reflected by the loss. Its practical relationship to final model drift depends on optimization and data, so it is not correct to say "larger beta always means better alignment".

At $\Delta=0$:

$$
\mathcal L=-\log\sigma(0)=\log 2\approx0.693
$$

The model is at the reference-relative indifference point, regardless of positive beta.

### Qwen Project Application

Suppose we train Qwen to produce a visual explanation and reviewers prefer one that correctly mentions an answer card over one that declares all whitespace junk. DPO would compare log probabilities of those **two explanations for the same screenshot**, not just the probabilities of the two class labels in our baseline classifier.

### Design Decision — Q11

If we ever experiment with DPO, inspect chosen/rejected **sequence log probabilities**, reference-adjusted margins, beta, response tokenization, and held-out task outcomes. Do not interpret the DPO loss as a calibrated class probability.

### Project Build Record Update — Q11

| Element | State | Type |
|---|---|---|
| DPO margin | Policy winner-minus-loser margin minus reference margin | Study Concept |
| DPO beta | Training hyperparameter, not chosen | Open Question |
| DPO margin versus class likelihood | Not interchangeable | Study Concept |

---

### Story Bridge 12 — DPO Is Simpler to Train, but Does That Make It Always Better Than RLHF?

We have seen two ways to use human preferences. PPO-based RLHF can generate new responses and optimize an explicit learned reward; standard DPO trains directly from a fixed comparison dataset.

Choosing between them requires understanding the **engineering trade-offs and limitations**, not just counting equations.

## Question 12 — How Do DPO and PPO-Based RLHF Compare?

### The short explanation

**PPO-based RLHF** explicitly uses reward scores to drive a policy-optimization loop. It can incorporate generated rollouts and additional reward signals, but the training system is relatively complicated.

**Standard offline DPO** directly fits pairwise preferences with policy and reference log probabilities. It is often easier to train and more resource-efficient, but its quality depends heavily on the fixed preference dataset and reference-policy assumptions.

| Dimension | PPO-based RLHF | Standard offline DPO |
|---|---|---|
| Supervision | Learned scalar reward, often from human pairs | Chosen/rejected response pairs |
| Separate reward model | Commonly yes | Not required by standard DPO loss |
| Value/critic model | Often yes | No |
| Online generated rollouts | Typical during training | Not required |
| Frozen reference | Often used for KL penalty | Used directly in DPO ratios |
| Training complexity | Higher; multiple interacting models and hyperparameters | Lower; supervised-style pairwise forward passes |
| Model exploration | Can sample fresh responses as the policy changes | Limited by its fixed data unless refreshed |
| Failure modes | Reward hacking, instability, critic/reward errors | Pair quality, reference mismatch, offline distribution limits |
| Guarantee of better alignment | None | None |

### Why offline data distribution matters

Suppose the preference dataset contains only easy, polite chat exchanges. After fine-tuning, the model encounters complicated scientific reasoning prompts. Standard DPO cannot be assumed to learn robust behavior for inputs and responses absent from its training distribution.

PPO-based RLHF has its own generalization problems, but a policy can collect new rollouts as it changes. That does not automatically solve label quality or safety concerns.

### What about other methods?

There are many related objectives and training algorithms, including online preference methods and reference-free or differently regularized losses. This chapter focuses on understanding the **standard conceptual forms** of reward-model RLHF and DPO first.

### Model-size and hardware considerations

DPO generally requires scoring two candidate responses under the trainable model and reference, often with teacher-forced forward passes. It can be implemented with LoRA, reference-log-probability caching or other memory-saving approaches, depending on library support.

PPO may need response generation, reward scoring, value estimation and policy updates. Its resource demands are typically more complex, but exact relative cost depends on implementation, model size, response length and hardware.

### Qwen Project Application

Neither method is the default solution for a known two-class screenshot label. If we had a separate natural-language explanation assistant with a high-quality ranked-response dataset, DPO would be a sensible first preference method to evaluate before designing a complex RLHF stack. This remains an experiment proposal, not a measured result.

### Design Decision — Q12

Choose methods based on **data format, production behavior, empirical quality, hardware and training complexity**. For current Little Content labels, keep SFT as the baseline; DPO/RLHF only enter when a preference-specific gap is identified.

### Project Build Record Update — Q12

| Element | State | Type |
|---|---|---|
| Offline preference optimization | DPO is a possible simpler baseline | Study Concept |
| PPO-RLHF alternative | More complex, may use fresh rollouts | Study Concept |
| Actual selection for screenshot classification | SFT remains preferred baseline | Design Decision |

---

### Story Bridge 13 — The DPO Equation Is Straightforward, but Training Can Still Fail Through Data Formatting

Part 3 taught us that chat templates, image processors and loss masks matter. DPO uses **two responses for the same prompt**, so a single example has even more opportunities for mismatched formatting, context or token-level scoring.

We must connect preference optimization to the practical data pipeline we already designed.

## Question 13 — How Do We Train with Preference Pairs in Practice, Especially for a VLM?

### The core rule

A valid preference pair uses the **same conditioning input** for both completions:

~~~text
Input:
    System policy
    User instruction
    Same SERP screenshot I

Completion 1:
    Preferred response y_w

Completion 2:
    Rejected response y_l
~~~

A pair is invalid for ordinary response-preference training if the two candidates answer different screenshots or materially different instructions.

### Processing workflow

~~~text
One labelled preference record
      |
      +--> shared raw conversation / image input
      |
      +--> chosen completion
      |
      +--> rejected completion
      ↓
Apply checkpoint-compatible chat template / processor
      ↓
Score chosen and rejected target sequences under policy
      ↓
Score chosen and rejected sequences under frozen reference
      ↓
Compute DPO margin and loss
      ↓
Backpropagate into trainable policy / LoRA parameters
~~~

The frozen reference provides probabilities but does not receive optimizer updates.

### What can go wrong?

**1. The prompt is not identical.**

If the chosen example contains one image and the rejected example contains another, the reward comparison is entangled with different evidence.

**2. The answer spans are misidentified.**

The log probabilities should cover the intended **response sequence**, not arbitrarily include different prompt lengths. The implementation must handle the same conventions for policy and reference.

**3. Completion lengths differ.**

A sequence log probability is a sum over tokens. Longer sequences can accumulate more negative log probability simply because they contain more factors.

This is not automatically repaired by dividing by response length; length normalization changes the objective. It requires task-specific evaluation.

**4. Chat templates differ between policy and reference.**

The two policies' log ratios are meaningful only when they score compatible prompt/response representations. Different or duplicated templates undermine the comparison.

**5. Image preprocessing differs across chosen/rejected paths.**

A consistent screenshot must result in consistent visual inputs for both candidates. Cached visual representations are only safe if the implementation preserves their association and model semantics.

**6. Preference labels reflect style rather than quality.**

If reviewers prefer longer, polished descriptions even when visually wrong, DPO can learn that bias.

### Teacher-forcing connection

We already know the two full responses. DPO can evaluate their token log probabilities **without sampling each output from the updated policy at training time**:

$$
\log\pi_\theta(y_w\mid x)
=
\sum_t\log\pi_\theta(y_{w,t}\mid x,y_{w,<t})
$$

$$
\log\pi_\theta(y_l\mid x)
=
\sum_t\log\pi_\theta(y_{l,t}\mid x,y_{l,<t})
$$

This is the same causal conditional-probability factorization from Parts 2 and 3, now applied twice for one preference pair.

### Qwen Project Application

Suppose we introduce a distinct, optional ability to explain the Little Content decision. We could compare two explanations for the *same* screenshot, provided human reviewers judge **groundedness, policy consistency and correctness**, rather than just eloquence.

For the production label-only interface, do not create artificial preference pairs by calling the known correct label "chosen" and the incorrect label "rejected" unless there is a clear, experimentally justified reason to abandon straightforward class-supervised learning.

### Design Decision — Q13

Preference training is hypothetical. If introduced, require strict shared input/image, matching chat templates, carefully scoped completion log probabilities, consistent response endings and expert-reviewed preference criteria.

### Project Build Record Update — Q13

| Element | State | Type |
|---|---|---|
| Chosen/rejected input equality | Must use same prompt/image | Design Decision |
| Preference completion masks | Score intended assistant response only | Design Decision |
| Multimodal DPO recipe | Not yet validated on any checkpoint | Open Question |

---

### Story Bridge 14 — Lower DPO Loss and Higher Reward Are Training Signals. How Do We Know People Actually Prefer the New Model?

Part 2 taught us that lower token loss is not the same as fewer business errors. The same principle applies more strongly to preference optimization: training objectives are only proxies for the behavior we actually want.

We need independent evaluation.

## Question 14 — How Do We Evaluate a Preference-Aligned Model?

### The central principle

**Do not declare alignment successful based solely on reward score or DPO training loss.**

Evaluate on new prompts, new response types and held-out preference judgments. For task-specific systems, measure the actual task objective too.

### Preference win rates

Let model A and model B each produce responses for the same held-out prompt set. Humans or a sufficiently validated judge compare them using a fixed rubric.

If A wins 60 out of 100 comparisons, B wins 30, and 10 are ties, one common reporting convention is:

$$
\mathrm{WinRate}_{A}=\frac{60}{100}=0.60
$$

Some evaluation protocols assign half credit to ties:

$$
\mathrm{WinRateWithHalfTies}_{A}
=
\frac{60+0.5(10)}{100}
=
0.65
$$

**Always specify the tie convention.** These are illustrative counts, not measured Qwen results.

### Common evaluation dimensions

| Dimension | Question |
|---|---|
| Pairwise preference win rate | Is the new model preferred over a baseline on held-out prompts? |
| Human consistency | Would qualified reviewers broadly agree with the preference rubric? |
| Groundedness / factuality | Does the response reflect evidence rather than invented details? |
| Instruction following | Does it obey the requested length, role and output format? |
| Safety and harmful failures | Does preference training introduce dangerous or inappropriate behavior? |
| Diversity and verbosity | Is the model becoming repetitive or unnecessarily long? |
| Robustness | Does behavior hold for unfamiliar prompts, hosts or layouts? |
| Computational cost | Is any gain worth serving/training memory, latency and complexity? |

### Judge-model caveats

A model-based judge can reduce evaluation cost but may show position bias, style bias, familiarity bias, verbosity bias or correlated model errors.

Good practice includes:

- Randomizing A/B response order.
- Using clear rubrics.
- Recording ties and uncertainty.
- Checking a human-reviewed subset.
- Avoiding judges that simply reward their own preferred writing style.
- Reporting evaluation-set composition and error slices.

### Why direct task metrics remain necessary

For the Little Content classifier, define positives as Little Content and calculate:

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

$$
\mathrm{FPR}=\frac{FP}{FP+TN}
$$

A model could win more explanation-preference judgments yet produce more false-positive class labels. Such a change may be unacceptable for the actual product.

### Qwen Project Application

If preferences were added for explanations, evaluate **explanation quality** separately from **classification correctness**. Preserve hard-negative screenshot slices, unfamiliar hosts and the required label-only production behavior.

### Design Decision — Q14

Require task-level evaluation, held-out preference comparisons, human audit and cost analysis before accepting any alignment method. Separate explanatory preference wins from classification improvements.

### Project Build Record Update — Q14

| Element | State | Type |
|---|---|---|
| Pairwise win rate | May evaluate optional explanation model | Open Question |
| Classification metrics | Primary for current task | Design Decision |
| Judge bias safeguards | Needed for preference-evaluation pipeline | Design Decision |

---

### Story Bridge 15 — We Know SFT, Reward Models, PPO and DPO. Which of Them Actually Belongs in Our Qwen Design?

Without a final design decision, the chapter would become a collection of equations. The point of an engineering curriculum is to decide *when each tool solves a real problem*.

We now combine the theory with the Qwen Little Content task and explicitly record what is known, proposed and unresolved.

## Question 15 — What Is Our Qwen Preference-Optimization Design, and When Would We Use It?

### Step 1 — State the actual product problem

The project description concerns identifying **Little Content** SERP pages using screenshots and a Qwen VLM.

Our theoretical training interface remains:

~~~text
Input:
    Screenshot of a SERP page
    Task instruction

Output:
    Little Content
      OR
    Not Little Content
~~~

We have reviewed supervised class labels from Parts 2/3. There is no established need for free-form comparison of multiple long responses.

### Step 2 — Identify the learning signal

If each screenshot has a trustworthy class label, the direct supervised task is:

$$
\mathcal D_{\mathrm{class}}
=
\{(I_i,u_i,y_i)\}_{i=1}^{N}
$$

with assistant-target cross-entropy:

$$
\mathcal L_{\mathrm{class}}
=
-\sum_i
\sum_{t\in\mathcal T_i}
\log\pi_\theta(y_{i,t}\mid I_i,u_i,y_{i,<t})
$$

This directly supervises the desired answer.

### Step 3 — Identify what preference training would add

Potential extra product requirements might include:

- Provide a concise, evidence-grounded explanation of a screenshot decision.
- Rank two borderline policy interpretations when no unique target explanation exists.
- Prefer a shorter, clearer or safer *explanation* among several factually acceptable ones.

These are **hypothetical extensions**. They should not be silently assumed to exist.

### Step 4 — Decide between methods using the problem, not popularity

| Situation | Most natural initial strategy | Why |
|---|---|---|
| Reliable screenshot → binary class labels | SFT or a dedicated classifier baseline | Direct target supervision expresses the goal |
| Unclear label policy / annotator disagreement | Improve policy, adjudication and data first | RLHF does not fix contradictory ground truth |
| Open-ended, rankable explanations | SFT then consider DPO on high-quality pairs | Pairwise behavior preference is relevant |
| Need online reward-guided exploration with validated scorer | Evaluate PPO-based RLHF or alternatives | Explicit rewards and fresh rollouts may matter |
| Primary memory limitation | LoRA/QLoRA as supported | Adaptation and memory strategy, not preference method |

### Step 5 — Form an honest proposed architecture

~~~mermaid
flowchart TD
    A["Labelled SERP screenshots"] --> B["Annotation and leakage audits"]
    B --> C["Prompt-only and simpler-classifier baselines"]
    C --> D["Supervised Qwen training with canonical class targets"]
    D --> E["Proposed PEFT/LoRA adaptation"]
    E --> F["Independent classification evaluation"]
    F --> G{"Demonstrated need for preference learning?"}
    G -->|No| H["Keep SFT + validated scoring"]
    G -->|Yes, explanation/ranking use case| I["Define preference rubric and collect pairs"]
    I --> J["Evaluate DPO baseline versus other methods"]
    J --> K["Validate human preference AND classification correctness"]
~~~

The diagram describes a **hypothetical development decision tree**, not a completed deployed pipeline.

### Step 6 — What would convince us to add DPO?

We would require all of the following:

1. A genuine preference-oriented capability that canonical classification targets do not sufficiently teach.
2. A trustworthy set of pairs built from the **same screenshot and instruction** with a clearly documented rubric.
3. A reproducible DPO baseline with compatible policy/reference processing.
4. Held-out evidence of improved preferred behavior **without** unacceptable classification regressions, hallucinations, or serving cost.
5. Human review of critical error slices.

If these conditions are not met, adding DPO is unnecessary complexity.

### Step 7 — What would convince us to add PPO-based RLHF?

We would need a substantial reason to favor online reward-guided policy optimization and sufficient evidence that an explicit reward signal can be trusted. We would also need the computing, software and evaluation capabilities to manage policy rollouts, reward scoring, reference drift and possible value estimation.

For a strict binary labeler, this is usually a much higher bar than SFT.

### Qwen Project Application — Final Part 4 Decision

**Our current theoretical Qwen Little Content system remains SFT-first, with the Part 2B LoRA baseline and Part 3 multimodal instruction formatting.**

- **Not adding by default:** explicit reward model, PPO loop or DPO preference stage.
- **Potentially adding later:** preference optimization for a separately justified response-explanation/ranking feature.
- **Still unresolved:** actual checkpoint details, training and evaluation metrics, dataset quality, hyperparameters and production scoring.

### Design Decision — Q15

Choose **direct class-supervised training** as the default. Preference optimization is conditional on a demonstrated preference-specific requirement, not a required stage of every fine-tuning project.

### Project Build Record Update — Q15

| Element | State | Type |
|---|---|---|
| Preference stage in baseline | Omitted | Design Decision |
| Reward model in baseline | Omitted | Design Decision |
| PPO in baseline | Omitted | Design Decision |
| DPO in baseline | Omitted unless separate need shown | Design Decision |
| Optional explanation preference research | Not established | Open Question |

---

# Qwen Project Build Record — After Part 4

This is a cumulative record. **Resume Fact** identifies described work/scope, not proof that the theoretical training configurations below were implemented. **Design Decision** identifies a proposal within this study. **Open Question** flags matters requiring real evidence.

| Design element | Current state | Type |
|---|---|---|
| Business objective | Little Content detection on SERP screenshots | Resume Fact |
| Model family | Qwen vision-language model | Resume Fact |
| Input modality | Labelled Bing SERP screenshots | Resume Fact |
| Described processing scale | About 50K URLs/day | Resume Fact |
| Core model task | Binary Little Content classification | Design Decision |
| Input format | Screenshot + task instruction | Design Decision |
| Supervised response | One of two canonical class labels | Design Decision |
| Optimization signal | Teacher-forced assistant-target cross-entropy | Design Decision |
| Message roles and chat format | Checkpoint-compatible multimodal instruction formatting | Design Decision |
| Image processing | Must match exact selected checkpoint | Design Decision |
| Trainable weights | Proposed language-side LoRA starting baseline | Design Decision |
| Pretrained vision and language weights | Initially frozen in proposal | Design Decision |
| Initial adapter target candidates | Language attention Query/Value projections | Design Decision |
| Multimodal connector | Trainable if separately available and justified | Design Decision |
| Quantized base / QLoRA | Possible memory-constrained alternative | Design Decision |
| Reliable raw labels and annotation policy | Must be verified | Open Question |
| Generalization target | Unseen hosts, layouts and hard negatives | Design Decision |
| Training/validation/test groups | Exact split not established | Open Question |
| Candidate class scoring and threshold | Validate experimentally | Open Question |
| Preference pairs for current classification task | Not defined | Open Question |
| Standalone reward model | Not part of baseline | Design Decision |
| PPO-based RLHF | Not part of baseline | Design Decision |
| DPO preference objective | Not part of baseline | Design Decision |
| Optional explanation/ranking feature | Not established | Open Question |
| Preference rubric and inter-rater agreement | Required only if preference feature is pursued | Open Question |
| DPO policy/reference setup and beta | Not chosen or trained | Open Question |
| RLHF reward model/critic/KL details | Not chosen or trained | Open Question |
| Exact Qwen checkpoint / tokenizer / image processing | Not established | Open Question |
| Exact LoRA ranks, optimizer and LR schedule | Deferred training configuration | Open Question |
| Held-out metrics and measured effect of any theoretical adaptation | No results established in these notes | Open Question |
| Production-serving architecture and system costs | Not established | Open Question |

---

# Part 4 — Key Equations and What They Mean

## Bradley–Terry preference probability

$$
P(y_w\succ y_l\mid x)
=
\sigma(r(x,y_w)-r(x,y_l))
$$

The reward difference—not the absolute reward offset—determines the pairwise win probability in this model.

## Reward-model loss

$$
\mathcal L_{\mathrm{RM}}
=
-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

This fits reward scores from preferred/rejected pairs.

## KL-regularized policy objective

$$
\max_\theta
\mathbb E_{\pi_\theta}[r(x,y)]
-
\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

This schematic objective trades reward against deviation from the reference, with implicit prompt conditioning.

## PPO clipped surrogate

$$
L_{\mathrm{clip}}
=
\mathbb E_t
\min\left(
\rho_tA_t,\;
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
$$

This discourages excessive policy probability-ratio updates to the sampled actions in the surrogate objective.

## DPO loss

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

This uses the policy/reference-relative preference margin directly from response log probabilities.

## DPO margin

$$
\Delta
=
\left[\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)\right]
-
\left[\log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)\right]
$$

A larger positive $\Delta$ means the trainable policy favors the chosen completion more relative to the rejected completion than the reference does.

---

# Part 4 — Final Mental Model

~~~text
SUPERVISED FINE-TUNING
"Here is the correct response."
       ↓
MODEL CAN FOLLOW BASIC INSTRUCTIONS
       ↓
SOME TASKS HAVE SEVERAL PLAUSIBLE RESPONSES
       ↓
HUMAN PREFERENCE PAIRS
"For the same prompt, response A is better than B."
       |
       +--> REWARD MODEL
       |       ↓
       |    RLHF / PPO WITH REFERENCE AND REWARD
       |
       +--> DIRECT PREFERENCE OPTIMIZATION
               DPO, using policy and reference log probabilities
       ↓
INDEPENDENT EVALUATION
Did actual task quality and human preference improve?
       ↓
CHOOSE ONLY IF THE METHOD SOLVES A REAL GAP
~~~

**Main Qwen decision:** Binary Little Content labels already support direct supervised learning. Preference optimization is **not a mandatory stage** of the proposed Qwen classifier and should not be claimed as performed.

# One-Line Summary

> **SFT teaches a model which responses to imitate; preference learning teaches which candidate responses are preferred; reward-model RLHF and DPO provide different ways to learn from comparisons, but the correct engineering choice depends on the actual task.**

---

# Bridge to the Next Study Unit — Alignment, Safety and Training Engineering

Part 4 focused on preference signals and optimization. The remaining curriculum will examine how to turn those ideas into robust training and evaluation practices: alignment/safety requirements, training configurations, data leakage, generalization and reliable task-level assessment.

Those belong to subsequent parts rather than silently expanding the preference-learning syllabus here.
