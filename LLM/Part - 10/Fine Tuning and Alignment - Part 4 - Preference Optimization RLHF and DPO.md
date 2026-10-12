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

