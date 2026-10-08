# Part 10 — Fine-Tuning and Alignment

# Fine-Tuning and Alignment — Part 2: SFT Foundations (Questions 1–11)

## 10.2 Supervised Fine-Tuning (SFT)

**Detailed Study Mode. Interview-style material remains separate.**

Part 1 established **why** fine-tuning is a reasonable tool for the Qwen Little Content project.

Part 2 now asks:

> **How are supervised examples represented, aligned, and scored during SFT?**

This file covers Questions 1–11: multimodal SFT examples, teacher forcing, token loss, assistant-only masking, and controlled class scores. Questions 12–25 have moved to the companion file for full fine-tuning, PEFT, LoRA, QLoRA and the training loop.

**Continue to:** [Part 2B — PEFT and LoRA](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20PEFT%20and%20LoRA.md).

Our fixed learning pattern remains:

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

# Qwen Project — End-to-End Roadmap Snapshot

This roadmap stays visible so that we always know what is already designed and what is intentionally deferred.

| Project layer | Current status |
|---|---|
| Business problem | Defined in Part 1 |
| Why Qwen / why a VLM | Defined in Part 1 |
| Why fine-tuning | Defined in Part 1 |
| Prompt-only baseline | Defined in Part 1 |
| **SFT training formulation** | **Designed in this Part** |
| **Loss construction** | **Designed in this Part** |
| **Full FT vs PEFT / LoRA** | Part 2B (companion file) |
| Multimodal chat template | Part 3 |
| Image preprocessing details | Part 3 |
| Preference tuning / RLHF / DPO decision | Part 4 |
| Safety / behavioral constraints | Part 5 |
| Dataset sampling + optimizer + LR + scheduler | Part 6 |
| Overfitting + leakage + generalization | Part 7 |
| Evaluation + thresholding + calibration | Part 8 |
| Deployment + 50K/day serving | Part 8 |
| Failure handling + monitoring + drift | Part 8 |

---

# Starting Project State From Part 1

~~~text
Task
Little Content detection

Input
Bing SERP screenshot
+
task instruction

Base model
Pretrained Qwen VLM

Desired output
Little Content
or
Not Little Content

Why fine-tune?
Prompting is the baseline,
but persistent task-specific errors
justify adaptation

Generalization goal
Unseen hosts / layouts / templates
~~~

The next question is:

> **What exactly happens during supervised training?**

---

### Story Bridge 1 — We Have Labels, but a Label Alone Does Not Train a Language Model

Part 1 ended with labelled screenshot pairs:

$$ (x_i, y_i) $$

But a generative VLM does not consume the abstract phrase "binary classification dataset."

It consumes a multimodal context and learns to predict target tokens.

## Question 1 — What is Supervised Fine-Tuning?

Supervised Fine-Tuning, or **SFT**, adapts a pretrained model using examples containing:

1. An input or context.
2. A desired target response.

A generic SFT dataset is:

$$
\mathcal{D}_{\\mathrm{SFT}} = \\{ (x_i, y_i) \\}_{i=1}^{N}
$$


where:

- $x_i$ is the input.
- $y_i$ is the desired output.

For a generative model, we want to maximize:

$$
P_\theta(y_i\mid x_i)
$$

Equivalently, we minimize negative log-likelihood:

$$
\mathcal{L}_{\mathrm{SFT}} = -
\sum_{i=1}^{N}
\log
P_\theta
\left(
y_i\mid x_i
\right)
$$

If:

$$
y_i =
\left(
y_{i,1},
y_{i,2},
\ldots,
y_{i,T_i}
\right)
$$

then:

$$
P_\theta(y_i \mid x_i) = \prod_{t=1}^{T_i} P_\theta \left( y_{i,t} \mid x_i, y_{i,\lt t} \right)
$$



and:

$$
\mathcal{L}_{\mathrm{SFT}} = - \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta \left( y_{i,t} \mid x_i, y_{i,<t} \right)
$$


### Intuition

SFT is learning from demonstrations:

~~~text
Here is the input.
Here is the response I want.
        ↓
Repeat across many examples.
        ↓
Adjust parameters so desired responses
become more probable.
~~~

### Qwen Project Application

For one example:

~~~text
Input:
SERP screenshot
+
classification instruction

Desired response:
Little Content
~~~

For another:

~~~text
Input:
SERP screenshot
+
classification instruction

Desired response:
Not Little Content
~~~

### Design Decision

The Qwen project will use **generative SFT** as the primary training formulation.

The exact chat serialization is deferred to Part 3.

---

### Story Bridge 2 — SFT Needs an Actual Training Example, Not Just an Abstract Pair

For a multimodal chat model, one example contains several semantic pieces:

- Image.
- Instruction.
- Assistant response.
- Potential system context.

We need to separate what conditions the model from what we want it to predict.

## Question 2 — What does one Qwen SFT example contain?

Conceptually:

~~~text
SYSTEM
You are a page-quality classifier.

USER
<image>
Classify this SERP screenshot as
Little Content or Not Little Content.

ASSISTANT
Little Content
~~~

The conditioning context is:

$$
x =
\{
\text{image},
\text{system instruction},
\text{user instruction}
\}
$$

The supervised target is:

$$
y =
\text{"Little Content"}
$$

So the task is:

$$
P_\theta
\left(
\text{"Little Content"}
\mid
\text{image},
\text{instruction}
\right)
$$

### Qwen Project Application

A training row should eventually contain at least:

~~~text
Image reference
Task instruction
Ground-truth label
~~~

Metadata such as host, URL, annotation source, and template identity may be useful for splitting and analysis, but should not automatically become model input.

### Design Decision

Each project example will conceptually contain:

$$
(\text{SERP image},\text{task instruction},\text{canonical target label})
$$

The exact Qwen chat format is deferred to Part 3.

---

### Story Bridge 3 — The Model Is Multimodal, So the Screenshot Must Enter the Language Model Somehow

Text becomes tokens.

A screenshot is not naturally a text-token sequence.

So we need a conceptual picture of how visual evidence reaches the generative model.

## Question 3 — How does a VLM conceptually combine image and text during SFT?

A generic VLM pipeline is:

~~~text
Screenshot
    ↓
Vision encoder
    ↓
Visual features
    ↓
Projector / connector
    ↓
Language-model embedding space

Text instruction
    ↓
Tokenizer
    ↓
Text embeddings

Visual + text representations
    ↓
Language model
    ↓
Assistant-token predictions
~~~

Let the vision encoder produce:

$$
V =
\left[
v_1,
v_2,
\ldots,
v_m
\right]
$$

where:

$$
v_j
\in
\mathbb{R}^{d_v}
$$

A projector can map a visual feature to the language-model hidden dimension:

$$
\tilde{v}_j =
W_pv_j
$$

with:

$$
W_p \in \mathbb{R}^{d_{\mathrm{model}}\times d_v}
$$

Different VLMs vary in how they implement image-token compression, connectors, image placeholders, cross-attention, and positional representations.

### Qwen Project Application

The screenshot provides evidence.

The instruction defines the task.

The assistant response supplies the supervised target.

### Design Decision

Until we select the exact Qwen variant, we will reason using:

$$
\text{Vision encoder}
\rightarrow
\text{multimodal connector}
\rightarrow
\text{language model}
$$

as the conceptual architecture.

---

### Story Bridge 4 — Once the Context Is Inside the Model, We Must Align Inputs With Next-Token Targets

A causal language model always learns one basic task:

> Given everything visible so far, what token comes next?

SFT can make this confusing because the complete correct response is already present in the training batch. The important point is that the causal mask prevents an earlier position from seeing future tokens.

So the full sequence can be present physically, while each prediction is still made only from the allowed prefix.

## Question 4 — How are next-token predictions aligned during SFT?

Let us rebuild this from a simple language-model example before returning to Qwen.

### Step 1 — Start with a plain sentence

Suppose the training sequence is:

~~~text
I | love | pizza
~~~

The model is not trained to copy the same token at the same position.

Instead:

~~~text
Given "I"
→ predict "love"

Given "I love"
→ predict "pizza"
~~~

Mathematically:

$$
P(\text{love}\mid\text{I})
$$

and:

$$
P(\text{pizza}\mid\text{I,love})
$$

That is next-token prediction.

---

### Step 2 — What enters the Transformer?

The complete known training sequence can be placed in one tensor:

~~~text
Position      0       1        2
Token         I      love     pizza
~~~

The Transformer produces a hidden state and a vocabulary-logit vector at every position.

For position $t$:

$$
z_t
\in
\mathbb{R}^{V}
$$

where $V$ is vocabulary size.

The crucial alignment rule is:

$$
\boxed{
\text{logits at position }t
\rightarrow
\text{target token at position }t+1
}
$$

Therefore:

| Logits produced from prefix ending at | Correct next-token target |
|---|---|
| I | love |
| I love | pizza |

The hidden state at the love position does not predict love. It predicts pizza.

---

### Step 3 — Why can the model not cheat?

Because of the causal mask.

At the position containing I:

~~~text
Visible:
I

Hidden:
love
pizza
~~~

So:

$$
h_0=f(\text{I})
$$

and its logits represent:

$$
P(\text{next token}\mid\text{I})
$$

The correct target is love.

At the position containing love:

~~~text
Visible:
I
love

Hidden:
pizza
~~~

So:

$$
h_1=f(\text{I,love})
$$

and its logits represent:

$$
P(\text{next token}\mid\text{I,love})
$$

The correct target is pizza.

Therefore the full sequence can exist in memory without future leakage.

---

### Step 4 — What does the one-position shift mean?

Suppose the input sequence is:

~~~text
A | B | C | D
~~~

The model produces:

~~~text
logits[0]
logits[1]
logits[2]
logits[3]
~~~

For next-token training, the useful comparisons are:

~~~text
logits[0] → B
logits[1] → C
logits[2] → D
~~~

So the target sequence is shifted one position to the left relative to the input positions used to produce the logits.

For a sequence of length $L$:

$$
\text{prediction positions} =
0,\ldots,L-2
$$

and:

$$
\text{target positions} =
1,\ldots,L-1
$$

Many causal-language-model training APIs perform this logits-versus-label shift internally.

---

### Step 5 — Concrete token-ID example

Suppose:

~~~text
Token        ID

I            10
love         25
pizza        91
EOS           2
~~~

Then:

~~~text
input_ids =
[10, 25, 91, 2]
~~~

The model produces:

$$
z_0,z_1,z_2,z_3
$$

The useful training pairs are:

~~~text
z0 → target 25 = love
z1 → target 91 = pizza
z2 → target 2  = EOS
~~~

Equivalently:

~~~text
Input prefix          Correct next token

I                  →  love
I love             →  pizza
I love pizza       →  EOS
~~~

---

# Now Apply the Same Idea to the Qwen Little Content Project

Suppose one training example is conceptually:

~~~text
USER:
<image>
Classify this page.

ASSISTANT:
Little Content
~~~

Let all screenshot and prompt context be represented by:

$$
x
$$

Suppose the assistant response tokenizes as:

$$
y_1=\text{Little}
$$

$$
y_2=\text{Content}
$$

and:

$$
y_3=\text{EOS}
$$

Then the full known training sequence is:

~~~text
[ screenshot + instruction ] | Little | Content | EOS
             x                   y1       y2       y3
~~~

### Prediction 1 — Little

The prefix is only:

$$
x
$$

So the model predicts:

$$
P_\theta(y_1\mid x)
$$

or:

$$
P_\theta
\left(
\text{Little}
\mid
\text{screenshot + instruction}
\right)
$$

Little itself is still a future token relative to the position whose logits make this prediction.

### Prediction 2 — Content

Now Little is part of the visible prefix.

So:

$$
P_\theta(y_2\mid x,y_1)
$$

or:

$$
P_\theta
\left(
\text{Content}
\mid
\text{screenshot + instruction + Little}
\right)
$$

Content itself is still hidden from that prediction position.

### Prediction 3 — EOS

After Little Content is visible:

$$
P_\theta
\left(
y_3
\mid
x,y_1,y_2
\right)
$$

or:

$$
P_\theta
\left(
\text{EOS}
\mid
\text{screenshot + instruction + Little + Content}
\right)
$$

Therefore:

$$
\boxed{ P_\theta(y_1,y_2,y_3\mid x) =
P_\theta(y_1\mid x)
P_\theta(y_2\mid x,y_1)
P_\theta(y_3\mid x,y_1,y_2)
}
$$

---

### Step 6 — The diagram to remember

~~~text
KNOWN TRAINING SEQUENCE

[ screenshot + instruction ] [ Little ] [ Content ] [ EOS ]
             x                 y1         y2        y3


PREFIX USED                    NEXT TOKEN TARGET

x
↓
predict y1 = Little


x + Little
↓
predict y2 = Content


x + Little + Content
↓
predict y3 = EOS
~~~

This is the cleanest mental model for SFT alignment.

---

### Step 7 — Why is Little both an input token and a target token?

Because it plays two different roles at two different positions.

Little is:

1. The target predicted from the previous position.
2. Part of the context used by the next position to predict Content.

Conceptually:

~~~text
Previous prefix
      ↓
predict Little

Then Little becomes part of the known prefix
      ↓
predict Content
~~~

There is no circular prediction.

The model never sees Little at a position and then uses that same position to predict Little.

That is exactly why the one-position shift exists.

---

### Step 8 — Why would training be wrong without the shift?

If we compared the logits produced at the Little position against the target Little, the hidden state at that position would already contain Little.

That would not be the desired next-token objective.

The correct relation is:

$$
\boxed{
z_t
\rightarrow
\text{token at position }t+1
}
$$

not:

$$
z_t
\rightarrow
\text{token at position }t
$$

---

### Step 9 — How does assistant-only masking fit into this?

Suppose the sequence is:

~~~text
USER | Classify | page | ASSISTANT | Little | Content | EOS
~~~

We want the user prompt to condition the response, but we may not want direct target loss on those prompt tokens.

Conceptually:

~~~text
Input:

USER  Classify  page  ASSISTANT  Little  Content  EOS


Labels:

-100   -100     -100    -100      Little  Content  EOS
~~~

Here the ignored positions mean:

> Use these tokens as context, but do not directly supervise them as assistant targets.

After the normal next-token shift, the important comparisons are:

~~~text
logits after ASSISTANT
        ↓
target Little


logits after Little
        ↓
target Content


logits after Content
        ↓
target EOS
~~~

So **label shifting** and **assistant-only loss masking** solve different problems and work together.

---

### Step 10 — Keep these three objects separate

#### Object 1 — Input sequence

During training:

~~~text
Prompt + ground-truth assistant response
~~~

is available.

#### Object 2 — Logits

At each position:

$$
z_t\in\mathbb{R}^{V}
$$

contains scores for the **next** token.

#### Object 3 — Labels

The labels tell us which next token was actually correct.

So remember:

~~~text
input at position t
        ↓
hidden state at t
        ↓
logits at t
        ↓
compare with target from position t+1
~~~

or:

$$
\boxed{
\text{Input at }t
\rightarrow
\text{logits at }t
\rightarrow
\text{target at }t+1
}
$$

---

### Step 11 — How can training still be parallel?

Logically:

$$
P(y_1,y_2,y_3\mid x) =
P(y_1\mid x)
P(y_2\mid x,y_1)
P(y_3\mid x,y_1,y_2)
$$

At inference time:

~~~text
y1 does not exist yet
→ generate y1

then y2 does not exist yet
→ generate y2

then y3 does not exist yet
→ generate y3
~~~

So inference is sequential.

During SFT training, however, the ground-truth values:

$$
y_1,y_2,y_3
$$

already exist in the dataset.

Therefore:

$$
[x,y_1,y_2,y_3]
$$

can be processed in one Transformer forward pass.

The causal mask ensures:

~~~text
Position predicting y1
can see x
cannot see y1, y2, y3

Position predicting y2
can see x, y1
cannot see y2, y3

Position predicting y3
can see x, y1, y2
cannot see y3
~~~

So the model can calculate all of the required next-token distributions during the same training forward pass.

This leads directly into **teacher forcing**, which we discuss in Question 5.

---

### Qwen Project Application

For our Little Content example:

~~~text
Screenshot + instruction
        ↓
predict Little

Screenshot + instruction + Little
        ↓
predict Content

Screenshot + instruction + Little + Content
        ↓
predict EOS
~~~

The full correct response is present during training, but the causal mask determines which part of that response each prediction position is allowed to use.

### Design Decision

We retain the standard autoregressive causal-LM objective.

The alignment rule to remember is:

$$
\boxed{
\text{logits at position }t
\text{ predict the token at position }t+1
}
$$

For the Qwen project:

> **The full response is available during SFT, but every supervised prediction is still made only from the valid causal prefix.**

---

### Story Bridge 5 — During Training the Correct Assistant Tokens Already Exist

Question 4 showed us how next-token prediction is aligned.

Now we need one more idea:

> During training, when the model predicts the next token, what previous assistant tokens should it use?

The answer is **teacher forcing**.

## Question 5 — What is teacher forcing during SFT?

Teacher forcing means:

> **During training, the model is given the correct previous target tokens from the training data while it learns to predict the next token.**

The model does **not** use its own previous prediction as the next training input.

---

### Start with our Qwen example

Suppose the correct assistant response is:

~~~text
Little Content
~~~

Assume it tokenizes as:

~~~text
y1 = Little
y2 = Content
~~~

Let:

$$
x
$$

represent the screenshot plus the task instruction.

For the first token, there is no previous assistant token yet.

So the model learns:

$$
P_\theta
\left(
y_1\mid x
\right)
$$

or:

$$
P_\theta
\left(
\text{Little}\mid x
\right)
$$

Conceptually:

~~~text
Screenshot + instruction
        ↓
predict "Little"
~~~

---

### Now the important part

Next, the model has to learn to predict:

~~~text
Content
~~~

During teacher forcing, we give it the **correct previous token**:

~~~text
Little
~~~

from the training dataset.

So the model learns:

$$
P_\theta
\left(
y_2
\mid
x,y_1^{\text{true}}
\right)
$$

or:

$$
P_\theta
\left(
\text{Content}
\mid
x,\text{Little}
\right)
$$

Conceptually:

~~~text
Screenshot + instruction + TRUE Little
        ↓
predict "Content"
~~~

The important word is:

> **TRUE**

The previous token comes from the labelled training example.

---

### Why is this called teacher forcing?

Think of a teacher correcting the model at every step.

Suppose the correct sequence is:

~~~text
Little → Content
~~~

But imagine that, early in training, the model would have predicted:

~~~text
Small
~~~

instead of:

~~~text
Little
~~~

Without teacher forcing, the next input could become:

~~~text
Screenshot + instruction + Small
        ↓
predict next token
~~~

Now the model has already moved away from the correct training sequence.

With teacher forcing, we ignore that wrong generated token for the purpose of the next training position.

The teacher says:

~~~text
The correct previous token was "Little".

Use "Little" as your history.

Now predict the next token.
~~~

So training remains:

~~~text
Screenshot + instruction
        ↓
target = Little


Screenshot + instruction + TRUE Little
        ↓
target = Content
~~~

That is teacher forcing.

---

### Mathematical notation

In general, when predicting target token $y_t$, the model conditions on:

$$
x
$$

and the correct earlier target tokens:

$$
y_{\lt t}^{\text{true}} = \left( y_1, y_2, \ldots, y_{t-1} \right)
$$


So the teacher-forced prediction is:

$$
\begin{array}{|c|}
\hline
P_\theta \left( y_t \mid x, y_{\lt t}^{\text{true}} \right) \\
\hline
\end{array}
$$


where:

- $x$ = original input, such as screenshot + instruction.
- $y_t$ = token currently being predicted.
- $y_{<t}^{\text{true}}$ = all correct target tokens before $y_t$.

---

### Training versus inference

This is the key distinction.

#### During training

The correct answer already exists in the dataset.

So:

~~~text
Screenshot + prompt
        ↓
predict Little

Screenshot + prompt + TRUE Little
        ↓
predict Content

Screenshot + prompt + TRUE Little + TRUE Content
        ↓
predict EOS
~~~

The model always receives the **correct previous target tokens**.

#### During inference

The correct answer is not available.

The model has to use what it generated itself:

~~~text
Screenshot + prompt
        ↓
model generates Little

Screenshot + prompt + GENERATED Little
        ↓
model generates Content
~~~

So the simplest comparison is:

~~~text
TRAINING
uses ground-truth previous tokens

INFERENCE
uses model-generated previous tokens
~~~

---

### What if the model makes a mistake during inference?

Suppose the correct answer should be:

~~~text
Little → Content
~~~

but the model generates:

~~~text
Small
~~~

as its first token.

During inference, the next step must continue from:

~~~text
Screenshot + prompt + Small
~~~

because there is no teacher available to replace Small with Little.

This is different from training, where the correct history is always available.

That is one reason training is easier than inference.

---

### Why can teacher-forced training be parallel?

The logical probability of a target sequence is still autoregressive:

$$
P_\theta \left( y_1, y_2, y_3 \mid x \right) = P_\theta \left( y_1 \mid x \right) P_\theta \left( y_2 \mid x, y_1 \right) P_\theta \left( y_3 \mid x, y_1, y_2 \right)
$$


That looks sequential.

But during training:

$$
y_1,y_2,y_3
$$

are already known.

So the whole sequence can be placed in one training input:

$$
[x,y_1,y_2,y_3]
$$

The causal mask makes sure that each position only sees the correct prefix.

Conceptually:

~~~text
Position predicting y1
sees:
x

Position predicting y2
sees:
x + TRUE y1

Position predicting y3
sees:
x + TRUE y1 + TRUE y2
~~~

Because all the correct tokens already exist in the training data, the Transformer can compute these token positions in the same forward pass.

At inference time, the future tokens do not exist yet, so generation must happen one token at a time.

---

### Connection to Part 8

This connects directly to what we learned about training versus inference.

~~~text
TRAINING

Future ground-truth tokens are already known
        ↓
Teacher forcing provides correct history
        ↓
Token positions can be processed in parallel
under a causal mask
~~~

but:

~~~text
INFERENCE

Future output tokens are unknown
        ↓
Generate one token
        ↓
append it
        ↓
generate the next token
~~~

---

### Qwen Project Application

For our Little Content classifier:

~~~text
TRAINING

Screenshot + instruction
        ↓
predict Little

Screenshot + instruction + TRUE Little
        ↓
predict Content
~~~

During inference:

~~~text
Screenshot + instruction
        ↓
generate token 1

Screenshot + instruction + GENERATED token 1
        ↓
generate token 2
~~~

So the core idea is:

> **Teacher forcing means that training uses the correct previous answer tokens, while inference uses the model's own generated previous tokens.**

### Design Decision

The Qwen project uses standard teacher-forced causal SFT.

---

### Story Bridge 6 — Teacher Forcing Gives Predictions, but We Still Need a Numeric Error Signal

Teacher forcing tells us **what context the model should use** while predicting each target token.

But after the model makes a prediction, training still needs another thing:

> **A number that tells us how good or bad that prediction was.**

That number is the **loss**.

For generative SFT, the standard loss is token-level **cross-entropy**, which in this setting is also the negative log-likelihood of the correct next token.

## Question 6 — What loss does generative SFT use?

The easiest way to understand the loss is to follow one prediction from beginning to end.

---

### Step 1 — The model does not directly output one word

Suppose Qwen is trying to predict the next token.

It does not directly output:

~~~text
Little
~~~

Instead, it produces one raw score for **every token in the vocabulary**.

Suppose our toy vocabulary contains only:

~~~text
Little
Content
Not
Other
~~~

The model might produce raw scores such as:

~~~text
Little   = 3.2
Content  = 1.1
Not      = 0.4
Other    = -0.2
~~~

These raw scores are called **logits**.

At supervised position $t$, we can write the full vocabulary-logit vector as:

$$
\mathbf{z}_t =
\left[
z_{t,1},
z_{t,2},
\ldots,
z_{t,V}
\right]
\in
\mathbb{R}^{V}
$$

where:

- $t$ = the current supervised prediction position.
- $V$ = vocabulary size.
- $z_{t,k}$ = the raw score assigned to vocabulary token $k$.

So the first important interview point is:

> **At every supervised position, the model produces one logit for every token in its vocabulary.**

---

### Step 2 — Logits are not probabilities

A logit can be any real number:

~~~text
3.2
1.1
0.4
-0.2
~~~

So we cannot directly interpret the logits as probabilities.

We apply **softmax**:

$$
P_\theta \left( k \mid c_t \right) = \frac{e^{z_{t,k}}}{\sum_{j=1}^{V} e^{z_{t,j}}}
$$


where:

- $k$ = a possible vocabulary token.
- $c_t$ = the context available when predicting at position $t$.
- $\theta$ = the model parameters.

Softmax converts all the logits into probabilities that add up to 1.

For example:

~~~text
Little   → 0.80
Content  → 0.10
Not      → 0.07
Other    → 0.03
~~~

and:

$$
0.80+0.10+0.07+0.03 = 1
$$

---

### Step 3 — What does $c_t$ mean?

The notation:

$$
c_t
$$

simply means:

> **Everything the model is allowed to see when making prediction $t$.**

For our Qwen project, if the model is predicting the first assistant token:

~~~text
Little
~~~

then the context might be:

~~~text
SERP screenshot
+
task instruction
+
assistant-start context
~~~

If the model is predicting:

~~~text
Content
~~~

teacher forcing means the context also contains the correct previous token:

~~~text
Little
~~~

So:

$$
P_\theta \left( y_t \mid c_t \right)
$$


means:

> **What probability did the model give to the correct next token, given the context available at that position?**

---

### Step 4 — Compare the prediction with the correct token

Suppose the correct next token is:

~~~text
Little
~~~

and the model predicts:

~~~text
Little      0.80   ← correct token
Content     0.10
Not         0.07
Other       0.03
~~~

For the loss, the most important value is:

$$
P_\theta \left( \text{Little} \mid c_t \right) = 0.8
$$


In simple terms:

> We ask how much probability the model assigned to the token that was actually correct.

If the correct token gets high probability, the prediction is good.

If it gets very low probability, the prediction is poor.

---

### Step 5 — Convert the correct-token probability into a loss

The token-level loss is:

$$
\begin{array}{|c|}
\hline
\mathcal{L}_t = -\log P_\theta \left( y_t \mid c_t \right) \\
\hline
\end{array}
$$


where $y_t$ is the correct target token.

#### Good prediction

Suppose:

$$
P_\theta \left( y_t \mid c_t \right) = 0.8
$$


Then:

$$
\mathcal{L}_t = -\log(0.8) \approx 0.223
$$


The loss is small.

#### Bad prediction

Suppose:

$$
P_\theta \left( y_t \mid c_t \right) = 0.1
$$


Then:

$$
\mathcal{L}_t =
-\log(0.1)
\approx
2.303
$$

The loss is much larger.

So:

~~~text
High probability on the correct token
        ↓
Small loss

Low probability on the correct token
        ↓
Large loss
~~~

---

### Step 6 — Why do we use the negative logarithm?

The negative log gives us a useful penalty curve.

| Probability assigned to the correct token | Loss |
|---:|---:|
| $0.99$ | $\approx 0.01$ |
| $0.80$ | $\approx 0.22$ |
| $0.50$ | $\approx 0.69$ |
| $0.10$ | $\approx 2.30$ |
| $0.01$ | $\approx 4.61$ |

So when:

$$
P_\theta \left( y_t \mid c_t \right) \rightarrow 1
$$


the loss approaches:

$$
0
$$

But when:

$$
P_\theta \left( y_t \mid c_t \right) \rightarrow 0
$$


the loss becomes very large.

A simple interview explanation is:

> **We use negative log probability because correct high-confidence predictions get a small loss, while low probability on the correct token gets a large penalty.**

---

# Now Apply It to the Qwen Little Content Example

Suppose the correct response is:

~~~text
Little Content
~~~

and assume:

~~~text
y1 = Little
y2 = Content
y3 = EOS
~~~

Teacher forcing tells us what context each prediction uses.

Cross-entropy tells us how wrong each prediction was.

### Prediction 1 — Little

Context:

~~~text
Screenshot + instruction
~~~

Correct next token:

~~~text
Little
~~~

Suppose:

$$
P_\theta \left( \text{Little} \mid x \right) = 0.8
$$


Then:

$$
\mathcal{L}_1 =
-\log(0.8)
\approx
0.223
$$

### Prediction 2 — Content

Because we are using teacher forcing, the context contains the **correct** previous token:

~~~text
Screenshot + instruction + TRUE Little
~~~

Suppose:

$$
P_\theta \left( \text{Content} \mid x, \text{Little} \right) = 0.6
$$

Then:

$$
\mathcal{L}_2 =
-\log(0.6)
\approx
0.511
$$

### Prediction 3 — EOS

Now the context contains:

~~~text
Screenshot + instruction + TRUE Little + TRUE Content
~~~

Suppose:

$$
P_\theta \left( \text{EOS} \mid x, \text{Little}, \text{Content} \right) = 0.9
$$


Then:

$$
\mathcal{L}_3 =
-\log(0.9)
\approx
0.105
$$

---

### Step 7 — How do we get one loss for the whole response?

We now have three token losses:

$$
\mathcal{L}_1
\approx
0.223
$$

$$
\mathcal{L}_2
\approx
0.511
$$

$$
\mathcal{L}_3
\approx
0.105
$$

For $T$ supervised assistant tokens, a common example-level objective is the mean token loss:

$$
\begin{array}{|c|}
\hline
\mathcal{L}_{\mathrm{example}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta \left( y_t \mid c_t \right) \\
\hline
\end{array}
$$


For this example:

$$
\mathcal{L}_{\mathrm{example}} =
\frac{
0.223+0.511+0.105
}{3}
$$

so:

$$
\mathcal{L}_{\mathrm{example}}
\approx
0.280
$$

Now we have **one scalar number** that represents how well the model predicted this assistant response.

---

### Step 8 — Why do we need one scalar loss?

Training needs an objective that can be minimized.

The model may produce millions of logits, but the optimizer needs a scalar objective:

$$
\mathcal{L}
$$

Then backpropagation computes gradients such as:

$$
\nabla_\theta
\mathcal{L}
$$

or, in our PEFT setup, with respect to trainable adapter parameters:

$$
\nabla_\phi
\mathcal{L}
$$

Those gradients tell the optimizer how the trainable parameters should change to reduce the loss.

The complete flow is:

~~~text
Context
        ↓
Qwen
        ↓
Vocabulary logits
        ↓
Softmax
        ↓
Vocabulary probabilities
        ↓
Probability of the correct token
        ↓
Negative log
        ↓
Token loss
        ↓
Average over supervised assistant tokens
        ↓
One scalar loss
        ↓
Backpropagation
        ↓
Update trainable parameters
~~~

---

### Step 9 — What does training actually improve?

Suppose before training:

$$
P_\theta \left( \text{Little} \mid x \right) = 0.20
$$


Then:

$$
-\log(0.20)
\approx
1.609
$$

After some training, suppose:

$$
P_\theta \left( \text{Little} \mid x \right) = 0.70
$$


Then:

$$
-\log(0.70)
\approx
0.357
$$

The loss has gone down.

That means the model has learned to assign more probability to the correct token for that kind of context.

---

### Step 10 — Cross-entropy and negative log-likelihood

An interviewer may ask:

> Are cross-entropy and negative log-likelihood the same thing here?

For standard next-token training, the target token can be represented as a one-hot distribution.

Suppose the toy vocabulary is:

~~~text
Little   Content   Not   Other
~~~

and the correct token is Little.

Then the target distribution is:

~~~text
Little   Content   Not   Other

1        0         0     0
~~~

The general cross-entropy is:

$$
\mathcal{L} = -
\sum_{k=1}^{V}
q_k
\log
p_k
$$

where:

- $q_k$ = target probability for token $k$.
- $p_k$ = model-predicted probability for token $k$.

Because only the correct token has:

$$
q_k=1
$$

all the other terms become zero.

Therefore:

$$
\mathcal{L} =
-\log
p_{\mathrm{correct}}
$$

That is exactly the negative log-likelihood of the correct token.

So for standard next-token SFT, **cross-entropy loss** and **negative log-likelihood of the correct token** describe the same token-level objective.

---

### Step 11 — Are we directly penalizing every wrong token?

Not separately.

The loss mainly asks:

> **How much probability did the correct token receive?**

For example:

~~~text
Before training:

Little                     0.20
all other tokens together  0.80
~~~

After training:

~~~text
Little                     0.75
all other tokens together  0.25
~~~

Because softmax probabilities must sum to 1, increasing probability on the correct token naturally reduces the total probability available to other tokens.

---

### Step 12 — Connection to teacher forcing

Question 5 and Question 6 fit together directly.

Teacher forcing answers:

> **What context should the model use while predicting token $y_t$?**

Cross-entropy answers:

> **Given that context, how wrong was the model's prediction for $y_t$?**

For our project:

~~~text
Screenshot + instruction
        ↓
Teacher-forced context
        ↓
Qwen predicts vocabulary probabilities
        ↓
Correct token = Little
        ↓
Loss = negative log probability of Little


Screenshot + instruction + TRUE Little
        ↓
Teacher-forced context
        ↓
Qwen predicts vocabulary probabilities
        ↓
Correct token = Content
        ↓
Loss = negative log probability of Content
~~~

So:

~~~text
Teacher forcing
        ↓
Correct context
        ↓
Next-token probability distribution
        ↓
Cross-entropy loss
        ↓
Backpropagation
        ↓
Parameter update
~~~

---

### Interview Explanation

A simple interview explanation is:

> **In generative SFT, the model produces a score, called a logit, for every token in the vocabulary at each supervised position. Softmax converts those logits into probabilities. We then look at the probability assigned to the correct next token. The token loss is the negative log of that probability. So if the correct token gets high probability, the loss is small; if it gets low probability, the loss is large. For a response with multiple supervised tokens, we calculate this loss for every assistant token and usually average them. That gives us one scalar loss, and backpropagation uses that loss to update the trainable parameters. In the Qwen example, for the answer "Little Content", we calculate the loss for predicting "Little", then the loss for predicting "Content" given the correct previous token "Little", and then the loss for EOS if EOS is supervised.**

If the interviewer asks:

> **Why negative log?**

A simple answer is:

> **Because it gives almost zero loss when the correct token has probability close to 1, and a large penalty when the correct token has very low probability.**

---

### Qwen Project Application

For the Little Content project:

~~~text
SERP screenshot + instruction
        ↓
Qwen vocabulary logits
        ↓
Softmax probabilities
        ↓
Probability of correct class-response token
        ↓
Token cross-entropy
        ↓
Average over supervised assistant tokens
        ↓
Backpropagation
        ↓
Update LoRA / other trainable parameters
~~~

The loss therefore pushes Qwen to assign more probability to the correct canonical Little Content response for screenshots with similar visual evidence.

### Design Decision

The primary SFT objective is:

> **Assistant-token cross-entropy / negative log-likelihood.**

For $T$ supervised assistant tokens:

$$
\begin{array}{|c|}
\hline
\mathcal{L}_{\mathrm{SFT}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta \left( y_t \mid c_t \right) \\
\hline
\end{array}
$$


Assistant-only loss masking, discussed in Question 7, determines **which token positions are included in this average**.

---

### Story Bridge 7 — The Sequence Contains Context Tokens We Do Not Want to Train the Model to Reproduce

Question 6 showed us how cross-entropy measures whether the model gave enough probability to the **correct next token**.

But an SFT example contains much more than the assistant answer.

It may contain:

- A system message.
- A user instruction.
- Image-related context.
- An assistant-start marker.
- The actual assistant response.

The model needs to **read** all of this context.

But we do not necessarily want to calculate supervised loss on all of it.

That leads to **loss masking**.

## Question 7 — Which tokens should contribute to the SFT loss?

A common instruction-SFT setup calculates loss only on the assistant-response tokens.

The basic idea is:

> **Context tokens help the model make the prediction, but assistant target tokens are the tokens we directly score with the supervised loss.**

This is called **assistant-only loss masking**.

---

### Step 1 — Start with one complete training example

Suppose the training conversation is:

~~~text
SYSTEM:
You are a page-quality classifier.

USER:
<image>
Classify this page.

ASSISTANT:
Little Content
~~~

Conceptually, the full sequence contains:

~~~text
SYSTEM TOKENS
+
USER TOKENS
+
IMAGE-RELATED CONTEXT
+
ASSISTANT START
+
Little
+
Content
+
EOS
~~~

All of these pieces may be needed to produce the correct answer.

But they do not all need to contribute directly to the supervised loss.

---

### Step 2 — What do we actually want the model to learn?

For this example, the task is:

~~~text
Given:

system instruction
+
SERP screenshot
+
user request

produce:

Little Content
~~~

We want the model to learn:

$$
P_\theta
\left(
\text{Little Content}
\mid
\text{system + screenshot + user instruction}
\right)
$$

We are **not** mainly trying to train the model to regenerate:

~~~text
You are a page-quality classifier.
~~~

or:

~~~text
Classify this page.
~~~

Those are the instructions that define the task.

They are input context.

The assistant response is the supervised output.

---

### Step 3 — Context versus target

This distinction is the key idea.

~~~text
SYSTEM MESSAGE
        ↓
Context

USER MESSAGE
        ↓
Context

IMAGE
        ↓
Context

ASSISTANT RESPONSE
        ↓
Supervised target
~~~

So:

> **The model should attend to the context, but the loss should be calculated on the assistant answer.**

This is why we need a loss mask.

---

### Step 4 — Define a loss mask

For every token position $t$, define:

$$
m_t
\in
\{0,1\}
$$

with:

$$
m_t = \begin{cases} 1, & \text{if position } t \text{ is a supervised assistant target token} \\ 0, & \text{if position } t \text{ is context or otherwise ignored} \end{cases}
$$


So:

~~~text
m_t = 1
→ include this token in the loss

m_t = 0
→ do not include this token in the loss
~~~

The masked loss is:

$$
\begin{array}{|c|}
\hline
\mathcal{L} = - \frac{\sum_t m_t \log P_\theta \left( y_t \mid c_t \right)}{\sum_t m_t} \\
\hline
\end{array}
$$

This looks complicated, but the meaning is simple:

> Calculate cross-entropy only for positions where the mask equals 1, then average over those supervised positions.

---

### Step 5 — Apply the mask to the Qwen example

Suppose the sequence is simplified to:

~~~text
SYSTEM | USER | IMAGE | ASSISTANT | Little | Content | EOS
~~~

A conceptual loss mask is:

~~~text
SYSTEM       → 0
USER         → 0
IMAGE        → 0
ASSISTANT    → 0
Little       → 1
Content      → 1
EOS          → 1
~~~

So:

~~~text
Context tokens
        ↓
used by the model
but not directly scored by supervised loss


Assistant answer tokens
        ↓
used by the model
and directly scored by supervised loss
~~~

---

### Step 6 — Very important: masked does NOT mean ignored by the model

This is one of the easiest points to misunderstand.

Suppose the user instruction is masked from the loss.

That does **not** mean the model cannot see it.

The model still attends to the instruction.

For example:

~~~text
USER:
Classify this page.
~~~

may have:

~~~text
attention = active
loss = masked
~~~

The model reads the instruction and uses it to predict the assistant answer.

We simply do not calculate a target loss saying:

~~~text
Please predict the token "Classify".
~~~

So:

> **Loss masking controls where we measure error. It does not necessarily control what the model can attend to.**

This distinction is crucial.

---

### Step 7 — Attention masking and loss masking are different

These two masks solve different problems.

#### Attention mask

Answers:

> Which positions are valid and available to participate in attention?

For a real prompt token:

~~~text
attention mask = 1
~~~

For padding:

~~~text
attention mask = 0
~~~

#### Loss mask

Answers:

> Which target positions should contribute to the supervised loss?

For a prompt token:

~~~text
loss mask = 0
~~~

For an assistant target token:

~~~text
loss mask = 1
~~~

So a user token can have:

~~~text
attention mask = 1
loss mask      = 0
~~~

That means:

> Read this token, but do not directly score the model on reproducing it.

We return to the attention-mask distinction again in Question 24.

---

### Step 8 — How does this connect to the one-token shift from Question 4?

This is important.

Suppose the simplified sequence is:

~~~text
USER | ASSISTANT | Little | Content | EOS
~~~

From Question 4, remember:

> Logits at one position predict the **next** token.

So:

~~~text
logits after ASSISTANT
        ↓
predict Little

logits after Little
        ↓
predict Content

logits after Content
        ↓
predict EOS
~~~

The assistant-only labels can therefore be represented conceptually as:

~~~text
Input tokens:

USER   ASSISTANT   Little   Content   EOS


Labels:

-100   -100        Little   Content   EOS
~~~

After the model's usual next-token alignment, the useful supervised comparisons are:

~~~text
logits after ASSISTANT
        ↓
target Little


logits after Little
        ↓
target Content


logits after Content
        ↓
target EOS
~~~

So **loss masking** and **next-token shifting** work together.

They are not the same thing.

---

### Step 9 — What does -100 mean?

Many causal-language-model training implementations use:

~~~text
-100
~~~

as an ignore value in the labels tensor.

For example:

~~~text
input_ids:

[system] [user] [assistant] [Little] [Content] [EOS]


labels:

[-100]  [-100] [-100]      [Little] [Content] [EOS]
~~~

The value:

~~~text
-100
~~~

does not mean that token ID -100 exists in the vocabulary.

It usually means:

> **Do not calculate cross-entropy loss for this label position.**

In common implementations, cross-entropy is configured with an ignore index, often -100.

So those positions are skipped when calculating the supervised loss.

---

### Step 10 — Numerical example

Suppose the assistant answer has three supervised tokens:

~~~text
Little
Content
EOS
~~~

and their losses are:

$$
\mathcal{L}_{\text{Little}} =
0.20
$$

$$
\mathcal{L}_{\text{Content}} =
0.50
$$

$$
\mathcal{L}_{\text{EOS}} =
0.10
$$

The prompt tokens are masked.

So the example loss is calculated only from those three assistant targets:

$$
\mathcal{L}_{\mathrm{example}} =
\frac{
0.20+0.50+0.10
}{3}
$$

Therefore:

$$
\mathcal{L}_{\mathrm{example}}
\approx
0.267
$$

The system prompt and user instruction do not add separate token-loss terms.

But they still influence these three predictions because they are part of the context.

---

### Step 11 — Why not calculate loss on the prompt as well?

We could train a causal language model on every token in the sequence.

But for instruction SFT, that may waste supervised capacity on predicting text that was **given to the model as input**.

For our task, we care most about:

~~~text
Given this screenshot and instruction
        ↓
produce the correct classification response
~~~

rather than:

~~~text
Given the beginning of the user instruction
        ↓
reconstruct the rest of the user instruction
~~~

Assistant-only masking focuses the supervised objective on the behavior we actually want.

---

### Step 12 — Does the image get a token loss?

Not in the same sense as the assistant text.

The screenshot is used as **conditioning information**.

Conceptually:

~~~text
SERP screenshot
        ↓
vision encoder
        ↓
visual representation
        ↓
Qwen uses that representation
to predict assistant tokens
~~~

The assistant token loss depends on the image.

So gradients can still flow through trainable parts of the visual/multimodal pathway even though we are not asking:

~~~text
What is the correct image token?
~~~

This connects directly to Question 8.

---

### Step 13 — Loss masking versus parameter freezing

Another important interview distinction:

~~~text
LOSS MASKING

asks:

Which prediction positions
should contribute to the loss?
~~~

while:

~~~text
PARAMETER FREEZING

asks:

Which model parameters
are allowed to update?
~~~

These are completely different decisions.

For example:

~~~text
User prompt
loss masked
        ↓
still affects prediction

LoRA adapter
trainable
        ↓
receives gradient from assistant loss

Frozen base weight
not updated
        ↓
still participates in forward pass
~~~

So:

> **Loss masking chooses the supervised positions. Freezing chooses the trainable parameters.**

---

### Step 14 — Full Qwen picture

For our project:

~~~text
SYSTEM
You are a page-quality classifier.
        ↓
context only


USER
<image>
Classify this page.
        ↓
context only


ASSISTANT
Little Content
        ↓
supervised target
~~~

Training conceptually becomes:

~~~text
System + user + image context
        ↓
Qwen
        ↓
predict Little
        ↓
calculate loss


System + user + image context
+ TRUE Little
        ↓
Qwen
        ↓
predict Content
        ↓
calculate loss


System + user + image context
+ TRUE Little + TRUE Content
        ↓
Qwen
        ↓
predict EOS
        ↓
calculate loss
~~~

But we do not calculate supervised target loss on the system/user context itself.

---

### Step 15 — Connection to Questions 4, 5, and 6

The pieces now fit together:

~~~text
QUESTION 4
Next-token alignment
        ↓
Which position predicts which token?


QUESTION 5
Teacher forcing
        ↓
Which previous target tokens are used as context?


QUESTION 6
Cross-entropy
        ↓
How wrong was each supervised prediction?


QUESTION 7
Loss masking
        ↓
Which token positions should be included
when calculating that loss?
~~~

This is the complete SFT supervision story so far.

---

### Interview Explanation

A simple interview answer is:

> **In instruction SFT, the input sequence contains both context tokens and assistant-response tokens. The model needs to read the system prompt, user instruction, and image context, but I usually do not want to calculate supervised loss on those context tokens. I only want to train the model on the assistant response. So I use a loss mask. Context positions get a mask value of 0, while assistant target positions get a value of 1. The model can still attend to the masked context; masking only means those positions do not directly contribute to cross-entropy. In many causal-LM implementations, ignored label positions are represented using -100. For my Qwen example, the screenshot and instruction are context, while the tokens for "Little Content" and possibly EOS are the supervised targets.**

If the interviewer asks:

> **Does masking the prompt mean the model cannot see the prompt?**

A simple answer is:

> **No. The prompt is still part of the input and the model attends to it. We are only saying that we do not calculate supervised loss on reproducing the prompt itself.**

If the interviewer asks:

> **What is the difference between loss masking and freezing?**

A simple answer is:

> **Loss masking decides which token positions contribute to the loss. Freezing decides which model parameters are allowed to update.**

---

### Qwen Project Application

For the Little Content project:

~~~text
SERP screenshot
+
system / user instruction
        ↓
CONTEXT
loss masked
but still visible to the model


Little
Content
EOS
        ↓
ASSISTANT TARGET
included in supervised loss
~~~

The screenshot and instruction tell Qwen **what to classify**.

The assistant tokens tell Qwen **what response should become more probable**.

### Design Decision

Our default theoretical design is **assistant-only loss masking**.

For supervised positions:

$$
m_t=1
$$

For context / ignored positions:

$$
m_t=0
$$

and the masked objective is:

$$
\begin{array}{|c|}
\hline
\mathcal{L} = - \frac{\sum_t m_t \log P_\theta \left( y_t \mid c_t \right)}{\sum_t m_t} \\
\hline
\end{array}
$$

In implementation, ignored textual label positions may commonly be represented by:

~~~text
-100
~~~

depending on the training framework and loss function configuration.

---

### Story Bridge 8 — A Masked Context Can Still Affect the Gradient

Question 7 introduced assistant-only loss masking.

That can create an easy misunderstanding:

> If the prompt and image do not receive their own token-level loss, does the model still learn from them?

Yes.

The key is to separate:

~~~text
Where the loss is measured
from
What information affects that loss
~~~

## Question 8 — If prompt and image positions are masked from the loss, can the model still learn from them?

Yes.

Questions 1–7 used $x$ for the full multimodal input context. For this gradient discussion, it is useful to split that input into:

$$
x =
\left(
I,
x_{\mathrm{text}}
\right)
$$

where:

- $I$ is the SERP screenshot.
- $x_{\mathrm{text}}$ is the textual context, such as the system message and user instruction.

For assistant token $t$, teacher forcing gives the causal context:

$$
c_t = \left( I, x_{\mathrm{text}}, y_{\lt t}^{\text{true}} \right)
$$


where $y_{<t}^{\mathrm{true}}$ contains the correct previous assistant tokens.

From Question 7, let:

$$
m_t
\in
\{0,1\}
$$

indicate whether token position $t$ contributes to the supervised loss.

The assistant-only SFT loss is therefore:

$$
\mathcal{L} = - \frac{1}{\sum_t m_t} \sum_t m_t \log P_\theta \left( y_t \mid I, x_{\mathrm{text}}, y_{\lt t}^{\text{true}} \right)
$$


Equivalently, using the shorter causal-context notation:

$$
\mathcal{L} = - \frac{1}{\sum_t m_t} \sum_t m_t \log P_\theta \left( y_t \mid c_t \right)
$$


The important point is that the image and prompt still appear inside the conditioning context.

So even though the prompt positions themselves have no direct token-level loss, changing the prompt or image can change the probability of the supervised assistant token and therefore change the loss.

---

### How can the image affect the gradient?

Let the vision encoder produce a visual representation:

$$
v =
f_{\theta_v}(I)
$$

where $\theta_v$ denotes the vision-encoder parameters.

Let the hidden state used to predict assistant token $t$ depend on the visual representation and the textual/assistant history:

$$
h_t = g_\theta \left( v, x_{\mathrm{text}}, y_{\lt t}^{\text{true}} \right)
$$

The probability of the correct assistant token depends on that hidden state:

$$
P_\theta \left( y_t \mid c_t \right) = P_\theta \left( y_t \mid h_t \right)
$$


Therefore, if the visual pathway is trainable, the chain rule gives a gradient path from the assistant loss back into the visual parameters:

$$
\frac{\partial \mathcal{L}}{\partial \theta_v} = \sum_t \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial v} \frac{\partial v}{\partial \theta_v}
$$


If we write the masked objective explicitly, the same idea is:

$$
\frac{\partial \mathcal{L}}{\partial \theta_v} = - \frac{1}{\sum_t m_t} \sum_t m_t \frac{\partial \log P_\theta \left( y_t \mid c_t \right)}{\partial h_t} \frac{\partial h_t}{\partial v} \frac{\partial v}{\partial \theta_v}
$$


So the gradient originates from supervised assistant-token loss, but it can flow through any trainable component that helped produce those assistant predictions.

---

### Qwen example

Suppose the correct response is:

~~~text
Little Content
~~~

For the first assistant token:

$$
\mathcal{L}_{\mathrm{Little}} = - \log P_\theta \left( \text{Little} \mid I, x_{\mathrm{text}} \right)
$$


For the second assistant token, teacher forcing adds the correct previous token:

$$
\mathcal{L}_{\mathrm{Content}} = - \log P_\theta \left( \text{Content} \mid I, x_{\mathrm{text}}, \text{Little} \right)
$$


If EOS is also supervised:

$$
\mathcal{L}_{\mathrm{EOS}} = - \log P_\theta \left( \text{EOS} \mid I, x_{\mathrm{text}}, \text{Little}, \text{Content} \right)
$$


The screenshot $I$ appears in every relevant conditional probability.

That is why the image can influence the loss even though we do not define a separate image-token cross-entropy target.

---

### What if the vision encoder is frozen?

If the vision encoder is frozen during optimization, its parameters remain unchanged:

$$
\theta_v^{(k+1)} = \theta_v^{(k)}
$$

The visual representation still enters the forward pass:

$$
v = f_{\theta_v}(I)
$$


and therefore still affects the assistant-token probabilities.

But the optimizer does not update $\theta_v$.

If a multimodal connector is trainable, let:

$$
u =
g_{\phi_c}(v)
$$

where $\phi_c$ denotes connector parameters.

Then, in general, assistant-token loss can produce:

$$
\frac{\partial \mathcal{L}} {\partial \phi_c} \neq 0
$$

so the connector can learn even while the vision encoder stays frozen.

---

### Crucial distinction

~~~text
LOSS MASKING

asks:

Which token positions contribute
directly to the supervised loss?


PARAMETER FREEZING

asks:

Which model parameters
are allowed to update?
~~~

These are different decisions.

A prompt token can be loss-masked and still affect the assistant prediction.

A vision encoder can be frozen and still provide visual features.

A trainable connector or LoRA adapter can still receive gradients from the assistant loss.

---

### Interview Explanation

A simple interview answer is:

> **Yes. Loss masking only decides where I measure the supervised error; it does not remove the prompt or image from the forward pass. In my Qwen example, the screenshot and instruction are part of the context used to predict the assistant tokens "Little" and "Content". I calculate cross-entropy only on those assistant targets, but their probabilities still depend on the image and prompt. Therefore the assistant loss can backpropagate through any trainable component that produced those predictions. If the vision encoder is frozen, its weights do not update, but its visual features still condition the prediction. A trainable connector or LoRA adapter can still receive gradients from that same assistant loss.**

### Qwen Project Application

For the Little Content project:

~~~text
Screenshot + instruction
        ↓
used as conditioning context
        ↓
Qwen predicts assistant token
        ↓
assistant-token cross-entropy
        ↓
gradient flows backward
through trainable components
~~~

There is no need for a separate supervised image-token loss for the screenshot to matter.

### Design Decision

We keep **loss masking** and **parameter freezing** separate:

- Loss masking selects the token positions included in the SFT objective.
- Parameter freezing selects the model parameters that the optimizer may update.

---

### Story Bridge 9 — The Human Class Name May Be Several Model Tokens

"Little Content" looks like one class label to us.

The tokenizer may represent it using multiple tokens.

That affects both training loss and later class scoring.

## Question 9 — What happens if a class label contains multiple tokens?

Suppose the positive label tokenizes as:

$$
y_1,y_2
$$

Then:

$$
P_\theta(y_1,y_2\mid x) =
P_\theta(y_1\mid x)
P_\theta(y_2\mid x,y_1)
$$

and:

$$
\log P_\theta(y_1, y_2 \mid x) = \log P_\theta(y_1 \mid x) + \log P_\theta(y_2 \mid x, y_1)
$$


SFT therefore supervises every token in the canonical target string.

### Why this matters for classification

Suppose one class label takes one token and another takes three.

Raw sequence log-probability contains more additive terms for the longer label.

Potential strategies include:

- Use short canonical labels.
- Compare length-normalized sequence scores.
- Add dedicated special class tokens if justified.
- Use a classification head instead of text generation.

### Qwen Project Application

We should not allow arbitrary equivalent outputs such as:

~~~text
Little Content

This is Little Content

The page has little content

LC
~~~

That creates unnecessary output variation.

### Design Decision

The project will use a **small canonical target vocabulary**.

The exact target strings and tokenizer behavior will be finalized in Part 3.

---

### Story Bridge 10 — A Binary Business Task Does Not Necessarily Require Free-Form Generation

We are using a generative VLM, but the business output is binary.

That creates a real architectural choice:

> Generate a label as text, or attach a dedicated classifier?

## Question 10 — Generative SFT or a classification head: which should we use?

### Option A — Generative label prediction

The model produces a canonical class response.

The training objective remains ordinary causal-LM SFT.

Advantages:

- Preserves the native VLM/chat interface.
- Reuses standard SFT tooling.
- Easy to extend to richer responses later.
- Fits naturally with instruction tuning.

Limitations:

- Label tokenization must be handled carefully.
- Free generation can produce formatting variants.
- Generative probabilities are not automatically calibrated binary probabilities.

### Option B — Add a classification head

Suppose a hidden representation $h\in\mathbb{R}^{d}$ feeds a two-class head:

$$
z =
W_c h+b_c
$$

with:

$$
W_c \in \mathbb{R}^{2\times d}
$$

$$
b_c \in \mathbb{R}^{2}
$$

and therefore:

$$
z
\in
\mathbb{R}^{2}
$$

The class probabilities are:

$$
P_\theta \left( y = k \mid x \right) = \frac{e^{z_k}}{\sum_{j=1}^{2} e^{z_j}}
$$


Advantages:

- Direct class logits.
- Efficient for pure classification.
- Easier to threshold and calibrate.

Limitations:

- Modifies the native model interface.
- Requires choosing the representation used for classification.
- Gives up some flexibility of a generative chat formulation.

### Qwen Project Application

Because the project is framed as fine-tuning a Qwen VLM and later parts explicitly study instruction/chat tuning, our main theoretical design will use **generative SFT**.

A classifier head remains a valid experimental baseline.

### Design Decision

Primary:

$$
\text{Generative SFT with canonical class responses}
$$

Secondary ablation:

$$
\text{Dedicated classification head}
$$

if supported cleanly by the implementation.


---

### Story Bridge 11 — A Generative Model Still Needs a Stable Production Score

Free-form generation is convenient for training, but production classification should not depend on wording variations.

A more controlled approach is to compare the likelihood of the allowed class responses.

## Question 11 — How can a generative model produce a classification score?

Let the positive canonical label be:

$$
y^{(+)}
$$

and the negative canonical label be:

$$
y^{(-)}
$$

Let the positive label contain $T_+$ tokens and the negative label contain $T_-$ tokens.

Their sequence log-scores are:

$$
S_+(x) = \sum_{t=1}^{T_+} \log P_\theta \left( y_t^{(+)} \mid x, y_{\lt t}^{(+)} \right)
$$

and:

$$
S_-(x) = \sum_{t=1}^{T_-} \log P_\theta \left( y_t^{(-)} \mid x, y_{\lt t}^{(-)} \right)
$$


A relative class score is:

$$
s(x) =
S_+(x)-S_-(x)
$$

A thresholded decision can then be written as:

$$
\hat{y} =
\mathbb{1}
\left[
s(x)\ge\tau
\right]
$$

where $\tau$ is chosen using validation data.

If the canonical labels have different token lengths, one possible comparison is the mean log-probability per token:

$$
\bar{S}_+(x) =
\frac{1}{T_+}
S_+(x)
$$

$$
\bar{S}_-(x) = \frac{1}{T_-}
S_-(x)
$$

with normalized relative score:

$$
\bar{s}(x) =
\bar{S}_+(x)-\bar{S}_-(x)
$$

### Why this helps

Instead of allowing unconstrained generation such as:

~~~text
This seems like a Little Content page.
~~~

the production layer can compare only the two known candidate responses.

### Qwen Project Application

The model can remain a generative VLM while the business decision layer behaves like a controlled classifier.

### Design Decision

The theoretical production design will prefer **canonical label likelihood comparison** over unrestricted text generation.

The final threshold and calibration design are deferred to Part 8.

---

# Qwen Project Build Record — After Questions 1–11

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
| Teacher forcing | Correct previous response tokens used in training | Design Decision |
| Loss scope | Assistant target tokens only | Design Decision |
| Prompt and image | Conditioning context, not direct text targets | Design Decision |
| Output approach | Generative canonical class response | Design Decision |
| Controlled class scoring | Compare canonical label likelihoods | Design Decision |
| Parameter updates and adapters | Covered in Part 2B | Next topic |

---
# Part 2 — Key Equations

## SFT dataset

$$
\mathcal{D}_{\mathrm{SFT}} =
\{
(x_i,y_i)
\}_{i=1}^{N}
$$

## Conditional response probability

$$
P_\theta \left( y_i \mid x_i \right) = \prod_{t=1}^{T_i} P_\theta \left( y_{i,t} \mid x_i, y_{i,\lt t}^{\text{true}} \right)
$$

## SFT negative log-likelihood

$$
\mathcal{L}_{\mathrm{SFT}} = - \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta \left( y_{i,t} \mid x_i, y_{i,\lt t}^{\text{true}} \right)
$$


## Softmax

$$
P_\theta(k \mid c_t) = \frac{e^{z_{t,k}}}{\sum_{j=1}^{V} e^{z_{t,j}}}
$$


## Token cross-entropy

$$
\mathcal{L}_t = - \log P_\theta \left( y_t \mid c_t \right)
$$


## Assistant-only masked loss

$$
\mathcal{L} = - \frac{\sum_t m_t \log P_\theta \left( y_t \mid c_t \right)}{\sum_t m_t}
$$

## Generative class score

$$
S_+(x) = \sum_{t=1}^{T_+} \log P_\theta \left( y_t^{(+)} \mid x, y_{\lt t}^{(+)} \right)
$$

$$
S_-(x) = \sum_{t=1}^{T_-} \log P_\theta \left( y_t^{(-)} \mid x, y_{\lt t}^{(-)} \right)
$$

$$
s(x) = S_+(x) - S_-(x)
$$


---

# Part 2 — Final Mental Model (Questions 1–11)

~~~text
LABELLED SERP SCREENSHOT + INSTRUCTION
        ↓
Qwen multimodal context
        ↓
Teacher-forced assistant response
        ↓
Next-token logits and softmax probabilities
        ↓
Assistant-only token loss
        ↓
Compare canonical class-response likelihoods
        ↓
SFT objective is ready
        ↓
Next: decide which parameters should update
~~~

These foundations lead directly to full fine-tuning and parameter-efficient fine-tuning.

---
# Continue to Part 2B — PEFT and LoRA

Questions 1–11 establish **what the model predicts, how the training targets are aligned, and where loss is measured**.

Questions 12–25 now live in [Part 2B — Full Fine-Tuning, PEFT and LoRA](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20PEFT%20and%20LoRA.md). They cover **which parameters are updated, how LoRA works, memory trade-offs, and the full Qwen training step**.
