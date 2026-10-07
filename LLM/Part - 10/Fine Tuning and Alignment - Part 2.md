# Part 10 — Fine-Tuning and Alignment

# Fine-Tuning and Alignment — Part 2

## 10.2 Supervised Fine-Tuning (SFT)

**Detailed Study Mode. Interview-style material remains separate.**

Part 1 established **why** fine-tuning is a reasonable tool for the Qwen Little Content project.

Part 2 now asks:

> **How do labelled examples actually change the model?**

This is where the project moves from a high-level idea into an actual training formulation.

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
| **Full FT vs PEFT / LoRA** | **Designed in this Part** |
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
\mathbf{z}_t
=
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
P_\theta
\left(
k\mid c_t
\right)
=
\frac{
e^{z_{t,k}}
}{
\sum_{j=1}^{V}
e^{z_{t,j}}
}
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
0.80+0.10+0.07+0.03=1
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
P_\theta
\left(
y_t\mid c_t
\right)
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
P_\theta
\left(
\text{Little}\mid c_t
\right)
=
0.8
$$

In simple terms:

> We ask how much probability the model assigned to the token that was actually correct.

If the correct token gets high probability, the prediction is good.

If it gets very low probability, the prediction is poor.

---

### Step 5 — Convert the correct-token probability into a loss

The token-level loss is:

$$
\boxed{
\mathcal{L}_t
=
-\log
P_\theta
\left(
y_t\mid c_t
\right)
}
$$

where $y_t$ is the correct target token.

#### Good prediction

Suppose:

$$
P_\theta
\left(
y_t\mid c_t
\right)
=
0.8
$$

Then:

$$
\mathcal{L}_t
=
-\log(0.8)
\approx
0.223
$$

The loss is small.

#### Bad prediction

Suppose:

$$
P_\theta
\left(
y_t\mid c_t
\right)
=
0.1
$$

Then:

$$
\mathcal{L}_t
=
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
P_\theta
\left(
y_t\mid c_t
\right)
\rightarrow 1
$$

the loss approaches:

$$
0
$$

But when:

$$
P_\theta
\left(
y_t\mid c_t
\right)
\rightarrow 0
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
P_\theta
\left(
\text{Little}\mid x
\right)
=
0.8
$$

Then:

$$
\mathcal{L}_1
=
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
P_\theta
\left(
\text{Content}
\mid
x,\text{Little}
\right)
=
0.6
$$

Then:

$$
\mathcal{L}_2
=
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
P_\theta
\left(
\text{EOS}
\mid
x,\text{Little},\text{Content}
\right)
=
0.9
$$

Then:

$$
\mathcal{L}_3
=
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
\boxed{
\mathcal{L}_{\mathrm{example}}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log
P_\theta
\left(
y_t\mid c_t
\right)
}
$$

For this example:

$$
\mathcal{L}_{\mathrm{example}}
=
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
P_\theta
\left(
\text{Little}\mid x
\right)
=
0.20
$$

Then:

$$
-\log(0.20)
\approx
1.609
$$

After some training, suppose:

$$
P_\theta
\left(
\text{Little}\mid x
\right)
=
0.70
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
\mathcal{L}
=
-
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
\mathcal{L}
=
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
\boxed{
\mathcal{L}_{\mathrm{SFT}}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log
P_\theta
\left(
y_t\mid c_t
\right)
}
$$

Assistant-only loss masking, discussed in Question 7, determines **which token positions are included in this average**.

---

### Story Bridge 7 — The Sequence Contains Context Tokens We Do Not Want to Train the Model to Reproduce

The SFT sequence includes:

- System text.
- User instruction.
- Image-related context.
- Assistant answer.

If we calculate loss everywhere, the model spends direct supervision on reproducing text that is only supposed to condition the task.

So we need a loss mask.

## Question 7 — Which tokens should contribute to the SFT loss?

A common instruction-SFT strategy is to compute loss only on assistant-response tokens.

Define:

$$
m_t
\in
\{0,1\}
$$

with:

$$
m_t
=
\begin{cases}
1 & \text{assistant target token}\\
0 & \text{context token}
\end{cases}
$$

Then:

$$
\mathcal{L}
=
-
\frac{
\sum_t
m_t
\log
P_\theta
\left(
y_t\mid c_t
\right)
}{
\sum_t m_t
}
$$

Conceptually:

~~~text
SYSTEM
attend to it
do not directly supervise it

USER
attend to it
do not directly supervise it

IMAGE
condition on it
not a text target

ASSISTANT TARGET
attend to it
supervise it
~~~

### Common implementation convention

Many causal-LM pipelines use an ignore value such as:

~~~text
-100
~~~

for positions that should not contribute to cross-entropy.

Conceptually:

~~~text
input_ids:
[system] [user] [image marker] [assistant marker] [Little] [Content]

labels:
[-100]  [-100] [-100]         [-100]             [Little] [Content]
~~~

The exact boundaries depend on the chosen Qwen chat template.

### Qwen Project Application

The screenshot and task instruction should condition the answer.

The assistant label supplies the direct supervision.

### Design Decision

Our default design is **assistant-only loss masking**.

We will not train the model to reconstruct the user's instruction.

---

### Story Bridge 8 — A Masked Context Can Still Affect the Gradient

Loss masking can sound as though the prompt and image are ignored during training.

That is not what happens.

The assistant prediction depends on those inputs.

We need to separate where loss is measured from where gradients can flow.

## Question 8 — If prompt and image positions are masked from the loss, can the model still learn from them?

Yes.

Suppose the supervised loss is:

$$
\mathcal{L}
=
-
\log
P_\theta
\left(
y
\mid
I,x
\right)
$$

where:

- $I$ is visual information.
- $x$ is textual context.

Even though the explicit target loss is evaluated on the assistant output, the prediction depends on the image and prompt.

For trainable visual-path parameters $\theta_v$:

$$
\frac{
\partial\mathcal{L}
}{
\partial\theta_v
}
=
\frac{
\partial\mathcal{L}
}{
\partial h
}
\frac{
\partial h
}{
\partial\theta_v
}
$$

if the visual path contributes to hidden representation $h$.

### Crucial distinction

~~~text
LOSS MASKING
Where is prediction error measured?

FREEZING
Which parameters are allowed to update?
~~~

These are different decisions.

### Example

The vision encoder may be frozen.

Then:

- Its output still conditions the assistant prediction.
- Gradients may mathematically reach its output.
- But its parameters are not updated.

A trainable multimodal connector can still receive gradient from the assistant loss.

### Qwen Project Application

The image can teach the downstream classifier behavior even though we do not define a separate "image-token loss."

### Design Decision

We will keep **loss masking** and **parameter freezing** conceptually separate throughout the project.

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
P_\theta(y_1,y_2\mid x)
=
P_\theta(y_1\mid x)
P_\theta(y_2\mid x,y_1)
$$

and:

$$
\log
P_\theta(y_1,y_2\mid x)
=
\log
P_\theta(y_1\mid x)
+
\log
P_\theta(y_2\mid x,y_1)
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

Suppose hidden representation $h$ feeds a two-class head:

$$
z
=
W_ch+b
$$

where:

$$
z
\in
\mathbb{R}^{2}
$$

Then:

$$
P(y=k\mid x)
=
\frac{
e^{z_k}
}{
\sum_{j=1}^{2}e^{z_j}
}
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

We can compute:

$$
S_+(x)
=
\log
P_\theta
\left(
y^{(+)}
\mid x
\right)
$$

and:

$$
S_-(x)
=
\log
P_\theta
\left(
y^{(-)}
\mid x
\right)
$$

A relative class score is:

$$
s(x)
=
S_+(x)-S_-(x)
$$

Then a thresholded decision is:

$$
\hat{y}
=
\mathbb{1}
\left[
s(x)\ge\tau
\right]
$$

where $\tau$ is chosen using validation data.

If the two canonical labels have different token lengths, one possible comparison is the length-normalized sequence score:

$$
\bar{S}(y\mid x)
=
\frac{1}{|y|}
\log
P_\theta
\left(
y\mid x
\right)
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

### Story Bridge 12 — SFT Defines the Loss, but It Does Not Decide Which Parameters Are Allowed to Move

We now know:

- What the model predicts.
- Which positions contribute to loss.
- How the classification score can be derived.

The next major decision is whether we update the entire model or only a small trainable subset.

## Question 12 — What is full fine-tuning?

In full fine-tuning, most or all pretrained model parameters remain trainable.

Let:

$$
\theta_0
$$

be the pretrained parameter state.

We optimize the model directly:

$$
\theta_0
\rightarrow
\theta^*
$$

through updates such as:

$$
\theta_{k+1}
=
\theta_k
-
\eta
\nabla_\theta
\mathcal{L}
$$

### Advantages

Full fine-tuning gives the model maximum adaptation capacity.

Every trainable representation can shift toward the downstream task.

This may help when:

- Domain shift is large.
- The downstream dataset is large and high quality.
- Compute is sufficient.
- The desired behavior differs substantially from the base model.

### Costs

It requires:

- Gradients for many parameters.
- Optimizer states for many parameters.
- More checkpoint storage.
- More training memory.
- More expensive distributed training.
- Greater risk of over-specializing a large model on a narrow dataset.

### Qwen Project Application

A full multimodal fine-tune could update:

~~~text
Vision encoder
+
multimodal connector
+
language backbone
~~~

That is more adaptation capacity than we may need for Little Content classification.

### Design Decision

Full fine-tuning is **not** our default first experiment.

It remains an escalation option if PEFT clearly underfits and sufficient data/compute are available.

---

### Story Bridge 13 — A Model That Fits for Inference May Still Be Too Expensive to Full-Fine-Tune

Inference mainly needs model weights, activations, and runtime state.

Training needs additional gradient and optimizer state.

So full fine-tuning can require far more memory than inference.

## Question 13 — Why is full fine-tuning so memory-intensive?

A simplified training-memory decomposition is:

$$
M_{\mathrm{train}}
\approx
M_{\mathrm{weights}}
+
M_{\mathrm{gradients}}
+
M_{\mathrm{optimizer}}
+
M_{\mathrm{activations}}
+
M_{\mathrm{runtime}}
$$

### Weight memory

For $P$ parameters stored using $b_w$ bytes each:

$$
M_{\mathrm{weights}}
\approx
Pb_w
$$

### Gradient memory

If every parameter is trainable:

$$
M_{\mathrm{gradients}}
\propto
P
$$

### Adam-like optimizer state

Adam maintains first and second moments:

$$
m_t
$$

and:

$$
v_t
$$

If both are FP32, those two tensors require roughly:

$$
8P
$$

bytes.

For a 7B-parameter model:

$$
7\times10^9\times8
=
56\times10^9
$$

bytes, or about 56 GB in decimal units, **only for the two FP32 moment tensors**.

This estimate still excludes:

- Model weights.
- Gradients.
- Activations.
- Possible FP32 master weights.
- Framework overhead.

### Important qualification

Modern techniques such as:

- ZeRO-style sharding.
- FSDP.
- Offloading.
- Lower-precision optimizer states.
- Quantization.

can change the actual memory footprint substantially.

The point is the scaling behavior, not a universal fixed number.

### Qwen Project Application

For a narrow VLM adaptation problem, paying optimizer-state cost for every pretrained parameter may be unnecessary.

### Design Decision

Training-memory efficiency is one of the reasons to prefer PEFT as our starting strategy.

---

### Story Bridge 14 — We Want Task Adaptation Without Paying to Train the Whole Model

Parameter-Efficient Fine-Tuning asks:

> Can we keep the pretrained model mostly fixed and learn only a small task-specific parameter set?

That is a natural match for our project goals.

## Question 14 — What is Parameter-Efficient Fine-Tuning?

Parameter-Efficient Fine-Tuning, or **PEFT**, keeps most pretrained parameters frozen and learns a much smaller trainable parameter set.

Let:

$$
\theta_0
=
\text{frozen base parameters}
$$

and:

$$
\phi
=
\text{trainable PEFT parameters}
$$

Then the model can be represented as:

$$
f(x;\theta_0,\phi)
$$

and the optimization becomes:

$$
\phi^*
=
\underset{\phi}{\operatorname{argmin}}
\;
\mathcal{L}
\left(
\theta_0,\phi
\right)
$$

while the base model is not directly updated.

### What PEFT can reduce

- Number of trainable parameters.
- Gradient storage for base weights.
- Optimizer-state memory for base weights.
- Task-specific checkpoint size.

### What PEFT does not remove

The frozen base model still participates in the forward computation.

So a model with only a small fraction of trainable parameters is not automatically proportionally cheaper in all forms of compute.

### Qwen Project Application

This aligns with our Part 1 design goal:

> Preserve broad Qwen capability while learning a narrow Little Content behavior.

### Design Decision

PEFT is the preferred adaptation family for the initial Qwen training design.

---

### Story Bridge 15 — LoRA Gives PEFT a Concrete Mathematical Form

Among PEFT methods, LoRA is especially attractive because it modifies the effect of a large linear layer through a small low-rank update.

We now need to derive that update.

## Question 15 — What is LoRA mathematically?

Consider a pretrained linear transformation:

$$
h
=
W_0x
$$

where:

$$
W_0
\in
\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

Full fine-tuning would directly update:

$$
W_0
$$

LoRA freezes $W_0$ and writes the task-specific update as:

$$
\Delta W
=
BA
$$

where:

$$
A
\in
\mathbb{R}^{r\times d_{\mathrm{in}}}
$$

and:

$$
B
\in
\mathbb{R}^{d_{\mathrm{out}}\times r}
$$

with:

$$
r
\ll
\min
\left(
d_{\mathrm{in}},
d_{\mathrm{out}}
\right)
$$

The effective transformation becomes:

$$
W
=
W_0+\Delta W
$$

and:

$$
h
=
W_0x
+
BAx
$$

A commonly used scaled form is:

$$
h
=
W_0x
+
\frac{\alpha}{r}
BAx
$$

where:

- $r$ is LoRA rank.
- $\alpha$ controls update scale.

### Intuition

Instead of learning an unrestricted update with:

$$
d_{\mathrm{out}}d_{\mathrm{in}}
$$

degrees of freedom, LoRA constrains the update to a lower-rank subspace.

### Qwen Project Application

The pretrained transformation remains intact.

The LoRA matrices learn a smaller task-specific correction for Little Content behavior.

### Design Decision

LoRA is our **primary PEFT method** for the theoretical project.

---

### Story Bridge 16 — Low Rank Is Useful Only If It Actually Saves Parameters

The factorization looks compact.

Now we should quantify how much smaller it is than a full matrix update.

## Question 16 — How many trainable parameters does LoRA add?

The original matrix has:

$$
N_{\mathrm{full}}
=
d_{\mathrm{out}}d_{\mathrm{in}}
$$

parameters.

LoRA adds:

$$
N_{\mathrm{LoRA}}
=
rd_{\mathrm{in}}
+
d_{\mathrm{out}}r
$$

or:

$$
N_{\mathrm{LoRA}}
=
r
\left(
d_{\mathrm{in}}
+
d_{\mathrm{out}}
\right)
$$

### Numerical example

Suppose:

$$
d_{\mathrm{in}}
=
d_{\mathrm{out}}
=
4096
$$

Then:

$$
N_{\mathrm{full}}
=
4096\times4096
=
16{,}777{,}216
$$

For:

$$
r=8
$$

LoRA uses:

$$
N_{\mathrm{LoRA}}
=
8\times4096
+
4096\times8
=
65{,}536
$$

The fraction is:

$$
\frac{
65{,}536
}{
16{,}777{,}216
}
\approx
0.003906
$$

or about:

$$
0.39\%
$$

of the original matrix size.

### Important qualification

This percentage is for **one matrix**.

The full model's trainable fraction depends on:

- Number of adapted layers.
- Number of adapted projections.
- Rank.
- Whether the connector is trainable.
- Whether vision modules are adapted.

### Qwen Project Application

This makes it possible to create a relatively small task-specific adapter while leaving the large Qwen backbone frozen.

### Design Decision

The final project will report the actual number and percentage of trainable parameters rather than merely saying "we used LoRA."


---

### Story Bridge 17 — LoRA Still Has Hyperparameters That Control Capacity and Update Strength

Saying "use LoRA" is not a complete design.

LoRA rank, scaling, and regularization determine how much task-specific capacity the adapter has and how strongly its update influences the frozen base model.

## Question 17 — What do LoRA rank, alpha, scaling, and dropout control?

### Rank

The rank is:

$$
r
$$

and controls the dimension of the low-rank update.

A larger rank gives:

- More trainable parameters.
- More adaptation capacity.
- More optimizer memory.
- Potentially greater overfitting risk.

A smaller rank gives:

- Fewer trainable parameters.
- Lower memory.
- Stronger capacity constraint.

### Alpha

A common LoRA scaling factor is:

$$
\frac{\alpha}{r}
$$

so the effective update is:

$$
\Delta W_{\mathrm{effective}}
=
\frac{\alpha}{r}BA
$$

Alpha controls the relative magnitude of the adapter contribution.

### LoRA dropout

A dropout operation can be applied on the LoRA path during training.

Conceptually:

~~~text
Input
   ↓
LoRA dropout
   ↓
A
   ↓
B
   ↓
Scaled adapter update
~~~

It can regularize the adapter, especially on smaller datasets.

### Initialization

A common design initializes the factors so that:

$$
BA
=
0
$$

at initialization.

Then:

$$
W
=
W_0
$$

at the start of training.

This lets the model begin from the pretrained behavior before the adapter learns a task-specific correction.

### Qwen Project Application

We should not choose:

~~~text
rank = 8
alpha = 16
dropout = 0.05
~~~

simply because those values are common online.

The appropriate values depend on:

- Model size.
- Dataset size.
- Adapted modules.
- Memory budget.
- Validation behavior.

### Design Decision

Part 2 defines the role of these hyperparameters.

The exact search values will be fixed in **Part 6**, where we design the complete training recipe.

---

### Story Bridge 18 — LoRA Has to Be Attached to Specific Matrices

A Transformer contains many linear transformations.

The adapter does not automatically know where to go.

So target-module selection is itself a model-capacity decision.

## Question 18 — Which Transformer matrices can receive LoRA adapters?

Self-attention commonly contains projection matrices:

$$
W_Q,\;
W_K,\;
W_V,\;
W_O
$$

corresponding to:

- Query.
- Key.
- Value.
- Attention output.

The feed-forward block also contains large linear transformations.

### Strategy A — Minimal attention adaptation

~~~text
Q projection
V projection
~~~

This is a relatively parameter-efficient starting point.

### Strategy B — Broader attention adaptation

~~~text
Q
K
V
O
~~~

This increases adaptation capacity.

### Strategy C — Attention + MLP adaptation

Adapters are placed on:

- Attention projections.
- Feed-forward linear layers.

This gives still more capacity.

### Tradeoff

~~~text
More target modules
        ↓
More trainable parameters
        ↓
More adaptation capacity
        ↓
More memory
+
potentially greater overfitting risk
~~~

### Qwen Project Application

The exact module names differ across Qwen variants and software implementations.

For example, generic names such as:

~~~text
q_proj
k_proj
v_proj
o_proj
~~~

should never be assumed without inspecting the selected model.

### Design Decision

Our **initial theoretical language-side LoRA baseline** will target:

> **Query and Value attention projections.**

If validation suggests under-capacity, we expand the target set systematically rather than immediately adapting every linear layer.

---

### Story Bridge 19 — A Vision-Language Model Has More Adaptation Choices Than a Text-Only LLM

Qwen receives the screenshot through a visual pathway before the language model makes the class decision.

That means we need to decide whether to adapt:

- Vision encoder.
- Multimodal connector.
- Language backbone.
- Some combination of them.

## Question 19 — Which VLM components should we freeze or adapt?

A simplified multimodal architecture is:

~~~text
Vision encoder
        ↓
Multimodal projector / connector
        ↓
Language model
~~~

There are several reasonable strategies.

### Strategy A — Freeze vision, adapt language only

Advantages:

- Very parameter-efficient.
- Preserves general visual representation.
- Good first test if pretrained visual features are already sufficient.

Risk:

- May not adapt enough to SERP-specific layout patterns.

### Strategy B — Freeze vision, train connector + language LoRA

Advantages:

- The connector can learn a more task-relevant mapping from visual features into language space.
- Much cheaper than full VLM fine-tuning.
- Preserves most pretrained visual capability.

### Strategy C — Add PEFT to upper vision layers

Advantages:

- Gives the visual representation some domain-specific flexibility.

Risks:

- More trainable parameters.
- More training complexity.
- Higher overfitting risk.

### Strategy D — Full multimodal fine-tuning

Advantages:

- Maximum adaptation capacity.

Risks:

- Highest memory and compute cost.
- Highest risk of over-specialization on a narrow dataset.

### Qwen Project Application

The task depends strongly on visual layout and content density.

Therefore, it would be risky to treat the screenshot pathway as irrelevant.

At the same time, the base Qwen model already has broad visual capability.

### Design Decision

Our initial theoretical configuration is:

~~~text
Vision encoder
Frozen initially

Multimodal connector / projector
Trainable if exposed separately

Language backbone
Frozen

Language attention
LoRA on Q and V projections
~~~

If validation shows systematic failure on visual-layout distinctions, we will test PEFT on upper vision layers.

---

### Story Bridge 20 — A Good Initial Adapter Placement Is a Hypothesis, Not a Truth

The previous configuration is intentionally a starting point.

A mature design should prove that additional adaptation capacity is useful before paying for it.

That calls for ablation experiments.

## Question 20 — How would we decide whether the initial LoRA placement is sufficient?

We can compare controlled variants while keeping the data and evaluation protocol fixed.

### Configuration A

~~~text
Language Q/V LoRA
Vision frozen
Connector trainable
~~~

### Configuration B

~~~text
Language Q/K/V/O LoRA
Vision frozen
Connector trainable
~~~

### Configuration C

~~~text
Language attention + MLP LoRA
Vision frozen
Connector trainable
~~~

### Configuration D

~~~text
Language LoRA
+
upper-vision PEFT
+
connector trainable
~~~

Compare:

- Validation precision.
- Validation recall.
- F1.
- Performance on unseen-host slices.
- Performance on hard visual-layout cases.
- Trainable parameter count.
- Peak training memory.
- Training time.

### How to interpret the result

If:

$$
\text{Performance}(A)
\approx
\text{Performance}(C)
$$

but A uses far fewer trainable parameters, A is preferable.

If A systematically fails on layout-specific examples and D fixes those failures without harming generalization, visual adaptation becomes justified.

### Qwen Project Application

This gives us a defensible interview answer:

> We started with the smallest plausible PEFT configuration and expanded capacity only when validation error analysis showed a need.

### Design Decision

Adapter placement will be selected through **ablation**, not habit.

---

### Story Bridge 21 — Fewer Trainable Parameters Do Not Mean the Whole Model Disappears During Training

LoRA can reduce trainable state dramatically.

But every training example still passes through the frozen base network.

So we need to understand what PEFT saves and what it does not.

## Question 21 — How does PEFT change training memory and compute?

With full fine-tuning, conceptually:

~~~text
Base weights
+
gradients for base weights
+
optimizer states for base weights
+
activations
~~~

With LoRA PEFT:

~~~text
Frozen base weights
+
LoRA gradients
+
optimizer states for LoRA
+
activations
~~~

Let:

$$
P
$$

be total base-model parameters and:

$$
p
$$

be trainable adapter parameters with:

$$
p\ll P
$$

Then gradient and optimizer-state storage associated with trainable parameters can shrink dramatically.

### But forward compute remains

The frozen model still performs:

- Attention.
- MLP computation.
- Vision processing.
- Multimodal fusion.

The base network is needed to compute the hidden states that the adapters modify.

### Important consequence

If only 1% of parameters are trainable, it does **not** mean training becomes 100 times faster.

The biggest PEFT savings are often:

- Optimizer memory.
- Gradient memory.
- Checkpoint size.

Compute savings can be much smaller than parameter-count savings.

### Qwen Project Application

LoRA makes adaptation practical without pretending that the underlying Qwen VLM is computationally tiny.

### Design Decision

When discussing LoRA, we will describe it as:

> **Parameter-efficient and optimizer-memory-efficient**, not computationally free.


---

### Story Bridge 22 — If the Frozen Base Model Is Still Too Large, We Can Compress the Base and Keep the Adapters Trainable

LoRA removes the need to optimize all base-model parameters.

But the frozen base weights still need to reside in memory.

If that memory requirement becomes the bottleneck, quantization can be combined with LoRA.

## Question 22 — What is QLoRA, and when would we use it?

QLoRA combines:

1. A quantized frozen base model.
2. Trainable LoRA adapters.

Conceptually:

~~~text
Base model weights
stored in quantized form
        ↓
Frozen

LoRA adapters
stored and trained at higher precision
        ↓
Updated during SFT
~~~

The goal is to reduce the memory required to hold the base model while retaining task-specific trainable adapters.

### Why this is different from ordinary LoRA

With ordinary LoRA:

~~~text
Base model
full training precision / normal inference precision
+
LoRA adapters
~~~

With QLoRA:

~~~text
Base model
quantized
+
LoRA adapters
trainable
~~~

### Important implementation details

Actual QLoRA behavior depends on:

- Quantization format.
- Compute dtype.
- Hardware support.
- Software stack.
- VLM implementation support.
- Which modules are quantized.

So QLoRA is not a purely mathematical replacement for LoRA; it is also an engineering choice.

### Qwen Project Application

If the selected Qwen VLM cannot fit comfortably for ordinary LoRA training on available hardware, QLoRA becomes an attractive option.

### Design Decision

Our preferred decision hierarchy is:

~~~text
LoRA
        ↓
If base-model memory is limiting:
QLoRA
        ↓
If PEFT underfits despite good data:
broader PEFT
        ↓
Full fine-tuning only if justified
~~~

QLoRA is primarily a **memory-efficiency strategy**, not automatically a better-performing training method.

---

### Story Bridge 23 — We Have All the Pieces; Now Follow One Example Through a Complete Training Step

At this point we understand:

- The input.
- The target.
- Teacher forcing.
- Token cross-entropy.
- Loss masking.
- Frozen versus trainable parameters.
- LoRA.

The best way to consolidate this is to follow one example from screenshot to optimizer update.

## Question 23 — What happens in one Qwen SFT training step?

Consider one labelled example.

### Step 1 — Load the screenshot

~~~text
SERP screenshot
~~~

### Step 2 — Apply the model's image processor

The exact processor depends on the selected Qwen variant.

It may perform operations such as:

- Resize.
- Normalize.
- Convert to tensor.
- Create model-specific image metadata.

We defer exact preprocessing details to Part 3.

### Step 3 — Construct the multimodal conversation

Conceptually:

~~~text
SYSTEM
You are a page-quality classifier.

USER
<image>
Classify the screenshot.

ASSISTANT
little_content
~~~

### Step 4 — Tokenize the textual portions

The system, user, and assistant text are converted into token IDs.

The image is represented through the model's visual pathway.

### Step 5 — Build attention and supervision masks

Context tokens:

~~~text
attention = active
loss = masked
~~~

Assistant target tokens:

~~~text
attention = active
loss = active
~~~

Padding:

~~~text
attention = inactive
loss = masked
~~~

### Step 6 — Forward pass

The model produces vocabulary logits:

$$
z_t
\in
\mathbb{R}^{V}
$$

for each relevant causal prediction position.

### Step 7 — Compute assistant-only cross-entropy

$$
\mathcal{L}
=
-
\frac{
\sum_t
m_t
\log
P_\theta
\left(
y_t\mid c_t
\right)
}{
\sum_t m_t
}
$$

### Step 8 — Backpropagate

The gradient flows through the computation graph.

Frozen base parameters do not receive optimizer updates.

Trainable LoRA parameters and selected connector parameters do.

### Step 9 — Optimizer update

For trainable parameter set $\phi$:

$$
\phi_{k+1}
=
\phi_k
-
\eta
\widehat{g}_k
$$

where $\widehat{g}_k$ represents the optimizer-adjusted gradient.

### Step 10 — Repeat over batches

Across many labelled screenshots, the model gradually assigns more probability to the correct canonical class response.

### Qwen Project Application

This is the first complete theoretical training loop for the resume project.

### Design Decision

This SFT loop becomes the backbone of the final Qwen project file.

---

### Story Bridge 24 — Real Training Uses Batches, So Padding and Supervision Need Different Masks

Examples do not all have the same text length.

Images may also require model-specific batching behavior.

That means batching introduces multiple masking concepts that should not be confused.

## Question 24 — What is the difference between attention masking and loss masking?

These masks solve different problems.

### Attention mask

The attention mask identifies valid sequence positions rather than padding.

Conceptually:

$$
a_t
=
\begin{cases}
1 & \text{real token}\\
0 & \text{padding token}
\end{cases}
$$

Causal masking separately prevents access to future positions.

### Loss mask

The loss mask identifies which valid tokens should contribute to the supervised objective:

$$
m_t
=
\begin{cases}
1 & \text{assistant target token}\\
0 & \text{context or padding}
\end{cases}
$$

Therefore a user prompt token may have:

~~~text
attention mask = 1
loss mask = 0
~~~

because the model must read the prompt but should not be directly trained to reproduce it.

Padding may have:

~~~text
attention mask = 0
loss mask = 0
~~~

### Qwen Project Application

This distinction becomes important when we build the Qwen data collator in Part 3.

### Design Decision

The training pipeline will explicitly maintain:

> **attention validity** and **assistant supervision** as separate masks.

---

### Story Bridge 25 — Part 2 Should End With a Concrete Fine-Tuning Design, Not a List of LoRA Definitions

Part 1 told us why fine-tuning was needed.

Part 2 has now told us how the supervision signal is constructed and where the trainable capacity lives.

We should finish by reconstructing the whole training design in one flow.

## Question 25 — What is the Qwen fine-tuning design after Part 2?

The theoretical training pipeline is now:

~~~text
Labelled SERP screenshot
        ↓
Qwen multimodal processor
        ↓
Screenshot representation
+
task instruction
        ↓
Pretrained Qwen VLM
        ↓
Teacher-forced causal SFT
        ↓
Assistant-only token loss
        ↓
Cross-entropy / negative log-likelihood
        ↓
Backpropagation
        ↓
Frozen base weights

Trainable:
LoRA adapters
+
multimodal connector if separable
        ↓
Task-specific update
        ↓
Higher probability for the
correct Little Content label
~~~

### Primary adaptation strategy

~~~text
Vision encoder
Frozen initially

Multimodal connector / projector
Trainable if exposed separately

Language backbone
Frozen

Language attention
LoRA on Q and V projections
~~~

### Expansion strategy

If the starting setup underfits:

~~~text
Q/V LoRA
    ↓
Q/K/V/O LoRA
    ↓
Attention + MLP LoRA
    ↓
Upper-vision PEFT
    ↓
Full fine-tuning only if justified
~~~

### Production classification interface

The model remains generative.

The downstream decision layer can compare the likelihoods of the two canonical class responses rather than rely on unconstrained free-form generation.

### What remains intentionally unresolved

- Exact Qwen variant.
- Exact image processor.
- Exact chat template.
- Exact canonical label strings.
- LoRA rank.
- LoRA alpha.
- LoRA dropout.
- Batch size.
- Gradient accumulation.
- Learning rate.
- Optimizer.
- Scheduler.
- Number of epochs.
- Dataset split.
- Class balancing.
- Threshold.
- Calibration.
- Deployment configuration.

These belong to later parts rather than being guessed prematurely.

---

# Qwen Project Build Record — After Part 2

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
| Teacher forcing | Used during SFT | Design Decision |
| Loss scope | Assistant target tokens only | Design Decision |
| Prompt tokens | Used as context; masked from direct loss | Design Decision |
| Image | Conditions the answer; not treated as a text target | Design Decision |
| Primary output formulation | Generative class label | Design Decision |
| Production class scoring | Compare canonical label likelihoods | Design Decision |
| Full fine-tuning | Not the default starting strategy | Design Decision |
| Preferred adaptation family | PEFT | Design Decision |
| Primary PEFT method | LoRA | Design Decision |
| Initial vision encoder | Frozen | Design Decision |
| Multimodal connector | Trainable if separable in chosen architecture | Design Decision |
| Language base weights | Frozen | Design Decision |
| Initial LoRA targets | Language attention Q and V projections | Design Decision |
| Expansion path | Q/K/V/O → MLP → upper-vision PEFT if validation requires | Design Decision |
| Adapter selection method | Controlled ablation | Design Decision |
| QLoRA | Memory-constrained alternative | Design Decision |
| Exact Qwen variant | Not established | Open Question |
| Exact target strings | Deferred to Part 3 | Open Question |
| Exact chat template | Deferred to Part 3 | Open Question |
| Exact image preprocessing | Deferred to Part 3 | Open Question |
| LoRA rank / alpha / dropout | Deferred to Part 6 | Open Question |
| Optimizer / LR / scheduler | Deferred to Part 6 | Open Question |
| Dataset split / leakage controls | Deferred to Part 7 | Open Question |
| Threshold / calibration | Deferred to Part 8 | Open Question |
| Deployment architecture | Deferred to Part 8 | Open Question |

---

# Part 2 — Key Equations

## SFT dataset

$$
\mathcal{D}_{\mathrm{SFT}}
=
\{
(x_i,y_i)
\}_{i=1}^{N}
$$

## Conditional response probability

$$
P_\theta(y_i\mid x_i)
=
\prod_{t=1}^{T_i}
P_\theta
\left(
y_{i,t}
\mid
x_i,
y_{i,<t}
\right)
$$

## SFT negative log-likelihood

$$
\mathcal{L}_{\mathrm{SFT}}
=
-
\sum_{i=1}^{N}
\sum_{t=1}^{T_i}
\log
P_\theta
\left(
y_{i,t}
\mid
x_i,
y_{i,<t}
\right)
$$

## Softmax

$$
P_\theta(k\mid c_t)
=
\frac{
e^{z_{t,k}}
}{
\sum_{j=1}^{V}e^{z_{t,j}}
}
$$

## Token cross-entropy

$$
\mathcal{L}_t
=
-
\log
P_\theta
\left(
y_t\mid c_t
\right)
$$

## Assistant-only masked loss

$$
\mathcal{L}
=
-
\frac{
\sum_t
m_t
\log
P_\theta
\left(
y_t\mid c_t
\right)
}{
\sum_t m_t
}
$$

## PEFT objective

$$
\phi^*
=
\underset{\phi}{\operatorname{argmin}}
\;
\mathcal{L}
\left(
\theta_0,\phi
\right)
$$

## LoRA update

$$
\Delta W
=
BA
$$

## Scaled LoRA layer

$$
h
=
W_0x
+
\frac{\alpha}{r}
BAx
$$

## LoRA parameter count

$$
N_{\mathrm{LoRA}}
=
r
\left(
d_{\mathrm{in}}
+
d_{\mathrm{out}}
\right)
$$

## Approximate training-memory decomposition

$$
M_{\mathrm{train}}
\approx
M_{\mathrm{weights}}
+
M_{\mathrm{gradients}}
+
M_{\mathrm{optimizer}}
+
M_{\mathrm{activations}}
+
M_{\mathrm{runtime}}
$$

## Generative class score

$$
s(x)
=
\log
P_\theta
\left(
y^{(+)}
\mid x
\right)
-
\log
P_\theta
\left(
y^{(-)}
\mid x
\right)
$$

---

# Part 2 — Final Mental Model

~~~text
LABELLED EXAMPLE
Screenshot
+
Instruction
+
Correct response
        ↓

QWEN MULTIMODAL FORWARD PASS
Vision features
+
Text context
        ↓

TEACHER FORCING
Ground-truth response history is known
        ↓

TOKEN LOGITS
        ↓

ASSISTANT-ONLY CROSS-ENTROPY
Prompt conditions the answer
but does not receive direct target loss
        ↓

BACKPROPAGATION
        ↓

FROZEN BASE MODEL
+
TRAINABLE LoRA ADAPTERS
+
TRAINABLE CONNECTOR
        ↓

TASK-SPECIFIC UPDATE
        ↓

MODEL BECOMES MORE LIKELY TO PRODUCE
THE CORRECT LITTLE CONTENT LABEL
~~~

The central distinction is:

> **SFT defines the training signal; PEFT defines which parameters are allowed to respond to that signal.**

---

# Bridge to Part 3

Part 2 answered:

> **How do labelled examples produce gradients, and which parameters should we adapt?**

Part 3 will study:

- **10.3 Instruction Tuning**
- **10.4 Chat Format Training**

That is where the example becomes fully concrete:

~~~text
SERP screenshot
        ↓
Qwen image processor
        ↓
System message
        ↓
User instruction
        ↓
Assistant target
        ↓
Chat template
        ↓
Tokenization
        ↓
Loss labels
        ↓
Data collator
~~~

Part 3 will answer:

> **Exactly how should the Qwen Little Content examples be formatted and fed into the model?**
