# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 1

## 8.1 What happens during inference?
## 8.2 Prompt tokens

This note follows a question-driven learning style. Each answer should naturally create the next question.

The story is:

```text
I type a prompt.
    ↓
What does the model actually do with it?
    ↓
Inference pipeline
    ↓
But the model cannot operate directly on words.
What numerical object does it actually receive?
    ↓
Prompt tokens
    ↓
Now all prompt tokens are already known.
Can the model process the whole prompt efficiently?
    ↓
Next topic: Prefill
```

---

# 8.1 What Happens During Inference?

## Question 1 — What does “inference” actually mean for an LLM?

Inference means using a model whose parameters have already been learned to make predictions on new input.

For a decoder-only language model, the prediction problem is:

> Given every token seen so far, what is the probability distribution over the next token?

Suppose the tokens currently available to the model are:

$$
x_1, x_2, x_3, \ldots, x_t
$$

The model estimates:

$$
P\left(x_{t+1}\mid x_1,x_2,\ldots,x_t\right)
$$

### How should you read this equation?

Read it as:

> **“The probability of the next token, $x_{t+1}$, given all tokens from $x_1$ through $x_t$.”**

Where:

- $x_1$ = first token in the current context
- $x_2$ = second token
- $x_t$ = latest token currently known
- $x_{t+1}$ = token we are trying to predict
- $P(\cdot)$ = probability
- the symbol $\mid$ means **“given”**

So the equation is simply a compact mathematical way of saying:

```text
Look at everything seen so far
→ estimate what token should come next
```

### Why do we need this mathematics?

Because this equation defines the exact problem an autoregressive language model solves.

The model is **not** directly predicting an entire paragraph.

Instead, it repeatedly predicts:

$$
\text{one next-token distribution}
$$

then another,

then another.

That single equation eventually explains:

- greedy decoding
- temperature
- top-k
- top-p
- beam search
- stop tokens
- sequential generation

### Example

Prompt:

```text
The capital of France is
```

Conceptually, the model might assign probabilities like:

```text
Paris      0.82
Lyon       0.03
located    0.02
France     0.01
...        ...
```

These numbers are illustrative.

The important point is:

> The model first produces a **distribution over possible next tokens**. A decoding rule later decides which token is actually selected.

### What if there are 100,000 vocabulary tokens?

Then the model conceptually assigns one score to each of those 100,000 possible next tokens.

Only after that do we decide which token should be chosen.

---

## Question 2 — How is inference different from training?

During training, the model changes its parameters.

During ordinary inference, the parameters remain fixed.

Let all trainable model parameters be represented by:

$$
\theta
$$

Training tries to find parameters that minimize a loss function:

$$
\theta^*
=
\arg\min_{\theta}\mathcal{L}(\theta)
$$

### How should you read this?

- $\theta$ = candidate model parameters
- $\mathcal{L}(\theta)$ = training loss produced by those parameters
- $\arg\min$ = “find the value that makes this quantity as small as possible”
- $\theta^*$ = learned parameters after optimization

So in plain English:

```text
Training:
Find model weights that minimize prediction error.
```

During inference, we do not solve that optimization problem again.

We simply use the learned parameters $\theta^*$:

$$
P_{\theta^*}\left(x_{t+1}\mid x_{\le t}\right)
$$

Here:

$$
x_{\le t}
$$

is shorthand for:

$$
x_1,x_2,\ldots,x_t
$$

### Why do we need this mathematics?

Because it cleanly separates two operations:

```text
Training
→ change θ

Inference
→ keep θ fixed and evaluate the model
```

During training we need gradients such as:

$$
\frac{\partial \mathcal{L}}{\partial \theta}
$$

which means:

> “How does the loss change if I slightly change each model parameter?”

Those gradients are used by an optimizer to update the weights.

During ordinary inference, we do not need to compute or store gradients for weight updates.

### What if the conversation teaches the model something new?

The model can use information placed in its current context without changing $\theta$.

For example, if you tell the model:

```text
For this conversation, call my project ORBIT.
```

then that information becomes part of the current context $x_{\le t}$.

The model may use it in future predictions even though its learned weights did not change.

### What if we fine-tune the model while serving?

Then that update step is training or adaptation, not pure inference.

---

## Question 3 — What is the complete path from text to a next-token prediction?

At a high level:

```text
Raw text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Embedding vectors
    ↓
Transformer layers
    ↓
Contextual hidden states
    ↓
Language-model head
    ↓
Vocabulary logits
    ↓
Probability distribution
    ↓
Decoding rule
    ↓
Selected next token
```

This is the complete map.

We will now unpack just enough of it to understand what inference is doing before later chapters examine prefill, decode, KV cache, and decoding methods in detail.

---

## Question 4 — Why can’t the model operate directly on text?

Neural networks perform numerical operations such as matrix multiplication.

Therefore the text must eventually become numerical tensors.

Suppose the tokenizer produces:

$$
[x_1,x_2,x_3,x_4,x_5]
=
[464,3139,286,4881,318]
$$

The token IDs here are only illustrative.

### Important point

These IDs are **labels**, not numerical magnitudes.

Token ID 8000 is not “eight times more meaningful” than token ID 1000.

They behave more like dictionary indexes.

---

## Question 5 — How does a token ID become something the Transformer can process?

The model stores an embedding table.

Let:

- $|V|$ = vocabulary size
- $d_{model}$ = model hidden dimension

Then the embedding matrix has shape:

$$
E\in\mathbb{R}^{|V|\times d_{model}}
$$

### How should you read this?

Suppose:

$$
|V|=100{,}000
$$

and:

$$
d_{model}=4096
$$

Then:

$$
E\in\mathbb{R}^{100000\times4096}
$$

That means:

```text
100,000 rows
×
4,096 numbers per row
```

Each vocabulary token has one learned vector of length 4096.

For token ID $x_i$, the model retrieves:

$$
e_i=E[x_i]
$$

where:

$$
e_i\in\mathbb{R}^{d_{model}}
$$

### Why do we need this mathematics?

Because it shows exactly how a discrete symbol becomes a continuous vector.

The integer ID itself cannot meaningfully participate in Transformer matrix multiplications.

The embedding lookup converts:

```text
categorical ID
→ continuous vector
```

### What if two token IDs are numerically close?

That tells us nothing about semantic similarity.

For example:

```text
ID 2001 and ID 2002
```

might represent completely unrelated tokens.

Semantic relationships live in the learned embedding vectors, not in the raw integer IDs.

---

## Question 6 — If tokens become embeddings, how does the model know their order?

A token embedding tells the model primarily **what token it is**.

The model also needs information about **where that token occurs**.

Compare:

```text
Dog bites man
```

with:

```text
Man bites dog
```

The same broad set of words appears, but order changes meaning.

Transformers therefore incorporate position information.

Conceptually:

$$
\text{representation used by attention}
=
\text{token information}
+
\text{position-dependent information}
$$

This is a conceptual statement, not a universal literal addition formula.

Some architectures use positional embeddings, while many modern decoder-only models use mechanisms such as RoPE inside attention.

### Why do we need positional mathematics?

Self-attention alone does not inherently know that one token is first and another is fifth.

Position-aware transformations allow attention to distinguish different relative or absolute locations.

### What if position information disappeared?

The model would lose a crucial signal for representing order.

Statements that contain similar tokens but different word order would become much harder to distinguish correctly.

---

## Question 7 — What comes out of the final Transformer layer?

After passing through all Transformer layers, each position has a contextual hidden representation.

For the latest relevant position $t$, write:

$$
h_t\in\mathbb{R}^{d_{model}}
$$

If:

$$
d_{model}=4096
$$

then $h_t$ contains 4096 numbers.

These numbers summarize the model's learned contextual representation at that position.

### Why do we need this vector?

Because the model now has to convert one contextual representation into a score for every possible vocabulary token.

That creates the next mathematical problem:

```text
4096-dimensional hidden state
→
100,000 vocabulary scores
```

---

## Question 8 — How does the hidden state become vocabulary scores?

The language-model head performs a linear projection.

A simple form is:

$$
z=W_{LM}h_t+b
$$

where:

$$
W_{LM}\in\mathbb{R}^{|V|\times d_{model}}
$$

$$
h_t\in\mathbb{R}^{d_{model}}
$$

and therefore:

$$
z\in\mathbb{R}^{|V|}
$$

### Dimension check

Suppose:

$$
|V|=100{,}000
$$

and:

$$
d_{model}=4096
$$

Then:

$$
W_{LM}: 100000\times4096
$$

multiplied by:

$$
h_t:4096\times1
$$

produces:

$$
z:100000\times1
$$

So there is now **one number for every vocabulary token**.

Each $z_i$ is called a **logit**.

### Why do we need this mathematics?

The dimensions prove how one hidden state can become one candidate score per vocabulary item.

This is the exact bridge between:

```text
Transformer representation space
→
vocabulary decision space
```

---

## Question 9 — What exactly is a logit?

A logit is an unnormalized score.

Suppose the model produces:

$$
z=[4,2,1]
$$

These are not yet probabilities.

For probabilities we want values that satisfy:

$$
0\le P_i\le1
$$

and:

$$
\sum_iP_i=1
$$

The logits $[4,2,1]$ do not satisfy that requirement.

So we use softmax.

---

## Question 10 — Why does softmax turn logits into probabilities?

Softmax is:

$$
P(i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

### How should you read this?

For one candidate token $i$:

1. take its logit $z_i$
2. exponentiate it: $e^{z_i}$
3. divide by the sum of exponentiated logits for every candidate token

The division makes all probabilities sum to 1.

### Numerical example

For:

$$
z=[4,2,1]
$$

we compute:

$$
e^4\approx54.60
$$

$$
e^2\approx7.39
$$

$$
e^1\approx2.72
$$

Total:

$$
54.60+7.39+2.72=64.71
$$

Therefore:

$$
P_1
=
\frac{54.60}{64.71}
\approx0.844
$$

$$
P_2
=
\frac{7.39}{64.71}
\approx0.114
$$

$$
P_3
=
\frac{2.72}{64.71}
\approx0.042
$$

Check:

$$
0.844+0.114+0.042=1.000
$$

So:

```text
logits           probabilities
4        →       0.844
2        →       0.114
1        →       0.042
```

### Why do we need softmax mathematics?

Because later sampling methods need a valid categorical probability distribution.

Top-k, top-p, temperature, and ordinary random sampling all depend on understanding how logits relate to probabilities.

### Edge case — Why does softmax use exponentials?

Exponentials have useful properties:

- outputs are always positive
- larger logits receive disproportionately larger weights
- normalization is straightforward

### Edge case — Numerical overflow

If a logit is extremely large, directly evaluating $e^{z_i}$ can overflow numerically.

Implementations therefore usually use a numerically stable form.

Let:

$$
m=\max_j z_j
$$

Then compute:

$$
P(i)
=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}}
$$

Subtracting the same constant from every logit does not change the softmax probabilities.

This keeps the exponentials numerically manageable.

### Why do we need this edge-case mathematics?

Because mathematically equivalent formulas are not always equally safe on real finite-precision hardware.

Production inference must care about numerical stability, not only symbolic correctness.

---

## Question 11 — Do we always need to compute softmax for greedy decoding?

No.

Softmax preserves the ordering of logits.

Therefore:

$$
\arg\max_i z_i
=
\arg\max_i P(i)
$$

Meaning:

> The token with the largest logit is also the token with the largest softmax probability.

So if all we want is greedy argmax, an implementation does not necessarily need the full normalized probability distribution.

### Why is this useful?

It reminds us that some mathematics is required for interpretation or sampling, while some decoding operations can use simpler equivalent computations.

---

## Question 12 — Does the model itself decide which next token to output?

It is useful to separate two stages.

### Stage 1 — Model

```text
context
→ logits
```

### Stage 2 — Decoding policy

```text
logits / probabilities
→ selected next token
```

For example, the model might imply:

```text
Paris      0.82
Lyon       0.03
located    0.02
...
```

Then the generation policy may:

```text
choose the maximum
sample randomly
apply temperature
apply top-k
apply top-p
maintain multiple beams
```

The model weights can remain exactly the same while generation behavior changes because the decoding policy changes.

---

## Question 13 — Once one token is selected, what happens next?

Suppose the current sequence is:

$$
x_1,x_2,\ldots,x_n
$$

The decoder selects:

$$
x_{n+1}
$$

Now the sequence becomes:

$$
x_1,x_2,\ldots,x_n,x_{n+1}
$$

The next prediction is:

$$
P\left(
 x_{n+2}
 \mid
 x_1,x_2,\ldots,x_n,x_{n+1}
\right)
$$

Then the process repeats.

This is **autoregressive generation**.

### Why do we need this mathematics?

Because it makes a crucial dependency explicit:

$$
\text{prediction at step }t+1
\text{ depends on the tokens chosen before it}
$$

This is the mathematical reason ordinary generation eventually becomes sequential.

We will study that deeply in 8.14.

---

## Question 14 — What important edge cases can happen during inference?

### Edge case 1 — Empty visible user message

An apparently empty user message does not necessarily produce zero model-input tokens.

A chat system may still add:

```text
system instructions
role tokens
message separators
generation-control tokens
```

So:

```text
empty visible text
≠
empty model context
```

### Edge case 2 — Extremely long input

If the prompt is longer than the supported context budget, a serving system may:

```text
reject it
truncate it
summarize older content
use a larger-context model
```

The exact behavior is a system decision.

### Edge case 3 — Equal or nearly equal logits

Suppose two candidate logits are:

$$
z_a=8.000001
$$

and:

$$
z_b=8.000000
$$

They are extremely close.

Small numerical changes from quantization, floating-point precision, different kernels, or hardware could potentially reverse their ordering.

This matters most when a deterministic decoding rule chooses only the maximum.

### Edge case 4 — Finite precision

Inference rarely uses infinite-precision mathematics.

It may use:

```text
FP32
BF16
FP16
FP8
INT8
INT4
```

The mathematical model may be the same, but finite precision introduces approximation.

### Edge case 5 — Wrong tokenizer for the model

The tensor shapes may still look valid, yet the meaning of token IDs can be completely wrong.

This is a dangerous silent failure.

---

# 8.1 Mental Model

The complete mathematical picture is:

$$
\boxed{
 x_{\le t}
 \;\longrightarrow\;
 h_t
 \;\longrightarrow\;
 z
 \;\longrightarrow\;
 P(x_{t+1}\mid x_{\le t})
 \;\longrightarrow\;
 \text{token selection}
}
$$

Read this as:

```text
current context
→ contextual hidden representation
→ vocabulary scores
→ next-token probabilities
→ decoding decision
```

A language model does not directly emit a finished response.

It repeatedly solves a next-token prediction problem.

---

# Bridge to 8.2

We now understand the whole inference pipeline.

But our equations keep using symbols like:

$$
x_1,x_2,\ldots,x_n
$$

What exactly are these $x_i$ values?

They are not English words.

They are tokens.

That creates the next question:

> **What exactly is a prompt token, how does text become tokens, and why does token count matter computationally?**

---

# 8.2 Prompt Tokens

## Question 1 — What exactly is a token?

A token is an element from the tokenizer's finite vocabulary.

Let the vocabulary be:

$$
V=\{v_1,v_2,\ldots,v_{|V|}\}
$$

where $|V|$ means:

> the number of entries in the vocabulary.

The tokenizer converts text into token IDs:

$$
\text{text}
\;\longrightarrow\;
[x_1,x_2,\ldots,x_n]
$$

where each token ID satisfies:

$$
x_i\in\{0,1,\ldots,|V|-1\}
$$

### How should you read this?

If:

$$
|V|=100{,}000
$$

then every token is assigned an ID from some finite set of vocabulary indices.

So the tokenizer converts an effectively unlimited space of strings into a finite alphabet that the neural network knows how to represent.

### Why do we need this mathematics?

Because both the embedding matrix and output vocabulary projection depend on a fixed vocabulary size.

Without a finite token vocabulary, the model could not maintain a finite embedding table or finite output layer.

---

## Question 2 — Is one word always one token?

No.

A word may become:

```text
one token
multiple subword tokens
multiple byte-oriented pieces
```

depending on the tokenizer.

For example, conceptually:

```text
unhappiness
```

might become:

```text
un + happiness
```

or some completely different segmentation.

The exact result depends on the model's tokenizer.

### Why not simply create one token for every word?

A pure word-level vocabulary creates problems with:

```text
rare words
new names
misspellings
morphological variants
code
URLs
multiple languages
```

Subword or byte-aware tokenization gives the model reusable smaller pieces.

### Edge case — One visible character can require multiple bytes

Unicode characters do not all have the same byte representation.

A byte-aware tokenizer therefore may treat visually similar strings very differently depending on their underlying encoding and learned vocabulary merges.

This is one reason **character count is not token count**.

---

## Question 3 — Why does token count matter for inference?

Suppose two prompts contain similar semantic information but tokenize into:

$$
n_1=1000
$$

and:

$$
n_2=1500
$$

The second prompt contains:

$$
\frac{1500-1000}{1000}=0.5=50\%
$$

more token positions.

The Transformer therefore has substantially more sequence positions to process.

### Why do we need mathematics here?

Because token count becomes the sequence-length variable $n$ in attention equations.

For ordinary dense self-attention, the pairwise attention-score matrix has shape:

$$
n\times n
$$

So the number of pairwise score locations is:

$$
n^2
$$

### Numerical example

For:

$$
n=1000
$$

we have:

$$
n^2=1{,}000{,}000
$$

If the prompt doubles to:

$$
n=2000
$$

then:

$$
n^2=4{,}000{,}000
$$

So doubling sequence length creates four times as many pairwise attention-score locations.

### Important edge case

Do **not** interpret this as:

> “The entire inference runtime is always exactly quadratic.”

That would be too simplistic.

Other costs include:

- feed-forward layers
- projection matrices
- memory movement
- optimized attention kernels
- model architecture
- hardware implementation

The $n^2$ term specifically explains why full dense attention over a known prompt can become expensive as context length grows.

We will return to this in **8.3 Prefill**.

---

## Question 4 — How do all prompt-token embeddings form one tensor?

For each token $x_i$:

$$
e_i=E[x_i]
$$

If the prompt has $n$ tokens, we stack their vectors:

$$
X=
\begin{bmatrix}
 e_1^T\\
 e_2^T\\
 \vdots\\
 e_n^T
\end{bmatrix}
$$

Therefore:

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

### How should you interpret this matrix?

- number of rows = number of tokens
- number of columns = number of features in each token representation

Suppose:

$$
n=2048
$$

and:

$$
d_{model}=4096
$$

Then:

$$
X\in\mathbb{R}^{2048\times4096}
$$

The number of scalar values is:

$$
2048\times4096
=8{,}388{,}608
$$

### Why do we need this mathematics?

Because this tensor shape is the starting point for the Transformer forward pass.

Later, when we derive $Q$, $K$, $V$, attention matrices, and KV-cache sizes, these dimensions will matter constantly.

---

## Question 5 — Are all prompt tokens actually typed by the user?

No.

In chat models, the serving system typically serializes multiple pieces of information.

A conceptual prompt may contain:

```text
system message
conversation history
current user message
role delimiters
special control tokens
```

So total prompt length can be written as:

$$
N_{prompt}
=
N_{system}
+
N_{history}
+
N_{user}
+
N_{special}
$$

### Why do we need this equation?

Because it explains why a tiny latest message can still create a very large prompt.

Example:

$$
N_{system}=500
$$

$$
N_{history}=6000
$$

$$
N_{user}=50
$$

$$
N_{special}=50
$$

Then:

$$
N_{prompt}=500+6000+50+50=6600
$$

So the user may type only 50 new tokens while the model actually receives 6600 prompt tokens.

---

## Question 6 — What are special tokens?

Special tokens are reserved vocabulary entries with structural or control meanings.

Examples may represent concepts such as:

```text
beginning of sequence
end of sequence
message-role boundary
separator
padding
```

Mathematically, they are still vocabulary entries.

They still have token IDs and embeddings.

The difference is their learned or system-defined role.

### Edge case — What if user text looks like a special token?

Tokenizer APIs often distinguish:

```text
literal user text
```

from:

```text
actual reserved control-token ID
```

This distinction is important so ordinary user text does not accidentally alter the chat template structure.

---

## Question 7 — What is the context window?

Let:

- $C$ = total context capacity
- $P$ = number of prompt tokens
- $G$ = generated tokens retained in the active context

A basic budget relationship is:

$$
P+G\le C
$$

### How should you read this?

> Prompt tokens and generated tokens must fit inside the model/runtime's active context budget.

### Numerical example

Suppose:

$$
C=8192
$$

and:

$$
P=7000
$$

Then the theoretical remaining space is:

$$
G\le8192-7000
$$

Therefore:

$$
G\le1192
$$

### Why do we need this mathematics?

Because it shows that prompt length and generation capacity compete for the same finite resource.

A longer prompt can leave less room for new output.

### Important edge case

Real APIs may impose a smaller generation maximum than $C-P$ due to:

```text
service policy
memory limits
batching strategy
user-specified max tokens
model configuration
```

So $C-P$ is an upper context bound, not necessarily the exact amount the serving system will allow.

---

## Question 8 — What happens if the prompt exceeds the context limit?

Suppose:

$$
P>C
$$

The complete prompt cannot fit in the available context.

A system may:

```text
reject the request
truncate older content
truncate another region
summarize previous conversation
use a larger-context model
```

### Why is naive truncation dangerous?

Suppose we have:

$$
[x_1,x_2,\ldots,x_{10000}]
$$

but only 8192 tokens can be retained.

If the system removes the first 1808 tokens, it might accidentally remove:

```text
system instructions
important definitions
user constraints
earlier facts needed later
```

So truncation is not mathematically equivalent to preserving the original prompt.

The new conditional distribution becomes:

$$
P(x_{t+1}\mid \text{truncated context})
$$

rather than:

$$
P(x_{t+1}\mid \text{full original context})
$$

Those two distributions can be different.

---

## Question 9 — Why can visually similar prompts have very different token counts?

Token count depends on tokenizer segmentation, not merely characters or words.

Things that can increase token count include:

```text
rare vocabulary
unusual spacing
code
long numbers
URLs
hashes
Unicode symbols
misspellings
some languages or scripts
```

### Edge case — Long numbers

A string like:

```text
7392048129384710293847
```

may be broken into many tokens.

The model does not automatically receive it as one abstract integer object.

### Edge case — Random hashes and IDs

Random strings have little reusable language structure, so tokenization can be inefficient.

### Edge case — Language differences

Different languages can consume different numbers of tokens for similar semantic content depending on how the tokenizer vocabulary was trained.

This can affect both context capacity and cost.

---

## Question 10 — Does token count alone determine prompt cost?

No.

Sequence length is important, but compute also depends on quantities such as:

$$
n=\text{sequence length}
$$

$$
d_{model}=\text{hidden width}
$$

$$
L=\text{number of Transformer layers}
$$

as well as:

```text
attention-head structure
feed-forward width
precision
sparsity
kernel implementation
hardware
```

### Why do we need this mathematical parameterization?

Because the statement:

> “Long prompts are expensive.”

is true but incomplete.

A 4000-token prompt on a small model and a 4000-token prompt on a very large model do not have the same compute cost.

The actual workload comes from tensor dimensions and architecture.

---

## Question 11 — What are the important prompt-token edge cases?

### Edge case 1 — 100 words can become far more than 100 tokens

Word count and token count are different measurements.

### Edge case 2 — Short latest message, long actual prompt

Conversation history can dominate the token count.

### Edge case 3 — Rare strings explode into many tokens

URLs, IDs, code, encoded text, unusual scripts, and random strings can consume context quickly.

### Edge case 4 — Padding during batching

If sequences of different lengths are grouped into a batch, implementations may need padding or more advanced packing/batching strategies.

Padding positions must not behave like real content positions, so attention masks or specialized kernels are used.

### Edge case 5 — Truncation changes meaning

Removing tokens changes the information available to the model.

### Edge case 6 — Special-token injection

A serving system must distinguish untrusted user content from actual control tokens.

### Edge case 7 — Tokenizer/model mismatch

A model used with the wrong tokenizer may receive token IDs whose learned meanings do not match the model's embedding table.

The code can still run.

The tensor dimensions can still be legal.

But the semantics are corrupted.

This is a classic silent failure.

---

# 8.2 Mental Model

From the model's perspective, a prompt is not a sentence.

It is first a sequence of vocabulary IDs:

$$
[x_1,x_2,\ldots,x_n]
$$

Then it becomes a matrix of continuous token representations:

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

The sequence length $n$ affects:

```text
context usage
attention work
memory
prefill latency
generation budget
```

Memory line:

> **LLMs do not process words directly. They process token IDs and tensors.**

---

# Final Bridge — Why Does This Naturally Lead to Prefill?

We now know that before generation begins, the complete prompt already exists as:

$$
[x_1,x_2,\ldots,x_n]
$$

Every one of those prompt tokens is already known.

That creates the next question:

> If every prompt token is already available, does the model really need to process them one at a time?

The answer is no.

The model can process the known prompt positions together while respecting causal attention.

That first large prompt-processing stage is called:

```text
PREFILL
```

And that is where Part 2 begins.

---

# Part 1 Summary

The story so far is:

```text
User text
↓
Tokenizer
↓
Token IDs
↓
Embedding vectors
↓
Transformer
↓
Final hidden state
↓
LM head
↓
Vocabulary logits
↓
Probability distribution
↓
Decoder selects next token
↓
Token is appended
↓
Repeat
```

Core equations:

### Next-token prediction

$$
P(x_{t+1}\mid x_1,x_2,\ldots,x_t)
$$

### Token embedding lookup

$$
e_i=E[x_i]
$$

### Prompt representation matrix

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

### LM-head projection

$$
z=W_{LM}h_t+b
$$

### Softmax

$$
P(i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

### Context budget

$$
P+G\le C
$$

The next detailed-study file begins with:

```text
8.3 Prefill phase
8.4 Decode phase
```

The key question carrying us forward is:

> **Since all prompt tokens are already known, how does the model process them efficiently before it begins generating unknown output tokens?**
