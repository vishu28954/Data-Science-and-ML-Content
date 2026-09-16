# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 1

## 8.1 What happens during inference?
## 8.2 Prompt tokens

This note follows a question-driven learning style. Each idea should create the need for the next idea.

The central story is:

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
Now all prompt tokens are known.
Can the model process the whole known prompt efficiently?
    ↓
Next topic: Prefill
```

---

# 8.1 What Happens During Inference?

## Question 1 — What does “inference” actually mean for an LLM?

Inference means using a model whose parameters have already been learned to compute predictions for new input.

For a decoder-only language model, the basic prediction problem is:

> Given all tokens seen so far, what should the probability distribution over the next token be?

If the current token sequence is

\[
x_1,x_2,\ldots,x_t,
\]

then the model estimates

\[
P(x_{t+1}\mid x_1,x_2,\ldots,x_t).
\]

### Why do we need this mathematics?

This equation defines the actual problem the model is solving.

It tells us that the model is not directly trying to write a paragraph, sentence, or answer all at once. At each generation step, it produces a conditional probability distribution for only the next token.

Everything later in this chapter — greedy decoding, temperature, top-k, top-p, beam search, stop tokens, and sequential generation — operates on or follows from this probability distribution.

### Example

Prompt:

```text
The capital of France is
```

Conceptually, the model estimates something like:

```text
Paris      0.82
Lyon       0.03
located    0.02
France     0.01
...        ...
```

The exact values depend on the model and tokenization.

The important point is that inference gives us a distribution, not automatically a final chosen word.

---

## Question 2 — How is inference different from training?

During training, the model changes its parameters so that future predictions improve.

During ordinary inference, the parameters remain fixed.

Let the model parameters be

\[
\theta.
\]

Training conceptually solves an optimization problem such as

\[
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta),
\]

where \(\mathcal{L}\) is the training loss.

During inference, we instead evaluate

\[
P_{\theta^*}(x_{t+1}\mid x_{\le t}),
\]

using the already learned \(\theta^*\).

### Why do we need this mathematics?

The distinction matters because inference is evaluation, not optimization.

During training, we need gradients such as

\[
\frac{\partial \mathcal{L}}{\partial \theta},
\]

and an optimizer updates the weights.

During inference, we normally do not need those gradients, so systems can avoid storing the intermediate state required for backpropagation.

That is one reason inference uses substantially less memory than training the same model, although inference can still be expensive.

### What if we fine-tune while serving?

Then we are no longer doing pure inference for those updates. We have introduced a training or adaptation step.

### What if the conversation teaches the model something new?

The model can use information placed in its context, but that does not imply that its model weights were updated. The information affects the current conditional prediction because it is part of \(x_{\le t}\).

---

## Question 3 — What is the complete path from my text to the next-token probabilities?

At a high level:

```text
Raw text
→ tokenizer
→ token IDs
→ token representations
→ Transformer blocks
→ final hidden states
→ language-model head
→ logits
→ probability distribution
→ token-selection rule
```

We will study several of these stages separately later.

For now, we want the map.

---

## Question 4 — Why does the model need numbers instead of text?

Neural networks operate on numerical tensors.

So the text must first be converted into discrete token IDs.

Suppose a tokenizer converts the prompt into

\[
[x_1,x_2,\ldots,x_n]
=
[464,3139,286,4881,318].
\]

The numbers here are only illustrative.

Each ID indexes a row in an embedding table.

If the vocabulary contains \(|V|\) tokens and the model dimension is \(d_{model}\), then the embedding matrix can be represented as

\[
E \in \mathbb{R}^{|V|\times d_{model}}.
\]

For token ID \(x_i\), the model retrieves

\[
e_i = E[x_i],
\]

where

\[
e_i \in \mathbb{R}^{d_{model}}.
\]

### Why do we need this mathematics?

This equation explains how a discrete symbolic object — a token ID — becomes a continuous vector that a neural network can process.

Without this mapping, matrix multiplications inside the Transformer would have no numerical input representation.

---

## Question 5 — Is the embedding vector enough to tell the model where a token occurs?

No.

The token embedding primarily identifies token content. A Transformer also needs position information so that

```text
Dog bites man
```

and

```text
Man bites dog
```

are not treated as the same unordered collection of tokens.

Different architectures introduce position differently.

Some use explicit positional embeddings. Many modern decoder-only LLMs use a positional mechanism such as rotary positional embeddings (RoPE) inside attention rather than simply adding a learned position vector to the token embedding.

The important conceptual point is:

\[
\text{representation used by attention}
=
\text{token information}
+
\text{position-dependent information}.
\]

This is conceptual rather than a universal literal addition formula.

### Edge case — What if position information were removed?

Self-attention by itself is permutation-equivariant. Without some mechanism that distinguishes positions, the model would have difficulty representing token order correctly.

---

## Question 6 — What happens inside the Transformer during inference?

The token representations pass through a stack of Transformer blocks.

A simplified block contains operations such as:

```text
normalization
→ self-attention
→ residual connection
→ feed-forward network
→ residual connection
```

For our current chapter, the most important fact is that the Transformer converts each token position into a contextual hidden representation.

After the final layer, suppose the hidden state at position \(t\) is

\[
h_t \in \mathbb{R}^{d_{model}}.
\]

This vector contains the model's contextual representation of the sequence up to that position, subject to causal attention.

### Why do we need this mathematical representation?

The next stage must convert the hidden state into a score for every possible vocabulary token.

So we need a vector with dimension \(d_{model}\) that can be projected into vocabulary space.

---

## Question 7 — How does a hidden state become scores for every possible next token?

The model uses a language-model output projection, often called the LM head.

A simple form is

\[
z = W_{LM} h_t + b,
\]

where

\[
W_{LM}\in\mathbb{R}^{|V|\times d_{model}},
\]

\[
h_t\in\mathbb{R}^{d_{model}},
\]

and therefore

\[
z\in\mathbb{R}^{|V|}.
\]

Each component \(z_i\) is a logit for token \(i\).

If the vocabulary size is 100,000, then the model produces 100,000 logits for the next token.

### Why do we need this mathematics?

The matrix dimensions explain exactly how one contextual vector becomes one score per vocabulary token.

This is the bridge from hidden representation space to vocabulary decision space.

### Edge case — Is the LM-head matrix always separate from the input embedding matrix?

Not necessarily. Some models tie the output projection weights to the input embedding weights, while others may use separate parameterization. The conceptual role remains the same: map hidden state to vocabulary logits.

---

## Question 8 — What exactly is a logit?

A logit is an unnormalized score.

For example:

\[
z=[4.0,2.0,1.0].
\]

We should not interpret these as probabilities because

\[
4+2+1\neq1.
\]

They can also be negative.

To obtain probabilities, we apply softmax:

\[
P(i)=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

For \(z=[4,2,1]\):

\[
e^4\approx54.60,
\]

\[
e^2\approx7.39,
\]

\[
e^1\approx2.72.
\]

Total:

\[
54.60+7.39+2.72=64.71.
\]

Therefore approximately:

\[
P_1=\frac{54.60}{64.71}\approx0.844,
\]

\[
P_2=\frac{7.39}{64.71}\approx0.114,
\]

\[
P_3=\frac{2.72}{64.71}\approx0.042.
\]

Now the values satisfy

\[
\sum_i P(i)=1.
\]

### Why do we need softmax mathematics?

Later decoding methods need a valid categorical probability distribution.

Greedy decoding needs to know which token has the greatest score; sampling methods need normalized probabilities from which a token can be drawn.

Temperature, top-k, and top-p all operate around this logit/probability stage.

### Edge case — Do we always need to explicitly compute softmax for greedy decoding?

No. Because softmax is monotonically increasing with respect to each logit when the other logits are fixed, the token with maximum logit is also the token with maximum softmax probability:

\[
\arg\max_i z_i
=
\arg\max_i P(i).
\]

So an implementation doing pure greedy argmax can avoid computing a full normalized probability distribution if it does not need it for anything else.

---

## Question 9 — Does the model itself “choose” the next token?

It is useful to separate two stages:

```text
Model
→ produces logits/probability information

Decoder / generation policy
→ chooses the next token
```

The model may assign probabilities such as

```text
Paris     0.82
Lyon      0.03
located   0.02
...
```

Then a decoding strategy decides whether to:

```text
choose the maximum
sample randomly
restrict to top-k
restrict to top-p
maintain multiple beams
```

This distinction will become central later.

### What if we change temperature but not model weights?

The underlying model is the same. We are changing how its output logits are transformed or sampled at inference time, not retraining it.

---

## Question 10 — Once one token is selected, what happens next?

Suppose the prompt is

\[
x_1,\ldots,x_n.
\]

The model predicts and the decoder selects

\[
x_{n+1}.
\]

The sequence becomes

\[
x_1,\ldots,x_n,x_{n+1}.
\]

Now the model must estimate

\[
P(x_{n+2}\mid x_1,\ldots,x_n,x_{n+1}).
\]

Then it selects \(x_{n+2}\), appends it, and repeats.

This is autoregressive generation.

### Why do we need this conditional-probability mathematics?

It makes the dependency explicit:

\[
\text{next prediction depends on the exact tokens generated before it.}
\]

This dependency will later explain why ordinary text generation is sequential.

---

## Question 11 — What are the main edge cases in the inference pipeline?

### Edge case 1 — Empty or almost-empty user text

The actual model input may still not be empty because chat systems often apply a template containing system, role, separator, or generation-control tokens.

So:

```text
empty visible user text
≠ necessarily zero model-input tokens
```

### Edge case 2 — Extremely long prompt

If tokenized prompt length exceeds the supported context budget, the serving system must use some policy such as rejection, truncation, summarization, or a model/runtime mechanism that supports a larger context.

The model cannot simply assume unlimited input length.

### Edge case 3 — Unknown characters

Modern subword/byte-aware tokenizers are generally designed to represent arbitrary text without a classic `<UNK>` failure for ordinary Unicode/byte sequences, but behavior depends on the tokenizer. Rare text may tokenize very inefficiently into many pieces.

### Edge case 4 — Special tokens inside user text

A tokenizer or chat formatter may distinguish literal text that resembles a special token from an actual reserved special-token ID. Systems must avoid accidentally allowing user text to escape the intended conversation template.

### Edge case 5 — Numerical ties in logits

Two or more tokens can in principle have equal logits. A deterministic argmax implementation then needs a tie-breaking convention, often inherited from the numerical library or token order.

### Edge case 6 — Finite-precision arithmetic

Real inference uses finite precision such as FP16, BF16, FP8, or quantized weights. Small numerical differences can slightly affect logits and, in near-tie situations, potentially change the selected token.

---

# 8.1 Mental Model

The simplest mathematically correct picture is:

\[
\boxed{
\text{context }x_{\le t}
\rightarrow
h_t
\rightarrow
z
\rightarrow
P(x_{t+1}\mid x_{\le t})
\rightarrow
\text{token selection}
}
\]

A language model does not directly output a finished answer.

It repeatedly constructs a distribution over the next token.

---

# Bridge to 8.2

We now understand the whole inference map.

But there is an unresolved problem at the very beginning:

> We keep writing \(x_1,x_2,\ldots,x_n\). What exactly are these \(x_i\)?

The model does not receive English words.

It receives tokens.

So the next question is:

> What exactly is a prompt token, how is text converted into tokens, and why does token count matter computationally?

That leads to 8.2.

---

# 8.2 Prompt Tokens

## Question 1 — What exactly is a token?

A token is an element from the tokenizer's finite vocabulary.

Let the vocabulary be

\[
V=\{v_1,v_2,\ldots,v_{|V|}\}.
\]

The tokenizer maps text into a sequence of vocabulary IDs:

\[
\text{text}\rightarrow[x_1,x_2,\ldots,x_n],
\]

where

\[
x_i\in\{0,1,\ldots,|V|-1\}.
\]

The exact indexing convention depends on the tokenizer.

### Why do we need this mathematics?

A tokenizer converts an unbounded space of possible strings into a finite discrete alphabet that the neural network can index.

The Transformer architecture needs a finite vocabulary because the input embedding table and final LM head both depend on \(|V|\).

---

## Question 2 — Is one word always one token?

No.

Depending on the tokenizer, a word can become:

```text
one token
multiple subword tokens
one or more byte-level pieces
```

Likewise, punctuation and whitespace may participate in token boundaries.

For example, conceptually:

```text
unhappiness
```

might be segmented as something like

```text
un + happiness
```

or another decomposition, depending on vocabulary and tokenizer training.

### Why not simply use whole words?

A pure word-level vocabulary creates several problems:

```text
huge vocabulary
rare words
new names
misspellings
morphological variants
multiple languages
```

Subword or byte-based tokenization allows the model to represent unseen or rare strings from reusable smaller pieces.

---

## Question 3 — Why does tokenization affect inference cost?

Suppose two prompts communicate roughly the same information but tokenize into different lengths:

\[
n_1=1000
\]

and

\[
n_2=1500.
\]

The second prompt gives the Transformer 50% more token positions to process.

The model does not care that both prompts contain a similar number of visible words. Its computational workload is driven much more directly by token count.

### Why do we need mathematics here?

Token count becomes the sequence-length variable \(n\) in later attention equations.

For ordinary dense self-attention over a prompt, the attention-score matrix has shape

\[
n\times n.
\]

So the number of pairwise attention-score locations is

\[
n^2.
\]

If \(n\) doubles from 1,000 to 2,000:

\[
1000^2 = 1,000,000
\]

but

\[
2000^2 = 4,000,000.
\]

The pairwise score count becomes four times larger.

This does not mean total end-to-end inference cost is exactly proportional to \(n^2\), because other operations such as feed-forward layers and optimized attention kernels also matter. But the equation explains why long prompts can make the attention portion expensive.

We will study this more carefully in Prefill.

---

## Question 4 — How does a token ID become a vector?

Token IDs are categorical integers. The integer value itself should not imply magnitude.

Token ID 9000 is not “nine times larger” in meaning than token ID 1000.

So we do not feed the raw scalar ID directly into the network.

Instead, we use an embedding lookup.

If

\[
E\in\mathbb{R}^{|V|\times d_{model}},
\]

then token \(x_i\) receives

\[
e_i=E[x_i].
\]

For a prompt of \(n\) tokens, stacking the embeddings gives

\[
X=
\begin{bmatrix}
 e_1^T\\
 e_2^T\\
 \vdots\\
 e_n^T
\end{bmatrix}
\in\mathbb{R}^{n\times d_{model}}.
\]

### Why do we need this mathematics?

This matrix shape will appear repeatedly later.

If

\[
n=2048
\]

and

\[
d_{model}=4096,
\]

then the prompt representation contains

\[
2048\times4096=8,388,608
\]

scalar elements before considering additional intermediate tensors.

This makes it clear why both sequence length and model width matter for memory and compute.

---

## Question 5 — Are all prompt tokens written by the user?

No.

In a chat model, the serving system often converts structured messages such as

```text
system: You are a helpful assistant.
user: Explain gravity.
```

into a model-specific serialized token sequence.

Conceptually it may look like:

```text
<system>
You are a helpful assistant.
<user>
Explain gravity.
<assistant>
```

The actual representation is model-specific.

Therefore total prompt length can be written conceptually as

\[
N_{prompt}
=
N_{system}
+
N_{history}
+
N_{user}
+
N_{template/special}.
\]

### Why do we need this mathematics?

It explains an important practical fact:

The user-visible message length is not necessarily equal to the model-input length.

A long conversation history or system prompt can consume most of the context even if the latest user message is short.

---

## Question 6 — What are special tokens?

Special tokens are reserved vocabulary entries that carry control or structural meaning.

Examples can include concepts such as:

```text
beginning of sequence
end of sequence
message-role boundaries
padding
separator tokens
```

The exact tokens differ across model families.

Mathematically, they are still IDs in the vocabulary:

\[
x_i\in V.
\]

The difference is semantic: the serving format or model training assigns them a structural role.

### Edge case — Can a normal token have the same visible text as a special token?

Tokenizer APIs often distinguish ordinary text encoding from explicitly allowed special tokens. This prevents literal user text from automatically becoming a control token unless the API intentionally permits it.

---

## Question 7 — What is the context window?

The context window is the maximum sequence budget the model/runtime allows for the active context.

Let

\[
C
\]

be the context capacity,

\[
P
\]

be the prompt-token count,

and

\[
G
\]

be the number of generated tokens retained in context.

A basic budget relationship is

\[
P+G\le C.
\]

### Why do we need this mathematics?

It exposes a direct tradeoff:

A longer prompt leaves less room for output if prompt and output share the same fixed context budget.

For example, if

\[
C=8192
\]

and

\[
P=7000,
\]

then the theoretical remaining token budget is at most

\[
8192-7000=1192.
\]

Actual serving systems can impose smaller generation limits for policy, memory, batching, or API reasons.

---

## Question 8 — What happens if the prompt exceeds the context limit?

The answer is system-dependent.

Possible behaviors include:

```text
reject the request
truncate older tokens
truncate from another region
summarize history
use a larger-context model
apply a specialized long-context strategy
```

### Why is naive truncation dangerous?

Suppose the original token sequence is

\[
[x_1,x_2,\ldots,x_{10000}]
\]

but the system can retain only 8192 tokens.

If it simply drops the first 1808 tokens, it may remove:

```text
system instructions
definitions
earlier user constraints
important context
```

So truncation policy is part of system design, not merely a mechanical operation.

---

## Question 9 — Why can two visually similar prompts have very different token counts?

Tokenization depends on the learned vocabulary and encoding scheme, not just character count.

Variables that can affect token count include:

```text
language
rare words
spacing
punctuation
code
long numbers
URLs
Unicode characters
misspellings
```

### Edge case — Long numeric strings

A long identifier such as

```text
7392048129384710293847
```

may break into several tokens rather than being represented as one numeric object.

The model receives token pieces, not an abstract integer value.

### Edge case — URLs and hashes

Random-looking strings have little reusable subword structure and can tokenize inefficiently.

### Edge case — Different languages

Token efficiency can vary across languages depending on how the tokenizer vocabulary was constructed.

This matters because the same semantic content can consume different context budgets.

---

## Question 10 — Does token count alone determine prompt compute?

No.

It is a major variable, but total compute also depends on model architecture and implementation.

Important quantities include:

\[
n = \text{sequence length},
\]

\[
d_{model} = \text{hidden width},
\]

\[
L = \text{number of Transformer layers},
\]

as well as attention-head structure, feed-forward width, precision, sparsity, kernel implementation, and hardware.

A simplified conceptual compute story is:

```text
more tokens
→ more token representations
→ more attention relationships
→ more layer operations
```

### Why do we need this mathematical parameterization?

It prevents the misleading statement:

> “A prompt is expensive only because it has many words.”

The cost is determined by tensor dimensions and model architecture, not natural-language word count.

---

## Question 11 — What are the most important prompt-token edge cases?

### Edge case 1 — User sees 100 words, model sees 150 tokens

Word count is not token count.

### Edge case 2 — Latest user message is short but conversation is long

Total input includes retained conversation history and formatting tokens.

### Edge case 3 — A rare string explodes into many pieces

The context budget can be consumed unexpectedly by code, IDs, encoded text, URLs, or rare scripts.

### Edge case 4 — Padding in batched inference

When multiple sequences are batched, some systems pad shorter sequences to align tensor shapes. Efficient serving implementations often use attention masks, sequence packing, or specialized batching strategies so padded positions do not behave like real content tokens.

### Edge case 5 — Truncation changes semantics

Dropping tokens is not semantically neutral. A prompt that fits after truncation may no longer represent the same instruction.

### Edge case 6 — Special-token injection

Chat systems must correctly separate user content from structural control tokens so user-provided text does not accidentally alter the intended message framing.

### Edge case 7 — Tokenizer/model mismatch

A model must be used with the tokenizer and chat template it expects. A mismatched tokenizer can map text to IDs with completely different learned meanings, making model behavior invalid even though tensor shapes still appear legal.

This is a classic silent-failure case.

---

# 8.2 Mental Model

A prompt is not a string from the model's perspective.

It is a finite sequence of discrete vocabulary IDs:

\[
[x_1,x_2,\ldots,x_n]
\]

which becomes a matrix of continuous representations:

\[
X\in\mathbb{R}^{n\times d_{model}}.
\]

The length \(n\) is fundamental because it affects:

```text
context usage
attention work
memory
prefill latency
generation budget
```

Memory line:

```text
LLMs do not pay for words; they process tokens and tensors.
```

---

# Final Bridge — Why Does This Naturally Lead to Prefill?

We now know that the complete prompt is already available as

\[
[x_1,x_2,\ldots,x_n].
\]

That creates a new question.

If every prompt token is already known before generation starts, do we really need to process them one token at a time?

The answer is no.

The model can process the known prompt positions together while respecting causal attention.

That first large prompt-processing stage is called:

```text
PREFILL
```

And that is where Part 2 begins.

---

# Part 1 Summary

The full story so far is:

```text
User text
↓
Tokenizer
↓
Token IDs x1 ... xn
↓
Embedding / position-aware representations
↓
Transformer
↓
Final hidden state ht
↓
LM head
↓
Vocabulary logits z
↓
Probability distribution P(x_{t+1} | x≤t)
↓
Decoder selects one token
↓
Token is appended
↓
Process repeats
```

Key equations:

\[
P(x_{t+1}\mid x_1,\ldots,x_t)
\]

\[
e_i=E[x_i]
\]

\[
X\in\mathbb{R}^{n\times d_{model}}
\]

\[
z=W_{LM}h_t+b
\]

\[
P(i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

\[
P+G\le C
\]

The next detailed-study file should begin with:

```text
8.3 Prefill phase
8.4 Decode phase
```

The key question carrying us there is:

> Since all prompt tokens are already known, how does the model process them efficiently before it begins generating unknown output tokens?
