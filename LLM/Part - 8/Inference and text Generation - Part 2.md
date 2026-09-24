# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 2

## 8.3 Prefill Phase
## 8.4 Decode Phase

This note continues directly from Part 1.

At the end of Part 1, we had reached this point:

```text
Raw text
→ tokenization
→ prompt token IDs
→ prompt embeddings
```

Suppose the prompt contains:

$$x_1, x_2, \ldots, x_n$$


Every one of those prompt tokens is already known before generation begins.

That creates the next question:

> If the entire prompt already exists, does the model really need to process its tokens one by one?

The answer is no.

That leads to the **prefill phase**.

But prefill immediately creates another problem:

> After the prompt ends, the next output token does not exist yet. Can future generated tokens still be processed together?

That answer is also no.

That leads to the **decode phase**.

The story of this part is therefore:

```text
All prompt tokens are known
        ↓
Process them together
        ↓
PREFILL
        ↓
First next-token distribution is produced
        ↓
Select one token
        ↓
Future token is now known
        ↓
Process the newly available token
        ↓
DECODE
        ↓
Repeat one generated token at a time
        ↓
But are we recomputing old attention information unnecessarily?
        ↓
Next topic: KV Cache
```

---

# 8.3 Prefill Phase

## Question 1 — What exactly is the prefill phase?

The **prefill phase** is the first Transformer forward pass over the known prompt.

Suppose the prompt is:

$$x_1, x_2, \ldots, x_n$$


All $n$ prompt tokens are already available before generation begins.

After embedding and position handling, the prompt can be represented as:

$$X \in \mathbb{R}^{n \times d_{\text{model}}}$$


where:

- $n$ = number of prompt tokens
- $d_{model}$ = hidden width of the Transformer

Instead of waiting for token $x_1$ to be processed before giving the model $x_2$, the model can send all $n$ known prompt positions through the Transformer together.

Conceptually:

```text
Known prompt:
x1 x2 x3 ... xn

        ↓

One large Transformer forward pass

        ↓

Contextual representations for all prompt positions
```

### Why is it called “prefill”?

Because this stage happens **before autoregressive output generation begins**.

It processes the prompt and prepares the model state needed to begin decoding.

### Why do we need this distinction?

Because prompt processing and output generation have very different computational behavior.

Later we will see:

```text
Prefill
→ many known token positions processed together

Decode
→ one newly generated token position becomes available at a time
```

That distinction is fundamental to LLM inference performance.

---

## Question 2 — Since every prompt token is already known, why can the Transformer compute all prompt positions together instead of running one separate forward pass per token?

This is one of the most important concepts in prefill.

The confusion usually comes from interpreting **autoregressive** as:

> “The computer must physically process token 1, then token 2, then token 3.”

That is **not** what autoregressive means.

Autoregressive means:

> **When computing the representation at position $i$, that position is only allowed to use information from itself and earlier positions.**

It describes an **information-dependency rule**, not necessarily the physical order in which the GPU performs the calculations.

### Step 1 — Start with a concrete prompt

Suppose the prompt is:

```text
I love machine learning
```

For simplicity, imagine tokenization gives:

```text
x1 = I
x2 = love
x3 = machine
x4 = learning
```

Before inference begins, the user has already supplied the whole prompt.

So the model already knows:

$[x_1, x_2, x_3, x_4]$

There is no uncertainty about what $x_2$, $x_3$, or $x_4$ are.

This is fundamentally different from generated output tokens, which do not yet exist.

### Step 2 — What does autoregressive actually restrict?

For the prompt above:

```text
Position 1: "I"
can use → I

Position 2: "love"
can use → I, love

Position 3: "machine"
can use → I, love, machine

Position 4: "learning"
can use → I, love, machine, learning
```

But position 1 must **not** use:

```text
love, machine, learning
```

because those are future positions relative to position 1.

Similarly, position 2 must not use:

```text
machine, learning
```

So the model must preserve the following dependency pattern:

| Query position ↓ / Key position → | I | love | machine | learning |
|---|---:|---:|---:|---:|
| **I** | ✅ | ❌ | ❌ | ❌ |
| **love** | ✅ | ✅ | ❌ | ❌ |
| **machine** | ✅ | ✅ | ✅ | ❌ |
| **learning** | ✅ | ✅ | ✅ | ✅ |

This is the autoregressive rule.

### Step 3 — But why can the GPU still compute all four positions together?

Because the identities of all four prompt tokens are already known.

Their representations can be placed into one tensor:

$$X \in \mathbb{R}^{4 \times d_{\text{model}}}$$

Then the Transformer can compute:

$Q=XW_Q$

$K=XW_K$

$V=XW_V$

for all four prompt positions using large matrix multiplications.

For example:

$$Q \in \mathbb{R}^{4 \times d_k}$$


and:

$$K \in \mathbb{R}^{4 \times d_k}$$


Therefore:

$$QK^T \in \mathbb{R}^{4 \times 4}$$

So the hardware can calculate the attention scores for all four query positions in the same matrix operation.

Conceptually, before masking, that matrix contains scores for all pairs:

| Query ↓ / Key → | I | love | machine | learning |
|---|---:|---:|---:|---:|
| **I** | score | score | score | score |
| **love** | score | score | score | score |
| **machine** | score | score | score | score |
| **learning** | score | score | score | score |

But some of these relationships are illegal in an autoregressive model.

That is where the causal mask enters.

### Step 4 — The causal mask removes illegal future attention

Conceptually, for four positions:

$$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\\\ 0 & 0 & -\infty & -\infty \\\\ 0 & 0 & 0 & -\infty \\\\ 0 & 0 & 0 & 0 \end{bmatrix}$$


The attention logits become:

$$A = \frac{QK^T}{\sqrt{d_k}} + M$$


After softmax:

$$\text{softmax}(A)$$


the masked future locations receive probability zero conceptually because:

$$e^{-\infty} = 0$$


So the GPU can calculate many prompt positions together, while the mask still guarantees that each position only uses legal past information.

### The crucial distinction

There are two different questions:

```text
1. What information is position i allowed to use?
   → controlled by causal attention

2. In what physical order must the hardware execute the calculations?
   → many known prompt positions can be calculated in parallel
```

These are not contradictory.

### Step 5 — What happens across Transformer layers?

This becomes even clearer if we think layer by layer.

Suppose layer 4 has already produced:

$$
H^{(4)} = \begin{bmatrix}
h_1^{(4)} \\
h_2^{(4)} \\
h_3^{(4)} \\
h_4^{(4)}
\end{bmatrix}
$$


Layer 5 does **not** need to do:

```text
position 1
then position 2
then position 3
then position 4
```

Instead, it can use the whole matrix $H^{(4)}$ and compute:

$Q=H^{(4)}W_Q$

$K=H^{(4)}W_K$

$V=H^{(4)}W_V$

for all four positions together.

The causal mask determines which positions may exchange information.

So the actual high-level execution is:

```text
All prompt tokens already known
        ↓
Create representations for all positions
        ↓
Transformer Layer 1
→ compute positions together
→ mask future attention
        ↓
Transformer Layer 2
→ compute positions together
→ mask future attention
        ↓
...
        ↓
Final Transformer layer
```

### Why does this work for prefill but not for decode?

Now compare the prompt with generated tokens.

Suppose the prompt is:

```text
I love machine learning
```

The next generated token might be:

```text
because
```

But before prediction, that token is not known.

The model first has to compute:

$$P(x_5 \mid x_1, x_2, x_3, x_4)$$


Then a decoding rule selects $x_5$.

Only after $x_5$ becomes known can the model compute:

$$P(x_6 \mid x_1, x_2, x_3, x_4, x_5)$$

So:

```text
PREFILL

x1   x2   x3   x4
↑    ↑    ↑    ↑
all token identities already known

→ positions can be processed together
```

But:

```text
DECODE

predict x5
    ↓
select x5
    ↓
now x5 exists
    ↓
predict x6
    ↓
select x6
    ↓
now x6 exists
```

cannot be parallelized in the same straightforward way because future token identities depend on earlier generated choices.

### Why do we need this concept?

Because this explains the fundamental computational difference between prefill and decode:

> **Prefill has many known token positions, so the model can exploit parallel matrix computation. Decode has unknown future token identities, so generation must progress autoregressively.**

### Memory line

> **Autoregressive controls information flow, not necessarily hardware execution order.**

And the most important comparison is:

```text
Prefill:
known token identities + causal mask
→ parallel computation across prompt positions

Decode:
future token identities unknown
→ sequential generation across output positions
```

---

## Question 3 — What gets computed during self-attention in prefill?

Let the prompt representation be:

$$X \in \mathbb{R}^{n \times d_{\text{text{model}}}}$$

The attention layer forms Queries, Keys, and Values.

For a simplified single-head view:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Suppose:

$$W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

and:

$$W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

Then:

$$Q \in \mathbb{R}^{n \times d_k}$$

$$K \in \mathbb{R}^{n \times d_k}$$

$$V \in \mathbb{R}^{n \times d_v}$$


### Why do we need these matrix shapes?

Because the shapes reveal what the attention mechanism is doing.

There is one query vector per prompt position.

There is one key vector per prompt position.

There is one value vector per prompt position.

So all prompt positions can participate in the same batched matrix operations.

---

## Question 4 — Why does the attention-score matrix become $n \times n$ ?

The unnormalized attention score matrix comes from:

$$QK^T$$

We have:

$$Q \in \mathbb{R}^{n \times d_k}$$


and:

$$K^T \in \mathbb{R}^{d_k \times n}$$

Therefore:

$$QK^T \in \mathbb{R}^{n \times n}$$

### Dimension check

The multiplication is:

$$(n \times d_k)(d_k \times n) = n \times n$$

So the matrix contains one score for every query-position/key-position pair.

Entry $(i,j)$ answers conceptually:

> How strongly should token position $i$ attend to token position $j$?

### Numerical example

If:

$$n=4$$

then the logical score matrix is:

$$4	\times4$$

or 16 pairwise positions.

If:

$$n=1000$$

then:

$$1000^2=1{,}000{,}000$$

pairwise score positions exist.

If:

$$n=2000$$

then:

$$2000^2=4{,}000{,}000$$

pairwise score positions exist.

### Why do we need this mathematics?

Because this is where the familiar quadratic attention dependence on prompt length comes from.

But there is an important edge case:

> Modern kernels such as memory-efficient or fused attention do not necessarily materialize the entire $n \times n$ matrix in GPU memory.

The **logical attention relationships** are still $n^2$ for ordinary dense attention, but optimized implementations can compute them in tiles and reduce memory traffic.

So:

```text
logical dense attention relationships ∝ n²
≠
every implementation literally stores an n×n matrix
```

That distinction matters in production systems.

---

## Question 5 — Why do we divide attention scores by $\sqrt{d_k}$?

Scaled dot-product attention uses:

$$\frac{QK^T}{\sqrt{d_k}}$$


Why not just use: $QK^T$ ?

The reason is statistical stability.

Suppose the components of a query vector and a key vector are roughly independent with:

$$\mathbb{E}[q_r] = 0$$

$$\mathbb{E}[k_r] = 0$$


and approximately unit variance.

Their dot product is:

$$q \cdot k = \sum_{r=1}^{d_k} q_r k_r$$


If each product contributes roughly variance 1, then the variance of the sum grows approximately as:

$$\text{Var}(q \cdot k) = d_k$$


Therefore the standard deviation grows approximately as:

$$\sqrt{d_k}$$

So as $d_k$ gets larger, raw dot products tend to become larger in magnitude.

Dividing by $\sqrt{d_k}$ gives:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1$$


under this simplified assumption.

### Why do we need this mathematics?

Because softmax is sensitive to scale.

Suppose the scores are:

$$[1,2,3]$$

Softmax is relatively smooth.

But if the scores become:

$$[10,20,30]$$

softmax becomes extremely peaked.

Very large dot products can therefore push softmax into saturated regions where almost all probability goes to one position.

The $\sqrt{d_k}$ scaling keeps attention logits in a more stable range.

### Edge case — Does this derivation exactly describe trained Transformers?

Not exactly.

Real hidden states are not perfectly independent, zero-mean, unit-variance random variables.

The derivation is an intuition for why the scale grows with dimensionality and why dividing by $\sqrt{d_k}$ is useful.

---

## Question 6 — How does the model stop a prompt token from seeing future prompt tokens?

It uses a **causal mask**.

Before softmax, the attention computation can be written as:

$$
A = \frac{QK^T}{\sqrt{d_k}} + M
$$

where $M$ is the causal mask.

For a four-token sequence, conceptually:

$$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\\\ 0 & 0 & -\infty & -\infty \\\\ 0 & 0 & 0 & -\infty \\\\ 0 & 0 & 0 & 0 \end{bmatrix}$$

Then:

$$\text{softmax}(A)$$

turns the masked future entries into probability approximately equal to zero.

Why?

Because:

$$e^{-\infty} = 0$$

conceptually.

So position 1 can attend only to position 1.

Position 2 can attend to positions 1 and 2.

Position 3 can attend to positions 1, 2, and 3.

And so on.

### Why do we need this mathematics?

Because it solves the apparent contradiction:

> “How can all prompt positions be processed in parallel without leaking future information?”

The answer is:

```text
Compute them together
+
mask illegal future dependencies
```

---

## Question 7 — What does the full attention equation look like during prefill?

A simplified single-head form is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Let:

$$S = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$$

Then:

$$S \in \mathbb{R}^{n \times n}$$

and:

$$V \in \mathbb{R}^{n \times d_v}$$

Therefore:

$$SV \in \mathbb{R}^{n \times d_v}$$

### What does this mean?

Every prompt position gets a new representation that is a weighted combination of allowed value vectors.

For position $i$:

$$o_i = \sum_{j=1}^{i} \alpha_{ij} v_j$$

where:

$$\sum_{j=1}^{i} \alpha_{ij} = 1$$


### Why do we need this mathematics?

It shows precisely what “attending to previous context” means.

The output at position $i$ is built by mixing information from positions available to it under the causal mask.

---

## Question 8 — If all prompt hidden states are computed, which one predicts the first generated token?

For a decoder-only language model, the final prompt position is normally the position used to produce the next-token logits.

Suppose the prompt ends at position $n$.

After the final Transformer layer we have:

$$
h_1,h_2,\dots,h_n
$$

The next-token prediction is based on:

$$
h_n
$$

The LM head produces:

$$z_{n+1} = W_{\text{LM}}h_n + b$$


and then conceptually:

$$P(x_{n+1} \mid x_1, \ldots, x_n) = \text{softmax}(z_{n+1})$$


### Then why compute hidden states for all earlier prompt positions?

Because the last position depends on them through causal self-attention.

Also, the attention information created for those positions becomes useful during later decoding.

We will make that second point concrete in the KV-cache lesson.

---

## Question 9 — What does prefill cost depend on?

There are multiple operations inside each Transformer layer.

A simplified view includes:

### Projection work

Creating $Q$, $K$, and $V$ from $X$ involves matrix multiplications that scale roughly linearly with sequence length $n$ for fixed model width.

Conceptually:

$$
O(n,d_{model},d_k)
$$

per projection.

### Dense attention-score work

The matrix:

$$
QK^T
$$

requires approximately:

$$
O(n^2d_k)
$$

operations per head in the standard dense formulation.

### Feed-forward work

The feed-forward network also processes every token position.

A simplified cost is approximately:

$$
O(n,d_{model},d_{ff})
$$

### Why do we need this breakdown?

Because saying:

> “Prefill is quadratic.”

is incomplete.

The attention-score component has an $n^2$ dependence, but projection and feed-forward operations have different scaling with $n$.

For many real LLM shapes and prompt lengths, all of these terms matter.

The exact bottleneck depends on:

- model size
- sequence length
- attention architecture
- GPU/accelerator
- numerical precision
- kernel implementation
- batching

---

## Question 10 — What is Time to First Token, and how does prefill affect it?

**Time to First Token (TTFT)** is the delay between submitting a request and receiving the first generated token.

At a model-compute level, a long prompt generally requires more prefill work before the model can produce the first next-token distribution.

Conceptually:

$$\text{TTFT} \approx \text{queueing} + \text{tokenization} + \text{prefill} + \text{first-token selection} + \text{serving/network overhead}$$

This is not an exact universal formula.

It is a systems decomposition.

### Why do we need this distinction?

Because TTFT is not identical to prefill time.

A request may have slow TTFT because of:

- long prompt processing
- request queueing
- cold model loading
- batching delay
- communication overhead

So:

```text
long prompt
→ often larger prefill cost
→ often higher TTFT

but

TTFT ≠ prefill time only
```

---

## Question 11 — What are important prefill edge cases?

### Edge case 1 — Prompt length is one token

If:

$$
n=1
$$

then the logical attention score matrix is:

$$
1 \times1
$$

There are no earlier prompt positions to combine with.

Prefill still exists, but it is very small.

### Edge case 2 — Empty visible user input

A chat system may still create prompt tokens from system messages, role markers, or templates.

So visible user length zero does not imply:

$$
n=0
$$

for actual model input.

### Edge case 3 — Batched prompts have different lengths

Suppose one request has 200 tokens and another has 2000.

A naive padded batch may require alignment to a common tensor shape.

Serving systems can use padding masks, sequence packing, or more advanced scheduling to avoid wasting too much computation.

### Edge case 4 — Extremely long context

As $n$ grows, attention and memory pressure increase.

Long-context models therefore rely heavily on efficient attention kernels, memory management, and sometimes architectural changes.

### Edge case 5 — Padding mask and causal mask are different

A causal mask prevents looking into the future.

A padding mask prevents artificial padding positions from being treated as real content.

In batched inference, both concerns may exist simultaneously.

### Edge case 6 — Optimized attention does not change causality

Flash-style or fused attention may compute the same logical attention more efficiently.

It does not mean the model is suddenly allowed to attend to future tokens.

Algorithmic optimization and model dependency rules are separate concepts.

---

# 8.3 Mental Model

Prefill is:

> **One large forward pass over all already-known prompt tokens, with causal masking enforcing autoregressive information flow.**

The core mathematical picture is:

$$X \rightarrow Q,K,V \rightarrow \frac{QK^T}{\sqrt{d_k}}+M \rightarrow \text{softmax} \rightarrow \text{contextual prompt representations}$$

The key insight is:

```text
Known positions can be computed together
even though
future information is still masked
```

Memory line:

> **Autoregressive does not mean the known prompt must be processed serially.**

---

# Bridge from Prefill to Decode

Prefill works beautifully because all prompt tokens are already known.

But eventually the model reaches the end of the prompt.

Suppose the prompt is:

$$
x_1,x_2,\dots,x_n
$$

The model predicts a distribution for:

$$
x_{n+1}
$$

But before the model chooses $x_{n+1}$, that token does not exist.

And if $x_{n+1}$ does not yet exist, then neither does:

$$
x_{n+2}
$$

That creates the limitation of prefill:

> **You cannot process future generated token embeddings before you know which tokens were actually generated.**

That leads to the decode phase.

---

# 8.4 Decode Phase

## Question 1 — What exactly is the decode phase?

The **decode phase** begins after prefill has produced the first next-token distribution.

Suppose the prompt is:

$$
x_1,x_2,\dots,x_n
$$

Prefill allows us to compute:

$$
P(x_{n+1} \mid x_1,\ldots,x_n)
$$

A decoding strategy chooses one specific token:

$$
x_{n+1}
$$

Now—and only now—the model knows the next input token.

It can then compute:

$$
P(x_{n+2} \mid x_1, \dots,x_n,x_{n+1})
$$

Then it selects:

$$
x_{n+2}
$$

and repeats.

So decode is:

```text
predict one token
→ select it
→ append it
→ use it to predict the next token
→ repeat
```

---

## Question 2 — Why can’t decode process 100 future output tokens at once like prefill?

Because those future token identities are unknown.

Suppose we want to generate:

$$
x_{n+1},x_{n+2},x_{n+3}
$$

The probability factorization is:

$$
P(x_{n+1},x_{n+2},x_{n+3} \mid x_{\le n})
$$

which becomes:

$$
P(x_{n+1} \mid x_{\le n})
\cdot
P(x_{n+2} \mid x_{\le n+1})
\cdot
P(x_{n+3} \mid x_{\le n+2})
$$

The second factor requires knowing $x_{n+1}$.

The third factor requires knowing $x_{n+2}$.

So the dependency chain is:

```text
x(n+1)
must be known before
x(n+2)

x(n+2)
must be known before
x(n+3)
```

### Why do we need this mathematics?

Because it shows that sequential generation is not merely an implementation accident.

It follows from autoregressive conditional dependence.

We will revisit this deeply in 8.14.

---

## Question 3 — What does one decode step look like mathematically?

Suppose the current context has length:

$$
t
$$

To predict token $x_{t+1}$, the model processes the newest available token representation and must attend to the relevant previous context.

For one attention head, let the new query be:

$$q_t \in \mathbb{R}^{1 \times d_k}$$

The keys for positions up to $t$ can be represented as:

$$
K_{1:t} \in \mathbb{R}^{\times d_k}
$$

Then:

$$
K_{1:t}^T \in \mathbb{R}^{d_k	\times t}
$$

So the attention-score vector is:

$$
q_tK_{1:t}^T \in \mathbb{R}^{1	\times t}
$$

### Dimension check

$$
(1	\times d_k)(d_k	\times t) = 1	\times t
$$

This means the new position obtains one attention score for every key position available in its context.

After scaling and softmax:

$$\alpha_t = \text{softmax}\left(\frac{q_tK_{1:t}^T}{\sqrt{d_k}}\right)$$


where:

$$\alpha_t \in \mathbb{R}^{1 \times t}$$


If:

$$V_{1:t} \in \mathbb{R}^{t \times d_v}$$


then:

$$o_t = \alpha_t V_{1:t}$$


has shape:

$$
o_t \in \mathbb{R}^{1	\times d_v}
$$

### Why do we need this mathematics?

It reveals the key contrast with prefill.

During prefill:

$$
QK^T \in \mathbb{R}^{n	\times n}
$$

for the known prompt.

During an optimized one-token decode step, the new query interacts with all available keys:

$$
q_tK_{1:t}^T \in \mathbb{R}^{1	\times t}
$$

So:

```text
Prefill:
many queries × many keys

Decode:
one new query × growing history
```

---

## Question 4 — Wait, where did $K_{1:t}$ and $V_{1:t}$ come from?

Excellent question.

The previous keys and values correspond to tokens that have already been processed.

A naive implementation could recompute them from scratch every decode step.

But that would be wasteful because the old tokens have not changed.

This observation creates the next major inference optimization:

> **Store the old Keys and Values and reuse them.**

That is exactly what the **KV cache** does.

We are deliberately not solving that problem fully yet because it is the next syllabus topic.

For now, remember:

```text
Decode needs old K/V information.

Question:
Do we recompute it every time?

Answer:
We should not.

Next topic:
KV cache.
```

---

## Question 5 — What would happen without a KV cache?

Suppose the original prompt has:

$$
P
$$

tokens and we generate:

$$
G
$$

new tokens.

Without caching, to produce each new token the model may need to recompute representations for the growing prefix.

The sequence lengths processed would roughly be:

$$
P,;P+1,;P+2, \ldots,P+G-1
$$

The total number of token positions repeatedly processed across decode steps would be:

$$\sum_{g=0}^{G-1} (P+g)$$

Using the arithmetic-series formula:

$$\sum_{g=0}^{G-1}(P+g) = GP + \frac{G(G-1)}{2}$$


### Numerical example

Suppose:

$$
P=1000
$$

and:

$$
G=100
$$

Then:

$$GP = 100 \times 1000 = 100{,}000$$


and:

$$\frac{G(G-1)}{2} = \frac{100 \times 99}{2} = 4950$$

So:

$$100{,}000 + 4950 = 104{,}950$$

token-position evaluations would be involved in this simplified repeated-prefix view.

Yet only 100 genuinely new output positions were created.

### Why do we need this mathematics?

Because it quantifies the redundancy.

Old context does not change, so repeatedly recomputing its attention state is wasteful.

This is the exact problem that motivates KV caching.

---

## Question 6 — Does decode become constant-time once we cache old states?

No.

KV caching removes a major source of recomputation, but the new query still needs to attend over a context whose length keeps growing.

At generation step $t$, the score vector has length:

$$
t
$$

because:

$$
q_tK_{1:t}^T \in \mathbb{R}^{1	\times t}
$$

So as the context gets longer, the new token interacts with more cached positions.

### Why is this important?

A common misconception is:

> “KV cache makes decode independent of context length.”

It does not.

KV cache means:

```text
do not recompute old K/V
```

It does **not** mean:

```text
ignore old K/V
```

The new query still needs access to the relevant history.

---

## Question 7 — Why is decode often discussed as latency-sensitive?

Each generated token depends on the token before it.

So if the system generates one token every:

$$
	au
$$

seconds, then generating $G$ tokens takes at least roughly:

$$
G	au
$$

seconds of sequential decode work, ignoring overlap and serving overhead.

For example, if the model effectively produces:

$$50 \text{ tokens/second}$$

then average model-side time per token is approximately:

$$\frac{1}{50} = 0.02 \text{ seconds} = 20 \text{ ms}$$


Generating 200 tokens would then require roughly:

$$200 \times 20 \text{ ms} = 4000 \text{ ms} = 4 \text{ s}$$


in this simplified example.

### Why do we need this mathematics?

Because it shows why output length has a direct effect on perceived response time.

Long prompt:

```text
primarily increases work before the first generated token
```

Long output:

```text
adds many sequential decode iterations
```

They create different latency patterns.

---

## Question 8 — What is Time Per Output Token?

A useful serving metric is often called **Time Per Output Token (TPOT)** or an equivalent decode-latency measure.

Conceptually:

$$\text{TPOT} \approx \frac{\text{decode time}}{\text{number of generated tokens}}$$


But real systems need careful measurement because:

- first-token behavior differs from later decode
- batching can change token cadence
- queueing may or may not be included
- network streaming can affect observed latency

### Why do we need this metric?

TTFT and TPOT answer different user-experience questions.

TTFT asks:

> How long until I see the response start?

TPOT asks:

> Once generation starts, how quickly do new tokens arrive?

This gives us the useful distinction:

```text
Prefill-heavy problem
→ TTFT becomes large

Decode-heavy problem
→ token streaming becomes slow
```

---

## Question 9 — Why can decode be inefficient on modern accelerators?

A large language model contains a huge number of parameters.

For each decode step, the system must perform computations through all Transformer layers for only a very small number of new token positions.

That means the hardware may have relatively little token-level parallel work within one request compared with prefill.

In addition, the model weights and KV state must be read from memory.

This is one reason decode is often discussed as having strong **memory-bandwidth** constraints in large-model serving.

We will study the full inference-cost story in 8.15.

### Important nuance

Do not reduce this to:

> “Decode is always memory-bound.”

Whether a workload is compute-bound or bandwidth-bound depends on:

- batch size
- model architecture
- hardware
- quantization
- kernel implementation
- sequence length
- parallelism strategy

The statement is a common serving tendency, not a universal law.

---

## Question 10 — Can decode be parallelized at all?

Yes—but we need to distinguish **different kinds of parallelism**.

### Across independent requests

If 100 users are each generating one token, those 100 decode operations can often be batched together.

### Across model layers/devices

Tensor parallelism, pipeline parallelism, or other distributed execution strategies can split model computation across hardware.

### Across future token positions in one ordinary autoregressive sequence

This is the difficult part.

You cannot know the exact token $x_{t+2}$ until the system has selected $x_{t+1}$.

So ordinary autoregressive dependency prevents straightforward parallel generation of exact future token identities.

### Edge case — Speculative decoding

Later techniques can **propose** multiple future tokens and verify them.

That improves throughput or latency without simply assuming future target-model tokens are already known.

But ordinary decode itself remains autoregressive.

---

## Question 11 — What are important decode edge cases?

### Edge case 1 — The model generates EOS immediately

The decode phase can end after a single generated token if that token is a valid stop token.

So maximum output length is a limit, not a promise.

### Edge case 2 — Very long generated output

Every generated token extends the active context.

So later decode steps may work with a longer history than earlier ones.

### Edge case 3 — Context-window exhaustion

If prompt tokens plus generated tokens approach the model/runtime context limit:

$$
P+G
ightarrow C
$$

generation may need to stop, truncate context, or use a serving-specific strategy.

### Edge case 4 — Batch members finish at different times

Suppose ten requests are being decoded together.

Some may generate EOS early while others continue.

Efficient serving systems dynamically manage these changing active sequences rather than wasting all work on finished requests.

### Edge case 5 — Different output lengths cause scheduling complexity

One request may need 20 output tokens while another needs 2000.

That makes continuous batching and resource scheduling much harder than fixed-shape offline tensor workloads.

### Edge case 6 — One token can represent very different amounts of visible text

A single decoded token might represent:

- one character
- part of a word
- an entire common word
- punctuation
- whitespace plus text

So “tokens per second” is not identical to “words per second.”

---

# 8.4 Mental Model

Decode is:

> **Repeatedly process the one newly available token position, use the accumulated context to predict the next token, select it, and repeat.**

The core conditional dependency is:

$$
P(x_{t+1} \mid x_{\le t})
$$

and one-head attention for the newest position can be viewed as:

$$
q_t \in \mathbb{R}^{1	\times d_k}
$$

attending over:

$$
K_{1:t} \in \mathbb{R}^{\times d_k}
$$

to produce:

$$
q_tK_{1:t}^T \in \mathbb{R}^{1	\times t}
$$

Memory line:

> **Prefill parallelizes over known prompt positions; decode cannot parallelize unknown future token identities in the same way.**

---

# Prefill vs Decode — Final Comparison

| Property | Prefill | Decode |
|---|---|---|
| Tokens available | Entire prompt is already known | Only the next generated token becomes known after selection |
| Main execution pattern | Many prompt positions processed together | One new position per autoregressive step |
| Attention view | Many queries × many prompt keys | One new query × growing history |
| Typical shape per head | $n\times n$ logical score matrix | $1\times t$ score vector for newest query |
| User-facing metric strongly affected | TTFT | TPOT / streaming speed |
| Main length driver | Prompt length | Generated output length + growing context |
| Natural next optimization question | — | How do we avoid recomputing old K/V? |

---

# Why Does This Naturally Lead to KV Cache?

We now have a clear problem.

During decode, the context contains tokens that were already processed earlier.

For example:

```text
Prompt:
x1 x2 x3 ... xn

Generated:
x(n+1)
```

When we generate the next token, the old prompt tokens have not changed.

Neither has the already-generated token history.

Yet attention still needs their Key and Value representations.

So the natural question becomes:

> **Why recompute Keys and Values for old tokens if those Keys and Values are unchanged?**

The answer is:

> **Cache them.**

That is the purpose of:

```text
8.5 KV Cache
```

---

# Part 2 Summary

The story of Part 2 is:

```text
Prompt tokens are already known
        ↓
Process them together
        ↓
PREFILL
        ↓
Use causal masking to preserve autoregressive rules
        ↓
Produce first next-token distribution
        ↓
Select one token
        ↓
Future output token becomes known
        ↓
DECODE
        ↓
Generate one token at a time
        ↓
Each new query needs old K/V information
        ↓
Recomputing old K/V is wasteful
        ↓
Next topic: KV Cache
```

Core equations:

### Prompt tensor

$$
X \in \mathbb{R}^{n \times d_{\text{model}}}
$$

### Query, Key, Value projections

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

### Prefill attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

### Prefill score shape

$$
QK^T \in \mathbb{R}^{n \times n}
$$

### One-token decode score shape

$$
q_t K_{1:t}^T \in \mathbb{R}^{1 \times t}
$$

### Autoregressive dependency

$$
P(x_{n+1}, x_{n+2}, \ldots) = \prod_{t} P(x_t \mid x_{<t})
$$

### Repeated-prefix work without caching

$$
\sum_{g=0}^{G-1} (P+g) = GP + \frac{G(G-1)}{2}
$$

The next detailed-study block is:

```text
8.5 KV Cache
```

The question carrying us forward is:


> **If old tokens do not change during decode, why should their Keys and Values ever be recomputed?**
