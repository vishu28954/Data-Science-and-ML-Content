# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 3 — Interview Style

## Covers

\`\`\`text
8.5 KV Cache
\`\`\`

This file contains only interview-style answers for KV Cache.

---

# 8.5 KV Cache — Interview Style

## 1. What is KV Cache?

### Interview answer

KV Cache is an inference optimization used during autoregressive generation.

During self-attention, every token produces a Query, Key, and Value. Once an old token has been processed, its Key and Value do not change for later decode steps. Instead of recomputing those old Keys and Values every time a new token is generated, the model stores them and reuses them.

Conceptually:

$$
K_{\text{cache}} = [k_1, k_2, \ldots, k_t]
$$

and:

$$
V_{\text{cache}} = [v_1, v_2, \ldots, v_t]
$$

For the newest token, the model computes a new Query and compares it with the cached Keys.

The main idea is:

> **KV Cache saves repeated computation by storing reusable attention state.**

---

## 2. Why is KV Cache needed during decoding?

### Interview answer

Autoregressive decoding generates one token at a time.

Suppose the model has already processed:

$$
x_1, x_2, \ldots, x_t
$$

and has selected a new token:

$$
x_{t+1}
$$

To process $x_{t+1}$ and predict the following token, the model needs attention over the previous context.

Without KV Cache, it could recompute the old Key and Value representations for:

$$
x_1, x_2, \ldots, x_t
$$

again.

That would be redundant because those old tokens have not changed.

With KV Cache, the old Keys and Values are reused, and only the new token's attention state is computed.

So:

\`\`\`text
without KV cache
→ repeatedly recompute old attention state

with KV cache
→ reuse old K/V
→ compute only the new token's state
\`\`\`

---

## 3. What exactly is stored in the KV Cache?

### Interview answer

The persistent cache stores the Key and Value vectors for previously processed token positions.

For one attention head, after $t$ positions:

$$
K_{\text{cache}}
\in
\mathbb{R}^{t \times d_k}
$$

and:

$$
V_{\text{cache}}
\in
\mathbb{R}^{t \times d_v}
$$

Conceptually:

$$
K_{\text{cache}}
=
\begin{bmatrix}
k_1 \\
k_2 \\
\vdots \\
k_t
\end{bmatrix}
$$

and:

$$
V_{\text{cache}}
=
\begin{bmatrix}
v_1 \\
v_2 \\
\vdots \\
v_t
\end{bmatrix}
$$

Each new processed token adds one new Key row and one new Value row per KV head, per Transformer layer.

---

## 4. Is the Query also stored in the KV Cache?

### Interview answer

Usually, no.

The Query is needed for the current token's attention computation, but future tokens do not reuse that old Query.

For a new token, the model computes a new Query:

$$
q_{t+1}
$$

and compares it with the cached Keys.

The old Keys remain useful because future Queries need to compare against them.

The old Values remain useful because future attention outputs may retrieve information from them.

So the mental model is:

\`\`\`text
Query
→ what am I looking for right now?
→ usually temporary

Key
→ what information do I represent?
→ reusable by future queries

Value
→ what information can I provide?
→ reusable by future queries
\`\`\`

That is why the cache is called **KV Cache**, not QKV Cache.

---

## 5. Are old Query–Key attention scores stored in the KV Cache?

### Interview answer

No. The persistent KV Cache stores Keys and Values, not the old pairwise Query–Key scores.

Suppose the current Query is:

$$
q_t
$$

The attention scores are:

$$
q_t K_{\text{cache}}^T
$$

These scores answer:

> Which previous positions are relevant to the current Query?

At the next step, the model creates a different Query:

$$
q_{t+1}
$$

and must compute new scores:

$$
q_{t+1} K_{\text{cache}}^T
$$

The previous scores cannot simply be reused because the Query has changed.

So:

\`\`\`text
persistent across decode steps:
K and V

temporary for one decode step:
Q
QKᵀ scores
softmax attention weights
\`\`\`

---

## 6. How is a new token added to the KV Cache?

### Interview answer

Suppose the existing cache contains $t$ token positions:

$$
K_{\text{cache}}^{(t)}
=
[k_1, k_2, \ldots, k_t]
$$

and:

$$
V_{\text{cache}}^{(t)}
=
[v_1, v_2, \ldots, v_t]
$$

After a new token $x_{t+1}$ has been selected and processed, the layer computes:

$$
q_{t+1}, \qquad k_{t+1}, \qquad v_{t+1}
$$

The new Key is appended:

$$
K_{\text{cache}}^{(t+1)}
=
[k_1, k_2, \ldots, k_t, k_{t+1}]
$$

and the new Value is appended:

$$
V_{\text{cache}}^{(t+1)}
=
[v_1, v_2, \ldots, v_t, v_{t+1}]
$$

The Query is used for the current attention computation but is not normally appended to a persistent Query cache.

---

## 7. What does one KV-cached decode attention step look like mathematically?

### Interview answer

Suppose the newly processed position is $t+1$.

For one attention head:

$$
q_{t+1}
\in
\mathbb{R}^{1 \times d_k}
$$

After appending the new Key, the Key cache has shape:

$$
K_{\text{cache}}^{(t+1)}
\in
\mathbb{R}^{(t+1) \times d_k}
$$

Therefore:

$$
\left(K_{\text{cache}}^{(t+1)}\right)^T
\in
\mathbb{R}^{d_k \times (t+1)}
$$

The score vector is:

$$
q_{t+1}
\left(K_{\text{cache}}^{(t+1)}\right)^T
\in
\mathbb{R}^{1 \times (t+1)}
$$

The dimension check is:

$$
(1 \times d_k)
(d_k \times (t+1))
=
1 \times (t+1)
$$

After scaling and softmax:

$$
\alpha_{t+1}
=
\text{softmax}
\left(
\frac{
q_{t+1}
\left(K_{\text{cache}}^{(t+1)}\right)^T
}{
\sqrt{d_k}
}
\right)
$$

If:

$$
V_{\text{cache}}^{(t+1)}
\in
\mathbb{R}^{(t+1) \times d_v}
$$

then:

$$
o_{t+1}
=
\alpha_{t+1}
V_{\text{cache}}^{(t+1)}
$$

with:

$$
o_{t+1}
\in
\mathbb{R}^{1 \times d_v}
$$

---

## 8. How does KV Cache relate to the first generated token after prefill?

### Interview answer

The first generated token is slightly special.

Suppose the prompt is:

$$
x_1, x_2, \ldots, x_n
$$

Prefill already processes the prompt and produces the next-token distribution:

$$
P(x_{n+1} \mid x_1, x_2, \ldots, x_n)
$$

A decoding strategy selects $x_{n+1}$ from that distribution.

Only after $x_{n+1}$ has been selected does the model process that new token in the next one-token forward pass.

During that pass, it computes the new token's:

$$
q_{n+1}, \qquad k_{n+1}, \qquad v_{n+1}
$$

and appends:

$$
k_{n+1}
$$

and:

$$
v_{n+1}
$$

to the cache.

That forward pass ultimately produces the distribution for:

$$
x_{n+2}
$$

So the sequence is:

\`\`\`text
prefill
→ distribution for x(n+1)
→ select x(n+1)

decode forward pass
→ process x(n+1)
→ append k(n+1), v(n+1)
→ distribution for x(n+2)
→ select x(n+2)

repeat
\`\`\`

---

## 9. What does KV Cache save computationally?

### Interview answer

Without caching, the model may repeatedly recompute attention state for the growing prefix.

If the prompt length is $P$ and the model generates $G$ new tokens, the repeatedly processed prefix lengths are approximately:

$$
P, P+1, P+2, \ldots, P+G-1
$$

The number of token positions repeatedly involved is:

$$
\sum_{g=0}^{G-1}(P+g)
$$

which equals:

$$
GP+\frac{G(G-1)}{2}
$$

KV Cache avoids recreating old Keys and Values for all those previously processed positions.

Instead, each decode step computes the new token's state and reuses the stored history.

---

## 10. How does KV Cache change the attention complexity during decoding?

### Interview answer

Without caching, recomputing dense attention over a prefix of length $s$ has an attention-score term approximately proportional to:

$$
O(s^2 d_k)
$$

With KV Cache, the newest token contributes one Query that attends over the existing history.

At a context length of approximately $t$, the score computation is closer to:

$$
O(t d_k)
$$

for that one Query per head.

So conceptually:

\`\`\`text
without KV cache:
many old queries are recomputed against many old keys

with KV cache:
one new query is computed against cached history
\`\`\`

KV Cache therefore removes a large amount of repeated work during autoregressive decoding.

---

## 11. Does KV Cache make decode constant-time?

### Interview answer

No.

KV Cache avoids recomputing old Keys and Values, but the new Query still attends over the growing history.

If the cache contains $t$ Keys, then:

$$
q_t K_{\text{cache}}^T
$$

produces approximately:

$$
t
$$

attention scores for that head.

So longer context still means:

- more cached data to read,
- more Key positions to compare against,
- more Value information to combine.

KV Cache reduces redundant computation, but it does not make attention independent of context length.

---

## 12. What is the memory cost of KV Cache?

### Interview answer

A useful approximate formula is:

$$
M_{\text{KV}}
\approx
2
\times
L
\times
B
\times
T
\times
H_{\text{KV}}
\times
d_h
\times
b
$$

where:

- $L$ = number of Transformer layers,
- $B$ = number of active sequences or batch size,
- $T$ = cached sequence length,
- $H_{\text{KV}}$ = number of KV heads,
- $d_h$ = dimension of each KV head,
- $b$ = bytes per stored element.

The factor:

$$
2
$$

appears because both Keys and Values are stored.

This formula shows that KV-cache memory grows linearly with sequence length, concurrency, number of layers, KV heads, head dimension, and bytes per element.

---

## 13. Give a numerical KV-cache memory example.

### Interview answer

Suppose:

$$
L=32
$$

$$
B=1
$$

$$
T=8192
$$

$$
H_{\text{KV}}=8
$$

$$
d_h=128
$$

and FP16 or BF16 storage is used:

$$
b=2 \text{ bytes}
$$

Then:

$$
M_{\text{KV}}
=
2
\times
32
\times
1
\times
8192
\times
8
\times
128
\times
2
$$

which equals:

$$
1{,}073{,}741{,}824
\text{ bytes}
$$

or approximately:

$$
1 \text{ GiB}
$$

for one sequence.

If the active batch size becomes 16 and everything else stays the same, the raw KV-cache requirement becomes approximately:

$$
16 \text{ GiB}
$$

This is why KV Cache can become a major serving-memory bottleneck.

---

## 14. How does sequence length affect KV-cache memory?

### Interview answer

KV-cache memory is approximately linear in cached sequence length:

$$
M_{\text{KV}} \propto T
$$

So if:

$$
T \rightarrow 2T
$$

then approximately:

$$
M_{\text{KV}} \rightarrow 2M_{\text{KV}}
$$

For example:

\`\`\`text
8192 cached tokens
→ about 1 GiB in the previous example

16384 cached tokens
→ about 2 GiB
\`\`\`

assuming all other values remain unchanged.

---

## 15. How does batch size or concurrency affect KV-cache memory?

### Interview answer

KV-cache memory is also linear in the number of active sequences:

$$
M_{\text{KV}} \propto B
$$

So doubling active batch size approximately doubles KV-cache memory.

This matters in production because a model can fit in GPU memory while high-concurrency KV state does not.

In serving systems, memory therefore has to accommodate:

\`\`\`text
model weights
+
runtime buffers
+
KV cache for active requests
\`\`\`

---

## 16. How does KV-cache precision affect memory?

### Interview answer

The storage formula contains:

$$
b
$$

which is the number of bytes per cached element.

Conceptually:

\`\`\`text
FP16 / BF16
≈ 2 bytes per element

FP8
≈ 1 byte per element
\`\`\`

So reducing KV-cache precision can significantly reduce memory use and memory traffic.

However, lower precision can introduce quantization error, so it creates a tradeoff between:

\`\`\`text
memory and bandwidth efficiency
↔
numerical fidelity
\`\`\`

The exact quality impact depends on the model and implementation.

---

## 17. Why do GQA and MQA help KV-cache efficiency?

### Interview answer

KV-cache memory depends on the number of KV heads:

$$
M_{\text{KV}} \propto H_{\text{KV}}
$$

In standard Multi-Head Attention, the number of Query heads and KV heads may be the same.

For example:

$$
H_Q = H_{\text{KV}} = 32
$$

In Grouped-Query Attention, multiple Query heads share fewer KV heads.

For example:

$$
H_Q=32
$$

but:

$$
H_{\text{KV}}=8
$$

Then, all else equal, the KV-head portion of the cache is reduced by:

$$
\frac{8}{32}
=
\frac{1}{4}
$$

In Multi-Query Attention:

$$
H_{\text{KV}}=1
$$

so all Query heads share one Key head and one Value head.

The key inference benefit is that fewer KV heads mean less KV-cache memory and lower KV-cache bandwidth requirements.

---

## 18. Why can long-context decoding still become slower even with KV Cache?

### Interview answer

Because the new Query must still interact with the cached history.

If the context contains $T$ positions, the newest Query must conceptually compare against approximately $T$ Keys:

$$
q_t K_{\text{cache}}^T
$$

So longer context means:

\`\`\`text
larger cache
→ more K/V data to read
→ more historical positions to attend over
\`\`\`

KV Cache removes repeated K/V recomputation, but it does not remove the cost of using the history.

---

## 19. Why is KV Cache important for memory bandwidth?

### Interview answer

During decode, the model processes only a small number of new positions per request, but it still needs to read:

- model weights,
- cached Keys,
- cached Values.

As context and concurrency grow, moving KV data through accelerator memory can become expensive.

This is why LLM serving systems care about:

- high-bandwidth memory capacity,
- memory bandwidth,
- cache layout,
- paging,
- batching,
- cache precision.

However, I would not say decode is always memory-bound. The bottleneck depends on model architecture, hardware, batch size, sequence length, quantization, and implementation.

---

## 20. Is there one KV Cache for the entire model?

### Interview answer

No.

Every Transformer layer has its own Key and Value state.

If a model has $L$ Transformer layers, then each layer stores its own cached Keys and Values.

Conceptually:

\`\`\`text
Layer 1
→ K cache
→ V cache

Layer 2
→ K cache
→ V cache

...

Layer L
→ K cache
→ V cache
\`\`\`

This is why the memory formula contains the factor:

$$
L
$$

---

## 21. How does positional information interact with KV Cache?

### Interview answer

A cached Key is associated with a specific token position.

If the model uses a position-dependent attention mechanism such as RoPE, the cached state reflects the positional treatment for that token.

So when a new token is appended at position:

$$
t+1
$$

its Query and Key must use the correct position.

KV Cache is therefore not an unordered set of vectors.

It is an ordered, position-sensitive history.

Incorrect cache-position alignment can produce incorrect attention behavior.

---

## 22. Can cached K/V always be reused if the prompt changes?

### Interview answer

No.

Cache reuse generally requires the corresponding prefix to remain unchanged and compatible.

Suppose the original sequence is:

\`\`\`text
A B C D
\`\`\`

and we change it to:

\`\`\`text
A B X D
\`\`\`

The representations after the changed position can differ, so downstream cached state may no longer be valid.

A useful rule is:

\`\`\`text
same compatible prefix
→ prefix cache may be reusable

changed prefix
→ affected downstream cache may need recomputation
\`\`\`

---

## 23. What is prefix caching?

### Interview answer

Prefix caching extends the same idea beyond one decode sequence.

If multiple requests share an identical long prefix, such as the same system prompt, a serving system may reuse the previously computed KV state for that prefix.

Conceptually:

\`\`\`text
same tokenized prefix
→ reuse compatible prefix KV state
→ reduce repeated prefill work
\`\`\`

But the reuse must be compatible with the exact model, tokenization, positional handling, and relevant serving configuration.

It is not simply semantic similarity caching.

---

## 24. How does beam search affect KV Cache?

### Interview answer

Beam search keeps multiple candidate continuations alive.

Once candidates diverge, each continuation may require its own continuation-specific KV state.

That can increase KV-cache memory.

Implementations may share common-prefix state and only duplicate or reorder the state needed after the branches diverge.

The main principle is:

> **More simultaneously maintained sequence hypotheses generally require more KV state.**

---

## 25. What happens with sliding-window or local attention?

### Interview answer

Not every architecture attends to the entire history.

If a layer uses an attention window of size:

$$
W
$$

then the active Keys and Values may only need to cover approximately the most recent:

$$
W
$$

positions.

Conceptually:

$$
K_{\text{active}}
=
[k_{t-W+1}, \ldots, k_t]
$$

instead of necessarily using:

$$
[k_1, \ldots, k_t]
$$

This can reduce active cache requirements, although the exact behavior depends on the model architecture.

---

# Complete KV Cache Interview Answer

If asked:

\`\`\`text
What is KV Cache and why is it important for LLM inference?
\`\`\`

You can say:

\`\`\`text
KV Cache is an optimization for autoregressive inference that stores the Key and Value representations of previously processed tokens.

During decoding, each new token creates a new Query, Key, and Value. The new Query needs to attend to the historical Keys and use the historical Values, but the old Keys and Values do not change. So instead of recomputing them at every generation step, we cache and reuse them.

For one attention head, if the current cache contains t tokens, K_cache has shape t by d_k and V_cache has shape t by d_v. A new Query has shape 1 by d_k, so multiplying it by K_cache-transpose produces a 1 by t attention-score vector.

We generally do not cache old Queries or old Query-Key scores because each future token creates a new Query and therefore needs new attention scores. What remains reusable is the historical K and V state.

KV Cache greatly reduces repeated decode computation, but it spends memory to do so. Its memory grows roughly linearly with the number of layers, active sequences, cached tokens, KV heads, head dimension, and bytes per element. So it is one of the main compute-memory tradeoffs in production LLM serving.
\`\`\`

---

# Crisp Interview Version

\`\`\`text
KV Cache stores the Keys and Values of previously processed tokens so they do not have to be recomputed during every autoregressive decode step.

For each new token, the model computes a fresh Query, Key, and Value. The new Key and Value are appended to the cache, while the Query is used immediately against all cached Keys.

We cache K and V, not old Q, because future tokens create new Queries but still need the historical Keys and Values.

KV Cache saves substantial repeated computation, but its memory grows with context length and concurrency. So the core tradeoff is: save compute by spending memory.
\`\`\`

---

# Important Interview Follow-Ups

## If asked: Does KV Cache store attention scores?

\`\`\`text
No. KV Cache stores Keys and Values. Attention scores depend on the current Query, so a new Query produces a new set of Query-Key scores at every decode step.
\`\`\`

## If asked: Does KV Cache make decode constant-time?

\`\`\`text
No. It removes repeated K/V recomputation, but the newest Query still attends over a growing historical cache, so long context still has a cost.
\`\`\`

## If asked: Why not cache Queries?

\`\`\`text
An old Query was only needed when that old position performed its own attention calculation. Future positions create new Queries, but they still reuse old Keys and Values.
\`\`\`

## If asked: What is the main production bottleneck of KV Cache?

\`\`\`text
Memory capacity and memory bandwidth can become major constraints, especially with long contexts and many concurrent requests.
\`\`\`

---

# Most Important Memory Lines

> **KV Cache saves compute by spending memory.**

> **Future Queries reuse old Keys and Values, not old Queries.**

> **The cache stores K and V vectors, not old Query-Key attention scores.**

> **With KV Cache, decode becomes one new Query against a growing cached history.**

> **KV Cache removes recomputation of old K/V, but it does not remove the cost of attending to long context.**

---

The next detailed-study block is:

\`\`\`text
8.6 Greedy Decoding
8.7 Temperature
\`\`\`
