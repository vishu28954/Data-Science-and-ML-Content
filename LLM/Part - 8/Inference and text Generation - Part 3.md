# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 3

## 8.5 KV Cache

This note continues directly from Part 2.

At the end of Part 2, we reached an important observation:

```text
Prefill
→ processes the known prompt

Decode
→ generates one new token at a time

But every new token still needs information from old tokens.

Question:
Should we recompute the old attention state every single time?
```

The answer is no.

That is exactly why **KV cache** exists.

The story of this part is:

```text
Prompt is processed during prefill
        ↓
Keys and Values are created for every prompt token
        ↓
Decode begins
        ↓
A new token needs to attend to old tokens
        ↓
Old Keys and Values have not changed
        ↓
Do not recompute them
        ↓
Store them and reuse them
        ↓
KV CACHE
        ↓
Faster decode, but higher memory usage
        ↓
Longer context and larger batches consume more KV memory
        ↓
After logits are produced efficiently,
how do we choose the next token?
        ↓
Next topics:
8.6 Greedy Decoding
8.7 Temperature
```

---

# 8.5 KV Cache

## Question 1 — What problem is KV cache trying to solve?

Suppose the prompt contains:

$$x_1, x_2, \ldots, x_n$$

During prefill, every Transformer layer computes Queries, Keys, and Values for the prompt tokens.

For one layer, conceptually:

$$Q = XW_Q$$

$$K = XW_K$$

$$V = XW_V$$

After the prompt is processed, the model predicts the first output token:

$$P(x_{n+1} \mid x_1, x_2, \ldots, x_n)$$

Suppose the decoder selects $x_{n+1}$.

To predict the next token, the model now needs:

$$P(x_{n+2} \mid x_1, x_2, \ldots, x_n, x_{n+1})$$

The new token $x_{n+1}$ must attend to the previous context.

But here is the important observation:

> The old prompt tokens $x_1, \ldots, x_n$ have not changed.

Therefore, the Key and Value vectors previously computed for those old tokens have not changed either.

So recomputing them would be redundant.

That is the exact problem KV cache solves.

---

## Question 2 — What exactly is stored in the KV cache?

For every Transformer layer, self-attention produces Key and Value vectors for every token position.

For one attention head, suppose token position $i$ has:

$$k_i in mathbb{R}^{d_h}$$

and:

$$v_i in mathbb{R}^{d_h}$$

where $d_h$ is the head dimension.

After processing $t$ tokens, we can stack the Keys as:

$$K_{	ext{cache}} =
egin{bmatrix}
k_1 \
k_2 \
dots \
k_t
end{bmatrix}
in mathbb{R}^{t 	imes d_h}$$

Similarly, the Values are:

$$V_{	ext{cache}} =
egin{bmatrix}
v_1 \
v_2 \
dots \
v_t
end{bmatrix}
in mathbb{R}^{t 	imes d_h}$$

So conceptually:

```text
KV cache
=
all previously computed Keys
+
all previously computed Values
```

### Why do we need this mathematics?

Because the shapes tell us what grows during generation.

If the sequence length grows from:

$$t = 1000$$

to:

$$t = 1001$$

the cache grows by exactly one new Key vector and one new Value vector per KV head, per layer, per active sequence.

So KV-cache memory grows approximately **linearly with sequence length**.

---

## Question 3 — Where does the KV cache come from initially?

It is created during **prefill**.

Suppose the prompt is:

$$x_1, x_2, ldots, x_n$$

During prefill, every Transformer layer already computes Keys and Values for those prompt positions.

So after prefill, each layer already has:

$$K_{	ext{cache}}^{(ell)} = [k_1^{(ell)}, k_2^{(ell)}, ldots, k_n^{(ell)}]$$

and:

$$V_{	ext{cache}}^{(ell)} = [v_1^{(ell)}, v_2^{(ell)}, ldots, v_n^{(ell)}]$$

where $ell$ denotes the Transformer layer.

This is important:

> There is not one single KV cache for the entire model.

There is a KV cache associated with the self-attention state of **each Transformer layer**.

So if the model has $L$ layers, KV state is stored for all $L$ layers.

---

## Question 4 — What happens during one decode step with KV caching?

Suppose the current sequence contains $t$ tokens.

The old cache contains:

$$K_{	ext{cache}}^{1:t}$$

and:

$$V_{	ext{cache}}^{1:t}$$

Now the newly available token produces a new Query, Key, and Value:

$$q_{t+1}$$

$$k_{t+1}$$

$$v_{t+1}$$

The new Key is appended to the Key cache:

$$K_{	ext{cache}}^{1:t+1}
=
left[
K_{	ext{cache}}^{1:t};
k_{t+1}

ight]$$

Similarly:

$$V_{	ext{cache}}^{1:t+1}
=
left[
V_{	ext{cache}}^{1:t};
v_{t+1}

ight]$$

The new query then attends over the accumulated Keys:

$$q_{t+1}
left(K_{	ext{cache}}^{1:t+1}
ight)^T$$

For one attention head:

$$q_{t+1} in mathbb{R}^{1 	imes d_h}$$

and:

$$K_{	ext{cache}}^{1:t+1}
in
mathbb{R}^{(t+1) 	imes d_h}$$

Therefore:

$$
q_{t+1}
left(K_{	ext{cache}}^{1:t+1}
ight)^T
in
mathbb{R}^{1 	imes (t+1)}
$$

After scaling and softmax:

$$
alpha_{t+1}
=
operatorname{softmax}
left(
rac{
q_{t+1}
left(K_{	ext{cache}}^{1:t+1}
ight)^T
}{
sqrt{d_h}
}

ight)
$$

Then the attention output is:

$$
o_{t+1}
=
alpha_{t+1}
V_{	ext{cache}}^{1:t+1}
$$

### What changed compared with recomputing everything?

Only the new token's Q, K, and V need to be newly computed.

The old Keys and Values are reused.

That is the central optimization.

---

## Question 5 — Why do we cache K and V, but usually not old Q?

This is one of the most important conceptual points.

At token position $t+1$, we need a **new query**:

$$q_{t+1}$$

because the new token is asking:

> Which previous pieces of information are relevant to me?

The new query is used to compare against previous Keys.

The old Keys are still needed because future queries must compare against them.

The old Values are still needed because future attention outputs may retrieve information from them.

But what about an old query, such as:

$$q_5$$

Once token 5 has already completed its attention computation, future token 20 does **not** need to reuse $q_5$.

Token 20 creates its own query:

$$q_{20}$$

and compares it against old Keys.

So:

```text
Old Query:
used when that old token was processed
→ usually not needed again

Old Key:
future queries must compare against it
→ cache it

Old Value:
future attention outputs may retrieve it
→ cache it
```

### Memory line

> **Queries ask the question now; Keys and Values represent the reusable history.**

---

## Question 6 — What would decode look like without KV cache?

Suppose the prompt length is:

$$P$$

and we generate:

$$G$$

tokens.

Without caching, the model could repeatedly recompute the entire growing prefix.

The sequence lengths would be:

$$P, P+1, P+2, ldots, P+G-1$$

Even if we count only how many token positions are repeatedly processed, the total is:

$$
sum_{g=0}^{G-1}(P+g)
$$

Using the arithmetic-series formula:

$$
sum_{g=0}^{G-1}(P+g)
=
GP+rac{G(G-1)}{2}
$$

### Numerical example

Suppose:

$$P=1000$$

and:

$$G=100$$

Then:

$$GP=100 	imes 1000=100{,}000$$

and:

$$
rac{G(G-1)}{2}
=
rac{100 	imes 99}{2}
=
4950
$$

Therefore:

$$
100{,}000+4950
=
104{,}950
$$

token positions are repeatedly involved in this simplified full-prefix view.

But only:

$$100$$

new token positions were actually generated.

That is a large amount of redundant work.

---

## Question 7 — Can we see the attention-cost difference more directly?

Yes.

Without a cache, if we recompute dense attention for a prefix of length $s$, the attention-score part is approximately:

$$O(s^2 d_h)$$

At successive generation steps, the lengths are approximately:

$$P, P+1, P+2, ldots, P+G-1$$

So the cumulative attention-score work behaves like:

$$
sum_{g=0}^{G-1}
Oleft((P+g)^2 d_h
ight)
$$

With a KV cache, each new token contributes only one new query against the existing history.

At step $g$, the attention-score work is approximately:

$$O((P+g)d_h)$$

So cumulative attention-score work behaves like:

$$
sum_{g=0}^{G-1}
Oleft((P+g)d_h
ight)
$$

### Why do we need this mathematics?

Because it shows that KV caching changes the nature of decode attention.

Conceptually:

```text
Without KV cache:
recompute a growing prefix repeatedly

With KV cache:
process one new token
+
reuse old K/V
+
attend over growing history
```

The cache does not eliminate attention over history.

It eliminates the need to **recreate the old history's K/V representations**.

---

## Question 8 — Does KV cache make each decode step constant-time?

No.

This is a very common misconception.

Suppose the current context length is:

$$t$$

The new query still attends to:

$$t$$

or approximately $t+1$ Keys.

The attention-score vector has shape:

$$1 	imes t$$

or, after appending the new position:

$$1 	imes (t+1)$$

So as context length grows, the new query must interact with more cached Keys.

That means:

> KV cache removes redundant recomputation, but it does not make attention independent of context length.

The new token still needs to read and use the accumulated history.

---

## Question 9 — What exactly does KV cache save, and what does it not save?

KV cache **saves**:

```text
recomputing old Key projections
recomputing old Value projections
reprocessing old token attention state unnecessarily
much of the repeated full-prefix work during decode
```

KV cache does **not** remove:

```text
the new token's Transformer computation
the new token's Query computation
the new token's Key and Value computation
attention of the new Query over cached Keys
reading cached K/V from memory
feed-forward computation for the new token
LM-head computation
autoregressive sequential dependency
```

This distinction is extremely important.

KV cache makes decode much more efficient.

It does **not** make decode free.

---

## Question 10 — Why is KV cache described as a compute-for-memory tradeoff?

Because we save computation by storing intermediate state.

Without caching:

```text
less persistent KV memory
but
more repeated computation
```

With caching:

```text
more persistent memory
but
much less repeated computation
```

So:

> **KV cache spends memory to save compute.**

This tradeoff becomes extremely important in production LLM serving.

---

## Question 11 — How much memory does the KV cache require?

A useful approximate formula is:

$$
M_{	ext{KV}}
approx
2
	imes
L
	imes
B
	imes
T
	imes
H_{	ext{KV}}
	imes
d_h
	imes
b
$$

where:

- $L$ = number of Transformer layers
- $B$ = batch size or number of active sequences
- $T$ = cached sequence length per sequence
- $H_{	ext{KV}}$ = number of Key/Value heads
- $d_h$ = dimension of each KV head
- $b$ = bytes per stored element

The factor:

$$2$$

exists because we store both:

$$K$$

and:

$$V$$

### Why do we need this mathematics?

Because it immediately tells us how KV memory scales.

KV-cache memory grows linearly with:

$$L$$

$$B$$

$$T$$

$$H_{	ext{KV}}$$

$$d_h$$

and the number of bytes used per element.

This is why long context and high concurrency can consume huge amounts of accelerator memory.

---

## Question 12 — Can we calculate a concrete KV-cache size?

Yes.

Suppose a model has:

$$L=32$$

layers.

Suppose:

$$B=1$$

active sequence.

Suppose the cached context length is:

$$T=8192$$

tokens.

Suppose the model uses:

$$H_{	ext{KV}}=8$$

KV heads.

Suppose:

$$d_h=128$$

and the cache is stored in BF16 or FP16:

$$b=2 	ext{ bytes}$$

Then:

$$
M_{	ext{KV}}
=
2
	imes
32
	imes
1
	imes
8192
	imes
8
	imes
128
	imes
2
$$

This gives:

$$
M_{	ext{KV}}
=
1{,}073{,}741{,}824
	ext{ bytes}
$$

which is approximately:

$$1 	ext{ GiB}$$

for a single 8192-token sequence.

### What if batch size becomes 16?

Because memory scales linearly with $B$:

$$
16 	imes 1 	ext{ GiB}
=
16 	ext{ GiB}
$$

approximately.

This immediately shows why KV memory can become a major concurrency bottleneck.

---

## Question 13 — What happens if sequence length doubles?

From:

$$
M_{	ext{KV}}
propto
T
$$

we know KV memory is linear in cached sequence length.

So if:

$$T 
ightarrow 2T$$

then approximately:

$$
M_{	ext{KV}}

ightarrow
2M_{	ext{KV}}
$$

For the previous example:

```text
8192 tokens
→ about 1 GiB

16384 tokens
→ about 2 GiB
```

assuming all other quantities remain unchanged.

This is different from the logical dense prefill attention relationship count, which has an $n^2$ dependence.

So:

```text
Prefill dense-attention relationships
→ quadratic in prompt length

KV-cache storage
→ linear in cached sequence length
```

---

## Question 14 — What happens if batch size doubles?

The same reasoning applies.

Since:

$$
M_{	ext{KV}}
propto B
$$

doubling active batch size approximately doubles KV-cache memory.

This is one of the most important serving implications.

A GPU might have enough memory for:

```text
model weights
+
one large KV cache
```

but not enough memory for:

```text
model weights
+
hundreds of large KV caches
```

So concurrency is often constrained not only by compute, but also by KV-cache capacity.

---

## Question 15 — What happens if we use FP16, FP8, or another lower-precision KV cache?

In the memory formula:

$$
M_{	ext{KV}}
propto b
$$

where $b$ is bytes per element.

For example:

```text
FP16 / BF16
≈ 2 bytes per element

FP8
≈ 1 byte per element
```

So, in an idealized storage calculation, moving from 2 bytes to 1 byte approximately halves raw KV-cache storage.

But there is a tradeoff.

Lower precision can introduce quantization error.

That can affect:

- attention scores
- retrieved values
- numerical stability
- model quality

The exact effect depends on the model and implementation.

So KV-cache quantization is another form of:

```text
memory efficiency
↔
numerical fidelity
```

---

## Question 16 — Why do GQA and MQA reduce KV-cache memory?

In ordinary Multi-Head Attention, the model may have the same number of Query heads and KV heads.

Suppose:

$$H_Q = H_{	ext{KV}} = 32$$

Then we store Keys and Values for all 32 KV heads.

But in Grouped-Query Attention, several Query heads can share the same Key/Value heads.

For example:

$$H_Q=32$$

but:

$$H_{	ext{KV}}=8$$

Then the KV-cache memory formula uses:

$$H_{	ext{KV}}=8$$

instead of:

$$H_{	ext{KV}}=32$$

So the raw KV-cache size becomes approximately:

$$
rac{8}{32}
=
rac{1}{4}
$$

of the corresponding 32-KV-head configuration, assuming the other dimensions are the same.

In Multi-Query Attention:

$$H_{	ext{KV}}=1$$

so all Query heads share a single Key head and a single Value head.

### Why does this matter?

Because reducing $H_{	ext{KV}}$ reduces:

- KV-cache memory
- KV-cache bandwidth requirements

This is one major reason GQA and MQA are attractive for efficient inference.

### Important nuance

This does **not** mean all architectures behave identically or that fewer KV heads are always better.

There are quality and architectural tradeoffs.

For this chapter, the key concept is simply:

> Fewer KV heads mean fewer Key and Value vectors must be cached per token.

---

## Question 17 — Why can decode still slow down for long contexts even with KV cache?

Because the cache must still be read.

At a decode step, the new Query attends over the cached Keys.

If the cache contains:

$$T$$

positions, then attention still needs to interact with approximately $T$ historical positions.

Conceptually:

$$
q_t K_{	ext{cache}}^T
$$

has a score vector whose length grows with context.

So even though we do not recompute old K/V:

```text
longer context
→ larger KV cache
→ more K/V data to read
→ more attention relationships for the new query
```

That is why KV caching removes a huge amount of redundant work but does not eliminate the cost of long context during decode.

---

## Question 18 — Why is KV-cache memory bandwidth important?

During decode, only a small number of new token positions may be processed at once for each request.

But the system must repeatedly read:

- model weights
- cached Keys
- cached Values

from accelerator memory.

As the cache grows, moving this data can become a major part of decode cost.

This is why production LLM serving often pays close attention to:

```text
HBM capacity
memory bandwidth
KV-cache layout
batching
cache paging
precision
```

### Important edge case

Do not turn this into the absolute claim:

> “Decode is always memory-bandwidth bound.”

Whether decode is compute-bound or bandwidth-bound depends on the workload, hardware, batch size, architecture, sequence length, and implementation.

---

## Question 19 — Does the KV cache preserve the causal order automatically?

The cache stores previously computed Keys and Values in token order.

For a standard left-to-right decode step, the newest token is at the end of the existing context.

That new token may attend to all valid earlier cached positions.

Because no future generated token exists yet, there is no future token in the cache for it to accidentally attend to.

This is why the causal mask during single-token decode can be simpler than during full-prompt prefill.

But real serving systems still need to manage:

- padding
- variable sequence lengths
- batched requests
- sliding windows
- architecture-specific masks

So masking logic does not disappear entirely.

---

## Question 20 — What happens to positional information when we cache K and V?

Position still matters.

Suppose token $x_t$ occurs at position $t$.

Its Key may already include a position-dependent transformation such as RoPE.

Conceptually, we can think of:

$$
k_t
=
	ext{PositionAwareKey}(x_t, t)
$$

When that Key is cached, the positional information associated with position $t$ must remain consistent.

When the next token arrives at position:

$$t+1$$

its Query and Key must use the correct new position.

If cache positions become misaligned, attention semantics become incorrect.

### Why is this an important edge case?

Because KV cache is not just a bag of vectors.

It is an **ordered history tied to token positions**.

---

## Question 21 — What if the earlier context changes after the KV cache has been created?

Then some cached states may no longer be valid.

Suppose the model originally processed:

```text
A B C D
```

and cached the associated K/V state.

If we change the middle of the sequence to:

```text
A B X D
```

then the contextual computations after that changed position can be different.

So we cannot blindly reuse all old cached states.

This is why cache reuse generally depends on an unchanged compatible prefix.

### Mental model

```text
same prefix
→ cached prefix state may be reusable

changed prefix
→ downstream cached state may need recomputation
```

---

## Question 22 — What is prefix caching conceptually?

Suppose many requests begin with exactly the same long prefix, such as a shared system prompt.

Instead of recomputing that prefix every time, a serving system may reuse previously computed KV state for the identical prefix.

Conceptually:

```text
same prompt prefix
        ↓
same compatible model state
        ↓
reuse prefix KV cache
        ↓
avoid repeated prefill work
```

This can reduce repeated prompt-processing cost.

But safe reuse requires compatibility in things such as:

- exact tokenized prefix
- model
- model weights
- attention configuration
- position handling
- relevant inference settings

So prefix caching is powerful, but it is not arbitrary text similarity caching.

---

## Question 23 — What happens to KV cache during beam search?

Beam search maintains multiple candidate sequences.

If one sequence branches into several beams, each beam may eventually need its own continuation-specific KV state.

That means beam search can increase KV-cache memory.

Some implementations share common-prefix cache state and copy or reorder only what is necessary.

The important principle is:

> More simultaneously maintained sequence hypotheses can require more KV state.

We will study beam search itself later in 8.10.

---

## Question 24 — What happens in sliding-window or local attention models?

Not every model attends to the entire history indefinitely.

Suppose a model uses an attention window of:

$$W$$

tokens.

Then the newest token may only need Keys and Values from the most recent:

$$W$$

positions.

Conceptually:

$$
K_{	ext{active}}
=
[k_{t-W+1}, ldots, k_t]
$$

instead of:

$$
[k_1, ldots, k_t]
$$

This can bound active KV memory for those layers.

But the exact behavior is architecture-dependent.

Some models use full attention in some layers and local attention in others.

So we should never assume that every model stores or attends to the exact same KV history.

---

## Question 25 — What happens when many requests have different sequence lengths?

Suppose a server is decoding:

```text
Request A → 500 cached tokens
Request B → 5000 cached tokens
Request C → 12000 cached tokens
```

Their KV-cache requirements are different.

If we allocated one large contiguous fixed-size block for every request, memory could be wasted.

Production serving systems therefore often use more flexible memory-management strategies.

Conceptually:

```text
logical KV sequence
does not necessarily require
one huge physically contiguous memory block
```

Paged or block-based KV-cache management can reduce fragmentation and improve utilization.

The exact implementation varies by serving system.

---

## Question 26 — What are the most important KV-cache edge cases?

### Edge case 1 — Extremely long context

KV memory grows approximately linearly with cached sequence length.

Eventually accelerator memory can become the limiting resource.

### Edge case 2 — Large batch or high concurrency

Every active sequence needs its own cache state.

So high concurrency can exhaust memory even if the model weights themselves fit comfortably.

### Edge case 3 — Beam search

Multiple active sequence hypotheses can multiply KV-state requirements.

### Edge case 4 — Context modification

Changing an earlier token can invalidate cached state after that point.

### Edge case 5 — Position mismatch

Incorrect positional indexing when appending cached tokens can corrupt attention behavior.

### Edge case 6 — Different attention architectures

MHA, GQA, MQA, local attention, and other architectures have different KV-memory behavior.

### Edge case 7 — KV-cache quantization

Lower-precision cache storage saves memory and bandwidth but can introduce numerical error.

### Edge case 8 — Context-window limit

The cache cannot normally grow forever.

Eventually the model/runtime must stop generation, evict context, use a sliding window, or apply some architecture-specific strategy.

---

# KV Cache — Complete Step-by-Step Example

Suppose the prompt contains three tokens:

```text
x1 x2 x3
```

During prefill, one Transformer layer computes:

$$k_1, k_2, k_3$$

and:

$$v_1, v_2, v_3$$

The cache becomes:

$$K_{	ext{cache}}=[k_1,k_2,k_3]$$

$$V_{	ext{cache}}=[v_1,v_2,v_3]$$

The model predicts:

$$P(x_4 mid x_1,x_2,x_3)$$

Suppose $x_4$ is selected.

For $x_4$, the layer computes only the new:

$$q_4, k_4, v_4$$

Then append:

$$K_{	ext{cache}}=[k_1,k_2,k_3,k_4]$$

$$V_{	ext{cache}}=[v_1,v_2,v_3,v_4]$$

The query $q_4$ attends over the cached Keys:

$$
q_4K_{	ext{cache}}^T
$$

and uses the cached Values to form its attention output.

The model then predicts:

$$P(x_5 mid x_1,x_2,x_3,x_4)$$

After $x_5$ is selected, compute only:

$$q_5,k_5,v_5$$

and append:

$$K_{	ext{cache}}=[k_1,k_2,k_3,k_4,k_5]$$

$$V_{	ext{cache}}=[v_1,v_2,v_3,v_4,v_5]$$

Then repeat.

So the full decode pattern is:

```text
Prefill:
compute K/V for prompt
        ↓
cache them

Decode step 1:
compute Q/K/V only for new token
        ↓
append new K/V
        ↓
new Q attends to cached K
        ↓
use cached V

Decode step 2:
compute Q/K/V only for next new token
        ↓
append new K/V
        ↓
repeat
```

---

# 8.5 Mental Model

The simplest mental model is:

> **KV cache remembers the reusable attention history.**

Mathematically:

$$
K_{	ext{cache}}
=
[k_1,k_2,ldots,k_t]
$$

$$
V_{	ext{cache}}
=
[v_1,v_2,ldots,v_t]
$$

For the newly available token:

$$q_{t+1}$$

we reuse the cached history:

$$
q_{t+1}
K_{	ext{cache}}^T
$$

instead of recomputing old Keys and Values.

The key tradeoff is:

```text
More memory
        ↓
Less repeated computation
        ↓
Much faster autoregressive decode
```

Memory line:

> **KV cache saves compute by spending memory.**

And another memory line:

> **We cache K and V because future Queries reuse them; old Queries are usually no longer needed.**

---

# Final Bridge — Why Does KV Cache Lead to Decoding Strategies?

We now have an efficient mechanism for producing next-token logits during autoregressive generation.

The model can repeatedly produce a vector:

$$
z in mathbb{R}^{|V|}
$$

containing one logit for every vocabulary token.

But KV cache does **not** answer the next question:

> Which token should we actually choose from those logits?

The simplest possible rule is:

> Choose the token with the highest score.

That leads directly to:

```text
8.6 Greedy Decoding
```

But greedy decoding creates another problem:

> What if always choosing the maximum-probability token makes generation too deterministic or locally shortsighted?

That leads to:

```text
8.7 Temperature
```

So the next study block is:

```text
8.6 Greedy Decoding
8.7 Temperature
```
