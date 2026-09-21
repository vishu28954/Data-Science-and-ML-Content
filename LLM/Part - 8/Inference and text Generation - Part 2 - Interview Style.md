# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 2 — Interview Style

## Covers

```text
8.3 Prefill phase
8.4 Decode phase
```

This file contains only interview-style answers. The detailed-study notes remain separate.

---

# 8.3 Prefill Phase — Interview Style

## 1. What is the prefill phase in LLM inference?

### Interview answer

Prefill is the first Transformer forward pass over the complete prompt.

If the prompt contains:

$$
x_1,x_2,ldots,x_n
$$

all of those prompt tokens are already known before generation begins.

After token embedding and positional handling, the prompt can be represented as:

$$
Xinmathbb{R}^{n	imes d_{model}}
$$

The model processes these known prompt positions together through the Transformer layers, while causal masking ensures that each position can only use information from itself and earlier positions.

The final prompt position then produces the logits for the first generated token.

---

## 2. If an LLM is autoregressive, why can prompt tokens be processed in parallel?

### Interview answer

Autoregressive does not mean that the computer must physically process prompt positions one after another.

It means that position $i$ must not use information from future positions.

Because the full prompt is already known, the model can compute representations for many prompt positions together in matrix operations while applying a causal mask.

So these two ideas are different:

```text
Autoregressive dependency
→ token i may only depend on positions ≤ i

Parallel computation
→ many allowed calculations can still run together
```

That is why prefill is parallelizable across known prompt positions.

---

## 3. What is the role of the causal mask during prefill?

### Interview answer

The causal mask prevents each prompt position from attending to future positions.

The attention logits can be written as:

$$
A
=
rac{QK^T}{sqrt{d_k}}
+
M
$$

where $M$ is the causal mask.

Conceptually, for four positions:

$$
M=
egin{bmatrix}
0 & -infty & -infty & -infty\\
0 & 0 & -infty & -infty\\
0 & 0 & 0 & -infty\\
0 & 0 & 0 & 0
end{bmatrix}
$$

After softmax, the $-infty$ positions receive probability zero.

This lets the model compute all known prompt positions together without leaking future information.

---

## 4. What are Q, K, and V during prefill?

### Interview answer

Starting from:

$$
Xinmathbb{R}^{n	imes d_{model}}
$$

the attention layer computes:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

For a simplified single-head view:

$$
Q,Kinmathbb{R}^{n	imes d_k}
$$

and:

$$
Vinmathbb{R}^{n	imes d_v}
$$

So each prompt position has a query, key, and value representation.

The queries determine what each position is looking for, the keys determine how positions can be matched, and the values contain the information that is aggregated.

---

## 5. Why does the attention score matrix have shape n by n?

### Interview answer

Because:

$$
Qinmathbb{R}^{n	imes d_k}
$$

and:

$$
K^Tinmathbb{R}^{d_k	imes n}
$$

Therefore:

$$
QK^Tinmathbb{R}^{n	imes n}
$$

Each entry $(i,j)$ represents the attention compatibility between query position $i$ and key position $j$.

So for ordinary dense attention, there are logically:

$$
n^2
$$

query-key relationships.

This is why the attention component becomes more expensive as prompt length increases.

---

## 6. Does an implementation always store the full n by n attention matrix?

### Interview answer

Not necessarily.

The logical dense-attention problem still has $n^2$ query-key relationships, but optimized kernels can compute attention in blocks or tiles without materializing the full matrix in memory.

So I would distinguish:

```text
logical dense-attention relationships
→ O(n²)

physical memory implementation
→ may avoid storing the entire n×n matrix
```

This distinction is important when discussing optimized attention kernels.

---

## 7. Why is attention divided by square root of d_k?

### Interview answer

The scaled dot-product attention uses:

$$
rac{QK^T}{sqrt{d_k}}
$$

The intuition is that if query and key components have roughly zero mean and unit variance, then their dot product:

$$
qcdot k
=
sum_{r=1}^{d_k}q_rk_r
$$

has variance that grows approximately with $d_k$:

$$
operatorname{Var}(qcdot k)approx d_k
$$

so its standard deviation grows approximately as:

$$
sqrt{d_k}
$$

Dividing by $sqrt{d_k}$ keeps the attention logits at a more stable scale before softmax.

Without scaling, large dot products can make softmax extremely peaked and numerically harder to optimize.

---

## 8. What is the complete attention equation during prefill?

### Interview answer

A simplified single-head attention equation is:

$$
operatorname{Attention}(Q,K,V)
=
operatorname{softmax}
left(
rac{QK^T}{sqrt{d_k}}+M
ight)V
$$

where $M$ is the causal mask.

If:

$$
S=
operatorname{softmax}
left(
rac{QK^T}{sqrt{d_k}}+M
ight)
$$

then:

$$
Sinmathbb{R}^{n	imes n}
$$

and:

$$
SVinmathbb{R}^{n	imes d_v}
$$

So each position gets a weighted combination of allowed value vectors from itself and earlier positions.

---

## 9. Which hidden state produces the first generated token?

### Interview answer

For a decoder-only model, the hidden state at the final prompt position is normally used to predict the first generated token.

If the prompt ends at position $n$, then after the final Transformer layer we have:

$$
h_1,h_2,ldots,h_n
$$

The LM head uses:

$$
h_n
$$

to produce next-token logits:

$$
z_{n+1}
=
W_{LM}h_n+b
$$

and then conceptually:

$$
P(x_{n+1}mid x_1,ldots,x_n)
=
operatorname{softmax}(z_{n+1})
$$

---

## 10. Why does a longer prompt usually increase Time to First Token?

### Interview answer

A longer prompt gives the model more token positions to process before the first generated token can be produced.

Prefill includes projection layers, attention, feed-forward layers, and other Transformer operations over the prompt.

The attention score computation for dense attention includes a term like:

$$
O(n^2d_k)
$$

while projection and feed-forward work scale differently, often linearly in $n$ for fixed model width.

So longer prompts usually increase prefill work, which often increases Time to First Token.

But TTFT is not only prefill time. It can also include queueing, batching delay, tokenization, networking, or cold-start overhead.

---

## 11. Is it correct to say prefill is always quadratic?

### Interview answer

No. That statement is too broad.

The dense attention-score component has a quadratic dependence on sequence length:

$$
O(n^2d_k)
$$

but other Transformer operations such as projections and feed-forward layers have different scaling.

So I would say:

> Dense self-attention contains a quadratic sequence-length component during prefill, but total prefill cost depends on the entire architecture and implementation.

That is a more accurate production answer.

---

## 12. What important prefill edge cases should I mention?

### Interview answer

Important cases include very short prompts, very long prompts, variable-length batched prompts, padding, and optimized attention kernels.

In a batch, padding positions should not be treated as real content, so padding masks may be needed in addition to the causal mask.

For long prompts, attention and memory pressure increase significantly.

And optimized kernels can reduce memory traffic without changing the model's causal dependency rules.

---

# Complete 8.3 Interview Answer

If asked:

```text
What is the prefill phase in LLM inference?
```

You can say:

```text
Prefill is the first Transformer forward pass over the complete known prompt.

Because all prompt tokens already exist before generation starts, the model can process many prompt positions together rather than running one separate forward pass for each token. The model still remains autoregressive because a causal mask prevents each position from attending to future positions.

Starting from the prompt tensor X with shape n by d_model, the model computes Q, K, and V. In dense self-attention, QK-transpose has logical shape n by n, so the attention part contains a quadratic dependence on prompt length. The scores are scaled by square root of d_k, masked, normalized with softmax, and used to aggregate the value vectors.

After the final Transformer layer, the hidden state at the last prompt position is projected through the LM head to produce the first next-token logits.

Longer prompts therefore usually increase prefill work and can increase Time to First Token.
```

---

# 8.4 Decode Phase — Interview Style

## 13. What is the decode phase?

### Interview answer

Decode begins after prefill has produced the first next-token distribution.

The model selects one token:

$$
x_{n+1}
$$

and appends it to the prompt.

Then it predicts:

$$
P(x_{n+2}mid x_1,ldots,x_n,x_{n+1})
$$

After selecting $x_{n+2}$, it repeats again.

So decode is the repeated token-by-token generation stage after the prompt has been processed.

---

## 14. Why can't decode process all future output tokens in parallel?

### Interview answer

Because future generated token identities are not known yet.

For example:

$$
P(x_{n+1},x_{n+2},x_{n+3}mid x_{le n})
$$

factorizes autoregressively as:

$$
P(x_{n+1}mid x_{le n})
cdot
P(x_{n+2}mid x_{le n+1})
cdot
P(x_{n+3}mid x_{le n+2})
$$

The second term depends on the actual token selected for $x_{n+1}$, and the third depends on $x_{n+2}$.

So ordinary generation has a sequential dependency across output positions.

---

## 15. What does one decode attention step look like mathematically?

### Interview answer

At decode step $t$, the newly available position produces a query:

$$
q_tinmathbb{R}^{1	imes d_k}
$$

The previous key vectors can be represented as:

$$
K_{1:t}inmathbb{R}^{t	imes d_k}
$$

Therefore:

$$
q_tK_{1:t}^T
in
mathbb{R}^{1	imes t}
$$

because:

$$
(1	imes d_k)(d_k	imes t)
=
1	imes t
$$

So the new token position obtains one attention score for every available key position in its context.

After softmax, those weights are applied to the previous values.

---

## 16. What is the key mathematical difference between prefill and decode?

### Interview answer

In a simplified single-head view:

During prefill:

$$
QK^Tinmathbb{R}^{n	imes n}
$$

because many known prompt queries interact with many prompt keys.

During one decode step:

$$
q_tK_{1:t}^T
in
mathbb{R}^{1	imes t}
$$

because only one new query is being processed against the accumulated history.

So the mental model is:

```text
Prefill:
many queries × many known keys

Decode:
one new query × growing history
```

---

## 17. Where do the old keys and values come from during decode?

### Interview answer

They were computed when the previous tokens were processed.

A naive implementation could recompute them from scratch every generation step, but that is redundant because old token representations do not change.

This observation motivates KV caching:

```text
old keys and values already exist
→ store them
→ reuse them on later decode steps
```

That is the next major inference optimization.

---

## 18. Why is decoding without a KV cache wasteful?

### Interview answer

Without caching, the model may repeatedly process the growing prefix.

If the prompt length is $P$ and we generate $G$ tokens, the prefix lengths are roughly:

$$
P,;P+1,;P+2,ldots,P+G-1
$$

The total number of repeatedly processed token positions is:

$$
sum_{g=0}^{G-1}(P+g)
=
GP+rac{G(G-1)}{2}
$$

For example, with:

$$
P=1000
$$

and:

$$
G=100
$$

this becomes:

$$
100	imes1000+rac{100	imes99}{2}
=
104{,}950
$$

token-position evaluations in this simplified repeated-prefix view, even though only 100 new positions were generated.

That redundancy is exactly why KV caching is so important.

---

## 19. Does KV caching make decode independent of context length?

### Interview answer

No.

Caching prevents recomputation of old keys and values, but the new query still has to attend over the available history.

At step $t$:

$$
q_tK_{1:t}^T
in
mathbb{R}^{1	imes t}
$$

So the number of key positions available to the new query still grows with context length.

KV cache removes redundant recomputation; it does not remove the historical context.

---

## 20. Why does output length directly affect latency?

### Interview answer

Because each new generated token requires another autoregressive decode step.

If average decode time per token is $	au$, then generating $G$ tokens takes roughly:

$$
G	au
$$

of sequential model-side generation work, ignoring overlap and serving overhead.

For example, at 50 tokens per second:

$$
	au=rac{1}{50}=0.02	ext{ s}=20	ext{ ms}
$$

Generating 200 tokens would take roughly:

$$
200	imes20	ext{ ms}=4	ext{ s}
$$

in this simplified calculation.

That is why long outputs affect latency differently from long prompts.

---

## 21. What is the difference between TTFT and TPOT?

### Interview answer

TTFT means Time to First Token.

It measures how long the user waits before generation starts.

TPOT means Time Per Output Token.

It measures how quickly new tokens arrive after generation has started.

A useful mental model is:

```text
Long prompt
→ more prefill work
→ often higher TTFT

Long output
→ more sequential decode steps
→ often higher total generation time / slower streaming experience
```

These metrics measure different parts of the inference path.

---

## 22. Why is decode often described as memory-bandwidth sensitive?

### Interview answer

During decode, a large model performs computation for only a small number of new token positions per request, while model weights and KV-cache data still need to be read from memory.

That can make memory movement a major bottleneck.

However, I would not say decode is universally memory-bound. Whether it is compute-bound or bandwidth-bound depends on batch size, hardware, quantization, model architecture, sequence length, kernels, and parallelism strategy.

---

## 23. Can decode be parallelized at all?

### Interview answer

Yes, but we need to distinguish types of parallelism.

Independent requests can be batched together.

Model computation can also be distributed across devices using tensor or pipeline parallelism.

What cannot be straightforwardly parallelized in ordinary autoregressive decoding is the exact future token sequence for one request, because token $t+1$ must be selected before token $t+2$ is known.

Techniques such as speculative decoding can propose multiple tokens and verify them, but they do not remove the underlying autoregressive dependency of the target model.

---

## 24. What decode edge cases should I know?

### Interview answer

Important cases include immediate EOS generation, very long outputs, context-window exhaustion, variable output lengths inside a batch, and requests finishing at different times.

These cases make serving harder because the active batch can change at every decode step.

Also, tokens per second is not the same as words per second because one token can represent a character, subword, full word, punctuation, or whitespace-plus-text.

---

# Complete 8.4 Interview Answer

If asked:

```text
What is the decode phase in LLM inference?
```

You can say:

```text
Decode is the token-by-token generation phase that begins after prefill.

After the prompt has been processed, the model produces a distribution for the first generated token. Once one token is selected, it becomes part of the context and the model performs another forward step to predict the next token.

This is sequential because autoregressive factorization means token t+1 depends on the actual tokens selected before it.

At one decode step, the new query can be represented as a 1 by d_k vector, while previous keys form a t by d_k matrix. Their product therefore has shape 1 by t, meaning the newest token attends over the growing history.

A naive implementation would repeatedly recompute old key and value representations. Since old tokens do not change, that is wasteful. This is exactly what motivates the KV cache.

Decode latency therefore grows with the number of generated tokens, and the user experience is often measured using metrics such as Time Per Output Token.
```

---

# Prefill vs Decode — Interview Comparison

## 25. How would you compare prefill and decode in an interview?

### Interview answer

Prefill processes the already-known prompt, while decode processes newly generated tokens one at a time.

During prefill, many prompt queries and keys are processed together, so dense attention has a logical $n	imes n$ structure.

During decode, one new query attends over the accumulated history, giving a $1	imes t$ attention-score vector in the simplified single-head view.

Prefill strongly influences Time to First Token, while decode strongly influences token streaming speed and total output latency.

The main systems optimization that naturally follows decode is KV caching, because previously computed keys and values should be reused rather than recomputed.

---

# Combined Interview Answer

If asked:

```text
Explain prefill and decode in LLM inference.
```

Answer:

```text
LLM inference has two distinct phases.

The first is prefill. The complete prompt is already known, so the model can process many prompt positions together in one large Transformer forward pass. Causal masking preserves autoregressive behavior by ensuring each position can only use itself and earlier positions. In dense self-attention, the logical score matrix has shape n by n, which is why long prompts can significantly increase prefill cost and Time to First Token.

After prefill, the model generates the first output token. That begins the decode phase. Future output tokens are not known ahead of time, so the model generates one token, appends it to the context, and then predicts the next token.

During one decode step, the new query attends over the accumulated history, so the attention-score shape is roughly 1 by t for one head. Because previously processed tokens do not change, repeatedly recomputing their key and value representations would be wasteful, which naturally motivates KV caching.

So the key distinction is: prefill parallelizes across known prompt positions, while decode proceeds sequentially across unknown future token positions.
```

---

# Crisp Version

```text
Prefill = process the known prompt in parallel while enforcing causality with a mask.

Decode = generate one new token at a time because each future token depends on the token selected before it.

Prefill mainly affects Time to First Token.

Decode mainly affects token streaming speed and total output latency.

The decode phase naturally motivates KV cache because old keys and values should be reused instead of recomputed.
```

---

# Most Important Memory Lines

```text
Autoregressive does not mean the known prompt must be processed serially.
```

```text
Prefill parallelizes across known prompt positions; decode cannot parallelize unknown future token identities in the same way.
```

```text
Prefill: many queries × many keys.
Decode: one new query × growing history.
```

```text
KV cache exists because old tokens do not change, so their key and value representations should not be recomputed.
```

---

The next detailed-study block is:

```text
8.5 KV Cache
```
