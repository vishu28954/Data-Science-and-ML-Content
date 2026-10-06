# Part 8 — Inference and Text Generation — Interview Style

This file is intentionally separate from Detailed Study Mode. The goal is to give **speakable interview answers** that can be expanded when the interviewer probes deeper.

The recommended pattern is:

~~~text
Start with the 30–60 second answer
        ↓
Add the key equation if asked
        ↓
Use the short example if asked
        ↓
Go deeper only when the interviewer probes
~~~

---

# 8.14 Why Generation Is Sequential

## Question 1 — Why is autoregressive generation sequential?

**Answer:**  
A decoder-only LLM generates autoregressively, so each output token is conditioned on the prompt and the tokens already generated:

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid x,y_{<t})
$$

The important point is that the distribution for $y_t$ depends on the **actual previously selected tokens**. Until $y_t$ is selected, the exact context needed to predict $y_{t+1}$ is not known. So accepted output positions form a sequential dependency chain.

**One-line memory:** Known history determines the next distribution.

---

## Question 2 — Why can't an LLM generate the entire answer in one normal forward pass?

**Answer:**  
Because later token distributions depend on earlier generated choices. For two output tokens:

$$
P(y_1,y_2\mid x)
=
P(y_1\mid x)\,P(y_2\mid x,y_1)
$$

The second factor cannot be the exact distribution for the final path until $y_1$ is known.

A simple example is:

~~~text
Prompt: "The capital of France is"
y1 = "Paris"
y2 = "."
~~~

If $P(y_1\mid x)=0.8$ and $P(y_2\mid x,y_1)=0.9$, then:

$$
P(y_1,y_2\mid x)=0.8\times0.9=0.72
$$

If a different first token were selected, the second-token distribution could change.

---

## Question 3 — If generation is sequential, why can prefill process the prompt in parallel?

**Answer:**  
Because all prompt tokens are already known. The Transformer can compute representations for all prompt positions in one batched forward pass while the causal mask prevents position $i$ from attending to future positions.

For one attention head:

$$
Q\in\mathbb{R}^{n\times d_k},\quad
K\in\mathbb{R}^{n\times d_k},\quad
V\in\mathbb{R}^{n\times d_v}
$$

and:

$$
QK^T\in\mathbb{R}^{n\times n}
$$

So causal dependency does not imply serial execution when the conditioned-on inputs are already known.

---

## Question 4 — What is the difference between prefill and decode?

**Answer:**  
**Prefill** processes the whole known prompt and produces the logits for the first generated token while building the prompt KV cache.

**Decode** processes one newly accepted token per sequence per ordinary forward step, reusing cached K/V from the existing history to produce logits for the following token.

So:

~~~text
Prefill → many known positions
Decode  → one new accepted position at a time
~~~

---

## Question 5 — What is the exact timeline from prefill to generation?

**Answer:**  
Suppose the prompt is $x_1,x_2,x_3,x_4$.

Prefill computes:

$$
P(y_1\mid x_1,x_2,x_3,x_4)
$$

Then $y_1$ is selected. The first one-token decode pass processes $y_1$ and produces:

$$
P(y_2\mid x_1,x_2,x_3,x_4,y_1)
$$

Then $y_2$ is selected and processed to obtain the distribution for $y_3$.

**Important:** The first generated token comes from the **prefill logits**. We do not need a separate decode pass before selecting $y_1$.

---

## Question 6 — Why doesn't a causal mask itself make prefill sequential?

**Answer:**  
The mask controls **information visibility**, not necessarily execution order. Prompt position $i$ can attend only to positions up to $i$, but all prompt embeddings are already available, so the GPU can compute many position-wise matrix operations together and apply the causal mask inside the same forward pass.

---

## Question 7 — Why can training parallelize token positions more than inference?

**Answer:**  
During teacher-forced training, the complete ground-truth sequence is already known. The model can compute:

$$
P(x_2\mid x_1),\;
P(x_3\mid x_1,x_2),\;\ldots
$$

for many positions in parallel under a causal mask.

At inference time, future generated tokens are not known. The model has to create them first, so the next position cannot use the final accepted history until that history exists.

---

## Question 8 — Why can't we just guess several future tokens in parallel?

**Answer:**  
We can propose them, but they are provisional. If the first proposed token differs from the token accepted by the target model, then the later conditional distributions change:

$$
P(y_2\mid x,A)
\neq
P(y_2\mid x,D)
$$

in general.

That is why ordinary exact autoregressive decoding advances one accepted token at a time.

---

## Question 9 — How does sampling show the sequential dependency?

**Answer:**  
Suppose the current distribution is:

| Token | Probability |
|---|---:|
| A | 0.50 |
| B | 0.30 |
| C | 0.20 |

Sampling may choose A in one run and B in another. The next distributions become:

$$
P(y_{t+1}\mid x,y_{<t},A)
$$

and:

$$
P(y_{t+1}\mid x,y_{<t},B)
$$

Those can be very different, so one stochastic decision can redirect the entire future sequence.

---

## Question 10 — Is greedy decoding sequential even though it is deterministic?

**Answer:**  
Yes. Greedy decoding selects:

$$
y_t
=
\underset{v}{\operatorname{argmax}}
\;
P(v\mid x,y_{<t})
$$

Determinism removes randomness, but the next distribution still depends on the selected $y_t$. So greedy decoding is still autoregressively sequential.

---

## Question 11 — Is beam search sequential?

**Answer:**  
Yes across generation depth, although it has parallelism across beams at the same depth.

With beam width $w$, the system can batch the current forward computation for the $w$ active hypotheses. But it must score and prune those hypotheses before it knows which histories exist at depth $t+1$.

So beam search has:

~~~text
Parallelism across beams
+
Sequential dependence across depths
~~~

---

## Question 12 — If one request is sequential, how can servers still use GPUs efficiently?

**Answer:**  
By batching the current decode steps of many independent requests.

For example:

~~~text
Request A → needs next token
Request B → needs next token
Request C → needs next token
~~~

The server can process those three current positions together. Each request remains sequential internally, but there is useful **across-request parallelism**.

---

## Question 13 — Does KV caching remove the sequential dependency?

**Answer:**  
No. KV caching removes redundant computation, not the autoregressive dependency.

Instead of recomputing K/V for the whole prefix at every step, the model stores historical K/V and computes only the new token's state. The next accepted token still has to be known before the following decode step can begin.

**Memory line:** KV cache removes recomputation, not causality.

---

## Question 14 — What does longer context change during decode?

**Answer:**  
The dependency chain remains one token after another, but each step can become more expensive because the new query attends over a longer cached history.

For one head:

$$
q_tK_{1:t}^{T}
\in
\mathbb{R}^{1\times t}
$$

So attention work and KV-cache reads grow as $t$ grows.

---

## Question 15 — What are TTFT and inter-token latency?

**Answer:**  
**TTFT**, or time to first token, is mainly the time needed to queue the request, prefill the prompt, and select the first output token.

A simplified view is:

$$
T_{\mathrm{TTFT}}
\approx
T_{\mathrm{queue}}
+
T_{\mathrm{prefill}}
+
T_{\mathrm{first\ selection}}
$$

After that, **inter-token latency** reflects the time between successive decode outputs.

---

## Question 16 — How do you quantify the sequential critical path?

**Answer:**  
If a request needs $N$ dependent decode iterations and iteration $j$ takes $t_j$:

$$
T_{\mathrm{decode}}
=
\sum_{j=1}^{N}t_j
$$

For example, 100 decode steps at roughly 20 ms each gives about:

$$
100\times20\ \mathrm{ms}
=
2\ \mathrm{s}
$$

ignoring queueing, overlap, and changing per-token cost.

---

## Question 17 — What is speculative decoding?

**Answer:**  
A smaller or faster draft model proposes several future tokens. The larger target model verifies the proposed block. If multiple tokens are accepted, the target can advance several positions per verification cycle.

It reduces target-model sequential cycles, but it does **not** remove the underlying autoregressive consistency requirement.

---

## Question 18 — Is sequential generation fundamental to every sequence model?

**Answer:**  
No. It follows from the autoregressive factorization used by GPT-style decoders:

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
$$

Non-autoregressive architectures can predict multiple positions together or iteratively refine a sequence. The sequential behavior discussed here is a consequence of the autoregressive modeling choice.

---

# 8.15 Why Inference Is Expensive

## Question 19 — Why is LLM inference expensive even without backpropagation?

**Answer:**  
Because inference still repeatedly executes a very large model. For each output token, the system may need to move large weight tensors, run billions of parameter operations, read and extend KV cache, perform attention over the growing context, and produce vocabulary logits.

So removing backpropagation makes inference cheaper than training, but it does not make inference cheap.

---

## Question 20 — How do you estimate model-weight memory?

**Answer:**  
A first-order estimate is:

$$
M_{\mathrm{weights}}
\approx
P\times b
$$

where $P$ is parameter count and $b$ is bytes per parameter.

For a 7B model in 16-bit storage:

$$
7\times10^9\times2
=
14\times10^9\ \mathrm{bytes}
$$

or roughly 14 GB in decimal units, before runtime overhead.

---

## Question 21 — Why do weights still matter every decode step if they are already in GPU memory?

**Answer:**  
Because the matrices must still be read through the memory hierarchy and consumed by the compute units for every forward pass. The weights do not come from disk each token, but repeatedly streaming large parameter tensors from accelerator memory can still dominate low-batch decode.

---

## Question 22 — What does memory-bandwidth limited mean?

**Answer:**  
It means performance is constrained more by how quickly data can be moved than by how many arithmetic operations the GPU can theoretically perform.

A useful concept is arithmetic intensity:

$$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

Low-batch decode often has relatively low reuse of each weight load, so memory bandwidth can become the bottleneck.

---

## Question 23 — Why is prefill often more compute-bound while decode is often more memory-bound?

**Answer:**  
Prefill processes many prompt tokens together, creating large matrix-matrix operations that reuse weights efficiently and achieve high arithmetic intensity.

Low-batch decode may process only one new token per active sequence while still touching large model matrices, so weight movement becomes comparatively more important.

This is a heuristic, not a universal rule.

---

## Question 24 — What is the rough dense-model FLOPs-per-token rule?

**Answer:**  
A common first-order estimate for the large parameterized matrix multiplications is:

$$
\mathrm{FLOPs}_{\mathrm{token}}
\sim
2P
$$

where $P$ is parameter count.

So a 7B dense model is roughly on the order of 14 GFLOPs of parameterized work per token, while a 70B model is roughly 140 GFLOPs, before sequence-dependent attention and other operations.

---

## Question 25 — Why doesn't the $2P$ rule fully capture long-context inference?

**Answer:**  
Because attention introduces sequence-length-dependent work.

At decode step $t$, one attention head computes:

$$
q_tK_{1:t}^{T}
$$

which costs approximately:

$$
O(td_k)
$$

per head, and then applies attention weights to $V_{1:t}$.

So longer context increases attention work and KV-cache reads even though the model parameter count has not changed.

---

## Question 26 — What would happen without a KV cache?

**Answer:**  
The model would repeatedly recompute historical representations for the old prefix at every decode step.

With a KV cache, old Keys and Values are stored once, the new token's K/V are appended, and the new query attends to the cached history.

That is one of the main reasons autoregressive inference is practical.

---

## Question 27 — Does KV caching make decode cheap after the model is loaded?

**Answer:**  
No. Each new token still requires the model to:

- Run through all Transformer layers.
- Read model weights.
- Compute the new Q/K/V and MLP activations.
- Read historical K/V for attention.
- Write new K/V.
- Produce vocabulary logits.
- Select the next token.

KV cache reduces recomputation; it does not eliminate per-token model execution.

---

## Question 28 — How do you estimate KV-cache memory?

**Answer:**  
A useful first-order estimate is:

$$
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{KV}}d_h b
$$

where $L$ is layers, $B$ active sequence histories, $T$ cached tokens, $H_{\mathrm{KV}}$ KV heads, $d_h$ head dimension, and $b$ bytes per cached element.

The factor 2 is for Keys and Values.

---

## Question 29 — Can you give a numerical KV-cache example?

**Answer:**  
For:

$$
L=32,\quad
B=1,\quad
T=8192,\quad
H_{\mathrm{KV}}=8,\quad
d_h=128,\quad
b=2
$$

the estimate gives:

$$
M_{\mathrm{KV}}
=
2\times32\times1\times8192\times8\times128\times2
$$

which equals:

$$
1{,}073{,}741{,}824\ \mathrm{bytes}
=
1\ \mathrm{GiB}
$$

for this illustrative configuration.

---

## Question 30 — Why do MQA and GQA help inference?

**Answer:**  
KV-cache memory is proportional to the number of KV heads:

$$
M_{\mathrm{KV}}\propto H_{\mathrm{KV}}
$$

Multi-Query Attention and Grouped-Query Attention use fewer K/V heads than query heads, so they reduce KV-cache size and K/V bandwidth during attention.

---

## Question 31 — How does prompt length affect inference cost?

**Answer:**  
A longer prompt increases prefill work, TTFT, and initial KV-cache size.

For prompt length $n$, the attention-score structure has roughly:

$$
n\times n
$$

position interactions per head, so the attention-score component scales approximately as:

$$
O(n^2d_k)
$$

Longer prompts also make decode start from a larger cached history.

---

## Question 32 — Is attention always the dominant inference cost?

**Answer:**  
No. At shorter contexts, dense linear and MLP layers can dominate because they involve huge parameter matrices. At very long contexts, attention and KV-cache traffic become increasingly significant.

The bottleneck depends on architecture, hidden size, context length, batch size, KV-head count, precision, hardware, and kernels.

---

## Question 33 — What does the LM head cost?

**Answer:**  
The final hidden state:

$$
h_t\in\mathbb{R}^{d_{\mathrm{model}}}
$$

is projected to vocabulary logits:

$$
z_t=W_{\mathrm{LM}}h_t+b
$$

with:

$$
W_{\mathrm{LM}}
\in
\mathbb{R}^{V\times d_{\mathrm{model}}}
$$

For $V=50{,}000$ and $d_{\mathrm{model}}=4096$, that matrix contains about 204.8 million weight entries.

---

## Question 34 — Is softmax or top-k/top-p the main inference cost?

**Answer:**  
They matter, especially with large vocabularies and batches, but for large LLMs the Transformer forward pass is usually the central cost.

Softmax is:

$$
P(i)
=
\frac{e^{z_i}}
{\sum_{j=1}^{V}e^{z_j}}
$$

Token selection may add top-k, top-p, repetition penalties, or masking, but these are usually smaller than executing the entire model.

---

## Question 35 — Why does output length strongly affect total inference cost?

**Answer:**  
Because every additional accepted output token requires another generation decision and usually another dependent decode pass.

A simplified request-time model is:

$$
T_{\mathrm{request}}
\approx
T_{\mathrm{queue}}
+
T_{\mathrm{prefill}}
+
\sum_{j=1}^{N_{\mathrm{decode}}}t_j
$$

Longer output means more model execution, more KV growth, more attention over history, and more token-selection work.

---

## Question 36 — How do prompt-heavy and output-heavy requests differ?

**Answer:**  
A long-prompt, short-output request mainly stresses prefill and TTFT.

A short-prompt, long-output request may have cheap prefill but a long sequential decode path.

For example:

~~~text
Request A: 8,000 prompt tokens + 20 output tokens
Request B:   200 prompt tokens + 2,000 output tokens
~~~

A and B are expensive in different ways.

---

## Question 37 — What is latency versus throughput?

**Answer:**  
**Latency** is how long one request waits: TTFT, inter-token latency, or total request time.

**Throughput** is how much total work the server completes per unit time, such as tokens per second or requests per second.

A serving optimization can improve throughput while slightly increasing individual queueing latency.

---

## Question 38 — Why does batching improve throughput?

**Answer:**  
Batching lets the hardware apply the same model weights to several active sequences in larger matrix operations.

That can improve:

- Weight reuse.
- Arithmetic intensity.
- Accelerator utilization.
- Total tokens per second.

The tradeoff is more KV memory, activation memory, and scheduling complexity.

---

## Question 39 — What is continuous batching?

**Answer:**  
Continuous batching dynamically changes the active batch as requests arrive and finish.

Instead of waiting for every sequence in a static batch to complete, the server can replace finished requests with new ones. That reduces idle capacity and improves serving throughput.

---

## Question 40 — Why can long outputs hurt serving efficiency?

**Answer:**  
A long-running request keeps its KV cache and scheduler slot alive for many more decode steps.

That can:

- Hold memory.
- Increase tail latency.
- Reduce admission capacity.
- Keep shared compute resources occupied.

So output-length distribution matters for system capacity planning.

---

## Question 41 — Why are large models split across multiple GPUs?

**Answer:**  
Because one GPU may not have enough memory, compute, or bandwidth.

Common approaches include tensor parallelism, pipeline parallelism, expert parallelism, and replicated data-parallel serving.

The benefit is more aggregate capacity; the cost is inter-device communication.

---

## Question 42 — Why can communication become a bottleneck?

**Answer:**  
With tensor parallelism, different GPUs compute partial results and may need collectives such as all-reduce or all-gather.

Decode is latency-sensitive, and even small communication delays can repeat across many layers and many output tokens. So interconnect bandwidth and latency become part of serving performance.

---

## Question 43 — How does quantization help inference?

**Answer:**  
Quantization stores weights, and sometimes activations or KV cache, using fewer bits.

Moving from 16-bit to 8-bit ideally halves weight memory; 16-bit to 4-bit ideally quarters it.

Benefits include lower memory footprint and bandwidth demand, but actual speedup depends on kernels, hardware, dequantization overhead, and accuracy tradeoffs.

---

## Question 44 — How does prefix caching help?

**Answer:**  
If many requests share an identical tokenized prefix—such as the same system prompt or document—the server can reuse the prefill/KV state for that prefix.

That reduces repeated prefill work and can improve TTFT. Once requests diverge, ordinary decode costs remain.

---

## Question 45 — What is paged KV-cache management?

**Answer:**  
Paged KV systems divide KV memory into blocks or pages and map logical sequence positions to physical memory blocks.

This helps reduce fragmentation and supports dynamic request lifetimes and continuous batching more efficiently.

It changes memory management, not the autoregressive mathematics.

---

## Question 46 — How does FlashAttention help?

**Answer:**  
FlashAttention reorganizes attention execution to reduce memory traffic and avoid materializing some large intermediates.

It does not change:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

It changes **how** that formula is executed on hardware.

---

## Question 47 — Why is long-context inference especially challenging?

**Answer:**  
Because several costs rise together:

- Prefill attention has a quadratic position-interaction component.
- KV-cache memory grows roughly linearly with cached token count.
- Decode reads a larger historical cache.
- More KV memory reduces concurrency.
- Data movement becomes increasingly important.

So long context stresses compute, bandwidth, and memory capacity at the same time.

---

## Question 48 — What is the best final answer to "Why is LLM inference expensive?"

**Answer:**  
LLM inference is expensive because a large model must repeatedly execute billions of parameter operations while moving large amounts of weight and KV-cache data, and autoregressive generation requires those forward passes along a sequential token dependency chain.

The main cost drivers are:

~~~text
Model size
+ Prompt length
+ Output length
+ KV-cache size
+ Memory bandwidth
+ Attention over context
+ Batch/concurrency pressure
+ Multi-GPU communication
~~~

A useful high-level decomposition is:

$$
T_{\mathrm{request}}
=
T_{\mathrm{queue}}
+
T_{\mathrm{prefill}}
+
T_{\mathrm{decode}}
+
T_{\mathrm{other}}
$$

and:

$$
M_{\mathrm{total}}
\approx
M_{\mathrm{weights}}
+
M_{\mathrm{KV}}
+
M_{\mathrm{activations}}
+
M_{\mathrm{runtime}}
$$

**Final memory line:** Inference cost is not one bottleneck; it is the interaction of repeated autoregressive execution, large weights, memory movement, KV state, context length, and serving concurrency.

---

# Final Interview Memory Map

~~~text
WHY SEQUENTIAL?
Autoregressive chain rule
→ next distribution depends on accepted history

PREFILL VS DECODE
Known prompt → parallel prefill
Unknown future → sequential decode

KV CACHE
Avoid old K/V recomputation
but keep sequential dependency

WHY EXPENSIVE?
Large weights
+ repeated decode
+ memory bandwidth
+ KV growth
+ long context
+ serving concurrency

SERVING
Latency ≠ throughput
Batching improves utilization
Continuous batching handles variable lengths

OPTIMIZATIONS
KV cache → recomputation
GQA/MQA → KV size
Quantization → weight bytes
Prefix cache → repeated prefill
Paged KV → memory management
FlashAttention → attention memory traffic
Speculative decoding → fewer target-model cycles
~~~
