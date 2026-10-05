# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 8

## 8.14 Why Generation Is Sequential
## 8.15 Why Inference Is Expensive

**Detailed Study Mode.** Interview-style answers will be prepared separately.

This is the final block of the inference syllabus. Earlier parts explained what happens during inference, prompt tokens, prefill, decode, KV cache, greedy decoding, temperature, top-k, top-p, beam search, repetition controls, stop tokens, and output limits.

Part 8 connects all of those ideas into a **systems-level explanation**:

1. Why can the prompt be processed in parallel while generated tokens still have to emerge one after another?
2. Why is inference expensive even though we are only doing forward passes and not training?
3. Where do compute cost, memory bandwidth, KV-cache memory, long context, batching, and output length enter the picture?
4. Which optimizations reduce cost, and which fundamental dependencies remain?

---

# 8.14 Why Generation Is Sequential

## Question 1 — What does "generation is sequential" actually mean?

For an autoregressive language model, a generated sequence is factorized as:

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid x,y_{<t})
$$

where:

- $x$ is the prompt.
- $y_t$ is the generated token at position $t$.
- $y_{<t}$ denotes all generated tokens before position $t$.
- $T$ is the final generated length.

The probability distribution for token $y_t$ depends on the **actual previously selected tokens**.

That creates the dependency:

~~~text
Choose y1
   ↓
Now y1 is known
   ↓
Compute distribution for y2
   ↓
Choose y2
   ↓
Now y1 and y2 are known
   ↓
Compute distribution for y3
   ↓
...
~~~

The key word is **actual**. Before $y_1$ is selected, we do not yet know the exact context needed to compute the target-model distribution for $y_2$.

## Question 2 — Why can't the model simply generate all output tokens in one forward pass?

Suppose the prompt is:

~~~text
The capital of France is
~~~

The model can produce:

$$
P(y_1\mid x)
$$

for the first generated token.

Assume the selected token is:

~~~text
Paris
~~~

The next distribution is then:

$$
P(y_2\mid x,\text{Paris})
$$

If the first token had instead been a different token, the next distribution would generally be different:

$$
P(y_2\mid x,\text{another token})
$$

So before the first token is selected, there is no single unique distribution for the second generated token that corresponds to the finally chosen history.

A standard autoregressive decoder therefore cannot exactly produce the entire sampled continuation in one ordinary forward pass.

## Question 3 — But during prefill, don't we process many prompt tokens in parallel?

Yes. This is one of the most important distinctions in LLM inference.

During **prefill**, all prompt tokens are already known:

$$
x_1,x_2,\ldots,x_n
$$

The Transformer can process their hidden states in parallel while a causal mask prevents position $i$ from attending to future positions $j>i$.

For one attention head:

$$
Q\in\mathbb{R}^{n\times d_k}
$$

$$
K\in\mathbb{R}^{n\times d_k}
$$

$$
V\in\mathbb{R}^{n\times d_v}
$$

and the attention-score matrix is:

$$
QK^T\in\mathbb{R}^{n\times n}
$$

All prompt positions can be represented in a batched matrix computation because their input token identities are known in advance.

**Causal dependence does not prevent parallel computation when the conditioned-on tokens are already known.**

## Question 4 — Why is decode different from prefill?

After prefill, the first output token $y_1$ is selected from the logits produced by the final prompt position.

Only after selecting $y_1$ can we perform the next one-token decode step that predicts $y_2$.

At a decode step, the latest token contributes a new Query, Key, and Value.

For one attention head, conceptually:

$$
q_t\in\mathbb{R}^{1\times d_k}
$$

$$
K_{1:t}\in\mathbb{R}^{t\times d_k}
$$

$$
V_{1:t}\in\mathbb{R}^{t\times d_v}
$$

The attention scores are:

$$
q_tK_{1:t}^{T}
\in
\mathbb{R}^{1\times t}
$$

The new hidden state generates logits for the **following** token.

This process cannot move to the next generation position until the current token has been selected.

## Question 5 — Can you show the exact timeline from prefill to multiple decode steps?

Suppose the prompt tokens are:

$$
x_1,x_2,x_3,x_4
$$

### Stage 1 — Prefill

The model processes all four known prompt tokens together.

From the final prompt hidden state, it computes:

$$
P(y_1\mid x_1,x_2,x_3,x_4)
$$

The decoding rule selects:

$$
y_1
$$

No one-token decode forward pass was needed to obtain $y_1$; its logits came from prefill.

### Stage 2 — First decode forward pass

Now $y_1$ is known.

Process $y_1$ using the cached prompt Keys and Values and the newly computed state for $y_1$.

This produces:

$$
P(y_2\mid x_1,x_2,x_3,x_4,y_1)
$$

Select $y_2$.

### Stage 3 — Second decode forward pass

Now process $y_2$ to produce:

$$
P(y_3\mid x_1,x_2,x_3,x_4,y_1,y_2)
$$

Select $y_3$.

The timeline is therefore:

~~~text
Prompt prefill
     ↓
P(y1 | prompt)
     ↓
select y1
     ↓
decode y1
     ↓
P(y2 | prompt, y1)
     ↓
select y2
     ↓
decode y2
     ↓
P(y3 | prompt, y1, y2)
     ↓
...
~~~

## Question 6 — Why doesn't the causal mask itself make prefill sequential?

The causal mask controls **which positions can attend to which other positions**. It does not require us to execute each prompt position in a separate Transformer invocation.

For prompt position $i$, attention can depend on:

$$
x_1,\ldots,x_i
$$

but not:

$$
x_{i+1},\ldots,x_n
$$

Because all prompt embeddings are already known, the GPU can compute many matrix operations for all positions simultaneously and then apply the causal mask.

The distinction is:

~~~text
Known inputs + causal dependencies
→ often parallelizable within a forward pass

Unknown future generated tokens
→ cannot be conditioned on until selected
~~~

## Question 7 — How is this different from training with teacher forcing?

During standard next-token training, the entire target sequence is already available in the training example.

Suppose the sequence is:

$$
x_1,x_2,\ldots,x_n
$$

The model can produce predictions for many next-token positions in parallel:

$$
P(x_2\mid x_1)
$$

$$
P(x_3\mid x_1,x_2)
$$

$$
\vdots
$$

$$
P(x_n\mid x_1,\ldots,x_{n-1})
$$

The causal mask prevents information leakage from future target tokens, but the ground-truth previous tokens are still known and can be supplied to the model simultaneously.

At inference time, those future ground-truth tokens do not exist. The model must generate them itself.

This is why **training can parallelize token positions far more aggressively than autoregressive generation**.

## Question 8 — Why can't we just use the most likely future token at every position in parallel?

Because the prediction for position $t+1$ depends on which token was actually selected at position $t$.

Even if we make a guess for multiple future positions, those guesses are only provisional.

For example:

~~~text
Prompt
  ↓
Guess: A → B → C
~~~

If the true target-model selection at the first position is D instead of A, then:

$$
P(y_2\mid x,D)
$$

need not resemble:

$$
P(y_2\mid x,A)
$$

and the guessed suffix B → C may no longer be valid.

This dependency is why ordinary exact autoregressive decoding advances one accepted token at a time.

## Question 9 — Does sampling make the sequential dependency even more obvious?

Yes.

Suppose at one step the model produces:

| Token | Probability |
|---|---:|
| A | 0.50 |
| B | 0.30 |
| C | 0.20 |

Sampling might select A on one run and B on another.

The next-step distributions are then different:

$$
P(y_{t+1}\mid x,y_{1:t-1},A)
$$

versus:

$$
P(y_{t+1}\mid x,y_{1:t-1},B)
$$

A stochastic choice at one step changes the context used for every subsequent step.

That means the model cannot know the exact future sampled path before those earlier random choices have been resolved.

## Question 10 — Is greedy decoding also sequential even though it is deterministic?

Yes.

Greedy decoding selects:

$$
y_t
=
\underset{v}{\operatorname{argmax}}
\;
P(v\mid x,y_{<t})
$$

Even though the choice is deterministic for fixed logits, the next distribution still depends on the selected token.

The model must first determine:

$$
y_t
$$

before it knows the exact context required to compute:

$$
P(y_{t+1}\mid x,y_{\le t})
$$

Determinism removes sampling randomness; it does **not** remove the autoregressive dependency.

## Question 11 — Is beam search sequential too?

Yes, but the unit of work is a **set of active hypotheses** rather than one sequence.

At depth $t$, beam search maintains up to $w$ partial sequences. It can batch the forward computations for those beams, producing one next-token distribution per beam.

Then it scores candidate extensions and selects the next set of beams.

Only after that pruning step can depth $t+1$ be evaluated.

So beam search has:

~~~text
Parallelism across beams at the same depth
+
Sequential dependence across generation depths
~~~

Increasing beam width increases same-step parallel work but does not remove the depth-by-depth dependency.

## Question 12 — Can multiple users be decoded in parallel even though one user's tokens are sequential?

Yes. This is a major serving optimization.

Suppose three requests are active:

~~~text
Request A: needs token 35
Request B: needs token 12
Request C: needs token 80
~~~

The server can batch the **current decode step** for all three requests:

$$
B=3
$$

and process their latest tokens together.

Each request still obeys its own sequential chain:

$$
y_t^{(r)}
\rightarrow
y_{t+1}^{(r)}
$$

but the hardware performs many requests' current steps in parallel.

This is one reason to distinguish:

- **Within-sequence parallelism:** limited during autoregressive decode.
- **Across-sequence batching:** highly useful for throughput.

## Question 13 — Does KV caching remove the sequential nature of generation?

No.

KV caching avoids recomputing Keys and Values for earlier tokens. It changes the amount of work per decode step, but not the dependency graph.

Without a KV cache, predicting token $t+1$ might repeatedly recompute the whole prefix.

With a KV cache:

~~~text
Reuse K/V for old tokens
Compute new state only for latest token
Attend to cached history
Produce next-token logits
~~~

This is much more efficient, but the next accepted token must still be known before the following decode step begins.

**KV cache removes redundant computation, not autoregressive dependency.**

## Question 14 — Does longer context make sequential generation more sequential?

The dependency chain is still one generated position after another regardless of context length.

However, longer context can make **each individual decode step more expensive**, because the new Query may attend over more cached Keys and Values.

For one head at sequence length $t$:

$$
q_tK_{1:t}^{T}
$$

requires work that grows with $t$.

So output generation remains sequential, while the per-token cost can also rise as context grows.

## Question 15 — What are TTFT and inter-token latency?

Two latency concepts help separate prefill from decode.

### Time to first token — TTFT

TTFT measures how long it takes before the first generated token becomes available.

Conceptually:

$$
T_{\mathrm{TTFT}}
\approx
T_{\mathrm{queue}}
+
T_{\mathrm{prefill}}
+
T_{\mathrm{first\ selection}}
$$

A long prompt can increase prefill time and therefore TTFT.

### Inter-token latency

After the first output token, each additional token requires another decode iteration.

If token $j$ takes time $t_j$:

$$
T_{\mathrm{decode}}
=
\sum_{j=1}^{N}t_j
$$

The delay between successive streamed output tokens is often called inter-token latency or time per output token.

**TTFT is dominated by getting to the first token; subsequent responsiveness is governed by decode-step latency.**

## Question 16 — Can we quantify the sequential critical path?

Suppose a response generates:

$$
N=100
$$

new tokens after the first token.

If each required decode iteration takes approximately:

$$
t_{\mathrm{decode}}=20\ \mathrm{ms}
$$

then a simplified lower-level decode timeline is approximately:

$$
100\times20\ \mathrm{ms}
=
2000\ \mathrm{ms}
=
2\ \mathrm{s}
$$

This assumes no overlap, no scheduling delays, and constant decode time. Real serving latency varies.

The important idea is that one request cannot simply run all 100 dependent decode positions simultaneously.

## Question 17 — Are there techniques that partially reduce sequential-generation latency?

Yes, but they do not erase the logical autoregressive dependency.

One important example is **speculative decoding**.

A smaller or faster draft model proposes multiple future tokens. The larger target model verifies several proposed positions in a more parallel fashion.

Conceptually:

~~~text
Draft model proposes:
A → B → C → D

Target model verifies the proposed block

Accepted prefix:
A → B → C

Reject D
Continue from accepted prefix
~~~

When several drafted tokens are accepted, the target model can advance multiple output positions per verification cycle.

However, the final accepted sequence must still be consistent with the target model's decoding rule. Speculation is an optimization around the dependency, not proof that autoregressive generation no longer depends on previous accepted tokens.

## Question 18 — Could a different model architecture generate all tokens in parallel?

Potentially, yes. Non-autoregressive sequence models attempt to predict multiple output positions simultaneously or iteratively refine a sequence.

But standard GPT-style decoder LLMs use autoregressive factorization:

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid x,y_{<t})
$$

The sequential behavior described in this syllabus follows from that modeling choice.

## Question 19 — What is the final mental model for sequential generation?

~~~text
PREFILL
All prompt tokens already known
→ process many positions in parallel
→ obtain first-token logits

DECODE
Select y1
→ process y1
→ obtain logits for y2
→ select y2
→ process y2
→ obtain logits for y3
→ repeat

PARALLELISM STILL EXISTS
→ across Transformer operations
→ across attention heads
→ across layers subject to dependencies
→ across requests
→ across beams at one generation depth

BUT THE ACCEPTED OUTPUT POSITIONS
remain autoregressively dependent
~~~

The core principle is:

> **The Transformer is massively parallel inside a forward pass, but autoregressive output positions form a sequential dependency chain across forward passes.**

---

# 8.15 Why Inference Is Expensive

## Question 1 — If inference uses only forward passes, why is it still expensive?

Inference avoids backpropagation and optimizer updates, so one training step is much more expensive than one inference forward pass.

But modern LLM inference is still costly because:

1. The model may contain billions of parameters.
2. The parameters must participate in repeated matrix operations.
3. Decode requires a new forward pass for each accepted output token.
4. Model weights and KV-cache data must move through the memory hierarchy.
5. Long contexts increase attention and cache costs.
6. Batching improves throughput but increases active memory demand.
7. Large models may require multiple accelerators and communication between them.
8. The vocabulary projection and token-selection pipeline also consume work.

The absence of backpropagation does not make a 7B, 70B, or larger model small.

## Question 2 — How much memory do model weights require?

A first approximation is:

$$
M_{\mathrm{weights}}
\approx
P\times b
$$

where:

- $P$ is number of parameters.
- $b$ is bytes per stored parameter.

### Example — 7B model

With 7 billion parameters in 16-bit storage:

$$
P=7\times10^9
$$

$$
b=2\ \mathrm{bytes}
$$

Therefore:

$$
M_{\mathrm{weights}}
\approx
14\times10^9\ \mathrm{bytes}
$$

or roughly 14 GB in decimal units, before additional runtime overhead.

### Example — 70B model

At 16 bits:

$$
70\times10^9\times2
=
140\times10^9\ \mathrm{bytes}
$$

or roughly 140 GB in decimal units.

At idealized 8-bit storage:

$$
70\times10^9\times1
\approx
70\ \mathrm{GB}
$$

At idealized 4-bit storage:

$$
70\times10^9\times0.5
\approx
35\ \mathrm{GB}
$$

Real quantized formats can require additional scales, metadata, padding, and kernels, so these are first-order estimates.

## Question 3 — Why do weights matter repeatedly during decode?

Every generated token must pass through the Transformer layers.

For a dense model, each decode iteration repeatedly applies operations involving matrices derived from the model's parameters:

- Attention projections.
- Attention output projection.
- Feed-forward or MLP projections.
- Normalization and other layer operations.
- Final vocabulary projection.

The model does not load its weights from disk for every token; weights remain resident in accelerator memory where possible. But data still has to move through the accelerator memory hierarchy and be consumed by compute units.

At small batch sizes, repeatedly streaming very large weight matrices can make decode **memory-bandwidth limited**.

## Question 4 — What does "memory-bandwidth limited" mean?

A processor has at least two relevant resource ceilings:

- How many arithmetic operations it can perform per second.
- How many bytes it can move from memory per second.

An operation with low **arithmetic intensity** performs relatively few calculations for each byte moved.

A simplified definition is:

$$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}
{\text{Bytes moved}}
$$

During single-token decode with a small batch, large model weights may be read to perform relatively little work per weight compared with a large matrix-matrix multiplication.

The hardware can therefore spend substantial time waiting on memory movement rather than reaching its peak arithmetic throughput.

This is why a GPU with enormous theoretical FLOPs can still have disappointing per-token latency on a large model.

## Question 5 — Is prefill expensive for the same reason as decode?

Not exactly.

### Prefill

A long prompt provides many token positions at once. Large matrix multiplications can reuse weights across many positions, giving high arithmetic intensity.

Prefill often behaves more **compute-intensive**, especially for large prompt batches and sufficiently large sequence matrices.

### Decode

A decode step may process only one new token per active sequence. At small batch sizes, weight reuse is lower and model execution is frequently **memory-bandwidth sensitive**.

This is a useful high-level distinction:

~~~text
Prefill
→ many tokens at once
→ large matrix-matrix operations
→ often more compute-oriented

Decode
→ one new token per sequence per step
→ repeated weight access
→ often more bandwidth-oriented
~~~

Real kernels, batch sizes, architectures, and hardware can shift the exact bottleneck.

## Question 6 — Can we estimate dense-model arithmetic per generated token?

A rough rule of thumb for a dense Transformer is that a forward pass requires on the order of:

$$
2P
$$

floating-point operations per token for the large parameterized matrix multiplications, where $P$ is parameter count.

Why approximately 2?

A matrix-vector or matrix-matrix multiply conceptually performs a multiply and an add per weight contribution, leading to roughly two floating-point operations.

This is only a **first-order estimate**. It ignores or simplifies:

- Attention-score operations.
- Softmax and normalization.
- Embedding and vocabulary details.
- Sparsity.
- Mixture-of-Experts routing.
- Quantized-kernel behavior.
- Hardware implementation details.

### 7B example

$$
2P
\approx
2\times7\times10^9
=
14\times10^9
$$

So approximately 14 GFLOPs of dense parameterized work per token as a rough scale estimate.

### 70B example

$$
2P
\approx
140\times10^9
$$

or roughly 140 GFLOPs per token before accounting for other work.

The important lesson is not the exact number; it is that **every output token invokes billions of parameter operations**.

## Question 7 — Why doesn't the 2P estimate fully explain long-context inference cost?

Because attention introduces sequence-length-dependent work.

With a KV cache, at decode step $t$, one attention head computes approximately:

$$
q_tK_{1:t}^{T}
$$

where:

$$
q_t\in\mathbb{R}^{1\times d_k}
$$

and:

$$
K_{1:t}\in\mathbb{R}^{t\times d_k}
$$

Computing the attention scores requires work proportional to:

$$
O(td_k)
$$

per head.

Then attention weights are applied to:

$$
V_{1:t}\in\mathbb{R}^{t\times d_v}
$$

which also scales with sequence length.

So even with KV caching, attention work and cache reads grow as the context becomes longer.

## Question 8 — What would happen without a KV cache?

Without a KV cache, every decode iteration would need to recompute representations for old tokens.

Suppose the current context length is $t$. A naïve no-cache approach could repeatedly process a sequence of length $t$ just to generate one new token.

After generating many tokens, this creates massive redundant work.

With a KV cache:

- Old Keys and Values are stored once.
- The new token's K/V are computed once.
- The new Query attends to cached K/V.

This changes decoding from repeated full-prefix recomputation to incremental processing.

**KV caching is one of the central reasons autoregressive inference is practical.**

## Question 9 — Does the KV cache make inference free after the weights are loaded?

No.

The KV cache eliminates repeated K/V computation for historical tokens, but the system still must:

1. Run the new token through all Transformer layers.
2. Read model weights.
3. Compute a new Query, Key, Value, MLP activations, and other layer outputs.
4. Read cached Keys and Values for attention.
5. Write the new Keys and Values.
6. Project the final hidden state to vocabulary logits.
7. Select the next token.

KV cache trades **extra memory** for **less recomputation**.

## Question 10 — How large can the KV cache become?

A useful first-order estimate is:

$$
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{KV}}d_h b
$$

where:

- $L$ = number of Transformer layers.
- $B$ = number of active sequence histories.
- $T$ = cached token count per sequence.
- $H_{\mathrm{KV}}$ = number of KV heads.
- $d_h$ = dimension per KV head.
- $b$ = bytes per cached value.
- Factor 2 accounts for Keys and Values.

### Numerical example

Let:

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
H_{\mathrm{KV}}=8
$$

$$
d_h=128
$$

$$
b=2
$$

Then:

$$
M_{\mathrm{KV}}
=
2\times32\times1\times8192\times8\times128\times2
$$

which equals:

$$
1{,}073{,}741{,}824\ \mathrm{bytes}
$$

or exactly:

$$
1\ \mathrm{GiB}
$$

for this illustrative configuration.

With:

$$
B=16
$$

the same simple formula becomes:

$$
16\ \mathrm{GiB}
$$

This illustrates why long contexts and large batches consume substantial memory even when model weights already fit.

## Question 11 — Why do MQA and GQA help inference?

Standard multi-head attention may use many Key/Value heads.

**Multi-Query Attention (MQA)** shares Key and Value projections across many Query heads.

**Grouped-Query Attention (GQA)** uses fewer KV heads than Query heads.

Since KV-cache memory is proportional to:

$$
H_{\mathrm{KV}}
$$

reducing the number of KV heads directly reduces cache size and the amount of K/V data that must be read during attention.

This can substantially improve serving efficiency, especially for long context and large batch sizes.

## Question 12 — How does prompt length affect inference cost?

Prompt length affects both prefill and later decode.

### Prefill

Self-attention over a prompt of length $n$ forms an attention-score structure with:

$$
n\times n
$$

position interactions per head.

The attention-score component therefore scales approximately as:

$$
O(n^2d_k)
$$

while large projection and MLP operations scale roughly linearly with $n$ for fixed hidden dimensions.

### Decode

After prefill, every new token attends over the growing cached context. Longer prompts therefore make early decode steps start with a larger $T$.

A longer prompt can increase:

- TTFT.
- Prefill compute.
- KV-cache memory.
- Per-token attention work during decode.

## Question 13 — Is attention always the dominant inference cost?

No.

For many dense LLMs, the parameterized linear and MLP layers represent enormous compute and weight-movement cost.

At relatively short contexts, these dense operations can dominate.

At very long contexts, attention over the growing KV cache can become increasingly significant because attention work and memory traffic scale with sequence length.

The dominant bottleneck depends on:

- Model architecture.
- Hidden width.
- Number of layers.
- Context length.
- Batch size.
- KV-head count.
- Precision.
- Accelerator architecture.
- Kernel implementation.

## Question 14 — What does the LM head cost?

After the final Transformer hidden state:

$$
h_t\in\mathbb{R}^{d_{\mathrm{model}}}
$$

the language-model head projects it to vocabulary logits:

$$
z_t=W_{\mathrm{LM}}h_t+b
$$

where:

$$
W_{\mathrm{LM}}
\in
\mathbb{R}^{V\times d_{\mathrm{model}}}
$$

and:

$$
z_t\in\mathbb{R}^{V}
$$

For a large vocabulary, this projection can be substantial.

For example, with:

$$
V=50{,}000
$$

and:

$$
d_{\mathrm{model}}=4096
$$

the projection contains:

$$
50{,}000\times4096
=
204{,}800{,}000
$$

weight entries.

Weight tying and optimized kernels can affect storage and execution, but the model still needs to produce vocabulary-scale scores for token selection.

## Question 15 — Is softmax over the vocabulary a major cost?

It can matter, especially for large vocabularies and large batches, but it is often not the dominant cost compared with running the full Transformer.

For logits:

$$
z\in\mathbb{R}^{V}
$$

softmax computes:

$$
P(i)
=
\frac{e^{z_i}}
{\sum_{j=1}^{V}e^{z_j}}
$$

Sampling strategies may add operations such as:

- Top-k selection.
- Top-p sorting or thresholding.
- Repetition-penalty processing.
- Constraint masking.

These are important, but the Transformer forward pass usually remains the central cost for large models.

## Question 16 — Why does output length strongly affect total inference cost?

If a request produces $N$ output tokens after the initial prefill, it requires roughly $N$ sequential generation decisions and nearly that many decode forward passes, depending on how the first token is counted.

A simplified request-time expression is:

$$
T_{\mathrm{request}}
\approx
T_{\mathrm{queue}}
+
T_{\mathrm{prefill}}
+
\sum_{j=1}^{N_{\mathrm{decode}}}
t_j
$$

Longer output means:

- More decode iterations.
- More repeated model-weight execution.
- More KV-cache growth.
- More attention over an expanding history.
- More token-selection work.

This is why a short prompt with a very long answer can still be expensive.

## Question 17 — Can you compare prompt-heavy and output-heavy requests?

Consider two simplified requests.

### Request A — Long prompt, short answer

~~~text
Prompt: 8,000 tokens
Output: 20 tokens
~~~

This request may have expensive prefill and relatively little decode.

### Request B — Short prompt, long answer

~~~text
Prompt: 200 tokens
Output: 2,000 tokens
~~~

This request may have cheap prefill but a very long sequential decode path.

Therefore:

~~~text
Long prompt
→ TTFT / prefill pressure

Long output
→ decode latency / repeated forward-pass pressure
~~~

Both can be expensive in different ways.

## Question 18 — What is latency versus throughput?

These terms describe different objectives.

### Latency

How long one request waits.

Examples:

- TTFT.
- Inter-token latency.
- End-to-end request time.

### Throughput

How much total work the server completes per unit time.

Examples:

$$
\text{tokens per second}
$$

or:

$$
\text{requests per second}
$$

A serving optimization can improve throughput while making an individual request wait slightly longer.

For example, waiting briefly to form a larger batch may improve accelerator utilization while increasing queueing latency.

## Question 19 — Why does batching improve throughput?

Suppose a model must apply the same weight matrix to several active sequences.

With batch size $B$, operations can be reorganized as larger matrix-matrix multiplications rather than many separate matrix-vector operations.

This enables:

- Better weight reuse.
- Higher arithmetic intensity.
- Better accelerator utilization.
- More total tokens processed per second.

But a larger batch also requires:

- More KV-cache memory.
- More activation memory.
- More scheduling complexity.

Batching is a throughput optimization, not removal of per-request autoregressive dependence.

## Question 20 — What is continuous batching?

Traditional static batching might wait for all requests in a batch to finish before forming the next batch.

LLM requests have variable output lengths, so this wastes capacity.

**Continuous batching** dynamically updates the active batch as requests arrive and finish.

Conceptually:

~~~text
Step 1:
A B C active

C finishes

Step 2:
A B D active

A finishes

Step 3:
E B D active
~~~

This keeps the accelerator busy and improves serving throughput.

Each individual sequence is still decoded autoregressively.

## Question 21 — Why can long output hurt batching efficiency?

Requests often finish at different times.

A request generating 20 tokens leaves the active set quickly. A request generating 2,000 tokens occupies decode and KV-cache resources for much longer.

Long outputs can:

- Hold KV-cache memory.
- Occupy active scheduling slots.
- Increase tail latency.
- Reduce the number of new requests the system can admit.

This is why output-length distributions matter for serving capacity planning.

## Question 22 — Why are large models often split across multiple GPUs?

A model may not fit on one accelerator, or one accelerator may not provide enough compute or memory bandwidth.

Common strategies include:

- **Tensor parallelism:** Split matrix operations across devices.
- **Pipeline parallelism:** Place different layers or layer groups on different devices.
- **Expert parallelism:** Distribute Mixture-of-Experts experts.
- **Data parallel serving:** Replicate model instances for independent request groups.

Multi-device inference introduces communication overhead.

The system may need to exchange partial activations or reductions every layer or group of layers, so network interconnect bandwidth and latency become part of inference performance.

## Question 23 — Why can communication become a bottleneck?

Suppose a tensor-parallel layer splits a matrix operation across multiple GPUs.

Each GPU computes a partial result, and the partial outputs may need a collective operation such as an all-reduce or all-gather.

The decode path is latency-sensitive. Even small communication delays can be repeated across many layers and many generated tokens.

Therefore, inference performance depends not only on GPU compute but also on:

- GPU-to-GPU interconnect.
- Topology.
- Collective communication implementation.
- Parallelism strategy.

## Question 24 — How does quantization reduce inference cost?

Quantization stores weights, and sometimes activations or KV cache, using fewer bits.

If weights move from 16-bit to 8-bit storage, idealized weight memory is approximately halved.

From 16-bit to 4-bit, idealized memory is approximately quartered.

Benefits can include:

- Smaller model memory footprint.
- Lower memory bandwidth demand.
- Larger possible batch size.
- Potentially higher throughput.

But quantization can also introduce:

- Dequantization overhead.
- Kernel constraints.
- Accuracy degradation.
- Hardware-specific performance behavior.

Smaller bit width does not automatically guarantee proportional speedup.

## Question 25 — How does prefix caching help?

Many requests may share an identical prefix, such as:

- The same system prompt.
- The same large document.
- The same application instructions.

A serving system can cache the corresponding prefill/KV state and reuse it for later requests with the same tokenized prefix.

This can reduce repeated prefill work and improve TTFT.

Prefix caching does not remove decode costs after requests diverge.

## Question 26 — What is paged KV-cache management?

Traditional KV allocation can waste memory because requests have different sequence lengths and finish at different times.

Paged KV-cache systems divide cache memory into manageable blocks or pages and map logical sequence positions onto physical memory blocks.

Benefits can include:

- Reduced fragmentation.
- More flexible allocation.
- Better support for continuous batching.
- Easier memory sharing or reuse in some designs.

Paged cache management optimizes **where KV data lives**, not the mathematical autoregressive dependency.

## Question 27 — How does FlashAttention help?

Standard attention can generate large intermediate matrices and move substantial data between high-bandwidth memory and on-chip memory.

FlashAttention-style algorithms reorganize attention computation to reduce memory traffic and avoid materializing some large intermediates.

This can improve speed and memory efficiency, especially during prefill and long-sequence attention.

It does not change the attention formula:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

It changes **how the computation is executed**.

## Question 28 — Why is long-context inference especially challenging?

Long context affects several resources simultaneously:

1. Prefill attention has a quadratic position-interaction component.
2. The KV cache grows roughly linearly with cached token count.
3. Decode attention reads a larger historical cache.
4. More cache memory reduces how many requests can fit concurrently.
5. Data movement can become increasingly important.

Long context is therefore not just a token-count issue; it is a compute, bandwidth, and memory-capacity problem.

## Question 29 — What does "prefill is compute-bound and decode is memory-bound" really mean?

It is a useful heuristic, not an absolute law.

### Prefill

Many prompt tokens are processed together. Large matrix-matrix multiplications can reuse weights efficiently and provide high arithmetic intensity.

### Decode

At low batch size, each step processes relatively few new token vectors while touching large model-weight matrices. Weight movement can dominate.

As batch size grows, decode can reuse weights across more active sequences and become more compute-efficient.

So the bottleneck depends on:

$$
\text{model}
+
\text{batch size}
+
\text{sequence length}
+
\text{hardware}
+
\text{kernels}
$$

## Question 30 — Why can a 70B model be slower than a 7B model even if both fit in memory?

A 70B dense model has roughly ten times as many parameters as a 7B model.

That generally means:

- More weight memory.
- More arithmetic per forward pass.
- More data movement.
- Greater need for multi-GPU execution.
- More communication if sharded.

Even if enough memory exists, repeatedly executing a much larger model for every output token increases latency and reduces the number of concurrent requests a fixed hardware pool can serve.

## Question 31 — Why isn't peak GPU FLOPs enough to predict LLM inference speed?

Peak FLOPs assume the hardware can keep arithmetic units fully utilized.

LLM inference may instead be limited by:

- Weight memory bandwidth.
- KV-cache bandwidth.
- Communication.
- Small-batch inefficiency.
- Kernel-launch overhead.
- Scheduling.
- Vocabulary operations.
- Memory fragmentation.

A complete performance analysis must consider the whole execution pipeline.

This is why **tokens per second cannot be predicted from theoretical FLOPs alone**.

## Question 32 — Can we build a simple request-cost mental model?

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

with:

$$
T_{\mathrm{decode}}
=
\sum_{j=1}^{N_{\mathrm{decode}}}t_j
$$

Memory usage can be thought of as:

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

This is not a precise capacity-planning equation, but it organizes the major components.

## Question 33 — What is a worked end-to-end example?

Consider a hypothetical model with:

- 7 billion parameters.
- 16-bit weights.
- 4,000 prompt tokens.
- 500 output tokens.
- KV cache enabled.

### Weight memory

Approximate weight memory:

$$
7\times10^9\times2
=
14\times10^9\ \mathrm{bytes}
$$

or about 14 GB in decimal units, before runtime overhead.

### Prefill

All 4,000 prompt positions are known and can be processed in parallel subject to causal masking.

The system creates KV state for those positions and obtains the first output-token logits.

### Decode

After first-token selection, each additional accepted output token requires another dependent decode step.

For 500 generated tokens, there are approximately 499 subsequent one-token decode passes after the first token came from prefill, assuming no speculative multi-token acceptance.

### Growing context

Near the end of generation, the cached sequence length is around:

$$
4000+500=4500
$$

tokens, ignoring extra special tokens.

Later output tokens therefore attend over more historical KV entries than early output tokens.

### Overall

The request cost is a combination of:

- One large prompt prefill.
- Hundreds of sequential decode steps.
- Repeated model-weight execution.
- Growing KV-cache storage and reads.
- Token-selection operations.

This example shows why both prompt length and output length matter, but in different ways.

## Question 34 — Which optimizations attack which bottlenecks?

| Optimization | Mainly helps |
|---|---|
| KV cache | Avoid recomputing old K/V during decode |
| GQA / MQA | Reduce KV-cache size and KV bandwidth |
| Quantization | Reduce weight memory and bandwidth |
| Continuous batching | Improve serving throughput |
| Prefix caching | Avoid repeated prefill for shared prefixes |
| FlashAttention | Reduce attention memory traffic |
| Paged KV cache | Improve cache-memory management |
| Tensor parallelism | Fit and compute larger models across devices |
| Speculative decoding | Reduce target-model decode cycles per accepted token |
| Shorter outputs | Reduce sequential decode steps |
| Shorter prompts | Reduce prefill and initial KV usage |

No single optimization eliminates every cost.

## Question 35 — What is the final answer to "Why is LLM inference expensive?"

Because a large model must repeatedly execute billions of parameter operations while moving large quantities of weight and KV-cache data, and autoregressive generation requires those forward passes to occur along a sequential output-token dependency chain.

The cost grows with:

- Model size.
- Prompt length.
- Output length.
- Batch size.
- Context length.
- KV-cache size.
- Vocabulary size.
- Beam width or number of active hypotheses.
- Precision.
- Hardware and communication topology.

The most important systems insight is:

> **Inference is not expensive for one single reason. It is the interaction of large model weights, repeated sequential decoding, memory bandwidth, KV-cache growth, attention over context, and serving-scale concurrency.**

---

# Part 8 — Prefill vs Decode Summary

| Property | Prefill | Decode |
|---|---|---|
| Input positions known? | Yes | Only current history is known |
| Tokens processed per sequence per forward pass | Many prompt positions | Usually one new position |
| Parallelism across positions | High | Limited by autoregressive dependency |
| Main user-visible metric | TTFT | Inter-token latency |
| Attention pattern | Full causal prompt attention | New Query attends to cached history |
| KV behavior | Build prompt KV cache | Append one position at a time |
| Typical bottleneck heuristic | Often compute-oriented | Often memory-bandwidth-oriented at small batch |
| Effect of long prompt | Strong | Raises starting cache length |
| Effect of long output | Indirect | Many sequential steps |

# Part 8 — Key Equations

**Autoregressive factorization:**

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid x,y_{<t})
$$

**Decode attention score shape:**

$$
q_tK_{1:t}^{T}
\in
\mathbb{R}^{1\times t}
$$

**Approximate model-weight memory:**

$$
M_{\mathrm{weights}}
\approx
P\times b
$$

**Simplified dense-model parameterized compute per token:**

$$
\mathrm{FLOPs}_{\mathrm{token}}
\sim
2P
$$

**KV-cache memory:**

$$
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{KV}}d_h b
$$

**Decode latency:**

$$
T_{\mathrm{decode}}
=
\sum_{j=1}^{N_{\mathrm{decode}}}t_j
$$

**Total request latency:**

$$
T_{\mathrm{request}}
\approx
T_{\mathrm{queue}}
+
T_{\mathrm{prefill}}
+
T_{\mathrm{decode}}
+
T_{\mathrm{other}}
$$

**Attention:**

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

# Final Mental Model for the Entire Inference Pipeline

~~~text
PROMPT
  ↓
Tokenization
  ↓
PREFILL
- all prompt tokens known
- causal attention
- prompt positions processed in parallel
- KV cache created
- first output-token logits produced
  ↓
TOKEN SELECTION
- greedy / sampling / beam search
- penalties and filters
  ↓
FIRST OUTPUT TOKEN
  ↓
DECODE LOOP
- process latest accepted token
- reuse cached K/V
- produce next-token logits
- select next token
- append its K/V
- check EOS / stop sequence / length limit
  ↓
REPEAT SEQUENTIALLY
  ↓
FINISH
~~~

# Final Synthesis — Why Generation Is Sequential and Expensive

Autoregressive generation is sequential because token $y_{t+1}$ is conditioned on the **actual accepted history** through $y_t$. Prompt processing can be parallel because the prompt tokens are already known, but generated future tokens are not.

Inference is expensive because each accepted output token requires another pass through a large model, repeatedly using huge parameter matrices and growing KV state. At serving scale, the system must balance latency, throughput, batch size, model size, context length, memory bandwidth, and accelerator memory.

That gives us the final connection:

~~~text
Autoregressive dependence
        ↓
Sequential decode iterations
        ↓
Repeated execution of a large model
        ↓
Weight + KV memory movement
        ↓
Latency, memory and throughput cost
~~~

This completes **Part 8 — Inference and Text Generation (8.1–8.15)**.
