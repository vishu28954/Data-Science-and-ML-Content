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

### Story Bridge 1 — Parallel Inside a Step, Sequential Across Output Positions

Transformers are famous for parallel matrix computation, so calling generation "sequential" can sound contradictory. The key is to separate **parallel work inside one forward pass** from the order in which **accepted output tokens** become available. Before going further, we need to identify exactly which part of generation forms the sequential chain.

## Question 1 — If Transformers are highly parallel, what exactly is sequential during generation?

For an autoregressive language model, a generated sequence is factorized as:

$$
P(y_{1:T} \vert x) = \prod_{t=1}^{T} P(y_t \vert x, y_{\lt t})
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

### Story Bridge 2 — The Next Token Changes the Context for the Token After It

Once we know that accepted output positions are sequential, the natural challenge is: why not compute all future positions together anyway? The obstacle is that the distribution for a later token depends on the **actual token selected earlier**. So we now need to connect sequential generation to conditional probability and the chain rule.

## Question 2 — If the model can compute logits in one pass, why can't it generate the whole continuation at once?

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

### A simple probability-chain-rule example

The same idea can be understood using the probability chain rule:

$$
P(y_1,y_2\mid x) = P(y_1\mid x)\,P(y_2\mid x,y_1)
$$

Suppose:

$$
x=\text{"The capital of France is"}
$$

$$
y_1=\text{"Paris"}
$$

and, for simplicity, let:

$$
y_2=\text{"."}
$$

Imagine an illustrative collection of 1,000 text situations. The prompt $x$ appears 100 times, and in 80 of those 100 cases the next token is $y_1=\text{"Paris"}$.

Then:

$$
P(x)=\frac{100}{1000}=0.1
$$

and:

$$
P(y_1,x)=\frac{80}{1000}=0.08
$$

Therefore:

$$
P(y_1\mid x) = \frac{P(y_1,x)}{P(x)} = \frac{0.08}{0.1} = \frac{80}{100} = 0.8
$$

Physically, once we know that the prompt $x$ occurred, we restrict attention from all 1,000 situations to the 100 situations containing that prompt. In 80 of those 100 cases, the next token is "Paris".

Now suppose that among those 80 cases containing both $x$ and $y_1=\text{"Paris"}$, 72 are followed by $y_2=\text{"."}$.

Then:

$$
P(y_2\mid x,y_1) = \frac{72}{80} = 0.9
$$

Equivalently:

$$
P(y_2\mid x,y_1) = \frac{P(x,y_1,y_2)}{P(x,y_1)} = \frac{72/1000}{80/1000} = 0.9
$$

Finally, by the chain rule:

$$
\begin{aligned}
P(y_1, y_2 \vert x) &= P(y_1 \vert x) \, P(y_2 \vert x, y_1) \\
&= 0.8 \times 0.9 = 0.72
\end{aligned}
$$


So, in this illustrative counting example, given the prompt $x$, the probability of the two-token continuation "Paris." is 0.72.

The important sequential point is that the second factor is:

$$
P(y_2\mid x,y_1)
$$

not merely $P(y_2\mid x)$. The probability distribution for the second generated token is conditioned on whichever first token was actually selected. If $y_1$ changes, the conditional distribution for $y_2$ can also change.

So before the first token is selected, there is no single unique distribution for the second generated token that corresponds to the finally chosen history.

A standard autoregressive decoder therefore cannot exactly produce the entire sampled continuation in one ordinary forward pass.

### Story Bridge 3 — Known History Changes the Parallelism Story

The chain-rule argument seems to suggest that every position should be processed one after another. But the prompt is different from future output: all prompt-token identities are already known before the forward pass begins. That creates an important exception we must understand—**prefill can exploit positional parallelism even under causal attention**.

## Question 3 — If generated tokens depend on earlier tokens, how can prefill process the whole prompt in parallel?

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

### Story Bridge 4 — The Boundary Between Known Tokens and Unknown Tokens

Prefill works in parallel because the entire prompt is already available. After the prompt ends, that advantage disappears: the next generated token does not exist until the current prediction is resolved. This boundary between **known prompt tokens** and **unknown future tokens** is exactly what makes decode behave differently.

## Question 4 — Once the first output token is selected, why does decode lose prefill's positional parallelism?

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

### Story Bridge 5 — Follow One Request Through the Boundary

We now understand the difference conceptually, but it is easy to lose track of which forward pass produces which token. The cleanest way to fix that is to follow one request from prompt prefill, to first-token selection, to successive one-token decode passes.

## Question 5 — What does the exact prefill-to-decode timeline look like token by token?

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

### Story Bridge 6 — Dependency Rules Do Not Always Imply Serial Execution

The timeline shows that prefill handles many positions together even though later positions are forbidden from looking into the future. That means **information dependency** and **execution order** are not the same thing. The next step is to understand why a causal mask restricts visibility without requiring one Transformer invocation per prompt token.

## Question 6 — If attention is causal, why doesn't the causal mask force prefill to run sequentially?

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

### Story Bridge 7 — Training Has Something Inference Does Not: Future Ground-Truth Tokens

Causal masking alone clearly does not prevent parallel position-wise computation. Training provides an even stronger example: many next-token predictions are computed together. The missing piece is that training already knows the ground-truth sequence, while inference must create the future tokens itself.

## Question 7 — Why can training predict many next tokens in parallel when inference cannot?

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

### Story Bridge 8 — What If We Guess the Future Instead of Waiting for It?

Teacher forcing works because the previous tokens are known. At inference time they are not, so a tempting shortcut is to predict several future positions using provisional guesses. But if the first guess changes, every later conditional distribution can change too. We need to see why a guessed suffix is not automatically the target model's true continuation.

## Question 8 — Could we guess the most likely future tokens in parallel and avoid sequential decode?

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

### Story Bridge 9 — A Random Choice Can Redirect the Entire Future

Parallel future guesses are fragile even under deterministic choices. Sampling makes the dependency easier to see because two runs can deliberately choose different valid tokens from the same distribution. Once that happens, the model is conditioning on two different histories, so the futures can diverge immediately.

## Question 9 — What changes when the current token is sampled rather than predetermined?

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

### Story Bridge 10 — Randomness Is Not the Source of Sequentiality

Sampling shows how one token choice redirects later predictions, but sequential dependence is not caused by randomness. Even when the next token is fixed by an argmax rule, the model still must know that selected token before constructing the exact context for the following position.

## Question 10 — If greedy decoding is deterministic, why doesn't determinism remove the sequential dependency?

Yes.

Greedy decoding selects:

$$
y_t = \arg\max_v P(v \vert x, y_{\lt t})
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

### Story Bridge 11 — More Hypotheses Create Width, Not Future Depth

Greedy decoding follows one deterministic path. Beam search keeps several paths alive, so it introduces much more same-step parallel work. But parallelizing **across hypotheses at one depth** is different from skipping the dependency between depth $t$ and depth $t+1$.

## Question 11 — Can beam search parallelize away the token-by-token dependency?

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

### Story Bridge 12 — Sequential Per Request Does Not Mean the GPU Must Be Idle

Beam search reveals one source of parallelism across hypotheses. Production serving provides another: many independent users can each be at their own current decode step. Their histories remain sequential internally, but the server can batch those current steps together.

## Question 12 — If one sequence is sequential, where can an inference server still find useful parallelism?

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

### Story Bridge 13 — Faster Reuse Is Not the Same as Breaking the Dependency

Across-request batching improves hardware utilization, while KV caching improves work inside each request by reusing old Keys and Values. That can dramatically reduce redundant computation. But an optimization can reduce the **cost of each step** without changing the fact that the next accepted token depends on the previous one.

## Question 13 — If KV caching makes decode much faster, does it remove the autoregressive dependency?

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

### Story Bridge 14 — The Chain Stays Sequential, but Each Link Can Become More Expensive

KV caching preserves the same token-by-token dependency while making it practical. As the cached history grows, however, the new query must attend over more Keys and Values. So longer context does not create "more sequentiality"; it changes the amount of work carried by each sequential step.

## Question 14 — If the dependency chain is unchanged, what does a longer context actually change during decode?

The dependency chain is still one generated position after another regardless of context length.

However, longer context can make **each individual decode step more expensive**, because the new Query may attend over more cached Keys and Values.

For one head at sequence length $t$:

$$
q_tK_{1:t}^{T}
$$

requires work that grows with $t$.

So output generation remains sequential, while the per-token cost can also rise as context grows.

### Story Bridge 15 — Turn the Computation Story Into User-Visible Time

We now have two distinct phases with different behavior: a prompt-heavy prefill and a token-by-token decode loop. Users experience those phases differently. That motivates two latency measurements—time to first token and the delay between later output tokens.

## Question 15 — How do prefill and sequential decode show up in the latency a user actually experiences?

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
T_{\mathrm{decode}} = \sum_{j=1}^{N}t_j
$$

The delay between successive streamed output tokens is often called inter-token latency or time per output token.

**TTFT is dominated by getting to the first token; subsequent responsiveness is governed by decode-step latency.**

### Story Bridge 16 — Once Latency Has a Name, We Can Put Numbers on It

TTFT separates the cost of reaching the first token from the repeated cost of producing later tokens. If each decode iteration takes some amount of time, then a long answer accumulates those dependent steps along a critical path. A simple numerical estimate makes that cost concrete.

## Question 16 — Can we estimate how much latency the sequential decode chain adds?

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
100\times20\ \mathrm{ms} = 2000\ \mathrm{ms} = 2\ \mathrm{s}
$$

This assumes no overlap, no scheduling delays, and constant decode time. Real serving latency varies.

The important idea is that one request cannot simply run all 100 dependent decode positions simultaneously.

### Story Bridge 17 — The Dependency Is Real, but We Can Sometimes Advance More Than One Token per Target Cycle

The critical-path calculation makes the limitation uncomfortable: many accepted tokens can mean many target-model decode cycles. That naturally leads to techniques such as speculative decoding, which try to preserve the target model's autoregressive behavior while reducing how often the expensive target must advance one step at a time.

## Question 17 — Can we reduce the number of target-model sequential steps without changing the final autoregressive rule?

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

### Story Bridge 18 — Separate a GPT Design Choice From a Universal Law

Speculative decoding works around the autoregressive dependency rather than eliminating it. That raises a broader architectural question: is token-by-token generation unavoidable for every sequence model, or is it a consequence of the particular probability factorization used by GPT-style decoders?

## Question 18 — Is sequential generation fundamental to all sequence models, or mainly to autoregressive architectures?

Potentially, yes. Non-autoregressive sequence models attempt to predict multiple output positions simultaneously or iteratively refine a sequence.

But standard GPT-style decoder LLMs use autoregressive factorization:

$$
P(y_{1:T} \vert x) = \prod_{t=1}^{T} P(y_t \vert x, y_{\lt t})
$$


The sequential behavior described in this syllabus follows from that modeling choice.

### Story Bridge 19 — Reconstruct the Whole Story Before Moving to Cost

We have followed the argument from chain-rule dependence to prefill, decode, masking, teacher forcing, decoding strategies, batching, KV caching, latency, and speculative decoding. Before asking why inference is expensive, we should compress these ideas into one picture that can be reconstructed from memory.

## Question 19 — How can we compress the entire sequential-generation story into one mental model?

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

### Story Bridge 20 — Sequential Decode Tells Us How Often the Model Runs; Now Ask How Expensive Each Run Is

Section 8.14 established that long outputs can require many dependent forward passes. Training is obviously expensive because it also performs backward passes and optimizer work, but inference removes those pieces. The next question is why repeated **forward-only** execution of a large model is still a major systems cost.

## Question 1 — If inference avoids backpropagation, why can serving an LLM still be expensive?

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

### Story Bridge 21 — Start With the Object That Must Be Present for Every Forward Pass

The first source of cost is simply the size of the model itself. Billions of parameters must live somewhere before the first token can be processed. So we begin with the simplest capacity calculation: parameter count multiplied by bytes per parameter.

## Question 2 — Before doing any computation, how much memory do the model weights alone require?

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
M_{\mathrm{weights}} \approx 14\times10^9\ \mathrm{bytes}
$$

or roughly 14 GB in decimal units, before additional runtime overhead.

### Example — 70B model

At 16 bits:

$$
70\times10^9\times2 = 140\times10^9\ \mathrm{bytes}
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

### Story Bridge 22 — Fitting in Memory Does Not Mean the Data Stops Moving

Knowing the weight footprint tells us whether the model fits, but not how quickly it runs. During every forward pass those matrices must be consumed by the accelerator's compute units through its memory hierarchy. So resident weights remain part of the per-token cost.

## Question 3 — If the weights already fit in GPU memory, why do they still cost us on every decode step?

Every generated token must pass through the Transformer layers.

For a dense model, each decode iteration repeatedly applies operations involving matrices derived from the model's parameters:

- Attention projections.
- Attention output projection.
- Feed-forward or MLP projections.
- Normalization and other layer operations.
- Final vocabulary projection.

The model does not load its weights from disk for every token; weights remain resident in accelerator memory where possible. But data still has to move through the accelerator memory hierarchy and be consumed by compute units.

At small batch sizes, repeatedly streaming very large weight matrices can make decode **memory-bandwidth limited**.

### Story Bridge 23 — Compute Is Only Fast When the Hardware Can Feed It

Repeatedly using large weight matrices exposes a hardware limit that raw FLOPs do not capture: the rate at which bytes can be delivered to the compute units. This is where arithmetic intensity and memory bandwidth become central to understanding decode performance.

## Question 4 — Why can moving model data be the bottleneck even on a GPU with enormous FLOPs?

A processor has at least two relevant resource ceilings:

- How many arithmetic operations it can perform per second.
- How many bytes it can move from memory per second.

An operation with low **arithmetic intensity** performs relatively few calculations for each byte moved.

A simplified definition is:

$$
\text{Arithmetic Intensity} =
\frac{\text{FLOPs}}
{\text{Bytes moved}}
$$

During single-token decode with a small batch, large model weights may be read to perform relatively little work per weight compared with a large matrix-matrix multiplication.

The hardware can therefore spend substantial time waiting on memory movement rather than reaching its peak arithmetic throughput.

This is why a GPU with enormous theoretical FLOPs can still have disappointing per-token latency on a large model.

### Story Bridge 24 — The Same Model Can Stress Hardware Differently in Different Phases

Decode can be bandwidth-sensitive because a small amount of new-token work repeatedly touches large weight matrices. Prefill, however, processes many known token positions together and can reuse those weights across larger matrix operations. We therefore need to separate the typical bottlenecks of the two phases.

## Question 5 — Do prefill and decode hit the same hardware bottleneck?

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

### Story Bridge 25 — After Memory Movement, Quantify the Arithmetic Scale

The compute-vs-bandwidth distinction tells us *what may bottleneck*, but we still need a sense of scale. A useful first-order estimate relates dense-model work to the number of parameters, giving us a rough FLOPs-per-token rule.

## Question 6 — Can we build a rough compute estimate for one token of dense-model inference?

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
2P \approx 2\times7\times10^9 = 14\times10^9
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

### Story Bridge 26 — Parameter Work Is Not the Only Work

The $2P$ rule captures the large parameterized matrix multiplications, but decode also performs attention over the growing history. As context length increases, that sequence-dependent work becomes increasingly visible. So the next step is to add the cost that parameter count alone misses.

## Question 7 — Why does the rough $2P$ compute rule become incomplete as context grows?

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

### Story Bridge 27 — Long History Would Be Far Worse If We Recomputed It Every Time

Attention already becomes more expensive as the history grows. Without reuse, each new output token could also force the model to recompute representations for the entire old prefix. This is the exact redundancy that KV caching is designed to remove.

## Question 8 — If attention needs the whole history, what would decoding cost without a KV cache?

Without a KV cache, every decode iteration would need to recompute representations for old tokens.

Suppose the current context length is $t$. A naïve no-cache approach could repeatedly process a sequence of length $t$ just to generate one new token.

After generating many tokens, this creates massive redundant work.

With a KV cache:

- Old Keys and Values are stored once.
- The new token's K/V are computed once.
- The new Query attends to cached K/V.

This changes decoding from repeated full-prefix recomputation to incremental processing.

**KV caching is one of the central reasons autoregressive inference is practical.**

### Story Bridge 28 — KV Cache Removes Redundant Work, Not the Forward Pass

KV caching transforms decoding from repeated full-prefix recomputation into incremental processing, which is a huge win. But the new token still has to pass through every Transformer layer, attend to history, update the cache, and produce vocabulary logits. We now need to identify what remains expensive.

## Question 9 — Once historical K/V are cached, what expensive work still remains?

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

### Story Bridge 29 — The Optimization Creates a New Resource Cost

KV caching trades computation for stored state. Every active sequence keeps Keys and Values across layers and positions, so longer contexts and larger batches can consume substantial accelerator memory. This tradeoff is important enough to quantify directly.

## Question 10 — KV caching saves compute, but how much memory can the cache consume?

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
M_{\mathrm{KV}} = 2\times32\times1\times8192\times8\times128\times2
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

### Story Bridge 30 — Once KV Memory Is the Problem, Reduce the Number of Stored KV Heads

The KV-memory formula shows exactly which dimensions drive cache size. One particularly useful lever is the number of Key/Value heads. MQA and GQA exploit that lever by sharing or grouping K/V projections across query heads.

## Question 11 — If KV-cache memory is large, how do MQA and GQA reduce it?

Standard multi-head attention may use many Key/Value heads.

**Multi-Query Attention (MQA)** shares Key and Value projections across many Query heads.

**Grouped-Query Attention (GQA)** uses fewer KV heads than Query heads.

Since KV-cache memory is proportional to:

$$
H_{\mathrm{KV}}
$$

reducing the number of KV heads directly reduces cache size and the amount of K/V data that must be read during attention.

This can substantially improve serving efficiency, especially for long context and large batch sizes.

### Story Bridge 31 — Context Length Starts Costing Us Before the First Output Token

Reducing KV heads helps with stored history, but the number of prompt positions still matters. A longer prompt increases the work required to build the initial states and also makes the very first decode step start with a larger cache.

## Question 12 — How does a longer prompt change both prefill cost and the decode steps that follow?

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

### Story Bridge 32 — Do Not Turn One Important Bottleneck Into a Universal Rule

Long prompts and long histories make attention increasingly important, but dense projections and MLPs also process enormous parameter matrices. Which component dominates depends on architecture, sequence length, batch size, precision, and hardware.

## Question 13 — With long-context attention growing, does attention always dominate inference cost?

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

### Story Bridge 33 — The Forward Pass Is Not Finished Until We Produce Vocabulary Scores

Even if we understand the Transformer blocks, generation still needs a score for every candidate token. The final hidden state must be projected into vocabulary-sized logits, which can itself involve a large matrix.

## Question 14 — After the Transformer layers finish, what does projecting to the vocabulary cost?

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
50{,}000\times4096 = 204{,}800{,}000
$$

weight entries.

Weight tying and optimized kernels can affect storage and execution, but the model still needs to produce vocabulary-scale scores for token selection.

### Story Bridge 34 — Logits Still Need to Become a Token

The LM head gives us vocabulary-scale logits, but the runtime may still apply softmax, top-k, top-p, penalties, masks, and sampling. Those operations matter, yet we need to place them in proportion to the cost of the full Transformer forward pass.

## Question 15 — Once vocabulary logits exist, do softmax and token-selection operations dominate the cost?

It can matter, especially for large vocabularies and large batches, but it is often not the dominant cost compared with running the full Transformer.

For logits:

$$
z\in\mathbb{R}^{V}
$$

softmax computes:

$$
P(i) = \frac{e^{z_i}} {\sum_{j=1}^{V}e^{z_j}}
$$

Sampling strategies may add operations such as:

- Top-k selection.
- Top-p sorting or thresholding.
- Repetition-penalty processing.
- Constraint masking.

These are important, but the Transformer forward pass usually remains the central cost for large models.

### Story Bridge 35 — Per-Token Cost Becomes Request Cost Through Repetition

We now understand many costs of one forward step: weights, attention, KV traffic, vocabulary projection, and token selection. Autoregressive generation repeats that machinery for each accepted output token. Output length therefore turns per-token expense into end-to-end request expense.

## Question 16 — If one token is expensive, why does a long output multiply that cost so strongly?

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

### Story Bridge 36 — Not All Long Requests Stress the Same Phase

Output length mainly stretches the sequential decode path, while prompt length mainly increases prefill and the starting context size. Two requests with similar total token counts can therefore have very different latency profiles and hardware pressure.

## Question 17 — How do long-prompt and long-output requests become expensive in different ways?

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


### Story Bridge 37 — Move From One Request to a Production Server

So far we have mostly followed one request. A real inference service has many users competing for the same accelerator, and the best strategy for total hardware utilization may not minimize each individual's wait. That creates the fundamental serving distinction between **latency** and **throughput**.

## Question 18 — Once many users share the server, how do latency and throughput become different goals?

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

### Story Bridge 38 — Many Sequential Requests Can Still Share the Same Weight Execution

Once throughput matters, serving requests one at a time can waste accelerator capacity. By grouping current work from several requests, the system can reuse weights more effectively and turn many small operations into larger, more efficient matrix computations.

## Question 19 — If requests are independent, why does batching them improve total throughput?

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

### Story Bridge 39 — Static Batches Waste Capacity When Sequence Lengths Differ

Batching improves utilization, but LLM requests rarely produce exactly the same number of output tokens. If the server waits for every sequence in a static batch to finish, completed slots sit idle. Continuous batching solves that scheduling problem by changing the active set over time.

## Question 20 — What happens when requests in the same batch finish at different times?

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

### Story Bridge 40 — Variable Lifetimes Turn Output Length Into a Shared Resource Problem

Continuous batching can replace finished requests, but a long-running request still keeps its KV state and decode slot active for many more steps. This means output length affects not only one user's latency, but also memory pressure and admission capacity for everyone else.

## Question 21 — Why can one very long response keep consuming shared serving resources after shorter requests finish?

Requests often finish at different times.

A request generating 20 tokens leaves the active set quickly. A request generating 2,000 tokens occupies decode and KV-cache resources for much longer.

Long outputs can:

- Hold KV-cache memory.
- Occupy active scheduling slots.
- Increase tail latency.
- Reduce the number of new requests the system can admit.

This is why output-length distributions matter for serving capacity planning.


### Story Bridge 41 — Batching Improves Utilization Until Capacity Becomes the Next Limit

As concurrency grows, KV memory accumulates; as model size grows, the weights themselves may exceed one accelerator's capacity. At that point the serving problem expands from using one GPU efficiently to distributing model execution across multiple devices.

## Question 22 — What do we do when one GPU is no longer enough for the model or the serving load?

A model may not fit on one accelerator, or one accelerator may not provide enough compute or memory bandwidth.

Common strategies include:

- **Tensor parallelism:** Split matrix operations across devices.
- **Pipeline parallelism:** Place different layers or layer groups on different devices.
- **Expert parallelism:** Distribute Mixture-of-Experts experts.
- **Data parallel serving:** Replicate model instances for independent request groups.

Multi-device inference introduces communication overhead.

The system may need to exchange partial activations or reductions every layer or group of layers, so network interconnect bandwidth and latency become part of inference performance.

### Story Bridge 42 — More Devices Add Capacity but Also Add Synchronization

Multi-GPU execution gives us more memory and compute, but model shards must exchange activations or partial results. Decode is latency-sensitive, so communication repeated across many layers and output tokens can become as important as arithmetic itself.

## Question 23 — Once the model is split across GPUs, why can communication become the new bottleneck?

Suppose a tensor-parallel layer splits a matrix operation across multiple GPUs.

Each GPU computes a partial result, and the partial outputs may need a collective operation such as an all-reduce or all-gather.

The decode path is latency-sensitive. Even small communication delays can be repeated across many layers and many generated tokens.

Therefore, inference performance depends not only on GPU compute but also on:

- GPU-to-GPU interconnect.
- Topology.
- Collective communication implementation.
- Parallelism strategy.


### Story Bridge 43 — Once Data Movement Is Expensive, Reduce the Number of Bits

The communication and bandwidth story reveals a broader principle: inference cost is often about **bytes moved**, not only operations performed. Quantization attacks that problem directly by representing weights—and sometimes other tensors—with fewer bits.

## Question 24 — If moving and storing weights is expensive, how does quantization attack that bottleneck?

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

### Story Bridge 44 — Some Work Is Expensive Only Because We Repeat It

Quantization reduces the cost of moving model data, but another waste source is repeated computation. Applications often send identical system prompts, documents, or instruction prefixes. If those tokenized prefixes are identical, their prefill state can potentially be reused.

## Question 25 — If many requests repeat the same prompt prefix, why recompute that prefill every time?

Many requests may share an identical prefix, such as:

- The same system prompt.
- The same large document.
- The same application instructions.

A serving system can cache the corresponding prefill/KV state and reuse it for later requests with the same tokenized prefix.

This can reduce repeated prefill work and improve TTFT.

Prefix caching does not remove decode costs after requests diverge.

### Story Bridge 45 — Saving KV Is Not Enough; We Must Allocate It Well

Prefix caching reuses useful state, while continuous batching constantly adds and removes requests of different lengths. That makes KV memory dynamic and prone to fragmentation. Paged KV management addresses **where and how** the cached blocks are allocated.

## Question 26 — If many variable-length requests keep KV state alive, how can the server manage that memory efficiently?

Traditional KV allocation can waste memory because requests have different sequence lengths and finish at different times.

Paged KV-cache systems divide cache memory into manageable blocks or pages and map logical sequence positions onto physical memory blocks.

Benefits can include:

- Reduced fragmentation.
- More flexible allocation.
- Better support for continuous batching.
- Easier memory sharing or reuse in some designs.

Paged cache management optimizes **where KV data lives**, not the mathematical autoregressive dependency.

### Story Bridge 46 — Optimize Attention by Changing the Execution, Not the Mathematics

Paged KV management improves storage layout, but attention itself can still move large intermediate tensors through memory. FlashAttention-style methods reduce that traffic by reorganizing the computation while preserving the same mathematical attention result.

## Question 27 — If attention moves too much intermediate data, how does FlashAttention improve execution?

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


### Story Bridge 47 — Long Context Is the Stress Test That Makes Several Bottlenecks Collide

We now have tools for weights, KV memory, repeated prefixes, memory allocation, and attention traffic. Increase the context dramatically and several pressures rise together: prefill work, KV capacity, decode reads, memory bandwidth, and reduced concurrency.

## Question 28 — What happens when we stress all of these mechanisms with a very long context?

Long context affects several resources simultaneously:

1. Prefill attention has a quadratic position-interaction component.
2. The KV cache grows roughly linearly with cached token count.
3. Decode attention reads a larger historical cache.
4. More cache memory reduces how many requests can fit concurrently.
5. Data movement can become increasingly important.

Long context is therefore not just a token-count issue; it is a compute, bandwidth, and memory-capacity problem.

### Story Bridge 48 — Turn the Detailed Cost Story Into a Hardware Heuristic

Long-context inference shows that bottlenecks can shift with workload shape. Still, practitioners often summarize serving behavior with one useful heuristic: prefill tends to be more compute-oriented, while low-batch decode tends to be more bandwidth-oriented. We need to understand both the intuition and the limits of that rule.

## Question 29 — After seeing all these costs, when is "prefill compute-bound, decode memory-bound" a useful rule?

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

### Story Bridge 49 — Capacity Is Only One Dimension of Model Size

The compute-vs-bandwidth heuristic explains why fitting a model is not the same as serving it quickly. A larger dense model carries more parameters through every token step, increasing arithmetic, movement, and often communication even when sufficient memory exists.

## Question 30 — If both models fit in memory, why is a 70B dense model still usually slower than a 7B model?

A 70B dense model has roughly ten times as many parameters as a 7B model.

That generally means:

- More weight memory.
- More arithmetic per forward pass.
- More data movement.
- Greater need for multi-GPU execution.
- More communication if sharded.

Even if enough memory exists, repeatedly executing a much larger model for every output token increases latency and reduces the number of concurrent requests a fixed hardware pool can serve.

### Story Bridge 50 — A Fast Arithmetic Engine Can Still Wait on Everything Around It

The 70B-versus-7B comparison reinforces that runtime depends on more than raw operation count. Bandwidth, KV traffic, communication, batch shape, kernels, scheduling, and memory management can all prevent the GPU from approaching its theoretical arithmetic peak.

## Question 31 — Why can't we predict LLM serving speed from peak GPU FLOPs alone?

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

### Story Bridge 51 — Many Bottlenecks Need One Organizing Equation

At this point we have accumulated many costs: queueing, prefill, decode, weights, KV state, activations, communication, and runtime overhead. Rather than memorize them separately, we can organize them into a simple latency-and-memory decomposition.

## Question 32 — Can we combine compute, memory, and scheduling into one simple request-cost model?

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


### Story Bridge 52 — Put Yourself Inside the Inference Server

The cost model gives us the pieces, but retention improves when we watch those pieces appear in sequence. So now we follow one concrete request through weight residency, prefill, first-token production, hundreds of decode steps, and a growing KV cache.

## Question 33 — What does the full cost story look like for one request from arrival to completion?

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

### Story Bridge 53 — Optimization Only Makes Sense After the Bottleneck Is Named

The end-to-end example shows that no single resource dominates every request. That means optimization should be diagnostic: remove recomputation with KV cache, reduce bytes with quantization, manage KV memory with paging, improve attention traffic with FlashAttention, and so on.

## Question 34 — Once we can locate each bottleneck, which optimization should attack which one?

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

### Story Bridge 54 — Finish by Reconstructing the Whole System, Not Memorizing a List

We can now trace inference cost from a single autoregressive token all the way to production serving across devices. The final step is to compress model size, repeated decode, memory bandwidth, KV growth, long context, batching, and communication into one coherent explanation.

## Question 35 — How can we compress the whole chapter into one answer to "Why is LLM inference expensive?"

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
