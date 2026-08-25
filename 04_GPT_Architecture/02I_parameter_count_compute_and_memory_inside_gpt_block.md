# GPT Architecture — Part 2I: Parameter Count, Compute Cost, and Memory Inside One GPT Block

## 1. Why This Part Matters

In the previous parts, we studied the internal components of one GPT block:

```text
Part 2A → Input and tensor shapes

Part 2B → LayerNorm

Part 2C → Q, K, V projections

Part 2D → Causal masked self-attention

Part 2E → Multi-head attention

Part 2F → Residual connections

Part 2G → Feed-forward network / MLP

Part 2H → Pre-LayerNorm vs Post-LayerNorm
```

Now we ask a very practical interview question:

```text
How expensive is one GPT block?
```

This means understanding:

```text
1. How many parameters are inside one GPT block?

2. How much computation happens inside one GPT block?

3. How much memory is needed during training and inference?
```

This part is very important for LLM system design because production LLMs are limited by:

```text
GPU memory

inference latency

training cost

batch size

context length

KV cache size

number of model parameters
```

The main idea is:

```text
Attention is expensive mainly because of sequence length.

FFN is expensive mainly because of large weight matrices.

KV cache is expensive during inference because it grows with context length and number of layers.
```

---

## 2. What Is Inside One GPT Block?

A modern GPT block has:

```text
Input X
  ↓
LayerNorm
  ↓
Masked Multi-Head Attention
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network / MLP
  ↓
Residual Connection
  ↓
Output Y
```

The main parameter-containing parts are:

```text
1. Attention projections:
   W_Q, W_K, W_V, W_O

2. Feed-forward / MLP projections:
   W_up, W_down
   or W_gate, W_up, W_down in gated FFNs

3. LayerNorm parameters:
   gamma and beta
```

Residual connections do not add parameters.

They are just addition.

---

## 3. Key Symbols

We will use the following symbols throughout this note:

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `T` | Sequence length / number of tokens |
| `d_model` | Hidden dimension / model dimension |
| `H` | Number of attention heads |
| `D` | Head dimension |
| `d_ff` | Feed-forward intermediate dimension |
| `V` | Vocabulary size |
| `L` | Number of GPT blocks / layers |
| `s` | Number of bytes used per stored value |

The most important relationship is:

`d_model = H × D`

where:

```text
H = number of attention heads
D = dimension of each head
```

Example:

```text
d_model = 768
H = 12
D = 64
```

So:

```text
768 = 12 × 64
```

Memory hook:

```text
The full model dimension is split across attention heads.
```

---

## 4. Three Kinds of Cost

When analyzing GPT cost, separate these three things:

```text
1. Parameter count

2. Compute cost

3. Memory cost
```

They are related, but they are not the same.

---

## 5. Parameter Count

Parameter count means:

```text
How many trainable numbers does the model contain?
```

Examples of parameters:

```text
W_Q

W_K

W_V

W_O

FFN weights

LayerNorm gamma and beta

Embedding matrix

Vocabulary output head
```

Parameter count affects:

```text
model size on disk

GPU memory needed to load the model

training memory

potential model capacity
```

---

## 6. Compute Cost

Compute cost means:

```text
How many mathematical operations are needed during a forward pass?
```

Compute cost affects:

```text
training time

inference latency

GPU utilization

throughput
```

Example:

```text
Matrix multiplication is compute-heavy.

Attention score calculation QK^T is compute-heavy.

FFN matrix multiplications are compute-heavy.
```

---

## 7. Memory Cost

Memory cost means:

```text
How much memory is needed to store weights, activations, gradients, optimizer states, and KV cache?
```

Different stages need different memory.

During training, we store:

```text
model weights

activations

gradients

optimizer states
```

During inference, we store:

```text
model weights

current activations

KV cache
```

Memory hook:

```text
Training memory is dominated by weights, activations, gradients, and optimizer states.

Inference memory is dominated by weights and KV cache.
```

---

# Part A: Parameter Count Inside One GPT Block

---

## 8. Attention Parameters

Inside one standard multi-head attention module, we have four main matrices:

`W_Q`, `W_K`, `W_V`, and `W_O`

where:

```text
W_Q = query projection

W_K = key projection

W_V = value projection

W_O = output projection
```

In standard multi-head attention, each matrix usually has shape:

```text
d_model × d_model
```

So each matrix has:

```text
d_model²
```

parameters.

Since there are four matrices, the total number of attention parameters is:

```text
N_attn = 4d_model²
```

---

## 9. Attention Parameter Example: d_model = 768

Suppose:

```text
d_model = 768
```

Each attention matrix has:

```text
768 × 768 = 589,824
```

parameters.

There are four matrices:

`W_Q`, `W_K`, `W_V`, and `W_O`

Total attention parameters:

```text
N_attn = 4 × 589,824

N_attn = 2,359,296
```

So one attention module has about:

```text
2.36 million parameters
```

---

## 10. Attention Parameter Example: d_model = 4096

Suppose:

```text
d_model = 4096
```

Each attention matrix has:

```text
4096 × 4096 = 16,777,216
```

parameters.

Total attention parameters:

```text
N_attn = 4 × 16,777,216

N_attn = 67,108,864
```

So one attention module has about:

```text
67.1 million parameters
```

inside one GPT block.

---

## 11. Feed-Forward Network Parameters

A standard FFN has two main matrices:

`W_up` and `W_down`

The FFN expands and compresses:

```text
d_model → d_ff → d_model
```

So the shapes are:

```text
W_up shape   = d_model × d_ff

W_down shape = d_ff × d_model
```

Total FFN parameters:

```text
N_ffn = d_model × d_ff + d_ff × d_model
```

Therefore:

```text
N_ffn = 2d_model d_ff
```

---

## 12. FFN Parameter Example: d_model = 768

Suppose:

```text
d_model = 768

d_ff = 3072
```

First matrix:

```text
W_up = 768 × 3072 = 2,359,296
```

Second matrix:

```text
W_down = 3072 × 768 = 2,359,296
```

Total FFN parameters:

```text
N_ffn = 2,359,296 + 2,359,296

N_ffn = 4,718,592
```

So the FFN has about:

```text
4.72 million parameters
```

inside one GPT block.

---

## 13. FFN Parameter Example: d_model = 4096

Suppose:

```text
d_model = 4096

d_ff = 11008
```

First matrix:

```text
W_up = 4096 × 11008 = 45,088,768
```

Second matrix:

```text
W_down = 11008 × 4096 = 45,088,768
```

Total standard FFN parameters:

```text
N_ffn = 90,177,536
```

So a standard FFN has about:

```text
90.2 million parameters
```

inside one GPT block.

---

## 14. Gated FFN Parameters

Many modern LLMs use gated FFNs such as SwiGLU.

A gated FFN often has three main matrices:

`W_gate`, `W_up`, and `W_down`

Simplified flow:

```text
x
  ↓
gate projection
  ↓
up projection
  ↓
element-wise gating
  ↓
down projection
```

Parameter shapes:

```text
W_gate shape = d_model × d_ff

W_up shape   = d_model × d_ff

W_down shape = d_ff × d_model
```

Total gated FFN parameters:

```text
N_gated_ffn = 3d_model d_ff
```

This is larger than a standard FFN.

---

## 15. Gated FFN Parameter Example: d_model = 4096

Suppose:

```text
d_model = 4096

d_ff = 11008
```

Each projection involving:

```text
d_model × d_ff
```

has:

```text
4096 × 11008 = 45,088,768
```

parameters.

For three matrices:

```text
N_gated_ffn = 3 × 45,088,768

N_gated_ffn = 135,266,304
```

So a gated FFN can have about:

```text
135.3 million parameters
```

inside one GPT block.

This is why the MLP/FFN part often contains a very large fraction of total model parameters.

---

## 16. LayerNorm Parameters

LayerNorm has two learnable vectors:

`gamma` and `beta`

Each has size:

```text
d_model
```

So one LayerNorm has:

```text
2d_model
```

parameters.

A GPT block usually has two LayerNorms:

```text
LayerNorm before attention

LayerNorm before FFN
```

So LayerNorm parameters per block:

```text
N_ln = 4d_model
```

Example:

```text
d_model = 768
```

LayerNorm parameters:

```text
N_ln = 4 × 768 = 3072
```

This is very small compared to attention and FFN parameters.

---

## 17. Residual Connection Parameters

Residual connections have:

```text
0 parameters
```

A residual connection is just addition.

Example:

```text
A = X + AttentionOutput
```

No matrix is learned for this addition.

Memory hook:

```text
Residuals are parameter-free but extremely important.
```

---

## 18. Total Parameters in One GPT Block: Standard FFN

For a standard FFN:

```text
N_attn = 4d_model²

N_ffn = 2d_model d_ff

N_ln = 4d_model
```

Therefore:

```text
N_block = N_attn + N_ffn + N_ln
```

Substituting each term:

```text
N_block = 4d_model² + 2d_model d_ff + 4d_model
```

Bias terms are often small compared to matrix weights, so they are usually ignored in rough estimates.

---

## 19. Total Parameter Example: d_model = 768, d_ff = 3072

Suppose:

```text
d_model = 768

d_ff = 3072
```

Attention:

```text
N_attn = 4 × 768²

N_attn = 2,359,296
```

FFN:

```text
N_ffn = 2 × 768 × 3072

N_ffn = 4,718,592
```

LayerNorm:

```text
N_ln = 4 × 768

N_ln = 3,072
```

Total:

```text
N_block = 2,359,296 + 4,718,592 + 3,072

N_block = 7,080,960
```

So one GPT block has about:

```text
7.08 million parameters
```

for this configuration.

---

## 20. Total Parameters in One GPT Block: Gated FFN

For a gated FFN:

```text
N_attn = 4d_model²

N_gated_ffn = 3d_model d_ff

N_ln = 4d_model
```

Therefore:

```text
N_block = N_attn + N_gated_ffn + N_ln
```

Substituting each term:

```text
N_block = 4d_model² + 3d_model d_ff + 4d_model
```

Important GitHub note:

```text
Do not write this formula as a heading like "# $$ N_block ... $$".
Keep it inside a normal text block like this.
```

---

## 21. Total Parameter Example: d_model = 4096, d_ff = 11008

Suppose:

```text
d_model = 4096

d_ff = 11008
```

Attention:

```text
N_attn = 4 × 4096²

N_attn = 67,108,864
```

Gated FFN:

```text
N_gated_ffn = 3 × 4096 × 11008

N_gated_ffn = 135,266,304
```

LayerNorm:

```text
N_ln = 4 × 4096

N_ln = 16,384
```

Total:

```text
N_block = 67,108,864 + 135,266,304 + 16,384

N_block = 202,391,552
```

So one GPT block has about:

```text
202.4 million parameters
```

for this larger gated-FFN configuration.

---

## 22. Important Observation

In many GPT-style models:

```text
N_ffn > N_attn
```

Example with a standard FFN:

```text
N_attn ≈ 4d_model²
```

If:

```text
d_ff = 4d_model
```

then:

```text
N_ffn = 2d_model(4d_model)

N_ffn = 8d_model²
```

So the FFN can contain about twice as many parameters as attention.

Memory hook:

```text
Attention is often expensive because of sequence length.

FFN is often parameter-heavy because of d_ff.
```

---

# Part B: Compute Cost Inside One GPT Block

---

## 23. Compute Cost Overview

Inside one GPT block, compute mainly comes from:

```text
1. Q, K, V projections

2. Attention score calculation QK^T

3. Attention weights multiplied by V

4. Output projection W_O

5. FFN / MLP matrix multiplications
```

LayerNorm and residual additions are usually cheaper compared to large matrix multiplications.

---

## 24. Compute Cost of Q, K, V Projections

Input shape:

```text
B × T × d_model
```

Each token vector is multiplied by projection matrices.

For Q, K, and V:

```text
Q = XW_Q

K = XW_K

V = XW_V
```

Each projection costs roughly:

```text
BTd_model²
```

For three projections:

```text
3BTd_model²
```

Output projection `W_O` adds another:

```text
BTd_model²
```

So all attention projections together cost roughly:

```text
C_proj ≈ 4BTd_model²
```

This grows linearly with sequence length `T`.

---

## 25. Compute Cost of Attention Scores

After Q and K are created, GPT computes:

```text
QK^T
```

For each head:

```text
Q_h shape = T × D

K_h shape = T × D

Q_h K_h^T shape = T × T
```

Cost per head:

```text
T²D
```

For all heads:

```text
HT²D
```

Since:

```text
H × D = d_model
```

this becomes:

```text
T²d_model
```

With batch size:

```text
BT²d_model
```

So attention score computation grows as:

```text
T²
```

with sequence length.

---

## 26. Compute Cost of Attention Weights Times V

After softmax, GPT computes:

```text
AttentionWeights × V
```

For each head:

```text
AttentionWeights shape = T × T

V_h shape = T × D

Output shape = T × D
```

Cost per head:

```text
T²D
```

For all heads:

```text
T²d_model
```

With batch size:

```text
BT²d_model
```

So this is also quadratic in sequence length.

---

## 27. Total Attention Compute

Attention has two major parts.

### Projection compute

Q, K, V, and output projection together cost approximately:

```text
C_proj ≈ 4BTd_model²
```

This is linear in `T`.

---

### Attention matrix compute

Computing attention scores and multiplying by values costs approximately:

```text
C_attn_matrix ≈ 2BT²d_model
```

This is quadratic in `T`.

---

### Total attention compute

So the approximate attention compute is:

```text
C_attn ≈ 4BTd_model² + 2BT²d_model
```

Memory hook:

```text
Projection part grows with T.

Attention matrix part grows with T².
```

For short sequences, projection and FFN can dominate.

For long sequences, the `T²` attention term becomes very important.

---

## 28. Compute Cost of FFN

For a standard FFN:

```text
d_model → d_ff → d_model
```

First matrix multiplication:

```text
BTd_model d_ff
```

Second matrix multiplication:

```text
BTd_ff d_model
```

Total FFN compute:

```text
C_ffn ≈ 2BTd_model d_ff
```

This grows linearly with sequence length `T`.

---

## 29. Compute Cost of Gated FFN

For a gated FFN, there are often three projections:

```text
gate projection

up projection

down projection
```

So compute is roughly:

```text
C_gated_ffn ≈ 3BTd_model d_ff
```

This is larger than a standard FFN.

---

## 30. Attention vs FFN Compute

Attention has a quadratic part:

```text
C_attn_matrix ∝ T²
```

FFN has a linear part:

```text
C_ffn ∝ T
```

So:

```text
For short context:
FFN can dominate compute.

For long context:
attention can become very expensive.
```

Important interview line:

```text
Attention scales quadratically with sequence length, while FFN scales linearly with sequence length but has large weight matrices.
```

---

## 31. Example: Effect of Increasing Sequence Length

Suppose sequence length doubles:

```text
T → 2T
```

Attention score matrix cost:

```text
T² → (2T)² = 4T²
```

So attention matrix compute becomes about:

```text
4 times larger
```

FFN compute:

```text
T → 2T
```

So FFN compute becomes about:

```text
2 times larger
```

This is why long context is difficult.

Memory hook:

```text
Double the context length:
attention matrix cost becomes 4x,
FFN cost becomes 2x.
```

---

# Part C: Memory Cost Inside One GPT Block

---

## 32. Types of Memory in GPT

There are several types of memory to understand.

```text
1. Parameter memory

2. Activation memory

3. Gradient memory

4. Optimizer state memory

5. KV cache memory
```

Different ones matter in different situations.

---

## 33. Parameter Memory

Parameter memory is the memory needed to store model weights.

Example:

```text
Number of parameters = 1 billion
```

If each parameter uses FP16 or BF16:

```text
2 bytes per parameter
```

Then parameter memory is:

```text
1,000,000,000 × 2 bytes = 2 GB
```

If each parameter uses FP32:

```text
4 bytes per parameter
```

Then:

```text
1,000,000,000 × 4 bytes = 4 GB
```

Memory hook:

```text
FP16/BF16 uses 2 bytes per parameter.

FP32 uses 4 bytes per parameter.
```

---

## 34. Parameter Memory for One GPT Block

Suppose one GPT block has:

```text
202 million parameters
```

Using FP16 or BF16:

```text
202,000,000 × 2 bytes ≈ 404 MB
```

Using FP32:

```text
202,000,000 × 4 bytes ≈ 808 MB
```

This is just one block.

A full model has many blocks.

---

## 35. Activation Memory

Activations are intermediate values produced during the forward pass.

Examples:

```text
hidden states

Q, K, V

attention scores

attention weights

FFN intermediate activations
```

During training, activations are stored because backpropagation needs them.

During inference, many activations can be discarded after use.

So activation memory is much more important during training.

---

## 36. Activation Memory Shape Examples

Input hidden states:

```text
B × T × d_model
```

Q, K, V:

```text
B × H × T × D
```

Attention scores:

```text
B × H × T × T
```

FFN intermediate:

```text
B × T × d_ff
```

Important:

```text
Attention scores grow with T².

FFN activations grow with T.
```

---

## 37. Attention Matrix Memory

The attention score matrix has shape:

```text
B × H × T × T
```

This can become very large.

Example:

```text
B = 1
H = 32
T = 4096
```

Attention scores:

```text
1 × 32 × 4096 × 4096
```

Number of values:

```text
536,870,912
```

If using FP16:

```text
536,870,912 × 2 bytes ≈ 1.07 GB
```

This is only the attention score matrix for one layer.

That is why efficient attention implementations are important.

---

## 38. FFN Activation Memory

The FFN intermediate activation has shape:

```text
B × T × d_ff
```

Example:

```text
B = 1
T = 4096
d_ff = 11008
```

Number of values:

```text
1 × 4096 × 11008 = 45,088,768
```

Using FP16:

```text
45,088,768 × 2 bytes ≈ 90 MB
```

This is smaller than the attention score example above, but still significant.

---

## 39. Training Memory

During training, memory is needed for:

```text
1. Weights

2. Forward activations

3. Gradients

4. Optimizer states
```

With Adam-like optimizers, optimizer state can be large because Adam stores:

```text
first moment estimate

second moment estimate
```

So training memory is much larger than inference memory.

Rough intuition:

```text
Inference:
mostly weights + KV cache

Training:
weights + activations + gradients + optimizer states
```

---

## 40. Optimizer State Memory

Adam optimizer stores extra values for every parameter.

For each parameter, Adam commonly stores:

```text
parameter value

gradient

first moment

second moment
```

So optimizer-related memory can be several times larger than the model weights alone.

Very rough memory intuition:

```text
Training can need 4x or more memory than just storing the model weights.
```

Exact memory depends on:

```text
precision

optimizer

gradient checkpointing

distributed training strategy

activation recomputation

ZeRO stage

offloading
```

---

## 41. Gradient Checkpointing

Gradient checkpointing is a training technique used to reduce activation memory.

Normally, training stores many activations from the forward pass.

Gradient checkpointing stores fewer activations and recomputes some of them during the backward pass.

Tradeoff:

```text
less memory usage

more computation
```

Memory hook:

```text
Gradient checkpointing trades compute for memory.
```

---

# Part D: KV Cache Memory During Inference

---

## 42. What Is KV Cache?

During autoregressive inference, GPT generates tokens one by one.

At each step, the model needs keys and values from previous tokens.

Without KV cache, the model would recompute K and V for all previous tokens again and again.

KV cache stores:

```text
K vectors from previous tokens

V vectors from previous tokens
```

This makes inference faster.

---

## 43. Why KV Cache Matters

Suppose the prompt is:

```text
I love machine learning
```

When generating the next token, GPT computes K and V for these tokens.

When generating the next token after that, the old tokens have not changed.

So their K and V can be reused.

Memory hook:

```text
Old tokens do not change during generation.

Therefore, their K and V can be cached.
```

---

## 44. KV Cache Shape for One Layer

For one GPT block/layer, KV cache stores K and V.

For standard multi-head attention:

```text
K_cache shape = B × H × T × D

V_cache shape = B × H × T × D
```

So total KV cache values per layer:

```text
2BHTD
```

Since:

```text
H × D = d_model
```

this becomes:

```text
2BTd_model
```

values per layer.

---

## 45. KV Cache Shape Across All Layers

For a model with:

```text
L = number of layers
```

KV cache values across the full model:

```text
2LBHTD
```

Since:

```text
H × D = d_model
```

this becomes:

```text
2LBTd_model
```

If each stored value uses `s` bytes, then KV cache memory is:

```text
M_KV = 2LBTd_model s
```

For FP16 or BF16:

```text
s = 2
```

So:

```text
M_KV = 4LBTd_model bytes
```

The first factor of `2` is for K and V.

The value `s` is the number of bytes per stored value.

---

## 46. KV Cache Example

Suppose:

```text
L = 32

B = 1

T = 4096

d_model = 4096

s = 2 bytes for FP16
```

KV cache values:

```text
2 × 32 × 1 × 4096 × 4096 = 1,073,741,824 values
```

Memory:

```text
M_KV = 1,073,741,824 × 2 bytes

M_KV = 2,147,483,648 bytes

M_KV ≈ 2.15 GB
```

So the KV cache alone is about:

```text
2.15 GB
```

for one sequence with 4096 tokens.

This is why long-context inference consumes a lot of memory.

---

## 47. KV Cache With Batch Size

If batch size increases, KV cache increases linearly.

Example:

```text
B = 8
```

Using the previous example:

```text
2.15 GB × 8 = 17.2 GB
```

So serving many users at once can require a lot of GPU memory.

Memory hook:

```text
KV cache grows with batch size and context length.
```

---

## 48. KV Cache With Longer Context

If context length doubles:

```text
T = 4096 → 8192
```

KV cache memory doubles.

Unlike attention score compute, KV cache memory grows linearly with context length.

```text
M_KV ∝ T

C_attn_matrix ∝ T²
```

So:

```text
KV cache memory grows linearly with T.

Attention score compute grows quadratically with T.
```

---

## 49. MHA vs GQA vs MQA and KV Cache

In standard multi-head attention:

```text
number of query heads = number of key/value heads
```

So KV cache uses:

```text
H key/value heads
```

In grouped-query attention:

```text
many query heads share fewer key/value heads
```

So KV cache is smaller.

In multi-query attention:

```text
many query heads share one key/value head
```

So KV cache is even smaller.

Memory hook:

```text
GQA and MQA reduce KV cache memory.
```

This is one reason modern LLMs often use GQA or MQA for efficient inference.

---

# Part E: Practical Production Implications

---

## 50. Why Larger d_model Increases Cost So Much

Many parameter formulas involve:

```text
d_model²
```

or:

```text
d_model d_ff
```

So increasing `d_model` significantly increases parameter count.

Example:

```text
d_model → 2d_model
```

Then:

```text
d_model² → (2d_model)² = 4d_model²
```

So doubling the hidden dimension can make some parameter and compute terms about four times larger.

---

## 51. Why Longer Context Is Expensive

Longer context affects:

```text
attention score matrix

KV cache

activation memory

latency
```

Attention scores grow with:

```text
T²
```

KV cache grows with:

```text
T
```

So long context is expensive in two ways:

```text
1. More computation because attention compares more token pairs.

2. More memory because KV cache stores more previous tokens.
```

---

## 52. Why Batch Size Matters

Batch size affects memory and throughput.

Larger batch size:

```text
improves GPU utilization

can increase throughput

uses more memory

can increase latency if batching waits too long
```

During inference serving, batching is a tradeoff:

```text
larger batch = better throughput

smaller batch = lower latency
```

---

## 53. Why Quantization Helps

Quantization reduces the number of bytes used per parameter.

Example:

| Precision | Bytes per parameter |
|---|---:|
| FP32 | 4 bytes |
| FP16 / BF16 | 2 bytes |
| INT8 | 1 byte |
| INT4 | 0.5 bytes |

So quantization reduces memory usage.

It can also improve inference speed depending on hardware support.

Tradeoff:

```text
lower memory and possibly faster inference

possible quality loss
```

---

## 54. Why Smaller Models Are Faster

Smaller models usually have:

```text
smaller d_model

fewer layers

smaller FFN dimension

fewer attention heads
```

This reduces:

```text
parameter memory

compute cost

latency

KV cache size
```

In production, smaller models are often used when:

```text
latency is strict

cost must be low

task is simple

high throughput is needed
```

---

## 55. Why Long Context Can Be More Expensive Than Expected

A user may think:

```text
I only doubled the context window.
```

But attention compute may become about:

```text
4 times larger
```

because of the `T²` attention matrix.

This is one reason RAG systems often retrieve only a few relevant chunks instead of putting every document into the prompt.

Memory hook:

```text
Do not put everything into the prompt.

Long context is expensive.
```

---

## 56. Connection to RAG System Design

In RAG, we often retrieve top-k chunks and place them in the prompt.

If we retrieve too many chunks:

```text
T increases
```

Then:

```text
attention cost increases

KV cache memory increases

latency increases

cost per request increases
```

So RAG design must balance:

```text
retrieval recall

context length

answer quality

latency

cost
```

This is why chunking, reranking, and prompt compression matter.

---

## 57. Connection to Inference Latency

Inference latency depends on:

```text
model size

number of layers

d_model

d_ff

context length

batch size

hardware

KV cache

quantization

serving engine
```

For each generated token, GPT must pass the new token through all layers.

Each layer runs:

```text
LayerNorm

attention using current Q and cached K/V

FFN

residual additions
```

So deeper and wider models are slower.

---

## 58. Prefill vs Decode

LLM inference has two phases:

```text
1. Prefill

2. Decode
```

### Prefill

Prefill processes the input prompt.

If prompt length is:

```text
T
```

the model processes all prompt tokens.

Attention during prefill can involve:

```text
T × T
```

attention patterns.

So long prompts make prefill expensive.

### Decode

Decode generates one token at a time.

At each decode step, the new token attends to previous cached K and V.

The KV cache helps avoid recomputing old K and V.

Memory hook:

```text
Prefill cost depends heavily on prompt length.

Decode cost depends heavily on generated length and KV cache.
```

---

## 59. Why KV Cache Speeds Up Decode

Without KV cache, when generating token number 1000, the model would recompute information for all previous 999 tokens.

With KV cache:

```text
previous K and V are reused
```

The model only computes new Q, K, and V for the new token.

Then the new query attends to cached keys and values.

This greatly reduces repeated computation.

---

## 60. Why KV Cache Uses Memory

The speed benefit comes at a memory cost.

For every generated or prompt token, for every layer, the model stores:

```text
K vector

V vector
```

So longer conversations increase KV cache memory.

This is why production systems often have limits on:

```text
maximum context length

maximum output length

batch size

number of simultaneous requests
```

---

## 61. Important System Design Tradeoff

In production LLM systems:

```text
More context
  → better access to information
  → more memory
  → more latency
  → higher cost

Smaller context
  → lower cost and latency
  → risk of missing useful information
```

A good ML system designer must balance this tradeoff.

---

## 62. Optimization Techniques

Common ways to reduce cost:

```text
use smaller model

use quantization

use KV cache

use batching carefully

use shorter prompts

retrieve fewer but better chunks

use reranking

use prompt compression

use speculative decoding

use GQA or MQA

cache frequent responses

route simple queries to cheaper models

distill large model into smaller model
```

These topics will become very important in LLM system design interviews.

---

## 63. Common Interview Calculation: Block Parameters

Question:

```text
A GPT block has d_model = 768 and d_ff = 3072.
Ignoring biases, estimate the number of parameters in one block.
```

Answer:

Attention:

```text
N_attn = 4d_model²

N_attn = 4 × 768²

N_attn = 2,359,296
```

FFN:

```text
N_ffn = 2d_model d_ff

N_ffn = 2 × 768 × 3072

N_ffn = 4,718,592
```

LayerNorm:

```text
N_ln = 4d_model

N_ln = 4 × 768

N_ln = 3,072
```

Total:

```text
N_block = 2,359,296 + 4,718,592 + 3,072

N_block = 7,080,960
```

So one block has about:

```text
7.1 million parameters
```

---

## 64. Common Interview Calculation: KV Cache

Question:

```text
Estimate KV cache memory for one sequence.

L = 32 layers
T = 4096 tokens
d_model = 4096
precision = FP16
B = 1
```

Formula:

```text
M_KV = 2LBTd_model s
```

For FP16:

```text
s = 2 bytes
```

Substitute:

```text
M_KV = 2 × 32 × 1 × 4096 × 4096 × 2 bytes
```

Number of bytes:

```text
M_KV = 2,147,483,648 bytes
```

Approximately:

```text
M_KV ≈ 2.15 GB
```

So KV cache is about:

```text
2.15 GB
```

for one request.

Important GitHub note:

```text
Do not write this formula as a heading like "# $$ M_KV ... $$".
Keep it inside a normal text block like this.
```

---

## 65. Common Interview Calculation: Effect of Context Length

Question:

```text
If context length doubles, what happens to attention compute and KV cache?
```

Answer:

Attention score compute grows as:

```text
T²
```

So doubling `T` gives:

```text
T → 2T

T² → (2T)² = 4T²
```

Therefore attention compute becomes about:

```text
4 times larger
```

KV cache grows as:

```text
T
```

So doubling `T` makes KV cache about:

```text
2 times larger
```

Memory hook:

```text
Attention compute: doubles context → 4x

KV cache memory: doubles context → 2x
```

---

## 66. Interview-Level Answer

If an interviewer asks:

```text
Where does the cost of a GPT block come from?
```

You can answer:

The main parameter cost in a GPT block comes from the attention projection matrices and the feed-forward network. Standard attention has roughly `4d_model²` parameters from `W_Q`, `W_K`, `W_V`, and `W_O`. A standard FFN has roughly `2d_model d_ff` parameters, and gated FFNs have roughly `3d_model d_ff` parameters. The FFN often contains more parameters than the attention module.

The compute cost comes from QKV projections, attention score computation, multiplying attention weights by V, output projection, and FFN matrix multiplications. Attention has a quadratic term in sequence length because the attention matrix is `T × T`, while the FFN grows linearly with sequence length but has large matrix multiplications.

During inference, KV cache is also important. It stores K and V vectors for previous tokens across all layers, with memory roughly proportional to `2LBTd_model s`. This speeds up decoding but increases memory usage as context length, batch size, and number of layers increase.

---

## 67. Common Interview Follow-Up Questions

### Q1. Which part of a GPT block has the most parameters?

Often the FFN/MLP has the most parameters, especially when `d_ff` is much larger than `d_model`.

---

### Q2. Why is attention expensive for long sequences?

Because each token compares with every other token.

The attention score matrix has shape:

```text
T × T
```

So attention cost grows roughly as:

```text
T²
```

---

### Q3. Why is FFN expensive?

Because it uses large matrix multiplications:

```text
d_model → d_ff → d_model
```

Since `d_ff` is often several times larger than `d_model`, the FFN has many parameters and high compute cost.

---

### Q4. What is the parameter count of standard attention?

Ignoring biases:

```text
N_attn = 4d_model²
```

for:

```text
W_Q, W_K, W_V, W_O
```

---

### Q5. What is the parameter count of a standard FFN?

Ignoring biases:

```text
N_ffn = 2d_model d_ff
```

---

### Q6. What is the parameter count of a gated FFN?

Ignoring biases:

```text
N_gated_ffn = 3d_model d_ff
```

---

### Q7. What is KV cache?

KV cache stores the key and value vectors of previous tokens during autoregressive inference.

It avoids recomputing K and V for old tokens at every generation step.

---

### Q8. What is the KV cache memory formula?

For standard multi-head attention:

```text
M_KV = 2LBTd_model s
```

where:

```text
2 = K and V

L = number of layers

B = batch size

T = sequence length

d_model = hidden dimension

s = bytes per stored value
```

---

### Q9. How does batch size affect KV cache?

KV cache grows linearly with batch size.

If batch size doubles, KV cache memory doubles.

---

### Q10. How does context length affect cost?

Attention score compute grows quadratically with context length:

```text
T²
```

KV cache grows linearly with context length:

```text
T
```

---

### Q11. Why do GQA and MQA reduce memory?

They reduce the number of key/value heads stored in the KV cache.

Fewer K/V heads means smaller KV cache.

---

### Q12. What is the difference between prefill and decode?

Prefill processes the full input prompt.

Decode generates one token at a time.

Prefill is expensive for long prompts.

Decode benefits heavily from KV cache.

---

## 68. Memory Hooks

```text
Attention parameters:
N_attn = 4d_model²

Standard FFN parameters:
N_ffn = 2d_model d_ff

Gated FFN parameters:
N_gated_ffn = 3d_model d_ff

LayerNorm parameters are tiny:
N_ln = 4d_model per block

Residual connections have zero parameters.

Attention score matrix:
B × H × T × T

Attention matrix compute:
C_attn_matrix ∝ T²

FFN compute:
C_ffn ∝ T

KV cache values:
2LBTd_model

KV cache memory:
M_KV = 2LBTd_model × bytes_per_value

For FP16/BF16:
M_KV = 4LBTd_model bytes

Training memory:
weights + activations + gradients + optimizer states

Inference memory:
weights + activations + KV cache

Long context is expensive.

More context means better information but higher latency and cost.
```

---

## 69. Final One-Line Summary

```text
The cost of a GPT block mainly comes from attention projections, the T² attention matrix, large FFN/MLP matrices, and KV cache memory during inference.
```
