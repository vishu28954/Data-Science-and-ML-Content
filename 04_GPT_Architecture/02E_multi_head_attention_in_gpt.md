# GPT Architecture — Part 2E: Multi-Head Attention in GPT

## 1. Why This Part Matters

In the previous parts, we studied:

```text
Part 2A → Input and tensor shapes inside a GPT block

Part 2B → LayerNorm in GPT

Part 2C → Q, K, V projections

Part 2D → Causal masked self-attention
```

Now we study how GPT performs attention using **multiple heads**.

The main idea is:

```text
Multi-head attention allows GPT to look at the same sequence in multiple different ways at the same time.
```

A single attention head may learn one type of relationship.

Multiple heads allow the model to capture many relationships in parallel.

Examples:

```text
Head 1 → nearby words

Head 2 → subject-verb relationship

Head 3 → pronoun reference

Head 4 → punctuation or syntax

Head 5 → long-range dependency

Head 6 → instruction-following pattern
```

---

## 2. Where Multi-Head Attention Appears Inside a GPT Block

A GPT block looks like this:

```text
Input X
  ↓
LayerNorm
  ↓
Create Q, K, V
  ↓
Split Q, K, V into multiple heads
  ↓
Causal masked self-attention per head
  ↓
Concatenate heads
  ↓
Output projection
  ↓
Residual connection
  ↓
LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual connection
  ↓
Output Y
```

In this part, we focus on:

```text
Q, K, V
  ↓
Split into heads
  ↓
Attention per head
  ↓
Concatenate heads
  ↓
Output projection
```

---

## 3. Why Do We Need Multiple Heads?

A sentence contains many different types of relationships.

Example:

```text
The animal did not cross the street because it was tired.
```

Different heads may focus on different relationships:

```text
Head 1:
"it" attends to "animal"

Head 2:
"did not" attends to "cross"

Head 3:
"street" attends to nearby location context

Head 4:
"tired" attends to "animal"

Head 5:
captures grammar or phrase structure
```

One attention head may not be enough to capture all these relationships.

So GPT uses multiple heads.

Memory hook:

```text
Single-head attention = one view of the sentence.

Multi-head attention = many views of the sentence in parallel.
```

---

## 4. Input to Multi-Head Attention

The input to attention is the normalized hidden representation:

```text
LayerNorm(X)
```

Shape:

```text
B × T × d_model
```

where:

```text
B       = batch size
T       = sequence length
d_model = hidden dimension
```

Example:

```text
B = 2
T = 5
d_model = 768
```

So:

```text
Input shape = 2 × 5 × 768
```

This means:

```text
2 sequences in the batch
5 tokens per sequence
768 numbers per token
```

---

## 5. Number of Heads and Head Dimension

Suppose:

```text
d_model = 768
number of heads = 12
```

Then each head dimension is:

```text
head_dim = d_model / number_of_heads

head_dim = 768 / 12

head_dim = 64
```

So instead of using one 768-dimensional attention operation, GPT uses:

```text
12 attention heads
each with 64 dimensions
```

Important:

```text
12 × 64 = 768
```

So the total representation size stays the same.

---

## 6. General Shape Rule

Let:

```text
B = batch size
T = sequence length
H = number of attention heads
D = head dimension
```

Then:

```text
d_model = H × D
```

Example:

```text
H = 12
D = 64

d_model = 12 × 64 = 768
```

This is why `d_model` is usually divisible by the number of heads.

---

## 7. Q, K, V Before Splitting Into Heads

From Part 2C, GPT creates Q, K, and V from the input.

For simplicity, assume:

```text
Q shape = B × T × d_model
K shape = B × T × d_model
V shape = B × T × d_model
```

Example:

```text
Q shape = 2 × 5 × 768
K shape = 2 × 5 × 768
V shape = 2 × 5 × 768
```

But attention heads need smaller subspaces.

So GPT splits the last dimension into:

```text
number_of_heads × head_dim
```

For this example:

```text
768 → 12 × 64
```

---

## 8. Reshaping Q, K, V Into Heads

Before splitting:

```text
Q shape = B × T × d_model
```

After splitting into heads:

```text
Q shape = B × T × H × D
```

Then frameworks often rearrange dimensions to:

```text
Q shape = B × H × T × D
```

Why?

Because this makes it easier to compute attention separately for each head.

Same for K and V:

```text
K shape = B × H × T × D

V shape = B × H × T × D
```

Example:

```text
B = 2
T = 5
H = 12
D = 64
```

Then:

```text
Q shape = 2 × 12 × 5 × 64

K shape = 2 × 12 × 5 × 64

V shape = 2 × 12 × 5 × 64
```

Meaning:

```text
For every batch item,
for every attention head,
for every token,
there is a 64-dimensional Q, K, and V vector.
```

---

## 9. Why Do We Rearrange to B × H × T × D?

Many implementations use this layout:

```text
B × H × T × D
```

because attention is computed independently for each head.

For each batch item and each head:

```text
Q_head shape = T × D
K_head shape = T × D
V_head shape = T × D
```

Then attention is computed as:

```text
Q_head K_head^T
```

which gives:

```text
T × T
```

So each head gets its own attention matrix.

---

## 10. Attention Per Head

For each attention head:

```text
Q_h shape = B × T × D
K_h shape = B × T × D
V_h shape = B × T × D
```

The head computes attention scores:

```text
Scores_h = Q_h K_h^T
```

After scaling:

```text
ScaledScores_h = Scores_h / sqrt(D)
```

Then causal mask is applied:

```text
MaskedScores_h = ScaledScores_h + Mask
```

Then softmax creates attention weights:

```text
AttentionWeights_h = softmax(MaskedScores_h)
```

Then values are mixed:

```text
HeadOutput_h = AttentionWeights_h × V_h
```

So every head produces its own output.

---

## 11. Shape of Attention Scores Per Head

For one head:

```text
Q_h shape = T × D
K_h shape = T × D
```

Then:

```text
K_h^T shape = D × T
```

So:

```text
Q_h K_h^T shape = T × T
```

This means:

```text
each token compares with every other token
```

In GPT, future positions are masked.

With batch and heads:

```text
Attention scores shape = B × H × T × T
```

Example:

```text
B = 2
H = 12
T = 5
```

Then:

```text
Attention scores shape = 2 × 12 × 5 × 5
```

Meaning:

```text
2 batch items
12 heads
each head has a 5 × 5 attention matrix
```

---

## 12. What Each Head Learns

Each head has a different learned projection space.

This means each head may learn different attention patterns.

Example sentence:

```text
The cat sat on the mat because it was tired.
```

Possible head behavior:

```text
Head 1:
"it" attends to "cat"

Head 2:
"sat" attends to "cat"

Head 3:
"mat" attends to "on"

Head 4:
"tired" attends to "cat"

Head 5:
focuses on nearby tokens

Head 6:
focuses on phrase boundaries
```

Important:

```text
All heads see the same sequence.
But each head sees it through different learned projections.
```

---

## 13. Do Different Heads Receive Different Tokens?

No.

All heads receive information from the same sequence.

But each head transforms the token representations differently.

So:

```text
Same tokens
different projections
different attention patterns
different outputs
```

This is why multiple heads are useful.

They do not split the sentence into different chunks.

They split the representation space into different subspaces.

---

## 14. Simple Analogy

Imagine reading a sentence with different specialists:

```text
Specialist 1 looks for grammar.

Specialist 2 looks for meaning.

Specialist 3 looks for references.

Specialist 4 looks for negation.

Specialist 5 looks for syntax.

Specialist 6 looks for long-distance context.
```

Each specialist reads the same sentence but focuses on a different aspect.

That is multi-head attention.

---

## 15. Tiny Conceptual Example

Sentence:

```text
I love machine learning
```

Suppose GPT has 4 attention heads.

Possible behavior:

```text
Head 1:
focuses on adjacent words

Head 2:
connects "machine" with "learning"

Head 3:
captures phrase "machine learning" as one concept

Head 4:
tracks the subject "I"
```

Each head outputs a different vector for every token.

For the token `learning`, each head may produce:

```text
Head 1 output → local context vector

Head 2 output → relation with machine

Head 3 output → phrase-level meaning

Head 4 output → subject/context information
```

Then these outputs are combined.

---

## 16. Head Outputs

Each head produces output shape:

```text
B × T × D
```

Example:

```text
B = 2
T = 5
D = 64
```

So each head output shape is:

```text
2 × 5 × 64
```

If there are 12 heads, we get:

```text
12 outputs
each of shape 2 × 5 × 64
```

Together, these represent many different attention views.

---

## 17. Concatenating Heads

After all heads compute their outputs, GPT concatenates them.

If:

```text
H = 12
D = 64
```

then concatenation gives:

```text
12 × 64 = 768
```

So:

```text
Head 1 output: 64 dimensions
Head 2 output: 64 dimensions
Head 3 output: 64 dimensions
...
Head 12 output: 64 dimensions
```

After concatenation:

```text
Concatenated output = 768 dimensions
```

Shape:

```text
Before concat:
B × H × T × D

After rearranging and concat:
B × T × (H × D)

Since H × D = d_model:

B × T × d_model
```

Example:

```text
Before concat: 2 × 12 × 5 × 64

After concat:  2 × 5 × 768
```

---

## 18. Why Concatenation Is Needed

Each head captures a different representation.

Concatenation keeps all these views.

Example:

```text
Head 1 = local context

Head 2 = grammar

Head 3 = long-range dependency

Head 4 = entity reference
```

If we only averaged them, we might lose useful differences.

Concatenation preserves information from all heads.

Then the model learns how to combine these views using an output projection.

---

## 19. Output Projection W_O

After concatenating heads, GPT applies an output projection.

```text
MultiHeadOutput = Concat(head_1, head_2, ..., head_H) W_O
```

where:

```text
W_O shape = d_model × d_model
```

Example:

```text
W_O shape = 768 × 768
```

The output projection does two things:

```text
1. Mixes information across heads
2. Produces the final attention output with shape B × T × d_model
```

Without this projection, heads would remain separate chunks.

With `W_O`, the model learns how to combine them.

---

## 20. Why We Need W_O

After concatenation, the vector looks like:

```text
[head_1 output | head_2 output | head_3 output | ... | head_H output]
```

This contains all head outputs side by side.

But the model still needs to mix them.

For example:

```text
Head 2 may detect a pronoun reference.

Head 5 may detect the subject.

Head 8 may detect tense.
```

The output projection can combine these signals into one final representation.

Memory hook:

```text
Concatenation collects head outputs.

Output projection mixes head outputs.
```

---

## 21. Full Multi-Head Attention Flow

The complete flow is:

```text
Input X
  ↓
LayerNorm(X)
  ↓
Create Q, K, V
  ↓
Reshape Q, K, V into heads
  ↓
For each head:
    compute QK^T
    scale by sqrt(head_dim)
    apply causal mask
    softmax
    multiply by V
  ↓
Get output from each head
  ↓
Concatenate all heads
  ↓
Apply output projection W_O
  ↓
Final multi-head attention output
```

Shape flow:

```text
X:
B × T × d_model

Q, K, V:
B × T × d_model

After split into heads:
B × H × T × D

Attention scores:
B × H × T × T

Head outputs:
B × H × T × D

After concat:
B × T × d_model

After W_O:
B × T × d_model
```

---

## 22. Concrete Shape Example

Suppose:

```text
B = 2
T = 5
d_model = 768
H = 12
D = 64
```

Input:

```text
X shape = 2 × 5 × 768
```

Q, K, V projection:

```text
Q shape = 2 × 5 × 768
K shape = 2 × 5 × 768
V shape = 2 × 5 × 768
```

Reshape into heads:

```text
Q shape = 2 × 12 × 5 × 64
K shape = 2 × 12 × 5 × 64
V shape = 2 × 12 × 5 × 64
```

Attention scores:

```text
Scores shape = 2 × 12 × 5 × 5
```

Attention output per head:

```text
Head outputs shape = 2 × 12 × 5 × 64
```

Concatenate heads:

```text
Concat output shape = 2 × 5 × 768
```

Output projection:

```text
Final attention output shape = 2 × 5 × 768
```

So multi-head attention preserves the original model dimension.

---

## 23. Why Not Use One Big Head of Size 768?

A natural question is:

```text
Why not use one attention head with 768 dimensions?
```

One large head can compute attention, but it gives only one attention pattern.

Multi-head attention gives multiple independent attention patterns.

Example:

```text
One big head:
one attention distribution per token

Twelve heads:
twelve attention distributions per token
```

So the model can attend to different parts of the context for different reasons.

Memory hook:

```text
One big head = one attention map.

Many heads = many attention maps.
```

---

## 24. Does Multi-Head Attention Increase Output Dimension?

No.

The final output dimension remains:

```text
d_model
```

Example:

```text
12 heads × 64 dimensions = 768 dimensions
```

After concatenation and output projection, the output is:

```text
B × T × 768
```

Same as the input.

This is necessary because the residual connection adds the attention output back to the original input.

For residual addition, shapes must match:

```text
X + AttentionOutput
```

Both must have shape:

```text
B × T × d_model
```

---

## 25. Multi-Head Attention and Residual Connection

Inside a GPT block:

```text
A = X + MultiHeadAttention(LayerNorm(X))
```

For this addition to work:

```text
X shape = B × T × d_model

MultiHeadAttention output shape = B × T × d_model
```

Then:

```text
A shape = B × T × d_model
```

So multi-head attention must return the same shape as the input.

---

## 26. Parameter Count of Multi-Head Attention

For standard multi-head attention, the main matrices are:

```text
W_Q
W_K
W_V
W_O
```

For simplicity, each has shape:

```text
d_model × d_model
```

So total parameters:

```text
4 × d_model × d_model
```

Example:

```text
d_model = 768
```

One matrix:

```text
768 × 768 = 589,824
```

Four matrices:

```text
4 × 589,824 = 2,359,296 parameters
```

So one attention module has about:

```text
2.36 million parameters
```

for `d_model = 768`.

For:

```text
d_model = 4096
```

One matrix:

```text
4096 × 4096 = 16,777,216
```

Four matrices:

```text
4 × 16,777,216 = 67,108,864 parameters
```

So one attention module has about:

```text
67 million parameters
```

for `d_model = 4096`.

This is only the attention part of one GPT block.

The feed-forward network often has even more parameters.

---

## 27. Compute Cost of Multi-Head Attention

The expensive part of attention is the attention score matrix:

```text
QK^T
```

For sequence length `T`, each head creates:

```text
T × T
```

attention scores.

With batch and heads:

```text
B × H × T × T
```

So attention memory and compute grow roughly with:

```text
T²
```

Example:

```text
T = 1,000  → 1,000,000 scores per head

T = 2,000  → 4,000,000 scores per head

T = 4,000  → 16,000,000 scores per head
```

This is why long-context attention is expensive.

Memory hook:

```text
Attention cost grows quadratically with sequence length.
```

---

## 28. Multi-Head Attention During Training

During training, GPT processes the full sequence at once.

Example:

```text
I love machine learning
```

Multi-head attention computes all token representations in parallel.

But causal masking prevents future leakage.

So each head computes attention patterns like:

```text
I        → can attend to I

love     → can attend to I, love

machine  → can attend to I, love, machine

learning → can attend to I, love, machine, learning
```

Every head follows the same causal rule, but each head may learn different attention weights.

---

## 29. Multi-Head Attention During Inference

During inference, GPT generates one token at a time.

Prompt:

```text
I love machine
```

The model predicts:

```text
learning
```

Then:

```text
I love machine learning
```

Then it predicts the next token.

Multi-head attention is still used at every generation step.

However, without optimization, GPT would recompute attention for all previous tokens again and again.

This is why GPT implementations use **KV cache**.

KV cache stores previous keys and values so they do not need to be recomputed every time.

We will study KV cache later in detail.

---

## 30. Multi-Head Attention and KV Cache Preview

During generation, old tokens do not change.

So their K and V vectors can be stored.

At the next generation step, GPT only needs to compute Q for the new token and attend to cached K and V from previous tokens.

Simple idea:

```text
Previous K and V are cached.

New token creates new Q.

New Q attends to cached K and V.
```

This makes inference much faster.

Important:

```text
KV cache is mainly an inference optimization.
```

---

## 31. Multi-Head Attention and LoRA Preview

When we later study LoRA fine-tuning, attention projection matrices become very important.

LoRA often modifies matrices such as:

```text
W_Q

W_V

W_K

W_O
```

The idea is:

```text
Instead of updating the full attention matrix,
train small low-rank adapter matrices.
```

So understanding Q, K, V, and W_O is essential before studying LoRA.

---

## 32. Modern Variants: MHA, MQA, and GQA

Standard multi-head attention is called:

```text
MHA = Multi-Head Attention
```

In standard MHA:

```text
Each head has its own Q, K, and V projections.
```

Modern LLMs sometimes use variants to reduce memory and speed up inference.

### Multi-Query Attention

```text
MQA = Multi-Query Attention
```

In MQA:

```text
Multiple query heads share fewer key and value heads.
```

This reduces KV cache size.

### Grouped-Query Attention

```text
GQA = Grouped-Query Attention
```

In GQA:

```text
Groups of query heads share key and value heads.
```

This is a compromise between MHA and MQA.

For now, remember:

```text
MHA = full multi-head attention

MQA/GQA = more efficient attention variants used in modern LLMs
```

We will study these later when we discuss inference optimization and production LLM serving.

---

## 33. Common Confusion: Are Heads Separate Models?

No.

Attention heads are not separate models.

They are parallel subspaces inside the same layer.

They share the same input but have different learned projections.

Memory hook:

```text
Heads are parallel attention views, not separate models.
```

---

## 34. Common Confusion: Do Heads Communicate With Each Other?

Inside the attention calculation, heads operate independently.

But after the head outputs are concatenated, the output projection `W_O` mixes information across heads.

So:

```text
Before concatenation:
heads are separate

After W_O:
head information is mixed
```

---

## 35. Common Confusion: Does Each Head Look at a Different Part of the Sentence?

Not necessarily.

Each head can attend to any allowed token.

Because GPT is causal, every head follows the same rule:

```text
Only attend to previous/current tokens.
```

But the learned attention weights may differ.

So one head may focus on nearby tokens while another focuses on long-distance tokens.

---

## 36. Common Confusion: Are More Heads Always Better?

Not always.

More heads can allow more attention patterns, but they also increase computation and may become redundant.

Model design balances:

```text
number of heads
head dimension
model size
training stability
inference speed
memory usage
```

Modern architectures often tune these choices carefully.

---

## 37. Common Confusion: Does Multi-Head Attention Change Sequence Length?

No.

Multi-head attention does not change sequence length.

Input:

```text
B × T × d_model
```

Output:

```text
B × T × d_model
```

The number of tokens remains the same.

Only token representations are updated.

---

## 38. Interview-Level Answer

If an interviewer asks:

```text
What is multi-head attention in GPT?
```

You can answer:

Multi-head attention allows GPT to compute several attention patterns in parallel. The model projects token representations into Q, K, and V, then splits these projections into multiple heads. Each head performs causal masked self-attention in its own lower-dimensional subspace. The outputs from all heads are concatenated and passed through an output projection, which mixes information across heads and returns the representation to the original model dimension. This lets GPT capture different relationships such as local context, syntax, long-range dependencies, and semantic references at the same time.

---

## 39. Common Interview Follow-Up Questions

### Q1. Why does GPT use multiple attention heads?

Because different heads can learn different relationships in the same sequence.

One head may focus on nearby context, another on grammar, another on long-range dependencies, and another on entity references.

---

### Q2. What is head dimension?

Head dimension is the size of each attention head.

```text
head_dim = d_model / number_of_heads
```

Example:

```text
d_model = 768
number_of_heads = 12
head_dim = 64
```

---

### Q3. What are the shapes of Q, K, and V in multi-head attention?

After projection and reshaping:

```text
Q shape = B × H × T × D

K shape = B × H × T × D

V shape = B × H × T × D
```

where:

```text
B = batch size
H = number of heads
T = sequence length
D = head dimension
```

---

### Q4. What is the shape of the attention score matrix?

The attention score matrix has shape:

```text
B × H × T × T
```

Each head has its own `T × T` attention matrix.

---

### Q5. Why do we concatenate heads?

Each head captures a different view of the sequence.

Concatenation preserves all these views before mixing them with the output projection.

---

### Q6. What does W_O do?

`W_O` is the output projection matrix.

It mixes information across attention heads and returns the final attention output to shape:

```text
B × T × d_model
```

---

### Q7. Does multi-head attention change the tensor shape?

Internally, yes, because the model reshapes into heads.

But the final output shape remains:

```text
B × T × d_model
```

---

### Q8. Are attention heads independent?

During the attention calculation, heads operate independently.

After concatenation, the output projection mixes information across heads.

---

### Q9. Why is attention expensive for long sequences?

Because each head creates a `T × T` attention matrix.

So computation and memory grow roughly with:

```text
T²
```

---

### Q10. What is the difference between MHA, MQA, and GQA?

```text
MHA:
each query head has its own key and value heads

MQA:
many query heads share one or fewer key/value heads

GQA:
groups of query heads share key/value heads
```

MQA and GQA reduce KV cache memory and improve inference efficiency.

---

## 40. Memory Hooks

```text
Multi-head attention = many attention views in parallel.

Each head sees the same tokens but through different projections.

head_dim = d_model / number_of_heads.

Each head computes its own attention matrix.

Attention scores shape = B × H × T × T.

Each head output shape = B × T × head_dim.

Concatenation combines all heads.

W_O mixes information across heads.

Final output shape = B × T × d_model.

More heads = more possible attention patterns, but also more compute.

Attention cost grows as T².

KV cache stores K and V during inference.
```

---

## 41. Final One-Line Summary

```text
Multi-head attention lets GPT compute multiple causal attention patterns in parallel, concatenate their outputs, and mix them through an output projection to produce a richer contextual token representation.
```
