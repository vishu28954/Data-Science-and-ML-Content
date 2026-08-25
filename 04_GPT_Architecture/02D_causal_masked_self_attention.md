# GPT Architecture — Part 2D: Causal Masked Self-Attention Step by Step

## 1. Why This Part Matters

In the previous part, we studied Q, K, and V.

```text
Q = Query
K = Key
V = Value
```

Now we will study how GPT uses Q, K, and V to perform **causal masked self-attention**.

The main idea is:

```text
Causal masked self-attention allows each token to attend only to itself and previous tokens.
```

This is the key mechanism that makes GPT a left-to-right text generation model.

---

## 2. Where Masked Self-Attention Appears Inside a GPT Block

A GPT block looks like this:

```text
Input X
  ↓
LayerNorm
  ↓
Create Q, K, V
  ↓
Causal Masked Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output Y
```

In this part, we focus on this section:

```text
Q, K, V
  ↓
Attention Scores
  ↓
Causal Mask
  ↓
Softmax
  ↓
Weighted Sum of Values
  ↓
Attention Output
```

---

## 3. What Is Self-Attention?

Self-attention means:

```text
Tokens in the same sequence attend to each other.
```

Example sequence:

```text
I love machine learning
```

Each token creates:

```text
Query
Key
Value
```

Then each token asks:

```text
Which previous tokens are useful for understanding me?
```

For example, the token `learning` may attend to:

```text
I
love
machine
learning
```

But because GPT is causal, `machine` cannot attend to `learning`.

---

## 4. Why GPT Needs Causal Masking

GPT is trained using next-token prediction.

Example:

```text
I love machine learning
```

The training task is:

```text
I             → predict love
I love        → predict machine
I love machine → predict learning
```

When GPT is predicting `machine`, it should not be allowed to see `machine` or `learning` as future answers.

If it could see future tokens, it would cheat.

So GPT uses a causal mask.

Memory hook:

```text
Causal mask = no looking ahead.
```

---

## 5. Encoder Attention vs GPT Attention

### Encoder-style attention

In an encoder model like BERT, every token can attend to every other token.

For:

```text
I love machine learning
```

The token `love` can attend to:

```text
I, love, machine, learning
```

This is bidirectional attention.

---

### GPT-style attention

In GPT, attention is causal.

The token `love` can attend only to:

```text
I, love
```

It cannot attend to:

```text
machine, learning
```

because those are future tokens.

---

## 6. Allowed Attention Table

For the sequence:

```text
I | love | machine | learning
```

The allowed attention pattern is:

| Query Token | Can Attend To |
|---|---|
| I | I |
| love | I, love |
| machine | I, love, machine |
| learning | I, love, machine, learning |

So token position `t` can attend only to positions:

```text
1, 2, ..., t
```

It cannot attend to positions greater than `t`.

---

## 7. Attention Score Matrix

After Q and K are created, GPT computes attention scores.

Formula:

```text
Scores = QK^T
```

Usually, GPT uses scaled scores:

```text
Scores = QK^T / sqrt(d_k)
```

where:

```text
d_k = key/query dimension
```

If the sequence length is:

```text
T = 4
```

then the attention score matrix has shape:

```text
T × T = 4 × 4
```

For:

```text
I | love | machine | learning
```

the score matrix looks like this:

| Query Token / Key Token | I | love | machine | learning |
|---|---:|---:|---:|---:|
| I | score | score | score | score |
| love | score | score | score | score |
| machine | score | score | score | score |
| learning | score | score | score | score |

Rows are query tokens.

Columns are key tokens.

Each row answers:

```text
For this query token, how much should it attend to each key token?
```

---

## 8. Why the Unmasked Score Matrix Is a Problem

Without masking, every token can attend to every other token.

For example, the row for `love` could attend to:

```text
I
love
machine
learning
```

But in GPT, `machine` and `learning` are future tokens relative to `love`.

That would leak future information.

So before softmax, GPT blocks future positions.

---

## 9. Causal Mask Matrix

For 4 tokens:

```text
I | love | machine | learning
```

the causal mask allows only the lower-triangular part of the attention matrix.

Allowed and blocked pattern:

| Query Token / Key Token | I | love | machine | learning |
|---|---:|---:|---:|---:|
| I | allowed | blocked | blocked | blocked |
| love | allowed | allowed | blocked | blocked |
| machine | allowed | allowed | allowed | blocked |
| learning | allowed | allowed | allowed | allowed |

In numeric form, the mask is often represented like this:

```text
Mask =

[
  [0,  -inf, -inf, -inf],
  [0,   0,   -inf, -inf],
  [0,   0,    0,   -inf],
  [0,   0,    0,    0  ]
]
```

Meaning:

```text
0    = allowed position
-inf = blocked future position
```

---

## 10. Applying the Mask

The mask is added to the attention scores before softmax.

```text
MaskedScores = Scores + Mask
```

Example:

Suppose the raw score row for token `love` is:

```text
[2.0, 4.0, 6.0, 8.0]
```

This row means `love` is comparing itself against:

```text
I, love, machine, learning
```

But `machine` and `learning` are future tokens.

So after applying the causal mask:

```text
[2.0, 4.0, -inf, -inf]
```

Now `love` can only attend to:

```text
I, love
```

---

## 11. Why Does Negative Infinity Become Zero After Softmax?

Softmax converts scores into probabilities.

Very simply:

```text
higher score → higher probability
lower score  → lower probability
-inf         → zero probability
```

So if a masked score is:

```text
-inf
```

then after softmax it becomes:

```text
0
```

This means the model gives zero attention to future tokens.

Example:

```text
Scores after mask:
[2.0, 4.0, -inf, -inf]
```

After softmax:

```text
[0.12, 0.88, 0.00, 0.00]
```

The future tokens receive zero attention.

---

## 12. Full Masked Attention Formula

The masked self-attention operation can be written as:

```text
Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + Mask) V
```

Step by step:

```text
1. Compute Q, K, V
2. Compute scores using QK^T
3. Scale scores by sqrt(d_k)
4. Add causal mask
5. Apply softmax
6. Multiply attention weights by V
7. Get attention output
```

---

## 13. Step-by-Step Example With 4 Tokens

Sequence:

```text
I | love | machine | learning
```

Suppose we have scaled attention scores:

```text
Scores =

[
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [1, 3, 5, 7],
  [2, 4, 6, 8]
]
```

Without a mask, every row can use every column.

Now apply causal mask:

```text
Mask =

[
  [0, -inf, -inf, -inf],
  [0, 0,    -inf, -inf],
  [0, 0,     0,   -inf],
  [0, 0,     0,    0  ]
]
```

Masked scores:

```text
MaskedScores =

[
  [1, -inf, -inf, -inf],
  [2, 3,    -inf, -inf],
  [1, 3,     5,   -inf],
  [2, 4,     6,    8  ]
]
```

Now softmax is applied row-wise.

The attention weights become something like:

```text
AttentionWeights =

[
  [1.00, 0.00, 0.00, 0.00],
  [0.27, 0.73, 0.00, 0.00],
  [0.02, 0.12, 0.86, 0.00],
  [0.00, 0.02, 0.12, 0.86]
]
```

The exact numbers depend on the score values, but the key idea is:

```text
All future-token probabilities are zero.
```

---

## 14. What Each Row Means

For:

```text
I | love | machine | learning
```

The attention weight matrix rows mean:

### Row 1: token `I`

```text
I can attend only to I.
```

So:

```text
[1.00, 0.00, 0.00, 0.00]
```

---

### Row 2: token `love`

```text
love can attend to I and love.
```

So future tokens get zero:

```text
[0.27, 0.73, 0.00, 0.00]
```

---

### Row 3: token `machine`

```text
machine can attend to I, love, and machine.
```

So:

```text
[0.02, 0.12, 0.86, 0.00]
```

---

### Row 4: token `learning`

```text
learning can attend to all previous tokens and itself.
```

So:

```text
[0.00, 0.02, 0.12, 0.86]
```

---

## 15. Multiplying Attention Weights by V

After attention weights are created, GPT multiplies them by the value matrix.

```text
Output = AttentionWeights × V
```

For each token, the output is a weighted sum of value vectors.

Example for token `machine`:

```text
machine_output =
  weight_to_I       × V_I
+ weight_to_love    × V_love
+ weight_to_machine × V_machine
+ weight_to_learning × V_learning
```

But because `learning` is future:

```text
weight_to_learning = 0
```

So:

```text
machine_output =
  weight_to_I       × V_I
+ weight_to_love    × V_love
+ weight_to_machine × V_machine
```

This means `machine` receives information only from allowed previous/current tokens.

---

## 16. Complete Flow for One Token

Let us track the token:

```text
machine
```

Sequence:

```text
I | love | machine | learning
```

For token `machine`:

### Step 1: Create query

```text
q_machine
```

This represents what `machine` is looking for.

### Step 2: Compare with keys

```text
q_machine · k_I
q_machine · k_love
q_machine · k_machine
q_machine · k_learning
```

### Step 3: Apply causal mask

Because `learning` is future:

```text
q_machine · k_learning → blocked
```

### Step 4: Softmax

Softmax gives attention weights only over:

```text
I, love, machine
```

### Step 5: Weighted sum of values

```text
output_machine =
  a_I × V_I
+ a_love × V_love
+ a_machine × V_machine
```

So `machine` gets contextual information from previous tokens without seeing the future.

---

## 17. Tensor Shapes in Masked Attention

For multi-head attention, common shapes are:

```text
B = batch size
H = number of attention heads
T = sequence length
D = head dimension
```

After Q, K, V projection and reshaping:

```text
Q shape = B × H × T × D
K shape = B × H × T × D
V shape = B × H × T × D
```

Attention scores:

```text
Scores = QK^T
```

Shape:

```text
Scores shape = B × H × T × T
```

Why?

For every batch item and every head:

```text
each query token compares with every key token
```

So we get a `T × T` attention matrix per head.

After softmax:

```text
AttentionWeights shape = B × H × T × T
```

Then:

```text
Output = AttentionWeights × V
```

Shape:

```text
Output shape = B × H × T × D
```

Then heads are combined back to:

```text
B × T × d_model
```

---

## 18. Why the Attention Matrix Is T × T

If a sequence has `T` tokens, every token may compare with every other token.

So:

```text
number of query positions = T
number of key positions   = T
```

Therefore:

```text
attention score matrix = T × T
```

Example:

```text
T = 4
```

Score matrix:

```text
4 × 4
```

Example:

```text
T = 1024
```

Score matrix:

```text
1024 × 1024
```

This is why attention can become expensive for long sequences.

---

## 19. Time and Memory Cost of Attention

The attention score matrix has shape:

```text
T × T
```

So the cost grows roughly like:

```text
T²
```

This means if sequence length doubles:

```text
attention cost becomes about 4 times larger
```

Examples:

```text
T = 1,000  → about 1,000,000 attention scores per head
T = 2,000  → about 4,000,000 attention scores per head
T = 4,000  → about 16,000,000 attention scores per head
```

This is why long-context models are expensive.

Memory hook:

```text
Self-attention is powerful but expensive because every token compares with every other token.
```

---

## 20. How GPT Trains in Parallel Without Cheating

This is one of the most important ideas.

During training, GPT receives the full sequence:

```text
I love machine learning
```

But the causal mask prevents future information leakage.

So the model can compute predictions for all positions in one forward pass:

```text
I       → predict love
love    → predict machine
machine → predict learning
```

Even though the full sentence is present, each position can only use previous/current tokens.

So training is:

```text
parallel over positions
```

but still:

```text
causal and non-cheating
```

Memory hook:

```text
Training sees the full sequence, but the mask hides the future.
```

---

## 21. Why Inference Is Still Sequential

During inference, future tokens do not exist yet.

Prompt:

```text
I love machine
```

GPT predicts:

```text
learning
```

Then `learning` is appended:

```text
I love machine learning
```

Then GPT predicts the next token.

So inference happens one token at a time.

Important difference:

```text
Training:
Full target sequence is available.
Causal mask enables parallel training.

Inference:
Future tokens do not exist.
Generation must happen step by step.
```

---

## 22. Causal Mask vs Padding Mask

There are two common masks in Transformer models.

### Causal Mask

Purpose:

```text
block future tokens
```

Used in:

```text
GPT-style decoder-only models
Transformer decoders
```

Example:

```text
love cannot attend to machine or learning.
```

---

### Padding Mask

Purpose:

```text
block padding tokens
```

Example sequence:

```text
I love cats <pad> <pad>
```

The model should not attend to:

```text
<pad>
```

So a padding mask blocks those positions.

---

### Combined Mask

In practice, GPT-style models may use:

```text
causal mask + padding mask
```

Causal mask blocks future tokens.

Padding mask blocks meaningless padding tokens.

---

## 23. Why Causal Masking Is Essential for Next-Token Prediction

GPT learns:

```text
Predict the next token from previous tokens.
```

Example:

```text
The capital of France is Paris
```

When predicting `Paris`, the model should only see:

```text
The capital of France is
```

It should not see:

```text
Paris
```

Otherwise, the model would simply copy the answer.

So causal masking preserves the true learning objective.

---

## 24. What Happens If We Remove the Causal Mask?

If we remove the causal mask during GPT training, then every token can see future tokens.

For example:

```text
I love machine learning
```

When predicting `machine`, the model could attend to:

```text
machine
learning
```

This leaks the answer.

The model may achieve low training loss but fail at real generation, because during inference future tokens are not available.

So without causal masking:

```text
training and inference become mismatched
```

This is a major problem.

---

## 25. Why GPT Can Still Use Earlier Tokens Fully

Causal masking does not mean GPT has poor context.

It means GPT uses only valid context.

For the last token in a prompt:

```text
I love machine
```

the last token `machine` can attend to:

```text
I, love, machine
```

So the final token representation contains information from the whole prompt.

This is why during inference we use the final token's output to predict the next token.

---

## 26. Attention Mask as a Lower-Triangular Matrix

A causal mask is often called a lower-triangular mask.

For 5 tokens:

```text
[
  [allowed, blocked, blocked, blocked, blocked],
  [allowed, allowed, blocked, blocked, blocked],
  [allowed, allowed, allowed, blocked, blocked],
  [allowed, allowed, allowed, allowed, blocked],
  [allowed, allowed, allowed, allowed, allowed]
]
```

The allowed region is the lower triangle of the matrix.

Memory hook:

```text
Lower triangle = allowed past and current tokens.
Upper triangle = blocked future tokens.
```

---

## 27. Masking and Autoregressive Probability

GPT models the probability of a sequence as a product of next-token probabilities.

For a sequence:

```text
x1, x2, x3, ..., xT
```

GPT models:

```text
P(x1, x2, ..., xT)
=
P(x1)
× P(x2 | x1)
× P(x3 | x1, x2)
× ...
× P(xT | x1, x2, ..., xT-1)
```

Causal masking enforces this structure.

At position `t`, the model can only use:

```text
x1, x2, ..., xt
```

to predict:

```text
x(t+1)
```

So causal masking matches the autoregressive training objective.

---

## 28. How This Relates to Cross-Entropy Loss

During training, GPT produces logits at every position.

Example:

```text
Input:
I | love | machine

Targets:
love | machine | learning
```

For each position, GPT predicts a probability distribution over the vocabulary.

Then cross-entropy loss checks:

```text
Did GPT assign high probability to the correct next token?
```

Causal mask ensures the prediction was made using only valid context.

So the training logic is:

```text
Causal mask prevents cheating.

Cross-entropy punishes low probability on the correct next token.
```

Both are required.

---

## 29. Tiny Full Example

Text:

```text
I love ML
```

Tokens:

```text
1: I
2: love
3: ML
```

Training targets:

```text
I    → love
love → ML
```

Attention allowed:

| Token | Can Attend To |
|---|---|
| I | I |
| love | I, love |
| ML | I, love, ML |

Prediction:

```text
hidden state at I    → predicts love
hidden state at love → predicts ML
```

The causal mask guarantees:

```text
I cannot see love while predicting love.

love cannot see ML while predicting ML.
```

That is why GPT learns real next-token prediction.

---

## 30. Common Confusion: Can a Token Attend to Itself?

Yes.

In causal self-attention, a token can attend to:

```text
previous tokens + itself
```

So token `machine` can attend to:

```text
I, love, machine
```

Why include itself?

Because the current token's own identity is also important for computing its representation.

---

## 31. Common Confusion: Does the Mask Remove Future Tokens From the Input?

No.

During training, the full sequence is still passed into the model.

The mask does not remove tokens from the input tensor.

It only blocks attention from looking at future positions.

So:

```text
Full sequence is present.
Future attention links are blocked.
```

This allows parallel training.

---

## 32. Common Confusion: Is the Mask Learned?

No.

The causal mask is not learned.

It is a fixed rule based on token positions.

For GPT:

```text
token t cannot attend to tokens after t
```

This rule is built into the attention computation.

The model learns Q, K, V projections.

But the causal mask itself is fixed.

---

## 33. Common Confusion: Is Causal Masking Used During Inference?

Yes, conceptually.

During inference, future tokens do not exist yet, so there are no future tokens to attend to.

But implementations still respect causal attention.

When using KV cache, the model only attends to cached previous keys and values plus the current token.

So the causal constraint remains.

---

## 34. Common Confusion: Why Does the Last Token Matter During Inference?

Prompt:

```text
I love machine
```

The last token `machine` can attend to:

```text
I, love, machine
```

So its hidden state has information from the full prompt.

That hidden state is projected into vocabulary logits to predict the next token.

Therefore:

```text
During inference, the last token's output predicts the next token.
```

---

## 35. Interview-Level Answer

If an interviewer asks:

```text
What is causal masked self-attention in GPT?
```

You can answer:

Causal masked self-attention is the attention mechanism used in GPT-style decoder-only Transformers. It allows each token to attend only to itself and previous tokens, while blocking future tokens. GPT first computes attention scores using QK^T, scales them by the square root of the key dimension, adds a causal mask that assigns very negative values to future positions, and then applies softmax. Future tokens receive zero attention weight. This prevents information leakage during next-token prediction and allows GPT to train on full sequences in parallel while preserving autoregressive behavior.

---

## 36. Common Interview Follow-Up Questions

### Q1. Why does GPT need a causal mask?

GPT needs a causal mask because it is trained to predict the next token using only previous tokens. Without the mask, the model could see future tokens and cheat during training.

---

### Q2. Where is the mask applied?

The causal mask is applied to the attention score matrix after QK^T is computed and before softmax.

```text
QK^T
  ↓
scale
  ↓
add mask
  ↓
softmax
```

---

### Q3. What values are used in the mask?

Allowed positions usually get:

```text
0
```

Blocked future positions get:

```text
-inf
```

or a very large negative number.

After softmax, those blocked positions become zero probability.

---

### Q4. Does causal masking make training sequential?

No.

Training can still be parallel because the full sequence is processed at once.

The causal mask prevents each position from attending to future tokens.

---

### Q5. Why is inference sequential?

Inference is sequential because future tokens do not exist yet.

GPT must generate one token, append it to the context, and then generate the next token.

---

### Q6. What is the shape of the attention score matrix?

For sequence length `T`, the attention score matrix has shape:

```text
T × T
```

With batch and heads:

```text
B × H × T × T
```

---

### Q7. Why is attention expensive for long sequences?

Because the attention matrix is `T × T`.

So memory and computation grow roughly with:

```text
T²
```

Longer context lengths make attention much more expensive.

---

### Q8. What is the difference between causal mask and padding mask?

A causal mask blocks future tokens.

A padding mask blocks meaningless padding tokens.

In practice, both masks may be combined.

---

### Q9. Can a token attend to itself?

Yes.

In GPT, each token can attend to previous tokens and itself.

---

### Q10. Is the causal mask learned?

No.

The causal mask is fixed based on token positions.

The attention weights are learned dynamically through Q, K, and V, but the mask rule itself is fixed.

---

## 37. Memory Hooks

```text
Causal mask = no looking ahead.

GPT attention = look left only.

QK^T gives attention scores.

Mask blocks future scores before softmax.

-inf becomes zero after softmax.

Attention weights decide how much information to take.

V provides the actual information.

Training sees full sequence but mask hides future tokens.

Training is parallel.

Inference is sequential.

Attention cost grows as T².
```

---

## 38. Final One-Line Summary

```text
Causal masked self-attention lets GPT process full sequences in parallel during training while ensuring each token can only use previous and current tokens, preserving next-token prediction.
```
