# GPT Architecture — Part 2C: Q, K, V Projections in GPT

## 1. Why This Part Matters

In the previous parts, we studied:

```text
Part 2A → Input and tensor shapes inside a GPT block
Part 2B → LayerNorm in GPT
```

Now we study what happens immediately after LayerNorm inside the attention layer:

```text
LayerNorm(X)
      ↓
Create Q, K, V
      ↓
Masked Self-Attention
```

The most important idea is:

```text
Q, K, and V are three different learned projections of the same token representations.
```

They help every token decide:

```text
What am I looking for?
What do other tokens contain?
What information should I take from them?
```

---

## 2. Where Q, K, and V Appear Inside a GPT Block

A GPT block looks like this:

```text
Input X
  ↓
LayerNorm
  ↓
Create Q, K, V
  ↓
Masked Multi-Head Self-Attention
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

Q, K, and V are not separate inputs provided by the user.

They are created inside the model from the hidden token representations.

---

## 3. Starting Point: Input X

The input to a GPT block has shape:

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
B = 1
T = 4
d_model = 768
```

Input sentence:

```text
I love machine learning
```

Then:

```text
X shape = 1 × 4 × 768
```

Ignoring the batch dimension:

```text
X shape = 4 × 768
```

Each row is one token vector:

```text
I        → vector of size 768
love     → vector of size 768
machine  → vector of size 768
learning → vector of size 768
```

So before Q, K, and V are created, each token already has a hidden representation.

---

## 4. Why Do We Need Q, K, and V?

Self-attention needs to answer three different questions.

Suppose the sentence is:

```text
The animal was tired
```

When the model processes the token:

```text
tired
```

it may need to ask:

```text
Who was tired?
```

The token:

```text
animal
```

should match that question.

Then useful information from `animal` should be passed to `tired`.

This gives us three roles:

| Component | Intuition | Simple Meaning |
|---|---|---|
| Query | What am I looking for? | Question asked by the current token |
| Key | What do I contain? | Matching signal advertised by each token |
| Value | What information do I provide? | Actual content passed forward |

So for token `tired`:

```text
Query of "tired":
Who was tired?

Key of "animal":
I am an entity that may answer this question.

Value of "animal":
Useful information about animal.
```

---

## 5. Search Engine Analogy

A useful analogy is a search engine.

```text
Query = what the user searches
Key   = searchable index or metadata of each document
Value = actual document content
```

Example:

```text
Query: best machine learning books
```

The search engine compares the query with document keys.

If a document key matches well, the document content is returned.

Similarly, in GPT:

```text
Query of current token compares with keys of previous tokens.

If a key matches well, its value contributes more to the output.
```

So attention follows this logic:

```text
Compare query with keys
        ↓
Get attention scores
        ↓
Convert scores into attention weights
        ↓
Use weights to combine values
```

---

## 6. How Q, K, and V Are Created

GPT creates Q, K, and V using learned linear projections.

Starting from input:

```text
X
```

GPT computes:

```text
Q = XW_Q

K = XW_K

V = XW_V
```

where:

```text
W_Q = query projection matrix
W_K = key projection matrix
W_V = value projection matrix
```

These matrices are trainable parameters.

They are learned during training.

---

## 7. What Does Projection Mean?

A projection here simply means:

```text
a learned linear transformation
```

Suppose a token vector has size:

```text
d_model = 768
```

GPT multiplies this vector by a learned matrix to create a new role-specific vector.

For query:

```text
token vector x
      ↓ multiply by W_Q
query vector q
```

For key:

```text
token vector x
      ↓ multiply by W_K
key vector k
```

For value:

```text
token vector x
      ↓ multiply by W_V
value vector v
```

So the same token representation is converted into three different roles.

---

## 8. Why Not Use X Directly?

A natural question is:

```text
Why not directly compare token embeddings?
Why create Q, K, and V separately?
```

Because the same token may need to play different roles.

A token may need to:

```text
1. Ask for information
2. Be matched by another token
3. Provide information if selected
```

These are different jobs.

So GPT learns different transformations:

```text
W_Q learns how to ask useful questions.

W_K learns how to create useful matching signals.

W_V learns what information should be passed forward.
```

This gives the model much more flexibility than using the same vector for everything.

---

## 9. Same Token, Three Different Vectors

Suppose we have the token:

```text
animal
```

Its hidden representation is:

```text
x_animal
```

GPT creates:

```text
q_animal = x_animal W_Q

k_animal = x_animal W_K

v_animal = x_animal W_V
```

These three vectors are not the same.

They represent different roles.

| Vector | Role |
|---|---|
| q_animal | What `animal` is looking for |
| k_animal | How `animal` can be matched by other tokens |
| v_animal | What information `animal` provides if attended to |

This is one of the most important ideas in attention.

A token does not have just one representation inside attention.

It has three role-specific representations.

---

## 10. Shape of Q, K, and V for One Attention Head

Input shape:

```text
X shape = B × T × d_model
```

For one attention head, suppose:

```text
d_k = query/key dimension
d_v = value dimension
```

Then the projection matrices have shapes:

```text
W_Q shape = d_model × d_k
W_K shape = d_model × d_k
W_V shape = d_model × d_v
```

After multiplication:

```text
Q shape = B × T × d_k
K shape = B × T × d_k
V shape = B × T × d_v
```

Usually:

```text
d_k = d_v
```

For simplicity, this is often called:

```text
head_dim
```

---

## 11. Shape Example for One Head

Suppose:

```text
B = 2
T = 5
d_model = 768
d_k = 64
d_v = 64
```

Then:

```text
X shape   = 2 × 5 × 768

W_Q shape = 768 × 64
W_K shape = 768 × 64
W_V shape = 768 × 64
```

After projection:

```text
Q shape = 2 × 5 × 64
K shape = 2 × 5 × 64
V shape = 2 × 5 × 64
```

Meaning:

```text
For every sequence in the batch,
for every token,
GPT creates a 64-dimensional query vector,
a 64-dimensional key vector,
and a 64-dimensional value vector.
```

---

## 12. GPT Uses Multi-Head Attention

In real GPT models, we do not create only one Q, K, and V set.

We create Q, K, and V for multiple attention heads.

Example:

```text
d_model = 768
number of heads = 12
head_dim = 64
```

because:

```text
768 / 12 = 64
```

Shape flow:

```text
X shape = B × T × 768
```

Project to Q:

```text
Q shape = B × T × 768
```

Then reshape into heads:

```text
Q shape = B × T × 12 × 64
```

Often frameworks rearrange this as:

```text
Q shape = B × 12 × T × 64
```

Similarly:

```text
K shape = B × 12 × T × 64

V shape = B × 12 × T × 64
```

For now, remember:

```text
Each attention head gets its own Q, K, and V subspace.
```

We will study multi-head attention in more detail in a later part.

---

## 13. Attention Scores Use Q and K

After Q and K are created, GPT computes similarity scores:

```text
Scores = QK^T
```

For one head, ignoring batch:

```text
Q shape = T × d_k
K shape = T × d_k
```

So:

```text
K^T shape = d_k × T
```

Then:

```text
QK^T shape = T × T
```

This gives one score for every pair of tokens.

Example with 4 tokens:

```text
I | love | machine | learning
```

Score matrix shape:

```text
4 × 4
```

Rows are query tokens.

Columns are key tokens.

| Query Token / Key Token | I | love | machine | learning |
|---|---:|---:|---:|---:|
| I | score | score | score | score |
| love | score | score | score | score |
| machine | score | score | score | score |
| learning | score | score | score | score |

Each row answers:

```text
For this query token, how relevant is every key token?
```

---

## 14. What Does One Dot Product Mean?

Suppose we are calculating how much `machine` should attend to `love`.

We compare:

```text
q_machine
```

with:

```text
k_love
```

using a dot product:

```text
q_machine · k_love
```

If the dot product is large:

```text
machine finds love relevant
```

If the dot product is small:

```text
machine does not find love very relevant
```

So:

```text
Query decides what it needs.
Key decides whether it matches that need.
```

---

## 15. Why Dot Product?

Dot product is a simple similarity measure.

If two vectors point in similar directions, their dot product is high.

If they point in very different directions, their dot product is low.

Example:

```text
q  = [1, 0]
k1 = [1, 0]
k2 = [0, 1]
```

Dot products:

```text
q · k1 = 1
q · k2 = 0
```

So `k1` matches the query better than `k2`.

This is how attention scores are formed.

---

## 16. Why Scale by Square Root of d_k?

The attention formula uses scaled scores:

```text
Scaled scores = QK^T / sqrt(d_k)
```

Why?

Because when vector dimensions are large, dot products can become large.

Large dot products can make softmax too sharp.

Example:

```text
Scores = [1, 2, 3]
```

Softmax gives a reasonable distribution.

But:

```text
Scores = [10, 20, 30]
```

Softmax becomes extremely peaked.

One token may get almost all the probability.

That can hurt training because gradients become unstable or too small.

So GPT divides by:

```text
sqrt(d_k)
```

This keeps attention scores in a more stable range.

Memory hook:

```text
Scaling prevents softmax from becoming too extreme.
```

---

## 17. Softmax Creates Attention Weights

After scores are computed and scaled, GPT applies softmax row-wise.

For each query token, softmax converts scores over key tokens into probabilities.

Example:

For query token `machine`, suppose the scores are:

| Key Token | Score |
|---|---:|
| I | 0.5 |
| love | 1.2 |
| machine | 2.0 |

After softmax:

| Key Token | Attention Weight |
|---|---:|
| I | 0.13 |
| love | 0.26 |
| machine | 0.61 |

The weights sum to 1.

Meaning:

```text
machine takes 13% information from I
machine takes 26% information from love
machine takes 61% information from machine itself
```

In GPT, future tokens are masked before softmax.

---

## 18. Values Provide the Actual Information

Q and K are used to decide attention weights.

V contains the information that gets mixed.

The attention operation is:

```text
Attention output = Attention weights × V
```

Think of it like this:

```text
Q and K decide how much to attend.

V decides what information is received.
```

Suppose attention weights for `machine` are:

```text
I        = 0.13
love     = 0.26
machine  = 0.61
```

Then output for `machine` is:

```text
0.13 × V_I + 0.26 × V_love + 0.61 × V_machine
```

So the new representation for `machine` becomes a weighted mixture of value vectors.

---

## 19. Tiny Numerical Example

Let us use a tiny example with 3 tokens:

```text
I | love | ML
```

Assume one attention head with:

```text
d_k = 2
d_v = 2
```

Suppose the query for `ML` is:

```text
q_ML = [1, 1]
```

Keys:

```text
k_I    = [1, 0]
k_love = [1, 1]
k_ML   = [0, 1]
```

Values:

```text
v_I    = [2, 0]
v_love = [0, 3]
v_ML   = [1, 1]
```

Now compute dot products:

```text
q_ML · k_I    = [1, 1] · [1, 0] = 1

q_ML · k_love = [1, 1] · [1, 1] = 2

q_ML · k_ML   = [1, 1] · [0, 1] = 1
```

Scores:

```text
[1, 2, 1]
```

Softmax roughly gives:

```text
[0.21, 0.58, 0.21]
```

So `ML` attends most to `love`.

Now combine the value vectors:

```text
output_ML = 0.21 × [2, 0] + 0.58 × [0, 3] + 0.21 × [1, 1]
```

Calculate each part:

```text
0.21 × [2, 0] = [0.42, 0]

0.58 × [0, 3] = [0, 1.74]

0.21 × [1, 1] = [0.21, 0.21]
```

Add them:

```text
output_ML = [0.63, 1.95]
```

So after attention, the token `ML` has a new vector:

```text
[0.63, 1.95]
```

This vector contains mixed information from:

```text
I, love, and ML
```

In GPT, it would only mix from allowed previous and current tokens.

---

## 20. In GPT, Q, K, and V Come From the Same Sequence

GPT uses self-attention.

That means Q, K, and V all come from the same input sequence.

```text
Q = XW_Q
K = XW_K
V = XW_V
```

This is why it is called **self-attention**:

```text
Tokens attend to other tokens in the same sequence.
```

Example prompt:

```text
Translate English to French:
English: I love machine learning
French:
```

The generated French tokens can attend back to the English sentence because the English sentence is part of the same prompt context.

No separate encoder is required.

---

## 21. Difference Between Self-Attention and Cross-Attention

In encoder-decoder Transformers, cross-attention is different.

For cross-attention:

```text
Q comes from decoder.

K comes from encoder.

V comes from encoder.
```

But in GPT self-attention:

```text
Q comes from the GPT input sequence.

K comes from the GPT input sequence.

V comes from the GPT input sequence.
```

Comparison:

| Attention Type | Q Source | K Source | V Source |
|---|---|---|---|
| GPT self-attention | Same sequence | Same sequence | Same sequence |
| Encoder-decoder cross-attention | Decoder | Encoder | Encoder |

Memory hook:

```text
Self-attention:
Q, K, and V come from the same sequence.

Cross-attention:
Q comes from decoder.
K and V come from encoder.
```

---

## 22. Q, K, and V Are Learned During Training

The matrices:

```text
W_Q
W_K
W_V
```

are trainable parameters.

During pretraining, GPT learns how to create useful query, key, and value vectors.

During full fine-tuning, these matrices can be updated.

During LoRA fine-tuning, we often inject trainable low-rank matrices into selected projection matrices such as:

```text
W_Q
W_K
W_V
W_O
```

Very commonly, LoRA is applied to:

```text
query projections
value projections
```

or sometimes all attention projections.

This will become very important when we study fine-tuning.

---

## 23. Parameter Count for Q, K, and V

Suppose:

```text
d_model = 768
```

For simplicity, assume:

```text
W_Q shape = 768 × 768
W_K shape = 768 × 768
W_V shape = 768 × 768
```

Each matrix has:

```text
768 × 768 = 589,824 parameters
```

All three together:

```text
3 × 589,824 = 1,769,472 parameters
```

So Q, K, and V projections contain many parameters.

For a larger model:

```text
d_model = 4096
```

Each matrix has:

```text
4096 × 4096 = 16,777,216 parameters
```

All three:

```text
3 × 16,777,216 = 50,331,648 parameters
```

That is just Q, K, and V in one layer.

This is why LLMs reach billions of parameters when many layers are stacked.

---

## 24. Implementation Detail: Combined QKV Projection

In many implementations, GPT does not perform three separate matrix multiplications.

Instead of:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

it may use one combined projection:

```text
QKV = XW_QKV
```

where:

```text
W_QKV shape = d_model × 3d_model
```

Then the result is split into Q, K, and V.

Example:

```text
X shape      = B × T × 768
W_QKV shape  = 768 × 2304
QKV shape    = B × T × 2304
```

Then split:

```text
Q shape = B × T × 768
K shape = B × T × 768
V shape = B × T × 768
```

This is computationally efficient.

Conceptually, it is still the same as three projections.

---

## 25. What QKV Projection Learns Conceptually

During training, the model learns useful attention behavior.

For example, one attention head may learn:

```text
When generating a verb, look for the subject.
```

Another head may learn:

```text
When resolving a pronoun, look for a previous noun.
```

Another head may learn:

```text
When writing code, look for the matching function or variable name.
```

These behaviors emerge because `W_Q` and `W_K` learn compatible matching spaces.

If a query needs subject information, it learns to align with keys of subject-like tokens.

Then `W_V` learns what information from those tokens should be passed forward.

---

## 26. Common Confusion: Are Q, K, and V Fixed Embeddings?

No.

Q, K, and V are not fixed embeddings.

They are computed dynamically from the current hidden states.

At layer 1:

```text
Q, K, and V are created from early token representations.
```

At layer 10:

```text
Q, K, and V are created from deeper contextual representations.
```

So Q, K, and V change from layer to layer.

They also change depending on context.

The token `bank` will produce different Q, K, and V representations in:

```text
river bank
```

versus:

```text
bank loan
```

because its hidden state entering that layer is different.

---

## 27. Common Confusion: Does Every Layer Have Its Own W_Q, W_K, and W_V?

Yes.

Each GPT block has its own attention projection matrices.

Example:

```text
Layer 1 has W_Q1, W_K1, W_V1

Layer 2 has W_Q2, W_K2, W_V2

Layer 3 has W_Q3, W_K3, W_V3
```

They are not shared across layers in standard GPT-style models.

This allows each layer to learn different kinds of relationships.

---

## 28. Common Confusion: Does Every Head Have Separate QKV Matrices?

Conceptually, yes.

Each attention head has its own projection subspace.

Implementation-wise, the model may use one large combined projection matrix and then split it into heads.

Conceptual view:

```text
Head 1 has its own Q, K, and V projections.

Head 2 has its own Q, K, and V projections.

Head 3 has its own Q, K, and V projections.
```

Implementation view:

```text
One large matrix creates all heads together.
```

Both views are correct.

---

## 29. Why Are Values Separate From Keys?

Another natural question is:

```text
If keys decide which tokens are relevant, why not use keys as values?
```

Because matching and information transfer are different tasks.

Example:

```text
Document title helps decide whether the document is relevant.

Document body contains the useful information.
```

Similarly:

```text
Key   = useful for matching
Value = useful for content transfer
```

A token may advertise one kind of signal for matching but provide different information once selected.

That is why K and V are separate.

---

## 30. QKV and Causal Masking

Q, K, and V themselves do not enforce causality.

Causality is enforced after computing attention scores.

The process is:

```text
1. Create Q, K, and V
2. Compute scores using QK^T
3. Apply causal mask
4. Apply softmax
5. Multiply attention weights by V
```

So the mask is not inside `W_Q`, `W_K`, or `W_V`.

The mask is applied to the score matrix before softmax.

We will study this deeply in the next part.

---

## 31. Full QKV Flow Summary

For one GPT block:

```text
Input X
  ↓
LayerNorm(X)
  ↓
Linear projection W_Q → Q
Linear projection W_K → K
Linear projection W_V → V
  ↓
QK^T gives attention scores
  ↓
Scale by sqrt(d_k)
  ↓
Apply causal mask
  ↓
Softmax gives attention weights
  ↓
Weighted sum of V gives attention output
```

So Q, K, and V are the preparation step for attention.

---

## 32. Interview-Level Answer

If an interviewer asks:

```text
What are Q, K, and V in GPT attention?
```

You can answer:

In GPT, Query, Key, and Value are learned linear projections of the token hidden states. The query represents what a token is looking for, the key represents what each token offers for matching, and the value represents the information passed forward if that token is attended to.

Attention scores are computed using the dot product between queries and keys, scaled by the square root of the key dimension, masked to prevent attending to future tokens, and passed through softmax. The resulting attention weights are used to take a weighted sum of the value vectors.

---

## 33. Common Interview Follow-Up Questions

### Q1. Why do we need separate Q, K, and V?

Because asking, matching, and information transfer are different roles.

```text
Q = what the token is looking for
K = how a token can be matched
V = what information the token provides
```

Separate projections give the model flexibility.

---

### Q2. What is the shape of Q, K, and V?

For one head:

```text
Input X shape = B × T × d_model

Q shape = B × T × d_k
K shape = B × T × d_k
V shape = B × T × d_v
```

For multi-head attention, these are reshaped into:

```text
B × num_heads × T × head_dim
```

---

### Q3. What is QK^T?

`QK^T` is the matrix of attention scores.

It compares every query token with every key token.

For sequence length `T`, the score matrix has shape:

```text
T × T
```

Each row says how much one token should attend to all other tokens.

---

### Q4. Why do we divide by sqrt(d_k)?

We divide by `sqrt(d_k)` because dot products can become large when vector dimension is high.

Large scores make softmax too sharp.

Scaling keeps scores stable and improves training.

---

### Q5. What is the role of V?

V contains the actual information that gets mixed and passed forward.

Q and K decide attention weights.

V provides the content.

---

### Q6. Are Q, K, and V learned?

The projection matrices `W_Q`, `W_K`, and `W_V` are learned parameters.

The actual Q, K, and V vectors are dynamically computed from the current hidden states during each forward pass.

---

### Q7. Are Q, K, and V the same in every layer?

No.

Each GPT block has its own Q, K, and V projection matrices.

So different layers can learn different attention behaviors.

---

### Q8. Are Q, K, and V fixed for a token?

No.

They depend on the token's current hidden state.

Since hidden states change by layer and context, Q, K, and V also change by layer and context.

---

### Q9. Is causal masking part of QKV projection?

No.

Q, K, and V are created first.

The causal mask is applied later to the attention score matrix before softmax.

---

## 34. Memory Hooks

```text
Q = what am I looking for?

K = what do I advertise for matching?

V = what information do I provide?

QK^T = relevance scores.

Softmax = attention weights.

Attention weights × V = mixed information.

Q and K decide where to look.

V decides what information to take.

In GPT self-attention, Q, K, and V come from the same sequence.

QKV projection matrices are learned parameters.

Q, K, and V are dynamic, not fixed.

Causal mask is applied after QK^T and before softmax.
```

---

## 35. Final One-Line Summary

```text
Q, K, and V are learned role-specific projections of token representations: Q asks what to look for, K decides what matches, and V provides the information that gets mixed into the next representation.
```
