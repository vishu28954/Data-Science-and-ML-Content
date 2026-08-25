# GPT Architecture — Part 2B: LayerNorm in GPT

## 1. Why This Part Matters

In the previous part, we studied the input and output tensor shape inside a GPT block.

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

Now we focus on one very important component inside a GPT block:

```text
LayerNorm
```

LayerNorm is used to make training stable.

The main idea is:

```text
LayerNorm stabilizes each token vector before it goes into attention and the feed-forward network.
```

---

## 2. Where LayerNorm Appears Inside a GPT Block

A modern GPT block usually looks like this:

```text
Input X
  ↓
LayerNorm
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

In simple equation form:

```text
A = X + MaskedMHA(LayerNorm(X))

Y = A + FFN(LayerNorm(A))
```

where:

```text
X = input to the GPT block
A = output after attention and residual connection
Y = final output of the GPT block
```

This is called **Pre-LayerNorm** because LayerNorm is applied before the major sublayers.

The two major sublayers are:

```text
1. Masked Multi-Head Self-Attention
2. Feed-Forward Network
```

---

## 3. Why Do We Need LayerNorm?

Deep neural networks are difficult to train because values can become unstable as they pass through many layers.

In GPT, token vectors pass through many blocks:

```text
Embeddings
  ↓
GPT Block 1
  ↓
GPT Block 2
  ↓
GPT Block 3
  ↓
...
  ↓
GPT Block N
```

If the values inside token vectors become too large, gradients can become unstable.

If the values become too small, gradients can become weak.

Example of very large activation values:

```text
[100, -80, 45, 200, -150, 90]
```

Example of very tiny activation values:

```text
[0.001, -0.002, 0.0005, 0.003]
```

Both can make training difficult.

LayerNorm helps keep the values in a stable range.

Memory hook:

```text
LayerNorm keeps token representations stable.
```

---

## 4. What Does LayerNorm Normalize?

This is the most important concept.

Suppose the input shape is:

```text
B × T × d_model
```

Example:

```text
B = 2
T = 4
d_model = 6
```

So the tensor shape is:

```text
2 × 4 × 6
```

This means:

```text
2 sequences
4 tokens per sequence
6 numbers per token vector
```

LayerNorm normalizes each token vector independently across its hidden dimensions.

For one token vector:

```text
x = [x1, x2, x3, x4, x5, x6]
```

LayerNorm calculates the mean and variance across these 6 values.

Important:

```text
LayerNorm does not normalize across the batch.

LayerNorm does not normalize across all tokens.

LayerNorm normalizes one token vector at a time.
```

Memory hook:

```text
LayerNorm normalizes across d_model.
```

---

## 5. LayerNorm Shape Behavior

LayerNorm changes the values inside the tensor, but it does not change the shape.

Input shape:

```text
B × T × d_model
```

Output shape:

```text
B × T × d_model
```

Example:

```text
Input shape:  2 × 4 × 768
Output shape: 2 × 4 × 768
```

So LayerNorm is a shape-preserving operation.

---

## 6. LayerNorm Formula

For one token vector:

```text
x = [x1, x2, ..., xd]
```

where:

```text
d = d_model
```

LayerNorm first calculates the mean:

```text
mean = average of all values in the token vector
```

Mathematically:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d}x_j
$$

Then it calculates the variance:

$$
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu)^2
$$

Then each dimension is normalized:

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Finally, LayerNorm applies learnable scale and shift parameters:

$$
y_j = \gamma_j \hat{x}_j + \beta_j
$$

So the full LayerNorm operation is:

$$
LayerNorm(x)_j = \gamma_j \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_j
$$

where:

```text
mu       = mean of the token vector
sigma^2  = variance of the token vector
epsilon  = small value for numerical stability
gamma    = learnable scale parameter
beta     = learnable shift parameter
```

---

## 7. Why Do We Subtract the Mean?

Suppose one token vector is:

```text
x = [2, 4, 6]
```

Mean:

```text
mean = (2 + 4 + 6) / 3 = 4
```

Subtract the mean:

```text
[2 - 4, 4 - 4, 6 - 4]
```

Result:

```text
[-2, 0, 2]
```

Now the values are centered around zero.

This helps neural networks train more smoothly because the activations are not shifted too far in one direction.

Memory hook:

```text
Subtracting the mean centers the token vector around zero.
```

---

## 8. Why Do We Divide by Standard Deviation?

After subtracting the mean, values may still have very different scales.

Example 1:

```text
[-2, 0, 2]
```

Example 2:

```text
[-200, 0, 200]
```

Both are centered around zero, but the second one has a much larger spread.

So LayerNorm divides by the standard deviation.

This makes the token vector have approximately:

```text
mean ≈ 0
variance ≈ 1
```

Memory hook:

```text
Dividing by standard deviation controls the scale.
```

---

## 9. Numerical Example

Take a small token vector:

```text
x = [2, 4, 6]
```

### Step 1: Calculate mean

```text
mean = (2 + 4 + 6) / 3
mean = 4
```

### Step 2: Subtract mean

```text
x - mean = [2 - 4, 4 - 4, 6 - 4]
x - mean = [-2, 0, 2]
```

### Step 3: Calculate variance

```text
variance = ((-2)^2 + 0^2 + 2^2) / 3
variance = (4 + 0 + 4) / 3
variance = 8 / 3
variance ≈ 2.67
```

### Step 4: Calculate standard deviation

```text
standard deviation = sqrt(2.67)
standard deviation ≈ 1.63
```

### Step 5: Normalize

```text
[-2 / 1.63, 0 / 1.63, 2 / 1.63]
```

Result:

```text
[-1.23, 0, 1.23]
```

So:

```text
[2, 4, 6] → [-1.23, 0, 1.23]
```

This is the normalized token vector before applying gamma and beta.

---

## 10. Why Do We Need Gamma and Beta?

After normalization, LayerNorm applies two learnable parameters:

```text
gamma = scale
beta  = shift
```

Final output:

```text
output = gamma × normalized_value + beta
```

At the beginning of training, usually:

```text
gamma = 1
beta = 0
```

So initially LayerNorm just normalizes.

But during training, the model learns better gamma and beta values.

This gives the model flexibility.

Important point:

```text
LayerNorm is not just fixed normalization.
LayerNorm also has trainable parameters.
```

---

## 11. How Many Parameters Does LayerNorm Have?

LayerNorm has two learnable vectors:

```text
gamma
beta
```

Each has size:

```text
d_model
```

So total LayerNorm parameters:

```text
2 × d_model
```

Example 1:

```text
d_model = 768
gamma parameters = 768
beta parameters  = 768
total = 1536
```

Example 2:

```text
d_model = 4096
gamma parameters = 4096
beta parameters  = 4096
total = 8192
```

Compared to attention and feed-forward layers, this is small.

But LayerNorm is very important for stable training.

---

## 12. LayerNorm in Tensor Indexing Terms

Input shape:

```text
X shape = B × T × d_model
```

LayerNorm operates on the last dimension:

```text
d_model
```

For every batch item and every token position, LayerNorm normalizes:

```text
X[b, t, :]
```

Meaning:

```text
one token vector at one position
```

Example:

```text
X[0, 2, :]
```

means:

```text
batch item 0
token position 2
all hidden dimensions
```

LayerNorm normalizes this vector independently.

It repeats this for every token in every sequence.

---

## 13. Mini Tensor Example

Suppose:

```text
B = 1
T = 2
d_model = 3
```

Input:

```text
Token 1: [2, 4, 6]
Token 2: [10, 20, 30]
```

LayerNorm processes each token separately.

### Token 1

```text
[2, 4, 6]
```

Mean:

```text
4
```

After normalization:

```text
[-1.23, 0, 1.23]
```

### Token 2

```text
[10, 20, 30]
```

Mean:

```text
20
```

Subtract mean:

```text
[-10, 0, 10]
```

Standard deviation:

```text
≈ 8.16
```

Normalize:

```text
[-1.23, 0, 1.23]
```

So both tokens are normalized independently.

Important:

```text
LayerNorm does not mix Token 1 and Token 2.
```

---

## 14. Does LayerNorm Mix Information Between Tokens?

No.

LayerNorm does not mix tokens.

It only normalizes the hidden dimensions inside each token vector.

Token mixing happens in attention.

Comparison:

```text
LayerNorm = normalize features inside each token

Attention = mix information across tokens

FFN = transform each token independently
```

This distinction is very important for interviews.

---

## 15. Does LayerNorm Depend on Sequence Length?

Not directly.

LayerNorm normalizes across:

```text
d_model
```

It does not normalize across:

```text
T
```

where:

```text
T = sequence length
```

So whether the sequence has 10 tokens or 1000 tokens, each token vector is normalized independently across its hidden dimensions.

---

## 16. LayerNorm vs BatchNorm

This is a very common interview topic.

### BatchNorm

BatchNorm normalizes using statistics across the batch.

It is commonly used in CNNs.

Problem for GPT-style models:

```text
Batch sizes can vary.
Sequence lengths can vary.
Autoregressive inference may happen one sequence at a time.
Batch statistics can be unstable.
```

### LayerNorm

LayerNorm normalizes each token vector independently.

It does not depend on other examples in the batch.

That makes it suitable for Transformers and autoregressive language models.

Comparison:

| Feature | BatchNorm | LayerNorm |
|---|---|---|
| Normalizes across | Batch dimension | Hidden feature dimension |
| Commonly used in | CNNs | Transformers |
| Depends on batch size? | Yes | No |
| Suitable for autoregressive generation? | Not ideal | Yes |
| Used in GPT-style models? | Usually no | Yes |

Memory hook:

```text
BatchNorm compares examples.
LayerNorm normalizes each token internally.
```

---

## 17. Pre-LayerNorm vs Post-LayerNorm

There are two common ways to place LayerNorm inside Transformer blocks.

### Post-LayerNorm

This was used in the original Transformer-style block.

```text
X
↓
Attention
↓
Add residual
↓
LayerNorm
↓
FFN
↓
Add residual
↓
LayerNorm
```

Equation style:

```text
A = LayerNorm(X + MHA(X))

Y = LayerNorm(A + FFN(A))
```

Here LayerNorm happens after residual addition.

---

### Pre-LayerNorm

Modern GPT-style models commonly use Pre-LayerNorm.

```text
X
↓
LayerNorm
↓
Attention
↓
Add residual
↓
A
↓
LayerNorm
↓
FFN
↓
Add residual
↓
Y
```

Equation style:

```text
A = X + MHA(LayerNorm(X))

Y = A + FFN(LayerNorm(A))
```

Here LayerNorm happens before attention and before the feed-forward network.

---

## 18. Why Modern GPT Models Prefer Pre-LayerNorm

Pre-LayerNorm makes deep Transformer models easier to train.

The residual path remains clean.

In Pre-LayerNorm:

```text
X travels directly through the residual path.
The normalized version of X goes into the attention layer.
```

So there are two paths:

```text
Path 1:
X directly passes through the residual connection.

Path 2:
X is normalized, processed by attention, and then added back.
```

This helps gradients flow through many layers.

Simple intuition:

```text
Pre-LN gives the model a clean highway for information and gradients.
```

This is especially important when the model has many Transformer blocks.

---

## 19. Residual Path Intuition

Look at this equation:

```text
A = X + MaskedMHA(LayerNorm(X))
```

There are two paths:

```text
Path 1:
X is carried forward directly.

Path 2:
X is normalized, processed by masked attention, and added back.
```

Now look at the FFN equation:

```text
Y = A + FFN(LayerNorm(A))
```

Again there are two paths:

```text
Path 1:
A is carried forward directly.

Path 2:
A is normalized, transformed by FFN, and added back.
```

This is one reason GPT can stack many deep blocks.

---

## 20. What Happens If We Remove LayerNorm?

If LayerNorm is removed from a deep GPT model, training can become unstable.

Possible problems:

```text
activations become too large
activations become too small
gradients become unstable
loss becomes noisy
training may diverge
learning rate becomes harder to tune
deep models become harder to optimize
```

LayerNorm is one of the key components that makes deep Transformer training practical.

---

## 21. LayerNorm During Training vs Inference

LayerNorm is used during both training and inference.

### During Training

```text
LayerNorm runs.
Gamma and beta can update.
Model weights update through backpropagation.
```

### During Inference

```text
LayerNorm runs.
Gamma and beta are fixed.
Model weights do not update.
```

So LayerNorm is not only a training trick.

It is part of the model computation during inference too.

---

## 22. LayerNorm and Fine-Tuning

During full fine-tuning, LayerNorm parameters can be updated.

These parameters are:

```text
gamma
beta
```

During parameter-efficient fine-tuning, such as LoRA, the base model is often frozen.

In that case, LayerNorm parameters may or may not be updated depending on the fine-tuning recipe.

Examples:

```text
Full fine-tuning:
Most or all model parameters update.

LoRA fine-tuning:
Usually small low-rank adapter matrices update.

LoRA + trainable LayerNorm:
Some recipes also allow LayerNorm parameters to update.
```

Why might someone update LayerNorm during fine-tuning?

```text
LayerNorm has few parameters.
Updating it can help the model adapt.
It is cheaper than full fine-tuning.
```

---

## 23. LayerNorm vs RMSNorm

Many modern LLMs use **RMSNorm**, which is related to LayerNorm.

### LayerNorm

LayerNorm does three main things:

```text
1. Subtract mean
2. Divide by standard deviation
3. Apply learnable scale and shift
```

### RMSNorm

RMSNorm is simpler.

It mainly normalizes by root mean square.

It usually does not subtract the mean.

Simple comparison:

```text
LayerNorm controls mean and variance.

RMSNorm mainly controls scale.
```

Why do some models use RMSNorm?

```text
It is simpler.
It can be faster.
It works well in many LLMs.
```

For now, the important shared idea is:

```text
Both LayerNorm and RMSNorm help stabilize hidden representations.
```

---

## 24. Common Confusion: Does LayerNorm Make All Tokens the Same?

No.

LayerNorm does not make all token vectors identical.

It only normalizes each token vector's scale and mean.

Example:

```text
Vector A = [2, 4, 6]
Vector B = [6, 4, 2]
```

Both have the same mean, but their patterns are different.

After normalization:

```text
[2, 4, 6] → [-1.23, 0, 1.23]

[6, 4, 2] → [1.23, 0, -1.23]
```

They remain different.

So LayerNorm preserves relative patterns inside the token vector.

---

## 25. Common Confusion: Is LayerNorm Learned?

Partly yes.

The normalization step itself is deterministic.

But LayerNorm has learnable parameters:

```text
gamma
beta
```

These allow the model to learn the best scale and shift after normalization.

So LayerNorm is both:

```text
a normalization operation
+
a small learned transformation
```

---

## 26. One GPT Block With LayerNorm Highlighted

Modern GPT-style block:

```text
Input X
  │
  ├── Residual path carries X directly
  │
  ↓
LayerNorm(X)
  ↓
Masked Multi-Head Attention
  ↓
Add with X
  ↓
A
  │
  ├── Residual path carries A directly
  │
  ↓
LayerNorm(A)
  ↓
Feed-Forward Network
  ↓
Add with A
  ↓
Output Y
```

Equation style:

```text
A = X + MaskedMHA(LayerNorm(X))

Y = A + FFN(LayerNorm(A))
```

LayerNorm appears before each major transformation.

---

## 27. Interview-Level Answer

If an interviewer asks:

```text
Why is LayerNorm used in GPT?
```

You can answer:

LayerNorm is used in GPT to stabilize hidden representations as they pass through many Transformer blocks. It normalizes each token vector across its hidden dimensions, independent of batch size. This makes it suitable for sequence models and autoregressive inference.

In modern GPT-style models, LayerNorm is often applied before attention and before the feed-forward network. This is called Pre-LayerNorm. Pre-LayerNorm improves gradient flow and makes deep Transformers easier to train.

---

## 28. Interview Follow-Up: LayerNorm vs BatchNorm

If asked:

```text
Why not BatchNorm in GPT?
```

You can answer:

BatchNorm depends on batch-level statistics, which can be unstable for variable-length sequences and autoregressive generation. LayerNorm normalizes each token independently across its hidden dimensions, so it does not depend on batch size and behaves consistently during both training and inference. That makes LayerNorm more suitable for GPT-style Transformer models.

---

## 29. Common Interview Questions

### Q1. What dimension does LayerNorm normalize over?

LayerNorm normalizes over the hidden dimension:

```text
d_model
```

For input shape:

```text
B × T × d_model
```

LayerNorm normalizes:

```text
X[b, t, :]
```

for each batch item and token position.

---

### Q2. Does LayerNorm change the tensor shape?

No.

LayerNorm changes the values but preserves the shape.

```text
Input:  B × T × d_model
Output: B × T × d_model
```

---

### Q3. Does LayerNorm mix tokens?

No.

LayerNorm does not mix tokens.

It only normalizes the hidden dimensions inside each token vector.

Token mixing happens in attention.

---

### Q4. Is LayerNorm used during inference?

Yes.

LayerNorm is part of the forward pass during both training and inference.

During inference, gamma and beta are fixed and not updated.

---

### Q5. What are gamma and beta?

Gamma and beta are learnable parameters.

```text
gamma = scale
beta  = shift
```

They allow the model to adjust the normalized vector.

---

### Q6. Why is Pre-LayerNorm useful?

Pre-LayerNorm keeps the residual path cleaner and improves gradient flow through deep Transformer stacks.

This makes modern GPT-style models easier to train.

---

## 30. Memory Hooks

```text
LayerNorm normalizes one token vector at a time.

It normalizes across d_model, not across batch or sequence.

It keeps activations stable.

It does not mix tokens.

Attention mixes tokens.

FFN transforms tokens.

Pre-LN means normalize before attention and FFN.

Pre-LN helps deep GPT models train better.

LayerNorm has learnable gamma and beta.

LayerNorm runs during both training and inference.
```

---

## 31. Final One-Line Summary

```text
LayerNorm stabilizes each token representation by normalizing across hidden dimensions, helping GPT blocks train deeply and reliably without changing tensor shape.
```
