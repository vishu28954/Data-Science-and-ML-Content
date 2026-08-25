# GPT Architecture — Part 2G: Feed-Forward Network / MLP Inside GPT

## 1. Why This Part Matters

In the previous parts, we studied:

```text
Part 2A → Input and tensor shapes inside a GPT block

Part 2B → LayerNorm in GPT

Part 2C → Q, K, V projections

Part 2D → Causal masked self-attention

Part 2E → Multi-head attention

Part 2F → Residual connections
```

Now we study the second major sublayer inside a GPT block:

```text
Feed-Forward Network
```

It is also called:

```text
FFN
MLP
Feed-forward layer
Position-wise feed-forward network
```

The main idea is:

```text
Attention mixes information across tokens.

The feed-forward network transforms each token independently.
```

This is one of the most important distinctions in GPT architecture.

---

## 2. Where FFN Appears Inside a GPT Block

A modern GPT block looks like this:

```text
Input X
  ↓
LayerNorm
  ↓
Masked Multi-Head Attention
  ↓
Residual Connection
  ↓
A
  ↓
LayerNorm
  ↓
Feed-Forward Network / MLP
  ↓
Residual Connection
  ↓
Output Y
```

The relevant equation is:

```text
Y = A + FFN(LayerNorm(A))
```

where:

```text
A = representation after the attention sublayer

LayerNorm(A) = normalized representation before FFN

FFN(LayerNorm(A)) = per-token learned transformation

Y = final output of the GPT block
```

So the FFN comes after attention.

---

## 3. Attention vs FFN

This is the key intuition.

### Attention

Attention allows tokens to communicate.

Example:

```text
The animal was tired
```

The token `tired` can attend to:

```text
The
animal
was
tired
```

So attention mixes information across tokens.

### FFN

The FFN does not mix tokens.

It transforms each token vector independently.

So after attention gives every token some context, the FFN processes each token's updated representation.

Memory hook:

```text
Attention = token communication

FFN = token thinking
```

---

## 4. What Does “Position-Wise” Mean?

The FFN is often called a **position-wise feed-forward network**.

This means:

```text
The same FFN is applied independently to each token position.
```

Suppose the sequence is:

```text
I | love | machine | learning
```

The FFN processes:

```text
token 1 representation independently

token 2 representation independently

token 3 representation independently

token 4 representation independently
```

It does not directly compare token 1 with token 2.

That comparison already happened inside attention.

---

## 5. Input and Output Shape of FFN

The input to the FFN has shape:

```text
B × T × d_model
```

The output also has shape:

```text
B × T × d_model
```

Example:

```text
B = 2
T = 5
d_model = 768
```

Input:

```text
2 × 5 × 768
```

Output:

```text
2 × 5 × 768
```

So FFN preserves the shape.

This is necessary because the FFN output is added back through a residual connection:

```text
Y = A + FFN(LayerNorm(A))
```

For this addition to work:

```text
A shape must equal FFN output shape
```

---

## 6. Basic FFN Formula

A simple Transformer FFN has two linear layers with a non-linear activation in between.

Formula:

```text
FFN(x) = W2 activation(W1 x + b1) + b2
```

The flow is:

```text
Input vector
  ↓
Linear layer 1
  ↓
Activation function
  ↓
Linear layer 2
  ↓
Output vector
```

In dimension terms:

```text
d_model → d_ff → d_model
```

where:

```text
d_model = original hidden dimension

d_ff = intermediate feed-forward dimension
```

---

## 7. Why FFN Expands the Dimension

Usually, the FFN expands the token vector to a larger dimension and then compresses it back.

Example:

```text
768 → 3072 → 768
```

Here:

```text
d_model = 768

d_ff = 3072
```

The expansion factor is:

```text
3072 / 768 = 4
```

So the FFN hidden dimension is often around 4 times larger than `d_model` in classic Transformer architectures.

Why expand?

Because the model gets more space to compute useful transformations.

Memory hook:

```text
FFN expands to think in a larger space, then compresses back to d_model.
```

---

## 8. Concrete Shape Example

Suppose:

```text
B = 2
T = 5
d_model = 768
d_ff = 3072
```

Input to FFN:

```text
A shape = 2 × 5 × 768
```

After first linear layer:

```text
2 × 5 × 3072
```

After activation:

```text
2 × 5 × 3072
```

After second linear layer:

```text
2 × 5 × 768
```

Final FFN output:

```text
2 × 5 × 768
```

Then residual addition:

```text
Y = A + FFN(LayerNorm(A))
```

Output:

```text
Y shape = 2 × 5 × 768
```

---

## 9. Token-Wise View

Suppose one token vector has size:

```text
d_model = 768
```

The FFN transforms it like this:

```text
token vector: 768 dimensions
       ↓
linear layer 1
       ↓
expanded vector: 3072 dimensions
       ↓
activation
       ↓
linear layer 2
       ↓
output vector: 768 dimensions
```

This happens separately for every token.

Example:

```text
FFN(token_1)

FFN(token_2)

FFN(token_3)

FFN(token_4)
```

The same FFN weights are reused for every token position.

---

## 10. Why the Same FFN Is Applied to Every Token

GPT does not have a different FFN for every position.

It uses the same FFN weights across all token positions.

Why?

Because the model should learn general transformations that can apply anywhere in the sequence.

For example:

```text
If a token representation indicates a noun-like concept, transform it in a useful way.

If a token representation indicates a code variable, transform it in a useful way.

If a token representation indicates a question, transform it in a useful way.
```

These transformations should not depend on a fixed absolute position like token 3 or token 10.

Memory hook:

```text
Same FFN weights are reused at every token position.
```

---

## 11. What Does the First Linear Layer Do?

The first linear layer maps:

```text
d_model → d_ff
```

Example:

```text
768 → 3072
```

This creates a larger intermediate representation.

Formula:

```text
h = W1 x + b1
```

where:

```text
x = input token vector

W1 = first FFN weight matrix

b1 = first bias vector

h = expanded hidden vector
```

The first linear layer creates many learned feature combinations from the input token vector.

---

## 12. What Does the Activation Function Do?

After the first linear layer, GPT applies a non-linear activation function.

Example:

```text
h = activation(W1 x + b1)
```

Why do we need activation?

Because without activation, two linear layers would collapse into one linear layer.

If there were no activation:

```text
FFN(x) = W2(W1x)
```

This is still just a linear transformation.

But with activation:

```text
FFN(x) = W2 activation(W1x)
```

the FFN can learn non-linear transformations.

Memory hook:

```text
Activation makes the FFN more expressive.
```

---

## 13. Why Non-Linearity Matters

Language understanding is not just linear.

The model needs to represent complex patterns like:

```text
negation

syntax

entity relations

code structure

instruction intent

conditional meaning

style

reasoning patterns
```

A purely linear transformation would be limited.

The activation function lets the FFN model more complex relationships inside each token representation.

---

## 14. Common Activation Functions

Different Transformer models use different activation functions.

Common examples:

```text
ReLU
GELU
SwiGLU
GeGLU
SiLU
```

Classic Transformer models often used ReLU.

Many GPT-style models use GELU or gated variants like SwiGLU.

---

## 15. ReLU Intuition

ReLU means:

```text
ReLU(x) = max(0, x)
```

Example:

```text
Input:  [-2, -0.5, 0, 3, 5]

Output: [0, 0, 0, 3, 5]
```

ReLU keeps positive values and removes negative values.

It is simple and efficient.

But modern LLMs often use smoother or gated activations.

---

## 16. GELU Intuition

GELU stands for:

```text
Gaussian Error Linear Unit
```

You do not need the exact formula for most interviews.

Intuition:

```text
GELU smoothly decides how much of each value should pass through.
```

Unlike ReLU, GELU does not suddenly cut negative values to zero.

It behaves more smoothly.

Simplified memory hook:

```text
ReLU = hard gate

GELU = smooth gate
```

---

## 17. SwiGLU Intuition

Many modern LLMs use gated FFNs such as SwiGLU.

A simplified SwiGLU-style FFN looks like:

```text
FFN(x) = W_down( activation(W_gate x) * (W_up x) )
```

The important idea is:

```text
One projection creates content.

Another projection creates a gate.

The gate controls how much content passes through.
```

So instead of just:

```text
expand → activate → compress
```

SwiGLU uses:

```text
expand content
expand gate
gate the content
compress back
```

Memory hook:

```text
SwiGLU lets the FFN choose what information to pass.
```

---

## 18. Standard FFN vs Gated FFN

### Standard FFN

```text
x
↓
W1
↓
activation
↓
W2
↓
output
```

Formula:

```text
FFN(x) = W2 activation(W1 x)
```

### Gated FFN

```text
x
↓
content projection
        ×
gate projection + activation
↓
down projection
↓
output
```

Simplified formula:

```text
GatedFFN(x) = W_down( activation(W_gate x) * W_up x )
```

The multiplication is element-wise.

Gated FFNs often work better in modern LLMs.

---

## 19. What Does the Second Linear Layer Do?

The second linear layer maps the expanded vector back to the model dimension.

```text
d_ff → d_model
```

Example:

```text
3072 → 768
```

Formula:

```text
output = W2 h + b2
```

where:

```text
h = activated expanded vector
```

The purpose is:

```text
compress the transformed representation back to d_model
```

This is required because the residual connection needs the output shape to match the input shape.

---

## 20. Full FFN Flow With Shapes

For a standard FFN:

```text
Input x:
d_model
```

First linear layer:

```text
d_model → d_ff
```

Activation:

```text
d_ff → d_ff
```

Second linear layer:

```text
d_ff → d_model
```

Output:

```text
d_model
```

For the full tensor:

```text
B × T × d_model
    ↓
B × T × d_ff
    ↓
B × T × d_ff
    ↓
B × T × d_model
```

---

## 21. FFN Does Not Mix Tokens

This is very important.

Suppose we have:

```text
I | love | machine | learning
```

The FFN processes:

```text
I vector independently

love vector independently

machine vector independently

learning vector independently
```

There is no direct token-to-token communication inside the FFN.

So:

```text
FFN(token_i) does not directly look at token_j
```

But each token vector already contains contextual information from attention.

Therefore, the FFN is transforming context-aware vectors.

Memory hook:

```text
Attention gathers context.

FFN processes the gathered context.
```

---

## 22. How FFN Helps After Attention

Attention gives each token a mixture of information from previous tokens.

For example:

```text
The animal was tired
```

After attention, the token `tired` may contain information from:

```text
The
animal
was
tired
```

Then the FFN transforms this enriched representation.

It may strengthen useful features and suppress irrelevant features.

So the combination is:

```text
Attention:
collect useful context

FFN:
process and refine that context
```

---

## 23. FFN as Feature Transformation

In classical ML, you can think of feature engineering as creating useful derived features.

The FFN does something similar, but learned automatically.

It transforms the hidden representation into more useful internal features.

Example:

```text
Input representation may contain:
subject information
verb information
tense information
negation information
instruction information
```

The FFN can combine and transform these internal signals.

This helps the model prepare better representations for later GPT blocks.

---

## 24. FFN and Representation Refinement Across Layers

A GPT model has many blocks.

Each block has attention and FFN.

Simplified view:

```text
Block 1:
attention gathers context
FFN refines token representation

Block 2:
attention gathers deeper context
FFN refines again

Block 3:
attention gathers even richer context
FFN refines again
```

So across many blocks:

```text
token representation becomes increasingly useful for next-token prediction
```

---

## 25. FFN Parameter Count

The FFN often contains many parameters.

For a standard two-layer FFN:

```text
W1 shape = d_model × d_ff

W2 shape = d_ff × d_model
```

Ignoring biases, total parameters:

```text
d_model × d_ff + d_ff × d_model
```

This simplifies to:

```text
2 × d_model × d_ff
```

---

## 26. Parameter Count Example: d_model = 768

Suppose:

```text
d_model = 768
d_ff = 3072
```

First matrix:

```text
W1 = 768 × 3072 = 2,359,296 parameters
```

Second matrix:

```text
W2 = 3072 × 768 = 2,359,296 parameters
```

Total FFN parameters:

```text
2,359,296 + 2,359,296 = 4,718,592
```

So the FFN has about:

```text
4.7 million parameters
```

inside one GPT block.

---

## 27. Compare FFN Parameters With Attention Parameters

For standard attention with:

```text
d_model = 768
```

Attention matrices:

```text
W_Q, W_K, W_V, W_O
```

Each is:

```text
768 × 768 = 589,824
```

Total attention parameters:

```text
4 × 589,824 = 2,359,296
```

FFN parameters with:

```text
d_ff = 3072
```

are:

```text
4,718,592
```

So in this example:

```text
FFN has about 2 times more parameters than attention.
```

This is why the FFN/MLP is a very large part of GPT block parameters.

---

## 28. Parameter Count Example: d_model = 4096

Suppose:

```text
d_model = 4096
d_ff = 11008
```

First matrix:

```text
W1 = 4096 × 11008 = 45,088,768 parameters
```

Second matrix:

```text
W2 = 11008 × 4096 = 45,088,768 parameters
```

Total standard FFN parameters:

```text
90,177,536
```

That is about:

```text
90 million parameters
```

inside one FFN layer.

For gated FFNs, there may be an additional projection, so the parameter count can be even larger.

---

## 29. Why FFN Often Has More Parameters Than Attention

Attention has roughly:

```text
4 × d_model × d_model
```

parameters.

A standard FFN has roughly:

```text
2 × d_model × d_ff
```

If:

```text
d_ff = 4 × d_model
```

then FFN parameters become:

```text
2 × d_model × 4d_model
= 8 × d_model × d_model
```

So FFN can have about twice as many parameters as attention.

Memory hook:

```text
Attention is expensive because of T² compute.

FFN is expensive because of large d_ff parameters.
```

---

## 30. Compute Cost of FFN

The FFN is applied to every token.

For each token, it performs large matrix multiplications.

The compute cost grows roughly with:

```text
B × T × d_model × d_ff
```

Because it must process every token in every batch item.

If sequence length doubles:

```text
FFN compute roughly doubles
```

This is different from attention.

Attention grows roughly as:

```text
T²
```

FFN grows roughly as:

```text
T
```

with respect to sequence length.

Important comparison:

```text
Attention cost grows quadratically with sequence length.

FFN cost grows linearly with sequence length.
```

---

## 31. FFN During Training

During training:

```text
1. Token representations enter FFN.
2. FFN transforms each token.
3. Output is added through residual connection.
4. Final logits are used for cross-entropy loss.
5. Gradients flow backward.
6. FFN weights update.
```

The FFN learns transformations that help the model predict the next token.

---

## 32. FFN During Inference

During inference:

```text
FFN still runs inside every GPT block.
```

But weights do not update.

For every generated token, the model must run:

```text
attention
+
FFN
```

through all layers.

KV cache helps reduce repeated attention computation for previous tokens.

However, FFN still needs to be applied to the new token at every layer.

Memory hook:

```text
KV cache helps attention reuse K and V.

KV cache does not remove the need to run FFN for the new token.
```

---

## 33. FFN and KV Cache

KV cache stores keys and values from previous tokens.

It helps avoid recomputing K and V for old tokens.

But FFN is not cached in the same way for generation.

Why?

Because for each new token, the model needs to compute the new token's hidden representation through every layer.

So at every layer, the new token goes through:

```text
LayerNorm
attention using cached K and V
residual
LayerNorm
FFN
residual
```

So FFN remains a major part of inference cost.

---

## 34. FFN and LoRA Fine-Tuning

LoRA can be applied to attention matrices, but it can also be applied to FFN matrices.

For example, LoRA may modify:

```text
W_Q
W_V
W_O
W1
W2
```

or in gated FFNs:

```text
up projection
gate projection
down projection
```

The idea is:

```text
Instead of updating the full FFN matrices,
train small low-rank adapter matrices.
```

Because FFN matrices are large, adapting them can be powerful but may cost more than adapting only attention projections.

---

## 35. FFN and Knowledge Storage Intuition

There is a useful intuition that FFNs store and transform a lot of model knowledge.

This does not mean all knowledge is literally stored only in FFNs.

But FFNs contain many parameters and are important for learned transformations.

Attention helps retrieve and mix context.

FFNs help process that context into useful representations.

Simplified intuition:

```text
Attention decides what context matters.

FFN helps interpret and transform that context.
```

---

## 36. Standard FFN vs Modern LLM MLP Names

Different codebases use different names.

You may see:

```text
MLP
FFN
FeedForward
feed_forward
mlp
```

They usually refer to the same conceptual component.

In many modern LLMs, the MLP has projections named something like:

```text
gate_proj
up_proj
down_proj
```

For standard FFNs, you may see:

```text
fc1
fc2
```

or:

```text
dense_h_to_4h
dense_4h_to_h
```

The naming differs, but the idea is the same:

```text
expand
activate or gate
compress
```

---

## 37. Common Confusion: Is FFN the Same as the Final Vocabulary Head?

No.

The FFN is inside every GPT block.

The vocabulary head is at the end of the model.

### FFN

```text
Inside each GPT block

Transforms hidden states

Shape:
B × T × d_model → B × T × d_model
```

### Vocabulary Head

```text
After all GPT blocks

Maps hidden states to vocabulary scores

Shape:
B × T × d_model → B × T × V
```

where:

```text
V = vocabulary size
```

Memory hook:

```text
FFN refines hidden representations.

Vocabulary head predicts tokens.
```

---

## 38. Common Confusion: Does FFN Know Other Tokens?

Directly, no.

The FFN processes each token independently.

But indirectly, yes.

Because before the FFN, attention has already mixed information from previous tokens into each token representation.

So the FFN processes a representation that may already contain context from other tokens.

This is the subtle point:

```text
FFN does not directly mix tokens.

But FFN works on token vectors that may already contain mixed context.
```

---

## 39. Common Confusion: Is FFN Linear?

The FFN contains linear layers, but the full FFN is not purely linear because it includes a non-linear activation.

Without activation:

```text
two linear layers = one linear layer
```

With activation:

```text
linear + nonlinearity + linear = nonlinear transformation
```

So FFN increases model expressiveness.

---

## 40. Common Confusion: Does FFN Change Sequence Length?

No.

The FFN does not change sequence length.

Input:

```text
B × T × d_model
```

Output:

```text
B × T × d_model
```

The same number of tokens comes out.

---

## 41. Common Confusion: Why Is d_ff Larger Than d_model?

Because expanding to a larger intermediate dimension gives the model more capacity to compute useful transformations.

It is similar to giving the model a larger workspace.

Then it compresses the result back to `d_model`.

Memory hook:

```text
d_ff is the FFN workspace.
```

---

## 42. Common Confusion: Is FFN Shared Across Layers?

No.

Each GPT block has its own FFN weights.

For example:

```text
Block 1 has FFN_1

Block 2 has FFN_2

Block 3 has FFN_3
```

They are not shared in standard GPT-style models.

This lets different layers learn different transformations.

---

## 43. Common Confusion: Is FFN Shared Across Tokens?

Yes.

Within the same layer, the same FFN weights are applied to every token position.

So:

```text
Same FFN across tokens in one layer.

Different FFN across layers.
```

This is an important distinction.

---

## 44. FFN Inside One GPT Block: Full View

The GPT block equation is:

```text
A = X + MaskedMHA(LayerNorm(X))

Y = A + FFN(LayerNorm(A))
```

The FFN part is:

```text
LayerNorm(A)
  ↓
Linear projection up
  ↓
Activation or gate
  ↓
Linear projection down
  ↓
FFN output
  ↓
Add residual connection
  ↓
Y
```

Shape flow:

```text
B × T × d_model
  ↓
B × T × d_ff
  ↓
B × T × d_ff
  ↓
B × T × d_model
```

---

## 45. Tiny Numerical Example

Suppose one token vector is:

```text
x = [1, 2]
```

Let the first linear layer expand it to 3 dimensions.

Assume:

```text
W1 transforms [1, 2] into [3, -1, 2]
```

Apply ReLU activation:

```text
ReLU([3, -1, 2]) = [3, 0, 2]
```

Now the second linear layer compresses it back to 2 dimensions.

Assume:

```text
W2 transforms [3, 0, 2] into [4, 1]
```

So:

```text
FFN(x) = [4, 1]
```

With residual connection, if the original representation was:

```text
A = [1, 2]
```

then final output:

```text
Y = A + FFN(A)

Y = [1, 2] + [4, 1]

Y = [5, 3]
```

This is a tiny simplified example.

Real GPT uses much larger vectors and learned weights.

---

## 46. Why FFN Is Important for Next-Token Prediction

GPT's goal is to predict the next token.

To do this, each token representation must contain useful information.

Attention helps gather context.

But the model still needs to transform that context into a better internal representation.

The FFN helps produce hidden features useful for:

```text
grammar

facts

style

reasoning patterns

code structure

instruction following

next-token prediction
```

So the FFN is not optional.

It is a major part of how GPT gains capacity.

---

## 47. Interview-Level Answer

If an interviewer asks:

```text
What is the feed-forward network inside a GPT block?
```

You can answer:

The feed-forward network, or MLP, is the second major sublayer inside a GPT block after masked multi-head attention. It is applied independently to each token position. A standard FFN expands the hidden dimension from d_model to a larger intermediate dimension d_ff, applies a non-linear activation such as GELU or a gated activation such as SwiGLU, and then projects it back to d_model. Attention mixes information across tokens, while the FFN transforms each token's contextual representation. The FFN output is added back through a residual connection, so the overall shape remains B × T × d_model.

---

## 48. Common Interview Follow-Up Questions

### Q1. What is the role of the FFN in GPT?

The FFN transforms each token representation after attention has gathered context.

It helps refine the representation and increases the model's expressive power.

---

### Q2. Does the FFN mix information between tokens?

No.

The FFN is applied independently to each token.

Token mixing happens in attention.

---

### Q3. What is the basic FFN formula?

```text
FFN(x) = W2 activation(W1 x + b1) + b2
```

---

### Q4. What are the usual dimensions inside FFN?

A common standard pattern is:

```text
d_model → d_ff → d_model
```

Example:

```text
768 → 3072 → 768
```

---

### Q5. Why is d_ff larger than d_model?

The larger intermediate dimension gives the model more capacity to compute useful non-linear transformations.

It acts like a larger internal workspace.

---

### Q6. Why do we need an activation function?

Without activation, two linear layers collapse into one linear transformation.

The activation introduces non-linearity, making the FFN more expressive.

---

### Q7. What activations are used in GPT-style models?

Common activations include:

```text
GELU

SwiGLU

GeGLU

SiLU

ReLU
```

Modern LLMs often use GELU or gated activations like SwiGLU.

---

### Q8. Does FFN change the tensor shape?

Internally, yes:

```text
B × T × d_model → B × T × d_ff → B × T × d_model
```

But the final output shape is the same as the input:

```text
B × T × d_model
```

---

### Q9. Is FFN the same as the vocabulary output head?

No.

The FFN is inside every GPT block and keeps shape:

```text
B × T × d_model
```

The vocabulary head is after all GPT blocks and maps:

```text
B × T × d_model → B × T × V
```

---

### Q10. Does each GPT block have its own FFN?

Yes.

Each GPT block has its own FFN weights.

The FFN is shared across token positions within a layer, but not usually shared across different layers.

---

### Q11. Why does FFN have many parameters?

Because it expands to a large intermediate dimension.

For standard FFN:

```text
parameters ≈ 2 × d_model × d_ff
```

Since `d_ff` is often much larger than `d_model`, FFN contributes many parameters.

---

### Q12. How does FFN relate to LoRA?

LoRA can be applied to FFN projection matrices such as up, gate, or down projections.

This allows fine-tuning part of the FFN behavior without updating the full large matrices.

---

## 49. Memory Hooks

```text
FFN = Feed-Forward Network.

MLP = another name for FFN.

Attention = token communication.

FFN = token thinking.

FFN is applied independently to each token.

FFN does not directly mix tokens.

Attention gathers context.

FFN refines context.

Shape:
B × T × d_model → B × T × d_ff → B × T × d_model

d_ff is usually larger than d_model.

Activation makes FFN non-linear.

Residual connection adds FFN output back to the representation.

FFN often has more parameters than attention.

FFN is inside every GPT block.

Vocabulary head is only at the end.
```

---

## 50. Final One-Line Summary

```text
The FFN or MLP inside a GPT block transforms each token's contextual representation independently by expanding it, applying a non-linear activation or gate, compressing it back to d_model, and adding it through a residual connection.
```
