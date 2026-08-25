# GPT Architecture — Part 2: One GPT Block in Detail

## 1. What is a GPT Block?

A GPT model is made by stacking multiple **GPT blocks** one after another.

In Part 1, we saw the complete flow:

```text
Text
  ↓
Tokens
  ↓
Embeddings
  ↓
GPT Blocks
  ↓
Logits
  ↓
Softmax
  ↓
Next Token
```

Now we zoom inside **one GPT block**.

The main idea is:

```text
A GPT block updates token representations using masked attention and a feed-forward network.
```

---

## 2. High-Level Structure of One GPT Block

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

In equation form:

$$
A = X + \text{MaskedMHA}(\text{LayerNorm}(X))
$$

$$
Y = A + \text{FFN}(\text{LayerNorm}(A))
$$

Where:

```text
X = input to the GPT block
A = output after masked attention and residual connection
Y = final output of the GPT block
```

This is called **Pre-LayerNorm architecture** because LayerNorm is applied before attention and before the feed-forward network.

---

## 3. Input Shape to One GPT Block

Suppose:

```text
B = batch size
T = sequence length
d_model = hidden dimension
```

Then the input shape is:

$$
X \in R^{B \times T \times d_{model}}
$$

Example:

```text
B = 2
T = 5
d_model = 768
```

Then:

$$
X \in R^{2 \times 5 \times 768}
$$

Meaning:

```text
2     = number of sequences in the batch
5     = number of tokens in each sequence
768   = vector size for each token
```

---

## 4. Step 1 — LayerNorm

Before attention, GPT applies LayerNorm.

```text
X
↓
LayerNorm(X)
```

LayerNorm normalizes each token vector independently.

For one token vector:

```text
x = [x1, x2, x3, ..., xd]
```

LayerNorm stabilizes the values inside this vector.

### Why LayerNorm is used

Deep neural networks can become unstable when activations become too large or too small.

LayerNorm helps by keeping token representations in a stable range.

Memory hook:

```text
LayerNorm keeps token representations stable before processing.
```

---

## 5. Step 2 — Create Q, K, and V

After LayerNorm, GPT creates three vectors for every token:

```text
Q = Query
K = Key
V = Value
```

These are created using learned weight matrices:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

Meaning:

| Term | Meaning |
|---|---|
| Query | What is this token looking for? |
| Key | What information does this token advertise? |
| Value | What information does this token provides if attended to? |

Example:

```text
Sentence: The animal was tired
```

When processing the token:

```text
tired
```

the query may ask:

```text
Who was tired?
```

The key of:

```text
animal
```

may match this query.

The value of:

```text
animal
```

provides useful information to the token `tired`.

---

## 6. Step 3 — Masked Multi-Head Self-Attention

GPT uses **masked self-attention**.

This means a token can attend only to:

```text
itself and previous tokens
```

It cannot attend to future tokens.

Example:

```text
I love machine learning
```

Allowed attention:

| Token | Can Attend To |
|---|---|
| I | I |
| love | I, love |
| machine | I, love, machine |
| learning | I, love, machine, learning |

This is what makes GPT autoregressive.

Memory hook:

```text
GPT attention = look only left.
```

---

## 7. Attention Formula

The masked attention formula is:

$$
\text{MaskedAttention}(Q,K,V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

Where:

```text
QK^T       = similarity scores between tokens
M          = causal mask
sqrt(d_k)  = scaling factor
softmax    = converts scores into attention weights
V          = information vectors to combine
```

The causal mask blocks future tokens.

A future token gets a very negative score, so after softmax its attention probability becomes almost zero.

Memory hook:

```text
Mask = no cheating.
```

---

## 8. Why Multi-Head Attention?

GPT does not use only one attention head.

It uses many attention heads in parallel.

Example:

```text
d_model = 768
number of heads = 12
head dimension = 64
```

Because:

```text
768 / 12 = 64
```

Each head can learn a different type of relationship.

Examples:

```text
Head 1 → nearby words
Head 2 → subject-verb relationship
Head 3 → pronoun reference
Head 4 → punctuation or syntax
Head 5 → long-range dependency
```

So multi-head attention gives GPT multiple ways to understand the same context.

Memory hook:

```text
Single-head attention = one view of the sentence.
Multi-head attention = many views of the sentence.
```

---

## 9. Output Projection

After all attention heads are computed, GPT concatenates them:

```text
head_1 | head_2 | head_3 | ... | head_h
```

Then it applies an output projection matrix:

$$
W_O
$$

This mixes information from all attention heads.

The output shape becomes the same as the input shape:

$$
R^{B \times T \times d_{model}}
$$

This is important because we need to add it back to the original input using a residual connection.

---

## 10. Step 4 — First Residual Connection

After attention, GPT adds the original input back.

$$
A = X + \text{MaskedMHA}(\text{LayerNorm}(X))
$$

This is called a **residual connection**.

### Why residual connections are used

Without residual connections, deep networks can lose information or suffer from unstable gradients.

Residual connections allow the model to keep old information and add a learned correction.

Memory hook:

```text
Residual connection = old information + new correction
```

So after attention:

```text
A = original token representation + context gathered from previous tokens
```

---

## 11. Step 5 — Second LayerNorm

After the first residual connection, GPT applies LayerNorm again.

```text
A
↓
LayerNorm(A)
```

This prepares the representation before sending it into the feed-forward network.

So far:

```text
X
↓
LayerNorm
↓
Masked Attention
↓
Add Residual
↓
A
↓
LayerNorm
```

---

## 12. Step 6 — Feed-Forward Network

The Feed-Forward Network, or FFN, is usually a two-layer neural network applied independently to each token.

The formula is:

$$
\text{FFN}(x) = W_2 \sigma(W_1x + b_1) + b_2
$$

Usually, the FFN expands the dimension and then compresses it back.

Example:

```text
768 → 3072 → 768
```

For larger modern LLMs, this may look like:

```text
4096 → 11008 → 4096
```

### Important point

The FFN does **not** mix information between tokens.

Token mixing already happens in attention.

The FFN improves each token representation independently.

Memory hook:

```text
Attention = token communication
FFN = token thinking
```

---

## 13. Step 7 — Second Residual Connection

After the FFN, GPT again adds the previous representation back.

$$
Y = A + \text{FFN}(\text{LayerNorm}(A))
$$

So the final output is:

```text
Y = representation after attention + FFN correction
```

This output goes into the next GPT block.

---

## 14. Full GPT Block Equation

A modern GPT block can be summarized as:

$$
A = X + \text{MaskedMHA}(\text{LayerNorm}(X))
$$

$$
Y = A + \text{FFN}(\text{LayerNorm}(A))
$$

This is one of the most important GPT architecture patterns.

---

## 15. What Does One GPT Block Actually Do?

Suppose the input sentence is:

```text
The animal was tired
```

Initially, each token mostly knows only itself.

After one GPT block:

```text
animal → understands it is likely a noun
was    → understands tense/context
tired  → can attend to animal
```

After many GPT blocks:

```text
Early blocks  → local relationships
Middle blocks → grammar and syntax
Deep blocks   → meaning, instructions, and generation patterns
```

This is a simplified intuition, but it is useful for interviews.

---

## 16. Why GPT Stacks Many Blocks

One block is not enough.

Each block refines the token representations.

Think of it like this:

```text
Embedding layer:
Token knows mostly itself.

After early GPT blocks:
Token understands local context.

After middle GPT blocks:
Token understands grammar and sentence structure.

After deeper GPT blocks:
Token understands meaning and generation behavior.
```

Stacking blocks gives the model depth and reasoning capacity.

---

## 17. GPT Block vs Original Transformer Decoder Block

This is an important interview distinction.

The original Transformer decoder block has three main sublayers:

```text
1. Masked self-attention
2. Cross-attention
3. Feed-forward network
```

But GPT is decoder-only.

So a GPT block has:

```text
1. Masked self-attention
2. Feed-forward network
```

GPT does **not** have cross-attention.

### Why GPT does not need cross-attention

GPT does not use a separate encoder.

It receives all context directly in the prompt.

Example:

```text
Translate English to French:
English: I love machine learning
French:
```

The English sentence is already part of the input context.

So GPT can use self-attention over the prompt instead of cross-attention to encoder outputs.

---

## 18. Training View of One GPT Block

During training:

```text
Input tokens
    ↓
GPT blocks
    ↓
Logits over vocabulary
    ↓
Cross-entropy loss
    ↓
Backpropagation
    ↓
Update weights
```

The weights updated during training can include:

```text
token embedding weights
attention matrices W_Q, W_K, W_V, W_O
feed-forward network weights
LayerNorm parameters
vocabulary output head
```

During full fine-tuning, many or all of these weights can update.

During LoRA fine-tuning, the base model is usually frozen, and small trainable adapter matrices are added to selected layers.

---

## 19. Inference View of One GPT Block

During inference:

```text
Prompt tokens
    ↓
GPT blocks
    ↓
Next-token logits
    ↓
Choose next token
    ↓
Append token
    ↓
Repeat
```

Weights do not change during inference.

The GPT block only transforms the current context into better hidden representations.

Later, we study **KV cache**, which makes inference faster by avoiding repeated computation of keys and values for previous tokens.

---

## 20. Interview-Level Answer

If an interviewer asks:

```text
What is inside a GPT block?
```

You can answer:

A GPT block contains masked multi-head self-attention followed by a feed-forward network, with residual connections and LayerNorm around both sublayers. In modern GPT-style models, LayerNorm is often applied before each sublayer, which is called Pre-LayerNorm.

The masked attention allows each token to attend only to previous tokens, preserving autoregressive generation. The feed-forward network transforms each token independently. Residual connections preserve information and improve gradient flow, while LayerNorm stabilizes training.

---

## 21. Common Interview Follow-Up Questions

### Q1. Why does GPT use masked attention?

GPT uses masked attention because it is trained for next-token prediction. If the model could see future tokens during training, it would leak the answer and cheat.

---

### Q2. What is the difference between attention and FFN?

Attention mixes information across tokens.

The FFN transforms each token independently.

```text
Attention = token-to-token communication
FFN = per-token transformation
```

---

### Q3. Why do we need residual connections?

Residual connections help preserve the original representation and improve gradient flow in deep networks.

They allow the model to learn corrections instead of completely replacing the representation.

---

### Q4. Why is LayerNorm used?

LayerNorm stabilizes training by normalizing token representations.

This is especially important in deep Transformer models where activations can become unstable.

---

### Q5. Why does GPT not use cross-attention?

GPT is decoder-only and has no separate encoder.

All information is placed directly in the prompt, so GPT uses masked self-attention over the full prompt context.

---

### Q6. What weights are learned inside a GPT block?

The learned weights include:

```text
W_Q, W_K, W_V for attention
W_O for attention output projection
FFN weights
LayerNorm scale and shift parameters
```

---

## 22. Memory Hooks

```text
GPT block = masked attention + FFN + residuals + LayerNorm

Masked attention = look left only

Attention = tokens communicate

FFN = each token thinks

Residual = preserve old + add correction

LayerNorm = stabilize training
```

Most important GPT block equations:

$$
A = X + \text{MaskedMHA}(\text{LayerNorm}(X))
$$

$$
Y = A + \text{FFN}(\text{LayerNorm}(A))
$$

---

## 23. Final One-Line Summary

```text
A GPT block takes token representations, lets each token attend to previous tokens using masked attention, transforms each token with an FFN, and stabilizes the process using residual connections and LayerNorm.
```
