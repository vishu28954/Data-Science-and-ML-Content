# GPT Architecture — Part 1: End-to-End GPT Flow

## 1. What is GPT?

GPT is a **decoder-only Transformer** trained using **next-token prediction**.

GPT stands for:

```text
Generative Pre-trained Transformer
```

Meaning:

- **Generative**: it generates text.
- **Pre-trained**: it is first trained on a large amount of text data.
- **Transformer**: it is based on the Transformer architecture.
- **Decoder-only**: it uses only the decoder part of the Transformer.

The core idea is:

```text
GPT takes previous tokens and predicts the next token.
```

---

## 2. Complete GPT Pipeline

The end-to-end GPT flow looks like this:

```text
Text Prompt
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Token Embeddings
    ↓
Positional Information
    ↓
GPT Decoder Blocks × N
    ↓
Final Hidden States
    ↓
Linear Layer over Vocabulary
    ↓
Logits
    ↓
Softmax
    ↓
Next Token Probabilities
    ↓
Generated Token
```

Example:

```text
Input:  I love machine
Output: learning
```

GPT learns the probability of the next token given the previous tokens:

$$
P(\text{next token} \mid \text{previous tokens})
$$

For example:

$$
P(\text{learning} \mid \text{I love machine})
$$

More generally:

$$
P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

This training objective is called **causal language modeling**.

---

## 3. Step 1 — Tokenization

GPT does not directly understand raw text.

So the input text:

```text
I love machine
```

is converted into tokens.

Example:

```text
["I", " love", " machine"]
```

Then each token is converted into a token ID:

```text
[40, 1842, 5780]
```

The model works with numbers, not raw words.

### Why tokenization matters

Tokenization decides how text is broken into smaller units.

Example:

```text
unhappiness → un + happy + ness
```

Modern LLMs often use subword tokenization so they can handle rare words, new words, and different languages more effectively.

---

## 4. Step 2 — Token Embeddings

Token IDs are converted into dense vectors using an embedding matrix.

If:

```text
Vocabulary size = V
Model dimension = d_model
```

then the embedding matrix has shape:

$$
E \in R^{V \times d_{model}}
$$

Example:

If:

```text
V = 50,000
d_model = 768
```

then:

$$
E \in R^{50000 \times 768}
$$

Each token ID selects one row from this matrix.

```text
"I"       → vector of size 768
"love"    → vector of size 768
"machine" → vector of size 768
```

For 3 tokens, the input matrix becomes:

$$
X \in R^{3 \times 768}
$$

where:

```text
3   = number of tokens
768 = embedding dimension
```

---

## 5. Step 3 — Positional Information

Transformers process tokens in parallel.

Because of that, they do not naturally know the order of tokens.

For example:

```text
Dog bites man
```

and:

```text
Man bites dog
```

contain the same words, but the meaning is different.

So GPT needs positional information.

The input representation becomes:

$$
X = \text{Token Embedding} + \text{Positional Information}
$$

Simple memory hook:

```text
Token embedding = what the token is
Positional information = where the token is
```

Earlier GPT models used learned positional embeddings.

Modern GPT-style models often use advanced positional methods such as **RoPE**, also called **Rotary Positional Embeddings**.

---

## 6. Step 4 — GPT Decoder Blocks

After embeddings and positional information, the input goes through multiple GPT blocks.

A GPT model contains many stacked decoder blocks:

```text
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

Each GPT block contains:

```text
Masked Multi-Head Self-Attention
    ↓
Residual Connection
    ↓
Layer Normalization
    ↓
Feed-Forward Network
    ↓
Residual Connection
    ↓
Layer Normalization
```

In modern GPT-style models, the block is often written in **Pre-LayerNorm** form:

$$
a = x + \text{MaskedMHA}(\text{LayerNorm}(x))
$$

$$
out = a + \text{FFN}(\text{LayerNorm}(a))
$$

Where:

```text
MaskedMHA = Masked Multi-Head Attention
FFN       = Feed-Forward Network
```

---

## 7. Step 5 — Masked Self-Attention

Masked self-attention is the heart of GPT.

GPT should not look at future tokens while predicting the next token.

Example sentence:

```text
I love machine learning
```

When GPT is predicting:

```text
machine
```

it can look at:

```text
I love
```

but not:

```text
learning
```

This is why GPT uses a **causal mask**.

The masked attention formula is:

$$
\text{MaskedAttention}(Q,K,V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

where:

```text
Q   = Query
K   = Key
V   = Value
M   = causal mask
d_k = dimension of key vectors
```

The causal mask blocks future tokens.

Memory hook:

```text
GPT attention = look only left.
```

---

## 8. Step 6 — Feed-Forward Network

After attention, each token has gathered context from previous tokens.

Then the Feed-Forward Network transforms each token independently.

The formula is:

$$
\text{FFN}(x) = W_2 \sigma(W_1x + b_1) + b_2
$$

Usually, the FFN expands the dimension and then compresses it back.

Example:

```text
768 → 3072 → 768
```

Memory hook:

```text
Attention mixes tokens.
FFN transforms tokens.
```

Attention allows tokens to communicate with each other.

The FFN performs deeper processing on each token representation.

---

## 9. Step 7 — Final Hidden States

After all GPT blocks, we get final contextual vectors.

For the input:

```text
I | love | machine
```

the model produces:

$$
H \in R^{3 \times d_{model}}
$$

Each token gets one final hidden vector.

For next-token prediction, the hidden vector of the last token is especially important.

Example:

```text
Input: I love machine
```

The hidden state of the token:

```text
machine
```

is used to predict the next token:

```text
learning
```

---

## 10. Step 8 — Linear Layer over Vocabulary

GPT converts the final hidden vector into logits over the vocabulary.

Suppose the final hidden vector is:

$$
h \in R^{d_{model}}
$$

and the vocabulary size is:

$$
V
$$

Then GPT applies a linear layer:

$$
z = hW_{vocab} + b
$$

where:

$$
z \in R^V
$$

If the vocabulary size is 50,000, then:

$$
z \in R^{50000}
$$

These values are called **logits**.

Example:

| Token | Logit |
|---|---:|
| learning | 8.2 |
| intelligence | 6.1 |
| models | 4.8 |
| coffee | 0.5 |

Logits are raw scores. They are not probabilities yet.

---

## 11. Step 9 — Softmax

Softmax converts logits into probabilities.

The formula is:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Example:

| Token | Probability |
|---|---:|
| learning | 0.72 |
| intelligence | 0.12 |
| models | 0.07 |
| coffee | 0.001 |

Now GPT can choose the next token.

```text
I love machine learning
```

---

## 12. Step 10 — Autoregressive Generation

GPT generates text one token at a time.

Step 1:

```text
Input:  I love machine
Output: learning
```

Step 2:

```text
Input:  I love machine learning
Output: because
```

Step 3:

```text
Input:  I love machine learning because
Output: it
```

So generation is autoregressive.

```text
Generate one token, append it, then generate the next token.
```

---

## 13. GPT Training

During training, GPT sees full text sequences.

Example:

```text
I love machine learning
```

The training targets are shifted by one position.

| Input Context | Target Token |
|---|---|
| I | love |
| I love | machine |
| I love machine | learning |

GPT predicts the next token at every position.

The loss function is cross-entropy:

$$
L = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

This means GPT is trained to assign high probability to the actual next token.

Training flow:

```text
Input sequence
    ↓
Predict next-token probabilities
    ↓
Compare with actual next tokens
    ↓
Compute cross-entropy loss
    ↓
Backpropagation
    ↓
Update model weights
```

During training, model weights change.

---

## 14. GPT Inference

During inference, weights do not change.

The model only generates text.

Inference flow:

```text
Prompt
    ↓
Predict next token
    ↓
Append generated token
    ↓
Predict next token
    ↓
Repeat
```

Comparison:

| Stage | What Happens? | Do Weights Change? |
|---|---|---|
| Pretraining | Learn from huge text data | Yes |
| Fine-tuning | Learn from task/domain data | Yes |
| Inference | Generate output from prompt | No |

---

## 15. Training vs Fine-Tuning vs Inference

### Pretraining

The model learns general language patterns from massive text data.

Example objective:

```text
Predict the next token in internet-scale text.
```

Weights are updated.

---

### Fine-Tuning

The already pretrained model is trained further on specific data.

Example:

```text
Instruction → Answer
Question → Response
User query → Ideal assistant reply
```

Weights are updated.

Fine-tuning is training.

---

### Inference

The trained model is used to answer a new prompt.

Example:

```text
Prompt: Explain gradient descent.
Output: Gradient descent is...
```

Weights are not updated.

Inference is only prediction or generation.

---

## 16. GPT Architecture Summary

The GPT architecture is:

```text
Tokenizer
    ↓
Token IDs
    ↓
Token Embeddings
    ↓
Positional Information
    ↓
Decoder-only Transformer Blocks
        - Masked Multi-Head Self-Attention
        - Feed-Forward Network
        - Residual Connections
        - Layer Normalization
    ↓
Final Hidden State
    ↓
Linear Vocabulary Head
    ↓
Logits
    ↓
Softmax
    ↓
Next-Token Probabilities
```

Most important summary:

```text
GPT is a decoder-only Transformer trained to predict the next token.
```

---

## 17. Interview-Level Answer

If an interviewer asks:

```text
Explain GPT architecture.
```

You can answer:

GPT is a decoder-only Transformer architecture trained using causal language modeling. The input text is tokenized into token IDs, converted into token embeddings, combined with positional information, and passed through a stack of Transformer decoder blocks.

Each block contains masked multi-head self-attention, feed-forward networks, residual connections, and layer normalization. The causal mask ensures that each token can only attend to previous tokens and not future tokens.

The final hidden state is projected through a linear layer to produce logits over the vocabulary, and softmax converts those logits into next-token probabilities. During training, GPT minimizes cross-entropy loss over the correct next tokens. During inference, it generates tokens autoregressively one at a time.

---

## 18. Memory Hooks

```text
Tokenizer = text to token IDs

Embedding = token IDs to vectors

Position information = order of tokens

Causal attention = look only left

GPT blocks = process context

Linear head = hidden vector to vocabulary logits

Softmax = probabilities

Generation = one token at a time
```

Most important formula:

$$
P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

That is GPT in one formula.

---

## 19. Common Interview Follow-Up Questions

### Q1. Why is GPT decoder-only?

GPT is decoder-only because it is trained for next-token prediction. It must generate text from left to right, so each token can only attend to previous tokens. This requires causal masked attention, which is the defining property of decoder-style Transformers.

---

### Q2. Why does GPT use masked attention?

GPT uses masked attention to prevent the model from seeing future tokens during training. Without masking, the model could cheat by looking at the token it is supposed to predict.

---

### Q3. What is the output of GPT before softmax?

The output before softmax is a vector of logits over the vocabulary. If the vocabulary size is 50,000, GPT outputs 50,000 logits for each prediction position.

---

### Q4. What is the loss function used in GPT training?

GPT uses cross-entropy loss for next-token prediction:

$$
L = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, ..., x_{t-1})
$$

This loss encourages the model to assign high probability to the actual next token.

---

### Q5. What is the difference between training and inference in GPT?

During training, GPT predicts the next token and updates its weights using backpropagation.

During inference, the weights are fixed, and GPT generates tokens one by one from the prompt.

---

## 20. Final One-Line Summary

```text
GPT converts text into tokens, processes them through causal Transformer blocks, and predicts the next token from the vocabulary.
```
