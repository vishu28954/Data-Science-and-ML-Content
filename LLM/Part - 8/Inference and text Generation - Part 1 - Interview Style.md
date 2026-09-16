# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 1 — Interview Style

## Covers

```text
8.1 What happens during inference?
8.2 Prompt tokens
```

This file contains only interview-style answers. The detailed-study notes remain separate.

---

# 8.1 What Happens During Inference? — Interview Style

## 1. What does inference mean in an LLM?

### Interview answer

Inference means using a trained model with fixed parameters to generate predictions for new input.

For a decoder-only language model, the core task is next-token prediction. If the current context is:

$$
x_1,x_2,\ldots,x_t
$$

then the model estimates:

$$
P\left(x_{t+1}\mid x_1,x_2,\ldots,x_t\right)
$$

So the model is not directly generating an entire paragraph at once. It repeatedly predicts a probability distribution over the next token, selects one token according to a decoding rule, appends it to the context, and repeats.

---

## 2. How is inference different from training?

### Interview answer

During training, the model updates its parameters to minimize a loss function. During inference, the learned parameters remain fixed.

Training can be written conceptually as:

$$
\theta^* = \underset{\theta}{\operatorname{argmin}}\;\mathcal{L}(\theta)
$$

where $\theta$ represents model parameters and $\mathcal{L}$ is the training loss.

During inference, I simply evaluate the trained model using $\theta^*$:

$$
P_{\theta^*}\left(x_{t+1}\mid x_{\le t}\right)
$$

The important systems difference is that inference normally does not require backpropagation or weight updates, so gradients do not need to be stored for training.

---

## 3. What is the end-to-end inference pipeline?

### Interview answer

At a high level, the pipeline is:

```text
Raw text
→ tokenizer
→ token IDs
→ embedding vectors
→ Transformer layers
→ contextual hidden states
→ LM head
→ vocabulary logits
→ probability distribution
→ decoding strategy
→ selected next token
```

After the token is selected, it is appended to the context and the model repeats the process autoregressively.

---

## 4. Why does the model need token IDs and embeddings instead of raw text?

### Interview answer

Neural networks operate on numerical tensors, so raw text first has to be converted into discrete token IDs.

Those IDs are categorical labels, not meaningful numerical magnitudes. Therefore, each token ID indexes a learned embedding vector.

If the vocabulary size is $|V|$ and the model hidden dimension is $d_{model}$, then the embedding matrix is:

$$
E\in\mathbb{R}^{|V|\times d_{model}}
$$

For token $x_i$, the model retrieves:

$$
e_i=E[x_i]
$$

where:

$$
e_i\in\mathbb{R}^{d_{model}}
$$

This converts a discrete symbol into a continuous vector that can participate in Transformer matrix operations.

---

## 5. Why does the model also need positional information?

### Interview answer

Token embeddings mainly tell the model what the token is, but the model also needs to know where that token occurs in the sequence.

For example, “Dog bites man” and “Man bites dog” contain similar tokens but have different meanings because order changes.

So Transformers include position-dependent information, for example positional embeddings or mechanisms such as rotary positional embeddings.

Without positional information, self-attention alone would not correctly distinguish token order.

---

## 6. What is the final hidden state used for?

### Interview answer

After all Transformer layers, each token position has a contextual hidden representation.

For the latest relevant position $t$:

$$
h_t\in\mathbb{R}^{d_{model}}
$$

The hidden state summarizes the model's contextual representation at that position.

The next step is to convert this $d_{model}$-dimensional vector into one score for every vocabulary token.

---

## 7. How does the hidden state become vocabulary logits?

### Interview answer

The model uses a language-model output projection, often called the LM head.

A simplified form is:

$$
z=W_{LM}h_t+b
$$

where:

$$
W_{LM}\in\mathbb{R}^{|V|\times d_{model}}
$$

and:

$$
z\in\mathbb{R}^{|V|}
$$

So if the vocabulary has 100,000 tokens, the model produces 100,000 logits, one for each possible next token.

---

## 8. What is a logit?

### Interview answer

A logit is an unnormalized score assigned to a candidate token.

For example:

$$
z=[4,2,1]
$$

These are not probabilities yet because they do not necessarily lie between zero and one or sum to one.

Softmax converts them into a probability distribution.

---

## 9. How does softmax convert logits into probabilities?

### Interview answer

Softmax is:

$$
P(i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

It exponentiates each logit to make the values positive and then normalizes them so the probabilities sum to one.

For logits:

$$
[4,2,1]
$$

we get approximately:

```text
0.844
0.114
0.042
```

### Important production detail

For numerical stability, implementations commonly subtract the maximum logit first:

$$
m=\max_j z_j
$$

and then compute:

$$
P(i)=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
$$

This gives the same probabilities but avoids numerical overflow.

---

## 10. Do we always need softmax for greedy decoding?

### Interview answer

No. Softmax preserves the ordering of logits, so:

$$
\arg\max_i z_i
=
\arg\max_i P(i)
$$

Therefore, if the generation strategy only needs the highest-scoring token, it can select the maximum logit directly without explicitly computing normalized probabilities.

Softmax becomes essential when I need the actual probability distribution for sampling methods such as temperature, top-k, or top-p.

---

## 11. Does the model itself choose the next token?

### Interview answer

I separate the model from the decoding policy.

The model produces logits or probabilities. The decoding policy decides which token to select.

For example, the same model output can be used with greedy decoding, random sampling, temperature scaling, top-k, top-p, or beam search.

So changing decoding behavior does not necessarily require changing the model weights.

---

## 12. Why is LLM generation called autoregressive?

### Interview answer

Because every new prediction depends on the tokens already present in the sequence.

If the current sequence is:

$$
x_1,x_2,\ldots,x_n
$$

and the model selects $x_{n+1}$, then the next prediction becomes:

$$
P\left(x_{n+2}\mid x_1,x_2,\ldots,x_n,x_{n+1}\right)
$$

The newly generated token becomes part of the context used to generate the following token.

This dependency is what eventually makes normal text generation sequential.

---

## 13. What important edge cases can occur during inference?

### Interview answer

Several subtle issues can occur.

An empty visible user message may still produce model input because the serving system adds system instructions or chat-template tokens. Very long prompts can exceed the context budget. Nearly tied logits can produce different outputs under different numerical precisions or hardware. Quantization can slightly change logits. And using the wrong tokenizer for a model can produce syntactically valid tensors but semantically incorrect token IDs.

The tokenizer-model mismatch is especially dangerous because the code may run without an obvious exception.

---

# Complete 8.1 Interview Answer

If asked:

```text
What happens during inference in a decoder-only LLM?
```

You can say:

```text
During inference, the model parameters are fixed and the model repeatedly performs next-token prediction.

The input text is tokenized into token IDs. Each token ID is mapped to an embedding vector and combined with position-dependent information before passing through the Transformer layers. The final Transformer hidden state at the current position is projected through the language-model head into one logit for every vocabulary token.

Those logits represent unnormalized next-token scores. If probabilities are required, softmax converts them into a categorical probability distribution. A decoding strategy such as greedy decoding or sampling then selects one token.

The selected token is appended to the context, and the model repeats the same process for the next token. This autoregressive dependency is the core mechanism behind text generation.
```

---

# 8.2 Prompt Tokens — Interview Style

## 14. What exactly is a token?

### Interview answer

A token is one entry in the tokenizer's finite vocabulary.

If the vocabulary is:

$$
V=\{v_1,v_2,\ldots,v_{|V|}\}
$$

then tokenization maps text into IDs:

$$
\text{text}\longrightarrow[x_1,x_2,\ldots,x_n]
$$

where each $x_i$ is an index into that vocabulary.

The model does not directly process words or characters. It processes token IDs and their learned vector representations.

---

## 15. Is one word always one token?

### Interview answer

No. A word may map to one token or several subword or byte-oriented tokens depending on the tokenizer.

Tokenization is influenced by vocabulary construction, frequency, language, punctuation, whitespace, code, numbers, URLs, and Unicode characters.

That is why word count, character count, and token count are different measurements.

---

## 16. Why not use one token for every whole word?

### Interview answer

A pure word-level vocabulary would become extremely large and would handle rare words, new names, misspellings, morphology, multiple languages, URLs, and code poorly.

Subword or byte-aware tokenization allows the model to represent rare or unseen strings using reusable smaller pieces while keeping the vocabulary finite.

---

## 17. Why does token count matter computationally?

### Interview answer

Token count becomes the sequence-length variable $n$ inside the Transformer.

For ordinary dense self-attention over a prompt, the pairwise attention-score matrix has shape:

$$
n\times n
$$

so the number of pairwise score positions is:

$$
n^2
$$

For example:

$$
1000^2=1{,}000{,}000
$$

while:

$$
2000^2=4{,}000{,}000
$$

So doubling prompt length creates four times as many pairwise attention-score positions.

I would not say total inference runtime is always exactly quadratic, because feed-forward layers, optimized kernels, model width, hardware, and other operations also matter. But this explains why long prompts can make prefill attention expensive.

---

## 18. What is the shape of the prompt representation passed into the Transformer?

### Interview answer

If the prompt contains $n$ tokens and the model dimension is $d_{model}$, then after embedding lookup the token vectors can be stacked into:

$$
X\in\mathbb{R}^{n\times d_{model}}
$$

For example, if:

$$
n=2048
$$

and:

$$
d_{model}=4096
$$

then:

$$
X\in\mathbb{R}^{2048\times4096}
$$

This tensor shape is important because later attention operations derive queries, keys, and values from this representation.

---

## 19. Are all prompt tokens typed by the user?

### Interview answer

No. In a chat system, the full model prompt may include the system message, previous conversation history, the current user message, role delimiters, separators, and other special tokens.

A useful conceptual equation is:

$$
N_{prompt}
=
N_{system}
+
N_{history}
+
N_{user}
+
N_{special}
$$

So a short latest user message can still produce a very large model input if the conversation history is long.

---

## 20. What are special tokens?

### Interview answer

Special tokens are reserved vocabulary entries used for structural or control purposes, such as beginning-of-sequence, end-of-sequence, message-role boundaries, separators, or padding.

Mathematically they are still token IDs with embeddings, but the model or serving system gives them special structural meaning.

A robust serving system should distinguish literal user text from actual control-token IDs so that user text cannot accidentally alter the intended chat template.

---

## 21. What is the context window?

### Interview answer

The context window is the maximum active token budget supported by the model and runtime.

If:

- $C$ = context capacity
- $P$ = prompt tokens
- $G$ = generated tokens retained in context

then a basic constraint is:

$$
P+G\le C
$$

For example, with:

$$
C=8192
$$

and:

$$
P=7000
$$

the theoretical remaining space is at most:

$$
G\le1192
$$

The actual API may impose a smaller output limit for policy or serving reasons.

---

## 22. What happens if the prompt exceeds the context window?

### Interview answer

The behavior depends on the serving system. It may reject the request, truncate part of the prompt, summarize older history, or route to a larger-context model.

The important point is that truncation changes the information available to the model.

Mathematically, the model is now estimating:

$$
P(x_{t+1}\mid \text{truncated context})
$$

instead of:

$$
P(x_{t+1}\mid \text{full original context})
$$

Those two conditional distributions can be different.

---

## 23. Why can two prompts with similar meaning have different token counts?

### Interview answer

Because tokenization depends on the learned vocabulary and encoding strategy, not just semantic meaning or character count.

Rare words, unusual spacing, long numbers, URLs, hashes, code, Unicode text, misspellings, and different languages can tokenize very differently.

This affects context usage, memory, prefill latency, and sometimes cost.

---

## 24. Does token count alone determine inference cost?

### Interview answer

No. Sequence length is important, but compute also depends on model architecture.

Important variables include:

$$
n=\text{sequence length}
$$

$$
d_{model}=\text{hidden width}
$$

$$
L=\text{number of Transformer layers}
$$

as well as feed-forward width, attention-head configuration, numerical precision, sparsity, kernels, and hardware.

So two models processing the same 4000-token prompt can have very different compute costs.

---

## 25. What prompt-token edge cases should an ML engineer know?

### Interview answer

I would watch for several cases: a short visible message with a very long conversation history, rare strings that explode into many tokens, code or URLs consuming context quickly, padding in batched inference, truncation that removes important instructions, special-token injection, and tokenizer-model mismatch.

The tokenizer-model mismatch is particularly important because it can be a silent failure: the tensor shapes can remain valid while the model receives token IDs with the wrong learned meanings.

---

# Complete 8.2 Interview Answer

If asked:

```text
What are prompt tokens and why do they matter for inference?
```

You can say:

```text
The model never directly receives words. A tokenizer converts the input text into a finite sequence of token IDs, and each ID indexes a learned embedding vector.

If the prompt contains n tokens and the model width is d_model, the prompt representation has shape n by d_model. Token count is important because it determines how many sequence positions the Transformer must process and, for dense self-attention during prefill, the attention-score matrix grows as n by n.

The full prompt also includes more than the latest user message. It may contain system instructions, previous conversation history, special tokens, and message-role formatting.

Prompt and generated tokens share a finite context budget, which can be written approximately as P + G <= C. If the prompt is too long, the system must reject, truncate, summarize, or otherwise reduce the context, and that can change model behavior.

So tokenization is not just a text-preprocessing detail; it directly affects model semantics, memory, latency, and available generation capacity.
```

---

# Combined Interview Answer

If asked:

```text
Walk me through what happens when I send a prompt to an LLM.
```

Answer:

```text
The serving system first tokenizes the text into token IDs. Those IDs may represent subwords, bytes, punctuation, or special control tokens rather than whole words. Each token ID is converted into a learned embedding vector and combined with position-dependent information.

The resulting prompt tensor is passed through the Transformer layers to produce contextual hidden states. The hidden state at the current prediction position is projected through the language-model head into one logit per vocabulary token. If probabilities are needed, softmax normalizes the logits into a next-token probability distribution.

A decoding strategy then selects one token. That token is appended to the context, and the model predicts the following token conditioned on the entire sequence seen so far.

The key mathematical object is P(x_{t+1} | x_1,...,x_t): the probability of the next token given the current context. Because generation repeatedly conditions on previously selected tokens, text generation is autoregressive.
```

---

# Crisp Version

```text
LLM inference is repeated next-token prediction.

Text is tokenized into IDs, IDs are mapped to embeddings, Transformer layers produce contextual hidden states, the LM head projects the latest hidden state into vocabulary logits, and a decoding rule selects the next token.

Prompt length matters because the model processes tokens rather than words, attention during prefill depends strongly on sequence length, and prompt plus generated tokens must fit inside the context window.
```

---

# Most Important Memory Lines

```text
Inference = fixed model weights + repeated next-token prediction.
```

```text
The model produces logits; the decoding policy chooses the token.
```

```text
LLMs do not process words directly. They process token IDs and tensors.
```

```text
Prompt tokens affect semantics, context usage, memory, prefill cost, and generation budget.
```

---

The next detailed-study block is:

```text
8.3 Prefill phase
8.4 Decode phase
```
