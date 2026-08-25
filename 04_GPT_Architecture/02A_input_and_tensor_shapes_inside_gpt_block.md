# GPT Architecture — Part 2A: Input and Tensor Shapes Inside a GPT Block

## 1. Why This Part Matters

Before understanding attention, Q/K/V, LayerNorm, FFN, or KV cache, we must understand one basic question:

```text
What exactly enters a GPT block?
```

A GPT block does not receive raw text.

It receives a tensor of token representations.

The most important idea is:

```text
Input to a GPT block = token vectors arranged as a 3D tensor.
```

The usual shape is:

```text
B × T × d_model
```

where:

```text
B       = batch size
T       = sequence length / number of tokens
d_model = hidden dimension / model dimension
```

---

## 2. Where Are We in the GPT Pipeline?

The full GPT pipeline is:

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Token Embeddings
   ↓
Positional Information
   ↓
GPT Block 1
   ↓
GPT Block 2
   ↓
...
   ↓
GPT Block N
   ↓
Final Hidden States
   ↓
Vocabulary Logits
   ↓
Softmax
   ↓
Next Token Probabilities
```

In this part, we are zooming into this section:

```text
Token Embeddings + Positional Information
        ↓
GPT Block
        ↓
Updated Token Representations
```

---

## 3. What Is the Input to a GPT Block?

The input to a GPT block is usually written as:

```text
X
```

But `X` is not one number.

It is not even one vector.

It is a collection of token vectors.

Example sentence:

```text
I love machine learning
```

Suppose tokenization gives:

```text
["I", "love", "machine", "learning"]
```

There are 4 tokens.

Each token becomes a vector.

If the model dimension is 768:

```text
"I"        → vector of size 768
"love"     → vector of size 768
"machine"  → vector of size 768
"learning" → vector of size 768
```

So for one sequence:

```text
X has shape: 4 × 768
```

Meaning:

```text
4   = number of tokens
768 = hidden dimension
```

---

## 4. What Does One Token Vector Contain?

At the very beginning, before the GPT blocks, a token vector contains mainly two types of information:

```text
1. Token identity
2. Positional information
```

Example:

```text
"machine" at position 3
```

The vector represents:

```text
what  = machine
where = position 3
```

But after passing through GPT blocks, the vector becomes contextual.

For example:

```text
The bank approved my loan.
```

and:

```text
The bank of the river was muddy.
```

The word `bank` may start with a similar token embedding in both sentences.

But after GPT blocks, the representation of `bank` becomes different because the surrounding context is different.

So a token vector evolves like this:

```text
Initial embedding:
mostly token identity + position

After early GPT blocks:
local context

After middle GPT blocks:
grammar and sentence relationships

After deeper GPT blocks:
meaning, instruction-following behavior, and generation patterns
```

---

## 5. Shape Without Batch

Let us first ignore batch size.

Suppose the sentence is:

```text
I love machine learning
```

Tokens:

```text
I | love | machine | learning
```

Number of tokens:

```text
T = 4
```

Hidden dimension:

```text
d_model = 768
```

Then the input shape is:

```text
T × d_model = 4 × 768
```

This means:

```text
4 rows
768 columns
```

Each row is one token vector.

```text
Row 1 → representation of "I"
Row 2 → representation of "love"
Row 3 → representation of "machine"
Row 4 → representation of "learning"
```

Each column is one learned hidden dimension.

---

## 6. Shape With Batch Size

In real training, GPT does not process just one sequence at a time.

It processes multiple sequences together in a batch.

Suppose:

```text
B = batch size
T = sequence length
d_model = hidden dimension
```

Then the input shape is:

```text
B × T × d_model
```

Example:

```text
B = 2
T = 4
d_model = 768
```

Then:

```text
X has shape: 2 × 4 × 768
```

Meaning:

```text
2 sequences in the batch
4 tokens per sequence
768 numbers per token
```

Visually:

```text
Batch
│
├── Sequence 1
│   ├── token 1 vector: 768 values
│   ├── token 2 vector: 768 values
│   ├── token 3 vector: 768 values
│   └── token 4 vector: 768 values
│
└── Sequence 2
    ├── token 1 vector: 768 values
    ├── token 2 vector: 768 values
    ├── token 3 vector: 768 values
    └── token 4 vector: 768 values
```

---

## 7. Concrete Tiny Example

To understand shapes clearly, let us use a fake tiny GPT model.

Assume:

```text
Batch size B = 1
Sequence length T = 4
Hidden dimension d_model = 6
```

Sentence:

```text
I love machine learning
```

Then the input shape is:

```text
1 × 4 × 6
```

If we ignore the batch dimension, it becomes:

```text
4 × 6
```

Example token vectors:

```text
I        → [0.2,  0.1, -0.3,  0.7,  0.0,  0.5]
love     → [0.4, -0.2,  0.8,  0.1, -0.5,  0.3]
machine  → [0.9,  0.3, -0.1, -0.4,  0.2,  0.6]
learning → [0.1,  0.7,  0.2,  0.5, -0.2, -0.1]
```

Matrix form:

```text
X =

[
  [0.2,  0.1, -0.3,  0.7,  0.0,  0.5],
  [0.4, -0.2,  0.8,  0.1, -0.5,  0.3],
  [0.9,  0.3, -0.1, -0.4,  0.2,  0.6],
  [0.1,  0.7,  0.2,  0.5, -0.2, -0.1]
]
```

Interpretation:

```text
Rows    = tokens
Columns = hidden features
```

So:

```text
Each row is one token representation.
Each column is one learned feature dimension.
```

---

## 8. What Do Hidden Dimensions Mean?

This is a common confusion.

When we say:

```text
A token vector has 768 dimensions.
```

It does not mean:

```text
dimension 1 = noun
dimension 2 = verb
dimension 3 = emotion
dimension 4 = tense
```

The dimensions are not usually human-interpretable one by one.

Instead, meaning is distributed across many dimensions.

For example, the meaning of:

```text
machine learning
```

is not stored in one single dimension.

It is stored as a pattern across many dimensions.

This is called a **distributed representation**.

Memory hook:

```text
A hidden vector is not a list of human-defined features.
It is a learned pattern spread across many dimensions.
```

---

## 9. What Does a GPT Block Do to X?

A GPT block takes:

```text
Input shape: B × T × d_model
```

and produces:

```text
Output shape: B × T × d_model
```

The shape stays the same.

Example:

```text
Input shape:  2 × 4 × 768
Output shape: 2 × 4 × 768
```

This is extremely important.

A GPT block does not change:

```text
number of sequences
number of tokens
hidden dimension
```

It only updates the meaning stored inside each token vector.

So:

```text
same shape
better representations
```

---

## 10. Why Does the Shape Stay the Same?

GPT stacks many blocks.

For stacking to work, the output of one block must be a valid input to the next block.

```text
X
↓
GPT Block 1
↓
same shape
↓
GPT Block 2
↓
same shape
↓
GPT Block 3
↓
same shape
```

So every GPT block is designed as a shape-preserving transformation:

```text
B × T × d_model → B × T × d_model
```

Internally, the shape may temporarily change.

For example, inside the Feed-Forward Network:

```text
768 → 3072 → 768
```

Inside multi-head attention:

```text
768 → 12 heads × 64 dimensions → 768
```

But the final output returns to:

```text
d_model
```

That is why blocks can be stacked repeatedly.

---

## 11. Token Representations Before and After a Block

Suppose the input is:

```text
The animal was tired
```

Before the GPT block:

```text
"The"     → vector
"animal"  → vector
"was"     → vector
"tired"   → vector
```

After the GPT block:

```text
"The"     → updated vector
"animal"  → updated vector
"was"     → updated vector
"tired"   → updated vector
```

The token count remains the same.

But each vector now contains more context.

For example, the token `tired` can now contain information from:

```text
The animal was
```

So `tired` becomes more context-aware.

---

## 12. GPT-Specific Point: Causal Direction

GPT is a decoder-only model.

That means it uses causal attention.

In causal attention, token position `t` can only use information from positions up to `t`.

For the sentence:

```text
The animal was tired
```

The allowed context is:

| Token | Can Use These Tokens |
|---|---|
| The | The |
| animal | The, animal |
| was | The, animal, was |
| tired | The, animal, was, tired |

So after a GPT block:

```text
representation at position t can only depend on tokens from position 1 to t
```

In simple notation:

```text
Y_t depends only on X_1, X_2, ..., X_t
```

This is the central causal property of GPT.

---

## 13. Difference From BERT-Style Encoder Shapes

BERT and GPT can both use tensors shaped like:

```text
B × T × d_model
```

But the attention pattern is different.

### BERT

BERT is bidirectional.

A token can attend to left and right context.

Example:

```text
The animal was tired
```

The token `animal` can attend to:

```text
The, animal, was, tired
```

### GPT

GPT is causal.

The token `animal` can attend only to:

```text
The, animal
```

So the shape may look similar, but the information flow is different.

Memory hook:

```text
BERT shape and GPT shape may be similar.
But BERT attention is bidirectional.
GPT attention is causal.
```

---

## 14. Every Position Predicts the Next Token During Training

During training, GPT produces an output vector for every token position.

Example input:

```text
I | love | machine | learning
```

GPT produces hidden states:

```text
h_I | h_love | h_machine | h_learning
```

Each hidden state is used to predict the next token.

```text
h_I        → predicts love
h_love     → predicts machine
h_machine  → predicts learning
```

Usually, the final token may predict an end token or may be ignored depending on the training setup.

So GPT training is parallel across positions.

The model computes many next-token predictions in one forward pass.

---

## 15. Input-Target Shifting

This is one of the most important GPT training ideas.

Suppose the text is:

```text
I love machine learning
```

Token IDs:

```text
[10, 245, 3912, 812]
```

During training, GPT uses shifted input and target sequences.

Input IDs:

```text
[10, 245, 3912]
```

Target IDs:

```text
[245, 3912, 812]
```

Meaning:

```text
10   should predict 245
245  should predict 3912
3912 should predict 812
```

In language:

```text
I       → love
love    → machine
machine → learning
```

Memory hook:

```text
GPT training = input sequence shifted left against target sequence.
```

---

## 16. What Happens at Each Position?

Sentence:

```text
I love machine learning
```

At position 1:

```text
Visible context: I
Target token: love
```

At position 2:

```text
Visible context: I love
Target token: machine
```

At position 3:

```text
Visible context: I love machine
Target token: learning
```

Even though the full sequence is passed into the model during training, the causal mask makes each position behave as if it only had access to previous tokens.

That is the power of masked self-attention.

---

## 17. Training vs Inference

Training and inference use the same GPT blocks, but the way we use the outputs is different.

### During Training

The full sequence is available.

Example:

```text
I love machine learning
```

The model predicts the next token at every position.

```text
I       → love
love    → machine
machine → learning
```

So training can compute many predictions in parallel.

But causal masking prevents cheating.

### During Inference

Future tokens do not exist yet.

Example prompt:

```text
I love machine
```

GPT predicts:

```text
learning
```

Then the generated token is appended:

```text
I love machine learning
```

Then GPT predicts another token.

So inference is sequential.

Important difference:

```text
Training:
Full sequence is available.
Causal mask prevents cheating.
Predictions are computed in parallel.

Inference:
Future tokens do not exist.
Tokens are generated one by one.
```

---

## 18. Shape Flow During Training

Suppose:

```text
B = 2
T = 4
d_model = 768
V = 50,000
```

Input token IDs:

```text
shape = B × T
shape = 2 × 4
```

After token embeddings:

```text
shape = B × T × d_model
shape = 2 × 4 × 768
```

After GPT blocks:

```text
shape = B × T × d_model
shape = 2 × 4 × 768
```

After vocabulary projection:

```text
shape = B × T × V
shape = 2 × 4 × 50000
```

Meaning:

```text
For every sequence in the batch,
for every token position,
GPT predicts scores over the entire vocabulary.
```

---

## 19. Why Output Logits Have Shape B × T × V

At each token position, GPT predicts the next token.

If sequence length is 4, GPT gives 4 vocabulary distributions.

Example:

```text
Input positions:
I | love | machine | learning
```

Output logits:

```text
position 1 → scores over vocabulary
position 2 → scores over vocabulary
position 3 → scores over vocabulary
position 4 → scores over vocabulary
```

If vocabulary size is 50,000:

```text
position 1 → 50,000 scores
position 2 → 50,000 scores
position 3 → 50,000 scores
position 4 → 50,000 scores
```

So logits have shape:

```text
B × T × V
```

---

## 20. Cross-Entropy Loss Shape View

During training, GPT outputs logits of shape:

```text
B × T × V
```

The target token IDs have shape:

```text
B × T
```

For each position, cross-entropy compares:

```text
predicted vocabulary distribution
vs
actual next token ID
```

Example:

```text
Prediction at position 1 → target token at position 2
Prediction at position 2 → target token at position 3
Prediction at position 3 → target token at position 4
```

The loss encourages GPT to give high probability to the actual next token.

Simple loss notation:

```text
Loss = negative log probability of the correct next token
```

For the full sequence:

```text
Loss = sum of next-token prediction losses across positions
```

---

## 21. What Is Predicted During Inference?

During inference, we usually care only about the last position.

Prompt:

```text
I love machine
```

GPT produces hidden states:

```text
h_I | h_love | h_machine
```

To generate the next token, we use:

```text
h_machine
```

Why?

Because `h_machine` has access to the full prompt:

```text
I love machine
```

Then:

```text
h_machine → vocabulary logits → softmax → next token
```

So:

```text
During inference, use the last token's output distribution to generate the next token.
```

During training:

```text
Use every position's output distribution to compute loss.
```

---

## 22. Why GPT Does Not Use a Single Sentence Vector Like BERT

In BERT, we often use a special `[CLS]` token representation for sentence-level tasks.

GPT does not naturally use a separate `[CLS]` token in the same way.

GPT is causal.

The last token representation contains information from all previous tokens.

For example:

```text
Explain gradient descent
```

The final hidden state has access to:

```text
Explain + gradient + descent
```

So the final hidden state is used to predict the first answer token.

This is why GPT generation usually depends heavily on the final token's representation.

---

## 23. Why Tensor Shapes Matter for Interviews

Understanding tensor shapes helps answer questions like:

```text
What is the input to a GPT block?
What is the output of a GPT block?
Why does the GPT block preserve shape?
What is the shape of the logits?
Why does GPT output a distribution for every token position during training?
Why does inference use only the last position?
How does causal masking affect information flow?
```

These are common follow-up questions in deep learning and LLM interviews.

---

## 24. Complete Shape Summary

| Stage | Shape | Meaning |
|---|---|---|
| Token IDs | B × T | One token ID per position |
| Token Embeddings | B × T × d_model | One vector per token |
| GPT Block Input | B × T × d_model | Contextual token representations |
| GPT Block Output | B × T × d_model | Updated token representations |
| Final Hidden States | B × T × d_model | Final contextual vectors |
| Vocabulary Logits | B × T × V | Scores over vocabulary per position |
| Targets | B × T | Correct next-token IDs |

---

## 25. Interview-Level Answer

If an interviewer asks:

```text
What is the input and output shape of a GPT block?
```

You can answer:

The input to a GPT block is a 3D tensor of shape batch size by sequence length by hidden dimension, usually written as B × T × d_model. Each token in each sequence is represented by a d_model-dimensional vector. The GPT block applies masked multi-head self-attention and a feed-forward network with residual connections and LayerNorm, but the output shape remains B × T × d_model. The shape is preserved so multiple GPT blocks can be stacked. During training, the final hidden states are projected to vocabulary logits of shape B × T × V, where each position predicts the next token.

---

## 26. Memory Hooks

```text
Input to GPT block = token representations, not raw text.

Shape = B × T × d_model.

Each token vector becomes more contextual after every block.

GPT block preserves shape.

Training uses all positions.

Inference uses the last position.

Causal attention means token t can only use tokens 1 to t.

Final logits shape = B × T × V.
```

---

## 27. Final One-Line Summary

```text
A GPT block receives token representations of shape B × T × d_model, updates each token using causal context, and outputs the same shape so the next GPT block can process it.
```
