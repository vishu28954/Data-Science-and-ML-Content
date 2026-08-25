# GPT Architecture — Part 2J: What Changes During Fine-Tuning Inside GPT Blocks

## 1. Why This Part Matters

In the previous parts, we studied the internal structure of one GPT block:

```text
Part 2A → Input and tensor shapes

Part 2B → LayerNorm

Part 2C → Q, K, V projections

Part 2D → Causal masked self-attention

Part 2E → Multi-head attention

Part 2F → Residual connections

Part 2G → Feed-forward network / MLP

Part 2H → Pre-LayerNorm vs Post-LayerNorm

Part 2I → Parameter count, compute cost, and memory
```

Now we ask a very important practical question:

```text
When we fine-tune GPT, what actually changes inside the model?
```

The simple answer is:

```text
Fine-tuning changes some or all of the trainable weights.

The architecture usually stays the same.
```

That means GPT still has:

```text
token embeddings

positional information

GPT blocks

masked self-attention

feed-forward networks

residual connections

LayerNorm

vocabulary head
```

But the numerical values inside the trainable matrices may change.

---

## 2. What Is Fine-Tuning?

Fine-tuning means taking a pretrained model and training it further on a smaller, more specific dataset.

Pretraining teaches the model broad language patterns.

Fine-tuning teaches the model a more specific behavior.

Examples:

```text
Pretraining:
learn general language, grammar, facts, reasoning patterns, code, world knowledge

Fine-tuning:
learn to answer as an assistant, follow instructions, write in a style, classify text, use domain-specific terminology
```

Memory hook:

```text
Pretraining gives broad capability.

Fine-tuning shapes behavior for a task.
```

---

## 3. What Does Not Change During Fine-Tuning?

Usually, fine-tuning does not change the model architecture.

The model still has the same structure:

```text
same number of layers

same hidden dimension

same number of attention heads

same FFN size

same causal mask

same tokenizer

same forward-pass structure
```

The computation graph remains mostly the same.

What changes is the value of trainable parameters.

---

## 4. What Can Change During Fine-Tuning?

Depending on the fine-tuning method, some or all of these parameters can change:

```text
token embedding weights

attention projection weights:
W_Q, W_K, W_V, W_O

feed-forward / MLP weights:
W_up, W_down, W_gate

LayerNorm parameters:
gamma and beta

vocabulary head weights

adapter weights if using PEFT methods like LoRA
```

But not every fine-tuning method updates all of them.

---

## 5. Full Fine-Tuning vs Parameter-Efficient Fine-Tuning

There are two broad categories:

```text
1. Full fine-tuning

2. Parameter-efficient fine-tuning
```

### Full fine-tuning

In full fine-tuning:

```text
most or all original model weights are updated
```

This means GPT can update:

```text
attention weights

FFN weights

LayerNorm parameters

embeddings

output head
```

This is powerful but expensive.

---

### Parameter-efficient fine-tuning

In parameter-efficient fine-tuning, also called PEFT:

```text
the base model is mostly frozen

only a small number of additional or selected parameters are trained
```

Examples:

```text
LoRA

QLoRA

adapters

prefix tuning

prompt tuning
```

This is cheaper and more memory-efficient.

---

## 6. Full Fine-Tuning: What Changes?

In full fine-tuning, the pretrained model starts with learned weights.

During training:

```text
1. Input tokens go through the model.

2. Model predicts output tokens.

3. Cross-entropy loss is calculated.

4. Gradients are computed.

5. Optimizer updates the model weights.
```

So the actual matrices inside GPT blocks change.

Example:

```text
Before fine-tuning:
W_Q = pretrained query projection matrix

After fine-tuning:
W_Q = updated query projection matrix
```

This can happen for many matrices.

---

## 7. What Changes in Attention During Fine-Tuning?

Inside attention, the main trainable matrices are:

```text
W_Q

W_K

W_V

W_O
```

Fine-tuning can change these matrices.

That means the model may learn different attention behavior.

For example, after fine-tuning on legal documents, the model may learn to attend more strongly to:

```text
clauses

dates

parties

obligations

exceptions

definitions
```

After fine-tuning on code, it may learn to attend more strongly to:

```text
variable names

function definitions

imports

syntax structure

previous lines of code
```

Memory hook:

```text
Fine-tuning attention changes what the model pays attention to.
```

---

## 8. What Changes in Q, K, and V?

Recall:

```text
Q = what this token is looking for

K = what each token offers for matching

V = what information each token provides
```

Fine-tuning can change:

```text
how queries are formed

how keys are matched

what values carry forward
```

Example:

```text
Before fine-tuning:
a token may attend broadly to general context.

After domain fine-tuning:
the same token may attend more strongly to domain-specific patterns.
```

So fine-tuning changes the learned projections that create Q, K, and V.

---

## 9. What Changes in Multi-Head Attention?

Each attention head has its own learned projection subspace.

Fine-tuning can shift what different heads specialize in.

Example:

```text
Before fine-tuning:
Head 3 may focus on general syntax.

After fine-tuning on customer support data:
Head 3 may become more useful for recognizing complaint patterns.
```

Important:

```text
The number of heads does not change.

The head dimension does not change.

The learned weights inside heads may change.
```

---

## 10. What Changes in the Output Projection W_O?

After all heads produce outputs, GPT concatenates them and applies:

```text
W_O
```

The output projection mixes information across heads.

Fine-tuning can change how head outputs are combined.

Example:

```text
If some heads become useful for domain-specific behavior,
W_O can learn to give those head outputs more useful influence.
```

Memory hook:

```text
W_Q, W_K, W_V change attention patterns.

W_O changes how attention-head information is mixed.
```

---

## 11. What Changes in the FFN / MLP?

The FFN contains large matrices such as:

```text
W_up

W_down

W_gate
```

In full fine-tuning, these weights can change.

The FFN transforms each token independently after attention has gathered context.

So fine-tuning can change how the model processes contextual token representations.

Example:

```text
Fine-tuning on medical text may change how the FFN represents symptoms, diagnosis terms, and treatment language.

Fine-tuning on coding data may change how the FFN represents functions, variables, and code patterns.
```

Memory hook:

```text
Fine-tuning FFN changes how the model interprets and transforms context.
```

---

## 12. Why FFN Fine-Tuning Can Be Powerful

The FFN often contains many parameters.

Because of this, changing FFN weights can significantly affect model behavior.

Recall from Part 2I:

```text
Attention parameters:
N_attn = 4d_model²

Standard FFN parameters:
N_ffn = 2d_model d_ff

Gated FFN parameters:
N_gated_ffn = 3d_model d_ff
```

Since `d_ff` is often large, the FFN/MLP can contain a major fraction of the model's parameters.

This means FFN updates can strongly influence the model's learned transformations.

---

## 13. What Changes in LayerNorm?

LayerNorm has learned parameters:

```text
gamma

beta
```

These control scaling and shifting after normalization.

In full fine-tuning, these parameters may update.

This can change the scale and distribution of hidden representations.

However, LayerNorm has very few parameters compared to attention and FFN.

So LayerNorm updates are usually small in parameter count but can still affect training stability and behavior.

---

## 14. What Changes in Token Embeddings?

Token embeddings map token IDs to vectors.

In full fine-tuning, embedding weights may change.

This can shift how the model represents tokens.

Example:

```text
In a medical fine-tuned model,
domain-specific terms may get representations better suited to medical contexts.

In a code fine-tuned model,
programming tokens may get more useful representations.
```

However, many fine-tuning setups freeze embeddings to reduce training cost or avoid damaging general language understanding.

---

## 15. What Changes in the Vocabulary Head?

The vocabulary head maps final hidden states to logits over the vocabulary.

Shape:

```text
B × T × d_model → B × T × V
```

Fine-tuning can update the vocabulary head.

This changes how hidden states are converted into next-token probabilities.

Example:

```text
Instruction fine-tuning may increase probability of helpful response patterns.

Domain fine-tuning may increase probability of domain-specific terminology.

Style fine-tuning may increase probability of certain phrasing patterns.
```

Memory hook:

```text
The vocabulary head affects what tokens become more likely.
```

---

## 16. What Does Not Change: Causal Mask

The causal mask does not change during fine-tuning.

GPT still follows the same rule:

```text
token t can attend only to tokens 1 through t
```

The model is still autoregressive.

It still predicts the next token from previous tokens.

Fine-tuning does not remove the no-looking-ahead rule.

Memory hook:

```text
Fine-tuning changes weights, not causality.
```

---

## 17. What Does Not Change: Residual Connections

Residual connections do not have trainable parameters.

They are just additions.

Example:

```text
A = X + MaskedMHA(LayerNorm(X))

Y = A + FFN(LayerNorm(A))
```

During fine-tuning, the attention and FFN outputs may change because their weights change.

But the residual addition itself does not change.

Memory hook:

```text
Residual connections are fixed pathways.

The updates flowing through them can change.
```

---

## 18. Fine-Tuning Still Uses Cross-Entropy Loss

Most supervised fine-tuning of GPT-style models still uses next-token prediction with cross-entropy loss.

The model receives an input sequence and learns to predict target tokens.

Simplified loss:

```text
Loss = -sum log probability of correct target tokens
```

Example:

```text
Prompt:
Explain gradient descent simply.

Target response:
Gradient descent is an optimization method...
```

The model learns to increase the probability of the target response tokens.

Memory hook:

```text
Fine-tuning still trains GPT by making correct target tokens more likely.
```

---

## 19. Supervised Fine-Tuning Data Format

A common supervised fine-tuning dataset contains examples like:

```text
Instruction:
Explain overfitting in machine learning.

Response:
Overfitting happens when a model performs well on training data but poorly on unseen data...
```

For chat models, the data may look like:

```text
User:
What is gradient descent?

Assistant:
Gradient descent is an optimization algorithm used to minimize a loss function...
```

The model is trained to produce the assistant response.

---

## 20. Input and Target During Fine-Tuning

For a training example:

```text
User: Explain attention.
Assistant: Attention lets tokens look at other tokens.
```

The model receives the full text as tokens.

But loss is often applied mainly to the assistant tokens.

Example:

```text
Input tokens:
User: Explain attention. Assistant: Attention lets tokens look at other tokens.

Loss applied to:
Attention lets tokens look at other tokens.
```

This teaches the model how to respond, not just how to copy the prompt.

---

## 21. Fine-Tuning Changes Probabilities

Fine-tuning changes the output probabilities produced by the model.

Before fine-tuning, the model may respond generally.

After fine-tuning, it may become more likely to produce:

```text
specific style

specific format

domain terminology

instruction-following behavior

structured answers

safer responses

tool-use patterns
```

The architecture is the same, but the probability distribution changes.

Memory hook:

```text
Fine-tuning reshapes the model's next-token probability distribution.
```

---

## 22. Example: Before and After Fine-Tuning

Prompt:

```text
Explain random forest.
```

Before fine-tuning, a base pretrained model may continue text in a broad or inconsistent way.

After instruction fine-tuning, it is more likely to respond like:

```text
Random forest is an ensemble learning algorithm that builds many decision trees and combines their predictions...
```

The model did not get a new architecture.

Its weights were adjusted so helpful assistant-style answers became more likely.

---

## 23. Instruction Fine-Tuning

Instruction fine-tuning trains the model on instruction-response pairs.

Goal:

```text
make the model follow user instructions
```

Examples:

```text
Summarize this paragraph.

Write Python code for binary search.

Explain this concept in simple language.

Classify this review as positive or negative.

Convert this text into JSON.
```

Instruction fine-tuning changes the model's behavior by making instruction-following outputs more likely.

---

## 24. Domain Fine-Tuning

Domain fine-tuning trains the model on data from a specific domain.

Examples:

```text
medical notes

legal contracts

financial reports

customer support tickets

code repositories

scientific papers
```

Goal:

```text
make the model better at domain-specific language and patterns
```

Important:

```text
Domain fine-tuning does not automatically guarantee factual accuracy.

It mainly changes the model's learned behavior and probability patterns.
```

For factual and up-to-date information, RAG is often useful.

---

## 25. Fine-Tuning vs RAG

Fine-tuning and RAG solve different problems.

| Need | Better Approach |
|---|---|
| Teach response style | Fine-tuning |
| Teach output format | Fine-tuning |
| Teach domain language | Fine-tuning |
| Use private documents | RAG |
| Use frequently changing facts | RAG |
| Reduce hallucination with citations | RAG |
| Teach task behavior | Fine-tuning |
| Retrieve exact source content | RAG |

Memory hook:

```text
Fine-tuning changes model behavior.

RAG changes what information the model sees at inference time.
```

---

## 26. Fine-Tuning vs Prompt Engineering

Prompt engineering changes the input prompt.

Fine-tuning changes model weights.

### Prompt engineering

```text
No weight updates

Fast to try

Good for simple behavior changes

Can be inconsistent for complex tasks
```

### Fine-tuning

```text
Updates weights or adapter weights

Requires data and training

Better for repeated behavior patterns

Can reduce prompt length for repeated instructions
```

Memory hook:

```text
Prompting changes instructions.

Fine-tuning changes learned behavior.
```

---

## 27. Fine-Tuning vs Pretraining

Pretraining is large-scale training from scratch or near-scratch on massive data.

Fine-tuning is smaller-scale training after pretraining.

| Feature | Pretraining | Fine-Tuning |
|---|---|---|
| Data size | Massive | Smaller |
| Goal | General language modeling | Specific behavior/task |
| Cost | Very high | Lower |
| Starting point | Random or partially trained weights | Pretrained model |
| Output | Base model | Specialized model |

Memory hook:

```text
Pretraining builds the foundation.

Fine-tuning adapts the foundation.
```

---

## 28. What Is LoRA?

LoRA stands for:

```text
Low-Rank Adaptation
```

It is a parameter-efficient fine-tuning method.

Instead of updating the full original weight matrix, LoRA freezes the original matrix and adds a small trainable update.

Simplified idea:

```text
Original weight:
W

LoRA update:
Delta_W

Effective weight during fine-tuning:
W_effective = W + Delta_W
```

But `Delta_W` is low-rank, so it uses far fewer trainable parameters.

---

## 29. LoRA Formula

A simple LoRA-style update can be written as:

```text
W_effective = W + Delta_W

Delta_W = A B
```

where:

```text
W = frozen pretrained weight matrix

A and B = small trainable low-rank matrices

Delta_W = learned adapter update
```

If the rank is small, LoRA trains far fewer parameters than full fine-tuning.

Memory hook:

```text
LoRA freezes the big matrix and trains a small correction.
```

---

## 30. Where Is LoRA Applied?

LoRA can be applied to different matrices inside GPT.

Common targets include:

```text
W_Q

W_K

W_V

W_O

W_up

W_down

W_gate
```

Many setups apply LoRA especially to attention projections:

```text
W_Q

W_V
```

or to all attention projections and MLP projections.

Which matrices to target depends on:

```text
task

model architecture

memory budget

quality requirement

training setup
```

---

## 31. What Changes During LoRA Fine-Tuning?

During LoRA fine-tuning:

```text
base model weights usually stay frozen

LoRA adapter weights are updated
```

The original matrix does not change.

Instead, the effective matrix changes because LoRA adds a learned update.

Example:

```text
Original:
W_Q is frozen

LoRA trains:
Delta_W_Q

Effective:
W_Q_effective = W_Q + Delta_W_Q
```

So the model behavior changes, but the original pretrained weights remain unchanged.

---

## 32. Why LoRA Is Efficient

LoRA is efficient because it trains fewer parameters.

Full fine-tuning may update billions of parameters.

LoRA may update only a small fraction.

Benefits:

```text
lower GPU memory

faster training

smaller checkpoint files

easier to store multiple task adapters

less risk of damaging the base model
```

Tradeoff:

```text
may be less flexible than full fine-tuning for very large behavior changes
```

---

## 33. What Is QLoRA?

QLoRA is a memory-efficient fine-tuning method that combines:

```text
quantized base model

LoRA adapters
```

Simple idea:

```text
store the base model in low precision

freeze the base model

train LoRA adapters
```

This greatly reduces memory usage and makes fine-tuning large models more accessible.

Memory hook:

```text
QLoRA = quantized base model + LoRA training.
```

---

## 34. Full Fine-Tuning vs LoRA

| Feature | Full Fine-Tuning | LoRA |
|---|---|---|
| Base weights updated? | Yes | Usually no |
| Extra adapter weights? | No | Yes |
| Memory cost | Higher | Lower |
| Training cost | Higher | Lower |
| Checkpoint size | Large | Small |
| Flexibility | Very high | High but constrained |
| Risk of forgetting | Higher | Usually lower |
| Common use | Maximum adaptation | Efficient adaptation |

---

## 35. What Is Catastrophic Forgetting?

Catastrophic forgetting happens when fine-tuning makes the model lose some useful general abilities.

Example:

```text
A model fine-tuned too aggressively on a narrow dataset may become worse at general questions.
```

This can happen because the weights shift too much toward the fine-tuning data.

Ways to reduce forgetting:

```text
use high-quality diverse data

use lower learning rate

use fewer epochs

mix general data with task data

use LoRA instead of full fine-tuning

evaluate on general benchmarks and task benchmarks
```

Memory hook:

```text
Fine-tuning should adapt the model, not erase its general ability.
```

---

## 36. Overfitting During Fine-Tuning

Overfitting happens when the model memorizes the fine-tuning dataset instead of learning general behavior.

Signs:

```text
training loss keeps improving

validation performance stops improving or gets worse

model repeats training-style phrases too much

model performs poorly on slightly different examples
```

Ways to reduce overfitting:

```text
use more data

clean duplicate examples

use validation set

early stopping

lower learning rate

regularization

data augmentation

LoRA with controlled rank
```

---

## 37. Learning Rate During Fine-Tuning

Fine-tuning usually uses a smaller learning rate than pretraining.

Why?

Because pretrained weights already contain useful knowledge.

If the learning rate is too high, training may damage the model.

Memory hook:

```text
Fine-tuning should gently adjust pretrained weights.
```

---

## 38. What Happens During Backpropagation?

During fine-tuning, the model performs:

```text
forward pass

loss calculation

backward pass

optimizer update
```

In full fine-tuning:

```text
gradients are computed for most or all model weights
```

In LoRA:

```text
gradients are computed mainly for LoRA adapter weights
```

Frozen parameters do not update.

---

## 39. Fine-Tuning and Optimizer State

Full fine-tuning requires optimizer states for many parameters.

For Adam-like optimizers, this can be memory-heavy.

LoRA requires optimizer states only for adapter parameters.

So LoRA saves memory in two ways:

```text
fewer trainable parameters

fewer optimizer states
```

This is one reason LoRA is popular for practical fine-tuning.

---

## 40. Fine-Tuning and Inference

After fine-tuning, inference is mostly the same forward pass.

The model still generates tokens autoregressively:

```text
prompt → logits → next token → append → repeat
```

For full fine-tuning:

```text
the model weights themselves are updated
```

For LoRA:

```text
the base model plus adapter is used
```

Sometimes LoRA adapters are merged into the base weights for inference.

---

## 41. What Does It Mean to Merge LoRA?

LoRA uses:

```text
W_effective = W + Delta_W
```

During inference, we can either:

```text
1. Keep W and Delta_W separate

2. Merge Delta_W into W
```

Merging means:

```text
W_merged = W + Delta_W
```

Then inference can use the merged matrix directly.

This can simplify deployment.

---

## 42. Fine-Tuning and Model Behavior

Fine-tuning can change model behavior in many ways.

Examples:

```text
more helpful answers

specific response format

domain-specific vocabulary

shorter or longer responses

better classification behavior

better tool-use behavior

better alignment with preference data
```

But fine-tuning is not magic.

It depends heavily on:

```text
data quality

data quantity

training objective

hyperparameters

evaluation
```

---

## 43. Fine-Tuning and Evaluation

Before deploying a fine-tuned model, evaluate it.

Evaluation should include:

```text
task performance

general capability

safety behavior

format correctness

hallucination rate

latency

cost

regression tests
```

For example, if you fine-tune for customer support, evaluate:

```text
answer correctness

tone

policy compliance

escalation behavior

refusal behavior

citation correctness if RAG is used
```

Memory hook:

```text
Never trust fine-tuning only because training loss went down.
```

---

## 44. Fine-Tuning and Data Quality

Fine-tuning quality depends strongly on data quality.

Bad data teaches bad behavior.

Good fine-tuning data should be:

```text
accurate

consistent

well-formatted

deduplicated

representative of real use cases

aligned with desired behavior

safe and policy-compliant
```

Small high-quality datasets can sometimes beat larger noisy datasets.

Memory hook:

```text
Fine-tuning is only as good as the examples you train on.
```

---

## 45. Common Confusion: Does Fine-Tuning Add New Layers?

Usually, no.

Full fine-tuning updates existing model weights.

LoRA adds small adapter matrices, but the main architecture stays the same.

So:

```text
Fine-tuning usually changes parameters, not the main architecture.
```

---

## 46. Common Confusion: Does Fine-Tuning Change the Tokenizer?

Usually, no.

Most fine-tuning keeps the same tokenizer.

Changing the tokenizer is difficult because embeddings and vocabulary head are tied to token IDs.

If new tokens are added, the embedding matrix and vocabulary head may need resizing and training.

For most fine-tuning:

```text
same tokenizer

same vocabulary

same token IDs
```

---

## 47. Common Confusion: Does Fine-Tuning Store New Facts Reliably?

Not always.

Fine-tuning can make the model more likely to produce certain information, but it is not ideal for frequently changing facts or exact document lookup.

For exact and updated knowledge, RAG is often better.

Use fine-tuning for:

```text
behavior

style

format

task pattern

domain language
```

Use RAG for:

```text
fresh facts

private documents

source-grounded answers

citations

large knowledge bases
```

---

## 48. Common Confusion: Is LoRA the Same as Prompting?

No.

Prompting changes only the input text.

LoRA trains new adapter parameters.

So LoRA changes the model's effective weights, while prompting does not.

---

## 49. Common Confusion: Is Fine-Tuning Always Better Than Prompting?

No.

Prompting may be enough when:

```text
the task is simple

the behavior change is small

you do not have training data

you need fast iteration
```

Fine-tuning is useful when:

```text
the same behavior is needed repeatedly

prompting becomes too long

format consistency matters

you have high-quality examples

you need domain-specific response style
```

---

## 50. Common Confusion: Is Fine-Tuning Always Better Than RAG?

No.

Fine-tuning and RAG solve different problems.

Fine-tuning changes behavior.

RAG provides external information at inference time.

For many production systems, the best solution is:

```text
RAG + prompt engineering + optional fine-tuning
```

---

## 51. Fine-Tuning Inside One GPT Block: Summary

Inside one GPT block, fine-tuning can update:

```text
attention projection weights

FFN / MLP weights

LayerNorm parameters

adapter weights if using LoRA or another PEFT method
```

Fine-tuning does not update:

```text
causal mask

residual connection structure

number of heads

sequence length rule

basic block architecture
```

The block still computes:

```text
A = X + MaskedMHA(LayerNorm(X))

Y = A + FFN(LayerNorm(A))
```

But the learned functions inside `MaskedMHA` and `FFN` may change.

---

## 52. Interview-Level Answer

If an interviewer asks:

```text
What changes inside GPT blocks during fine-tuning?
```

You can answer:

During fine-tuning, the architecture of GPT usually stays the same, but some or all trainable parameters are updated. In full fine-tuning, weights inside attention projections such as `W_Q`, `W_K`, `W_V`, and `W_O`, as well as FFN or MLP weights, LayerNorm parameters, embeddings, and the vocabulary head may change. These updates change how the model forms attention patterns, transforms token representations, and assigns probabilities to output tokens.

In parameter-efficient fine-tuning methods like LoRA, the base model is mostly frozen and small low-rank adapter matrices are trained instead. These adapters modify the effective weights of selected matrices, often in attention or MLP layers, while using far fewer trainable parameters. The causal mask, residual connections, number of heads, and overall GPT block structure do not change.

---

## 53. Common Interview Follow-Up Questions

### Q1. Does fine-tuning change the GPT architecture?

Usually, no.

Fine-tuning changes weights, not the main architecture.

---

### Q2. Which weights can change during full fine-tuning?

Full fine-tuning can update:

```text
attention weights

FFN weights

LayerNorm parameters

embedding weights

vocabulary head weights
```

---

### Q3. What changes during LoRA fine-tuning?

The base model is usually frozen.

Small trainable low-rank adapter matrices are updated.

The effective weight becomes:

```text
W_effective = W + Delta_W
```

---

### Q4. Does the causal mask change during fine-tuning?

No.

The causal mask is fixed.

GPT still cannot attend to future tokens.

---

### Q5. Do residual connections change during fine-tuning?

No.

Residual connections have no trainable parameters.

They remain fixed addition paths.

---

### Q6. Does fine-tuning change attention behavior?

Yes.

If attention projection weights are updated, the model may learn different attention patterns.

---

### Q7. Does fine-tuning change FFN behavior?

Yes.

If FFN weights are updated, the model changes how it transforms token representations.

---

### Q8. Is fine-tuning useful for adding fresh facts?

Usually, RAG is better for fresh or changing facts.

Fine-tuning is better for behavior, style, format, and task patterns.

---

### Q9. Why is LoRA memory-efficient?

Because it trains only small adapter matrices instead of updating all model weights.

It also needs optimizer states only for those adapter parameters.

---

### Q10. What is catastrophic forgetting?

Catastrophic forgetting happens when fine-tuning damages the model's general abilities by over-specializing it on a narrow dataset.

---

## 54. Memory Hooks

```text
Fine-tuning changes weights, not usually architecture.

Full fine-tuning updates many original parameters.

LoRA freezes the base model and trains small adapter updates.

Attention fine-tuning changes what tokens attend to.

FFN fine-tuning changes how token representations are transformed.

LayerNorm parameters may change, but they are small.

Causal mask does not change.

Residual connections do not change.

Tokenizer usually does not change.

Fine-tuning changes next-token probabilities.

Prompting changes input.

RAG changes available context.

Fine-tuning changes learned behavior.

Good fine-tuning depends on good data.
```

---

## 55. Final One-Line Summary

```text
During fine-tuning, GPT usually keeps the same architecture, but updates selected trainable weights or adapter parameters, changing attention patterns, representation transformations, and next-token probabilities.
```
