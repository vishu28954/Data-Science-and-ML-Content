# Part 10 — Fine-Tuning and Alignment

# Part 3 — Interview Style: Instruction Tuning and Chat Fine-Tuning

## Companion to detailed Part 3 (Questions 1–15)

**Detailed source:** [Part 3 — Instruction Tuning and Chat Fine-Tuning](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%203%20-%20Instruction%20Tuning%20and%20Chat%20Fine%20Tuning.md)

**Purpose:** Answers you can explain naturally in a technical interview. This is based on the **15 questions actually covered in the detailed Part 3 chapter**, not a replacement for it.

**How to use:** First say the simple 30–60-second answer aloud. Then explain the steps and Qwen example. Open the mathematical explanation and follow-ups when deeper reasoning is needed. Avoid memorizing every sentence.

**Project evidence boundary:** The Qwen Little Content VLM is a learning case based on the project description. Exact checkpoint, chat tokens, model modules, training hyperparameters, datasets, LoRA runs and measured improvements are **not established by this chapter**. Treat implementation choices below as proposals rather than accomplished work.

~~~text
Direct answer
    ↓
How it works
    ↓
Qwen Little Content example
    ↓
Mathematics where relevant
    ↓
Interviewer follow-ups and trade-offs
    ↓
One-line memory aid
~~~

---

# Question 1 — What is Instruction Tuning?

### Simple interview answer — say this aloud

Instruction tuning teaches an already pretrained model to follow requests by showing examples of instructions and appropriate responses. For instance, we can ask Qwen to classify a screenshot and give it the correct class as the target during training. The model learns to make that response more likely when it receives a similar request. The important difference from ordinary text continuation is that the input clearly specifies a task the assistant should perform.

### How it works — step by step

First define a task. Then collect requests, their inputs and correct answers. Format them for the model, apply supervised next-token loss to the intended assistant response, and evaluate on examples the model has not seen. Training on varied instructions can improve robustness, but does not guarantee the model will always follow instructions correctly.

### Qwen Little Content example

Instruction: Classify the attached SERP screenshot and return one allowed label. Input: screenshot. Target response: Little Content. The screenshot supplies evidence; the instruction states the task.

### How to understand the mathematics

Let $u$ be the instruction, $I$ the image and $y$ the target response. The model learns $P_\theta(y\mid u,I)$. For a response with tokens $y_1,\ldots,y_T$, teacher forcing trains each token given the correct preceding response tokens.

$$
\mathcal L_{\mathrm{IT}}
=
-\sum_{t=1}^{T}
\log P_\theta(y_t\mid u,I,y_{<t}^{\mathrm{true}})
$$

This is familiar supervised next-token loss, not a separate magical instruction loss.

### Interviewer follow-ups

**Interviewer:** Does instruction tuning mean using RLHF or DPO?

**Answer:** No. Standard instruction tuning can use supervised demonstrations alone. RLHF and DPO are separate preference-based methods discussed later.

**Interviewer:** Does it guarantee the model uses the screenshot rather than shortcuts?

**Answer:** No. That needs checks on image sensitivity, honest labels, and independent evaluation.

### Memory line

> Instruction tuning uses supervised demonstrations to teach what to do when a request is given.

---

# Question 2 — How Is Instruction Tuning Different from SFT?

### Simple interview answer — say this aloud

SFT is the training method: learn from input-and-correct-output examples. Instruction tuning is a way of using SFT to teach request-following behavior. We don't need a different optimizer or a different loss just because we call the data instruction tuning. The difference is what the examples represent.

### How it works — step by step

An ordinary classifier could learn a review-to-sentiment mapping using SFT without any explicit instruction. An instruction-tuned assistant sees something like 'Classify this review' and learns to answer that request. Both may use the same cross-entropy training mechanism.

### Qwen Little Content example

Part 2 supplied the image-to-class supervision and teacher forcing. Part 3 explicitly supplies the task instruction, conversational roles and model-compatible formatting. The training objective can stay the same.

### How to understand the mathematics

General SFT data can be written:

$$
\mathcal D_{\mathrm{SFT}}=\{(x_i,y_i)\}_{i=1}^{N}
$$

Instruction-based examples can be written:

$$
\mathcal D_{\mathrm{IT}}=\{(u_i,x_i,y_i)\}_{i=1}^{N}
$$

The notation highlights the task instruction $u_i$; it does not require a distinct optimization algorithm.

### Interviewer follow-ups

**Interviewer:** Can every SFT dataset be called an instruction-tuning dataset?

**Answer:** Not necessarily. A dataset might simply map inputs to labels without any natural-language request. Instruction tuning specifically teaches behavior conditioned on instructions.

**Interviewer:** Would you change LoRA because you're instruction-tuning?

**Answer:** Not automatically. LoRA controls which parameters train; instruction tuning describes the supervised task examples.

### Memory line

> SFT is the method; instruction tuning is one common goal and dataset format for SFT.

---

# Question 3 — How Do We Construct an Instruction-Tuning Dataset?

### Simple interview answer — say this aloud

I would start by defining exactly what a correct answer means. Then I would collect inputs, assign trustworthy labels, write clear instructions, and keep the correct answer separate from the input. Before training I would check image-label pairing, inconsistent labels, duplicates and leakage between training and evaluation. A large but poorly labelled dataset can teach the wrong behavior.

### How it works — step by step

A sensible record contains an example ID, image reference, instruction or instruction-template identifier, canonical target label and audit metadata. The record is then converted into model-compatible messages. Split strategy should reflect how we expect to use the model—for example, new sites or visual templates.

### Qwen Little Content example

For one SERP screenshot, store the screenshot and reviewed Little Content label separately. Do not write 'Classify this empty useless page' in the user prompt because that wording gives away the answer.

### How to understand the mathematics

Suppose two reviewers agree on 92 of 100 screenshots:

$$
P_{\mathrm{agreement}}=\frac{92}{100}=0.92
$$

This is raw agreement, not proof of correct labels or a chance-corrected score such as Cohen's kappa. Inspect disagreements, especially rare and borderline examples.

### Interviewer follow-ups

**Interviewer:** How can your training data leak the answer?

**Answer:** If positive examples have prompts saying 'low-quality page' while negatives say 'useful page', the model can infer the label from the prompt instead of the screenshot.

**Interviewer:** Why not randomly split every screenshot?

**Answer:** Near-duplicate screenshots, hosts or templates might appear across sets and make generalization look better than it is.

### Memory line

> Define the rule, keep the answer out of the input, audit labels and protect the evaluation split.

---

# Question 4 — What Are System, User and Assistant Roles?

### Simple interview answer — say this aloud

Roles tell a chat model who is speaking. A system message commonly provides overarching instructions, a user message contains the request and its inputs, and an assistant message contains the answer. During SFT, the assistant message can hold the correct target we want the model to learn to produce. These roles are parts of the conversation representation, not separate neural networks.

### How it works — step by step

Build structured messages rather than placing all content in one ambiguous text string. Keep the task policy in an appropriate instruction message, the screenshot and request in the user message, and the class label in the assistant target. The model's supported chat template later turns these roles into its expected sequence.

### Qwen Little Content example

System: Return one of two approved labels. User: Classify this attached screenshot [image]. Assistant target: Not Little Content. The answer should not contain the user's request.

### How to understand the mathematics

This topic is mostly about representation rather than a new equation. The model is trained on a serialized conversation; incorrect placement changes which content becomes input and which becomes a supervised answer.

### Interviewer follow-ups

**Interviewer:** What happens if the user prompt accidentally becomes part of the assistant target?

**Answer:** We train the model to reproduce user text that it should only read as context. Output may become repetitive or badly formatted.

**Interviewer:** Does having a system role guarantee perfect instruction priority?

**Answer:** No. Roles give structure and help training, but are not a guarantee of reliable or secure behavior.

### Memory line

> Roles mark who gave the instruction and who should produce the answer.

---

# Question 5 — What Is a Chat Template?

### Simple interview answer — say this aloud

A chat template is the formatting recipe that turns structured system, user and assistant messages into the specific sequence a chat model expects. Different model families and checkpoints can use different markers. I would use the template belonging to the exact Qwen checkpoint rather than write role markers myself.

### How it works — step by step

Start with structured messages. Apply the model-compatible chat template to place message and speaker boundaries. Tokenize the text using the matching tokenizer, and process the image with the multimodal processor. Inspect a sample of the resulting input and target labels before training.

### Qwen Little Content example

Our user message contains classification text and an actual image content part, while the assistant target contains the class label. The chosen Qwen processor determines how the roles and image input are represented.

### How to understand the mathematics

If $M$ denotes structured messages and $T$ denotes the checkpoint's chat-template procedure:

$$
s=T(M)
$$

For the text portion, tokenization gives:

$$
z=\operatorname{Tokenizer}(s)
$$

A multimodal processor may also produce image tensors and other metadata. The entire model input is not necessarily just a flat text-token list.

### Interviewer follow-ups

**Interviewer:** Why not manually write 'User:' and 'Assistant:'?

**Answer:** That may look correct to a person but differ from the representation the model expects. The wrong format can cause training–inference mismatch.

**Interviewer:** Are a chat template and a tokenizer the same?

**Answer:** No. The template organizes messages; the tokenizer maps textual content and recognized special markers into IDs.

### Memory line

> A chat template converts a conversation into the exact structured sequence the checkpoint expects.

---

# Question 6 — What Are Role Delimiters, Special Tokens and Turn Boundaries?

### Simple interview answer — say this aloud

These markers tell the model where messages start, which role is speaking and where messages or assistant turns end. A message-ending marker and an assistant-turn-ending marker are not necessarily the same thing. Getting them wrong can make the model continue speaking when it should stop or start its answer in the wrong place.

### How it works — step by step

Think of a generic sequence: start-system, policy, end-message; start-user, request, end-message; start-assistant, target label, end-turn. Those names are only illustrative. At inference, the input may end with an unfinished assistant prefix so the next token begins the response.

### Qwen Little Content example

We need to inspect Qwen's actual supported markers, generation prefix and stop behavior. We should supervise the intended response-ending marker where that is part of the output format.

### How to understand the mathematics

If $p$ is the serialized conversation prefix, the first assistant target token is predicted as:

$$
P_\theta(y_1\mid p)
$$

Later targets use:

$$
P_\theta(y_t\mid p,y_{<t}^{\mathrm{true}})
$$

A wrong assistant prefix changes the context $p$, even when the screenshot is identical.

### Interviewer follow-ups

**Interviewer:** Can I assume the special tokens from another Qwen variant are the same?

**Answer:** No. Check the tokenizer and template of the selected model.

**Interviewer:** Why might a fine-tuned model not stop generating?

**Answer:** One possibility is incorrect turn-end targets or mismatched stopping configuration. Inspect both training labels and inference settings.

### Memory line

> Message boundaries tell Qwen where a role starts; turn boundaries tell it when the assistant is finished.

---

# Question 7 — How Does Loss Masking Work in Chat Fine-Tuning?

### Simple interview answer — say this aloud

The model needs to read the instruction and screenshot, but we usually want the training loss to focus on the correct assistant answer. So we keep the prompt as context and ignore its positions when calculating the supervised target loss. The assistant label tokens, and any intended ending token, remain supervised.

### How it works — step by step

Serialize the conversation, then build token labels. Prompt positions get ignored labels; selected assistant target positions keep their correct token IDs. Verify the causal shift: the logits after a prefix are scored against the next token. The screenshot remains part of the input even though it isn't an output text label.

### Qwen Little Content example

For the label Little Content, the real tokenizer may use several tokens. All intended target pieces must remain unmasked. If the first label piece is ignored accidentally, the model loses direct supervision on an important decision.

### How to understand the mathematics

Let $z_t$ be the token at position $t$, and $m_t=1$ only for supervised assistant targets. A representative objective is:

$$
\mathcal L
=
-
\frac{\sum_t m_t\log P_\theta(z_t\mid z_{<t},I)}
{\sum_t m_t}
$$

Here $I$ is the screenshot and $m_t=0$ for ignored target positions. The causal next-token alignment must be handled exactly once.

### Interviewer follow-ups

**Interviewer:** If prompt tokens are masked from loss, can the model still attend to them?

**Answer:** Yes. Loss masking controls which predictions we score; causal and padding masks control visibility and input validity.

**Interviewer:** Loss becomes meaningless—what do you inspect?

**Answer:** Decode non-ignored labels and ensure the assistant response actually has supervised positions. Also check label shifting and padding.

### Memory line

> Keep instructions visible as context; apply training loss to the intended assistant answer.

---

# Question 8 — How Does Multi-Turn Conversation Fine-Tuning Work?

### Simple interview answer — say this aloud

Multi-turn training gives the model a history of several messages before a response. It can help the model learn to use earlier questions, answers and clarifications. We can train only on the final assistant response, or on multiple selected assistant responses, while keeping the appropriate earlier conversation as context.

### How it works — step by step

Prepare the ordered conversation with explicit roles. Choose which assistant turns should be supervised. For each selected assistant target, the model predicts its tokens from the conversation prefix available at that point. Build loss masks accordingly and check truncation doesn't remove essential earlier context.

### Qwen Little Content example

Our Little Content classifier probably doesn't need conversation history for a single screenshot. Multi-turn examples would be useful only if its product design includes clarification or follow-up conversations.

### How to understand the mathematics

For selected assistant turns $a\in\mathcal A$:

$$
\mathcal L_{\mathrm{multi}}
=
-\sum_{a\in\mathcal A}
\sum_{t=1}^{T_a}
\log P_\theta(y_{a,t}\mid\mathrm{history}_a,y_{a,<t}^{\mathrm{true}})
$$

The set $\mathcal A$ tells us which assistant messages are trained as targets; it need not include every assistant message.

### Interviewer follow-ups

**Interviewer:** Do you need to calculate loss on every previous assistant message?

**Answer:** No. You can supervise only the final response or several selected responses depending on the task.

**Interviewer:** Does training on multi-turn data give the model memory across unrelated requests?

**Answer:** No. It uses the conversation context supplied in that input; the dataset alone doesn't create persistent memory.

### Memory line

> Multi-turn SFT teaches the model to answer using conversation history, with carefully selected target turns.

---

# Question 9 — What Is the Difference Between Single-Turn and Multi-Turn Training?

### Simple interview answer — say this aloud

In single-turn training, one user request leads to one target assistant response. In multi-turn training, the model sees a conversation containing earlier messages and may need that history to answer. Neither approach is automatically better: I would choose based on what the deployed system actually needs to do.

### How it works — step by step

Single-turn examples are shorter and easier to format, mask and evaluate. Multi-turn examples help when earlier context or clarifications matter, but they are longer and may include response styles that conflict with a strict one-label classifier.

### Qwen Little Content example

Our initial Qwen service receives a screenshot and should return Little Content or Not Little Content. A single-turn example directly matches that job. Introducing explanatory multi-turn dialogue may add complexity without helping the binary decision.

### How to understand the mathematics

There is no special loss function unique to single-turn versus multi-turn SFT. The difference is the amount and structure of the conditioning context and which assistant spans are supervised.

### Interviewer follow-ups

**Interviewer:** Should we use multi-turn data because Qwen is a chat model?

**Answer:** Not by default. We need a product requirement and evidence that history use matters.

**Interviewer:** Why could multi-turn training hurt a strict classifier?

**Answer:** If its targets encourage long explanatory replies, they may conflict with the requirement to output exactly one class label.

### Memory line

> Match the number of turns in training to how the real application will be used.

---

# Question 10 — What Happens When Conversations Are Truncated or Packed?

### Simple interview answer — say this aloud

Truncation cuts a sequence to fit the model's length limit. Packing places multiple examples together to waste less space on padding. Both can improve efficiency, but incorrect handling can remove important instructions or labels, or let one training example see another example's content.

### How it works — step by step

For truncation, make sure the screenshot context and supervised answer are still present. For packing, inspect whether examples are properly separated and whether attention is allowed across boundaries. Text-packing shortcuts can be unsafe for multimodal examples if image positions and metadata no longer align.

### Qwen Little Content example

If we cut away the assistant label, Qwen may have no useful supervised target. If we crop away the screenshot's meaningful content, the class might change. If we naively pack unrelated screenshot conversations together, one example may condition on another.

### How to understand the mathematics

For two independent examples A and B, correctly isolated packed attention can be described schematically as:

$$
M=
\begin{bmatrix}
M_A&0\\
0&M_B
\end{bmatrix}
$$

The off-diagonal parts represent blocked cross-example attention. Exact masks may be encoded using booleans or additive values, not necessarily literal zeros.

### Interviewer follow-ups

**Interviewer:** Doesn't a separator token automatically prevent cross-example attention?

**Answer:** No. A separator is not necessarily an attention barrier. The trainer must explicitly handle example isolation where required.

**Interviewer:** Would you start by enabling aggressive packing for Qwen VLM?

**Answer:** No. I'd establish correct multimodal batching first, then test only packing supported by the checkpoint and trainer.

### Memory line

> Truncation can remove context; packing can mix contexts. Check both before optimizing speed.

---

# Question 11 — How Does Multimodal Instruction Tuning Work?

### Simple interview answer — say this aloud

Multimodal instruction tuning teaches the model to follow a written request while using another input type, such as an image. For our Qwen classifier, the screenshot supplies visual information, the instruction tells Qwen to classify it, and the target is a short text label. The model does not need to generate the screenshot to learn from it.

### How it works — step by step

Load the actual image, process it with the checkpoint-compatible image processor, create the user message with the text and image content, apply the chat template, and prepare assistant-token labels. The model combines visual and text information to predict response tokens. Gradients update whichever components are trainable.

### Qwen Little Content example

A text record saying 'image attached' isn't enough. The processed batch must actually contain the image inputs and any visual metadata required by the selected Qwen VLM. Check that switching screenshots changes the model's evidence, while keeping the instruction fixed.

### How to understand the mathematics

With screenshot $I$, instruction $u$ and assistant tokens $y$:

$$
P_\theta(y\mid I,u)
=
\prod_{t=1}^{T}
P_\theta(y_t\mid I,u,y_{<t}^{\mathrm{true}})
$$

The loss can apply only to intended assistant tokens, although their probabilities depend on $I$. Frozen visual layers can still provide conditioning representations.

### Interviewer follow-ups

**Interviewer:** Can Qwen train successfully if the image tensor is missing?

**Answer:** The text-only computation might still run, but it wouldn't teach visual grounding. Check processor outputs and image-to-label alignment.

**Interviewer:** Why is image preprocessing more than a minor detail?

**Answer:** Cropping or reducing resolution can remove the very layout or text evidence needed for classification.

### Memory line

> Multimodal SFT uses the image as evidence and the answer tokens as supervision.

---

# Question 12 — How Do Data Quality, Diversity and Dataset Mixtures Affect Instruction Tuning?

### Simple interview answer — say this aloud

The model learns from patterns in its training examples. Quality means correct images and labels. Diversity means realistic types of pages and layouts. A dataset mixture means how frequently examples come from different classes or tasks. If we get these wrong, the model can memorize shortcuts or focus on the wrong behavior even if training loss looks good.

### How it works — step by step

Audit human labels and disagreements; include confusing but important examples; check hosts/templates for duplicates; record class balance and source proportions. If mixing general chat or image tasks with Little Content detection, measure whether that mixture helps the actual classifier.

### Qwen Little Content example

Thousands of near-identical screenshots from one host may look like lots of data but teach little about unseen hosts. Sparse pages containing one genuinely useful answer should be included as hard negatives.

### How to understand the mathematics

For data sources $k$ with sampling weights $w_k$:

$$
\mathcal L_{\mathrm{mix}}
=
\sum_{k=1}^{K}w_k
\mathbb E_{(x,y)\sim\mathcal D_k}[\mathcal L_\theta(x,y)],
\qquad
\sum_k w_k=1
$$

Changing the weights changes the contribution of each source. Also, if 90% of examples are negative, always predicting negative can give 90% accuracy while positive-class recall is zero.

### Interviewer follow-ups

**Interviewer:** Why isn't accuracy sufficient on an imbalanced dataset?

**Answer:** A majority-class predictor can look accurate while missing all positive cases. Examine precision, recall, false positives and intended operating thresholds.

**Interviewer:** Should we add general chat data automatically?

**Answer:** No. Only if it serves a requirement and controlled validation shows the mixture helps.

### Memory line

> Clean, diverse examples teach the intended rule; duplicated or biased examples teach shortcuts.

---

# Question 13 — How Does Instruction Tuning Generalize to Unseen Instructions?

### Simple interview answer — say this aloud

We want Qwen to learn what a request means, not memorize one exact sentence. If several equivalent phrasings consistently lead to the same intended behavior, the model may learn to handle new wording. But that doesn't prove it will also handle unseen screenshots. We should test instruction wording and visual generalization separately.

### How it works — step by step

Hold out some instruction paraphrases during training, and separately hold out some hosts or layouts. Test four cases: familiar wording/familiar layout, new wording/familiar layout, familiar wording/new layout, and new wording/new layout. This helps distinguish language sensitivity from visual-domain failure.

### Qwen Little Content example

A model might respond correctly to 'Classify this screenshot' but fail with 'Apply the Little Content policy'. Or it could handle both phrases while failing on an unfamiliar page template. These are different weaknesses.

### How to understand the mathematics

No additional loss function is required just to evaluate paraphrase robustness. Use the same task metrics while changing one evaluation dimension at a time. Holdout design is more informative here than a new equation.

### Interviewer follow-ups

**Interviewer:** Do many paraphrases guarantee generalization?

**Answer:** No. They can be repetitive or accidentally leak the answer. Paraphrases must preserve the task definition and be evaluated on genuinely unseen examples.

**Interviewer:** What matters most for our production classifier?

**Answer:** Generalization to new hosts and layouts is central; broad prompt flexibility is secondary if production uses a stable prompt.

### Memory line

> Test new wording and new screenshots separately; passing one doesn't prove the other.

---

# Question 14 — What Are Common Failures in Chat Fine-Tuning, and How Do We Debug Them?

### Simple interview answer — say this aloud

I would debug a chat-fine-tuning failure from the data outward before changing LoRA rank or training for longer. First verify the raw input and label. Then inspect the serialized chat format, processed image input, supervised target tokens and loss mask. Finally check training gradients and performance on genuinely held-out examples.

### How it works — step by step

If Qwen repeats the user prompt, inspect roles and target masks. If it outputs paragraphs instead of labels, inspect canonical targets and inference scoring. If it ignores screenshots, verify image tensors and image-label pairing. If loss is low but new-host precision is poor, check data leakage, class balance and overfitting.

### Qwen Little Content example

Take a screenshot whose correct class is Not Little Content. Inspect the raw image; decode the prompt and the target tokens; confirm the processor passed that image; verify the model is not receiving the target class within the user instruction. Only after that debug optimization and capacity.

### How to understand the mathematics

A useful diagnostic is whether supervised label positions actually exist and whether gradients on intended trainable parameters are nonzero. A small-set overfit test can check training wiring, but passing it is not evidence of generalization.

### Interviewer follow-ups

**Interviewer:** Why can applying the chat template twice be harmful?

**Answer:** It can duplicate or scramble role delimiters, changing the input the model sees.

**Interviewer:** Does low training loss prove correct fine-tuning?

**Answer:** No. It can reflect leaked labels, duplicates, masked-target mistakes or overfitting. Task evaluation is required.

### Memory line

> Debug the raw data, template, image, labels, masking, gradients and evaluation—in that order.

---

# Question 15 — How Would We Build an Instruction-Tuning Dataset for Qwen Little Content Detection?

### Simple interview answer — say this aloud

I would first define the class policy and audit labelled SERP screenshots. Then I would store each screenshot with its reviewed label and metadata, create a consistent classification instruction, and build system/user/assistant messages. I would use the exact Qwen checkpoint's image processor and chat template, train on the correct assistant label tokens using SFT, and initially compare a modest LoRA setup against prompt-only Qwen. I'd validate on unseen hosts and layouts and measure precision, recall, errors and cost before claiming success.

### How it works — step by step

The design sequence is: policy → labelled screenshot → raw record → structured messages → checkpoint-compatible multimodal processing → assistant-target loss mask → teacher-forced forward pass → selected LoRA updates → independent evaluation. Debug a few processed examples and a tiny batch before scaling. Keep parameter selection and output-threshold experiments separate.

### Qwen Little Content example

System instruction: Return exactly one allowed label. User: Classify this image, with the real screenshot attached. Assistant target: Little Content or Not Little Content. The target is not included in the user message. Q/V language-side LoRA and a possible connector are theoretical experiments, not confirmed historical implementation details.

### How to understand the mathematics

The assistant-only loss is:

$$
\mathcal L
=
-\frac{\sum_t m_t\log P_\theta(z_t\mid z_{<t},I)}{\sum_t m_t}
$$

For task-level evaluation:

$$
\mathrm{Precision}=\frac{TP}{TP+FP},
\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

The first optimizes assistant-token probabilities; the second pair evaluates binary classification behavior. They answer different questions.

### Interviewer follow-ups

**Interviewer:** Why not select LoRA rank and Qwen special tokens right now?

**Answer:** The actual checkpoint and its modules must be inspected, and rank should be chosen through resource-aware validation, not fabricated as a deployed fact.

**Interviewer:** What is the biggest failure mode in this data pipeline?

**Answer:** A model that appears well-trained but learns labels from leaked prompts, repeated hosts or incorrectly paired screenshots instead of visual content.

### Memory line

> Correct data representation comes before optimization; reliable held-out classification comes before success claims.

---

---

# Integrated Interview Case 1 — Fix a Broken Chat Fine-Tuning Example

**Interviewer:** Training loss is low, but the model frequently replies with "Classify this screenshot" before its class label. How would you investigate?

**Strong answer:** I would first inspect how one dataset record was converted into structured messages. The user request should be input context, not part of the assistant target. Next I would render the chat template and decode the non-ignored loss labels. If the instruction appears among supervised assistant target tokens, we are teaching Qwen to repeat it. I would correct message roles and masking, verify causal label alignment, then run a small controlled test before full retraining.

**Follow-up:** Would higher LoRA rank solve this?

**Answer:** No. More adapter capacity does not correct supervision on the wrong tokens.

---

# Integrated Interview Case 2 — The Model Is Ignoring Images

**Interviewer:** Qwen predicts Little Content for almost everything, even pages full of useful content. Training appears normal.

**Strong answer:** First verify that each label belongs to the correct screenshot and that the user instruction isn't leaking the correct class. Then inspect the multimodal processor output to confirm the actual image tensor and related metadata are passed to the model. Compare image swaps under a fixed instruction as a diagnostic, and look for cropping or resolution problems. Check class imbalance and held-out host/layout errors. Only then consider whether language LoRA, a multimodal connector or vision-side adaptation needs to change.

**Follow-up:** Can a frozen vision encoder still support classification learning?

**Answer:** Yes. Frozen visual representations still condition predictions, and trainable downstream parts can learn to use them.

---

# Integrated Interview Case 3 — Choosing Single-Turn, Multi-Turn and Packing

**Interviewer:** You have 100,000 labelled screenshot examples and a strict one-label classification API. Should you create multi-turn dialogues and aggressively pack everything for speed?

**Strong answer:** I would start with single-turn examples because they match the API. Multi-turn data is useful only if conversational history is a product requirement. I would first implement correct padding and multimodal batching. Sequence packing is an optimization to investigate only if the selected Qwen trainer can preserve image alignment, correct loss targets and appropriate example isolation. I would prioritize verified supervision over a speculative speed improvement.

**Follow-up:** Why might a packed sample be wrong despite a separator token?

**Answer:** A separator does not necessarily block causal attention across examples. The trainer needs correct handling of attention and boundaries.

---

# Integrated Interview Case 4 — Design the Dataset and Evaluation

**Interviewer:** How would you build a trustworthy instruction-tuning dataset for a SERP screenshot classifier?

**Strong answer:** Define the two classes and ambiguous edge cases; audit annotation quality and screenshot-label pairing; store raw image references separately from target responses; construct a stable instruction and role-correct messages; use the exact checkpoint's chat template and image processor; inspect the assistant-only labels and end-of-turn behavior; begin with prompt-only and efficient fine-tuning baselines. Evaluate on independent sites and layouts and measure precision, recall and false positives, not only training loss.

**Follow-up:** A model scores well on new prompts but fails on new hosts. What does that indicate?

**Answer:** It may have learned instruction phrasing robustly while failing to generalize its visual classification rule. Separate the two evaluation dimensions.

---

# Part 3 — Rapid Interview Memory Map

~~~text
What should the model do?
        ↓
Instruction + correct answer
        ↓
Dataset quality and label policy
        ↓
System / User / Assistant roles
        ↓
Checkpoint-specific chat template
        +
Image processor
        ↓
Tokens + real image inputs
        ↓
Correct assistant targets and loss masks
        ↓
Teacher-forced SFT / selected LoRA parameters
        ↓
Check unseen hosts, unseen layouts, instruction variations
        ↓
Reliable production-format class responses
~~~

# Concepts Often Confused

| A | B | Difference |
|---|---|---|
| SFT | Instruction tuning | SFT is a training method; instruction tuning teaches request-following behavior |
| Chat template | Tokenizer | Template arranges messages; tokenizer converts text and markers into IDs |
| System role | Guaranteed instruction security | Roles represent message structure, not an infallible security boundary |
| Loss mask | Attention mask | Loss mask selects supervised targets; attention mask controls visibility and padding |
| End of message | End of assistant turn | The format may distinguish finishing one message from finishing the full assistant turn |
| Single-turn | Multi-turn | One request/response versus history-dependent conversation |
| Truncation | Packing | Cut long input versus combine independent examples |
| Text saying "[image]" | Actual image input | Placeholder words cannot replace multimodal processor inputs |
| Training loss | Business metric | Token likelihood isn't the same as precision/recall on the real task |
| Instruction robustness | Visual generalization | New wording and new screenshot layouts require separate evaluation |

# What We Should Not Claim Yet

Do not invent an exact Qwen checkpoint, official special token spellings, image grid scheme, processed visual-token count, LoRA rank, optimizer or actual training results. For interviews, separate **reported project scope**, **proposed design**, and **experimentally verified facts**.

# One-Line Summary

> **Instruction and chat fine-tuning are about making the model see the right request, image and conversation structure—and teaching it to generate only the intended assistant response.**
