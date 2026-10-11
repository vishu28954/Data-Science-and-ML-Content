# Part 10 — Fine-Tuning and Alignment

# Part 3 — Instruction Tuning and Chat Fine-Tuning (Questions 1–15)

## 10.3 Instruction Tuning and 10.4 Chat Format Training

**Detailed Study Mode.** This chapter uses our original **visible Story Bridge → natural question → detailed answer → mathematics/example → Qwen application → Design Decision → Project Build Record → next Story Bridge** pattern. The separate Interview Style track is not used here.

**Previously:** [Part 2 — SFT Foundations](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202.md) and [Part 2B — PEFT and LoRA](https://github.com/vishu28954/Data-Science-and-ML-Content/blob/main/LLM/Part%20-%2010/Fine%20Tuning%20and%20Alignment%20-%20Part%202B%20-%20PEFT%20and%20LoRA.md).

**Part 1:** Why fine-tune? **Part 2:** How does teacher-forced loss work? **Part 2B:** Which parameters learn through LoRA? **Part 3:** How should instructions, conversations and multimodal targets be structured so that optimization actually teaches the intended behavior?

**Evidence boundary:** The project description refers to Qwen VLM Little Content detection using labelled Bing SERP screenshots and a pipeline of roughly 50K URLs/day. Our specific model version, chat template, LoRA settings, data size, splits and measured fine-tuning outcomes have **not** been established as implementation facts. We describe these as proposed designs, not completed experiments.

## Learning map

| Question | Subject |
|---|---|
| 1 | What is instruction tuning? |
| 2 | SFT versus instruction tuning |
| 3 | Building instruction datasets |
| 4 | System, user and assistant roles |
| 5 | Chat templates |
| 6 | Role delimiters and special tokens |
| 7 | Assistant-only loss in chat training |
| 8 | Multi-turn conversation training |
| 9 | Single-turn versus multi-turn data |
| 10 | Truncation and packing |
| 11 | Multimodal instruction tuning |
| 12 | Dataset quality and mixtures |
| 13 | Generalizing to unseen instructions |
| 14 | Common chat-fine-tuning failures |
| 15 | Complete Qwen training-data design |

---

### Story Bridge 1 — SFT Can Make a Label More Likely, But How Does It Teach the Meaning of a Request?

Part 2 taught us how to increase the probability of a correct assistant response. But the same screenshot could arrive with instructions phrased in many ways. A model trained on one fixed sentence might accidentally rely on the wording rather than the task. So we need to understand what it means to tune on **instructions**, not just labels.

## Question 1 — What is Instruction Tuning?

### Simple intuition

Instruction tuning means **teaching a pretrained model to respond to requests by showing examples of requests and appropriate responses**.

A pretrained model is very good at continuing text. But continuing text is not the same as obeying a request such as "summarize this paragraph", "translate this sentence", or "classify this image".

In instruction tuning, the request becomes part of the input and the correct response becomes the supervised target.

### Step-by-step technical mechanism

1. Collect requests representing tasks we want the assistant to perform.
2. Pair each request with the desired response, ideally checked for correctness.
3. Format the request and response using the model's supported input representation.
4. Train with the same teacher-forced next-token likelihood we studied in Part 2.
5. Evaluate on unseen requests and examples, including alternative phrasings.

Let $u$ be the instruction, $x$ additional input such as an image, and $y$ the assistant response.

$$
P_\theta(y\mid u,x)
=
\prod_{t=1}^{T}P_\theta(y_t\mid u,x,y_{<t}^{\mathrm{true}})
$$

We can minimize:

$$
\mathcal L_{\mathrm{IT}}
=
-\sum_{t=1}^{T}\log P_\theta(y_t\mid u,x,y_{<t}^{\mathrm{true}})
$$

This is not a new kind of loss. The important change is **what the labelled data asks the model to learn**.

### Worked example

~~~text
Instruction: Translate "Good morning" into Hindi.
Correct response: सुप्रभात

Instruction: Return the sentiment of this review.
Input: "The food was excellent."
Correct response: Positive

Instruction: Classify the screenshot using one allowed label.
Image: [processed SERP screenshot]
Correct response: Little Content
~~~

Training on different requests encourages the model to learn what the instructions call for; it does not guarantee reliable generalization by itself.

### Common misunderstanding

Instruction tuning does **not** automatically require RLHF or DPO. Those are different methods studied later. It also does not guarantee accuracy, visual grounding or perfect output formatting.

### Qwen Project Application

Our screenshot tells Qwen *what it should inspect*. The instruction tells Qwen *what decision to make and what format to use*. Both must be present as conditioning information, while the correct canonical class is the supervised target.

### Design Decision — Q1

Begin with a stable classification instruction and two canonical responses. We may later introduce carefully controlled instruction paraphrases to evaluate robustness, not to hide changes to the label policy.

### Project Build Record Update — Q1

| Element | Proposed state | Type |
|---|---|---|
| Behavioral goal | Follow screenshot classification instructions | Design Decision |
| Instruction diversity | Test controlled paraphrases | Design Decision |
| Guaranteed instruction following | Not assumed | Open Question |

---

### Story Bridge 2 — This Sounds Like SFT. Are We Learning a New Algorithm?

Our equation above looks almost identical to SFT from Part 2. That is intentional. To avoid mixing terms, we must separate the **training procedure** from the **kind of behavior the examples teach**.

## Question 2 — How Is Instruction Tuning Different from SFT?

### Simple intuition

**SFT is a general method. Instruction tuning is a common application of SFT.**

SFT says: show the model labelled input–output examples and adjust it to predict the correct outputs.

Instruction tuning says: organize those examples as requests and demonstrations of how to respond.

| Dataset | SFT? | Instruction tuning? |
|---|---|---|
| Review text mapped directly to positive/negative without a request | Yes | Not necessarily |
| "Translate this sentence" followed by translation | Yes | Yes |
| Screenshot + classification request + class label | Yes | Yes |
| More domain text for continued next-token training | Usually classed as continued pretraining | Not necessarily |

Names vary in literature, but this distinction remains useful.

### Mathematical connection

General supervised data:

$$
\mathcal D_{\mathrm{SFT}}=\{(x_i,y_i)\}_{i=1}^{N}
$$

Instruction-style data:

$$
\mathcal D_{\mathrm{IT}}=\{(u_i,x_i,y_i)\}_{i=1}^{N}
$$

Once serialized, both can be trained with conditional causal next-token loss. **They need not use different optimizers, Transformer architectures or LoRA variants.**

### Qwen example

Part 2 already defined assistant-only classification loss. Part 3 adds the **explicit request, message roles and compatible chat formatting**. We can reuse the same SFT objective rather than invent another objective.

### Design Decision — Q2

Continue with teacher-forced SFT and the previously proposed PEFT approach. Treat instruction wording and conversation construction as **separate data-engineering choices**.

### Project Build Record Update — Q2

| Element | Proposed state | Type |
|---|---|---|
| Optimization | Assistant-token SFT | Existing Design Decision |
| Instruction tuning | Instruction-conditioned demonstration dataset | Design Decision |
| Preference training | Not assumed necessary | Open Question |

---

### Story Bridge 3 — We Know the Objective, But What Does a Training Record Actually Contain?

An optimizer works with tensors and labels. We therefore need an exact way to store requests, screenshots, human labels and metadata before converting them to a model input. Poor dataset design can make even perfect training code learn the wrong thing.

## Question 3 — How Do We Construct an Instruction-Tuning Dataset?

### First intuition

An instruction-tuning dataset contains **requests paired with correct responses**. If the request needs extra information, such as an image or paragraph, that information must also be included.

A basic text-oriented record might be:

~~~json
{
  "instruction": "Classify this screenshot.",
  "image_id": "example_001",
  "target": "Little Content"
}
~~~

This is a **storage example**, not the literal Qwen processor input. The image ID must resolve to real image data.

For a conversation-based model, we can store structured messages:

~~~json
{
  "messages": [
    {"role": "system", "content": "Return one allowed class."},
    {"role": "user", "content": "Classify the attached screenshot."},
    {"role": "assistant", "content": "Little Content"}
  ],
  "image_id": "example_001"
}
~~~

### Six dataset-building steps

1. **Define the label policy.** Decide what qualifies as Little Content, including confusing examples such as a single useful answer block on an otherwise sparse page.
2. **Collect valid image inputs.** Preserve screenshot quality, correct image–label pairing and appropriate source metadata.
3. **Annotate consistently.** Audit disagreement among reviewers and check the most ambiguous cases.
4. **Design the instruction.** It should tell the model what to do without revealing the correct label in the input.
5. **Separate training from evaluation.** Look for duplicates, host repetition and template overlap.
6. **Validate before tokenizing.** Verify permitted target strings and inspect records manually.

### Mathematical example: human-label agreement

Suppose two reviewers agree on 92 of 100 screenshots.

$$
P_{\mathrm{agreement}}=\frac{92}{100}=0.92
$$

This is **92% raw agreement**, not proof of correct labels, and not the same as a chance-corrected measure such as Cohen's kappa. In an imbalanced dataset, two reviewers may agree frequently simply because most pages are not Little Content. Review disagreements by class and error type.

### Dangerous counterexample: instruction leakage

~~~text
Bad positive prompt:
"Classify this low-quality empty page."
Target: Little Content

Bad negative prompt:
"Classify this useful page with meaningful content."
Target: Not Little Content
~~~

The instruction practically contains the answer. The model might learn the adjectives instead of inspecting the screenshot.

**Better:** Use an instruction whose wording is independent of the label.

### Qwen Project Application

Keep the raw human judgment separate from the instruction, image reference and any metadata available at inference. This lets us test whether Qwen is learning visual evidence rather than prompt shortcuts.

### Design Decision — Q3

Design a record with example ID, image reference, canonical label, instruction-template ID, reviewer/audit metadata and grouping metadata for eventual split design. Separate the supervised answer from the conditioning prompt.

### Project Build Record Update — Q3

| Element | Proposed state | Type |
|---|---|---|
| Dataset schema | Structured record plus image reference and target | Design Decision |
| Annotation policy | Needs explicit borderline-case rules | Open Question |
| Leak-safe data split | To be validated in evaluation design | Open Question |

---

### Story Bridge 4 — An Instruction and Answer Are Not Yet a Conversation

A chat assistant might receive a broad behavioral rule, a user request with an image, and then produce its response. If we do not identify who says what, we can accidentally teach the model to generate the user's request instead of its own answer.

## Question 4 — What Are System, User and Assistant Roles?

### First intuition

A **role** identifies the speaker or purpose of a message.

~~~text
System: You classify SERP pages.
User: Classify this screenshot. [IMAGE]
Assistant: Little Content
~~~

- **System** generally holds higher-level behavior instructions in the model's supported chat interface.
- **User** contains the request and its inputs, such as a screenshot.
- **Assistant** contains the model's answer; during supervised training it can hold the correct target response.

A role is not a separate neural network, loss function, or guaranteed security boundary.

### Why roles change learning

Imagine we mistakenly format the assistant target as:

~~~text
Assistant target:
"Classify this screenshot. Little Content"
~~~

We are now asking the model to learn to repeat the *user's instruction* before the label. That may be wrong for our production interface.

A properly constructed example supervises only the intended answer while preserving the instruction as input context.

### Technical distinction: messages versus tokens

A message role is usually represented in the serialized input using checkpoint-specific delimiters or special tokens. The trained Transformer processes the resulting representation, not an abstract programming-language message object.

### Qwen Project Application

We can use a constant classification policy in an appropriate high-level instruction and put the screenshot plus request in the user message. The assistant target remains exactly one canonical class response.

Different Qwen variants may handle system instructions differently; we must inspect the actual checkpoint's interface.

### Design Decision — Q4

Use a consistent system/instruction policy where supported, user image/request content, and assistant canonical target. Keep target loss limited to what the assistant should actually produce.

### Project Build Record Update — Q4

| Element | Proposed state | Type |
|---|---|---|
| Roles | Instruction/system, user/screenshot, assistant/class | Design Decision |
| Exact role handling in chosen Qwen checkpoint | To verify | Open Question |
| Target response | Canonical label only | Existing Design Decision |

---

### Story Bridge 5 — The Dataset Contains Messages, Yet a Causal Transformer Needs a Model-Compatible Input Sequence

Our records now have system, user and assistant roles. But the model ultimately operates over tokens and image features. Something must convert structured messages into the exact representation the selected checkpoint expects.

## Question 5 — What Is a Chat Template?

### Simple intuition

A **chat template** is a formatting recipe that turns structured conversation messages into a sequence with the proper message boundaries, speaker information and assistant-response position.

A **generic invented illustration** is:

~~~text
[START_SYSTEM] Return one allowed label. [END_MESSAGE]
[START_USER] Classify this screenshot. [IMAGE] [END_MESSAGE]
[START_ASSISTANT] Little Content [END_TURN]
~~~

The bracketed markers are **illustrative**, not claimed Qwen special tokens.

### Three different jobs

| Component | Responsibility |
|---|---|
| Chat template | Organizes message roles and boundaries |
| Tokenizer | Converts text and recognized special tokens into token IDs |
| Image / multimodal processor | Converts the image and other modality information into supported model inputs |

We should not confuse a chat template with the tokenizer itself.

### Mathematical representation

Let $M$ be structured messages and $T$ the model-specific chat-template procedure.

$$
s=T(M)
$$

For the textual part, the tokenizer maps the serialized text to token IDs:

$$
z=\operatorname{Tokenizer}(s)
$$

For a VLM, the full processor also prepares image tensors and possibly image-grid metadata or placeholders. **The resulting inputs are not necessarily just one flat array of text token IDs.**

### Why manual formatting can fail

Suppose training uses the checkpoint's special turn delimiters, but inference uses plain text such as "User:" and "Assistant:". Human readers may see equivalent conversations, but the model sees different token patterns. That can degrade performance.

Always use the supported processor/template and check exactly what is passed to the model.

### Qwen Project Application

We have not selected an exact Qwen VLM checkpoint. Therefore we cannot responsibly state its exact chat delimiters, token IDs, multimodal packing behavior or image-size limits. The design requires inspecting those after the checkpoint is chosen.

### Design Decision — Q5

Store structured messages and call the checkpoint-compatible processor/chat-template implementation. Inspect several encoded examples and decoded targets before training.

### Project Build Record Update — Q5

| Element | Proposed state | Type |
|---|---|---|
| Raw format | Structured messages with linked image input | Design Decision |
| Serialization | Checkpoint-compatible chat template | Design Decision |
| Exact tokens and image processor settings | Unresolved until checkpoint selection | Open Question |

---

### Story Bridge 6 — A Chat Template Needs Markers So the Model Knows Where a Message Ends

A template cannot merely concatenate message text without boundaries. The model must have some way to distinguish the user request from the assistant answer and to recognize when the assistant is supposed to begin or finish responding.

## Question 6 — What Are Role Delimiters, Special Tokens and Turn Boundaries?

### First intuition

A **delimiter** marks a boundary. In a chat model, delimiters can say "new message begins", "this message is from the user", or "assistant's turn is over".

In ordinary text, punctuation helps us distinguish sentences. Chat delimiters play a similar organizational role for model inputs, although they may be learned special token IDs rather than normal punctuation.

~~~text
[BEGIN_MESSAGE role=system] Policy text [END_MESSAGE]
[BEGIN_MESSAGE role=user] Screenshot request [END_MESSAGE]
[BEGIN_MESSAGE role=assistant] Little Content [END_TURN]
~~~

These markers are **schematic**, not official Qwen tokens.

### A subtle but important distinction

A marker ending a **message** need not be the same as a marker ending the **assistant turn**.

Consider an assistant message that is followed by another assistant message or an interaction involving tools. Some formats distinguish "the current message ended" from "the assistant is done for now".

A model may need to predict the correct end marker so that its output terminates correctly.

### Special tokens versus visible strings

Some strings that look like markup in documentation correspond to unique special token IDs. Others may simply be normal text split into several tokenizer pieces.

We cannot safely identify which are special until we inspect the exact checkpoint and tokenizer vocabulary.

### Worked example: why the assistant prefix matters

At inference, the model may be given the conversation ending at an **unfinished assistant prefix**:

~~~text
[BEGIN_MESSAGE role=system] ... [END_MESSAGE]
[BEGIN_MESSAGE role=user] ... [END_MESSAGE]
[BEGIN_MESSAGE role=assistant]
~~~

The next generated token should begin the assistant response.

In training, the target class tokens follow that prefix. The positioning of that prefix affects the next-token targets from Part 2.

### Mathematical link

Let a serialized sequence contain a prefix $p$ and assistant answer $y$. The first assistant token is predicted as:

$$
P_\theta(y_1\mid p)
$$

Later assistant tokens are predicted as:

$$
P_\theta(y_t\mid p,y_{<t}^{\mathrm{true}})
$$

If the serialized prefix is wrong, the conditioning context has changed even if the image and text are unchanged.

### Qwen Project Application

We must verify the exact assistant-start representation, role delimiters, end-of-turn marker, and whether the model's multimodal processor inserts image-related markers automatically.

### Design Decision — Q6

Never manually guess Qwen special tokens. Validate the selected checkpoint's template, generation prefix and stopping behavior with decoded examples.

### Project Build Record Update — Q6

| Element | Proposed state | Type |
|---|---|---|
| Role delimiters | Provided by selected checkpoint's template | Design Decision |
| Assistant-turn ending | Supervise correct marker where intended | Design Decision |
| Literal Qwen token names and IDs | Await checkpoint inspection | Open Question |

---

### Story Bridge 7 — Now We Know the Conversation Sequence, But Which Parts Are We Actually Training the Model to Generate?

A batch may contain system instructions, user requests, image placeholders, assistant responses and end-of-turn markers. Part 2 established assistant-only supervision in principle. Now we must do it **correctly after chat templating**.

## Question 7 — How Does Loss Masking Work in Chat Fine-Tuning?

### First intuition

**The model reads the conversation but usually receives direct SFT loss only on the assistant answer we want it to produce.**

Think of the system and user messages as the exam question, and the assistant message as the answer sheet. The question must be visible, but we do not want the model to spend loss predicting the question itself.

### Toy aligned sequence

Suppose the processed sequence, in a fictional simple tokenizer, is:

~~~text
Position:  0      1       2           3        4          5
Token:   SYS    USER    ASSISTANT    Little   Content   END_TURN
Label:   IGN    IGN       IGN        Little   Content   END_TURN
~~~

**Be careful:** Real causal LM training uses *next-token alignment*. The table expresses which target tokens we want to supervise, not a claim that each logit predicts its own current token. The framework may perform the causal shift internally.

The image and instruction remain conditioning inputs.

### Exact masked objective

Let $z_1,\ldots,z_L$ denote token IDs in the formatted sequence, and let $m_t=1$ only when $z_t$ is an intended assistant target.

$$
\mathcal L_{\mathrm{chat}}
=
-
\frac{
\sum_{t=1}^{L}m_t\log P_\theta(z_t\mid z_{<t},I)
}{
\sum_{t=1}^{L}m_t
}
$$

Here:

- $I$ is the screenshot or other multimodal input, if any.
- $m_t$ is a loss-selection indicator.
- $z_{<t}$ includes earlier system/user/assistant tokens allowed by causal attention.
- The denominator counts supervised target tokens.

### Critical distinction: three types of masking

| Mask | What it does |
|---|---|
| Causal attention mask | Prevents a position from seeing future positions |
| Padding/validity mask | Avoids treating batch padding as meaningful content |
| Loss mask / ignored labels | Determines which next-token predictions contribute to the objective |

**Ignoring a prompt position for loss does not remove it from context.**

### A specific engineering failure

Suppose the correct label is split into two or more token IDs. If the data collator accidentally masks the first assistant token, the model never receives direct supervision for the critical class-prefix decision.

An even worse failure is masking **all** assistant target positions. Then the loss may be undefined or effectively provide no useful gradient, depending on implementation.

### Debugging checklist

- Render and inspect the processed conversation.
- Decode non-ignored training targets.
- Confirm the first assistant class token and intended end marker are supervised.
- Confirm prompt and image positions are not mistakenly the supervised answer.
- Verify whether the model shifts logits/labels internally to avoid a double shift.
- Verify multi-image inputs and padding are aligned with the processor output.

### Qwen Project Application

For the theoretical binary classifier, we supervise only canonical label tokens and any intended assistant completion marker. The screenshot remains visible to the model but is not turned into a sequence of output text targets.

### Design Decision — Q7

Use assistant-target loss masks generated from model-compatible formatting and inspect them with real encoded examples before optimization.

### Project Build Record Update — Q7

| Element | Proposed state | Type |
|---|---|---|
| Supervised positions | Assistant class response and intended ending | Existing Design Decision |
| System/user tokens | Context, not direct supervised output | Design Decision |
| Shift and masking audit | Required on actual tokenized batch | Design Decision |

---

### Story Bridge 8 — A Chat Model Must Sometimes Respond After More Than One User Message

The examples so far contain one request and one answer. But assistants are often asked follow-ups, corrections and clarifications. They must use earlier conversation turns while predicting their next answer.

## Question 8 — How Does Multi-Turn Conversation Fine-Tuning Work?

### First intuition

In multi-turn training, we show the model **a sequence of messages**, not a single isolated request.

For example:

~~~text
User: Does this screenshot look empty?
Assistant: It appears visually sparse.
User: Apply the Little Content policy and return one label.
Assistant target: Little Content
~~~

When generating the final answer, the model should use the relevant earlier conversation as context.

### Two valid training choices

**Choice A: supervise only the final assistant response.**

Use all prior messages as context, but calculate loss only on the last target. This is appropriate when we specifically want to train the model for the last action.

**Choice B: supervise multiple assistant responses.**

If earlier assistant answers are also good training targets, compute loss on each selected assistant span. Each selected answer is predicted from the conversation prefix available before it.

### Formal objective

Suppose a conversation contains messages $M_1,\ldots,M_K$, with chosen assistant target spans indexed by $a$.

A schematic objective is:

$$
\mathcal L_{\mathrm{multi}}
=
-\sum_{a\in\mathcal A}
\sum_{t=1}^{T_a}
\log
P_\theta
\left(
y_{a,t}
\mid
\mathrm{history}_a,
y_{a,<t}^{\mathrm{true}}
\right)
$$

The set $\mathcal A$ contains the assistant messages selected for supervision. Training code usually implements this through one serialized sequence and a mask over the corresponding target-token positions.

### Why the context can matter

Imagine:

~~~text
User: The screenshot has a blank top half but a complete answer card below.
Assistant: Understood.
User: Give the class label.
~~~

If earlier messages are present, the last instruction is understandable. If truncation removes the relevant earlier user message, "Give the class label" may be ambiguous.

The model does **not** get persistent memory across examples merely because a dataset contains multi-turn records. Context must actually appear in the current training or inference sequence.

### Qwen Project Application

Our primary classifier likely needs **single-turn** image-and-instruction supervision, because it should classify a screenshot directly. Multi-turn data could be useful if the product truly supports clarifications, revisions or follow-up questions, but we should not introduce unnecessary conversation complexity into a fixed binary service.

### Design Decision — Q8

Keep the core Qwen classifier single-turn initially. Treat multi-turn supervision as a separate possible extension tied to demonstrated interaction requirements.

### Project Build Record Update — Q8

| Element | Proposed state | Type |
|---|---|---|
| Primary task | Single-turn screenshot classification | Design Decision |
| Multi-turn examples | Optional future extension | Open Question |
| Multi-assistant-turn masking | Use selected target spans if such data is trained | Design Decision |

---

### Story Bridge 9 — If Multi-Turn Is Possible, Does It Always Make the Model Better?

A model that performs a one-shot classification and a model that conducts a conversation have different requirements. Multi-turn data can teach helpful history use, but could add irrelevant behavior to a tightly constrained classifier.

## Question 9 — What Is the Difference Between Single-Turn and Multi-Turn Training?

### First intuition

- **Single-turn:** One user request is followed by one target assistant response.
- **Multi-turn:** Multiple messages appear before or among supervised assistant responses.

Neither is universally superior. It depends on what the deployed model is expected to do.

| Dimension | Single-turn | Multi-turn |
|---|---|---|
| Typical example | Screenshot → class label | Screenshot → clarification → final answer |
| Context | Usually short and focused | Longer, may contain prior decisions |
| Complexity | Easier formatting, masking and evaluation | More complicated supervision and truncation |
| Potential benefit | Strong focused behavior | Learn to use conversation history |
| Risk | Limited practice with follow-up requests | Learning unnecessary chat verbosity or inconsistent earlier answers |

### Understanding what the loss teaches

Suppose a service will always receive a screenshot and must output exactly one label. Multi-turn examples where the assistant explains its reasoning in long paragraphs may teach a response style that conflicts with the production requirement.

The issue is not that multi-turn training is mathematically wrong. The target behavior is different.

### Practical experiment

Construct two controlled training variants:

~~~text
Variant S:
Single screenshot + classification instruction
→ One canonical label

Variant M:
Mixed single-turn and multi-turn examples
→ Some intermediate explanatory responses
→ Final label
~~~

Evaluate both on the **actual production input format**, not only on the training-style conversations. If Variant M introduces explanatory text or raises latency without improving decision quality, it may be a poor fit.

### Qwen Project Application

Because Little Content detection is fundamentally a binary classification task, choose single-turn SFT as the default, unless product requirements demand conversational follow-up behavior.

### Design Decision — Q9

Do not add multi-turn training merely because Qwen is a chat-capable model. Match the conversation format to the expected inference task.

### Project Build Record Update — Q9

| Element | Proposed state | Type |
|---|---|---|
| Default interaction | One screenshot/request → one label | Design Decision |
| Chatting/clarification feature | Not established | Open Question |
| Evaluation format | Must match intended serving task | Design Decision |

---

### Story Bridge 10 — Even Correctly Formatted Conversations Can Break When We Batch Them Together

Training needs efficient batches. Some examples are long, some are short, and multimodal inputs can add many positions. When we join or shorten sequences, the boundaries and supervised target positions must remain correct.

## Question 10 — What Happens When Conversations Are Truncated or Packed?

### First intuition

**Truncation** means cutting a sequence to fit an input-length limit.

**Packing** means placing multiple examples into one training sequence or batch container to reduce wasted padding and improve utilization.

Both can be useful. Both can cause serious mistakes if implemented carelessly.

### Truncation problem

Suppose our serialized conversation is:

~~~text
SYSTEM RULE
USER SCREENSHOT + INSTRUCTION
ASSISTANT CLASS LABEL
ASSISTANT END MARKER
~~~

A careless length cap might remove the assistant label while retaining the prompt. The training sample now has no valid answer to supervise.

Or the crop could retain the label but remove the image/prompt that explains it. The model is then trained under the wrong context.

**Safe principle:** Validate that retained tokens still contain the full required conditioning information and selected supervised target. In some applications, discard overly long samples or use a task-aware truncation policy rather than cut blindly.

### Packing problem

Imagine two independent examples:

~~~text
Example A: screenshot A → Little Content
Example B: screenshot B → Not Little Content
~~~

A naive concatenation produces:

~~~text
[Example A tokens][Example B tokens]
~~~

Under plain causal attention, tokens in Example B can potentially attend to Example A unless the training approach deliberately prevents that or intentionally treats the concatenation as one context.

That creates **cross-example context leakage**. Correctly implemented packing may use block-diagonal attention, separate position handling, or another method appropriate to the model/trainer. A simple separator alone does not necessarily block attention.

### Mathematical view

Suppose two examples have lengths $L_1$ and $L_2$, and we pack into a length-$L_1+L_2$ sequence.

Without an example-isolation mechanism, causal masking permits attention from Example B's later positions to earlier positions in Example A.

A block-isolated attention validity structure instead has:

$$
M=
\begin{bmatrix}
M_A & 0 \\
0 & M_B
\end{bmatrix}
$$

Here each $M_A$ and $M_B$ is its own valid causal attention pattern, and the off-diagonal blocks indicate no cross-example attention. The actual mask encoding may use booleans or additive negative-infinity bias, not literal zero probability values at all stages.

### Multimodal complication

Packing multiple image-containing conversations may require correct image-to-token alignment, image-grid metadata, position handling and model-specific attention behavior. Not every text-packing optimization is safe for a VLM.

### Qwen Project Application

Begin with reliable padding/batching without experimental packing. Once correctness is established, benchmark whether the selected Qwen trainer supports multimodal packing and isolation correctly. Screenshot preprocessing should also avoid discarding visually meaningful regions just to force a convenient length.

### Design Decision — Q10

Prioritize **correct multimodal examples over packing efficiency**. Preserve required screenshot context and supervised label; inspect packed samples and attention behavior before enabling sequence packing.

### Project Build Record Update — Q10

| Element | Proposed state | Type |
|---|---|---|
| Truncation | Must preserve relevant conditioning and assistant targets | Design Decision |
| Packing | Disabled until checkpoint/trainer support is verified | Design Decision |
| Multimodal attention and position behavior | To inspect in selected implementation | Open Question |

---

