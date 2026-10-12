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
P_\theta(y \mid u, x) = \prod_{t=1}^{T} P_\theta(y_t \mid u, x, y_{\lt t}^{\text{true}})
$$


We can minimize:

$$
\mathcal{L}_{\mathrm{IT}} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid u, x, y_{\lt t}^{\text{true}})
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
z = \text{Tokenizer}(s)
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
z = \text{Tokenizer}(s)
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
\mathcal{L}_{\mathrm{chat}} = - \frac{\sum_{t=1}^{L} m_t \log P_\theta(z_t \mid z_{\lt t}, I)}{\sum_{t=1}^{L} m_t}
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

### Story Bridge 11 — Chat Formatting Is Clear for Text. What Changes When the User Attaches an Image?

Our Qwen project is multimodal. The message structure may still look like a conversation, but the processor must also turn screenshot pixels into visual model inputs. This raises a new question: how do image inputs and assistant text targets coexist in one training example?

## Question 11 — How Does Multimodal Instruction Tuning Work?

### Simple intuition

Multimodal instruction tuning teaches a model to follow instructions **using more than one kind of input**. For Qwen, the relevant input is a screenshot together with a written classification request.

The desired assistant response can still be text. We do not need to make the model reproduce the screenshot; we need it to use the screenshot to decide which label to produce.

### The complete input story

~~~text
Raw screenshot (pixels) ──→ image processor ──→ visual inputs
                                               |
Instruction + message roles ──→ chat template ─┤
                                               ↓
                                     Qwen multimodal forward pass
                                               ↓
                                Next-token predictions for assistant
                                               ↓
                                    Assistant-only SFT loss
~~~

The exact implementation depends on the checkpoint. Some VLMs represent an image with a variable number of visual tokens or patches; others include special placeholders, grid metadata and resolution-specific transformations. These implementation details should not be guessed.

### Conceptual mathematical model

Let $I$ denote the screenshot and $u$ the instruction. Let the image encoder produce visual features:

$$
v=f_{\theta_v}(I)
$$

The language-side computation, combined with the model-specific multimodal connection, produces probabilities for the response sequence:

$$
P_\theta(y \mid I, u) = \prod_{t=1}^{T} P_\theta(y_t \mid I, u, y_{\lt t}^{\text{true}})
$$


The supervised token objective remains:

$$
\mathcal{L} = -\sum_{t \in \mathcal{T}_{\mathrm{assistant}}} \log P_\theta(y_t \mid I, u, y_{\lt t}^{\text{true}})
$$


Only selected assistant token targets contribute directly to this loss. The image is **conditioning information**, so it can still affect the gradients of trainable modules.

### What the dataset should store versus what the model should receive

| Stage | Representation |
|---|---|
| Dataset storage | Screenshot path/bytes, instruction, label and metadata |
| Model-compatible messages | User message containing text and an image content part |
| Processor output | Token IDs, image tensors/features and checkpoint-specific metadata |
| Training labels | Supervised assistant target token IDs and ignored positions |
| Model output | Logits over vocabulary at prediction positions |

A JSON-style storage record alone does not guarantee that the image was actually included in the model's forward pass.

### Common failure: image forgotten

Imagine an incorrect data collator that builds only text tokens:

~~~text
User: Classify the attached screenshot.
Assistant target: Little Content
~~~

but never loads or passes the image tensor. The training loop could still run, but the model would be learning from **instruction and label frequencies**, not visual evidence.

To detect this, inspect processor outputs and test **image sensitivity**: does changing the screenshot while keeping the instruction fixed change model scores in a meaningful, task-relevant way? The test must be designed carefully; score changes alone do not prove correct grounding.

### Another failure: wrong resizing or cropping

A low-content decision can depend on the relative layout of page elements. A crop that removes the main content panel can turn a normal page into an apparently empty one. Resizing may also make small text unreadable.

Therefore image preprocessing is part of model quality, not just data plumbing.

### Qwen Project Application

Use the checkpoint's supported multimodal chat-message format and processor. Keep the output restricted to canonical class labels. Confirm that visual inputs reach Qwen and remain aligned with the correct sample and assistant target.

### Design Decision — Q11

Select the exact Qwen VLM checkpoint first, then implement its documented processor and check image dimensions, aspect ratio, special markers, visual metadata and token-label alignment using real samples.

### Project Build Record Update — Q11

| Element | Proposed state | Type |
|---|---|---|
| Input modalities | Screenshot + instruction | Existing Design Decision |
| Image processing | Must use checkpoint-compatible processor | Design Decision |
| Image resolution / patch behavior | Not specified until checkpoint inspection | Open Question |
| Output modality | Canonical assistant text class label | Existing Design Decision |

---

### Story Bridge 12 — Correct Formatting Does Not Guarantee the Model Learns the Right Decision Rule

Even a perfectly encoded screenshot and correct loss function can train a poor classifier if examples are mislabeled, repetitive, imbalanced, or unrepresentative of production.

We now need to examine the *information content* of the dataset.

## Question 12 — How Do Data Quality, Diversity and Dataset Mixtures Affect Instruction Tuning?

### First intuition

A model learns patterns from the examples we show it. If the examples teach incorrect patterns, the model can become excellent at repeating those mistakes.

Three important dataset properties are:

- **Quality:** Are labels, images and response formats correct?
- **Diversity:** Do examples represent the actual variations the system will encounter?
- **Mixture:** What proportion of training examples comes from different tasks, classes or sources?

### Example: a model memorizes page templates

Suppose our training data contains thousands of near-identical screenshots from a handful of hosts. Randomly splitting individual screenshots might place the same layout in training and evaluation.

Qwen could seem accurate simply because it has already seen similar templates. That is not evidence that it generalizes to new hosts.

In an appropriate evaluation design, group related screenshots, hosts or templates to prevent misleading overlap. The exact grouping should reflect the deployment question.

### Example: inconsistent supervision

Consider the same screenshot receiving different labels:

~~~text
Screenshot X → Little Content
Screenshot X → Not Little Content
~~~

Some annotation disagreements may be legitimate consequences of an unclear policy. We should resolve or characterize them; training harder does not fix contradictory supervision.

### Numerical example: why class balance matters

Suppose 90% of a dataset is Not Little Content and 10% is Little Content.

A classifier that always predicts Not Little Content has:

$$
\mathrm{Accuracy}=0.90
$$

Yet its recall on the Little Content class is zero:

$$
\mathrm{Recall}_{+}=\frac{TP}{TP+FN}=0
$$

This is why accuracy alone cannot determine success. Depending on the real business goal, precision, recall, false-positive rate and calibration are important.

### The mixture question

If we combine:

~~~text
80% general chat instructions
15% broad image question answering
 5% SERP screenshot classification
~~~

we may teach broader assistant behavior, but the precise Little Content behavior could be underrepresented. Conversely, training only on the narrow task may make the model highly specialized but gives less practice with varied natural-language requests.

**This ratio is illustrative, not an advised recipe.**

### Data weighting: how sampling changes learning

Let group $k$ have dataset distribution $\mathcal D_k$ and sampling weight $w_k$, with $\sum_k w_k=1$.

A mixture objective is:

$$
\mathcal{L}_{\mathrm{mix}} = \sum_{k=1}^{K} w_k \mathbb{E}_{(x,y) \sim \mathcal{D}_k} [\mathcal{L}_\theta(x,y)]
$$


Changing $w_k$ changes which examples contribute more often or more strongly to training. A larger weight is not always better; noisy or duplicated sources can become overrepresented.

### Qwen Project Application

Start with accurately labelled SERP screenshots representing sparse useful pages, login-dominant pages, navigation-heavy sites, novel templates and different content densities. Document class proportions and human-review consistency. A mixture of other instruction tasks should be added only if a clear evaluation need justifies it.

### Design Decision — Q12

Prioritize data quality, hard negatives and diversity of hosts/layouts over adding large volumes of redundant screenshots. Record dataset-source mixtures rather than relying on an undocumented blend.

### Project Build Record Update — Q12

| Element | Proposed state | Type |
|---|---|---|
| Hard negatives | Sparse-but-useful pages must be represented | Design Decision |
| Class/source mixture | Record and evaluate empirically | Open Question |
| Leakage controls | Host/template overlaps require audit | Design Decision |

---

### Story Bridge 13 — If We Train on One Set of Instructions, Will Qwen Handle New Wording?

We now have clean data with valid chat formatting. But training on the sentence "Classify this screenshot" does not prove the model will behave correctly when the request is phrased differently.

The next challenge is **generalization beyond memorized examples and prompts**.

## Question 13 — How Does Instruction Tuning Generalize to Unseen Instructions?

### Simple intuition

We want the model to learn the **task relationship** rather than memorize one exact sentence.

For instance, these requests should usually call for the same behavior:

~~~text
Classify this SERP screenshot.
Does this screenshot violate the Little Content policy?
Use the allowed label to assess this search-result page.
~~~

But they are not necessarily equivalent unless the policy and required response are held constant.

### How generalization may happen

A pretrained model already has broad linguistic representations. Supervised examples can connect varied instruction phrasings to desired behavior. That transfer can work because related instructions may activate useful pretrained features.

However, the training objective does not prove that the model has learned a perfect abstract notion of "following any instruction". We need tests on **unseen paraphrases and genuinely different inputs**.

### A controlled evaluation design

Create two independent variations:

1. **Screenshot variation:** new hosts/layouts with the same canonical instruction.
2. **Instruction variation:** previously unseen paraphrases on held-out screenshots.

Then compare four conditions:

| Condition | Instruction | Screenshot |
|---|---|---|
| A | Seen wording | Seen-like layout |
| B | New wording | Seen-like layout |
| C | Seen wording | New host/layout |
| D | New wording | New host/layout |

A model might pass B but fail C, meaning it understands alternate wording but not unfamiliar visual layouts. Or it might pass C but fail B, revealing instruction sensitivity.

### Why pure paraphrase augmentation can mislead

If every paraphrase subtly suggests the answer, the model can learn lexical shortcuts. If we generate dozens of nearly identical prompts, the extra volume may contribute little genuinely new information.

Instruction diversity should preserve the **same task definition and target meaning**.

### Qwen Project Application

Because production may use a fixed instruction, broad conversational paraphrase robustness may be a secondary objective. We should still test a few safe reformulations, but not sacrifice binary classification quality to optimize for arbitrary chat flexibility without a product need.

### Design Decision — Q13

Evaluate task generalization separately across **unseen screenshots** and **unseen instructions**. Do not conflate robust language following with robust visual classification.

### Project Build Record Update — Q13

| Element | Proposed state | Type |
|---|---|---|
| Primary generalization | Unseen SERP hosts and layouts | Design Decision |
| Secondary generalization | Controlled instruction paraphrases | Design Decision |
| Robustness on unseen prompts | Must be measured | Open Question |

---

### Story Bridge 14 — Suppose Training Runs Successfully But the Model Behaves Strangely. Where Could the Error Be?

By now, we have multiple distinct stages: dataset construction, message roles, chat template, image processing, token IDs, masked labels, training and evaluation.

Failures may arise at *any* of these stages. We should diagnose them systematically rather than immediately changing LoRA rank or training for more epochs.

## Question 14 — What Are Common Failures in Chat Fine-Tuning, and How Do We Debug Them?

### A practical failure map

| Observed failure | Plausible root causes | First checks |
|---|---|---|
| Assistant repeats user instruction | Wrong roles or loss masking | Inspect decoded supervised labels |
| Assistant outputs long paragraphs instead of class | Inconsistent targets; unconstrained decoding | Check label strings and candidate scoring |
| Model never ends response | Wrong or unsupervised turn ending | Inspect end-marker targets and stopping config |
| Loss looks excellent but classification is poor | Label leakage, wrong image pairing, overfitting | Verify raw images and independent test data |
| Different output at train vs inference | Different chat templates or image processing | Compare processor paths |
| Model predicts one class constantly | Class imbalance, shortcut learning, scoring | Confusion matrix and per-class metrics |
| Multi-turn history ignored | Relevant context truncated or improperly encoded | Decode whole retained sequence |
| Packed samples interfere | Cross-example attention leakage | Inspect packing and isolation behavior |
| Screenshot seems irrelevant | Missing image tensor or wrong preprocessing | Confirm visual inputs and run image-swap diagnostics |

### A debugging sequence that avoids random guessing

**Step 1 — Verify one raw example.**

Open the screenshot, read its true label, inspect the instruction and verify that they all correspond to the same sample.

**Step 2 — Verify the serialized representation.**

Render the checkpoint-compatible conversation. Confirm the expected roles, assistant-start marker and image content part are present.

**Step 3 — Verify labels and masking.**

Decode the positions with non-ignored labels and check that they correspond to the canonical assistant response and intended turn ending. Confirm the training implementation does not shift targets twice.

**Step 4 — Verify that image information enters the model.**

Inspect processor outputs, any image batch tensors and checkpoint-required visual metadata. If possible, compare model responses when the image changes but the text remains constant. Interpret results using a controlled sample set.

**Step 5 — Verify a small optimization run.**

On a tiny, internally consistent set, check that trainable LoRA parameters receive nonzero gradients and the intended training loss can decline. A tiny-set overfit test is a **pipeline debugging technique**, not evidence of generalization.

**Step 6 — Verify actual task performance.**

Test on genuinely held-out hosts/layouts, calculate confusion matrices and inspect the false positives or false negatives that matter most. Revisit capacity and training hyperparameters only after data and formatting errors are ruled out.

### A deceptively easy bug: double application of the chat template

If stored training text already includes message markers and the data loader adds the template again, the model receives duplicated or conflicting delimiters.

Prevent this by defining whether the dataset stores **structured messages** or **already serialized content**, and applying the appropriate processor path **exactly once**.

### Qwen Project Application

If our proposed Qwen classifier predicts Little Content even for a visually rich screenshot, first verify it actually receives the image and that the allowed label sequences are scored consistently. Do not assume the problem is a weak visual encoder before ruling out data-path errors.

### Design Decision — Q14

Include dataset audits, decoded-token inspections, masked-label checks, tiny-batch training tests and held-out evaluation in the implementation checklist before tuning model capacity.

### Project Build Record Update — Q14

| Element | Proposed state | Type |
|---|---|---|
| Formatting smoke tests | Required before full training | Design Decision |
| Tiny-batch overfit test | Debugging only, not final evidence | Design Decision |
| Task-level evaluation and error slices | Required for claims about improvement | Design Decision |

---

### Story Bridge 15 — We Understand Each Piece. Can We Assemble One Correct End-to-End Qwen Example?

We have studied the purpose of instruction tuning, the message roles, the template, token targets, multimodal processing, and data-quality risks.

The final question asks us to turn these separate ideas into one **coherent training-data design** for the theoretical project. This is where we record the decisions and the unresolved details together.

## Question 15 — How Would We Build an Instruction-Tuning Dataset for Qwen Little Content Detection?

### The goal

Build labelled screenshot examples that can be processed by a selected Qwen VLM and used for supervised, assistant-target training. The model should output one of two canonical class labels when given a screenshot and classification instruction.

**We are designing a pipeline; no code is claimed to have been deployed or validated here.**

### Step 1 — Define the user-facing classification behavior

~~~text
Input:
    Screenshot of a search-result page
    Classification instruction

Output:
    Little Content
        OR
    Not Little Content
~~~

Do not allow the training label policy to shift silently between annotators or dataset batches.

### Step 2 — Store raw examples separately from the model-formatting layer

A proposed conceptual record is:

~~~json
{
  "example_id": "SERP_00042",
  "image_ref": "screenshots/SERP_00042.png",
  "label": "Not Little Content",
  "instruction_template_id": "LC_V1",
  "host_group": "example.test",
  "annotation_status": "reviewed"
}
~~~

The file reference, host and ID are **illustrative**, not actual project data. The host should be treated as metadata for audits and evaluation grouping unless it is intentionally approved as an inference feature.

### Step 3 — Construct messages with the correct roles

Conceptual data representation:

~~~text
System:
    You are a SERP Little Content classifier.
    Return exactly one of the two approved class labels.

User:
    Classify the attached screenshot according to the policy.
    [actual image content]

Assistant target:
    Not Little Content
~~~

The screenshot must be passed in the image content structure expected by the selected Qwen processor. A plain text string saying "[actual image content]" is not a substitute for an image tensor.

### Step 4 — Apply the correct checkpoint processor

~~~text
Structured conversation + screenshot bytes
              ↓
Official model-compatible chat template
              +
        Image processor
              ↓
  Token IDs + visual model inputs
              ↓
        Causal Qwen VLM
~~~

Exact Qwen model name, special token IDs and pixel/grid requirements cannot be settled without first choosing and inspecting the checkpoint.

### Step 5 — Construct the supervised labels

The system instruction and user request are conditioning context. We generally calculate direct loss on the assistant class response and its intended ending.

Consider a **fictional** tokenizer in which "Not Little Content" splits into three tokens:

~~~text
Prefix (context):  system + user + assistant-start
Target sequence:   Not  Little  Content  END_TURN
Loss applies to:   Not  Little  Content  END_TURN
~~~

We must not assume that these are the actual Qwen tokens or that the response has exactly four tokens.

With mask $m_t$:

$$
\mathcal{L} = - \frac{\sum_t m_t \log P_\theta(z_t \mid z_{\lt t}, I)}{\sum_t m_t}
$$


### Step 6 — Apply the parameter-update design from Part 2B

Once the training batch is correct, we can use the already proposed LoRA baseline:

~~~text
Frozen pretrained model
    +
Trainable language-side LoRA on selected Q/V modules
    +
Potential trainable connector if separable and justified
    ↓
Assistant-token SFT loss
    ↓
Gradients update only allowed trainable parameters
~~~

This section does **not** choose the final rank, alpha, dropout, optimizer or learning rate. Those are separate engineering decisions planned for later chapters.

### Step 7 — Define trustworthy evaluation

At minimum, compare:

- A prompt-only Qwen baseline versus a fine-tuned candidate.
- Performance on the two canonical classes using task metrics, not merely token loss.
- False-positive errors on sparse-but-useful pages.
- Unseen-host and unusual-layout slices.
- Effect of changes to instruction wording.
- Memory and latency implications of the chosen adaptation.

Decision thresholds and calibration are part of later evaluation/serving design; they must be fitted on validation data rather than inferred from training loss.

### Step 8 — Check input/output consistency before training

**Minimal smoke-test record:**

| Check | Why it matters |
|---|---|
| Screenshot decodes and matches human label | Prevents visual label corruption |
| Correct Qwen processor produces the image inputs | Ensures a real multimodal forward pass |
| Message roles and template are correct | Prevents the model learning the wrong sequence |
| Correct assistant tokens are unmasked | Ensures direct target supervision |
| End-of-turn target is intentional | Prevents malformed stopping behavior |
| Frozen base and trainable LoRA sets are as intended | Prevents accidental full fine-tuning |
| Tiny-batch loss and gradients behave sensibly | Confirms training wiring |
| Held-out images have no inappropriate overlap | Protects evaluation integrity |

### Qwen Project Application — Final integration

The complete proposed path is:

~~~mermaid
flowchart TD
    A["Human-labelled SERP screenshot"] --> B["Raw sample with label + audit metadata"]
    B --> C["Structured system/user/assistant messages"]
    C --> D["Checkpoint-compatible chat template + image processor"]
    D --> E["Token IDs and visual inputs"]
    E --> F["Teacher-forced Qwen forward pass"]
    F --> G["Assistant-target masked token loss"]
    G --> H["Backpropagation through trainable LoRA/connector"]
    H --> I["Candidate checkpoint"]
    I --> J["Held-out host/layout evaluation"]
    J --> K["Decision on quality, cost and next experiments"]
~~~

The Mermaid diagram is **a proposed learning and engineering architecture**, not a documented production pipeline.

### Design Decision — Q15

Adopt structured multimodal examples, canonical label targets, checkpoint-compatible templating, assistant-only SFT, and a modest LoRA starting configuration for experimentation. Do not commit to exact model-dependent details until we have evidence from the chosen Qwen checkpoint.

### Project Build Record Update — Q15

| Element | Proposed state | Type |
|---|---|---|
| Training-data interface | Screenshot + instruction → one canonical class label | Design Decision |
| Raw record versus processed batch | Separate clearly | Design Decision |
| Input formatting | Exact Qwen processor/chat template to be used | Design Decision |
| Exact checkpoint and hyperparameters | Not yet selected | Open Question |
| End-to-end training run and observed performance | Not established | Open Question |

---

# Qwen Project Build Record — After Part 3

This record is cumulative. It includes earlier Part 2/2B assumptions and the new instruction/chat-tuning decisions. **Resume Fact** means the project description asserts it; it does not independently verify experimental configuration or outcomes.

| Design element | Current state | Type |
|---|---|---|
| Business problem | Identify Little Content SERP screenshots | Resume Fact |
| Model family | Qwen Vision-Language Model | Resume Fact |
| Training input type | Labelled Bing SERP screenshots | Resume Fact |
| Described processing scale | Approximately 50K URLs/day | Resume Fact |
| Chosen learning formulation | Generative supervised classification | Design Decision |
| Desired response | Canonical Little Content / Not Little Content label | Design Decision |
| Loss | Causal assistant-target cross-entropy | Design Decision |
| Teacher forcing | Correct previous assistant target tokens used in training | Design Decision |
| Prompt and screenshot | Conditioning information; not direct target-text loss | Design Decision |
| Adaptation starting point | PEFT with LoRA | Design Decision |
| Frozen pretrained language weights | Proposed initial configuration | Design Decision |
| Frozen visual encoder | Proposed initial configuration | Design Decision |
| Initial language-side LoRA targets | Query and Value projection candidates | Design Decision |
| Optional multimodal connector | Trainable if architecture supports and validation justifies | Design Decision |
| QLoRA | Alternative if base model weight memory is limiting | Design Decision |
| Instruction formulation | Explicit task policy and canonical output constraints | Design Decision |
| Raw data schema | Image reference, label, instruction ID, audit/group metadata | Design Decision |
| Message roles | System/instruction, user/screenshot, assistant/label | Design Decision |
| Chat template | Checkpoint-compatible implementation; not hand-guessed | Design Decision |
| Special tokens / generation prefix | Inspect selected checkpoint | Open Question |
| Image preprocessing | Use checkpoint-compatible multimodal processor | Design Decision |
| Exact Qwen checkpoint | Not selected | Open Question |
| Exact image resolution / patch handling | Not selected | Open Question |
| Multi-turn data | Not needed in initial binary-classifier dataset | Design Decision |
| Truncation | Preserve image/instruction and selected assistant targets | Design Decision |
| Packing | Do not enable without tested multimodal isolation/support | Design Decision |
| Dataset quality checks | Pairing, annotation agreement, duplicates and leakage audits | Design Decision |
| Generalization checks | New hosts/layouts plus controlled prompt paraphrases | Design Decision |
| Tiny-batch / loss-mask smoke tests | Required before scaled training | Design Decision |
| Exact LoRA rank, alpha and dropout | Not chosen | Open Question |
| Optimizer, LR, scheduler, batch size, epochs | Deferred training-engineering decisions | Open Question |
| Data splitting and sampling weights | Exact choices not established | Open Question |
| Inference threshold and calibration | Deferred evaluation decisions | Open Question |
| Production serving architecture | Not established in this theoretical design | Open Question |
| Empirical improvements from proposed configuration | No experiment results established | Open Question |

---

# Part 3 — Key Equations and What They Mean

## Instruction-conditioned likelihood

$$
P_\theta(y \mid u, I) = \prod_{t=1}^{T} P_\theta(y_t \mid u, I, y_{\lt t}^{\text{true}})
$$


$u$ is the request; $I$ is the screenshot; $y$ is the intended answer sequence.

## Assistant-only masked loss

$$
\mathcal{L} = - \frac{\sum_t m_t \log P_\theta(z_t \mid z_{\lt t}, I)}{\sum_t m_t}
$$


$m_t=1$ for supervised assistant targets and $0$ for ignored positions. The screenshot can influence all supervised probabilities even though it does not carry a direct text-token target.

## Dataset-mixture objective

$$
\mathcal{L}_{\mathrm{mix}} = \sum_{k=1}^{K} w_k \mathbb{E}_{(x,y) \sim \mathcal{D}_k} [\mathcal{L}_\theta(x,y)], \qquad \sum_{k=1}^{K} w_k = 1
$$


$w_k$ changes the contribution of task/data source $k$; it does not guarantee better downstream performance.

## Precision and recall

$$
\mathrm{Precision}=\frac{TP}{TP+FP},
\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

These task metrics are different from next-token training loss.

---

# Part 3 — Final Mental Model

~~~text
TASK POLICY
What does the class label mean?
        ↓
RAW EXAMPLES
Image, label, audit metadata
        ↓
INSTRUCTION + ROLES
System / user / assistant
        ↓
CHECKPOINT CHAT TEMPLATE + IMAGE PROCESSOR
Messages → token IDs + visual inputs
        ↓
LOSS-LABEL CONSTRUCTION
Assistant targets unmasked
        ↓
TEACHER-FORCED SFT
Cross-entropy for the allowed label response
        ↓
LoRA / TRAINABLE MODULES
Only intended parameters update
        ↓
EVALUATION
New hosts, new layouts, robust class scoring
~~~

**One-line summary:**

> **Part 2 established how supervised tokens create a training signal; Part 3 establishes how correct instructions, messages, images and masks create the right supervised tokens in the first place.**

---

# Bridge to Part 4 — Preference Optimization

We have now taught Qwen using examples of **correct target responses**. But what if the desired behavior cannot easily be represented by one objectively correct answer? In some tasks, humans can say which of two responses they prefer even if neither is the unique correct response.

That motivates preference datasets, reward models, RLHF and DPO, which belong to the next unit—not this chapter.

**This chapter ends here at Question 15.** Further training hyperparameters and production evaluation remain in their designated later modules.
