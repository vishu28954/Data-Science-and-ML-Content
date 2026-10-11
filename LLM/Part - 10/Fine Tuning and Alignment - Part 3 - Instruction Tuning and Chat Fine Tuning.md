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

