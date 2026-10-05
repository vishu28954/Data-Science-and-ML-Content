# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 7

## 8.12 Stop Tokens
## 8.13 Maximum Output Length

**Detailed Study Mode.** The interview-style version remains a separate file.

Parts 1–6 explained prefill, decode, KV cache, greedy selection, temperature, top-k, top-p, beam search, and repetition controls. We now need to answer **when generation stops**.

Stop tokens or configured stop sequences can terminate an answer before its budget is exhausted. Maximum output length imposes a ceiling even if the model never produces a natural ending.

---

# 8.12 Stop Tokens

### Story Bridge 1 — A Decoder Can Keep Going, So Something Must End the Loop

Parts 1–6 taught us how the model repeatedly predicts and selects the next token. But that process, by itself, does not answer one basic question:

> **When should generation end?**

If every selected token simply becomes the context for another prediction, the model can keep extending the sequence. Before we discuss EOS, custom stop markers, or length limits, we first need to understand why an autoregressive decoder requires an explicit stopping mechanism.

## Question 1 — If autoregressive generation can keep predicting tokens, what actually tells it to stop?

At each position, a decoder-only language model predicts a distribution for the next token:

$$
P(y_{t+1}\mid x,y_{1:t})
$$

Here $x$ is the prompt, $y_{1:t}$ is the generated continuation so far, and $y_{t+1}$ is the next candidate token.

Next-token prediction alone does not enforce an ending. A serving system therefore needs stopping criteria, including recognized ending tokens, custom stop conditions, and an output-token budget.

A generation request may finish naturally after a few tokens or be interrupted mechanically at its configured limit.

### Story Bridge 2 — From Needing a Stop to Learning an Ending Signal

So we know the decode loop needs some way to terminate. The next question is whether the **model itself can produce a learned ending signal**, rather than relying only on an external counter.

## Question 2 — Can the model learn its own ending signal — what is EOS?

EOS stands for **end of sequence**. It is typically a dedicated tokenizer token ID that may be predicted at positions where the model has learned a sequence should end.

Different model families use different token spellings and IDs. Examples of visible token spellings include:

~~~text
</s>
<|endoftext|>
<|eot_id|>
~~~

These markers are **not universally interchangeable**. Some chat formats distinguish end of message, end of assistant turn, and control handoff. Inspect the tokenizer and chat template to determine which marker should terminate a particular response.

### Story Bridge 3 — If EOS Is Learned, It Must Compete During Token Selection

EOS gives us a learned ending marker. But the model must still decide *when* to use it. That means EOS has to participate in the same prediction process as ordinary next-token candidates.

## Question 3 — How can EOS compete with ordinary vocabulary tokens?

Yes, when EOS is an eligible output token. The model can assign EOS a score just as it assigns scores to words, punctuation, and other tokens.

Imagine this next-token distribution:

| Candidate | Probability |
|---|---:|
| EOS | 0.65 |
| Period | 0.20 |
| because | 0.10 |
| Other tokens combined | 0.05 |

With greedy decoding, EOS is selected because it has the highest probability. With sampling, EOS may or may not be selected according to the final distribution after configured filters and penalties.

A high EOS probability does not itself stop generation: the relevant EOS token must actually be selected and recognized by the runtime.


### Story Bridge 4 — From EOS Probability to Runtime Behavior

Now EOS exists inside the probability distribution. The next step is to connect that model-side probability to the running inference loop: **what changes once EOS actually wins token selection?**

## Question 4 — Once EOS is selected, what exactly happens in the decode loop?

A typical generation loop does the following:

1. Prefill the prompt and select the first generated token from the resulting logits.
2. For subsequent positions, process the most recently selected token using a decode forward pass and relevant KV cache.
3. Use the decoding rule to select the next token.
4. Check the selected token against configured ending-token IDs and other stopping conditions.
5. If an enabled stopping condition has occurred, finish; otherwise continue within the token budget.

Serving implementations may store EOS internally while hiding its special-token spelling from returned text.

**Important distinction:** The first generated token can be selected directly from prefill logits. A separate one-token decode forward pass is not needed before that first selection.

### Story Bridge 5 — Looking Finished Is Not the Same as Being Stopped

A recognized EOS can stop the loop immediately. But humans often judge an answer as “finished” from its words or punctuation. We therefore need to separate **semantic-looking completion** from an actual runtime stopping signal.

## Question 5 — If the text sounds finished, is that enough to stop generation?

No. A phrase that *sounds* final is still ordinary generated text unless the serving application explicitly treats it as a stopping pattern.

For example:

~~~text
Case A:
Answer: The result is 42.
Next selected token: EOS
→ Runtime recognizes the ending token

Case B:
Answer: The result is 42. I am done.
Next selected token: an ordinary text token
→ Generation need not stop
~~~

Punctuation and newlines are likewise not universal stopping signals.

### Story Bridge 6 — Applications Sometimes Need Their Own Ending Rule

If ordinary text does not automatically stop generation, an application may still want its own delimiter or protocol marker. That creates a second type of stopping mechanism: **custom stop sequences**.

## Question 6 — What if the application wants to stop on something other than EOS?

An EOS token is usually a tokenizer-defined special token ID that may have been learned as an ending marker.

A **custom stop sequence** is a caller-configured pattern the serving system detects in generated output. For example, an application may configure:

~~~text
[END]
~~~

A custom marker can span several tokenizer tokens. Its detection may be based on decoded characters, token IDs, or a streaming matcher; behavior depends on the API.

| Feature | EOS | Custom stop sequence |
|---|---|---|
| Defined by | Model and tokenizer | Application or request |
| Typical representation | Dedicated special token ID | Text or token pattern |
| Can span multiple tokens? | Generally one ID | Yes |
| Typical detection | Selected special token | Runtime pattern matching |
| Returned to user? | Often hidden | Depends on API |

### Story Bridge 7 — A Text Marker May Span Several Tokens

A custom marker looks simple as text, but tokenization can split it into several pieces. The runtime therefore has to decide whether it has seen a complete delimiter or only the beginning of one.

## Question 7 — What if that custom stop sequence spans multiple tokenizer tokens?

Suppose the application uses the marker:

~~~text
[END]
~~~

Imagine the tokenizer splits it into three tokens:

~~~text
[   END   ]
~~~

The model produces text ending in the complete sequence. The runtime may not know that the marker has been generated until the final closing bracket arrives.

A streaming implementation needs to consider **partial matches**. It may temporarily buffer characters that could be the beginning of a stop marker instead of immediately exposing them to the client.

This is why streamed output and the final returned text may differ near a custom stop boundary. Exact inclusion/exclusion behavior depends on the serving API.


### Story Bridge 8 — Real Requests Can Have Several Exit Doors

We now have more than one possible exit from the decode loop: EOS, custom delimiters, and other runtime conditions. A real request therefore needs a policy for handling several possible stopping criteria together.

## Question 8 — What if several stopping conditions are active at the same time?

A request may configure EOS IDs, custom stop strings, a maximum new-token limit, and possibly cancellation or other application-defined controls.

The first applicable condition encountered under the implementation's evaluation rules terminates the relevant generation.

If the API returns a finish reason, it may distinguish normal stopping from length exhaustion. Finish-reason field names are provider-specific.

Distinguish **the model selecting EOS** from **the runtime detecting a text stop marker** and from **exhausting the output budget**.

### Story Bridge 9 — Stopping Depends on Which Candidates the Decoder Allows

Those stopping rules matter only if the decoding process actually allows the relevant ending candidate to survive. So the next issue is how greedy decoding, sampling filters, and beam search interact with EOS.

## Question 9 — Can the decoding strategy change whether EOS is selected or considered finished?

**Greedy:** If EOS has the greatest eligible score, it is selected and the relevant response usually ends.

**Top-k:** EOS must be among the retained top-k candidates to remain sampleable.

**Top-p:** EOS must fall inside the nucleus after probability-based filtering to remain sampleable.

**Beam search:** An extension that emits EOS becomes a completed hypothesis. Other unfinished beams may continue expanding. Final selection and early stopping depend on the configured scoring, length penalty, and finishing rules.

Minimum-length constraints can suppress EOS until a certain output length; other constraints can alter which ending markers remain eligible.

# Deep Dive 2 — EOS Handling During Beam Search

The existing EOS discussion covers single-sequence decoding. **Beam search requires extra care because one hypothesis can finish while several others remain active.**

## 1. Why doesn't EOS in one beam immediately end the whole search?

Each beam represents a different partial output. For beam width $w=2$, imagine these two hypotheses are active:

~~~text
Beam A: "travel"     cumulative probability 0.60
Beam B: "learn"      cumulative probability 0.40
~~~

A beam producing EOS becomes a *completed candidate*. But another unfinished beam might later produce a completed sequence with a better score under the search's scoring policy.

Therefore, most beam-search algorithms distinguish:

- **Active hypotheses:** Unfinished sequences eligible for another decode step.
- **Completed hypotheses:** Sequences that already produced EOS and can be considered for final output.

A completed hypothesis is not normally expanded again.

## 2. A complete numerical example

Suppose Beam A has cumulative probability 0.60 and Beam B has 0.40. Their next-token distributions are:

| Parent | Proposed next token | Conditional probability | Candidate sequence probability |
|---|---|---:|---:|
| A | EOS | 0.50 | $0.60\times0.50=0.30$ |
| A | X | 0.30 | $0.60\times0.30=0.18$ |
| A | Y | 0.20 | $0.60\times0.20=0.12$ |
| B | EOS | 0.10 | $0.40\times0.10=0.04$ |
| B | Z | 0.80 | $0.40\times0.80=0.32$ |
| B | W | 0.10 | $0.40\times0.10=0.04$ |

The **global candidate ranking** is:

| Rank | Candidate | Cumulative probability | Status |
|---:|---|---:|---|
| 1 | B → Z | 0.32 | Unfinished |
| 2 | A → EOS | 0.30 | Completed |
| 3 | A → X | 0.18 | Unfinished |
| 4 | A → Y | 0.12 | Unfinished |
| 5–6 | B → EOS or W | 0.04 | Completed / unfinished |

A common implementation stores A → EOS in its **completed-hypothesis pool** and continues with high-scoring unfinished hypotheses. It may retain B → Z and A → X as its next two active beams, depending on its candidate-pool and EOS-processing rules.

**Important:** Implementation details vary. A naïve scheme that selects exactly two candidates and then simply discards completed ones would have only one active survivor here. Many practical beam-search implementations inspect more than $w$ candidate extensions to keep enough unfinished beams after processing EOS.

## 3. Why can't we directly compare a completed score to an unfinished prefix as if both were final outputs?

A completed candidate, A → EOS, has raw probability:

$$
P(A,\mathrm{EOS})=0.30
$$

The unfinished prefix B → Z has probability:

$$
P(B,Z)=0.32
$$

But B → Z is **not a completed answer**. It still needs to generate at least one ending token under a policy that requires EOS.

Suppose B → Z predicts:

$$
P(\mathrm{EOS}\mid B,Z)=0.99
$$

Its completed-sequence probability becomes:

$$
P(B,Z,\mathrm{EOS})=0.32\times0.99=0.3168
$$

Now compare the two completed candidates:

$$
0.3168>0.30
$$

B → Z → EOS wins under **raw sequence probability**.

If instead its next EOS probability had been 0.80:

$$
P(B,Z,\mathrm{EOS})=0.32\times0.80=0.256
$$

then A → EOS would have the greater raw completed score among those two.

This explains why **a high-scoring unfinished beam must be allowed to continue when it can still produce a better completed sequence**.

## 4. What happens when enough completed hypotheses have been found?

Finding $w$ completed candidates does **not automatically imply** that every unfinished candidate is hopeless. A stopping rule must account for whether continuing an active beam could change the final ranking.

For raw cumulative log probability, a useful upper-bound observation is:

$$
S(y_{1:t+1}) = S(y_{1:t}) + \log P(y_{t+1} \vert x, y_{1:t}) \le S(y_{1:t})
$$


So the raw log score of an unfinished prefix is an **upper bound** on the raw score of any continuation of that exact prefix. If even the best possible unfinished continuation cannot outrank the completed candidates needed for the final result, an implementation can safely stop under those simplified scoring assumptions.

Length penalties and other modified scoring rules complicate this reasoning: extending a sequence can alter its normalized score, so stopping criteria must match the **actual** scoring scheme.

## 5. How do custom stop strings interact with beam search?

EOS naturally produces a distinct completed-token hypothesis. A custom stop string may instead be detected by the **serving layer** after examining generated text. Its treatment in beam search depends on whether the implementation supports matching per beam, which candidate histories are inspected, and whether the detected string is retained or removed.

Do not assume that every sampling-oriented custom-stop API automatically implements identical semantics for beam search. For a given library, inspect its beam-search stop-criteria behavior.

## 6. What is the final conceptual algorithm?

~~~text
Active beam hypotheses
         ↓
Beam-specific next-token distributions
         ↓
Candidate beam–token extension scores
         ↓
Rank candidate extensions globally
         ↓
EOS-ending extension? ── Yes → Completed-hypothesis pool
         ↓ No
Choose high-scoring unfinished successors
         ↓
Check whether the stopping rule is satisfied
         ↓ No
Expand unfinished successors again
         ↓ Yes
Select best completed result under configured scoring
~~~

**Core lesson:** EOS completes **a beam**, not necessarily **the whole beam-search procedure**. The search finishes under a separate global stopping rule.

---

### Story Bridge 10 — Before Leaving Stop Tokens, Resolve the Remaining Edge Cases

Beam search shows that even “EOS was generated” can have different meanings depending on whether we are following one sequence or several hypotheses. Before leaving stop tokens, we should collect the remaining edge cases that can make stopping behavior surprising.

## Question 10 — What stop-token edge cases remain before we move to output limits?

- **Immediate EOS:** The model may choose EOS as its first output token, producing no ordinary text.
- **EOS never selected:** Generation can continue until another stop condition or maximum length is reached.
- **Masked EOS:** Constraints may temporarily forbid EOS.
- **Different special markers:** End-of-message and end-of-assistant-turn need not mean the same thing.
- **Stop marker inside content:** A custom delimiter inside quoted text or code can trigger premature stopping under naïve matching.
- **Multi-token marker:** A partial stop-string match may need streaming buffer management.
- **EOS versus visible output:** Special tokens may count internally but be omitted from the user-visible response.

---


### Story Bridge 11 — Natural Endings Are Not Enough

At this point, natural and application-defined stopping are clear. But one problem remains: **what if none of those stopping conditions ever occurs?** A production system still needs a hard boundary on how long generation is allowed to continue. That is the role of maximum output length.


# 8.13 Maximum Output Length

## Question 1 — What if no stop condition ever occurs — what prevents generation from continuing indefinitely?

It is a configured cap on how many **new output tokens** a generation request may produce. Let the requested budget be:

$$
M
$$

Then a typical rule is:

$$
N_{\mathrm{generated}}\le M
$$

where the counted generated tokens follow the API's conventions. An earlier EOS or custom stop condition can terminate the response before $M$ is reached.

The cap is an **upper bound**, not an instruction to produce exactly that many tokens.

### Story Bridge 12 — Output Budget and Context Capacity Are Different Limits

A hard output cap prevents unbounded generation. But the model already has another limit—the context window. The two limits interact, but they are not the same quantity.

## Question 2 — Is maximum output length the same thing as the context window?

The **context window** describes the token capacity available to the model at a time. The **maximum output length** is the limit on new tokens generated for this particular request.

Let:

- $C$ = context-window capacity in tokens.
- $N_{\mathrm{prompt}}$ = tokens consumed by the input.
- $M$ = requested maximum new output tokens.

In a simplified fixed-context setup with no extra reserved tokens or context-management mechanisms:

$$
N_{\mathrm{prompt}}+M\le C
$$

If this condition is not met, the runtime must apply its documented policy, such as rejecting the request, restricting output, or managing context. Some models also have a separate model-specific maximum output cap.

An output budget of 500 tokens does **not** mean the model has only a 500-token context window.

### Story Bridge 13 — The Prompt Consumes Part of the Available Context

Once we separate the output budget from context capacity, we can calculate the practical room left for generation instead of treating the requested maximum as automatically available.

## Question 3 — Given the prompt size and context window, how much output can actually fit?

Suppose:

$$
C=8192
$$

and:

$$
N_{\mathrm{prompt}}=7000
$$

The unused context capacity is:

$$
C-N_{\mathrm{prompt}}=8192-7000=1192
$$

If the request specifies:

$$
M=500
$$

then the effective output ceiling in this simplified case is 500 new tokens.

If the request instead specifies 2,000 new tokens, the available 1,192 context positions are insufficient under a fixed-window policy.

Conceptually, with no additional constraints:

$$
N_{\mathrm{generated}}
\le
\min\left(M,\;C-N_{\mathrm{prompt}}\right)
$$

Real systems may reserve special tokens, impose separate output caps, or use other context-management strategies.


### Story Bridge 14 — Now Two Independent Stopping Mechanisms Can Compete

Now a request has two independent ways to stop: a recognized ending can occur first, or the mechanical token budget can run out first. The runtime stops according to whichever applicable condition is reached first.

## Question 4 — What happens when EOS and the output limit compete?

Suppose the maximum output budget is:

$$
M=20
$$

If recognized EOS is selected after 12 counted output tokens, generation stops naturally before the cap.

If no stopping condition occurs before the output count reaches 20, the runtime ends generation because the budget is exhausted.

~~~text
EOS or custom stop first → natural/configured stopping
Maximum output count first → length-limited termination
~~~

Neither case implies the other. A length-limited response can look complete by coincidence, and a naturally stopped response need not use its entire allowance.

# Worked Example — EOS, Custom Stop String, and Output Cap

Suppose the request has these settings:

| Setting | Value |
|---|---:|
| Context window | 8,192 tokens |
| Prompt tokens | 7,000 |
| Maximum new tokens | 500 |
| Custom stop string | [END] |
| EOS stopping | Enabled |

The unused context space is:

$$
8192-7000=1192
$$

The requested 500-token output budget fits in the remaining context capacity under the simplified fixed-window assumptions.

Now consider four possible runs:

| Run | Event | Finish condition |
|---|---|---|
| A | EOS selected at generated token 75 | EOS stop |
| B | Complete [END] marker detected at token 120 | Custom stop |
| C | No prior stop; generated-token count reaches 500 | Length limit |
| D | EOS selected as first generated token | EOS, possibly empty visible reply |

For B, whether the stop marker is included in the returned text depends on API semantics.

For C, the output may be incomplete because the cap is unrelated to the meaning of the generated text.

## Conceptual generation pseudocode

~~~python
generated = []
finish_reason = None

# After prefill, the first next-token logits are already available.
logits = prefill_logits

for step in range(max_new_tokens):
    token = select_token(logits, decoding_config)
    generated.append(token)

    if token in configured_end_token_ids:
        finish_reason = "stop"
        break

    if complete_custom_stop_match(generated):
        finish_reason = "stop"
        break

    # The selected token becomes the input for the next decode step.
    logits = decode_forward(token, kv_cache)
else:
    finish_reason = "length"
~~~

This is **conceptual pseudocode**, not a universal provider implementation. Real runtimes differ in how they buffer streaming stop sequences, handle multiple active beams or requests, count output tokens, and expose stop reasons.

---

### Story Bridge 15 — A Counter Does Not Know Whether the Answer Is Complete

The combined example makes the distinction concrete: an EOS stop reflects a recognized ending, while a length stop reflects a counter. That means a length-limited response can end even when the model was still in the middle of expressing something.

## Question 5 — Why can reaching the length limit leave an incomplete response?

Maximum length is a mechanical constraint. It does not know whether the output has finished a sentence, closed a bracket, completed a JSON object, or answered the user.

For example:

~~~text
Three reasons explain the result:
1. Insufficient data
2. Training–serving mismatch
3.
~~~

A generation result with a length-related finish reason should be treated as **potentially truncated** even if the returned text looks superficially plausible.

# Deep Dive 1 — Token-by-Token Trace of Three Different Stopping Outcomes

The earlier sections explain stopping conditions individually. Now we will follow **three requests with identical settings**, observing every selected token, the output counter, and the reason generation terminates.

## 1. Define the request and counting conventions

Assume the following illustrative configuration:

| Setting | Value |
|---|---|
| Prompt | "Give the result." |
| Maximum new tokens | 8 |
| EOS | Enabled |
| Custom stop string | [END] |
| Decoding method | Greedy, for simplicity |

For this **toy example only**, assume the visible strings in the next table correspond to the individual token IDs shown. Actual tokenization is model-dependent: a word, leading space, punctuation mark, or the string "[END]" can be split differently.

We will also assume:

- Each selected token—including EOS and the tokens forming a stop string—counts toward this illustrative eight-token budget.
- The runtime checks enabled stopping conditions after each selection.
- The complete custom stop string is removed from the user-visible result.
- Each of the three traces starts from the **same prompt and settings**, but represents a hypothetical different generation outcome. Under fixed greedy decoding with the *same unmodified model and identical prompt*, three different outcomes would not all occur; these are **counterfactual traces** for learning the stopping logic, not claims that greedy decoding randomly changes its output.

This last distinction matters: our purpose here is to compare termination rules while holding the **configuration** constant.

## 2. Run A — EOS is selected before the budget expires

The illustrative token sequence is:

~~~text
The | result | is | 42 | EOS
~~~

| Selection step | Selected token | Count after selection | Has EOS appeared? | Action |
|---:|---|---:|---|---|
| 1 | The | 1 | No | Continue |
| 2 | result | 2 | No | Continue |
| 3 | is | 3 | No | Continue |
| 4 | 42 | 4 | No | Continue |
| 5 | EOS | 5 | Yes | Finish normally |

At step 5, the ending token is selected **before the eight-token limit**. Under the stated assumptions:

$$
N_{\mathrm{generated}}=5<8
$$

The user-visible answer may contain just:

~~~text
The result is 42
~~~

EOS is an internal special token and is commonly omitted from returned text. The finish reason would be a stop-related value under a typical API.

**Why is there no sixth decode step?** Because the runtime has already recognized a stopping condition after selecting EOS. It does not need another forward pass.

## 3. Run B — A custom stop string completes first

Now suppose the selected token sequence is:

~~~text
The | result | is | 42 | [ | END | ]
~~~

Assume the last three selected token IDs decode to the characters of the configured stop string "[END]".

| Selection step | Selected token | Count after selection | Stop-string state | Action |
|---:|---|---:|---|---|
| 1 | The | 1 | No match | Continue |
| 2 | result | 2 | No match | Continue |
| 3 | is | 3 | No match | Continue |
| 4 | 42 | 4 | No match | Continue |
| 5 | [ | 5 | Possible partial match | Continue; buffer if streaming |
| 6 | END | 6 | Longer partial match | Continue; buffer if streaming |
| 7 | ] | 7 | Complete "[END]" match | Finish |

There is **no EOS token** in this hypothetical run. The serving runtime finishes because it has detected the configured stop pattern.

Under our illustrative counting convention:

$$
N_{\mathrm{generated}}=7<8
$$

If the API excludes the matching stop sequence from returned text, the visible result may again be:

~~~text
The result is 42
~~~

Notice that Runs A and B can return **the same visible text but have different internal stopping triggers and different token counts**.

Why is streaming tricky? At step 5, the runtime has seen only "[". It may hold this possible delimiter in a short buffer until it knows whether the following output completes "[END]" or should be released as ordinary text. A stop string can also cross arbitrary tokenizer-token boundaries.

## 4. Run C — No stop condition occurs before the output budget runs out

Suppose the selected tokens are:

~~~text
The | result | is | 42 | because | the | answer | continues
~~~

| Selection step | Selected token | Count after selection | EOS or completed stop string? | Action |
|---:|---|---:|---|---|
| 1 | The | 1 | No | Continue |
| 2 | result | 2 | No | Continue |
| 3 | is | 3 | No | Continue |
| 4 | 42 | 4 | No | Continue |
| 5 | because | 5 | No | Continue |
| 6 | the | 6 | No | Continue |
| 7 | answer | 7 | No | Continue |
| 8 | continues | 8 | No | Finish: length limit |

Now:

$$
N_{\mathrm{generated}}=M=8
$$

The model did not choose EOS, and no complete stop string appeared. The runtime terminates because the token budget has been exhausted.

The visible output might be grammatically incomplete:

~~~text
The result is 42 because the answer continues
~~~

The system cannot infer that the sentence is semantically complete merely from the output counter.

## 5. Compare the three traces

| Run | Counted generated tokens | Actual termination trigger | Illustrative finish category |
|---|---:|---|---|
| A | 5 | Recognized EOS | Stop |
| B | 7 | Complete custom stop string | Stop |
| C | 8 | Output cap reached | Length |

**Core distinction:** Stop tokens and patterns indicate *an ending condition*; maximum output length indicates *a budget boundary*. A length-related finish reason warns that the text may be truncated.

If a stop token and the output budget are encountered on the same counted position, the reported finish reason may depend on the runtime's evaluation order. Do not assume a universal tie-breaking rule.

## 6. How does this connect to prefill and decode?

For an existing prompt of $n$ tokens:

1. **Prefill** processes the prompt and produces logits for the first generated token $y_1$.
2. Select $y_1$ and check the stop conditions and output count.
3. If generation continues, the **first one-token decode forward pass** processes $y_1$ with its relevant KV state, producing logits for $y_2$.
4. Select $y_2$, check the stopping rules, and continue.

The same logic repeats until EOS, a recognized stop string, a configured cancellation, or an output limit terminates generation.

The selected token $y_t$ is known **before** the forward pass that predicts $y_{t+1}$; no forward pass is performed to predict a token after generation has already terminated.

---

### Story Bridge 16 — Before Using a Limit, We Must Know What It Counts

Once truncation is possible, the next practical question is what the configured number actually counts. Different APIs may limit newly generated tokens or the total input-plus-output sequence.

## Question 6 — What exactly does the limit count — new tokens or total sequence length?

Some APIs count only newly generated tokens; others expose a limit on total input-plus-output sequence length.

For a prompt with 800 tokens:

| Configured restriction | Maximum possible new tokens |
|---|---:|
| Maximum new tokens = 200 | 200 |
| Maximum total sequence length = 900 | 100 |

For the second row:

$$
900-800=100
$$

Confirm a particular library's parameter names, precedence rules, reserved special-token accounting, and any model-specific maximum.


### Story Bridge 17 — The Budget Unit Is Tokens, Not Human Words

Knowing whether the limit applies to new or total tokens still leaves one basic unit question: the model does not budget output in human words or characters—it works in tokenizer units.

## Question 7 — Are those limits measured in words, characters, or tokens?

Usually not. They are measured in **tokens**, which can correspond to whole words, parts of words, punctuation, spaces attached to text, or other tokenizer units.

A limit of 100 output tokens does not mean 100 words or 100 characters.

# Deep Dive 3 — Realistic Token Accounting and Finish Reasons

The basic relationship between context size and maximum output length is useful, but real API requests contain more than the text typed by the user.

## 1. What can contribute to input-token usage?

A chat request may include:

- The system and developer instructions supplied to the model.
- Prior conversation history and the current user message.
- Chat-template delimiters, message-role markers, and other special tokens.
- Tool schemas, tool outputs, retrieved context, or multimodal tokenized content, where applicable.

An API may represent, count, reserve, or report these categories differently. The point of this example is to demonstrate **budget arithmetic**, not assert a universal tokenizer accounting policy.

## 2. A numerical context-budget example

Assume a hypothetical serving configuration with:

| Category | Illustrative token count |
|---|---:|
| Model context capacity | 8,192 |
| Current and prior user/assistant text | 6,800 |
| System and developer instructions | 120 |
| Chat-template and message delimiters | 80 |
| Tool and retrieved-context material | 60 |
| Additional reserved context positions | 32 |
| Requested maximum new output tokens | 1,500 |
| Separate model-specific maximum output | 1,024 |

The counted prompt occupancy is:

$$
N_{\mathrm{prompt}} = 6800 + 120 + 80 + 60 = 7060
$$


Subtract it and the illustrative 32-position reserve:

$$
C_{\mathrm{remaining}} = 8192 - 7060 - 32 = 1100
$$

The request asks for 1,500 new tokens, but the remaining simplified context capacity is 1,100, and the model also has a separate 1,024-token output cap.

If the serving system implements all three as hard ceilings, the effective maximum is:

$$
M_{\mathrm{effective}} = \min(1500,\;1100,\;1024) = 1024
$$

This does **not** mean the provider must silently lower the user's requested maximum from 1,500 to 1,024. Depending on the API, the request may be rejected, adjusted, or handled under a different context-management policy. The calculation describes the intersection of those ceilings **if generation is allowed under this simplified setup**.

If the request instead asks for only 900 new tokens, the requested cap becomes the tightest:

$$
\min(900,\;1100,\;1024)=900
$$

## 3. Why is the distinction between visible text tokens and billed/reported tokens important?

Some services account for output categories that are not fully visible in the final answer, such as special tokens or model-internal reasoning tokens. Some count these against a shared output budget; others impose different budgets or reporting conventions.

Therefore, a final user-visible reply that looks short does not necessarily imply that an API request used only a small number of counted output tokens.

**Do not assume identical accounting across model families, SDKs, or providers.** Inspect the particular API's usage fields and output-limit documentation.

## 4. How do finish reasons help distinguish natural completion from truncation?

Consider these illustrative outcomes:

| Returned result | Why generation ended | Interpretation |
|---|---|---|
| "The answer is 42." | EOS selected | Model/runtime recognized a natural ending |
| "The answer is 42." | Custom stop marker matched | Runtime recognized a configured delimiter |
| "The answer is 42 becau" | Token budget exhausted | Output may be truncated |
| "The answer is 42." | Token budget exhausted immediately after the period | Text looks complete, but termination was mechanical |

The final two rows are both **length-based termination**, even though one appears complete to a human reader.

For a provider exposing stop and length finish categories:

~~~text
finish_reason = "stop"
→ Recognized ending token or configured stop condition

finish_reason = "length"
→ Output or model-generation budget exhausted
~~~

These names are illustrative common conventions; the real service may use different names or expose additional finish reasons.

## 5. How should an application react to a length-related finish reason?

The appropriate response depends on the task.

For conversational prose, the application may ask for a continuation or inform the user that the answer was truncated. For structured outputs, it should not assume an incomplete JSON object is valid merely because some fields have already been produced.

If continuation is supported, the developer needs to account for the existing generated history, remaining context capacity, duplicate text, and whether the interrupted format can be continued safely.

Increasing the token budget can help **only if** the model and context limits permit it. Otherwise, summarizing or reducing the input, using a supported longer-context configuration, or splitting the task may be necessary.

## 6. A checklist for reasoning about any generation limit

Before concluding that a response "should have had room to finish", identify:

1. The full tokenized input length, including applicable chat-template and tool-context overhead.
2. The requested *new-token* budget and whether it includes any non-visible output categories.
3. The model's independent maximum output limit, if any.
4. The available context capacity and any reserved positions or context-management behavior.
5. The configured EOS, end-of-turn, and custom stop conditions.
6. The actual finish reason returned by the API.

**Core lesson:** The number of words visible to the user, the model's context capacity, and the runtime's output-token budget are related but different quantities.

---

### Story Bridge 18 — A Maximum Is Permission, Not a Target

Real token accounting shows why a configured maximum is only one ceiling among several. A larger ceiling therefore means “generation may continue longer,” not “the model must fill the allowance.”

## Question 8 — Does a larger output limit force the model to generate a longer answer?

No. A model can select EOS early, or the runtime can detect another configured stop condition, leaving much of the budget unused.

A higher ceiling *permits* longer outputs and increases the possible worst-case runtime. It does not force the model to fill the entire budget.

### Story Bridge 19 — More Output Means More Sequential Decode Work

If a larger budget permits more tokens, the first systems consequence is time: autoregressive generation requires additional sequential decode iterations for additional output positions.

## Question 9 — Why does a longer output increase inference latency?

Autoregressive decoding typically produces new tokens sequentially. If $N$ tokens are generated and iteration $j$ takes $t_j$ seconds, decode time is:

$$
T_{\mathrm{decode}}=\sum_{j=1}^{N}t_j
$$

With roughly constant average iteration time:

$$
T_{\mathrm{decode}}\approx N\bar{t}
$$

Total request time also includes prefill and serving overhead:

$$
T_{\mathrm{request}}
\approx
T_{\mathrm{prefill}}+
T_{\mathrm{decode}}+
T_{\mathrm{overhead}}
$$

Generating more tokens generally increases decode time. The rate need not be constant: context length, batching, hardware, and memory bandwidth affect individual token latency.

### Story Bridge 20 — More Output Also Means a Larger Cached History

More decode iterations do not only consume time. Every additional generated position can also extend the cached history, so output length has a memory consequence as well.

## Question 10 — Why does a longer output increase KV-cache requirements?

For ordinary cached decoding, additional generated positions add Keys and Values to the cache, unless a specialized cache-management method changes that behavior.

A simplified KV-memory estimate is:

$$
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{KV}}d_h b
$$

where:

- $L$ = Transformer layer count.
- $B$ = number of active sequence histories.
- $T$ = cached sequence length.
- $H_{\mathrm{KV}}$ = number of KV heads.
- $d_h$ = head dimension.
- $b$ = bytes per cached element.

Allowing more output tokens raises the potential cached sequence length $T$. Actual physical memory depends on allocation strategy, prefix sharing, cache eviction, sliding-window attention, and serving implementation.

The maximum output budget does not imply that every request consumes its full possible cache allocation.


### Story Bridge 21 — A Single Request Becomes a Shared-Serving Problem

For one request, longer output means more latency and potentially more KV memory. In production, many such requests coexist, so their different lifetimes begin to affect the shared batch and scheduler.

## Question 11 — What happens when requests with very different output lengths are batched together?

Different requests can finish at different times:

~~~text
Request A: finishes after 20 output tokens
Request B: finishes after 100 output tokens
Request C: hits its cap at 200 output tokens
~~~

Serving systems can remove completed requests from the active batch and often schedule waiting requests into newly available capacity.

Stop conditions and output ceilings therefore affect token throughput, active batch composition, waiting time, and maximum resource demand.

### Story Bridge 22 — Finish Reasons Tell Us Which Exit Door Was Used

Once output limits affect both correctness and serving resources, the final practical question is diagnostic: **when a response ends, how do we know whether it ended naturally or because some limit was reached?**

## Question 12 — What should we check when a response stops near or at its output limit?

The first thing to inspect is the runtime's **finish reason**, when the API provides one. A stop-related finish reason generally indicates a recognized ending condition such as EOS or a configured stop marker, while a length-related finish reason indicates that an output or generation budget was exhausted.

That distinction matters because visible text alone can be misleading: a response can look complete and still have ended mechanically at its limit, or it can look abruptly cut off because the length cap arrived first.

Beyond the finish reason, several edge cases affect how the limit should be interpreted:


- **Very small budget:** Output may truncate mid-sentence or mid-structure.
- **Very large budget:** Actual generation can still finish early if EOS is selected.
- **Prompt nearly fills context:** Remaining space can limit output unless context management is available.
- **No natural stop:** The hard cap bounds generation.
- **Structured output:** An insufficient budget can truncate JSON or other required formats.
- **Special token accounting:** Hidden output tokens and EOS counting may differ across runtimes.
- **Provider finish reason:** Use it to distinguish normal stopping from length-limited output.

---


We can now reconstruct the chapter as one causal chain: generation needs a stopping rule; EOS provides a learned ending; custom sequences add application-defined endings; a hard token cap protects us when no ending occurs; and that cap affects truncation, latency, KV memory, and batched serving. The final comparison compresses that whole story.

# Part 7 — Final Comparison

| Property | Stop tokens and stop sequences | Maximum output length |
|---|---|---|
| Question | Has a recognized stopping event occurred? | Has the output budget been exhausted? |
| Trigger | Selected ending token or matched pattern | Counted token limit reached |
| Can terminate before the budget? | Yes | No; it defines the budget |
| Typical finish reason | Stop or equivalent | Length or equivalent |
| Main concern | Correct delimiter semantics and detection | Truncation and resource limits |

# Most Important Equations

**Next-token probability:**

$$
P(y_{t+1}\mid x,y_{1:t})
$$

**Output budget:**

$$
N_{\mathrm{generated}}\le M
$$

**Simplified context-constrained output ceiling:**

$$
N_{\mathrm{generated}}
\le
\min\left(M,\;C-N_{\mathrm{prompt}}\right)
$$

**Decode time:**

$$
T_{\mathrm{decode}}=\sum_{j=1}^{N}t_j
$$

**Simplified KV-cache memory:**

$$
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{KV}}d_h b
$$

# Part 7 Mental Model

~~~text
Prompt prefill
    ↓
First-token selection
    ↓
Recognized EOS or custom stop?
    ├── Yes → Finish
    └── No
         ↓
    Output budget exhausted?
         ├── Yes → Finish: length
         └── No → Decode next token and repeat
~~~

**Stop criteria determine whether a recognized ending occurred. Maximum output length limits how long the system may continue when it has not.**

# Bridge to Part 8

The remaining syllabus topics explain the system-level consequences of generating one token at a time:

- **8.14 Why Generation Is Sequential**
- **8.15 Why Inference Is Expensive**

These connect autoregressive dependence, prefill, decode, KV-cache growth, attention costs, memory bandwidth, and token throughput.
