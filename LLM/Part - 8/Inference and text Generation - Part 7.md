# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 7

## 8.12 Stop Tokens
## 8.13 Maximum Output Length

**Detailed Study Mode.** The interview-style version remains a separate file.

Parts 1–6 explained prefill, decode, KV cache, greedy selection, temperature, top-k, top-p, beam search, and repetition controls. We now need to answer **when generation stops**.

Stop tokens or configured stop sequences can terminate an answer before its budget is exhausted. Maximum output length imposes a ceiling even if the model never produces a natural ending.

---

# 8.12 Stop Tokens

## Question 1 — Why does autoregressive generation need a stopping condition?

At each position, a decoder-only language model predicts a distribution for the next token:

$$
P(y_{t+1}\mid x,y_{1:t})
$$

Here $x$ is the prompt, $y_{1:t}$ is the generated continuation so far, and $y_{t+1}$ is the next candidate token.

Next-token prediction alone does not enforce an ending. A serving system therefore needs stopping criteria, including recognized ending tokens, custom stop conditions, and an output-token budget.

A generation request may finish naturally after a few tokens or be interrupted mechanically at its configured limit.

## Question 2 — What is an EOS token?

EOS stands for **end of sequence**. It is typically a dedicated tokenizer token ID that may be predicted at positions where the model has learned a sequence should end.

Different model families use different token spellings and IDs. Examples of visible token spellings include:

~~~text
</s>
<|endoftext|>
<|eot_id|>
~~~

These markers are **not universally interchangeable**. Some chat formats distinguish end of message, end of assistant turn, and control handoff. Inspect the tokenizer and chat template to determine which marker should terminate a particular response.

## Question 3 — Does EOS belong to the same vocabulary distribution as ordinary tokens?

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

## Question 4 — What happens after EOS is selected?

A typical generation loop does the following:

1. Prefill the prompt and select the first generated token from the resulting logits.
2. For subsequent positions, process the most recently selected token using a decode forward pass and relevant KV cache.
3. Use the decoding rule to select the next token.
4. Check the selected token against configured ending-token IDs and other stopping conditions.
5. If an enabled stopping condition has occurred, finish; otherwise continue within the token budget.

Serving implementations may store EOS internally while hiding its special-token spelling from returned text.

**Important distinction:** The first generated token can be selected directly from prefill logits. A separate one-token decode forward pass is not needed before that first selection.

## Question 5 — Is the ordinary text "I am done" equivalent to EOS?

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

## Question 6 — How are special EOS tokens different from custom stop sequences?

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

## Question 7 — What happens when a custom stop sequence spans multiple tokens?

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

## Question 8 — What if there are several possible stopping conditions?

A request may configure EOS IDs, custom stop strings, a maximum new-token limit, and possibly cancellation or other application-defined controls.

The first applicable condition encountered under the implementation's evaluation rules terminates the relevant generation.

If the API returns a finish reason, it may distinguish normal stopping from length exhaustion. Finish-reason field names are provider-specific.

Distinguish **the model selecting EOS** from **the runtime detecting a text stop marker** and from **exhausting the output budget**.

## Question 9 — How do greedy, top-k, top-p, and beam search interact with EOS?

**Greedy:** If EOS has the greatest eligible score, it is selected and the relevant response usually ends.

**Top-k:** EOS must be among the retained top-k candidates to remain sampleable.

**Top-p:** EOS must fall inside the nucleus after probability-based filtering to remain sampleable.

**Beam search:** An extension that emits EOS becomes a completed hypothesis. Other unfinished beams may continue expanding. Final selection and early stopping depend on the configured scoring, length penalty, and finishing rules.

Minimum-length constraints can suppress EOS until a certain output length; other constraints can alter which ending markers remain eligible.

## Question 10 — What EOS and stop-sequence edge cases should we understand?

- **Immediate EOS:** The model may choose EOS as its first output token, producing no ordinary text.
- **EOS never selected:** Generation can continue until another stop condition or maximum length is reached.
- **Masked EOS:** Constraints may temporarily forbid EOS.
- **Different special markers:** End-of-message and end-of-assistant-turn need not mean the same thing.
- **Stop marker inside content:** A custom delimiter inside quoted text or code can trigger premature stopping under naïve matching.
- **Multi-token marker:** A partial stop-string match may need streaming buffer management.
- **EOS versus visible output:** Special tokens may count internally but be omitted from the user-visible response.

---

# 8.13 Maximum Output Length

## Question 1 — What is maximum output length?

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

## Question 2 — How is output length different from the context window?

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

## Question 3 — Can you calculate a realistic token budget?

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

## Question 4 — What happens when EOS and maximum length compete?

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

## Question 5 — Why can a length limit produce incomplete output?

Maximum length is a mechanical constraint. It does not know whether the output has finished a sentence, closed a bracket, completed a JSON object, or answered the user.

For example:

~~~text
Three reasons explain the result:
1. Insufficient data
2. Training–serving mismatch
3.
~~~

A generation result with a length-related finish reason should be treated as **potentially truncated** even if the returned text looks superficially plausible.

## Question 6 — Maximum new tokens versus maximum total sequence length: what changes?

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

## Question 7 — Are output limits measured in words?

Usually not. They are measured in **tokens**, which can correspond to whole words, parts of words, punctuation, spaces attached to text, or other tokenizer units.

A limit of 100 output tokens does not mean 100 words or 100 characters.

## Question 8 — Does a higher maximum output length force longer generation?

No. A model can select EOS early, or the runtime can detect another configured stop condition, leaving much of the budget unused.

A higher ceiling *permits* longer outputs and increases the possible worst-case runtime. It does not force the model to fill the entire budget.

## Question 9 — How does output length affect inference latency?

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

## Question 10 — How does output length affect the KV cache?

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

## Question 11 — Why does maximum output length matter for batched serving?

Different requests can finish at different times:

~~~text
Request A: finishes after 20 output tokens
Request B: finishes after 100 output tokens
Request C: hits its cap at 200 output tokens
~~~

Serving systems can remove completed requests from the active batch and often schedule waiting requests into newly available capacity.

Stop conditions and output ceilings therefore affect token throughput, active batch composition, waiting time, and maximum resource demand.

## Question 12 — What are important maximum-output-length edge cases?

- **Very small budget:** Output may truncate mid-sentence or mid-structure.
- **Very large budget:** Actual generation can still finish early if EOS is selected.
- **Prompt nearly fills context:** Remaining space can limit output unless context management is available.
- **No natural stop:** The hard cap bounds generation.
- **Structured output:** An insufficient budget can truncate JSON or other required formats.
- **Special token accounting:** Hidden output tokens and EOS counting may differ across runtimes.
- **Provider finish reason:** Use it to distinguish normal stopping from length-limited output.

---

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
