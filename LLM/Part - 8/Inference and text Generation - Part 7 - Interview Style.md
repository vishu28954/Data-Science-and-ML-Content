# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 7 — Interview Style

## Topics
- **8.12 Stop Tokens**
- **8.13 Maximum Output Length**

These notes are for **interview preparation only**. The separate [Part 7 Detailed Study](./Inference%20and%20text%20Generation%20-%20Part%207.md) contains the full derivations, token-by-token traces, beam-search stopping examples, and token-accounting discussion.

---

# 8.12 Stop Tokens

## Q1. Why does an autoregressive LLM need a stopping condition?

**Interview answer:** An autoregressive model predicts one token at a time:

$$
P(y_{t+1}\mid x,y_{1:t})
$$

This next-token process does not itself enforce a maximum response length. The serving system therefore needs stopping criteria such as EOS tokens, custom stop sequences, or a maximum output-token budget.

The key idea is that **generation can end because the model/runtime recognized an ending condition, or because the system exhausted a resource limit**.

## Q2. What is an EOS token?

**Interview answer:** EOS means **end of sequence**. It is typically a special tokenizer token ID that the model can predict when a sequence or turn should end. If the serving system recognizes the selected EOS token as a stopping token, it terminates generation for that sequence.

The exact EOS ID and visible spelling are model-specific. Chat models may distinguish end-of-message, end-of-turn, and tool-related control tokens, so those markers should not automatically be treated as equivalent.

## Q3. Is EOS part of the same next-token distribution as normal tokens?

**Interview answer:** Yes, when EOS is an eligible vocabulary token. The model produces a logit for EOS just like it produces logits for ordinary text tokens.

Suppose the next-token probabilities are:

| Candidate | Probability |
|---|---:|
| EOS | 0.65 |
| Period | 0.20 |
| because | 0.10 |
| Other tokens | 0.05 |

With greedy decoding, EOS is selected because it has the highest probability. With sampling, EOS is selected only if the sampling process chooses it after all configured filters and penalties.

**A high EOS probability alone does not stop generation. EOS must actually be selected and recognized.**

## Q4. What happens after EOS is selected?

**Interview answer:** Once a recognized EOS token is selected, the current sequence is marked complete and no additional decode step is needed for that sequence. The EOS token may exist internally in the token sequence but is often omitted from user-visible text.

The first generated token is selected from the logits produced by prefill. For later tokens, each selected token is processed in the next one-token decode pass only if generation has not already terminated.

## Q5. Is the phrase "I am done" equivalent to EOS?

**Interview answer:** No. "I am done" is ordinary text unless the application explicitly configures it as a stop pattern. The model can generate a sentence that sounds complete and still continue producing tokens.

Similarly, a period, newline, or closing quotation mark is not universally a stopping token.

## Q6. What is the difference between EOS and a custom stop sequence?

**Interview answer:** EOS is generally a model/tokenizer-defined special token ID. A custom stop sequence is a caller-defined pattern monitored by the serving runtime.

A custom stop sequence can span multiple tokenizer tokens. For example, an application might use:

~~~text
[END]
~~~

The runtime may need to watch a token stream or decoded-text buffer until the complete marker appears.

| EOS | Custom stop sequence |
|---|---|
| Usually tokenizer-defined | Usually application-defined |
| Typically a dedicated token ID | Can span multiple tokens |
| Can be learned as a natural ending | Enforced by runtime matching |
| Usually hidden from visible output | Inclusion depends on API |

## Q7. How does a multi-token custom stop sequence work during streaming?

**Interview answer:** The runtime needs to track partial matches. If the configured stop marker is split into several tokenizer tokens, it cannot know that the stop condition has occurred until the complete sequence is generated.

During streaming, an implementation may temporarily buffer a possible partial match. If later tokens complete the stop marker, the buffered marker can be suppressed; if not, it can be released as normal text.

Exact matching and inclusion semantics are API-specific.

## Q8. How do stop tokens interact with greedy, top-k, and top-p decoding?

**Interview answer:** With greedy decoding, EOS stops generation if it is the highest-ranked eligible token. With top-k, EOS must remain inside the retained top-k candidate set. With top-p, EOS must fall inside the nucleus.

If a generation constraint masks EOS—for example, to enforce a minimum response length—EOS cannot be selected until the constraint allows it.

## Q9. How does EOS work in beam search?

**Interview answer:** In beam search, each beam is a different partial sequence. If one beam generates EOS, that beam becomes a **completed hypothesis**, but unfinished beams may continue.

The beam-search procedure ends only when its global stopping rule is satisfied. Therefore:

> **EOS completes one beam; it does not necessarily terminate the entire beam search.**

## Q10. Why can't beam search always stop as soon as one beam generates EOS?

**Interview answer:** Another unfinished beam may later produce a completed sequence with a better score.

For example, suppose one beam completes with:

$$
P(A,\mathrm{EOS})=0.60\times0.50=0.30
$$

Another unfinished beam has:

$$
P(B,Z)=0.40\times0.80=0.32
$$

If that beam then predicts EOS with probability 0.99:

$$
P(B,Z,\mathrm{EOS})
=
0.32\times0.99
=
0.3168
$$

Since:

$$
0.3168>0.30
$$

the later-completing beam is better under raw sequence probability.

## Q11. What happens when multiple stopping criteria are configured?

**Interview answer:** A request may include EOS tokens, custom stop sequences, a maximum output-token limit, cancellation, or other supported runtime criteria. The implementation stops when an applicable condition is satisfied under its evaluation rules.

A provider may return a **finish reason** indicating whether generation stopped naturally, because of a custom stop condition, because the output limit was reached, or because of another runtime event.

---

# 8.13 Maximum Output Length

## Q12. What is maximum output length?

**Interview answer:** Maximum output length is an upper bound on how many new tokens a request may generate.

If the configured limit is $M$:

$$
N_{\mathrm{generated}}\le M
$$

Generation may stop earlier because of EOS or another stop condition. Therefore, the maximum output length is a **ceiling**, not a target.

## Q13. How is maximum output length different from the context window?

**Interview answer:** The context window limits the total tokenized context that can be processed at once. Maximum output length limits how many **new tokens** may be generated for one request.

In a simplified fixed-context setting:

$$
N_{\mathrm{prompt}}+N_{\mathrm{generated}}\le C
$$

where $C$ is context capacity.

A model can therefore have a very large context window but still be configured to generate only a small number of output tokens.

## Q14. How do you compute the effective output budget?

**Interview answer:** In a simplified setup, generation can be constrained by the requested output limit, remaining context capacity, and any separate model output cap.

A useful conceptual expression is:

$$
M_{\mathrm{effective}}
=
\min\left(
M_{\mathrm{requested}},
C-N_{\mathrm{prompt}},
M_{\mathrm{model}}
\right)
$$

For example, if the requested output is 1,500 tokens, remaining context capacity is 1,100 tokens, and the model-specific output cap is 1,024 tokens:

$$
M_{\mathrm{effective}}
=
\min(1500,\;1100,\;1024)
=
1024
$$

Actual APIs may reject, truncate, or otherwise manage requests differently, so this equation is a conceptual intersection of ceilings, not a universal provider behavior.

## Q15. Can chat-template and system tokens affect the available generation budget?

**Interview answer:** Yes. The model processes more than just the user's visible text. Depending on the system, the input may contain system/developer instructions, conversation history, role markers, special tokens, tool schemas, retrieved context, and other serialized content.

Therefore, visible word count is not the same as tokenized prompt occupancy. Token accounting is provider- and tokenizer-specific.

## Q16. What happens when maximum output length is reached before EOS?

**Interview answer:** The serving system terminates generation even though the model has not produced a recognized natural ending. The result may therefore be truncated mid-sentence, mid-code-block, or mid-JSON object.

A length-related finish reason should be treated as a warning that the result may be incomplete.

## Q17. Does increasing maximum output length force a longer response?

**Interview answer:** No. It only increases the maximum permitted length. The model can still select EOS much earlier.

A larger maximum increases the **worst-case** decoding work and potential KV-cache usage, but it does not guarantee that the request will use the full budget.

## Q18. What is the difference between maximum new tokens and maximum total sequence length?

**Interview answer:** Maximum new tokens counts only generated output. Maximum total sequence length includes prompt tokens plus generated tokens.

For an 800-token prompt:

- Maximum new tokens = 200 allows up to 200 generated tokens.
- Maximum total length = 900 leaves only:

$$
900-800=100
$$

positions for generated tokens, assuming no additional reserved positions.

Library parameter names and special-token accounting differ, so the exact API semantics should be checked.

## Q19. How does output length affect inference latency?

**Interview answer:** Autoregressive decode is sequential across output positions. If token $j$ takes time $t_j$:

$$
T_{\mathrm{decode}}
=
\sum_{j=1}^{N}t_j
$$

With approximately constant average decode latency $\bar{t}$:

$$
T_{\mathrm{decode}}
\approx
N\bar{t}
$$

So generating more output tokens generally increases total decode latency. Actual token latency can vary with context length, hardware, batching, memory bandwidth, and other serving details.

## Q20. How does maximum output length affect KV-cache memory?

**Interview answer:** During normal KV-cached decoding, each additional generated position can add Keys and Values to the cache.

A simplified estimate is:

$$
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{KV}}d_h b
$$

where:

- $L$ is number of layers.
- $B$ is number of active sequence histories.
- $T$ is cached sequence length.
- $H_{\mathrm{KV}}$ is number of KV heads.
- $d_h$ is head dimension.
- $b$ is bytes per cached element.

A larger possible output increases the possible value of $T$, although real memory behavior depends on cache allocation, prefix sharing, sliding windows, and other serving optimizations.

## Q21. Why does maximum output length matter in batched serving?

**Interview answer:** Requests in the same batch finish at different times. A short response may terminate after 20 tokens while another runs for 200 tokens.

Completed requests can be removed from active decoding, allowing the scheduler to reclaim compute and KV-cache capacity. Output-length distributions therefore influence throughput, latency, active batch size, and resource planning.

## Q22. What is the difference between a stop finish reason and a length finish reason?

**Interview answer:** A stop-related finish reason indicates that a recognized ending condition occurred, such as EOS or a configured stop marker. A length-related finish reason indicates that the generation budget was exhausted.

A response that *looks complete* can still have a length finish reason if the cap happened to be reached immediately after a complete-looking sentence. Applications should rely on the returned finish metadata rather than only the visible text.

## Q23. Can visible output length differ from reported output-token usage?

**Interview answer:** Yes. Token accounting depends on the provider. Special tokens may be hidden, and some systems may count non-visible output categories against a broader generation budget.

Therefore, a short visible answer does not necessarily imply an equally small reported output-token count.

---

# Practical Interview Questions

## Q24. Walk me through EOS, custom-stop, and maximum-length termination using the same token budget.

**Interview answer:** Suppose the output budget is eight counted tokens.

**Run A:** The model generates four normal tokens and selects EOS as token 5. Generation finishes naturally.

**Run B:** A custom multi-token stop marker completes at token 7. The runtime stops because the configured pattern matched.

**Run C:** No stop condition appears, and token 8 is an ordinary token. The runtime stops because the output budget is exhausted.

The important distinction is that **A and B end because a stop condition occurs; C ends because a resource limit is reached**.

## Q25. What should an application do when generation ends because of a length limit?

**Interview answer:** Treat the response as potentially incomplete. For ordinary prose, the application may request a continuation or increase the budget if the model and context limits permit it. For structured output, it should validate the result instead of assuming that partially generated JSON or another schema is complete.

If more output cannot fit, the application may need to reduce input context, summarize earlier content, use a longer-context model, or split the task.

---

# Complete Spoken Answers

## Stop Tokens — concise

Stop tokens are special tokens or configured patterns that tell the inference system when a generated sequence should end. An EOS token is typically a tokenizer-defined special token that the model can predict as part of its next-token distribution. Custom stop sequences are usually enforced by the serving runtime and may span several tokenizer tokens. Once a recognized stop condition occurs, generation terminates for that sequence, usually before the maximum output-token budget is exhausted.

## Maximum Output Length — concise

Maximum output length is the upper bound on how many new tokens a request may generate. It is separate from the context window, which limits the total tokenized context the model can process. If EOS or another stop condition occurs first, generation ends naturally. If the output budget is exhausted first, generation terminates mechanically and the answer may be truncated. Larger output allowances also increase worst-case decode latency and potential KV-cache usage.

## Combined Answer

Stop criteria and maximum output length provide complementary termination mechanisms. Stop tokens and custom stop sequences let generation end when an appropriate ending event occurs, while maximum output length provides a hard upper bound when no earlier stop occurs. In production, I would configure both, inspect the API's finish reason, and ensure the requested output budget fits the model's context and output limits.

---

# Quick Follow-up Answers

**Does EOS always have the highest probability near the end?** No. It is just another eligible token score until the decoding strategy selects it.

**Can EOS be excluded by top-k or top-p?** Yes, if it does not survive the corresponding filter.

**Can a model continue after saying a complete sentence?** Yes. Natural-language completeness is not the same as a runtime stop condition.

**Does one beam reaching EOS terminate all beams?** No. It completes that beam; global beam-search stopping is separate.

**Can maximum output length stop text in the middle of JSON?** Yes.

**Does a larger output limit force a longer answer?** No.

**Is context window the same as maximum output tokens?** No.

**Can a custom stop marker span several tokenizer tokens?** Yes.

**Should a length finish reason be treated as potentially incomplete?** Yes.

**Can visible text length and reported token usage differ?** Yes, depending on provider accounting.

---

# Key Memory Lines

- **EOS:** A model/tokenizer-defined special ending token.
- **Custom stop sequence:** A runtime-detected application pattern.
- **EOS in beam search:** Completes a beam, not necessarily the whole search.
- **Maximum output length:** A hard ceiling on generated tokens.
- **Context window:** Capacity for prompt plus generated context, subject to system-specific rules.
- **Stop vs length:** Natural/configured ending versus exhausted generation budget.
- **Latency:** More generated tokens generally mean more decode iterations.
- **KV cache:** More generated positions can increase cached state.
- **Finish reason:** Use runtime metadata to distinguish natural stopping from truncation.

**Next:** Part 8 — 8.14 Why Generation Is Sequential and 8.15 Why Inference Is Expensive.
