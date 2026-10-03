# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 6 — Interview Style

## Topics
- **8.10 Beam Search**
- **8.11 Repetition Penalty**

**Interview preparation only.** For slow, question-led derivations, consult the separate [Part 6 Detailed Study](./Inference%20and%20text%20Generation%20-%20Part%206.md).

---

# 8.10 Beam Search

## Q1. What is beam search, and why do we use it instead of greedy decoding?

**Interview answer:** Beam search is an approximate sequence-search decoding strategy that retains up to $w$ promising partial sequences, called beams, at each generation step. It expands all active beams, ranks their candidate extensions by accumulated sequence score, and preserves the highest-scoring candidates. Unlike greedy decoding, which commits to one locally optimal next token, beam search keeps alternatives that may lead to stronger complete sequences.

It optimizes its configured search score, not guaranteed factual accuracy or human preference.

## Q2. What is beam width?

**Interview answer:** Beam width $w$ is the maximum number of active partial sequence hypotheses retained for further expansion. It counts **sequences**, not vocabulary tokens. A basic unmodified beam with $w=1$ behaves like greedy decoding, assuming equivalent scoring, stopping rules, and tie-breaking.

A larger beam explores more competing continuations but consumes more compute and continuation KV-cache memory.

## Q3. What is the mathematical scoring function used in beam search?

**Interview answer:** We accumulate conditional token log probabilities using the autoregressive chain rule:

$$
P(y_{1:T}\mid x)
=
\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
$$

$$
S(y_{1:T})
=
\sum_{t=1}^{T}
\log P(y_t\mid x,y_{<t})
$$

When extending a partial hypothesis with token $v$:

$$
S(y_{1:t}\mathbin{\|}v)
=
S(y_{1:t})
+
\log P(v\mid x,y_{1:t})
$$

Here $x$ is the prompt, $y_{1:t}$ is the existing generated continuation, and the concatenation symbol means "append".

Log scores turn a product of many tiny probabilities into a numerically stable sum.

## Q4. Describe one complete beam-search iteration.

**Interview answer:** First, compute a next-token distribution separately for every active beam, often in one batched forward pass. For each beam and each eligible vocabulary token, add that token's conditional log probability to the parent's cumulative score. Compare these **hypothetical extensions globally across all parents**, keep the top $w$ active extensions, and append the winning tokens to their corresponding parent histories. EOS-ending candidates are managed as completed hypotheses under the implementation's finishing rules. Repeat with unfinished survivors.

**Key point:** We do not keep a fixed number of children from every parent. Both winning extensions may come from the same parent.

## Q5. If each beam predicts the entire vocabulary, how can we choose the next token without first appending it?

**Interview answer:** We do **not** append tokens first. A forward pass already gives each active beam the conditional probabilities of all eligible next tokens. For parent beam $b$ and candidate token $v$, score the *hypothetical* extension:

$$
C_{b,v}
=
S_b+\log P(v\mid B_b)
$$

Here $B_b$ is beam $b$'s complete history, and $S_b$ is its accumulated log score.

With $w=2$ and vocabulary size $V=50{,}000$, the candidate-score matrix has shape:

$$
C\in\mathbb{R}^{2\times50{,}000}
$$

That represents 100,000 hypothetical beam–token pairs. A global top-2 operation finds the winning pairs. **Only those tokens are actually appended.** This does not require 100,000 Transformer forward passes: the current pass already produced their one-step probabilities.

## Q6. Does every beam require a separate full-vocabulary distribution? Can beams share a forward pass?

**Interview answer:** Each beam has its **own conditional distribution** because its token history may differ. The model weights are shared, and the computations can be batched into one forward pass. With beam width $w$ and vocabulary size $V$, the batched logits have shape:

$$
Z\in\mathbb{R}^{w\times V}
$$

Softmax is applied **independently across each beam's vocabulary row**:

$$
P_{b,v}
=
\frac{\exp(Z_{b,v})}
{\sum_{j=1}^{V}\exp(Z_{b,j})}
$$

Each row sums to one; beams do not share a single probability distribution.

Immediately after prefill, before beams diverge, the prompt has just **one** first-token distribution. Once different first tokens create distinct histories, later decode steps generally require distinct distributions, even if processed in one batch.

## Q7. Walk through a numerical beam-search example with width two.

**Interview answer:** Suppose the first-token probabilities are:

$$
P(A\mid x)=0.60,\qquad P(B\mid x)=0.40
$$

With $w=2$, retain A and B. Their next-token probabilities are:

| Parent | Next token X | Next token Y |
|---|---:|---:|
| A | 0.55 | 0.45 |
| B | 0.90 | 0.10 |

Multiply each parent's cumulative probability by the conditional next-token probability:

| Candidate | Cumulative probability |
|---|---:|
| B → X | $0.40\times0.90=0.36$ |
| A → X | $0.60\times0.55=0.33$ |
| A → Y | $0.60\times0.45=0.27$ |
| B → Y | $0.40\times0.10=0.04$ |

Global top-2 selection retains **B → X and A → X**.

Greedy would have committed to A initially and selected A → X with cumulative probability 0.33. Beam search preserved B and found a stronger two-token continuation with probability 0.36.

## Q8. Can both surviving beams come from the same parent?

**Interview answer:** Yes. Suppose $P(A)=0.60$, $P(B)=0.40$, and:

| Parent | $P(X)$ | $P(Y)$ |
|---|---:|---:|
| A | 0.60 | 0.40 |
| B | 0.50 | 0.50 |

The extension scores are proportional to:

$$
P(A,X)=0.60\times0.60=0.36
$$

$$
P(A,Y)=0.60\times0.40=0.24
$$

$$
P(B,X)=P(B,Y)=0.40\times0.50=0.20
$$

Global top-2 keeps **A → X and A → Y**, eliminating B entirely. Ordinary beam search allocates survivor slots globally, not one slot per parent.

## Q9. Does the highest-scoring partial beam necessarily become the final winner?

**Interview answer:** No. A current leader can receive weaker continuation probabilities than a competing beam, changing their eventual completed-sequence rankings. For example, if active beams B → X and A → X have probabilities 0.36 and 0.33 respectively, but EOS probabilities of 0.80 and 0.90, their completed-sequence probabilities are:

$$
P(B,X,\mathrm{EOS}\mid x)=0.36\times0.80=0.288
$$

$$
P(A,X,\mathrm{EOS}\mid x)=0.33\times0.90=0.297
$$

The second sequence is now ahead despite its lower score at the previous step.

## Q10. Does beam search guarantee the globally most probable complete output?

**Interview answer:** Not generally. Beam search is approximate because it prunes candidates at each step. A discarded prefix cannot return even if its future continuation would have been strong. Larger beam width explores more paths but is not a guarantee of globally optimal completion, better human-perceived quality, factual accuracy, or diversity.

## Q11. What happens when a hypothesis generates EOS?

**Interview answer:** EOS marks a completed sequence. Most implementations collect completed hypotheses and stop expanding them, while continuing to expand unfinished active beams. The EOS token's conditional probability usually contributes to the completed sequence's score. Stopping may depend on the number and ranking of completed hypotheses, early-stopping bounds, minimum or maximum length, and length-normalization rules.

A high-scoring unfinished prefix should not automatically be treated as the best completed answer.

## Q12. Why can raw beam-search scores favor shorter completed sequences?

**Interview answer:** An additional token contributes a conditional probability between zero and one. Therefore, extending the *same prefix* can never increase its raw probability:

$$
P(y_{1:t+1}\mid x)
=
P(y_{1:t}\mid x)
P(y_{t+1}\mid x,y_{1:t})
\le P(y_{1:t}\mid x)
$$

In log space:

$$
S(y_{1:t+1})
=
S(y_{1:t})
+
\log P(y_{t+1}\mid x,y_{1:t})
\le S(y_{1:t})
$$

Because longer sequences accumulate more non-positive log-probability terms, raw scoring can favor short completed hypotheses. **This is a length bias, not a claim that every shorter sequence scores above every longer sequence.** EOS probabilities and stopping rules matter.

## Q13. Can you numerically show the short-sequence bias?

**Interview answer:** Consider two hypothetical completed outputs. The probabilities listed are conditional on each sequence's own history.

| Completed sequence | Conditional probabilities | Raw sequence probability |
|---|---|---:|
| travel EOS | $0.60\times0.40$ | 0.240 |
| travel abroad tomorrow EOS | $0.60\times0.50\times0.80\times0.90$ | 0.216 |

Their cumulative log scores are:

$$
S_{\mathrm{short}}=\log(0.240)\approx-1.427
$$

$$
S_{\mathrm{long}}=\log(0.216)\approx-1.532
$$

With raw log scores, the shorter sequence ranks higher because $-1.427>-1.532$.

Notice that the probability of the second token is conditioned on each candidate's actual prior context. The comparison is between two completed sequences, not two lengths of an identical continuation.

## Q14. What is length normalization?

**Interview answer:** Length normalization adjusts cumulative sequence scores to reduce the tendency to penalize longer completed outputs merely for containing more probability factors. A simple example is average log probability:

$$
S_{\mathrm{avg}}(y_{1:T})
=
\frac{1}{T}
\sum_{t=1}^{T}
\log P(y_t\mid x,y_{<t})
$$

For the previous example, if EOS counts as a token:

| Sequence | Raw log score | Length | Average log score |
|---|---:|---:|---:|
| travel EOS | -1.427 | 2 | -0.714 |
| travel abroad tomorrow EOS | -1.532 | 4 | -0.383 |

Average log scoring reverses their ordering: $-0.383>-0.714$.

Another frequently used family is:

$$
S_{\mathrm{len}}(y_{1:T})
=
\frac{\log P(y_{1:T}\mid x)}
{\left(\frac{5+T}{6}\right)^\alpha}
$$

The length-penalty parameter $\alpha$ controls its strength; $\alpha=0$ recovers the raw log score. Exact conventions, including whether EOS counts toward length, depend on the library.

## Q15. What is the computational and memory impact of beam width?

**Interview answer:** With $w$ active beams and vocabulary size $V$, a naïve expansion has approximately:

$$
w\times V
$$

candidate beam–token scores per step. The model also produces up to $w$ next-token distributions, often batched. Larger beams generally increase compute and continuation KV-cache memory, though implementations can use efficient top-selection and prefix-state sharing.

The $wV$ expression counts **candidate extension scores**, not $wV$ Transformer forward passes.

## Q16. How does beam search use the KV cache?

**Interview answer:** Every beam represents a token history. Beams with a shared prompt or identical prefix can reuse common prefix KV state. Once they diverge, they need continuation-specific KV state. When pruning or reordering beams, the serving system must reorder or reindex their cache references consistently with the selected parents.

## Q17. How is beam search different from top-k sampling?

| Property | Beam search | Top-k sampling |
|---|---|---|
| Retained objects | Partial **sequences** | Top-ranked **next-token candidates** |
| Main score | Accumulated sequence score | Current next-token probabilities |
| Basic selection | Global ranking and pruning | Renormalize and randomly sample |
| Main parameter | Beam width $w$ | Candidate count $k$ |
| Typical basic output | Deterministic under fixed scores and tie rules | Stochastic when multiple candidates survive |

**Interview emphasis:** Beam search explores competing future *paths*. Top-k restricts a single next-token sampling distribution.

## Q18. When is beam search a poor fit?

**Interview answer:** It can be expensive for long open-ended generation and may favor generic high-likelihood phrasing. It is not inherently a diversity method or a factuality verifier. Its usefulness depends on the generation task, model, length scoring, stopping rules, and whether multiple competing high-scoring sequences are valuable.

---

# 8.11 Repetition Penalty

## Q19. Why do autoregressive models sometimes generate repetitive text?

**Interview answer:** Each generated token is appended to the context and conditions the next prediction. Previously generated tokens or phrases can remain highly probable, leading to repeated continuations under greedy decoding, sampling, or beam search. Repetition control modifies next-token scores or eligibility to discourage undesirable loops, while allowing necessary repetition where possible.

## Q20. What is a repetition penalty?

**Interview answer:** A repetition penalty is a decoding-time adjustment to the logits or eligibility of tokens that appeared in a configured history. Depending on the implementation, the history can contain the prompt, generated tokens, or a recent token window. Its purpose is to discourage repeated choices, not remove already generated text.

## Q21. What is a common sign-aware multiplicative repetition-penalty formula?

**Interview answer:** For a previously seen token $i$ with logit $z_i$, a common convention uses a factor $r>1$:

$$
z_i'=
\begin{cases}
\displaystyle\frac{z_i}{r}, & z_i>0\\[8pt]
z_i r, & z_i<0\\[4pt]
0, & z_i=0
\end{cases}
$$

An unseen token remains unchanged. When $r=1$, the transformation is neutral.

The rule is sign-aware because dividing a negative logit by a number greater than one would make it *less negative*, accidentally boosting the repeated token.

## Q22. Show a numerical repetition-penalty example.

**Interview answer:** Suppose the logits are A = 3.0, B = 2.5, C = -2.0. A and C previously appeared; B did not. Let $r=1.5$.

| Token | Seen before? | Before | After |
|---|---|---:|---:|
| A | Yes | 3.0 | $3.0/1.5=2.0$ |
| B | No | 2.5 | 2.5 |
| C | Yes | -2.0 | $-2.0\times1.5=-3.0$ |

Greedy decoding chooses A before the penalty but B afterward. The penalty changes the **relative next-token scores** and can therefore affect any decoding method that consumes those scores.

## Q23. Why not simply divide every repeated token's logit by the penalty factor?

**Interview answer:** For positive logits, division by $r>1$ lowers the score. For negative logits, it raises the score toward zero. For example:

$$
\frac{-2}{1.5}\approx-1.33>-2
$$

That would unintentionally *reward* the repeated token. The sign-aware rule instead multiplies negative logits:

$$
-2\times1.5=-3
$$

## Q24. How is a repetition penalty different from presence and frequency penalties?

**Interview answer:** The simple multiplicative repetition penalty typically depends on whether a token has appeared at all, not how many times. A presence penalty subtracts a fixed amount for a previously seen token. A frequency penalty subtracts an amount proportional to its occurrence count.

If $c_i$ is token $i$'s historical count:

$$
z_i'=z_i-\lambda_{\mathrm{presence}}\mathbf{1}[c_i>0]
$$

$$
z_i'=z_i-\lambda_{\mathrm{frequency}}c_i
$$

If $c_i=3$ and the coefficient is $0.5$, the presence adjustment is 0.5 while the frequency adjustment is 1.5.

These are representative conventions; actual serving APIs can differ.

## Q25. What is no-repeat n-gram blocking?

**Interview answer:** It is a hard decoding constraint that blocks a next token if adding it would recreate an already occurring contiguous $n$-token sequence. A forbidden extension may have its logit set to:

$$
z_i'=-\infty
$$

so its probability becomes zero after softmax. Unlike a soft repetition penalty, this rule makes the exact repeated pattern impossible under the applied constraint.

The check generally uses token IDs, not semantic equivalence or word-level understanding.

## Q26. How do repetition penalties interact with temperature, top-k, top-p, and beam search?

**Interview answer:** Repetition adjustments change selected token logits before subsequent token selection. This can change logit ranking and therefore change greedy outcomes, top-k membership, top-p cumulative membership, and beam-search candidate scores.

A common conceptual order is logit constraints and penalties, then temperature, then any sampling filter, then final selection, but **actual processor order is library-specific**. For beam search, penalties depend on each beam's own history, and modified beam scores need not equal the raw model's unmodified sequence log probabilities.

## Q27. Can a repetition penalty completely eliminate repetition?

**Interview answer:** No. A soft penalty only reduces some token scores; the repeated token may still win. A hard no-repeat n-gram constraint blocks exact token patterns but not semantic paraphrases or slightly modified repetitions. Repetition also depends on the model, prompt, decoding method, output length, and stop conditions.

## Q28. What happens if the repetition penalty is too strong?

**Interview answer:** Legitimate repetition may be suppressed: recurring names, technical terms, code identifiers, punctuation, structured-output keys, and phrases copied from the user's prompt. Excessive penalties can force awkward or incorrect alternatives. The goal is to discourage unwanted loops rather than forbid necessary repetition.

## Q29. Does repetition penalty operate on prompt tokens or only generated tokens?

**Interview answer:** That depends on the implementation and the configured history scope. Some apply it to prompt and generated tokens, while others use only generated output or a recent window. Penalizing prompt tokens can be undesirable when a response must reuse terminology or quote the input.

## Q30. How can repetition penalties change beam-search scoring?

**Interview answer:** Each beam can have a different token history, so the same token may receive different penalties across beams. After modification, the next-token scores used during search may differ from the unmodified model's conditional log probabilities. When discussing likelihood, distinguish **raw model likelihood** from **decoding scores modified by penalties or constraints**.

---

# Complete Spoken Answers

## Beam search — concise

Beam search is an approximate sequence decoding algorithm that keeps multiple partial continuations rather than committing to one token at every step. At each iteration, it obtains a next-token distribution for every active beam, adds candidate token log probabilities to their parents' cumulative scores, and selects the highest-scoring extensions *globally*. The winners become the next active beams. It handles EOS-ending hypotheses separately and may use length normalization to reduce short-sequence bias. Larger beam width explores more continuations at increased computational and KV-cache cost.

## Beam search — if the interviewer asks about batching

Each beam needs its own next-token distribution because its token history is different. However, multiple beams can share model weights and run in one batched forward pass. With beam width $w$ and vocabulary size $V$, the logits have shape $w\times V$. Each row receives its own vocabulary softmax. Broadcasting each parent's cumulative log score over its row produces a $w\times V$ candidate-score matrix, from which we choose the global top-$w$ beam–token pairs before appending the winning tokens.

## Repetition penalty — concise

A repetition penalty adjusts next-token scores for tokens that appeared previously, reducing the chance of undesirable repeated output. One common multiplicative version divides positive logits of repeated tokens by a factor above one and multiplies their negative logits by that factor so the scores decrease in both cases. Presence penalties depend on whether a token appeared, frequency penalties depend on how often it appeared, and no-repeat n-gram blocking can forbid exact repeated token patterns. Strong penalties risk suppressing legitimate repetition.

## Combined answer

Beam search and repetition penalties solve different problems. Beam search maintains multiple competing partial sequences and ranks extensions by accumulated score. A repetition penalty modifies a beam's next-token scores based on that beam's own history, making certain repeated continuations less attractive. They can be combined, but the resulting search score may not match the unmodified model likelihood.

---

# Quick Follow-ups

**Does one batched forward pass mean all beams share one probability distribution?** No. It produces one distribution per active beam row.

**Do we append a proposed token before knowing its candidate score?** No. Score hypothetical beam–token pairs first and append only global winners.

**Is one Transformer forward pass needed per vocabulary candidate?** No. One next-token distribution already contains all candidates' one-step probabilities for its beam.

**Does each parent get one surviving child?** Not necessarily. Multiple winners may share the same parent.

**Is beam width the same as top-k?** No. Beam width counts partial sequences; top-k counts retained tokens at the current position.

**Does extending a particular prefix ever increase raw probability?** No; the appended conditional probability is at most one.

**Does the shortest completed sequence always win?** No; raw sequence scoring creates a length bias, not an absolute length-based ordering.

**Is a higher beam width guaranteed to improve output quality?** No.

**Can a soft repetition penalty ban a token?** Usually not. Hard constraints such as no-repeat n-gram blocking can make specified extensions ineligible.

**Does a basic multiplicative repetition penalty scale with occurrence count?** Typically no. Frequency penalties explicitly use occurrence counts.

**Does penalty factor $r=1$ change logits?** No, for the sign-aware multiplicative formula shown.

**Are repetition penalties guaranteed to improve factual accuracy?** No. They are decoding heuristics.

---

# Key Memory Lines

- **Beam search:** Keep multiple partial sequences, not multiple tokens from one sequence.
- **Full-vocabulary expansion:** Score every eligible beam–token pair before appending survivors.
- **Batching:** Shared model and possibly one forward pass; separate conditional distribution per beam.
- **Scoring:** Sum conditional log probabilities along each path.
- **EOS:** A completed hypothesis is handled under finishing and stopping rules.
- **Length bias:** More generated tokens add more non-positive log-probability terms.
- **Length normalization:** Adjust scores for comparing differently sized completed candidates.
- **Repetition penalty:** Reduce scores of seen tokens; its formula and history scope depend on implementation.
- **Presence vs frequency:** Appeared at all versus number of appearances.
- **No-repeat n-gram:** Hard-block an exact repeated token pattern.

**Next:** Part 7 — 8.12 Stop Tokens and 8.13 Maximum Output Length.
