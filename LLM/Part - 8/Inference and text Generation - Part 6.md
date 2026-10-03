# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 6

## 8.10 Beam Search
## 8.11 Repetition Penalty

**Detailed Study Mode.** Interview answers will be prepared separately.

This part continues the sequence from greedy decoding, temperature, top-k and top-p. Those techniques determine how to select the *next token*. Two remaining questions motivate this chapter:

1. Can we retain several **partial sequences** instead of committing to one next token?
2. What if the model repeatedly generates the same token or phrase?

Beam search addresses the first problem by searching over competing continuations. Repetition penalties address the second by modifying next-token scores or blocking repeated patterns.

---

# 8.10 Beam Search

## Question 1 — Why do we need beam search if we already have greedy decoding?

Greedy decoding chooses the highest-probability next token:

$$
y_t=\underset{i}{\operatorname{argmax}}\;P(i\mid x,y_{<t})
$$

But a full sequence has probability:

$$
P(y_1,\ldots,y_T\mid x)
=
\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
$$

The locally most probable next token need not lead to the most probable complete sequence.

Greedy decoding discards every alternative immediately. If an initially less probable token would lead to a stronger future continuation, greedy cannot recover it.

**Beam search keeps several promising partial continuations alive at the same time.**

## Question 2 — What exactly is a beam?

A beam is a candidate **partial output sequence**, together with its accumulated score and the model state needed to continue it.

Let beam width be $w$.

~~~text
w = 1 → retain only one active candidate
w = 2 → retain up to two active candidates
w = 4 → retain up to four active candidates
~~~

Beam width does not mean the number of tokens retained in a single next-token distribution. That is a top-k sampling concept. Beam width counts competing **sequence hypotheses**.

With $w=1$, a basic unmodified beam search behaves like greedy decoding, assuming equivalent scoring, stopping and tie rules.

## Question 3 — How do we calculate the score of a partial sequence?

Suppose we have generated:

$$
y_{1:t}=(y_1,\ldots,y_t)
$$

The probability of that partial sequence is:

$$
P(y_{1:t}\mid x)
=
\prod_{j=1}^{t}P(y_j\mid x,y_{<j})
$$

Instead of multiplying many tiny numbers, beam search usually accumulates their log probabilities:

$$
S(y_{1:t})
=
\log P(y_{1:t}\mid x)
=
\sum_{j=1}^{t}\log P(y_j\mid x,y_{<j})
$$

When appending a new token $v$:

$$
S(y_{1:t}\mathbin{\|}v)
=
S(y_{1:t})
+
\log P(v\mid x,y_{1:t})
$$

The concatenation symbol $\mathbin{\|}$ means "append token $v$".

**Why is this mathematics needed?** Every hypothesis carries a cumulative sequence score. Each extension requires only one new conditional log probability added to the parent's existing score.

## Question 4 — What operations happen in one beam-search step?

For each active beam:

1. Obtain the next-token probability distribution.
2. Generate candidate one-token extensions.
3. Add each extension's log probability to its parent's cumulative score.
4. Rank extensions **globally across every parent**, not separately inside each parent.
5. Keep up to $w$ active candidates under the search's scoring and stopping rules.
6. Handle completed EOS-ending hypotheses separately, depending on implementation.

Two surviving beams can come from the same parent if its two highest-scoring extensions outperform all other candidates.

## Question 5 — Can you show a complete numerical example?

Set beam width:

$$
w=2
$$

### Step 1: First-token distribution

Suppose:

$$
P(A\mid x)=0.60,\qquad P(B\mid x)=0.40
$$

Greedy would select A and discard B. Beam search keeps both:

| Active hypothesis | Probability | Approximate log score |
|---|---:|---:|
| A | 0.60 | -0.511 |
| B | 0.40 | -0.916 |

### Step 2: Expand both hypotheses

Suppose the model predicts:

| Parent | Child | Conditional probability |
|---|---|---:|
| A | X | 0.55 |
| A | Y | 0.45 |
| B | X | 0.90 |
| B | Y | 0.10 |

We calculate **four sequence probabilities**:

$$
P(A,X\mid x)=0.60\times0.55=0.33
$$

$$
P(A,Y\mid x)=0.60\times0.45=0.27
$$

$$
P(B,X\mid x)=0.40\times0.90=0.36
$$

$$
P(B,Y\mid x)=0.40\times0.10=0.04
$$

Now rank all four:

| Rank | Candidate sequence | Probability | Approx. log score |
|---|---|---:|---:|
| 1 | B → X | 0.36 | -1.022 |
| 2 | A → X | 0.33 | -1.109 |
| 3 | A → Y | 0.27 | -1.309 |
| 4 | B → Y | 0.04 | -3.219 |

Since $w=2$, retain:

$$
\{B\rightarrow X,\ A\rightarrow X\}
$$

**What did we learn?** Greedy's local decision A would produce A → X with probability 0.33. Beam search preserved B and discovered B → X, with probability 0.36.

The initially lower-probability first token can lead to the higher-probability two-token path.

## Question 6 — Does the highest-scoring beam at one step always win at the end?

No. Continue the previous example. Our two active hypotheses are B → X (0.36) and A → X (0.33).

Suppose the next-step distributions are:

| Parent | EOS | Token Z |
|---|---:|---:|
| B → X | 0.80 | 0.20 |
| A → X | 0.90 | 0.10 |

The probabilities of completed hypotheses are:

$$
P(B,X,\mathrm{EOS}\mid x)=0.36\times0.80=0.288
$$

$$
P(A,X,\mathrm{EOS}\mid x)=0.33\times0.90=0.297
$$

The probabilities of continuing with Z are:

$$
P(B,X,Z\mid x)=0.36\times0.20=0.072
$$

$$
P(A,X,Z\mid x)=0.33\times0.10=0.033
$$

Even though B → X was ahead before this step, A → X → EOS has the highest score among the illustrated extensions.

**A beam's relative ranking can change as future tokens are generated.**

The EOS probability belongs in the probability of a completed sequence.

## Question 7 — Does beam search guarantee the globally optimal completed output?

Not in general.

Beam search prunes candidate prefixes at every depth. A discarded prefix may have had an excellent continuation, but the search will never explore it.

Larger width retains more alternatives and generally uses more compute and memory. It does **not** guarantee better human-perceived output, perfect factuality or greater diversity. Exact global optimization would require exhaustive search of the relevant complete-sequence space or another method with a provable guarantee.

## Question 8 — What happens when a beam generates EOS?

EOS is a special end-of-sequence token. A beam emitting it becomes a **completed hypothesis**.

A typical implementation stores completed hypotheses and avoids expanding them again. It continues expanding unfinished beams until a stopping condition is satisfied, such as enough suitable completed hypotheses, an early-stopping criterion or a maximum generation length.

Different libraries differ in how completed and unfinished hypotheses compete and when early stopping is safe.

**Important:** A high-scoring unfinished prefix is not necessarily a valid final answer. The completed sequence typically includes its EOS probability in the cumulative score.

## Question 9 — Why can beam search prefer short completed sequences?

Every conditional probability is at most one:

$$
0\le P(y_t\mid x,y_{<t})\le1
$$

For positive probabilities, its logarithm is non-positive:

$$
\log P(y_t\mid x,y_{<t})\le0
$$

Adding tokens therefore generally makes the raw cumulative log score more negative:

$$
S(y_{1:t+1})
=
S(y_{1:t})+\log P(y_{t+1}\mid x,y_{1:t})
\le S(y_{1:t})
$$

When comparing completed outputs of different lengths, this creates a tendency for raw sequence likelihood to favor shorter possibilities, depending on EOS probabilities and stopping rules.

## Question 10 — What is length normalization?

Length normalization modifies sequence scores to address differences in sequence length.

One simple conceptual score is average log probability:

$$
S_{\mathrm{avg}}(y_{1:T})
=
\frac{1}{T}\sum_{t=1}^{T}\log P(y_t\mid x,y_{<t})
$$

Another commonly used family of scores is:

$$
S_{\mathrm{len}}(y_{1:T})
=
\frac{\log P(y_{1:T}\mid x)}
{\left(\frac{5+T}{6}\right)^{\alpha}}
$$

Here $\alpha$ controls the penalty's strength. With $\alpha=0$, the denominator is one and the score is the raw log probability.

**Why do we need it?** Comparing raw probabilities across outputs of different lengths can systematically disadvantage longer completed outputs because they contain more probability factors.

There is no single universal length-penalty formula. Check how a particular inference library counts EOS and applies normalization during pruning and final ranking.

## Question 11 — How expensive is beam search?

With beam width $w$ and eligible vocabulary size $V$, a basic step may consider up to:

$$
w\times V
$$

one-token candidate extensions before pruning.

The model also evaluates up to $w$ active sequence continuations at each step, often using batching.

A wider beam usually increases compute and memory demand, though precise latency depends on batching, attention implementation, search kernels and the number of completed beams.

## Question 12 — How does beam search interact with KV cache?

Each active beam has its own token history. Beams sharing an identical prefix may share corresponding cached Keys and Values, while divergent continuations need their own subsequent KV state.

~~~text
Shared prompt KV state
         ↓
      A     B
      ↓     ↓
     A-X   B-X
~~~

When candidate beams are reordered or discarded, the serving system must reorder or release the corresponding continuation KV state correctly.

More active beams can require more KV-cache memory, especially once their histories diverge.

## Question 13 — Beam search versus top-k sampling: what is the difference?

| Aspect | Beam search | Top-k sampling |
|---|---|---|
| Retains | Multiple candidate **sequences** | Highest-ranked $k$ **tokens** for the current position |
| Scores | Accumulated sequence scores | Current next-token probabilities |
| Basic choice | Rank and prune | Randomly sample from a filtered distribution |
| Typical goal | Search competing continuations | Control current-step sampling candidates |
| Basic randomness | None with fixed scores and tie rules | Present when multiple candidates survive |

Beam search can still become repetitive or produce generic high-likelihood text. It optimizes its configured score, not necessarily creativity or factual accuracy.

### Beam-search mental model

> **Greedy keeps one path. Beam search keeps several paths and repeatedly compares their accumulated scores.**

---

# 8.11 Repetition Penalty

## Question 1 — Why does autoregressive generation sometimes repeat itself?

The model predicts one token at a time using its current context. Sometimes a previously generated token remains a high-probability continuation even after it has just been selected.

For example:

~~~text
The result is very very very very ...
~~~

This can arise under greedy, sampling or beam-based decoding. Repetition is not always wrong: code, quotations, names, headings and grammatical structure may legitimately reuse tokens.

The goal is to discourage **unwanted repetition**, not to forbid all repetition.

## Question 2 — What is a repetition penalty?

A repetition penalty is a decoding-time adjustment that changes the scores of tokens that already occurred in a configured history.

Conceptually:

~~~text
Model produces next-token logits
         ↓
Identify tokens seen in the relevant history
         ↓
Modify their scores
         ↓
Apply subsequent decoding controls
         ↓
Select the next token
~~~

Unlike temperature, which scales every token's logits, repetition penalties selectively affect previously seen tokens.

**Important:** There is no universal repetition-penalty formula. Libraries may implement different algorithms and use different history scopes.

## Question 3 — What is a common sign-aware multiplicative repetition penalty?

One widely used convention applies a factor $r>1$ to the logit of a token that has previously appeared:

$$
z_i'=
\begin{cases}
\displaystyle\frac{z_i}{r}, & z_i>0\\[8pt]
z_i r, & z_i<0\\[4pt]
0, & z_i=0
\end{cases}
$$

Tokens not seen in the relevant history retain their original logits:

$$
z_j'=z_j
$$

With $r=1$, no change occurs.

### Why are positive and negative logits treated differently?

Consider a repeated token with negative logit:

$$
z_i=-2
$$

Dividing by $1.5$ would produce:

$$
\frac{-2}{1.5}\approx-1.33
$$

But:

$$
-1.33>-2
$$

This **increases** the logit, making the repeated token more attractive. The sign-aware rule instead multiplies a negative logit:

$$
-2\times1.5=-3
$$

That makes the repeated token less attractive relative to competitors.

## Question 4 — Can we work through the repetition penalty with actual numbers?

Suppose the model's unmodified next-token logits are:

| Token | Appeared earlier? | Original logit |
|---|---|---:|
| A | Yes | 3.0 |
| B | No | 2.5 |
| C | Yes | -2.0 |

Apply:

$$
r=1.5
$$

For repeated A:

$$
z_A'=\frac{3.0}{1.5}=2.0
$$

For unseen B:

$$
z_B'=2.5
$$

For repeated C:

$$
z_C'=-2.0\times1.5=-3.0
$$

The modified logits are:

| Token | Before | After |
|---|---:|---:|
| A | 3.0 | 2.0 |
| B | 2.5 | 2.5 |
| C | -2.0 | -3.0 |

Before applying the penalty, greedy decoding would choose A. After applying it, B has the highest logit.

**The penalty changes the probability distribution for the next token. It does not alter or delete previously generated tokens.**

## Question 5 — Does this penalty care whether a token occurred once or five times?

Not in the basic sign-aware multiplicative form described above. It typically depends on **whether** the token is in the configured history, rather than applying an additional multiplier for every occurrence.

Different systems may include the prompt and generated tokens, only generated tokens, or only a recent history window.

This distinction matters: penalizing a token from the prompt can interfere with answers that need to repeat exact terminology.

## Question 6 — How are presence and frequency penalties different?

Let $c_i$ denote the number of times token $i$ appeared in the relevant history.

A simple presence penalty subtracts a fixed amount if the token appeared at least once:

$$
z_i'=z_i-\lambda_{\mathrm{presence}}\mathbf{1}[c_i>0]
$$

A frequency penalty subtracts an amount proportional to its count:

$$
z_i'=z_i-\lambda_{\mathrm{frequency}}c_i
$$

For a token that occurred three times and penalty coefficient 0.5:

- Presence adjustment: 0.5
- Frequency adjustment: 1.5

Presence asks **has this token appeared?** Frequency asks **how many times has this token appeared?**

These are representative formulas; actual API behavior can vary.

## Question 7 — What is no-repeat n-gram blocking?

An $n$-gram is a contiguous sequence of $n$ tokens.

No-repeat n-gram blocking prevents an $n$-gram that already appeared in the configured history from appearing again.

For a forbidden next-token completion, a hard constraint may assign:

$$
z_i'=-\infty
$$

After softmax, that token has zero probability.

Compare the two techniques:

~~~text
Soft repetition penalty
→ decreases the score of a repeated token
→ token can still be selected

Hard no-repeat n-gram blocking
→ masks a token that would recreate an n-gram
→ forbidden completion has zero probability
~~~

Matching usually happens using token IDs, not semantic meaning. A paraphrased repeated phrase may evade exact n-gram blocking.

## Question 8 — Does repetition penalty change the ranking of tokens?

Yes. Positive temperature preserves logit ranking, but a repetition penalty modifies only certain token logits.

In our example:

~~~text
Before:
A = 3.0  ← repeated
B = 2.5
Greedy chooses A

After repetition penalty:
A = 2.0
B = 2.5
Greedy chooses B
~~~

It can therefore alter greedy choices, top-k membership, top-p nucleus membership, and beam expansions.

## Question 9 — How does repetition penalty combine with temperature and other decoding rules?

A typical conceptual pipeline is:

~~~text
Model logits
       ↓
Task constraints and repetition adjustments
       ↓
Temperature scaling
       ↓
Top-k / Top-p filtering (if enabled)
       ↓
Softmax and sampling, or beam-search scoring
~~~

The exact order is implementation-specific. Penalty adjustment and temperature scaling do not always commute, so processor order can change the final distribution.

For beam search, every beam has its own history and may receive different repetition adjustments.

## Question 10 — What happens when the penalty is too strong?

Excessive penalties can make normal language generation worse:

- Technical terminology and names may need to recur.
- Code frequently repeats identifiers, punctuation, and syntax tokens.
- Structured outputs repeat delimiters and field labels.
- Answering a question may require reusing words from the prompt.

A stronger penalty is not automatically better. It can push probability mass toward unnatural or incorrect alternatives.

## Question 11 — Does repetition penalty completely eliminate repetitive outputs?

No. A soft penalty can still leave a repeated token as the highest-ranked option. A hard n-gram constraint can block exact repeats but not all semantic repetition or slight paraphrases.

Repetition also depends on prompt, model behavior, decode strategy and stopping conditions.

## Question 12 — How does repetition penalty affect beam-search scoring?

Each active beam has its own sequence history. A token already present in one beam may be penalized differently than the same token in another beam.

When penalties change logits, a beam's search score can differ from the **unmodified model log probability**. Distinguish the model's original likelihood from a score containing decoding heuristics.

### Repetition-penalty mental model

> **Beam search controls which partial sequences survive. A repetition penalty changes the next-token scores inside those sequences.**

---

# Beam Search vs Repetition Penalty — Final Comparison

| Property | Beam Search | Repetition Penalty |
|---|---|---|
| Main question | Which partial continuations should remain alive? | Which previously used tokens or patterns should be discouraged? |
| Operates on | Partial sequence hypotheses | Next-token logits or constraints |
| Main controls | Beam width and sequence scoring | Penalty factor, token counts, history or n-gram size |
| Main tradeoff | More search versus more compute and KV memory | Less repetition versus suppression of legitimate repeats |
| Works with | Configured scores and stopping rules | Greedy, sampling or beam-based decoding |

# Part 6 — Key Equations

**Sequence probability:**

$$
P(y_{1:T}\mid x)=\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
$$

**Accumulated beam score:**

$$
S(y_{1:T})=\sum_{t=1}^{T}\log P(y_t\mid x,y_{<t})
$$

**Add one token to a beam:**

$$
S(y_{1:t}\mathbin{\|}v)
=
S(y_{1:t})+\log P(v\mid x,y_{1:t})
$$

**Example length-adjusted score:**

$$
S_{\mathrm{len}}(y_{1:T})
=
\frac{\log P(y_{1:T}\mid x)}
{\left(\frac{5+T}{6}\right)^{\alpha}}
$$

**Example sign-aware multiplicative repetition penalty for an already-seen token:**

$$
z_i'=
\begin{cases}
\displaystyle\frac{z_i}{r}, & z_i>0\\[8pt]
z_i r, & z_i<0\\
0, & z_i=0
\end{cases}
$$

**Simple presence and frequency penalties:**

$$
z_i'=z_i-\lambda_{\mathrm{presence}}\mathbf{1}[c_i>0]
$$

$$
z_i'=z_i-\lambda_{\mathrm{frequency}}c_i
$$

# Bridge to Part 7

Beam search helps manage alternative continuations, while repetition controls help discourage undesirable repetition. The next two questions concern when generation should terminate:

- **8.12 Stop Tokens**
- **8.13 Maximum Output Length**
