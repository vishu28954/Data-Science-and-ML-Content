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
y_t = \arg\max_i P(i \vert x, y_{\lt t})
$$


But a full sequence has probability:

$$
P(y_1, \ldots, y_T \vert x) = \prod_{t=1}^{T} P(y_t \vert x, y_{\lt t})
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
P(y_{1:t} \vert x) = \prod_{j=1}^{t} P(y_j \vert x, y_{\lt j})
$$


Instead of multiplying many tiny numbers, beam search usually accumulates their log probabilities:

$$
S(y_{1:t}) = \log P(y_{1:t} \vert x) = \sum_{j=1}^{t} \log P(y_j \vert x, y_{\lt j})
$$


When appending a new token $v$:

$$
S(y_{1:t} \parallel v) = S(y_{1:t}) + \log P(v \vert x, y_{1:t})
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

### Question 4A — If each beam produces a distribution over the full vocabulary, when do we select a token to append?

**We calculate the scores of hypothetical extensions before selecting and appending any token.** Beam search does not choose one next token from each beam first. It evaluates possible beam–token combinations, compares them globally, and only then forms the new active beams.

#### 1. Start with two active beams

Suppose our prompt is "I want to" and beam width is $w=2$.

| Beam | Partial sequence | Cumulative probability |
|---|---|---:|
| 1 | I want to travel | 0.60 |
| 2 | I want to learn | 0.40 |

We store the cumulative **log scores**:

$$
S_1=\log(0.60),\qquad S_2=\log(0.40)
$$

#### 2. Obtain a full-vocabulary distribution for each beam

If vocabulary size is $V=50{,}000$, the two beam-specific probability distributions can be arranged into a matrix:

$$
P\in\mathbb{R}^{2\times50{,}000}
$$

Each row corresponds to one beam and each column to one vocabulary token. Each row sums to one:

$$
\sum_{v=1}^{V}P(v\mid B_b)=1
$$

for each active beam $b$.

**At this point, no next token has been selected or appended.** The forward pass has already produced the probabilities of *all* possible next tokens for every beam.

#### 3. Calculate hypothetical extension scores

For beam $b$ with cumulative score $S_b$ and vocabulary token $v$:

$$
\boxed{
C_{b,v}=S_b+\log P(v\mid B_b)
}
$$

The score matrix has shape:

$$
C\in\mathbb{R}^{2\times50{,}000}
$$

Thus we have up to:

$$
2\times50{,}000=100{,}000
$$

candidate one-token extensions.

Crucially, **calculating these scores does not require 100,000 separate Transformer forward passes**. Every candidate's current conditional probability is already available from the two beam-specific distributions.

#### 4. A fully worked miniature example

To make the mechanism easy to see, suppose we only have three eligible next tokens per parent in this simplified toy example:

| Parent beam | Parent probability | Proposed next token | Next-token probability | Cumulative candidate probability |
|---|---:|---|---:|---:|
| travel | 0.60 | abroad | 0.50 | 0.30 |
| travel | 0.60 | tomorrow | 0.30 | 0.18 |
| travel | 0.60 | soon | 0.20 | 0.12 |
| learn | 0.40 | Python | 0.70 | 0.28 |
| learn | 0.40 | English | 0.20 | 0.08 |
| learn | 0.40 | more | 0.10 | 0.04 |

For example:

$$
P(\text{travel abroad}\mid x)=0.60\times0.50=0.30
$$

$$
P(\text{learn Python}\mid x)=0.40\times0.70=0.28
$$

In log-score form:

$$
S(\text{learn Python}) = \log(0.40)+\log(0.70)=\log(0.28)
$$

We compute these candidate scores *before appending any of the proposed tokens*.

#### 5. Rank globally and append only the winners

Rank all six extensions together:

| Global rank | Hypothetical sequence | Cumulative probability |
|---:|---|---:|
| 1 | travel abroad | 0.30 |
| 2 | learn Python | 0.28 |
| 3 | travel tomorrow | 0.18 |
| 4 | travel soon | 0.12 |
| 5 | learn English | 0.08 |
| 6 | learn more | 0.04 |

Because beam width is $w=2$, the surviving beams become:

$$
B_1'=\text{travel abroad}
$$

$$
B_2'=\text{learn Python}
$$

**Only now** do we append "abroad" to the travel parent and "Python" to the learn parent. The next decode forward pass runs for the two surviving unfinished sequences; the discarded hypothetical candidates are not expanded further.

#### 6. Can both surviving extensions come from the same parent?

Yes. Suppose the existing beams A and B have cumulative probabilities 0.60 and 0.40, and their candidate token distributions are:

| Parent | P(X) | P(Y) |
|---|---:|---:|
| A | 0.60 | 0.40 |
| B | 0.50 | 0.50 |

All four extension probabilities are:

$$
P(A,X)=0.60\times0.60=0.36
$$

$$
P(A,Y)=0.60\times0.40=0.24
$$

$$
P(B,X)=0.40\times0.50=0.20
$$

$$
P(B,Y)=0.40\times0.50=0.20
$$

Global top-2 selection retains **A → X and A → Y**. Parent B disappears completely. Ordinary beam search does not reserve one winning extension for each parent.

**Efficiency note:** Actual implementations need not fully sort every vocabulary extension. Top-selection algorithms can find the required high-scoring candidates. In unconstrained global top-$w$ selection, each parent's top $w$ candidate extensions suffice as potential global winners, though EOS handling and other constraints can require a larger candidate pool.

---

### Question 4B — Does each beam produce its own full-vocabulary probability distribution, or does one forward pass produce one common distribution for all beams?

**Each active beam has its own conditional next-token distribution over the entire vocabulary.** Several beams can nevertheless be processed together in a **single batched Transformer forward pass**.

#### 1. Why are the distributions different?

Consider two beam histories:

~~~text
Beam 1: I want to travel
Beam 2: I want to learn
~~~

Their predictions are conditioned on different contexts:

$$
P(v\mid\text{I want to travel})
$$

$$
P(v\mid\text{I want to learn})
$$

The model weights are shared, but the beam histories are different. Therefore the predicted distributions will generally differ. For example, "abroad" may be likely after "travel", whereas "Python" may be likely after "learn".

#### 2. How does one batched forward pass produce both distributions?

Suppose beam width is $w=2$ and vocabulary size is $V=50{,}000$. An inference system can batch the two beam continuations, preserving their separate contexts and appropriate cached attention state.

The model's output logits have shape:

$$
Z\in\mathbb{R}^{w\times V}
$$

In our example:

$$
Z\in\mathbb{R}^{2\times50{,}000}
$$

We apply **softmax along the vocabulary dimension separately for each beam**:

$$
P_{b,v} = \frac{\exp(Z_{b,v})} {\sum_{j=1}^{V}\exp(Z_{b,j})}
$$

This gives:

$$
P\in\mathbb{R}^{2\times50{,}000}
$$

Each row independently sums to one:

$$
\sum_{v=1}^{V}P_{b,v}=1
$$

We **do not** compute one softmax across the logits of all beams combined.

For multiple independent prompts, an implementation may conceptually have logits of shape:

$$
Z\in\mathbb{R}^{B\times w\times V}
$$

where $B$ is the number of prompts. The prompt and beam dimensions may be flattened internally.

#### 3. What happens to the KV cache?

All beams originated from the same prompt and can share its cached prefix. Once their generated tokens diverge, they generally have different continuation-specific KV states.

A GPU can process their current decode steps together as a batch without making their conditional distributions identical. When beams are reordered after pruning, the implementation must reorder or reindex their corresponding KV states.

#### 4. What is special about the very first generation step?

Immediately after prefill, no beams have diverged yet. There is only **one known prompt and one next-token distribution** for that prompt. Beam search uses that distribution to choose its initial candidate tokens, thereby creating separate beam histories.

After those selected tokens diverge, each surviving beam generally needs its own next-token distribution at the following decode iteration. Their forward computations can still be batched.

#### 5. How does this connect back to scoring every extension?

For $w$ active beams and vocabulary size $V$:

$$
S\in\mathbb{R}^{w}
$$

contains the parent cumulative log scores. The per-beam probability matrix is:

$$
P\in\mathbb{R}^{w\times V}
$$

Broadcast each parent's score across its own vocabulary row:

$$
C_{b,v}=S_b+\log P_{b,v}
$$

The result:

$$
C\in\mathbb{R}^{w\times V}
$$

contains the candidate scores. Beam search selects the **global top-$w$ eligible beam–token pairs**, then appends their tokens to the corresponding parents and repeats for surviving unfinished beams.

**Mental model:** One shared Transformer model; one *conditional full-vocabulary distribution per active beam*; possibly one *batched* forward pass; and one *global selection* over all beam–token extensions.

---

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
0 \le P(y_t \vert x, y_{\lt t}) \le 1
$$

For positive probabilities, its logarithm is non-positive:

$$
\log P(y_t \vert x, y_{\lt t}) \le 0
$$


Adding tokens therefore generally makes the raw cumulative log score more negative:

$$
S(y_{1:t+1}) = S(y_{1:t}) + \log P(y_{t+1} \vert x, y_{1:t}) \le S(y_{1:t})
$$


When comparing completed outputs of different lengths, this creates a tendency for raw sequence likelihood to favor shorter possibilities, depending on EOS probabilities and stopping rules.


The fundamental reason is that **every additional token introduces another conditional probability factor, and that factor cannot exceed 1**.

Consequently, extending a particular sequence can never increase its raw sequence probability. This can create a preference for shorter completed sequences when beam search compares outputs of different lengths using their cumulative log probabilities.

Let's derive this step by step.

### 1. Why does generating another token reduce cumulative probability?

Suppose the prompt is:

`I want to`

The model predicts the following next-token probabilities:

| Next token | Probability |
|---|---:|
| travel | 0.60 |
| learn | 0.30 |
| sleep | 0.10 |

Suppose we choose `travel`.

The cumulative probability of this one-token continuation is:

$$
P(\text{travel}\mid\text{prompt})=0.60
$$

Now suppose the model predicts:

$$
P(\text{abroad}\mid\text{prompt, travel})=0.80
$$

The probability of generating the entire sequence `travel abroad` is:

$$
P(\text{travel abroad}\mid\text{prompt}) = 0.60\times0.80 = 0.48
$$

Notice that the cumulative probability has decreased from **0.60 to 0.48**, even though `abroad` had a high conditional probability of 0.80.

If the next token is `tomorrow`, with conditional probability 0.50:

$$
P(\text{travel abroad tomorrow}\mid\text{prompt}) = 0.60\times0.80\times0.50 = 0.24
$$

Our cumulative probabilities are now:

| Generated continuation | Cumulative probability |
|---|---:|
| travel | 0.60 |
| travel abroad | 0.48 |
| travel abroad tomorrow | 0.24 |

**Key observation:** Every new token multiplies the existing probability by a number between 0 and 1.

Mathematically:

$$
0\le P(y_{t+1}\mid x,y_{1:t})\le1
$$

Therefore:

$$
P(y_{1:t+1}\mid x) = P(y_{1:t}\mid x) P(y_{t+1}\mid x,y_{1:t})
$$

And hence:

$$
P(y_{1:t+1}\mid x) \le P(y_{1:t}\mid x)
$$

Extending a particular prefix can never increase its raw probability.

If the next token has probability exactly 1, the cumulative probability remains unchanged.

---

### 2. What happens when probabilities are converted into log scores?

Beam search generally accumulates log probabilities rather than multiplying probabilities directly.

The score of a partial sequence is:

$$
S(y_{1:t}) = \sum_{j=1}^{t} \log P(y_j \vert x, y_{\lt j})
$$


When another token is appended:

$$
S(y_{1:t+1}) = S(y_{1:t}) + \log P(y_{t+1}\mid x,y_{1:t})
$$

For probabilities between 0 and 1, their logarithms are negative:

$$
\log(0.80)\approx-0.223
$$

$$
\log(0.50)\approx-0.693
$$

Consequently, every additional token usually adds another negative value to the accumulated log score.

Consider our previous example:

| Generated continuation | Cumulative probability | Approximate log score |
|---|---:|---:|
| travel | 0.60 | -0.511 |
| travel abroad | 0.48 | -0.734 |
| travel abroad tomorrow | 0.24 | -1.427 |

For the first two tokens:

$$
S(\text{travel abroad}) = \log(0.60)+\log(0.80)
$$

$$
=-0.511-0.223
$$

$$
\approx-0.734
$$

For the third token:

$$
S(\text{travel abroad tomorrow}) = -0.734+\log(0.50)
$$

$$
=-0.734-0.693
$$

$$
\approx-1.427
$$

Thus:

$$
S(y_{1:t+1})\le S(y_{1:t})
$$

**Interpretation:** The longer the continuation becomes, the more negative its raw cumulative log score generally becomes.

This is a property of extending the *same prefix*, not a guarantee that every short sequence scores higher than every long sequence.

---

### 3. How does this create a preference for shorter completed sequences?

The important distinction is that beam search ultimately compares **different completed candidate sequences**, not merely successive prefixes.

A sequence is typically considered completed when it generates the end-of-sequence token, or EOS.

Consider two hypothetical completed outputs for the same prompt:

**Candidate A — Shorter sequence**

`travel EOS`

Its conditional probabilities are:

- `travel`: 0.60
- `EOS` after `travel`: 0.40

Its total sequence probability is:

$$
P(\text{travel,}\ \text{EOS} \vert x) = 0.60 \times 0.40 = 0.240
$$


**Candidate B — Longer sequence**

`travel abroad tomorrow EOS`

Suppose its conditional probabilities are:

- `travel`: 0.60
- `abroad` after `travel`: 0.50
- `tomorrow` after `travel abroad`: 0.80
- `EOS` after `travel abroad tomorrow`: 0.90

Its total probability is:

$$
\begin{aligned}
P(\text{travel,}\ \text{abroad,}\ \text{tomorrow,}\ \text{EOS} \vert x) &= 0.60 \times 0.50 \times 0.80 \times 0.90 \\
&= 0.216
\end{aligned}
$$

Compare the completed candidates:

| Candidate | Number of generated tokens (including EOS) | Raw sequence probability |
|---|---:|---:|
| travel EOS | 2 | 0.240 |
| travel abroad tomorrow EOS | 4 | 0.216 |

Since:

$$
0.240>0.216
$$

the shorter completed sequence has the higher raw probability.

Equivalently, their log scores are:

$$
S_{\text{short}} = \log(0.240) \approx-1.427
$$

$$
S_{\text{long}} = \log(0.216) \approx-1.532
$$

And:

$$
-1.427>-1.532
$$

Therefore, beam search using raw cumulative log probability would rank the shorter completed sequence higher.

**Why did this happen?**

The longer sequence has additional probability factors. Even when these additional tokens are individually likely, their probabilities multiply together, reducing the raw probability of the complete sequence.

**Important qualification:** A longer candidate can still outperform a different shorter candidate if its conditional probabilities are sufficiently high. Raw likelihood creates a length bias; it does not impose a rule that the shortest sequence always wins.

---

### 4. How does length normalization address this problem?

Length normalization adjusts the cumulative score to account for sequence length.

One simple approach is to calculate the **average log probability per generated token**:

$$
S_{\mathrm{avg}}(y_{1:T}) = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t \vert x, y_{\lt t})
$$

Here:

- $T$ is the number of generated tokens.
- $y_t$ is the token generated at position $t$.
- $x$ is the original prompt.
- $S_{\mathrm{avg}}$ is the average log probability.

For consistency in the following numerical example, we count EOS as a generated token.

**Shorter completed sequence**

$$
S_{\mathrm{short}}=\log(0.240)\approx-1.427
$$

It contains two generated tokens, including EOS:

$$
S_{\mathrm{avg,short}} = \frac{-1.427}{2} \approx-0.714
$$

**Longer completed sequence**

$$
S_{\mathrm{long}}=\log(0.216)\approx-1.532
$$

It contains four generated tokens, including EOS:

$$
S_{\mathrm{avg,long}} = \frac{-1.532}{4} \approx-0.383
$$

Now compare both scoring methods:

| Sequence | Raw log score | Average log score |
|---|---:|---:|
| travel EOS | -1.427 | -0.714 |
| travel abroad tomorrow EOS | -1.532 | -0.383 |

Using raw log probability:

$$
-1.427>-1.532
$$

The shorter candidate ranks higher.

Using average log probability:

$$
-0.383>-0.714
$$

The longer candidate ranks higher.

**What changed?**

The longer candidate has a higher average conditional probability per token, even though multiplying all its token probabilities gives a smaller raw sequence probability.

Length normalization reduces the disadvantage of accumulating additional negative log-probability terms.

---

### 5. Is average log probability the only length-normalization formula?

No. Different beam-search implementations use different length penalties.

Another commonly used scoring family is:

$$
S_{\mathrm{len}}(y_{1:T}) = \frac{\log P(y_{1:T}\mid x)} {\left(\frac{5+T}{6}\right)^{\alpha}}
$$

Here:

- $T$ is the generated sequence length under the implementation's counting convention.
- $\alpha$ controls the strength of the length penalty.
- $\alpha=0$ removes the length adjustment.

When:

$$
\alpha=0
$$

the denominator becomes:

$$
\left(\frac{5+T}{6}\right)^0=1
$$

Therefore:

$$
S_{\mathrm{len}}(y_{1:T}) = \log P(y_{1:T}\mid x)
$$

The score is simply the original cumulative log probability.

Length-penalty formulas, EOS counting, pruning rules and early-stopping behavior vary across inference libraries.

---

### Final takeaway

**Why can beam search prefer shorter completed sequences?**

Every new token multiplies a sequence's raw probability by an additional factor that is at most 1. Equivalently, it adds a non-positive log probability to the cumulative score.

As a result, longer sequences accumulate additional negative score contributions and may rank below shorter completed alternatives.

Length normalization adjusts scores to make comparisons across different output lengths less sensitive to the number of probability factors.

**Remember the distinction:**

- **Raw cumulative log probability:** Scores the probability of the entire generated sequence.
- **Length-normalized score:** Adjusts that score to account for how many tokens the sequence contains.
- **Beam search:** Ranks candidate sequences using whichever scoring rule its implementation specifies.


## Question 10 — What is length normalization?

Length normalization modifies sequence scores to address differences in sequence length.

One simple conceptual score is average log probability:

$$
S_{\mathrm{avg}}(y_{1:T}) = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t \vert x, y_{\lt t})
$$


Another commonly used family of scores is:

$$
S_{\mathrm{len}}(y_{1:T}) = \frac{\log P(y_{1:T} \vert x)}{\left(\frac{5+T}{6}\right)^{\alpha}}
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
P(y_{1:T} \vert x) = \prod_{t=1}^{T} P(y_t \vert x, y_{\lt t})
$$


**Accumulated beam score:**

$$
S(y_{1:T}) = \sum_{t=1}^{T} \log P(y_t \vert x, y_{\lt t})
$$


**Add one token to a beam:**

$$
S(y_{1:t} \parallel v) = S(y_{1:t}) + \log P(v \vert x, y_{1:t})
$$


**Example length-adjusted score:**

$$
S_{\mathrm{len}}(y_{1:T}) = \frac{\log P(y_{1:T} \vert x)}{\left(\frac{5+T}{6}\right)^{\alpha}}
$$


**Example sign-aware multiplicative repetition penalty for an already-seen token:**

$$
z_i' =
\begin{cases}
\frac{z_i}{r}, & z_i > 0 \\
z_i r, & z_i \text{ } \lt 0 \\
0, & z_i = 0
\end{cases}
$$


**Simple presence and frequency penalties:**

$$
\begin{aligned}
z_i' &= z_i - \lambda_{\mathrm{presence}} \mathbf{1}[c_i \gt 0] \\
z_i' &= z_i - \lambda_{\mathrm{frequency}} c_i
\end{aligned}
$$


# Bridge to Part 7

Beam search helps manage alternative continuations, while repetition controls help discourage undesirable repetition. The next two questions concern when generation should terminate:

- **8.12 Stop Tokens**
- **8.13 Maximum Output Length**
