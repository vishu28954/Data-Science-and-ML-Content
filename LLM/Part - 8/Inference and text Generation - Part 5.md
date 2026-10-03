# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 5

## 8.8 Top-k Sampling
## 8.9 Top-p / Nucleus Sampling

**Detailed Study Mode** — Interview answers will be kept in a separate file.

This part continues our story from greedy decoding and temperature:

~~~text
Transformer → next-token logits
                 ↓
       optional temperature
                 ↓
     decide which tokens
     remain eligible
          ↙      ↘
        top-k   top-p
          ↘      ↙
       renormalize
             ↓
         sample
             ↓
   append token and repeat
~~~

Greedy always picks the locally highest-scoring token. Temperature changes probability concentration but, on its own, does not remove finite-logit tokens from the sampling pool. We now need a way to restrict which tokens can be sampled.

---

# 8.8 Top-k Sampling

## Question 1 — Why do we need top-k if we already have temperature?

Suppose the model's next-token distribution is:

| Token | Probability |
|---|---:|
| A | 0.40 |
| B | 0.30 |
| C | 0.15 |
| D | 0.10 |
| E | 0.05 |

Greedy selects A every time. Ordinary sampling allows all five tokens to appear. Temperature can sharpen or flatten their relative probabilities, but a finite positive temperature does not remove a token that has a finite logit.

**Top-k sampling keeps only the $k$ highest-probability eligible tokens, excludes the rest, renormalizes, and randomly samples from the remaining candidates.**

For $k=3$, the eligible tokens are A, B and C. D and E receive zero sampling probability.

### Why does this help?

Instead of relying entirely on the distribution's very low-probability tail, we restrict random choices to the highest-ranked candidates. However, being high-ranked does not guarantee factual correctness or semantic quality.

## Question 2 — What is the mathematical definition of top-k?

Let the model produce logits:

$$
z_{t+1}\in\mathbb{R}^{|\mathcal{V}|}
$$

where $\mathcal{V}$ is the vocabulary. If temperature $T>0$ is applied, the probabilities are:

$$
p_i=\frac{\exp(z_i/T)}{\sum_{j\in\mathcal{V}}\exp(z_j/T)}
$$

Sort eligible tokens by descending probability:

$$
p_{(1)}\ge p_{(2)}\ge\cdots\ge p_{(|\mathcal{V}|)}
$$

Let $S_k$ be the set containing the first $k$ tokens. Its original probability mass is:

$$
Z_k=\sum_{j\in S_k}p_j
$$

Then define the filtered distribution:

$$
P_k(i) = \begin{cases} \frac{p_i}{Z_k}, & i \in S_k \\ 0, & i \notin S_k \end{cases}
$$

Finally, sample:

$$
x_{t+1} \sim \mathop{\text{Categorical}}(P_k)
$$


### Why do we need renormalization?

A probability distribution must sum to one. Removing candidates generally leaves less than one unit of probability mass. Dividing by $Z_k$ restores the total to one while preserving the ratios among retained candidates.

We divide by *retained probability mass*, not by the number $k$.

## Question 3 — Can we calculate top-k using actual numbers?

For $k=3$, retain A, B, C:

$$
S_3=\{A,B,C\}
$$

The retained mass is:

$$
Z_3=0.40+0.30+0.15=0.85
$$

Normalize:

$$
P_3(A)=\frac{0.40}{0.85}\approx0.470588
$$

$$
P_3(B)=\frac{0.30}{0.85}\approx0.352941
$$

$$
P_3(C)=\frac{0.15}{0.85}\approx0.176471
$$

| Token | Before filtering | After top-k |
|---|---:|---:|
| A | 40.00% | 47.06% |
| B | 30.00% | 35.29% |
| C | 15.00% | 17.65% |
| D | 10.00% | 0.00% |
| E | 5.00% | 0.00% |
| **Total** | **100%** | **100%** |

The ratio between A and B is preserved:

$$
\frac{P_3(A)}{P_3(B)}=\frac{0.40}{0.30}=\frac43
$$

The candidate set changes; the relative odds among the retained candidates do not.

## Question 4 — What does sampling actually do after filtering?

The new distribution is approximately:

$$
[0.470588,\;0.352941,\;0.176471]
$$

Imagine drawing a random number uniformly from $[0,1)$. The cumulative intervals are approximately:

~~~text
A: [0.000000, 0.470588)
B: [0.470588, 0.823529)
C: [0.823529, 1.000000)
~~~

A draw of $0.64$ selects B; a draw of $0.90$ selects C. These intervals are a conceptual implementation of categorical sampling, not a required implementation detail.

**Top-k does not always select A.** Multiple retained tokens remain possible unless the set contains only one token.

After the token is selected, it is appended to the context. The model produces *new logits for the following token*, and the filtering and sampling process repeats. A previous step's candidate set is not automatically reused.

## Question 5 — What edge cases matter for top-k?

- **$k=1$:** Only the highest-ranked token survives. The choice is greedy-like, with implementation-specific tie handling.
- **$k=|\mathcal{V}|$:** No initially eligible vocabulary token is removed.
- **$k>|\mathcal{V}|$:** The setting must be rejected or clamped.
- **$k=0$:** Not a meaningful candidate count mathematically; some libraries use it to mean *disable top-k*. Check the API.
- **Ties at the cutoff:** A tie-breaking convention determines which $k$ tokens survive.
- **Previously forbidden tokens:** Top-k cannot restore tokens already masked out by generation constraints.
- **Model confidently wrong:** Filtering by probability does not ensure factual correctness.

## Question 6 — What is top-k's fundamental limitation?

It keeps the *same number* of candidates even when the probability distribution looks very different.

Consider:

| Token | Peaked distribution | Flatter distribution |
|---|---:|---:|
| A | 0.90 | 0.25 |
| B | 0.06 | 0.22 |
| C | 0.02 | 0.20 |
| D | 0.01 | 0.18 |
| E | 0.01 | 0.15 |

If $k=3$, both distributions retain three tokens.

For the peaked distribution, retained mass is:

$$
0.90+0.06+0.02=0.98
$$

For the flatter distribution:

$$
0.25+0.22+0.20=0.67
$$

The exact same $k$ retains 98% in one situation but only 67% in another.

### The unresolved question

Why fix the number of candidates instead of fixing how much probability mass we want the eligible candidates to cover?

That leads to **top-p**.

---

# 8.9 Top-p / Nucleus Sampling

## Question 1 — What does top-p do differently from top-k?

Top-p, or **nucleus sampling**, keeps the *smallest descending-probability prefix of tokens* whose cumulative probability reaches or exceeds a threshold $p$.

Consider:

| Token | Probability | Cumulative probability |
|---|---:|---:|
| A | 0.50 | 0.50 |
| B | 0.25 | 0.75 |
| C | 0.12 | 0.87 |
| D | 0.08 | 0.95 |
| E | 0.05 | 1.00 |

Let:

$$
p=0.80
$$

A and B together cover only 0.75. Adding C brings the total to 0.87, crossing the threshold. Therefore:

$$
S_p=\{A,B,C\}
$$

D and E are excluded.

We do not try to reach *exactly* 0.80; entire tokens must be included. The first prefix that reaches or exceeds the threshold is sufficient.

## Question 2 — What is the mathematical definition of nucleus sampling?

Sort the probability distribution:

$$
p_{(1)} \ge p_{(2)} \ge \cdots \ge p_{( \vert \mathcal{V} \vert )}
$$


For $0<p\le1$, choose the smallest integer $m$ satisfying:

$$
m = \min \left\lbrace r : \sum_{i=1}^{r} p_{(i)} \ge p \right\rbrace
$$



The nucleus is the set of the first $m$ ranked tokens:

$$
S_p=\{(1),(2),\ldots,(m)\}
$$

Its retained mass is:

$$
Z_p=\sum_{j\in S_p}p_j
$$

The filtered distribution is:

$$
P_p(i) =
\begin{cases}
\frac{p_i}{Z_p}, & i \in S_p \\
0, & i \notin S_p
\end{cases}
$$

Then:

$$
x_{t+1} \sim \mathop{\text{Categorical}}(P_p)
$$


### Why do we need this mathematics?

The cumulative-sum inequality decides **when to stop adding tokens**. The minimum ensures that we keep no additional lower-ranked candidates once the threshold is reached. Division by $Z_p$ produces a valid distribution for sampling.

## Question 3 — Can we calculate top-p completely?

For the example above:

$$
Z_p=0.50+0.25+0.12=0.87
$$

Therefore:

$$
P_p(A)=\frac{0.50}{0.87}\approx0.574713
$$

$$
P_p(B)=\frac{0.25}{0.87}\approx0.287356
$$

$$
P_p(C)=\frac{0.12}{0.87}\approx0.137931
$$

And D and E now have zero probability.

| Token | Original | Top-p, $p=0.80$ |
|---|---:|---:|
| A | 50.00% | 57.47% |
| B | 25.00% | 28.74% |
| C | 12.00% | 13.79% |
| D | 8.00% | 0.00% |
| E | 5.00% | 0.00% |
| **Total** | **100%** | **100%** |

The 87% original mass of A, B, C has been rescaled to 100%.

## Question 4 — Why is top-p called adaptive?

Apply the same $p=0.80$ to the peaked and flatter distributions from the top-k section.

### Peaked distribution

Token A alone has probability 0.90:

$$
0.90\ge0.80
$$

So the nucleus contains **one token**.

### Flatter distribution

Cumulative probability grows as:

$$
0.25,\quad0.47,\quad0.67,\quad0.85
$$

We need four tokens to pass 0.80.

Thus the same threshold keeps one token in the peaked case but four tokens in the flatter case.

~~~text
Same p = 0.80

Peaked:
A alone covers 90%
→ retain 1 token

Flatter:
A+B+C cover 67%
A+B+C+D cover 85%
→ retain 4 tokens
~~~

This is the main distinction: top-k fixes candidate count; top-p adapts it to the probability distribution.

### Important limitation

Distribution concentration is not a guarantee of correctness. A model can be highly concentrated on a false answer.

## Question 5 — How does temperature interact with top-p?

Temperature-scaled probabilities are:

$$
P_T(i)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

For positive $T$, temperature preserves **ranking**, so it ordinarily does not change which tokens occupy the top $k$. However, it **changes cumulative probability mass**, so the nucleus size for top-p can change.

Take:

$$
z=[4,2,1]
$$

At $T=1$:

$$
P_1\approx[0.8438,\;0.1142,\;0.0420]
$$

With $p=0.80$, the top token alone already crosses the threshold: retain **one** token.

At $T=2$:

$$
P_2\approx[0.6285,\;0.2312,\;0.1402]
$$

Now the top token alone is insufficient:

$$
0.6285<0.80
$$

But the top two together reach approximately:

$$
0.6285+0.2312=0.8597
$$

So top-p retains **two** tokens.

The original logits, token ranking, and top-p threshold are identical. Only the temperature changed.

## Question 6 — Can top-k and top-p be combined?

Yes. Many inference libraries allow both parameters.

One conceptual pipeline is:

~~~text
Logits
  ↓
Optional logit penalties / constraints
  ↓
Temperature
  ↓
Top-k filtering
  ↓
Top-p filtering over the eligible distribution
  ↓
Renormalize
  ↓
Sample
~~~

**The exact ordering is implementation-specific.** Applying one filter first changes the distribution seen by the next filter, particularly when top-p calculates cumulative mass. Do not assume every library simply intersects two independently computed candidate sets.

For top-k alone, an implementation may mask excluded logits to negative infinity before softmax instead of explicitly computing all probabilities and then filtering.

## Question 7 — What edge cases matter for top-p?

- **$p=1$:** For ordinary positive probabilities, retain the full eligible distribution. Tokens already masked to zero do not become available.
- **Very small positive $p$:** The top token alone often reaches the threshold; sampling is greedy-like for that step.
- **$p=0$:** Usually invalid mathematically or specially handled by the library; implementations must keep at least one token to sample.
- **Threshold reached exactly:** Stop when the cumulative probability equals $p$; no further token is required.
- **Threshold exceeded:** Keep the entire crossing token; do not try to keep a fractional token.
- **Equal-ranked tokens near cutoff:** Tie-handling and ordering can determine which tokens survive.
- **Numerical rounding:** Cumulative sums near the threshold can be sensitive to floating-point precision.
- **All tokens masked:** The inference system needs an explicit error or fallback policy.

## Question 8 — Do these filters guarantee diversity or factuality?

No.

Top-k and top-p restrict which tokens are eligible; they do not search for the globally best complete sequence or verify generated claims. Small candidate sets often reduce diversity. Large candidate sets increase the chance of selecting less-probable alternatives, including unsuitable ones.

The user-visible effect also depends on temperature, the prompt, the model, and how sampling is implemented.

## Question 9 — What is their computational cost?

Both techniques require identifying high-ranked candidate tokens. A straightforward full vocabulary sort has cost:

$$
O(V\log V)
$$

where $V$ is the number of eligible vocabulary items.

Optimized top-k can avoid fully sorting the vocabulary. Top-p implementations require some way of ordering or identifying candidates and accumulating probability mass. Exact overhead depends on kernels, vocabulary size, batching, and hardware.

**Neither method removes the Transformer forward pass or the sequential autoregressive dependency across output tokens.**

---

---

# Worked Example — Temperature, Top-k and Top-p Together

This example connects both topics by following **one next-token distribution from beginning to end**. It also shows why the order of candidate filters can matter.

## Step 1 — Start with the next-token probabilities

Suppose the Transformer has finished one forward pass and, after softmax at **temperature $T=1$**, its next-token distribution is:

| Token | Original probability |
|---|---:|
| A | 0.40 |
| B | 0.30 |
| C | 0.15 |
| D | 0.10 |
| E | 0.05 |
| **Total** | **1.00** |

We will use the same settings throughout:

$T=1$, &nbsp;&nbsp;&nbsp;&nbsp; $k=3$, &nbsp;&nbsp;&nbsp;&nbsp; $p=0.80$

Temperature $T=1$ leaves the original softmax distribution unchanged. This isolates the effect of the two filters. We will first apply **top-k, then top-p**, and then reverse their order.

## Step 2 — Apply top-k with k = 3

Keep the three highest-probability tokens:

$S_k=\lbrace A,B,C \rbrace$


The retained mass is:

$Z_k = 0.40 + 0.30 + 0.15 = 0.85$


After renormalization:

$$
\begin{aligned}
P_k(A) &= \frac{0.40}{0.85} = \frac{8}{17} \approx 0.4706 \\
P_k(B) &= \frac{0.30}{0.85} = \frac{6}{17} \approx 0.3529 \\
P_k(C) &= \frac{0.15}{0.85} = \frac{3}{17} \approx 0.1765
\end{aligned}
$$



So the distribution is now:

| Token | After top-k |
|---|---:|
| A | 0.4706 |
| B | 0.3529 |
| C | 0.1765 |
| D | 0 |
| E | 0 |

Notice that top-k has **changed the numerical probabilities** by renormalizing the retained mass.

## Step 3 — Apply top-p = 0.80 to the top-k distribution

Top-p now sees the **renormalized distribution from Step 2**, not the original distribution.

Accumulate probabilities in descending order:

| Token | Current probability | Cumulative mass |
|---|---:|---:|
| A | 0.4706 | 0.4706 |
| B | 0.3529 | 0.8235 |
| C | 0.1765 | 1.0000 |

The first token does not reach 0.80, but the first two do:

$P_k(A) + P_k(B) = \frac{14}{17} \approx 0.8235 \ge 0.80$


So top-p keeps only:

$S_p = \lbrace A, B \rbrace$

Renormalize again. Because A and B originally had probability 0.40 and 0.30, their final relative proportions are:

$$
\begin{aligned}
P_{\text{final}}(A) &= \frac{0.40}{0.40 + 0.30} = \frac{4}{7} \approx 0.5714 \\
P_{\text{final}}(B) &= \frac{0.30}{0.40 + 0.30} = \frac{3}{7} \approx 0.4286
\end{aligned}
$$


The final sampling distribution is:

$$
P_{\text{final}} = \begin{bmatrix} 0.5714 & 0.4286 & 0 & 0 & 0 \end{bmatrix}
$$

**Top-k followed by top-p has left only two eligible tokens.**

## Step 4 — Sample one token

Suppose we illustrate categorical sampling by drawing:

$
u=0.55
$

from a uniform distribution on $[0,1)$.

The final cumulative intervals are:

~~~text
A: [0.0000, 0.5714)
B: [0.5714, 1.0000)
~~~

The draw lies in A's interval, so this illustrative sampling step selects **A**.

The selected token is appended to the sequence. The model then runs another decode forward pass to compute a **new** next-token distribution. It does not reuse the current distribution for the following token.

## Step 5 — What changes if we reverse the filter order?

Reset to the **original** distribution and apply **top-p first**, still with $p=0.80$.

Its cumulative mass is:

| Token | Original probability | Cumulative mass |
|---|---:|---:|
| A | 0.40 | 0.40 |
| B | 0.30 | 0.70 |
| C | 0.15 | 0.85 |
| D | 0.10 | 0.95 |
| E | 0.05 | 1.00 |

The smallest prefix reaching 0.80 is:

$$
S_p = \left\lbrace A, B, C \right\rbrace
$$


Its retained mass is 0.85. Renormalization gives:

$$
P_p = \begin{bmatrix} 0.4706 & 0.3529 & 0.1765 & 0 & 0 \end{bmatrix}
$$


Now apply **top-k with $k=3$**. All three currently eligible tokens survive, so the distribution stays the same:

$$
P_{\text{reverse}} = \begin{bmatrix} 0.4706 & 0.3529 & 0.1765 & 0 & 0 \end{bmatrix}
$$

Using the same illustrative draw:

$$
u = 0.55
$$


the draw now falls in B's cumulative interval:

~~~text
A: [0.0000, 0.4706)
B: [0.4706, 0.8235)
C: [0.8235, 1.0000)
~~~

So the reverse order selects **B**, not A, for this particular shared draw.

## Step 6 — Compare the outcomes

| Operation order | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| Top-k 3, then top-p 0.80 | 0.5714 | 0.4286 | 0 | 0 | 0 |
| Top-p 0.80, then top-k 3 | 0.4706 | 0.3529 | 0.1765 | 0 | 0 |

Both orders use the **same original distribution** and **same settings**, but produce different final sampling distributions.

Why? Because top-p's cumulative threshold is calculated on the distribution it receives. If top-k renormalizes that distribution first, top-p can reach its threshold with fewer candidates.

**Implementation note:** This is a conceptual example of sequential filtering with renormalization between stages. In real inference libraries, the order of logit processors, masking, temperature, and top-p calculation is implementation-dependent; inspect the library rather than assuming one universal order.

### Final mental model

~~~text
Temperature → controls relative probability concentration

Top-k → keeps a fixed number of highest-ranked tokens

Top-p → keeps enough tokens to cross a probability threshold

Renormalization → restores total probability to one

Sampling → draws one eligible token

New forward pass → produces a new distribution for the next token
~~~


# Top-k vs Top-p — Final Comparison

| Property | Top-k | Top-p / Nucleus |
|---|---|---|
| Main question | How many tokens can be sampled? | How much probability mass must remain eligible? |
| Main parameter | Integer $k$ | Threshold $p$ |
| Candidate count | Fixed, subject to eligibility | Adaptive |
| Selection rule | Keep highest $k$ | Smallest sorted prefix reaching $p$ |
| Original mass retained | Varies | At least $p$ |
| After selection | Renormalize and sample | Renormalize and sample |
| Main limitation | Ignores distribution shape | May include many candidates on flat distributions |

### Three memory lines

1. Greedy chooses the highest-ranked token directly.
2. Top-k keeps a fixed number of high-ranked tokens and samples among them.
3. Top-p keeps the smallest high-ranked set covering a target probability mass and samples among them.

Temperature controls the probability ratios. Top-k and top-p control the eligible candidate set. Sampling makes the actual random selection.

---

# Why Does This Lead to Beam Search?

Top-k and top-p improve the flexibility of **one-step token selection**. But neither one evaluates every possible *complete continuation*. Greedy can commit early to a token whose future continuations are weaker, and sampling can make the same kind of local choice.

The next question is:

> Can we retain multiple partial sequences at once instead of committing to only one evolving path?

That leads to **8.10 Beam Search**. After that, we will study **8.11 Repetition Penalty** to understand how generation systems discourage unhelpful repeated tokens.
