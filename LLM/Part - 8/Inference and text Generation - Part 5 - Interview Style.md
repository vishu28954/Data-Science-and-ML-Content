# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 5 — Interview Style

## Topics
- **8.8 Top-k Sampling**
- **8.9 Top-p / Nucleus Sampling**

These notes are exclusively for interview preparation. The separate [Part 5 Detailed Study](./Inference%20and%20text%20Generation%20-%20Part%205.md) contains the full question-led explanations and worked derivations.

---

# 8.8 Top-k Sampling

## Q1. What is top-k sampling?

**Interview answer:** Top-k is a decoding strategy that retains only the $k$ highest-probability eligible tokens at each generation step. It masks the others, renormalizes the retained probability mass, and samples one of those $k$ candidates. Unlike greedy decoding, multiple tokens can be chosen when $k>1$.

## Q2. Why use top-k when we already have temperature?

**Interview answer:** Temperature adjusts how sharply probability mass is concentrated, but positive finite temperature does not remove tokens with finite logits. Top-k restricts the sampling pool to a fixed number of high-ranked candidates, excluding the low-ranked tail. They solve different problems: **temperature changes probabilities; top-k changes eligibility**.

## Q3. What is the mathematical formula for top-k sampling?

Let $p_i$ be the token probabilities and $S_k$ be the set of the $k$ highest-probability eligible tokens. The retained mass is:

$$
Z_k=\sum_{j\in S_k}p_j
$$

The filtered distribution is:

$$
P_k(i)=
\begin{cases}
\displaystyle\frac{p_i}{Z_k}, & i\in S_k\\[8pt]
0, & i\notin S_k
\end{cases}
$$

The token is then sampled:

$$
x_{t+1}\sim\operatorname{Categorical}(P_k)
$$

**Interview emphasis:** We divide by *retained probability mass*, not by $k$.

## Q4. Can you give a numerical top-k example?

**Interview answer:** Suppose:

| Token | Original probability |
|---|---:|
| A | 0.40 |
| B | 0.30 |
| C | 0.15 |
| D | 0.10 |
| E | 0.05 |

With $k=3$, retain A, B and C. The retained mass is:

$$
Z_k=0.40+0.30+0.15=0.85
$$

After renormalization:

$$
P_3(A)=\frac{0.40}{0.85}\approx0.4706
$$

$$
P_3(B)=\frac{0.30}{0.85}\approx0.3529
$$

$$
P_3(C)=\frac{0.15}{0.85}\approx0.1765
$$

D and E have probability zero. The final distribution is approximately:

$$
[0.4706,\;0.3529,\;0.1765,\;0,\;0]
$$

A is most likely, but sampling can still select B or C.

## Q5. What exactly happens during categorical sampling after top-k?

**Interview answer:** We draw one token according to the renormalized probabilities. A conceptual implementation generates $u$ uniformly from $[0,1)$ and locates the cumulative probability interval containing it.

For the previous example:

~~~text
A: [0.000000, 0.470588)
B: [0.470588, 0.823529)
C: [0.823529, 1.000000)
~~~

A draw of $u=0.64$ selects B; $u=0.90$ selects C. After selection, the token is appended to the context, the model produces new logits, and filtering is repeated for the next token.

## Q6. Is top-k the same as selecting the top token?

**Interview answer:** No. Top-k selects a **candidate set**, then normally samples from it. When $k=1$, only the highest-ranked token remains, so selection is greedy-like. For $k>1$, lower-ranked retained tokens can still be selected.

## Q7. What are the important top-k edge cases?

**Interview answer:** $k=1$ yields a single candidate. Choosing $k$ equal to the number of eligible tokens removes no additional tokens. A value larger than the eligible vocabulary needs clamping or rejection. Exact ties at the cutoff need a tie-breaking rule. Some libraries use $k=0$ as a special *disable filtering* setting, although a zero-sized candidate set is not mathematically usable.

## Q8. What is the main limitation of top-k?

**Interview answer:** It always keeps a fixed number of candidates, regardless of distribution shape. A peaked distribution might assign 98% of its probability to the top three, while a flatter one assigns only 67% to its top three. The same $k$ does not represent the same amount of probability mass at every generation step.

---

# 8.9 Top-p / Nucleus Sampling

## Q9. What is top-p sampling?

**Interview answer:** Top-p, also called nucleus sampling, sorts eligible tokens in descending probability order and retains the **smallest set whose cumulative probability reaches or exceeds threshold $p$**. It then renormalizes the retained probabilities and samples one token.

Unlike top-k, it does not fix the number of candidates in advance; the nucleus size adapts to the current probability distribution.

## Q10. What is the exact mathematical definition of top-p?

Sort token probabilities:

$$
p_{(1)}\ge p_{(2)}\ge\cdots\ge p_{(|\mathcal{V}|)}
$$

For $0<p\le1$, choose the smallest integer:

$$
m=\min\left\{r:\sum_{i=1}^{r}p_{(i)}\ge p\right\}
$$

Keep:

$$
S_p=\{(1),(2),\ldots,(m)\}
$$

Define retained mass:

$$
Z_p=\sum_{j\in S_p}p_j
$$

Then:

$$
P_p(i)=
\begin{cases}
\displaystyle\frac{p_i}{Z_p}, & i\in S_p\\[8pt]
0, & i\notin S_p
\end{cases}
$$

Finally:

$$
x_{t+1}\sim\operatorname{Categorical}(P_p)
$$

## Q11. Give a numerical top-p example.

**Interview answer:** Suppose the sorted probabilities are:

| Token | Probability | Cumulative probability |
|---|---:|---:|
| A | 0.50 | 0.50 |
| B | 0.25 | 0.75 |
| C | 0.12 | 0.87 |
| D | 0.08 | 0.95 |
| E | 0.05 | 1.00 |

For $p=0.80$, A and B together cover only 0.75. Adding C reaches 0.87, so A, B and C form the nucleus. Renormalizing:

$$
P_p(A)=\frac{0.50}{0.87}\approx0.5747
$$

$$
P_p(B)=\frac{0.25}{0.87}\approx0.2874
$$

$$
P_p(C)=\frac{0.12}{0.87}\approx0.1379
$$

D and E cannot be sampled.

## Q12. Why can retained probability mass exceed the top-p threshold?

**Interview answer:** We retain whole tokens. The final included token may push cumulative mass beyond the threshold. For example, with $p=0.80$, mass can jump from 0.75 to 0.87. The algorithm stops at the first prefix *at or above* the threshold; it does not split the final token.

## Q13. Why is top-p adaptive?

**Interview answer:** Its candidate count varies with probability concentration. For $p=0.80$, if the top token already has 0.90 probability, one token is enough. If probabilities are $[0.25,0.22,0.20,0.18,0.15]$, four tokens are required because the first three reach only 0.67, while the first four reach 0.85.

So the same threshold can lead to a one-token nucleus in a peaked distribution and a larger nucleus in a flatter distribution.

## Q14. What are the main top-p edge cases?

**Interview answer:** $p=1$ ordinarily retains the full eligible distribution. A very small positive $p$ often retains just the top token. The nucleus stops when the cumulative mass reaches the threshold exactly or first exceeds it. Ties, near-threshold floating-point effects and handling of $p=0$ depend on implementation; there must always be at least one token available to sample.

---

# Combining Temperature, Top-k and Top-p

## Q15. Does temperature affect top-k and top-p differently?

**Interview answer:** Yes. Positive temperature preserves logit ranking, so by itself it does not change the identity of the top $k$ tokens. But temperature changes the cumulative probability mass of ranked tokens, so it can change the number retained by top-p.

For logits:

$$
z=[4,2,1]
$$

at $T=1$, the approximate distribution is:

$$
[0.8438,\;0.1142,\;0.0420]
$$

At $p=0.80$, the first token is enough.

At $T=2$:

$$
[0.6285,\;0.2312,\;0.1402]
$$

the top two together reach approximately 0.8597, so the nucleus now contains two tokens.

## Q16. Can top-k and top-p both be applied?

**Interview answer:** Yes. They may both restrict the eligible set before sampling. The exact order is implementation-specific and matters because the first filter changes the distribution on which the second filter operates. Temperature and any logit processors or constraints can also affect the probabilities supplied to these filters.

## Q17. Show why filter order can change the result.

**Interview answer:** Start with:

$$
[0.40,\;0.30,\;0.15,\;0.10,\;0.05]
$$

and use $k=3$ and $p=0.80$.

**Top-k first:** Keep A, B and C; their renormalized probabilities are approximately:

$$
[0.4706,\;0.3529,\;0.1765,\;0,\;0]
$$

Top-p now sees:

$$
0.4706+0.3529\approx0.8235\ge0.80
$$

so it keeps A and B only. The final distribution is:

$$
[0.5714,\;0.4286,\;0,\;0,\;0]
$$

**Top-p first:** On the original distribution, A and B total only 0.70, so C is also needed to reach 0.85. After normalization the distribution is:

$$
[0.4706,\;0.3529,\;0.1765,\;0,\;0]
$$

Top-k with $k=3$ removes none of those three.

**Conclusion:** The order can change both the eligible tokens and their final sampling probabilities. This example assumes renormalization between the two filters.

## Q18. What happens after sampling one token?

**Interview answer:** The chosen token is appended to the current context. In the next autoregressive decode forward pass, the model processes that token using the existing KV cache and produces **new logits for the following token**. Temperature and sampling filters are then applied to this new distribution. We do not reuse the previous generation step's candidate probabilities.

## Q19. Are top-k and top-p methods for improving factuality?

**Interview answer:** They are probability-filtering and sampling controls, not factuality verification methods. They can reduce sampling from very low-ranked tokens, but they cannot correct a model that places high probability on a false continuation. Likewise, a concentrated distribution should not be treated as a guarantee of correctness.

## Q20. What is the computational overhead of top-k and top-p?

**Interview answer:** Both involve selecting or ordering high-ranked vocabulary candidates. A straightforward full sort of $V$ candidates costs:

$$
O(V\log V)
$$

Optimized top-k can use partial selection, and top-p implementations can use suitable ranking or threshold-finding methods. Exact overhead depends on implementation, vocabulary size and hardware. Neither technique eliminates the Transformer forward pass or autoregressive generation's sequential dependency.

---

# Complete Spoken Answer — Top-k

Top-k sampling retains the $k$ highest-probability tokens, masks all other eligible candidates, renormalizes the retained probabilities and randomly samples one token. Unlike greedy decoding, it can choose several different continuations when $k$ is greater than one. Its main limitation is that it retains a fixed number of tokens, irrespective of whether the next-token distribution is very peaked or relatively flat.

# Complete Spoken Answer — Top-p

Top-p, or nucleus sampling, sorts tokens by probability and retains the smallest prefix whose cumulative probability reaches or exceeds a chosen threshold. It renormalizes that subset and samples one token. Unlike top-k, its candidate count adapts to the probability distribution. For example, a very peaked distribution may require only one candidate, while a flatter distribution may require several to cover the same cumulative mass.

# Combined Spoken Answer — Top-k vs Top-p

Both top-k and top-p restrict the set of tokens eligible for sampling. Top-k retains a fixed number of highest-ranked tokens, while top-p retains a variable number sufficient to cover a specified cumulative probability. In both cases, excluded tokens receive zero probability and the retained tokens are renormalized before sampling. Temperature is separate: it reshapes the probabilities and can therefore alter the size of a top-p nucleus without changing the token ranking. If both filters are enabled, their application order can matter.

---

# Quick Follow-up Answers

**Is top-k always random?** Not when $k=1$; only one candidate survives.

**Is top-p always random?** Not when the nucleus contains just one candidate.

**Does top-p retain exactly $p$ probability mass?** No. It retains at least $p$ before renormalization.

**Does top-k retain a fixed amount of probability mass?** No. It retains a fixed candidate count, but the covered probability mass changes across generation steps.

**Does sampling always select the highest-probability token?** No. Higher probability means a greater *chance* of being chosen, not a guaranteed choice.

**Are candidate sets reused at every step?** No. A new next-token distribution is produced after each generated token.

**Can these techniques guarantee factual accuracy?** No. They only control the sampling distribution.

# Key Memory Lines

- **Greedy:** Choose the top token directly.
- **Top-k:** Keep the best *how many* tokens?
- **Top-p:** Keep enough tokens to cover *how much* probability?
- **Temperature:** How strongly should high-scoring tokens be preferred?
- **Sampling:** Randomly draw a token according to the final normalized distribution.
- **Filter order:** It can change the result, especially when top-p sees a previously filtered distribution.

**Next topics:** 8.10 Beam Search and 8.11 Repetition Penalty.
