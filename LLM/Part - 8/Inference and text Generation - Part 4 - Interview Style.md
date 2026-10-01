# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 4 — Interview Style

**Topics:** 8.6 Greedy Decoding; 8.7 Temperature

These are interview answers only. Detailed derivations are kept in the separate Part 4 study file.

---

# 8.6 Greedy Decoding

## Q1. What is greedy decoding?

Greedy decoding chooses the highest-probability token at each generation step:

$$
x_{t+1} = \underset{i}{\operatorname{argmax}}\; P(i \mid x_{\le t})
$$

For example, if the next-token probabilities are A: 0.60, B: 0.25, C: 0.10, and D: 0.05, greedy decoding selects A. The rule involves no random sampling.

## Q2. Do we have to calculate softmax to perform greedy decoding?

No. Softmax preserves the ordering of finite logits, so:

$$
\underset{i}{\operatorname{argmax}}\;z_i
=
\underset{i}{\operatorname{argmax}}\;P(i\mid x_{\le t})
$$

An implementation can select the highest logit directly if normalized probabilities are not otherwise required.

## Q3. Is greedy decoding deterministic?

The rule itself is deterministic: identical logits and the same tie-breaking rule yield the same selected token. Exact end-to-end reproducibility can still be affected by numerical differences across devices and inference implementations, particularly when logits are nearly tied.

## Q4. Does greedy decoding find the most probable complete sequence?

Not necessarily. It optimizes the next token, not the entire continuation.

Suppose:

$$
P(A)=0.55,\qquad P(B)=0.45
$$

and the strongest continuation after each choice has:

$$
P(C\mid A)=0.51,\qquad P(C\mid B)=0.90
$$

Greedy chooses A first, producing the two-token path A–C with:

$$
P(A,C)=0.55\times0.51=0.2805
$$

But the alternative B–C has:

$$
P(B,C)=0.45\times0.90=0.405
$$

The lower-probability first token leads to a higher-probability complete path in this example. Greedy decoding is **locally optimal, not guaranteed globally optimal**.

## Q5. How do you score the probability of an entire generated sequence?

For an output sequence \(y_1,\ldots,y_T\) given prompt \(x\), autoregressive factorization gives:

$$
P(y_1,\ldots,y_T\mid x)
=
\prod_{t=1}^{T} P(y_t\mid x,y_{<t})
$$

Taking logs turns the product into a sum:

$$
\log P(y_1,\ldots,y_T\mid x)
=
\sum_{t=1}^{T}\log P(y_t\mid x,y_{<t})
$$

Log-probability scores avoid numerical underflow when comparing long sequences.

## Q6. What are greedy decoding's strengths and limitations?

It is simple, inexpensive as a selection rule, and deterministic under identical logits and tie handling. Its limitations are lack of diversity, possible repetitive continuation, and inability to recover from a locally optimal choice that leads to a worse overall sequence. Repetition also depends on the model, prompt, and other generation settings.

## Q7. What if multiple tokens share the maximum logit?

The mathematical argmax is tied. The implementation must specify a tie-breaking rule, such as choosing the first maximum token ID. Greedy decoding should not be described as uniquely defined in an exact tie without that rule.

## Complete spoken answer — Greedy decoding

Greedy decoding selects the highest-probability token at every generation step. Since softmax preserves the ranking of logits, we can take argmax directly over the logits. It is simple and deterministic, but it optimizes only the immediate token choice. Because full-sequence probability is a product of conditional token probabilities, choosing the locally most probable token does not guarantee the most probable complete continuation. It also offers no sampling diversity and may follow repetitive high-probability patterns.

---

# 8.7 Temperature

## Q8. What is temperature in LLM generation?

Temperature is a positive parameter that rescales logits before softmax:

$$
P_T(i)
=
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},
\qquad T>0
$$

A temperature below 1 sharpens the next-token distribution; a temperature above 1 flattens it; a temperature of 1 leaves it unchanged.

## Q9. Why does a lower temperature make the distribution sharper?

Dividing by \(T<1\) magnifies logit differences. For example, at \(T=0.5\), every logit is divided by 0.5, doubling the gaps between logits. Softmax then places more probability mass on the highest-scoring tokens.

## Q10. Why does a higher temperature make the distribution flatter?

Dividing by \(T>1\) reduces differences between logits. At \(T=2\), every gap is halved, and lower-scoring tokens gain relative probability mass after softmax.

## Q11. Can you demonstrate temperature numerically?

Take the logits:

$$
z=[4,2,1]
$$

Approximate softmax probabilities are:

| Temperature | Token A | Token B | Token C |
|---|---:|---:|---:|
| \(T=0.5\) | 0.9796 | 0.0179 | 0.0024 |
| \(T=1\) | 0.8438 | 0.1142 | 0.0420 |
| \(T=2\) | 0.6285 | 0.2312 | 0.1402 |

The ranking stays A, B, C; only the concentration changes.

## Q12. What mathematical expression best explains temperature's effect?

For tokens \(i\) and \(j\), the probability ratio is:

$$
\frac{P_T(i)}{P_T(j)}
=
e^{(z_i-z_j)/T}
$$

Thus:

$$
\log\left(\frac{P_T(i)}{P_T(j)}\right)
=
\frac{z_i-z_j}{T}
$$

Temperature directly controls how strongly a given logit gap influences relative probability.

## Q13. Does temperature change the highest-ranked token?

Not for positive temperature. Dividing logits by the same positive value preserves their ranking:

$$
\underset{i}{\operatorname{argmax}}\;z_i
=
\underset{i}{\operatorname{argmax}}\;\frac{z_i}{T}
$$

Therefore, **positive temperature followed by greedy argmax selects the same token**. Temperature becomes meaningful when the transformed distribution is sampled.

## Q14. Does temperature itself introduce randomness?

No. Temperature transforms probabilities; sampling makes a random choice from those probabilities:

$$
x_{t+1}
\sim
\operatorname{Categorical}
\left(P_T(1),\ldots,P_T(|\mathcal{V}|)\right)
$$

Temperature and sampling are distinct operations.

## Q15. What happens as temperature approaches zero?

When there is a unique highest logit:

$$
T\rightarrow0^+
\quad\Longrightarrow\quad
P_T(\text{top token})\rightarrow1
$$

Thus sampling approaches greedy behavior. Exactly \(T=0\) is undefined in the temperature-softmax formula, although some inference APIs interpret it as a special request for greedy generation. If several logits tie for the maximum, the limiting probability is split equally among those tied tokens rather than assigned entirely to one.

## Q16. What happens as temperature approaches infinity?

For finite, unmasked logits, dividing by an increasingly large positive value makes them approach zero. Softmax then approaches a uniform distribution across the available tokens:

$$
T\rightarrow\infty
\quad\Longrightarrow\quad
P_T(i)\rightarrow\frac{1}{|\mathcal{V}|}
$$

Here \(|\mathcal{V}|\) denotes the number of available finite-logit tokens.

## Q17. Does raising temperature give the model new information?

No. Temperature neither updates model weights nor adds knowledge or vocabulary items. It changes how likely the sampler is to choose tokens already represented in the model's output distribution. Higher temperature can produce more varied choices, including implausible ones; it does not guarantee creativity or factual accuracy.

## Q18. How do implementations compute temperature-scaled softmax safely?

Subtract the maximum logit before exponentiating:

$$
z_{\max}=\max_j z_j
$$

$$
P_T(i)
=
\frac{\exp((z_i-z_{\max})/T)}
{\sum_j\exp((z_j-z_{\max})/T)}
$$

Subtracting a common constant leaves the softmax distribution unchanged while reducing overflow risk.

## Q19. Do masked tokens become available at high temperature?

No. A forbidden token with effective logit \(-\infty\) remains at \(-\infty\) after division by any positive finite temperature, so its softmax probability remains zero.

## Q20. Is a lower temperature always better?

No. Lower temperature produces a more concentrated and usually more repeatable sampled continuation; higher temperature enables greater exploration but also increases the chance of low-probability choices. The useful setting depends on the model, task, and other decoding controls.

## Complete spoken answer — Temperature

Temperature rescales the model's logits before softmax. A value below one amplifies differences and concentrates probability on top-ranked tokens; a value above one reduces differences and flattens the distribution. Temperature does not itself perform sampling or introduce randomness. Positive temperature preserves token ranking, so applying temperature and then taking greedy argmax yields the same token. Its main purpose is to control how strongly the model prefers high-scoring tokens when sampling.

---

# Combined interview answer

Greedy decoding and temperature operate at different stages of token selection. Greedy decoding takes the highest-scoring token at each step, making a deterministic local choice. Temperature rescales logits before softmax, making the resulting distribution sharper or flatter; a sampler can then draw from it. Because positive temperature preserves logit ranking, applying temperature alone cannot change a greedy argmax choice. Greedy's main limitation is local optimality, while temperature's main tradeoff is between concentrated and exploratory sampling.

# Quick memory lines

- **Greedy:** choose the highest-probability token now; not guaranteed to maximize full-sequence probability.
- **Temperature:** reshape the distribution, not the vocabulary or model knowledge.
- **Low \(T\):** sharper; **high \(T\):** flatter.
- **Temperature + greedy:** the same argmax for every positive \(T\).
- **Temperature + sampling:** changes how often alternative tokens are selected.

**Next study block:** 8.8 Top-k Sampling and 8.9 Top-p / Nucleus Sampling.
