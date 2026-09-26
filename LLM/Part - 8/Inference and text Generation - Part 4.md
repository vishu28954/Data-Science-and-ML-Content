# Part 8 — Inference and Text Generation

# Inference and Text Generation — Part 4

## 8.6 Greedy Decoding
## 8.7 Temperature

This note continues directly from Part 3.

At the end of the KV Cache lesson, we had reached this point:

\`\`\`text
Prompt is processed
        ↓
KV Cache makes repeated decode efficient
        ↓
Transformer produces logits for the next token
        ↓
But which token should actually be chosen?
\`\`\`

This creates a new problem.

Suppose the vocabulary contains many possible next tokens and the model produces a score for every one of them.

What should we do with those scores?

The simplest answer is:

> Choose the token with the highest probability.

That gives us **Greedy Decoding**.

But greedy decoding creates another question:

> Do we always want the highest-probability token, or should generation sometimes explore other plausible tokens?

That leads to **Temperature**.

The story of this part is:

\`\`\`text
Transformer produces logits
        ↓
Convert logits into a next-token distribution
        ↓
Simplest choice:
pick the maximum
        ↓
GREEDY DECODING
        ↓
Deterministic and simple
        ↓
But local maximum is not necessarily the best complete sequence
        ↓
What if we want controlled randomness?
        ↓
Reshape the probability distribution
        ↓
TEMPERATURE
        ↓
Low temperature → sharper distribution
High temperature → flatter distribution
        ↓
But temperature still does not decide which subset of tokens is allowed
        ↓
Next topics:
Top-k and Top-p sampling
\`\`\`

---

# 8.6 Greedy Decoding

## Question 1 — Where does token selection begin?

At the end of a Transformer forward pass, the model produces a hidden state for the latest known position.

Suppose that hidden state is:

$$h_t$$

The language-model head converts it into a vector of logits:

$$z_{t+1} = W_{\text{LM}}h_t + b$$

If the vocabulary contains:

$$|\mathcal{V}|$$

tokens, then:

$$z_{t+1} \in \mathbb{R}^{|\mathcal{V}|}$$

So there is one logit for every vocabulary token.

For example:

\`\`\`text
Token        Logit
------------------
"Paris"       5.1
"London"      2.7
"France"      1.8
"."           0.9
...
\`\`\`

The logits are not probabilities yet.

To obtain probabilities, we can apply softmax:

$$
P(i \mid x_{\le t})
=
\frac{e^{z_i}}
{\sum_{j=1}^{|\mathcal{V}|} e^{z_j}}
$$

Now we have a probability distribution over possible next tokens.

The decoding algorithm decides what to do with that distribution.

---

## Question 2 — What is greedy decoding?

Greedy decoding chooses the token with the highest probability at the current generation step.

Mathematically:

$$
x_{t+1}
=
\underset{i}{\operatorname{argmax}}
\;
P(i \mid x_{\le t})
$$

In plain English:

> Look at every possible next token and choose the one with the largest probability.

Suppose:

\`\`\`text
Token       Probability
-----------------------
A              0.60
B              0.25
C              0.10
D              0.05
\`\`\`

Greedy decoding chooses:

$$x_{t+1}=A$$

because:

$$0.60$$

is the largest probability.

There is no random sampling involved.

---

## Question 3 — Do we actually need to compute softmax for greedy decoding?

Not necessarily.

Suppose two logits satisfy:

$$z_i > z_j$$

Softmax is monotonic with respect to the logits, so:

$$
P(i \mid x_{\le t})
>
P(j \mid x_{\le t})
$$

Therefore:

$$
\underset{i}{\operatorname{argmax}}
\;
z_i
=
\underset{i}{\operatorname{argmax}}
\;
P(i \mid x_{\le t})
$$

So if the only goal is to find the highest-scoring token, the implementation can choose the maximum logit directly.

For example:

\`\`\`text
Logits:
A → 5.0
B → 3.0
C → 1.0

Highest logit:
A

Softmax would preserve the same ranking,
so greedy decoding still chooses A.
\`\`\`

### Why is this useful?

Softmax is necessary when we need actual normalized probabilities for sampling or interpretation.

But for pure argmax selection, normalization is mathematically unnecessary.

---

## Question 4 — Is greedy decoding deterministic?

Under the same model state and the same logits, yes.

If:

$$
z_A > z_B > z_C
$$

then greedy decoding always selects:

$$A$$

There is no random draw.

So conceptually:

\`\`\`text
same context
+
same model
+
same logits
+
greedy decoding
        ↓
same selected token
\`\`\`

### Important practical nuance

Exact reproducibility across different hardware, kernels, numerical precisions, or serving implementations is a separate systems issue.

Tiny numerical differences can matter if two logits are extremely close.

But the decoding rule itself is deterministic.

---

## Question 5 — Does greedy decoding choose the most probable complete sequence?

No.

This is one of the most important limitations of greedy decoding.

Greedy decoding chooses:

> the best token **right now**

It does not search all possible future sequences.

Suppose the first-step probabilities are:

$$
P(A)=0.55
$$

and:

$$
P(B)=0.45
$$

Greedy chooses:

$$A$$

because:

$$0.55 > 0.45$$

Now suppose that after choosing $A$:

$$
P(C \mid A)=0.51
$$

while after choosing $B$:

$$
P(C \mid B)=0.90
$$

The greedy two-token sequence is:

$$A \rightarrow C$$

with probability:

$$
P(A,C)
=
P(A)P(C \mid A)
$$

Therefore:

$$
P(A,C)
=
0.55 \times 0.51
=
0.2805
$$

But the alternative sequence:

$$B \rightarrow C$$

has probability:

$$
P(B,C)
=
P(B)P(C \mid B)
$$

which gives:

$$
P(B,C)
=
0.45 \times 0.90
=
0.405
$$

So:

$$
0.405 > 0.2805
$$

Even though:

$$
P(B) < P(A)
$$

at the first step.

### What happened?

Greedy decoding made the locally best choice:

$$A$$

but that choice led to a weaker continuation.

The initially less probable token:

$$B$$

led to a more probable complete two-token path.

This is why:

> **Locally highest probability does not necessarily imply globally highest sequence probability.**

---

## Question 6 — Why does sequence probability involve multiplication?

Autoregressive language models factor a sequence probability into conditional next-token probabilities.

For an output sequence:

$$
y_1, y_2, \ldots, y_T
$$

conditioned on prompt $x$, we have:

$$
P(y_1, \ldots, y_T \mid x)
=
\prod_{t=1}^{T}
P(y_t \mid x, y_{<t})
$$

So the probability of the full sequence depends on every token decision.

For numerical stability, sequence scores are often expressed using log probabilities:

$$
\log P(y_1, \ldots, y_T \mid x)
=
\sum_{t=1}^{T}
\log P(y_t \mid x, y_{<t})
$$

because multiplying many probabilities can produce extremely small numbers.

### Why do we need this mathematics?

It explains exactly why a locally optimal token does not guarantee a globally optimal sequence.

Greedy decoding optimizes one factor at a time.

It does not optimize the entire product over all possible future paths.

---

## Question 7 — Can greedy decoding become repetitive?

It can.

Suppose a particular continuation reinforces a pattern such as:

\`\`\`text
very very very very ...
\`\`\`

At each step, the repeated token may continue to receive the highest probability.

Greedy decoding has no randomness that would naturally move generation away from that local pattern.

However, repetition is not caused by greedy decoding alone.

It also depends on:

- the model,
- the prompt,
- learned probability distributions,
- repetition penalties,
- stop conditions,
- other generation settings.

So the accurate statement is:

> Greedy decoding can make deterministic repetitive patterns harder to escape when the repeated continuation remains the local maximum.

We will study repetition penalties later in 8.11.

---

## Question 8 — What happens if two tokens have exactly the same maximum score?

Suppose:

$$
z_A=z_B
$$

and both are larger than every other logit.

Mathematically, the maximum is tied.

The abstract greedy rule does not uniquely determine whether $A$ or $B$ should be selected.

A concrete implementation needs a tie-breaking rule.

For example, it may choose:

- the first maximum index,
- the lowest token ID,
- or another deterministic implementation-specific choice.

Exact ties are uncommon with ordinary floating-point model outputs, but they are an important mathematical edge case.

---

## Question 9 — How do constraints affect greedy decoding?

The model may initially produce logits for the entire vocabulary:

$$
z \in \mathbb{R}^{|\mathcal{V}|}
$$

But a generation system can modify or mask some logits before selection.

For example, if a token is forbidden, its effective logit might be set conceptually to:

$$-\infty$$

Then it cannot win the argmax.

So greedy decoding really selects the highest-scoring token **after any active decoding constraints or logit transformations have been applied**.

This becomes important for:

- structured generation,
- stop-token handling,
- vocabulary restrictions,
- repetition penalties.

---

## Question 10 — What are the main strengths and weaknesses of greedy decoding?

### Strengths

\`\`\`text
simple
deterministic
no sampling randomness
cheap token-selection rule
easy to reproduce conceptually
\`\`\`

### Weaknesses

\`\`\`text
locally optimal only
can miss better future sequences
no diversity
can become trapped in repetitive continuations
often too rigid for open-ended generation
\`\`\`

The key mental model is:

> **Greedy decoding asks only: “Which token is best right now?”**

It does not ask:

> “Which entire future sequence will ultimately be best?”

---

# 8.6 Mental Model

Suppose the next-token distribution is:

\`\`\`text
A → 0.60
B → 0.25
C → 0.10
D → 0.05
\`\`\`

Greedy decoding simply does:

\`\`\`text
find maximum
        ↓
A = 0.60
        ↓
select A
        ↓
append A to context
        ↓
run next decode step
        ↓
repeat
\`\`\`

Mathematically:

$$
x_{t+1}
=
\underset{i}{\operatorname{argmax}}
\;
P(i \mid x_{\le t})
$$

Memory line:

> **Greedy decoding chooses the locally highest-probability token at every step; it does not search for the globally highest-probability sequence.**

---

# Bridge from Greedy Decoding to Temperature

Greedy decoding gives us one extreme:

\`\`\`text
always choose the maximum
→ completely deterministic token selection
\`\`\`

But imagine the model produces:

\`\`\`text
Token A → 0.40
Token B → 0.35
Token C → 0.20
Token D → 0.05
\`\`\`

Greedy decoding always chooses:

$$A$$

even though:

$$B$$

is also highly plausible.

For open-ended generation, we may want the model to sometimes explore other plausible continuations.

But before sampling, we need a way to answer:

> How strongly should the model prefer high-scoring tokens over lower-scoring ones?

That is what **temperature** controls.

---

# 8.7 Temperature

## Question 1 — What is temperature?

Temperature is a parameter that rescales logits before softmax.

If the original logits are:

$$
z_1, z_2, \ldots, z_{|\mathcal{V}|}
$$

then temperature-adjusted probabilities are:

$$
P_T(i)
=
\frac{
e^{z_i/T}
}{
\sum_{j=1}^{|\mathcal{V}|} e^{z_j/T}
}
$$

where:

$$T>0$$

The important operation is:

$$
z_i
\rightarrow
\frac{z_i}{T}
$$

before softmax.

Temperature therefore changes the **shape of the probability distribution**.

It does not change the model weights.

---

## Question 2 — Why do we divide logits by temperature?

Because division gives the conventional behavior we want.

If:

$$T<1$$

then:

$$
\frac{z_i}{T}
$$

magnifies differences between logits.

For example, with:

$$T=0.5$$

we get:

$$
\frac{z_i}{0.5}
=
2z_i
$$

So logit gaps become larger.

Softmax becomes sharper.

If:

$$T>1$$

the differences shrink.

For:

$$T=2$$

we get:

$$
\frac{z_i}{2}
$$

so logits move closer together.

Softmax becomes flatter.

Therefore:

\`\`\`text
T < 1
→ enlarge logit differences
→ sharper probability distribution

T = 1
→ original softmax distribution

T > 1
→ shrink logit differences
→ flatter probability distribution
\`\`\`

---

## Question 3 — Can we see temperature numerically?

Yes.

Suppose the logits are:

$$
z=[4,2,1]
$$

### Case 1 — Temperature equals 1

With:

$$T=1$$

the probabilities are approximately:

$$
P_{T=1}
\approx
[0.8438,\;0.1142,\;0.0420]
$$

The first token is strongly preferred.

### Case 2 — Temperature equals 0.5

Now:

$$T=0.5$$

so the scaled logits are:

$$
\frac{z}{T}
=
[8,4,2]
$$

The probabilities become approximately:

$$
P_{T=0.5}
\approx
[0.9796,\;0.0179,\;0.0024]
$$

The distribution becomes extremely sharp.

### Case 3 — Temperature equals 2

Now:

$$T=2$$

so:

$$
\frac{z}{T}
=
[2,1,0.5]
$$

The probabilities become approximately:

$$
P_{T=2}
\approx
[0.6285,\;0.2312,\;0.1402]
$$

The distribution is much flatter.

So:

| Temperature | Token 1 | Token 2 | Token 3 | Behavior |
|---|---:|---:|---:|---|
| $T=0.5$ | 0.9796 | 0.0179 | 0.0024 | very sharp |
| $T=1$ | 0.8438 | 0.1142 | 0.0420 | original |
| $T=2$ | 0.6285 | 0.2312 | 0.1402 | flatter |

This shows exactly what temperature changes.

---

## Question 4 — What is the cleanest mathematical explanation of temperature?

Look at the probability ratio between two tokens $i$ and $j$.

Under temperature:

$$
\frac{P_T(i)}{P_T(j)}
=
e^{(z_i-z_j)/T}
$$

Taking the natural logarithm:

$$
\log
\left(
\frac{P_T(i)}{P_T(j)}
\right)
=
\frac{z_i-z_j}{T}
$$

This equation is extremely useful.

It tells us that temperature directly controls the strength of the preference implied by a logit difference.

Suppose:

$$
z_i-z_j=2
$$

### At temperature 0.5

$$
\frac{z_i-z_j}{T}
=
\frac{2}{0.5}
=
4
$$

So:

$$
\frac{P_T(i)}{P_T(j)}
=
e^4
\approx
54.6
$$

Token $i$ is roughly 54.6 times as probable as token $j$.

### At temperature 1

$$
\frac{P_T(i)}{P_T(j)}
=
e^2
\approx
7.39
$$

### At temperature 2

$$
\frac{P_T(i)}{P_T(j)}
=
e^1
\approx
2.72
$$

So:

\`\`\`text
same model
same logits
same ranking

but

lower T
→ stronger preference

higher T
→ weaker preference
\`\`\`

This is the mathematical heart of temperature.

---

## Question 5 — Does temperature change which token has the highest logit?

For any positive temperature:

$$T>0$$

the answer is no.

If:

$$z_i>z_j$$

then:

$$
\frac{z_i}{T}
>
\frac{z_j}{T}
$$

because division by a positive number preserves ordering.

Therefore the rank of the logits does not change.

So:

$$
\underset{i}{\operatorname{argmax}}
\;
z_i
=
\underset{i}{\operatorname{argmax}}
\;
\frac{z_i}{T}
$$

for:

$$T>0$$

This gives us an extremely important consequence.

> **If we apply positive temperature and then still use greedy argmax, the selected token does not change.**

Temperature becomes useful when the adjusted distribution is used for **sampling**.

---

## Question 6 — Does temperature itself add randomness?

No.

Temperature only transforms the probability distribution.

It does not perform the random draw.

After temperature, a sampling algorithm might draw:

$$
x_{t+1}
\sim
\text{Categorical}
\left(
P_T(1), P_T(2), \ldots, P_T(|\mathcal{V}|)
\right)
$$

The randomness comes from sampling from this categorical distribution.

Temperature controls how concentrated that distribution is.

So:

\`\`\`text
temperature
→ reshapes probabilities

sampling
→ actually chooses a random token according to those probabilities
\`\`\`

This distinction is essential.

---

## Question 7 — What happens as temperature approaches zero?

Consider:

$$
P_T(i)
=
\frac{
e^{z_i/T}
}{
\sum_j e^{z_j/T}
}
$$

As:

$$
T \rightarrow 0^+
$$

differences between logits are magnified enormously.

If one token has a unique maximum logit, its probability approaches:

$$1$$

while the probabilities of other tokens approach:

$$0$$

So in the limit:

\`\`\`text
T → 0+
→ distribution approaches greedy behavior
\`\`\`

### Important mathematical edge case

Exactly:

$$T=0$$

cannot be substituted into:

$$
\frac{z_i}{T}
$$

because division by zero is undefined.

Some APIs may treat temperature zero as a special instruction for deterministic or greedy generation, but mathematically the temperature-softmax formula requires:

$$T>0$$

---

## Question 8 — What if two tokens tie for the maximum as temperature approaches zero?

Suppose:

$$
z_A=z_B
$$

and both are strictly greater than every other logit.

As:

$$
T \rightarrow 0^+
$$

probability mass concentrates on the tied maxima.

Because the two maximum logits are equal, their exponentiated values remain equal.

In the idealized two-way tie:

$$
P_T(A)
\rightarrow
\frac{1}{2}
$$

and:

$$
P_T(B)
\rightarrow
\frac{1}{2}
$$

while lower-logit tokens approach zero probability.

So the statement:

> “Temperature approaching zero always produces one token with probability one”

assumes a unique maximum.

---

## Question 9 — What happens as temperature becomes extremely large?

As:

$$
T \rightarrow \infty
$$

for finite logits:

$$
\frac{z_i}{T}
\rightarrow
0
$$

Therefore:

$$
e^{z_i/T}
\rightarrow
1
$$

for every finite unmasked token.

If there are:

$$|\mathcal{V}|$$

available tokens, then the distribution approaches:

$$
P_T(i)
\rightarrow
\frac{1}{|\mathcal{V}|}
$$

So very high temperature approaches a uniform distribution over the available finite-logit tokens.

Conceptually:

\`\`\`text
very high T
→ model preferences become weak
→ even low-scoring tokens receive substantial probability
→ generation can become incoherent
\`\`\`

---

## Question 10 — Does temperature create new knowledge or new tokens?

No.

Temperature does not change:

- model weights,
- vocabulary,
- learned facts,
- hidden representations already computed.

It only modifies the relative probabilities used during token selection.

A token that already exists in the vocabulary can become more or less likely.

Temperature cannot create information the model never learned.

So:

> **Higher temperature means more exploration of the model's existing distribution, not more knowledge.**

---

## Question 11 — Why can low temperature improve consistency but reduce diversity?

Suppose the model assigns:

\`\`\`text
A → high probability
B → moderately high probability
C → somewhat plausible
D → low probability
\`\`\`

At low temperature, the probability gap between $A$ and the others increases.

So repeated generations are much more likely to choose $A$.

This makes outputs:

- more consistent,
- more predictable,
- less diverse.

But it can also make the model repeatedly follow the same local path.

So low temperature is not automatically equivalent to higher quality.

It simply concentrates probability mass around already high-scoring tokens.

---

## Question 12 — Why can high temperature increase diversity but also errors?

At high temperature, probability differences shrink.

Tokens that originally had relatively low logits receive more probability mass.

That increases the chance of exploring alternative continuations.

But it also increases the chance of selecting weak or implausible tokens.

So:

\`\`\`text
higher temperature
→ more diversity
→ more exploration
→ greater chance of low-probability choices
\`\`\`

Again, temperature is not a direct factuality control.

It changes the distribution from which token choices are made.

---

## Question 13 — What is the relationship between temperature and the original probabilities?

Suppose the original probabilities at temperature 1 are:

$$
P_1(i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

Temperature transformation can also be understood conceptually as raising the original probabilities to the power:

$$
\frac{1}{T}
$$

and renormalizing:

$$
P_T(i)
=
\frac{
P_1(i)^{1/T}
}{
\sum_j P_1(j)^{1/T}
}
$$

This gives another intuitive view.

If:

$$T<1$$

then:

$$
\frac{1}{T}>1
$$

which exaggerates differences.

If:

$$T>1$$

then:

$$
\frac{1}{T}<1
$$

which compresses differences.

This is mathematically equivalent to scaling logits before softmax.

---

## Question 14 — How is temperature implemented safely with softmax?

Directly computing:

$$
e^{z_i/T}
$$

can overflow if logits are large.

A numerically stable form subtracts the maximum logit before exponentiation.

Let:

$$
z_{\max}
=
\max_j z_j
$$

Then:

$$
P_T(i)
=
\frac{
\exp\left((z_i-z_{\max})/T\right)
}{
\sum_j
\exp\left((z_j-z_{\max})/T\right)
}
$$

Subtracting the same constant from every logit does not change softmax probabilities.

Why?

Because:

$$
\frac{
e^{(z_i-c)/T}
}{
\sum_j e^{(z_j-c)/T}
}
=
\frac{
e^{z_i/T}e^{-c/T}
}{
e^{-c/T}\sum_j e^{z_j/T}
}
$$

The common factor:

$$
e^{-c/T}
$$

cancels.

This improves numerical stability without changing the mathematical distribution.

---

## Question 15 — What happens to masked tokens under temperature?

Suppose a token has been assigned an effective logit:

$$-\infty$$

because it is forbidden.

For any positive finite temperature:

$$
\frac{-\infty}{T}
=
-\infty
$$

and:

$$
e^{-\infty}
=
0
$$

So its probability remains zero.

Temperature does not make an explicitly masked token available again.

This matters when temperature is combined with:

- vocabulary masks,
- structured decoding,
- generation constraints.

---

## Question 16 — What if temperature is negative?

Standard temperature-based decoding assumes:

$$T>0$$

A negative temperature would reverse the ordering of logits because division by a negative number reverses inequalities.

That would make low-logit tokens become high-scoring relative to high-logit tokens.

This is not the usual meaning of temperature in language-model decoding and is generally not offered as a normal generation setting.

So for ordinary LLM decoding, think:

$$T>0$$

---

## Question 17 — Is there a universally best temperature?

No.

The useful temperature depends on:

- the task,
- model,
- prompt,
- other sampling parameters,
- desired diversity,
- desired consistency.

A factual extraction task may benefit from a concentrated distribution.

Creative generation may benefit from more exploration.

But temperature alone does not guarantee correctness, creativity, or safety.

It only controls how strongly token probability differences influence sampling.

---

# 8.7 Mental Model

Suppose the model produces:

$$
z=[4,2,1]
$$

Think of temperature as a **contrast control for logits**.

\`\`\`text
Low temperature

[4, 2, 1]
     ↓ divide by 0.5
[8, 4, 2]
     ↓
differences become larger
     ↓
softmax becomes sharper
\`\`\`

\`\`\`text
High temperature

[4, 2, 1]
     ↓ divide by 2
[2, 1, 0.5]
     ↓
differences become smaller
     ↓
softmax becomes flatter
\`\`\`

The mathematical core is:

$$
P_T(i)
=
\frac{
e^{z_i/T}
}{
\sum_j e^{z_j/T}
}
$$

and the deepest intuition is captured by:

$$
\log
\left(
\frac{P_T(i)}{P_T(j)}
\right)
=
\frac{z_i-z_j}{T}
$$

Memory line:

> **Temperature does not decide the token; it controls how strongly the probability distribution prefers high-logit tokens before sampling.**

---

# Greedy Decoding vs Temperature — Final Comparison

| Property | Greedy Decoding | Temperature |
|---|---|---|
| Main purpose | Select the highest-scoring token | Reshape the token probability distribution |
| Randomness | None in the rule itself | None by itself; randomness comes from sampling |
| Core mathematics | $\underset{i}{\operatorname{argmax}}\;P(i \mid x_{\le t})$ | $P_T(i)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$ |
| Effect on ranking | Selects the top-ranked token | Positive $T$ preserves ranking |
| Low value behavior | Not applicable | Sharper distribution |
| High value behavior | Not applicable | Flatter distribution |
| Main limitation | Local optimum only | Does not restrict the candidate set |

---

# Why Does This Naturally Lead to Top-k and Top-p?

Temperature can flatten a probability distribution.

But consider a vocabulary containing tens of thousands of tokens.

At high temperature, many very low-probability tokens can receive more probability mass.

This creates the next question:

> Do we really want to sample from the entire vocabulary every time?

One solution is:

> Keep only the top $k$ highest-probability candidates.

That gives us:

\`\`\`text
8.8 Top-k Sampling
\`\`\`

But a fixed number of candidates creates another problem.

Sometimes the model is extremely confident and only two tokens are plausible.

Other times twenty tokens may be plausible.

So instead of keeping a fixed number, we can keep the smallest set whose cumulative probability reaches a threshold.

That gives us:

\`\`\`text
8.9 Top-p / Nucleus Sampling
\`\`\`

---

# Part 4 Summary

The story of Part 4 is:

\`\`\`text
Transformer produces next-token logits
        ↓
Need to choose a token
        ↓
GREEDY DECODING
        ↓
choose the local maximum
        ↓
simple and deterministic
        ↓
but local maximum does not guarantee the best sequence
        ↓
allow alternative plausible tokens
        ↓
TEMPERATURE
        ↓
rescale logits before softmax
        ↓
low T → sharper
high T → flatter
        ↓
temperature reshapes probabilities
but does not itself perform sampling
        ↓
Next:
Top-k and Top-p
\`\`\`

Core equations:

### Greedy selection

$$
x_{t+1}
=
\underset{i}{\operatorname{argmax}}
\;
P(i \mid x_{\le t})
$$

### Sequence probability

$$
P(y_1, \ldots, y_T \mid x)
=
\prod_{t=1}^{T}
P(y_t \mid x, y_{<t})
$$

### Temperature-scaled softmax

$$
P_T(i)
=
\frac{
e^{z_i/T}
}{
\sum_j e^{z_j/T}
}
$$

### Temperature log-odds relationship

$$
\log
\left(
\frac{P_T(i)}{P_T(j)}
\right)
=
\frac{z_i-z_j}{T}
$$

### Temperature limits

For a unique maximum logit:

$$
T \rightarrow 0^+
\quad
\Rightarrow
\quad
\text{distribution approaches greedy selection}
$$

For finite unmasked logits:

$$
T \rightarrow \infty
\quad
\Rightarrow
\quad
\text{distribution approaches uniform}
$$

The next detailed-study block is:

\`\`\`text
8.8 Top-k Sampling
8.9 Top-p / Nucleus Sampling
\`\`\`
