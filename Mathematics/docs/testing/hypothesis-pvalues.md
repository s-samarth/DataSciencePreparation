# Hypothesis Testing & p-values

Hypothesis testing is a formal, disciplined procedure for separating real signal from random noise, so you
do not fool yourself. This page builds the courtroom analogy, explains why the boring null is always the
default, and defines what a p-value actually is (and is not).

!!! tip "Rapid Recall"
    A hypothesis test asks whether an observed effect is real or just lucky sampling. The null $H_0$ is the
    specific, boring default ("no effect," "p = 0.5") and carries the equality, because you need a precise
    value to compute probabilities; whatever you want to prove goes in $H_1$. You never accept the null, only
    fail to reject it, because absence of evidence is not proof. The p-value is the probability of data this
    extreme or more *given the null is true*, a tail area, and it is *not* the probability the null is true.
    Reject when $p<\alpha$, with $\alpha$ set in advance as the false-positive rate you will tolerate.

## §1 Hypothesis Testing

**The core problem, why we need it.** You see something in your data: a drug seems to lower blood pressure, a
new button seems to lift conversions, a coin seems to land heads too often. The question that haunts every
analysis is the same: **is this effect real, or did I just get lucky with my sample?**

You can never measure the whole population, so you take a noisy sample. Flip a fair coin 10 times and 7 heads
is just randomness; flip it 1000 times and 700 heads is genuinely off. Hypothesis testing is a **formal,
disciplined procedure for separating real signal from random noise**, it stops you from fooling yourself.
Humans are catastrophically bad at this intuitively (hot hands, stocks "due" for a rebound, a redesign that
"feels" better). The test forces a number onto how surprised we should be, given that nothing interesting is
actually happening.

| Courtroom | Hypothesis Test |
|---|---|
| Presumed innocent | Null hypothesis $H_0$ (nothing is happening) |
| The evidence | Your data |
| Beyond reasonable doubt | Significance threshold $\alpha$ |
| "Guilty" verdict | Reject $H_0$ |
| "Not guilty" verdict | Fail to reject $H_0$ |

### Null and alternate hypotheses

**The null hypothesis ($H_0$)** is the boring, skeptical default: "no effect," "no difference," "the coin is
fair." **The alternate hypothesis ($H_1$ or $H_a$)** is the interesting claim you want to demonstrate: "the
drug works," "the coin is biased." Why is the boring statement the default? Two deep reasons:

- **The null is specific; the alternate is vague.** "The coin is fair" means exactly $p = 0.5$, one precise number you can build a probability model on. "The coin is biased" could be 0.51 or 0.9; you cannot compute probabilities from a vague claim. We need the precise statement to do the math, so the precise statement becomes the working assumption.
- **The burden of proof sits on the new claim.** Extraordinary claims need extraordinary evidence. The data must drag you, kicking and screaming, away from disbelief.

**How to frame the null.** The null always contains the **equality** ($=$, $\le$, or $\ge$), it needs a
specific value to assume so probabilities are computable. Whatever you set out to prove goes in $H_1$.

| Claim | $H_0$ | $H_1$ |
|---|---|---|
| New method raises avg above 70 | $\mu \le 70$ | $\mu > 70$ |
| Coin is biased (either way) | $p = 0.5$ | $p \neq 0.5$ |
| Drug changes BP (unknown direction) | $\mu = 0$ | $\mu \neq 0$ |

### The language: "reject" versus "fail to reject"

Two concrete reasons we can never "accept" the null:

- **Small samples hide real effects.** A drug that truly lowers BP by a tiny amount, tested on 5 people, gives noisy data, so you fail to reject "no effect," but the effect is real. You simply lacked the *power* to see it.
- **$H_0$ is a single exact point.** "$\mu = 100$" or "effect is exactly zero" is almost never *precisely* true (maybe it is 0.0001). You can never gather enough evidence to confirm a value to infinite precision; you can only fail to find it different.

## §2 The p-value

$$\text{p-value} = P(\text{ data this extreme or more} \mid H_0 \text{ is true})$$

The p-value is the probability of the data (or more extreme), **given the null is true**, not the probability
the null is true. The vertical bar means "given." We live inside the assumption that $H_0$ holds and ask how
weird the data looks from in there. Small p means "if nothing interesting were happening I would almost never
see this." Large p means "even in a boring world this shows up all the time."

**Why "or more extreme"?** For continuous data the probability of any single exact value is essentially zero,
so that alone is useless. Deeper: what makes 8/10 heads surprising is not the exact 8, it is landing far out in
the tail; 9 or 10 would be even more surprising, more evidence against fairness. So "how extreme" must include
everything *at least* as extreme, the whole tail. That tail area *is* the p-value.

**How we use it.**

- Set **$\alpha$** in advance (usually 0.05). This means: "I am willing to be fooled by noise 5% of the time."
- Compute the p-value.
- $p < \alpha$ means reject $H_0$. $p \ge \alpha$ means fail to reject $H_0$.

If the universe is genuinely boring and $\alpha = 0.05$, you will still wrongly reject about 1 in 20 times, the
price of business (this is the Type I error rate).

**Is the p-value only for normal distributions? No.** The p-value is a general concept: a tail probability
under whatever distribution the null implies. Coin flips give binomial; rare counts give Poisson; variances
give F; categorical gives chi-squared; small-sample means give t; large-sample means give normal. The
distribution changes; the logic never does.

### The normal case, the z-score

$$z = \frac{\text{observed} - \text{expected under } H_0}{\text{standard error}}$$

How many standard deviations away from the boring-world center did I land? Further out means more surprising
means smaller p. Reference points worth memorizing:

- $|z| = 1.645$ gives one-tailed $p = 0.05$
- $|z| = 1.96$ gives two-tailed $p = 0.05$ (*the* famous number)
- $|z| = 2.576$ gives two-tailed $p = 0.01$

**One-tailed versus two-tailed:** a directional $H_1$ ($\mu > 100$) looks at one tail; a non-directional $H_1$
($\mu \neq 100$) counts both tails, so the p-value is doubled (by symmetry, $2\times$ one-tail area).

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Normal curve with two-tailed rejection regions">
      <defs>
        <linearGradient id="bell" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#5db0ff" stop-opacity="0.35"/>
          <stop offset="1" stop-color="#5db0ff" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      <line x1="40" y1="210" x2="660" y2="210" stroke="#2a3140" stroke-width="1.5"/>
      <path d="M40,210 C180,210 230,40 350,40 C470,40 520,210 660,210 Z" fill="url(#bell)" stroke="#5db0ff" stroke-width="2"/>
      <path d="M40,210 C120,210 150,150 175,120 L175,210 Z" fill="#ef6f6f" fill-opacity="0.5"/>
      <path d="M660,210 C580,210 550,150 525,120 L525,210 Z" fill="#ef6f6f" fill-opacity="0.5"/>
      <line x1="175" y1="40" x2="175" y2="218" stroke="#ef6f6f" stroke-width="1.5" stroke-dasharray="4 3"/>
      <line x1="525" y1="40" x2="525" y2="218" stroke="#ef6f6f" stroke-width="1.5" stroke-dasharray="4 3"/>
      <line x1="350" y1="40" x2="350" y2="218" stroke="#e8b04b" stroke-width="1.5" stroke-dasharray="2 4"/>
      <text x="350" y="234" fill="#9aa4b6" font-size="13" text-anchor="middle" font-family="monospace">0 (null center)</text>
      <text x="160" y="234" fill="#ef6f6f" font-size="13" text-anchor="middle" font-family="monospace">&minus;1.96</text>
      <text x="540" y="234" fill="#ef6f6f" font-size="13" text-anchor="middle" font-family="monospace">+1.96</text>
      <text x="110" y="150" fill="#ef6f6f" font-size="12" text-anchor="middle">reject</text>
      <text x="590" y="150" fill="#ef6f6f" font-size="12" text-anchor="middle">reject</text>
      <text x="350" y="150" fill="#5db0ff" font-size="13" text-anchor="middle">fail to reject (95%)</text>
    </svg>
<figcaption>The two-tailed rejection regions at plus or minus 1.96 each hold 2.5% of the area; the central 95% is where you fail to reject.</figcaption>
</figure>

A worked z-test for a mean, with $\sigma=60$, $n=50$, observed 980 against an expected 1000:

$$SE = \frac{\sigma}{\sqrt n} = \frac{60}{\sqrt{50}} = \frac{60}{7.071} = 8.485$$
$$z = \frac{980 - 1000}{8.485} = \frac{-20}{8.485} = -2.357$$

And a z-test for a proportion, testing $\hat p = 0.15$ against $p_0 = 0.12$ with $n=400$:

$$SE = \sqrt{\frac{p_0(1-p_0)}{n}} = \sqrt{\frac{0.12 \times 0.88}{400}} = \sqrt{0.000264} = 0.01625$$
$$z = \frac{0.15 - 0.12}{0.01625} = \frac{0.03}{0.01625} = 1.846$$

## Interview Questions

**Q1: Why is the null hypothesis always the boring, specific statement?**
Because you need a precise value to build a probability model, and only the null ("no effect," "p equals 0.5")
supplies one, while the alternative is vague and uncomputable. The null also reflects the burden of proof: the
new claim must be demonstrated by evidence strong enough to drag you away from skepticism, so the skeptical
statement is the working assumption.

**Q2: What exactly is a p-value, and what is it not?**
A p-value is the probability of observing data at least as extreme as yours, computed under the assumption
that the null is true, a tail area. It is not the probability that the null is true, nor the probability your
result happened by chance in any broader sense. A small p means the data would be surprising in a boring world,
which is evidence against the null.

**Q3: Why do we say "fail to reject" rather than "accept" the null?**
Because failing to find an effect is not proof there is none. A small or noisy sample can lack the power to
detect a real effect, and the null is an exact point that is almost never precisely true, so you can never
confirm it to infinite precision. You can only fail to find a difference, which is a weak conclusion.

**Q4: When do you double the p-value, and why?**
For a two-tailed test, where the alternative is non-directional like $\mu\neq\mu_0$. Surprising evidence can
fall in either tail, so you count both, which by symmetry doubles the one-tail area. A directional alternative
like $\mu>\mu_0$ uses only one tail and is not doubled.
