# t, z, Chi-Squared & F Tests

The test family all descends from the normal. This page covers the t-distribution that appears when you
estimate the population spread, the z-test it converges to, the chi-squared distribution built from squared
normals, and the F-distribution that compares two variances.

!!! tip "Rapid Recall"
    The t-distribution appears when $\sigma$ is unknown and estimated by $s$, which injects a second source of
    uncertainty and fattens the tails; degrees of freedom index the family, and by about df 30 it matches the
    normal. The t-statistic is $(\bar x-\mu_0)/(s/\sqrt n)$, more honest than z and converging to it as $n$
    grows. A z-test applies when $\sigma$ is known or $n$ is large. Chi-squared is a sum of $k$ squared standard
    normals, nonnegative and right-skewed, used for goodness-of-fit and independence and always right-tailed.
    The F-distribution is a ratio of two scaled chi-squareds, centering near 1, the basis of variance tests and
    ANOVA.

## §5 t-Distribution, Degrees of Freedom & the t-Test

**The problem that created the t-distribution.** Everything with z-scores assumed we **knew the population
$\sigma$**. But in reality you almost never do, if you knew $\sigma$ that precisely you would probably know
$\mu$ too. So you estimate $\sigma$ from the sample with the sample SD, **s**. The catch: **s is itself a noisy
estimate** from the same small sample. You have injected a second source of uncertainty. Naively using the
normal with a noisy s makes you *overconfident*, you would reject too often because the normal's tails are too
thin to account for it.

**Degrees of freedom.** The t-distribution is a *family* of curves indexed by **degrees of freedom (df)**. More
df gives thinner tails gives closer to the normal; by df $\approx 30$ they are nearly identical; at
df $= \infty$ they are the same. More data means s is a more reliable stand-in for $\sigma$ means less extra
uncertainty means convergence to normal.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Normal vs t-distribution with low df">
      <line x1="30" y1="200" x2="670" y2="200" stroke="#2a3140" stroke-width="1.5"/>
      <path d="M30,200 C200,200 250,55 350,55 C450,55 500,200 670,200" fill="none" stroke="#5db0ff" stroke-width="2.5"/>
      <path d="M30,200 C190,195 250,95 350,95 C450,95 510,195 670,200" fill="none" stroke="#e8b04b" stroke-width="2.5" stroke-dasharray="6 4"/>
      <text x="360" y="48" fill="#5db0ff" font-size="13">Normal (z) &mdash; thin tails</text>
      <text x="360" y="120" fill="#e8b04b" font-size="13">t, low df &mdash; fat tails, shorter peak</text>
      <circle cx="80" cy="197" r="3" fill="#e8b04b"/><circle cx="620" cy="197" r="3" fill="#e8b04b"/>
      <text x="80" y="185" fill="#e8b04b" font-size="11" text-anchor="middle">more</text>
      <text x="620" y="185" fill="#e8b04b" font-size="11" text-anchor="middle">more</text>
    </svg>
<figcaption>The t with low df has fatter tails and a shorter peak than the normal, putting more probability far from center.</figcaption>
</figure>

**Finding df by test.**

| Test | df |
|---|---|
| One-sample t-test | $n - 1$ (one mean estimated) |
| Paired t-test | $n - 1$, n = number of pairs (reduce each pair to a difference) |
| Two-sample, pooled (equal var.) | $n_1 + n_2 - 2$ (two means estimated) |
| Two-sample, Welch (unequal var.) | Welch-Satterthwaite approximation (software) |

General principle: **df = (independent observations) minus (parameters estimated from the data)**. Every
parameter you estimate costs one df.

**What the t-test is and why it is better or worse than z.**

$$t = \frac{\bar x - \mu_0}{s / \sqrt n}$$

Just like z, but with the sample's estimated spread s in place of the known $\sigma$.

- **One-sample:** is my mean different from a claimed value?
- **Two-sample (independent):** do two groups' means differ? (A/B testing bread and butter)
- **Paired:** same subjects measured twice (before/after).

**Better (more honest):** the t-test accounts for the uncertainty in estimating $\sigma$. Since you essentially
never know $\sigma$, it is the honest choice; a z-test with estimated $\sigma$ on a small sample gives
artificially small p-values and inflated false positives. **"Worse" (only narrowly):** it is more conservative,
fat tails push critical values out (e.g. about 2.78 at df=4 versus 1.96 for z), so slightly less power. But
that "extra power" of z was an illusion from pretending you had certainty you did not. As n grows the two
become identical, so the cost vanishes. **The t-test is correctly humble, the safer, more defensible choice.**

A worked one-sample t-test, with $s=9$, $n=10$, observed 342 against a claimed 350:

$$SE = \frac{s}{\sqrt n} = \frac{9}{\sqrt{10}} = \frac{9}{3.162} = 2.846$$
$$t = \frac{342 - 350}{2.846} = \frac{-8}{2.846} = -2.811$$

## §6 The z-Test & the z-vs-t Comparison

A **z-test** is a hypothesis test whose statistic follows the standard normal under the null. Its defining
feature: you either know the population $\sigma$, or n is large enough that estimating it is negligible. It
follows a normal shape because of the CLT, and all its machinery (1.96, tail-area p-values) rests on that.

$$z = \frac{\text{observed} - \text{expected under } H_0}{\text{standard error}}$$

| Dimension | z-test | t-test |
|---|---|---|
| Population $\sigma$ | Known (or estimated with very large n) | Unknown, estimated from sample with s |
| Reference distribution | Standard normal $N(0,1)$ | Student's t with df |
| Tail thickness | Thin, fixed | Fat; fatter for smaller df |
| Sample size | Best for large n ($>30$) | For small n; correct for any n |
| Models $\sigma$-estimation uncertainty? | No (assumes $\sigma$ certain) | Yes |
| Critical value (two-sided 0.05) | Always 1.96 | Larger, df-dependent (2.78 at df=4, 2.06 at df=25), to 1.96 as df grows |
| Degrees of freedom | Not needed | Required ($n-1$ for one sample) |
| As n grows | — | Converges to the z-test |
| Risk if misused | z with small n plus estimated $\sigma$ gives overconfident, inflated FP | Essentially none; only slightly conservative |
| Typical use | Large-n proportions; rare known-$\sigma$ cases | Comparing means (one/two-sample, paired), the workhorse |

## §7 Chi-Squared Distribution & Tests

**Where the chi-squared distribution comes from.** Build it from the normal. Take a standard normal Z, square
it, and sum several independent squared standard normals:

$$\chi^2_{k} = Z_1^2 + Z_2^2 + \dots + Z_{k}^2$$

A chi-squared distribution is what you get when you sum $k$ squared standard normals. The parameter $k$ is just
how many you added. This immediately gives its properties:

- **Never negative**, you are summing squares. Lives on $[0, \infty)$.
- **Right-skewed**, most squared normals are small (a normal is usually near 0), but occasionally a large value squares to something huge, stretching a long right tail. Skew is strongest for small $k$.
- **Center grows with $k$**, mean $= k$, because each squared standard normal contributes $E[Z^2] = 1$ on average.
- **Becomes more symmetric (normal-looking) as $k$ grows**, the CLT again: summing many independent things trends toward normal.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Chi-squared distribution shapes for different df">
      <line x1="40" y1="195" x2="670" y2="195" stroke="#2a3140" stroke-width="1.5"/>
      <path d="M40,60 C70,150 110,185 200,190 C400,194 600,195 670,195" fill="none" stroke="#ef6f6f" stroke-width="2.5"/>
      <path d="M40,195 C90,120 130,80 200,80 C320,80 360,190 670,194" fill="none" stroke="#e8b04b" stroke-width="2.5"/>
      <path d="M40,195 C140,193 220,110 320,105 C430,100 520,185 670,192" fill="none" stroke="#5db0ff" stroke-width="2.5"/>
      <text x="70" y="50" fill="#ef6f6f" font-size="12">k = 1 (very skewed)</text>
      <text x="210" y="70" fill="#e8b04b" font-size="12">k = 3</text>
      <text x="340" y="92" fill="#5db0ff" font-size="12">k = 6 (more symmetric)</text>
      <text x="350" y="215" fill="#9aa4b6" font-size="12" text-anchor="middle">&chi;&sup2; value &rarr; (always &ge; 0)</text>
    </svg>
<figcaption>As the degrees of freedom k grow, the chi-squared shifts right and grows more symmetric, lying entirely on the nonnegative axis.</figcaption>
</figure>

**Test 1, goodness-of-fit.** Does observed categorical data match a hypothesized distribution? (A die rolled
120 times: if fair, expect 20 each.)

$$\chi^2 = \sum \frac{(O_{i} - E_{i})^2}{E_{i}}$$

For each category: how far is observed from expected, squared (direction-free, big misses punished), divided by
expected (scale the deviation), then summed. **Why divide by $E_i$?** A deviation of 10 means something
different when you expected 20 versus 2000. Dividing normalizes "how surprising is this miss relative to
scale," which is exactly what turns each term into an approximately squared-standard-normal, so the sum is
chi-squared. **df = categories minus 1** (the counts must sum to n). A large $\chi^2$ means big deviation means
evidence against $H_0$; chi-squared tests are **always right-tailed**.

**Test 2, test of independence.** Are two categorical variables related, or independent? Build a contingency
table; if independent, expected cell counts come from the margins (the multiplication rule
$P(A\cap B)=P(A)P(B)$).

$$E_{ij} = \frac{(\text{row}_{i}\text{ total}) \times (\text{col}_{j}\text{ total})}{\text{grand total}}$$
$$\chi^2 = \sum_{\text{cells}} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \qquad df = (\text{rows} - 1)(\text{cols} - 1)$$

**Parameters and practical caveat.** The chi-squared distribution has exactly **one parameter: df**, which sets
its whole shape. You compute df from the problem's structure (categories minus constraints), then use it to get
the p-value. Inputs to any test: observed counts O, expected counts E (under $H_0$), and df. **Caveat:** the
approximation breaks when expected counts are tiny (rule of thumb: every $E \ge 5$); for tiny counts use an
exact test (Fisher's exact for 2x2).

## §8 The F-Distribution

Chi-squared describes variance-like things (summed squared deviations). To **compare two variances** we take
their ratio:

$$F = \frac{\chi^2_1 / d_1}{\chi^2_2 / d_2}$$

Take two independent chi-squared variables, each divided by its own df, and form their ratio, that ratio is the
F-distribution. It is a ratio of two scaled variances.

- **Never negative** (ratio of non-negatives); lives on $[0, \infty)$.
- **Right-skewed** (inherited from chi-squared).
- **Centers near 1**, if the two variances are equal, the ratio is about 1. $F \gg 1$ means the numerator's variation dwarfs the denominator's: the signal we hunt.
- **Two df parameters** (numerator $d_1$, denominator $d_2$), it is built from two chi-squared pieces, so F-tables are indexed by two numbers.

**When we use it:** directly comparing two variances (an F-test for equality of variances, "do these two
processes have the same consistency?"), and, the dominant use, **ANOVA**, which reframes a question about means
into a ratio of variances. See [Variance Tests, ANOVA & Non-Parametric](variance-anova-nonparam.md).

## Interview Questions

**Q1: Why does the t-distribution exist, and when does it matter most?**
It arises because you rarely know the population standard deviation and must estimate it with the sample $s$,
which is itself noisy and adds a second source of uncertainty. The t has fatter tails than the normal to
account for this, and the effect matters most for small samples; by about 30 degrees of freedom the t is nearly
identical to the normal.

**Q2: When is a z-test appropriate rather than a t-test?**
When the population standard deviation is known, or when the sample is large enough that estimating it is
negligible, so the statistic follows the standard normal under the null. In practice means are usually compared
with t because $\sigma$ is unknown, while z is reserved for large-sample proportions and the rare known-$\sigma$
case. Using z with a small sample and estimated $\sigma$ produces overconfident, inflated false positives.

**Q3: How is the chi-squared distribution constructed, and what are its tests?**
It is the sum of $k$ independent squared standard normals, so it is nonnegative and right-skewed with mean $k$.
Its main tests are goodness-of-fit, asking whether observed category counts match a hypothesized distribution,
and the test of independence on a contingency table, both using $\sum (O-E)^2/E$ and always right-tailed. The
degrees of freedom are categories minus one, or (rows minus one)(columns minus one).

**Q4: What does the F-distribution represent, and where is it used?**
It is the distribution of a ratio of two independent chi-squared variables, each divided by its degrees of
freedom, so it is a ratio of two scaled variances that centers near one when the variances are equal. It is
used to test equality of two variances and, dominantly, in ANOVA, which reframes a question about several means
as a ratio of between-group to within-group variance.
