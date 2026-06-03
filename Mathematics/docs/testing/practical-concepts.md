# Practical Concepts

A cluster of ideas that surround every test in practice: how correlated data shrinks your effective sample,
the difference between spread of data and spread of an estimate, bootstrapping for any statistic, the proof
behind the $n-1$ in sample variance, the logit that powers logistic regression, and the two limit theorems
that make inference possible at all.

!!! tip "Rapid Recall"
    Effective sample size is how many independent, equally-informative points you really have, shrunk by
    correlation or unequal weighting. Standard deviation describes the spread of data; standard error,
    $\text{SD}/\sqrt n$, describes the spread of a statistic. Bootstrapping resamples your data with
    replacement to estimate the uncertainty of any statistic with no distributional assumption. Sample variance
    divides by $n-1$ because using $\bar x$ in place of $\mu$ makes the squared deviations too small, and the
    expectation works out to $(n-1)\sigma^2$. Log-odds, $\ln(p/(1-p))$, is unbounded and symmetric, which is
    why logistic regression models it linearly. The LLN makes estimation possible and the CLT makes p-values
    and confidence intervals computable.

## §13 Sample Size vs. Effective Sample Size

**Sample size (n):** how many data points you collected. **Effective sample size (ESS):** how many
*independent, equally-informative* points you *effectively* have, often far fewer. Not every point carries a
full unit of fresh information.

**Why they diverge.**

- **Correlation between observations.** Measuring one patient's BP 100 times in an hour is not 100 independent facts, readings are highly correlated. Time-series, repeated measures, clustered data (students within a classroom) all suffer this. More correlation gives lower ESS.
- **Unequal weighting.** If a few responses are heavily weighted (one stands in for 50 people), a handful dominate and the effective sample shrinks.
- **Importance sampling / MCMC.** Samples drawn from the wrong distribution and reweighted; ESS measures how many "good" samples you really have. A common formula is $\text{ESS} = (\sum w_i)^2 / \sum w_i^2$, which equals n only when all weights are equal and collapses toward 1 when one weight dominates.

## §14 Standard Deviation vs. Standard Error

**Standard deviation (SD)** describes the spread of *individual data points*, "how much do observations vary
around the mean?" It is the square root of variance, a property of the data. **Standard error (SE)** describes
the spread of a *statistic* (usually the mean) across hypothetical repeated samples, "if I redid the experiment
many times, how much would my sample mean bounce around?" A property of your estimate, not the data.

$$SE = \frac{SD}{\sqrt n}$$

The standard error is the standard deviation shrunk by the square root of how many points you averaged over.

## §15 Bootstrapping

**Core idea:** estimate the uncertainty of *any* statistic by **resampling your own data, with replacement,
many times.** No formula, no distributional assumption, let the data simulate "what other samples might have
looked like."

**Mechanics.**

- You have a sample of size n.
- Draw n points from it **with replacement** (some appear multiple times, others not at all), one "bootstrap sample."
- Compute your statistic (mean, median, correlation, 90th percentile) on it.
- Repeat thousands of times, giving thousands of values of the statistic.
- Their spread estimates the standard error; their percentiles give a CI (2.5th and 97.5th give a 95% CI).

**Why "with replacement"?** Without it you would just get your original dataset back every time, no variation.
Replacement creates the variation that mimics genuine resampling. **Why valuable:** works for statistics with
*no clean SE formula* (median, percentiles, ratios, correlations, model metrics) and makes *no distributional
assumption*.

**Using bootstrapping to calculate a p-value.** Simulate the null by resampling. Example, do two groups have
different means?

$$p = \frac{\#\,\text{resampled statistics} \ge \text{observed}}{\text{total resamples}}$$

Out of all the fake differences generated assuming no effect, what fraction were as big as the real one?

## §16 Population vs. Sample Variance, Why n−1

$$\sigma^2 = \frac{1}{N} \sum (x_{i} - \mu)^2 \qquad\text{(population, you know the true mean } \mu)$$
$$s^2 = \frac{1}{n - 1} \sum (x_{i} - \bar x)^2 \qquad\text{(sample, you must use } \bar x)$$

**The proof (why exactly $n-1$).** We want an **unbiased** estimator, its average over all possible samples
equals the true $\sigma^2$. Start from the identity linking deviations-from-$\bar x$ to deviations-from-$\mu$:

$$\sum (x_{i} - \bar x)^2 = \sum (x_{i} - \mu)^2 - n(\bar x - \mu)^2$$

In words: scatter around the sample mean equals scatter around the true mean *minus* a positive term
$n(\bar x-\mu)^2$, exactly the "shrinkage" from using $\bar x$ instead of $\mu$. (It comes from expanding
$(x_i-\mu)^2 = ((x_i-\bar x)+(\bar x-\mu))^2$ and summing; the cross term vanishes because
$\sum(x_i-\bar x) = 0$.) Take expectations of both sides:

$$E\left[ \sum(x_{i} - \mu)^2 \right] = n\sigma^2$$
$$E\left[ n(\bar x - \mu)^2 \right] = n \cdot \text{Var}(\bar x) = n \cdot \frac{\sigma^2}{n} = \sigma^2$$
$$E\left[ \sum(x_{i} - \bar x)^2 \right] = n\sigma^2 - \sigma^2 = (n - 1)\sigma^2$$

The expected sum of squared deviations around $\bar x$ is exactly **$(n-1)\sigma^2$**, not $n\sigma^2$.
Therefore:

- **Divide by n:** $E[s^2] = (n-1)\sigma^2/n < \sigma^2$, biased downward, underestimates (confirming the intuition).
- **Divide by $n-1$:** $E[s^2] = (n-1)\sigma^2/(n-1) = \sigma^2$, unbiased.

## §17 Odds & Log-Odds

$$\text{odds} = \frac{p}{1 - p}$$

Odds are the ratio of "happens" to "does not happen." $p = 0.75$ gives odds 3 ("3 to 1"); $p = 0.5$ gives odds
1.

$$\text{logit}(p) = \ln\!\left( \frac{p}{1 - p} \right)$$

Log-odds (the logit) is the natural log of the odds.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ranges of probability, odds, log-odds">
      <text x="20" y="40" fill="#5db0ff" font-size="13" font-family="monospace">Probability</text>
      <line x1="160" y1="35" x2="420" y2="35" stroke="#5db0ff" stroke-width="3"/>
      <circle cx="160" cy="35" r="4" fill="#5db0ff"/><circle cx="420" cy="35" r="4" fill="#5db0ff"/>
      <text x="150" y="22" fill="#9aa4b6" font-size="11">0</text><text x="415" y="22" fill="#9aa4b6" font-size="11">1</text>
      <text x="440" y="40" fill="#9aa4b6" font-size="11">bounded both ends</text>
      <text x="20" y="100" fill="#e8b04b" font-size="13" font-family="monospace">Odds</text>
      <line x1="160" y1="95" x2="630" y2="95" stroke="#e8b04b" stroke-width="3"/>
      <circle cx="160" cy="95" r="4" fill="#e8b04b"/>
      <text x="150" y="82" fill="#9aa4b6" font-size="11">0</text>
      <text x="290" y="82" fill="#9aa4b6" font-size="11">1 (neutral, off-center)</text>
      <text x="600" y="82" fill="#9aa4b6" font-size="11">&infin;</text>
      <text x="640" y="100" fill="#9aa4b6" font-size="11">asymmetric</text>
      <text x="20" y="165" fill="#5ad19a" font-size="13" font-family="monospace">Log-odds</text>
      <line x1="120" y1="160" x2="660" y2="160" stroke="#5ad19a" stroke-width="3"/>
      <text x="110" y="147" fill="#9aa4b6" font-size="11">&minus;&infin;</text>
      <line x1="390" y1="150" x2="390" y2="170" stroke="#fff" stroke-width="1.5"/>
      <text x="370" y="147" fill="#9aa4b6" font-size="11">0 (p=0.5)</text>
      <text x="650" y="147" fill="#9aa4b6" font-size="11">+&infin;</text>
      <text x="120" y="190" fill="#9aa4b6" font-size="11">unbounded &amp; symmetric around 0</text>
    </svg>
<figcaption>Probability is bounded in [0,1], odds run [0, infinity) asymmetrically, and log-odds span all reals symmetrically around 0.</figcaption>
</figure>

**Why take the log, what it solves.**

- **Unbounded range solves the modeling problem.** Linear models output $w\cdot x + b \in (-\infty,\infty)$, but probability is trapped in $[0,1]$, a mismatch. Log-odds also spans all reals, so you cleanly set **log-odds $= w\cdot x + b$**. *This is exactly logistic regression*: model log-odds linearly, then convert back to probability via the sigmoid (the sigmoid is the inverse of the logit). The log-odds is the bridge that lets a linear model produce a valid probability.
- **Symmetry around zero is interpretable.** 0 gives $p=0.5$; positive means more likely than not, negative means less, magnitude means strength. $\ln(2) \approx +0.69$ and $\ln(0.5) \approx -0.69$ are exact mirrors, "twice as likely" and "half as likely" become symmetric.
- **Addition replaces multiplication (the deep reason).** Logs turn multiplication into addition. Odds combine multiplicatively (awkward); in log-odds space, *evidence adds up*. Each feature contributes additively, so logistic-regression coefficients are interpretable ("this feature adds 0.7 to the log-odds"), and it connects to information theory, log-likelihoods, and "bits of evidence" (Naive Bayes sums log-probabilities for the same reason).
- **Numerical stability (engineering reason).** Probabilities can be tiny ($10^{-300}$); multiplying many underflows to zero. Log space turns products into sums of manageable numbers. ML libraries compute log-probabilities and log-odds internally, cross-entropy, log-loss, softmax-with-logits. The "logits" a network outputs *are* unnormalized log-odds; sigmoid/softmax converts back only at the very end.

## §18 CLT & LLN in Inferential Statistics

**Law of Large Numbers, why estimation works at all.** LLN: as the sample grows, the sample mean converges to
the true population mean. It is the reason **estimation is even possible**, the guarantee that with enough data
your estimates home in on truth rather than wandering off. It underwrites:

- **Consistency of estimators**, MLE, sample mean, etc. converge to the right answer with enough data.
- **The shrinking standard error**, $SE = SD/\sqrt n \to 0$ as n grows is LLN in action.
- **Bootstrapping's validity**, a large sample faithfully represents the population, so resampling from it approximates resampling from the population.

But LLN alone is not enough, it says the mean *converges*, not *how it is distributed* or *how uncertain* it is
along the way. That is CLT's job.

**Central Limit Theorem, why we can compute p-values and CIs.** CLT: regardless of the population's shape, the
distribution of the *sample mean* becomes approximately **normal** as n grows, centered at the true mean with
SD $= \sigma/\sqrt n$. The single most important theorem in inference, because:

- **It is why z/t-tests work on non-normal data.** Raw data can be wildly skewed, but the sample *mean* is approximately normal for decent n. Since tests about means use the distribution *of the mean*, they are valid even when the data is not normal, you can t-test skewed income data.
- **It is why CIs have their form.** "mean $\pm 1.96 \times$ SE" comes directly from CLT: because the mean is normal, 95% of the time it lands within 1.96 SE of the truth. No CLT, no 1.96.
- **It is why the SE formula is meaningful**, CLT pins the mean's spread as exactly $\sigma/\sqrt n$, giving the scale for all tests and intervals.
- **It is the root of the chi-squared, t, F family.** Each was built from normal variables; the reason those normals appear in real data is CLT. So CLT is the root of the entire distribution family tree.

See [Law of Large Numbers & CLT](../advanced/lln-clt.md) for the full development of both theorems.

## Interview Questions

**Q1: What is effective sample size and why is it often smaller than n?**
Effective sample size is the number of independent, equally-informative observations you really have, which can
be far below the raw count. Correlation between observations, such as repeated measures or clustered data,
reduces it, as does unequal weighting where a few points dominate. In importance sampling it is
$(\sum w_i)^2/\sum w_i^2$, equal to n only when all weights match.

**Q2: Distinguish standard deviation from standard error.**
Standard deviation measures the spread of individual data points around the mean, a property of the data
itself. Standard error measures the spread of a statistic, usually the mean, across hypothetical repeated
samples, a property of your estimate. They are linked by $SE=SD/\sqrt n$, so averaging more points shrinks the
standard error while the standard deviation stays put.

**Q3: Why does the sample variance divide by $n-1$ rather than $n$?**
Because using the sample mean $\bar x$ in place of the unknown true mean $\mu$ makes the squared deviations
systematically too small, since $\bar x$ minimizes that sum. Taking expectations, the sum of squared deviations
around $\bar x$ averages to $(n-1)\sigma^2$, so dividing by $n-1$ yields an unbiased estimate of $\sigma^2$,
while dividing by $n$ underestimates it.

**Q4: Why does logistic regression model the log-odds linearly instead of the probability?**
Because probability is bounded in $[0,1]$ while a linear function $w\cdot x+b$ ranges over all reals, a
mismatch. The log-odds also spans all reals and is symmetric around zero, so setting log-odds equal to the
linear predictor is clean, interpretable (evidence adds), and numerically stable, with the sigmoid converting
back to a probability at the end.
