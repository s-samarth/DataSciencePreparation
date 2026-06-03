# Law of Large Numbers & CLT

The two halves of the heart of statistics. The law of large numbers says averaging gets you to the truth,
fixing the destination. The central limit theorem says the error around that destination is always normal,
fixing the shape and scale of the approach. Together they justify nearly every estimate and error bar in
practice.

!!! tip "Rapid Recall"
    For i.i.d. data the sample mean is unbiased, $E(\bar X_n)=\mu$, with shrinking variance
    $\mathrm{Var}(\bar X_n)=\sigma^2/n$, so it locks onto $\mu$: the law of large numbers. The strong version
    holds almost surely, the weak version in probability per $n$, and Chebyshev proves the weak law in one
    line. The central limit theorem says $\sqrt{n}(\bar X_n-\mu)/\sigma \to \mathcal{N}(0,1)$ regardless of the
    source distribution, because $\sqrt{n}$ rescaling annihilates every moment past the mean and variance.
    Error shrinks like $1/\sqrt{n}$, so four times the data halves it, and finite variance is required (Cauchy
    breaks it).

## §8 Sample Statistics & the Law of Large Numbers

For i.i.d. $X_1,\dots,X_n$ with true mean $\mu$ and variance $\sigma^2$, the *sample mean* is:

$$\bar X_n = \frac1n\sum_{j=1}^n X_j$$

The deep question of all statistics: $\mu$ is an unobservable fact about the world; $\bar X_n$ is computable
from data. How are they related as data grows?

### Two facts that power everything

| Fact | Statement | Meaning |
|---|---|---|
| Unbiased | $E(\bar X_n) = \mu$ | aimed at the right target |
| Shrinking variance | $\text{Var}(\bar X_n) = \dfrac{\sigma^2}{n}$ | spread goes to 0 as $n\to\infty$ |

The variance fact comes from independence (covariances zero): summing $n$ variances gives $n\sigma^2$, and the
$1/n$ squares to $1/n^2$, leaving $\sigma^2/n$. As $n$ grows the sample mean stops wobbling and locks onto
$\mu$.

### The Law of Large Numbers

$$\bar X_n \to \mu \quad\text{as } n\to\infty$$

| Version | Statement | Why this strength |
|---|---|---|
| Strong | $\bar X_n\to\mu$ with probability 1 (almost surely) | the whole path settles at $\mu$ and never escapes |
| Weak | $P(|\bar X_n-\mu|>c)\to 0$ for any $c>0$ | each fixed large $n$ is probably close, a per-$n$ statement |

**Why weak is weaker:** the weak law promises closeness at each large $n$ separately; the strong law promises
the entire tail of the sequence converges. Strong implies weak, not the reverse. The weak law's proof is one
line, Chebyshev on $\bar X_n$:

$$P(|\bar X_n-\mu|>c) \le \frac{\text{Var}(\bar X_n)}{c^2} = \frac{\sigma^2}{nc^2} \to 0$$

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 620 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Running average of coin flips converging to one half">
  <line x1="60" y1="30" x2="60" y2="210" stroke="#39414f" stroke-width="1"/>
  <line x1="60" y1="210" x2="600" y2="210" stroke="#39414f" stroke-width="1"/>
  <line x1="60" y1="115.0" x2="600" y2="115.0" stroke="#54c7b8" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="606" y="119.0" fill="#54c7b8" font-family="monospace" font-size="11">μ = 0.5</text>
  <path d="M60.0,30.0 L62.7,30.0 L65.4,86.7 L68.1,72.5 L70.9,98.0 L73.6,86.7 L76.3,78.6 L79.0,93.8 L81.7,86.7 L84.4,81.0 L87.1,76.4 L89.8,72.5 L92.6,69.2 L95.3,78.6 L98.0,75.3 L100.7,72.5 L103.4,80.0 L106.1,86.7 L108.8,92.6 L111.6,89.5 L114.3,94.8 L117.0,91.8 L119.7,96.5 L122.4,93.8 L125.1,91.2 L127.8,88.8 L130.6,86.7 L133.3,90.7 L136.0,88.6 L138.7,92.3 L141.4,95.8 L144.1,93.8 L146.8,97.0 L149.5,95.0 L152.3,93.1 L155.0,91.4 L157.7,94.3 L160.4,92.6 L163.1,91.0 L165.8,93.8 L168.5,92.2 L171.3,90.7 L174.0,93.3 L176.7,95.7 L179.4,94.2 L182.1,96.5 L184.8,98.7 L187.5,100.8 L190.3,102.9 L193.0,101.4 L195.7,103.3 L198.4,101.9 L201.1,100.6 L203.8,102.4 L206.5,101.1 L209.2,99.8 L212.0,98.6 L214.7,100.3 L217.4,102.0 L220.1,103.7 L222.8,105.2 L225.5,104.0 L228.2,105.6 L231.0,107.0 L233.7,108.5 L236.4,107.3 L239.1,108.7 L241.8,110.0 L244.5,108.8 L247.2,110.1 L249.9,109.0 L252.7,110.3 L255.4,111.5 L258.1,112.7 L260.8,113.9 L263.5,112.8 L266.2,111.7 L268.9,112.8 L271.7,111.8 L274.4,110.8 L277.1,109.8 L279.8,108.8 L282.5,107.8 L285.2,108.9 L287.9,108.0 L290.7,107.1 L293.4,106.2 L296.1,107.3 L298.8,106.4 L301.5,105.6 L304.2,106.6 L306.9,107.6 L309.6,108.6 L312.4,109.6 L315.1,108.7 L317.8,107.9 L320.5,107.1 L323.2,108.1 L325.9,109.0 L328.6,108.2 L331.4,107.4 L334.1,106.7 L336.8,105.9 L339.5,105.2 L342.2,106.1 L344.9,105.4 L347.6,104.7 L350.4,104.0 L353.1,103.3 L355.8,104.2 L358.5,105.0 L361.2,105.9 L363.9,106.7 L366.6,107.5 L369.3,108.3 L372.1,107.7 L374.8,108.5 L377.5,109.2 L380.2,110.0 L382.9,110.8 L385.6,110.1 L388.3,109.4 L391.1,108.8 L393.8,109.5 L396.5,108.9 L399.2,108.3 L401.9,107.6 L404.6,107.0 L407.3,106.4 L410.1,105.8 L412.8,105.3 L415.5,104.7 L418.2,104.1 L420.9,103.6 L423.6,103.0 L426.3,103.8 L429.0,104.5 L431.8,103.9 L434.5,103.4 L437.2,102.9 L439.9,102.3 L442.6,101.8 L445.3,102.5 L448.0,103.2 L450.8,102.7 L453.5,102.2 L456.2,101.7 L458.9,101.2 L461.6,100.7 L464.3,100.3 L467.0,100.9 L469.7,100.5 L472.5,100.0 L475.2,100.6 L477.9,101.3 L480.6,100.8 L483.3,101.5 L486.0,101.0 L488.7,101.6 L491.5,102.3 L494.2,102.9 L496.9,103.5 L499.6,103.0 L502.3,102.6 L505.0,102.1 L507.7,102.7 L510.5,103.3 L513.2,103.9 L515.9,103.4 L518.6,103.0 L521.3,103.6 L524.0,104.1 L526.7,104.7 L529.4,105.2 L532.2,105.8 L534.9,106.3 L537.6,105.9 L540.3,106.4 L543.0,106.0 L545.7,105.6 L548.4,105.1 L551.2,104.7 L553.9,104.3 L556.6,104.8 L559.3,105.4 L562.0,104.9 L564.7,105.5 L567.4,106.0 L570.2,106.5 L572.9,106.1 L575.6,105.7 L578.3,105.3 L581.0,104.9 L583.7,104.5 L586.4,105.0 L589.1,105.5 L591.9,105.9 L594.6,105.6 L597.3,106.0 L600.0,106.5" fill="none" stroke="#e0a23b" stroke-width="1.8"/>
  <text x="52" y="35" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="end">1.0</text>
  <text x="52" y="213" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="end">0.0</text>
  <text x="330" y="228" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">number of flips n →</text>
</svg>
<figcaption>Running proportion of heads over 200 flips: early jitter gets swamped, converging to μ = 0.5.</figcaption>
</figure>

!!! warning "Trap, no balancing force (gambler's fallacy)"
    10 heads in a row does *not* make tails "due." The streak is not corrected, it gets **swamped** by the
    volume of later flips. The average drifts to $0.5$ because the extra heads become negligible, not because
    the future compensates.

**Where LLN lives.** *Life:* casinos and insurance (tiny per-bet edge gives guaranteed profit over millions of
bets), polling. *ML:* Monte Carlo estimation *is* the LLN; mini-batch SGD (batch gradient is a sample-mean
estimate of the true gradient, variance $\sigma^2/n$); empirical risk minimization (minimizing average loss
approximates expected loss, the reason finite data generalizes); test-set evaluation. LLN says **averaging gets
you to the truth.**

## §9 The Central Limit Theorem

The LLN says $\bar X_n\to\mu$ but nothing about the *shape* of the error. The CLT fills that in:

$$\frac{\sqrt{n}\,(\bar X_n - \mu)}{\sigma} \to \mathcal{N}(0,1) \quad\text{as } n\to\infty \qquad\Longleftrightarrow\qquad \frac{\sum_{j=1}^n X_j - n\mu}{\sqrt{n}\,\sigma} \to \mathcal{N}(0,1)$$

Take the sample mean, subtract the truth, magnify by $\sqrt{n}$, and the distribution of that error is always
a standard normal, **regardless of the source distribution**.

**The magic, two astonishing claims.** **1. Universality.** The $X$'s can be anything, uniform, exponential,
coin flips, lopsided custom. Average enough and the shape is always the same bell. The source's identity is
*erased*. The bell is an **attractor**: sums flow toward it. Why? Adding many independent things averages out
their idiosyncrasies (skew, lumps), leaving only the most generic symmetric shape consistent with a fixed mean
and variance, the max-entropy normal. **2. The $\sqrt{n}$ rate.** Error shrinks like $1/\sqrt{n}$: to **halve**
error you need **4 times** the data. This is the standard error $\sigma/\sqrt{n}$ behind every confidence
interval and A/B sample-size calculation.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 620 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Skewed source distribution becomes a normal bell after averaging">
  <defs><linearGradient id="g2" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0%" stop-color="#54c7b8" stop-opacity="0.55"/><stop offset="100%" stop-color="#54c7b8" stop-opacity="0.04"/>
  </linearGradient></defs>
  <line x1="60" y1="190" x2="300" y2="190" stroke="#39414f" stroke-width="1"/>
  <rect x="70" y="20" width="22" height="170" fill="#e0686f" opacity="0.8"/><rect x="96" y="70" width="22" height="120" fill="#e0686f" opacity="0.8"/><rect x="122" y="110" width="22" height="80" fill="#e0686f" opacity="0.8"/><rect x="148" y="138" width="22" height="52" fill="#e0686f" opacity="0.8"/><rect x="174" y="156" width="22" height="34" fill="#e0686f" opacity="0.8"/><rect x="200" y="168" width="22" height="22" fill="#e0686f" opacity="0.8"/><rect x="226" y="176" width="22" height="14" fill="#e0686f" opacity="0.8"/><rect x="252" y="181" width="22" height="9" fill="#e0686f" opacity="0.8"/><rect x="278" y="184" width="22" height="6" fill="#e0686f" opacity="0.8"/><rect x="304" y="186" width="22" height="4" fill="#e0686f" opacity="0.8"/>
  <text x="175" y="210" fill="#e0686f" font-family="monospace" font-size="11" text-anchor="middle">skewed source X</text>
  <text x="330" y="120" fill="#e0a23b" font-family="monospace" font-size="22" text-anchor="middle">→</text>
  <text x="330" y="140" fill="#7c8593" font-family="monospace" font-size="9" text-anchor="middle">average</text>
  <text x="330" y="151" fill="#7c8593" font-family="monospace" font-size="9" text-anchor="middle">many</text>
  <line x1="360" y1="190" x2="600" y2="190" stroke="#39414f" stroke-width="1"/>
  <path d="M360,189 C410,187 430,175 460,95 C475,55 465,50 480,50 C495,50 485,55 500,95 C530,175 550,187 600,189 L600,190 L360,190 Z" transform="translate(0,0)" fill="url(#g2)" stroke="#54c7b8" stroke-width="2"/>
  <text x="480" y="210" fill="#54c7b8" font-family="monospace" font-size="11" text-anchor="middle">bell of means X̄</text>
</svg>
<figcaption>Any finite-variance source, once averaged, yields a normal distribution of sample means.</figcaption>
</figure>

### LLN versus CLT, complementary zoom levels

|  | What it tells you |
|---|---|
| LLN | the **destination**: $\bar X_n\to\mu$ (where it lands) |
| CLT | the **shape of the approach**: error times $\sqrt{n}$ is normal, width $\sigma$ |

LLN sees $\text{Var}(\bar X_n)=\sigma^2/n\to 0$ (convergence); CLT rescales by $\sqrt{n}$ to hold the variance
at $\sigma^2$ and reveal the normal underneath.

### Why it works, the MGF proof

Standardize ($\mu=0,\sigma=1$), $S_n=\sum X_j$. Independent sums multiply MGFs:

$$E\!\left(e^{tS_n/\sqrt n}\right) = \left[M\!\left(\tfrac{t}{\sqrt n}\right)\right]^n$$

Take logs (the form is $1^\infty$), substitute $y=1/\sqrt n$, apply L'Hopital twice; the two derivatives pull
down $M'(0)=0$ and $M''(0)=1$ (the first two moments):

$$\lim_{n\to\infty} n\ln M\!\left(\tfrac{t}{\sqrt n}\right) = \frac{t^2}{2} \;\Rightarrow\; \text{MGF}\to e^{t^2/2} = \text{MGF of }\mathcal{N}(0,1)$$

**What the math reveals.** The proof used **only the first two moments**. Skew, kurtosis, and all higher
moments got divided away by powers of $\sqrt{n}$ and vanished. *That* is the mathematical reason the source's
identity disappears, only mean and variance survive, and those define a unique normal.

!!! warning "Caveat, finite variance required"
    The Cauchy distribution (no mean or variance) breaks the CLT: averaging Cauchys returns a Cauchy, never a
    normal. "Just average it, it will be normal" hides the finite-variance assumption; heavy-tailed data can
    violate it.

### Normal approximation in practice

A $\text{Bin}(n,p)$ is a sum of $n$ Bernoullis, so for large $n$ it is $\approx\mathcal{N}(np,\,np(1-p))$:

$$P(a\le X\le b) \approx \Phi\!\left(\frac{b-np}{\sqrt{np(1-p)}}\right) - \Phi\!\left(\frac{a-np}{\sqrt{np(1-p)}}\right)$$

| Approximation | When |
|---|---|
| Poisson | $n$ large, $p$ small, $\lambda=np$ moderate (rare events) |
| Normal | $n$ large, $p$ near $\tfrac12$ (symmetric) |

**Continuity correction:** approximating discrete with continuous, use
$P(X=a) = P(a-\tfrac12 < X < a+\tfrac12)$, smear the spike over a unit interval. Worked example, with
$X\sim\text{Bin}(100,0.1)$: $np=10$, $np(1-p)=9$, SD $=3$, and $P(X\le 15)\to P(X<15.5)$:

$$\approx \Phi\!\left(\tfrac{15.5-10}{3}\right) = \Phi(1.83) \approx 0.966$$

A Chebyshev sample-size example, with $\mu=0.5,\ \sigma^2=0.25$, $\text{Var}(\bar X_n)=0.25/n$:

$$P(|\bar X_n-0.5|\ge 0.1) \le \frac{0.25/n}{0.01} = \frac{25}{n}$$
$$\frac{25}{n}\le 0.05 \;\Rightarrow\; n\ge 500$$

**Where CLT lives.** Every confidence interval and error bar (estimate $\pm 1.96\,\sigma/\sqrt n$), all of
hypothesis testing and A/B testing (the test statistic is normal *because* of the CLT, why z-scores and
p-values work without knowing the data distribution), sample-size planning, and the justification for Gaussian
noise assumptions.

> **Interview compression.** LLN: sample mean goes to truth because $\text{Var}(\bar X_n)=\sigma^2/n\to 0$.
> Weak is in probability per $n$; Strong is almost-sure whole path. No balancing force, streaks get swamped.
> CLT: sums of many finite-variance independents are normal regardless of source, because $\sqrt n$ rescaling
> annihilates every moment past mean and variance. Error is about $1/\sqrt n$ (4 times the data to halve it).
> LLN gives the destination; CLT gives the shape and scale of the approach.

## Interview Questions

**Q1: What does the law of large numbers guarantee, and what does it not?**
It guarantees that the sample mean of i.i.d. data converges to the true mean as $n$ grows, because the sample
mean is unbiased with variance $\sigma^2/n$ shrinking to zero. It does not say anything about the shape of the
error at finite $n$, and it provides no balancing force: a run of heads is not corrected, it is simply swamped
by later data.

**Q2: State the central limit theorem and the two things that make it remarkable.**
The CLT says $\sqrt{n}(\bar X_n-\mu)/\sigma$ converges to a standard normal as $n\to\infty$. It is remarkable
first for universality, the limiting shape is normal regardless of the source distribution, and second for the
$1/\sqrt{n}$ rate, which means you need four times the data to halve the standard error.

**Q3: Why does the CLT depend only on the mean and variance of the source?**
In the MGF proof, standardizing and raising the per-term MGF to the $n$-th power, then taking logs, leaves a
limit determined by $M'(0)=0$ and $M''(0)=1$. The $\sqrt{n}$ rescaling divides away every higher moment, so
skew and kurtosis vanish and only the first two moments survive, which is why the source's identity disappears.

**Q4: When does the central limit theorem fail?**
When the source distribution does not have finite variance. The Cauchy distribution has no mean or variance, so
averaging Cauchy variables returns another Cauchy rather than converging to a normal. Heavy-tailed real data can
approach this regime, so "just average it" silently assumes finite moments.
