# More Discrete Distributions

Beyond Bernoulli and binomial, the discrete family extends to waiting-time distributions and rare-event
counts. This page covers the geometric and negative binomial (counting failures before successes), the
Poisson (rare events per interval), the limit that connects binomial to Poisson, and three worked examples
that recur in interviews.

!!! tip "Rapid Recall"
    The geometric distribution counts failures before the first success with PMF $q^k p$ and mean $q/p$, and
    it is the only memoryless discrete distribution. The negative binomial counts failures before the $r$th
    success, a sum of $r$ geometrics with mean $rq/p$. The Poisson counts rare events per interval with PMF
    $e^{-\lambda}\lambda^k/k!$ and the signature fact that mean equals variance equals $\lambda$. When trials
    are many and per-trial probability is tiny with $\lambda=np$ fixed, the binomial converges to Poisson.
    Three transferable moves: linearity beats dependence (local maxima), the mean can mislead (St.
    Petersburg), and the Poisson approximation tames brutal combinatorics (birthday triples).

## §9 Geometric Distribution

This source counts **failures before the first success**. For $X\sim\text{Geom}(p)$:

$$P(X=k)=q^k p,\qquad q=1-p,\quad k=0,1,2,\dots$$

**Reading the PMF.** $k$ failures in a row ($q^k$) then one success ($p$). No binomial coefficient, only one
arrangement works (fail, ..., fail, succeed); the order is forced.

**Valid PMF (geometric series, hence the name).**

$$\sum_{k=0}^{\infty}q^k p=p\cdot\frac{1}{1-q}=p\cdot\frac{1}{p}=1$$

**Mean.**

$$E(X)=\frac{q}{p}=\frac{1-p}{p}$$

Rare success (small $p$) means many failures first. $p=0.5\Rightarrow1$ failure; $p=0.1\Rightarrow9$. A
first-step derivation confirms it:

$$E(X)=0\cdot p+q\,(1+E(X))\ \Rightarrow\ E(X)-qE(X)=q\ \Rightarrow\ E(X)=\frac{q}{1-q}=\frac{q}{p}$$

**Memorylessness, the defining property.**

$$P(X\ge m+n \mid X\ge m)=P(X\ge n)$$

The *only* discrete distribution that is memoryless. Past failures carry no information about the future,
which is why "I am due for a win" is a fallacy and why first-step analysis is so clean.

- **Retry and waiting counts** (network retries, rejection sampling): expected attempts $=q/p$ drives capacity planning.
- **Rejection sampling efficiency**: proposals per accepted sample is geometric; low acceptance $p$ means a slow sampler.
- **RL discounting**: if an episode continues with prob $\gamma$ each step, the horizon is geometric with mean $\gamma/(1-\gamma)$; the discounted return is an expectation over a geometric horizon.
- **Churn and survival**: periods until churn (constant per-period rate); discrete cousin of the exponential.

Worked example: $X\sim\text{Geom}(0.2)$, $q=0.8$.

$$E(X)=\frac{q}{p}=\frac{0.8}{0.2}=4$$
$$P(X=3)=(0.8)^3(0.2)=0.512\times0.2=0.1024$$

## §10 Negative Binomial Distribution

$$P(X=n)=\binom{n+r-1}{r-1}p^r(1-p)^n,\qquad n=0,1,2,\dots$$

written $X\sim\text{NB}(r,p)$.

**Reading the PMF, why the coefficient looks odd.**

- $p^r$, $r$ successes; $(1-p)^n$, $n$ failures.
- $\binom{n+r-1}{r-1}$, the **last trial must be the $r$th success** (that is what stops the process), so it is pinned. Arrange the other $r-1$ successes among the first $n+r-1$ slots.

If the last trial were a failure, the $r$th success already happened earlier and you would have stopped. With
$r=1$, $\binom{n}{0}=1$, recovering the geometric's missing coefficient.

**Mean (via decomposition into geometrics).**

$$X=X_1+\dots+X_r,\quad X_i\sim\text{Geom}(p)\ \text{i.i.d.}$$

Each $X_i$ is the failures between consecutive successes (memorylessness makes them i.i.d.). By linearity:

$$E(X)=r\cdot\frac{q}{p}=\frac{rq}{p}$$

Waiting for $r$ successes takes about $r$ times as long as waiting for 1.

Worked example: failures before the $r$th success, $\text{NB}(r{=}3,p{=}0.7)$, $q=0.3$, want $P(X=2)$:

$$P(X=2)=\binom{2+3-1}{3-1}p^3 q^2=\binom{4}{2}(0.7)^3(0.3)^2$$
$$\binom{4}{2}=6,\quad (0.7)^3=0.343,\quad (0.3)^2=0.09$$
$$P(X=2)=6\times0.343\times0.09=0.18522$$
$$E(X)=\frac{rq}{p}=\frac{3\times0.3}{0.7}\approx1.286$$

## §11 Poisson Distribution

$$P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!},\qquad k=0,1,2,\dots$$

written $X\sim\text{Pois}(\lambda)$. The parameter $\lambda$ is the **rate**: the average number of
occurrences in the interval.

**Valid PMF (Taylor series of $e^\lambda$).**

$$\sum_{k=0}^{\infty}\frac{e^{-\lambda}\lambda^k}{k!}=e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}=e^{-\lambda}e^{\lambda}=1$$

**Mean (and the signature fact).**

$$E(X)=e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^k}{(k-1)!}=\lambda e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^{k-1}}{(k-1)!}=\lambda e^{-\lambda}e^{\lambda}=\lambda$$

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 640 220" role="img" aria-label="Poisson PMF for three rates">
      <text x="320" y="16" fill="#5ad1c4" font-family="sans-serif" font-size="12" text-anchor="middle">Poisson PMF as the rate λ grows (1 → 4 → 10): peak shifts right, spreads out</text>
      <line x1="40" y1="185" x2="610" y2="185" stroke="#2a323d"/>
      <g fill="#ffb347" opacity="0.95">
        <rect x="48" y="118" width="10" height="67"/><rect x="60" y="118" width="10" height="67"/>
        <rect x="72" y="152" width="10" height="33"/><rect x="84" y="174" width="10" height="11"/>
        <rect x="96" y="182" width="10" height="3"/>
      </g>
      <text x="70" y="200" fill="#ffb347" font-size="10" font-family="sans-serif">λ=1</text>
      <g fill="#5ad1c4" opacity="0.9">
        <rect x="200" y="180" width="10" height="5"/><rect x="212" y="160" width="10" height="25"/>
        <rect x="224" y="135" width="10" height="50"/><rect x="236" y="120" width="10" height="65"/>
        <rect x="248" y="120" width="10" height="65"/><rect x="260" y="132" width="10" height="53"/>
        <rect x="272" y="150" width="10" height="35"/><rect x="284" y="167" width="10" height="18"/>
        <rect x="296" y="177" width="10" height="8"/>
      </g>
      <text x="245" y="200" fill="#5ad1c4" font-size="10" font-family="sans-serif">λ=4</text>
      <g fill="#9d8cff" opacity="0.85">
        <rect x="420" y="178" width="9" height="7"/><rect x="431" y="168" width="9" height="17"/>
        <rect x="442" y="155" width="9" height="30"/><rect x="453" y="142" width="9" height="43"/>
        <rect x="464" y="133" width="9" height="52"/><rect x="475" y="129" width="9" height="56"/>
        <rect x="486" y="129" width="9" height="56"/><rect x="497" y="134" width="9" height="51"/>
        <rect x="508" y="143" width="9" height="42"/><rect x="519" y="155" width="9" height="30"/>
        <rect x="530" y="167" width="9" height="18"/><rect x="541" y="176" width="9" height="9"/>
      </g>
      <text x="480" y="200" fill="#9d8cff" font-size="10" font-family="sans-serif">λ=10</text>
    </svg>
<figcaption>As the rate λ grows, the peak sits at λ and the distribution widens (variance equals λ too). For large λ it starts to look bell-shaped, a preview of the normal approximation.</figcaption>
</figure>

- **Poisson regression** (a core GLM) when the *target is a count* (purchases, clicks, claims).
- **Overdispersion bridge**: when variance far exceeds the mean, upgrade to negative binomial.
- **Queueing and systems**: request arrivals (M/M/1 queues), relevant for ML systems design, throughput, tail latency.
- **Anomaly detection**: Poisson baseline for rare-event counts per window; deviations from $\lambda$ flag anomalies.

**Sum of Poissons.**

$$X\sim\text{Pois}(\lambda_1),\ Y\sim\text{Pois}(\lambda_2)\ \text{independent}\ \Rightarrow\ X+Y\sim\text{Pois}(\lambda_1+\lambda_2)$$

Rates just add.

## §12 The Binomial to Poisson Limit

$$X\sim\text{Bin}(n,p),\quad n\to\infty,\ p\to0,\ \lambda=np\ \text{fixed}\ \Longrightarrow\ X\to\text{Pois}(\lambda)$$

Three conditions together: many trials ($n\to\infty$), tiny per-trial probability ($p\to0$), fixed average
($np=\lambda$). Plain English: **rare event, many chances, fixed rate gives Poisson.**

**Why it works (the limit).** Substitute $p=\lambda/n$ into the binomial PMF and take the limit. Three pieces
survive:

$$\binom{n}{k}\approx\frac{n^k}{k!},\qquad p^k=\frac{\lambda^k}{n^k},\qquad \left(1-\frac{\lambda}{n}\right)^{n}\to e^{-\lambda}$$

The $n^k$ cancels, leaving:

$$\lim_{\substack{n\to\infty\\ p\to0}}\binom{n}{k}p^k(1-p)^{n-k}=\frac{\lambda^k}{k!}\,e^{-\lambda}$$

**When to invoke it (the cases).** $n$ large (rule of thumb $n\ge20$, ideally $\ge100$), $p$ small
($p\le0.05$), and $\lambda=np$ moderate.

|  | Binomial | Poisson |
|---|---|---|
| You know | trials $n$ and per-trial $p$ | only the rate $\lambda$ over an interval |
| Support | bounded, $0..n$ | unbounded, $0..\infty$ |
| Mean | $np$ | $\lambda$ |
| Variance | $np(1-p)\to np$ as $p\to0$ | $\lambda$ (equals mean) |

As $p\to0$ the binomial's mean and variance both approach $\lambda$, the approximation literally drags them
together into the Poisson's mean-equals-variance signature.

Worked example: $n=400$ large, $p=0.005$ small, so $\lambda=np=2$:

$$P(X=3)=\frac{e^{-2}\,2^3}{3!}=\frac{8e^{-2}}{6}=\tfrac{8}{6}\times0.1353\approx0.180$$

## §13 Three Famous Worked Examples

These three recur in interviews and each teaches a transferable move: linearity beats dependence, the mean
can mislead, and the Poisson approximation tames brutal combinatorics.

### Expected number of local maxima

The count has overlapping, dependent conditions. Linearity of expectation ignores dependence, so define an
indicator per position and sum.

$$I_j=\mathbf{1}[\text{position }j\text{ is a local max}],\qquad X=\sum_{j=1}^{n} I_j$$

Look at the 3 values at positions $j-1,j,j+1$. Only their *relative rank* matters; by symmetry the largest is
in the center with probability $\tfrac13$.

$$E[I_j]=\tfrac13\quad(\text{there are } n-2 \text{ interior positions})$$

For an endpoint there is only one neighbor; the endpoint is the larger of 2 with probability $\tfrac12$
(2 endpoints).

$$E(X)=(n-2)\cdot\tfrac13+2\cdot\tfrac12=\frac{n-2}{3}+1=\frac{n+1}{3}$$

### St. Petersburg paradox

First success on flip $k$ (counting the success): $X\sim\text{FS}(\tfrac12)$,

$$P(X=k)=\left(\tfrac12\right)^{k-1}\cdot\tfrac12=\frac{1}{2^k}$$
$$E(Y)=\sum_{k=1}^{\infty}2^k\cdot\frac{1}{2^k}$$

The doubling payout and halving probability cancel exactly, every outcome contributes 1.

$$E(Y)=\sum_{k=1}^{\infty}1=\infty$$

### Birthday triples via Poisson

There are $\binom{n}{3}$ triples; $I_{ijk}=1$ if all three share a birthday. Anchor on person $i$'s birthday;
the other two must match it.

$$P(I_{ijk}=1)=\frac{1}{365}\cdot\frac{1}{365}=\frac{1}{365^2}$$
$$\lambda=E(X)=\binom{n}{3}\frac{1}{365^2}$$

Many triples, each rare, weakly dependent, so $X\approx\text{Pois}(\lambda)$.

$$P(X\ge1)=1-P(X=0)=1-\frac{e^{-\lambda}\lambda^0}{0!}=1-e^{-\lambda}$$

## Interview Questions

**Q1: Why is the geometric distribution memoryless, and why does that matter?**
Memorylessness means $P(X\ge m+n\mid X\ge m)=P(X\ge n)$: having already waited through $m$ failures gives no
information about how many more failures remain. The geometric is the only discrete distribution with this
property, which is why "I am due for a win" is a fallacy and why first-step analysis on geometric waiting
times is so clean.

**Q2: What is the signature fact about the Poisson distribution, and when do you reach for it?**
For a Poisson, the mean and the variance both equal the rate $\lambda$. You reach for it when modeling counts
of rare events over an interval, such as arrivals, clicks, or claims, and especially as the limit of a
binomial with many trials and tiny per-trial probability. If you observe variance much larger than the mean,
that signals overdispersion and a move to the negative binomial.

**Q3: State the conditions under which a binomial converges to a Poisson.**
The number of trials goes to infinity, the per-trial probability goes to zero, and the product $\lambda=np$
stays fixed. In practice the approximation is good when $n$ is at least about 20 (ideally 100 or more), $p$ is
at most about 0.05, and $\lambda$ is moderate. As $p\to0$, the binomial's mean and variance both approach
$\lambda$, matching the Poisson's mean-equals-variance signature.

**Q4: How does linearity of expectation solve the expected number of local maxima?**
Define an indicator for each position being a local maximum and sum them. Each interior position is a local
max with probability $1/3$ by symmetry of the three relevant ranks, and each of the two endpoints with
probability $1/2$. Summing the indicator means gives $E(X)=(n-2)/3+1=(n+1)/3$, despite the indicators being
dependent, because linearity never requires independence.
