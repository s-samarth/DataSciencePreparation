# Continuous Random Variables

The leap from discrete to continuous swaps sums for integrals and the probability mass function for a
density, with one conceptual shift: probability becomes area, not height. This page builds the PDF,
continuous expectation, the uniform distribution, LOTUS, and closes with the master cheat sheet that ties
the whole discrete and continuous family together.

!!! tip "Rapid Recall"
    A PDF gives probability as area: $P(a\le X\le b)=\int_a^b f_X(x)\,dx$, the density itself is not a
    probability and can exceed 1. The PDF and CDF are linked by the fundamental theorem of calculus,
    $f_X=F_X'$. Continuous expectation is $\int x f_X(x)\,dx$, the integral version of the discrete sum. The
    uniform distribution has constant density $1/(b-a)$, mean $(a+b)/2$, and variance $(b-a)^2/12$. LOTUS,
    the law of the unconscious statistician, computes $E[g(X)]$ by weighting $g(x)$ with the same density,
    and it is the engine behind variance, moments, moment-generating functions, entropy, and expected loss.

## §13 Probability Density Function (PDF)

$$P(a\le X\le b)=\int_a^b f_X(x)\,dx$$

The probability $X$ falls in $[a,b]$ is the area under the density there. The PDF itself is not a
probability; its *area* is.

**Two properties of a valid PDF.**

$$f_X(x)\ge 0\ \text{for all }x \qquad\text{and}\qquad \int_{-\infty}^{\infty}f_X(x)\,dx=1$$

**The crucial intuition: density times width is approximately probability.**

$$f_X(x_0)\cdot\epsilon \approx P\!\left(X\in\left[x_0-\tfrac{\epsilon}{2},\,x_0+\tfrac{\epsilon}{2}\right]\right)$$

Height times small width is approximately the probability of that little window. The PDF is the *rate* at
which probability accumulates, like physical density (mass per unit length), not mass itself. High density
means values cluster there.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 520 220" role="img" aria-label="PDF with shaded area equal to probability">
      <text x="260" y="16" fill="#5ad1c4" font-family="sans-serif" font-size="12" text-anchor="middle">P(a ≤ X ≤ b) = shaded area under the density</text>
      <line x1="40" y1="180" x2="500" y2="180" stroke="#2a323d"/>
      <defs><clipPath id="clipArea"><rect x="200" y="20" width="110" height="160"/></clipPath></defs>
      <path id="bell" d="M40,178 C140,176 180,60 260,55 C340,60 380,176 480,178" fill="none" stroke="#9d8cff" stroke-width="3"/>
      <path d="M40,178 C140,176 180,60 260,55 C340,60 380,176 480,178 L480,180 L40,180 Z" fill="#9d8cff" opacity="0.18" clip-path="url(#clipArea)"/>
      <line x1="200" y1="180" x2="200" y2="92" stroke="#ffb347" stroke-dasharray="4 3" stroke-width="1.5"/>
      <line x1="310" y1="180" x2="310" y2="92" stroke="#ffb347" stroke-dasharray="4 3" stroke-width="1.5"/>
      <text x="200" y="196" fill="#ffb347" font-size="12" font-family="sans-serif" text-anchor="middle">a</text>
      <text x="310" y="196" fill="#ffb347" font-size="12" font-family="sans-serif" text-anchor="middle">b</text>
    </svg>
<figcaption>For a continuous variable the curve is the density; the probability of landing between a and b is the shaded area, computed by the integral.</figcaption>
</figure>

**PDF and CDF (the bridge).**

$$F_X(x)=\int_{-\infty}^{x}f_X(t)\,dt \qquad\Longleftrightarrow\qquad f_X(x)=F_X'(x)$$

Integrate the PDF to get the CDF; differentiate the CDF to get the PDF (fundamental theorem of calculus).
Also $P(a<X<b)=F_X(b)-F_X(a)$. For continuous variables $<$ and $\le$ agree (single points have zero
probability).

## §14 Continuous Expectation

$$E(X)=\int_{-\infty}^{\infty} x\,f_X(x)\,dx$$

The continuous expectation is the integral analog of the discrete sum:

$$\underbrace{\sum_x x\,P(X=x)}_{\text{discrete}}\ \longleftrightarrow\ \underbrace{\int_{-\infty}^{\infty} x\,f_X(x)\,dx}_{\text{continuous}}$$

## §16 Uniform Distribution

$$f_U(x)=\begin{cases}c & a\le x\le b\\ 0 & \text{otherwise}\end{cases}$$

A PDF integrates to 1, and the area is a rectangle of width $(b-a)$ and height $c$:

$$\int_a^b c\,dx=c(b-a)=1\ \Longrightarrow\ c=\frac{1}{b-a}$$

Density equals 1 over the interval length. Wider interval means shorter rectangle. (The height can exceed 1,
the concrete "PDF is not a probability" case.)

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 420 190" role="img" aria-label="Uniform distribution flat PDF">
      <text x="210" y="16" fill="#5ad1c4" font-family="sans-serif" font-size="12" text-anchor="middle">Unif(a, b) — flat density, height 1/(b−a)</text>
      <line x1="40" y1="150" x2="400" y2="150" stroke="#2a323d"/>
      <line x1="40" y1="40" x2="40" y2="150" stroke="#2a323d"/>
      <rect x="120" y="78" width="180" height="72" fill="#ffb347" opacity="0.22"/>
      <line x1="120" y1="78" x2="300" y2="78" stroke="#ffb347" stroke-width="3"/>
      <line x1="120" y1="78" x2="120" y2="150" stroke="#ffb347" stroke-width="2" stroke-dasharray="4 3"/>
      <line x1="300" y1="78" x2="300" y2="150" stroke="#ffb347" stroke-width="2" stroke-dasharray="4 3"/>
      <line x1="210" y1="150" x2="210" y2="92" stroke="#5ad1c4" stroke-width="1.5" stroke-dasharray="3 3"/>
      <text x="120" y="166" fill="#aeb9c4" font-size="12" font-family="sans-serif" text-anchor="middle">a</text>
      <text x="300" y="166" fill="#aeb9c4" font-size="12" font-family="sans-serif" text-anchor="middle">b</text>
      <text x="210" y="166" fill="#5ad1c4" font-size="11" font-family="sans-serif" text-anchor="middle">(a+b)/2</text>
      <text x="92" y="82" fill="#ffb347" font-size="11" font-family="sans-serif" text-anchor="end">1/(b−a)</text>
    </svg>
<figcaption>A flat rectangle of total area 1. The mean is the midpoint by symmetry; spread grows with the square of the width.</figcaption>
</figure>

**Expected value and variance.**

$$E(U)=\frac{a+b}{2}\qquad\text{(midpoint, by symmetry)}$$

Variance via $E[U^2]-(E[U])^2$:

$$E[U^2]=\int_a^b x^2\frac{1}{b-a}\,dx=\frac{b^3-a^3}{3(b-a)}$$
$$\text{Var}(U)=\frac{(b-a)^2}{12}$$

Variance grows with the *square* of the width: double the interval, quadruple the variance. (This
$\tfrac{(b-a)^2}{12}$ appears in quantization-error analysis.)

Worked example: $U\sim\text{Unif}(0,30)$ minutes.

$$E(U)=\frac{0+30}{2}=15\text{ min}$$
$$\text{Var}(U)=\frac{30^2}{12}=\frac{900}{12}=75$$
$$\text{SD}(U)=\sqrt{75}\approx8.66\text{ min}$$

## §17 LOTUS, the Law of the Unconscious Statistician

$$E[g(X)]=\sum_x g(x)\,P(X=x)\qquad\text{(discrete)}$$
$$E[g(X)]=\int_{-\infty}^{\infty} g(x)\,f_X(x)\,dx\qquad\text{(continuous)}$$

Ordinary expectation has $x$ times the density; LOTUS just swaps the $x$ for $g(x)$. The weights stay
untouched.

- **Variance**: $E[X^2]$ is LOTUS with $g(x)=x^2$, every variance you computed used it.
- **Higher moments** (skew, kurtosis): $E[X^3],E[X^4]$.
- **Moment generating functions** $E[e^{tX}]$ (used to prove the CLT).
- **Expected loss or risk** $E[\text{loss}(X)]$, averaging loss over data is empirical LOTUS.
- **Entropy** $E[-\log p(X)]$ and KL divergence.

A worked density $f_X(x)=2x$ on $[0,1]$:

$$\int_0^1 2x\,dx=x^2\big|_0^1=1\ \checkmark$$
$$E[X]=\int_0^1 x\cdot2x\,dx=\int_0^1 2x^2\,dx=\frac{2}{3}$$
$$E[X^2]=\int_0^1 x^2\cdot2x\,dx=\int_0^1 2x^3\,dx=\frac{1}{2}$$
$$\text{Var}(X)=\frac12-\left(\frac23\right)^2=\frac{9}{18}-\frac{8}{18}=\frac{1}{18}$$

## §18 Master Cheat Sheet

**The discrete distribution family, what you fix and what you count.**

| Distribution | Counts | PMF | Mean | Key trait |
|---|---|---|---|---|
| $\text{Bern}(p)$ | one trial: 0/1 | $p^x(1-p)^{1-x}$ | $p$ | the atom |
| $\text{Bin}(n,p)$ | successes in fixed $n$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | indep., fixed $p$ |
| $\text{HGeom}(w,b,n)$ | whites drawn (no replacement) | $\frac{\binom{w}{k}\binom{b}{n-k}}{\binom{w+b}{n}}$ | $n\frac{w}{w+b}$ | dependent draws |
| $\text{Geom}(p)$ | failures before 1st success | $q^k p$ | $q/p$ | memoryless |
| $\text{NB}(r,p)$ | failures before $r$th success | $\binom{n+r-1}{r-1}p^r q^n$ | $rq/p$ | sum of $r$ geometrics; overdispersion |
| $\text{Pois}(\lambda)$ | rare events per interval | $\frac{e^{-\lambda}\lambda^k}{k!}$ | $\lambda$ | mean = variance = $\lambda$ |
| $\text{Unif}(a,b)$ | random point in $[a,b]$ (continuous) | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | variance $\frac{(b-a)^2}{12}$ |

**Discrete versus continuous, the same machinery with sum and integral.**

| Concept | Discrete | Continuous |
|---|---|---|
| Prob. function | PMF $P(X=x)$ | PDF $f_X(x)$ |
| CDF | $F_X(x)=\sum_{t\le x}P(X=t)$ | $F_X(x)=\int_{-\infty}^{x}f_X(t)\,dt$ |
| Expectation | $\sum_x x\,P(X=x)$ | $\int_{-\infty}^{\infty} x\,f_X(x)\,dx$ |
| Variance | $E[X^2]-(E[X])^2$ | $E[X^2]-(E[X])^2$ (identical both worlds) |
| LOTUS | $\sum_x g(x)P(X=x)$ | $\int_{-\infty}^{\infty} g(x)f_X(x)\,dx$ |

**The five highest-leverage takeaways.**

- **Linearity of expectation ignores dependence.** Decompose any count into indicators, take each mean, sum. This cracked $E[\text{Bin}]=np$, the card problems, local maxima, and the birthday triples.
- **Continuous equals discrete with sum to integral, PMF to PDF.** Expectation, variance, LOTUS all transfer for free. The only genuinely new ideas: "probability is area, not height" and "PDF values are not probabilities."
- **The mean hides spread (and can mislead).** Variance $=E[X^2]-(E[X])^2$ exists for this; St. Petersburg shows the mean alone can be a terrible decision rule.
- **The discrete family is one idea** separated by what you fix and what you count: Bernoulli, binomial, geometric, negative binomial, Poisson.
- **LOTUS is the engine** behind variance, moments, MGFs, entropy, and expected loss, the difference between "I can find the average" and "I can find the average of anything."

## Interview Questions

**Q1: Why can a PDF value exceed 1 when a probability cannot?**
Because a PDF is a density, not a probability: only its integral over an interval is a probability. On a
narrow support the density must be tall to enclose unit area, for example $\text{Unif}(0,0.5)$ has height 2.
Probability is the area under the curve, so the height alone carries no upper bound.

**Q2: How are the PDF and CDF related?**
They are inverse operations under the fundamental theorem of calculus: the CDF is the integral of the PDF up
to $x$, and the PDF is the derivative of the CDF. Consequently $P(a<X<b)=F_X(b)-F_X(a)$, and for continuous
variables strict and non-strict inequalities agree because any single point has zero probability.

**Q3: State LOTUS and name three things it computes.**
LOTUS says $E[g(X)]=\int g(x)f_X(x)\,dx$ (or the analogous sum), computing the expectation of a function of
$X$ by reweighting $g(x)$ with the original density without first finding the distribution of $g(X)$. It is
the engine behind variance (with $g(x)=x^2$), higher moments and moment-generating functions, and expected
loss or entropy.

**Q4: For a uniform distribution on $[a,b]$, give the mean and variance and explain the variance scaling.**
The mean is the midpoint $(a+b)/2$ by symmetry, and the variance is $(b-a)^2/12$. Because variance scales with
the square of the interval width, doubling the width quadruples the variance, a relationship that also
appears in quantization-error analysis.
