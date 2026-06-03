# Universality of the Uniform & the Normal

Two pillars of continuous probability: the uniform distribution is the universal raw material from which any
distribution can be simulated, and the normal distribution is the inevitable bell that emerges from sums of
many small effects. This page derives both from first principles.

!!! tip "Rapid Recall"
    Universality of the uniform has two directions: simulation, where $X=F^{-1}(U)$ turns a uniform into any
    distribution, and flattening, where $F(X)$ turns any continuous variable back into a uniform. That is why
    a single uniform RNG can simulate anything. The standard normal PDF $\frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ has a
    shape forced by symmetry (the square) and decay (the negative exponent), with the $\sqrt{2\pi}$ pinned by
    the Gaussian integral. The z-score $Z=(X-\mu)/\sigma$ standardizes, variances add even on subtraction, and
    the empirical rule gives 68, 95, and 99.7 percent within one, two, and three standard deviations.

## §1 Universality of the Uniform

The uniform is the universal raw material of all randomness. Two claims, inverses of each other.
**(A) Simulation:** any distribution with CDF $F$ can be generated from a uniform by plugging it into the
inverse CDF. **(B) Flattening:** feed any continuous variable through its *own* CDF and the output is always
$\text{Uniform}(0,1)$.

**Intuition, the y-axis is always uniform.** Picture the CDF $F$ climbing from 0 to 1. Drop a point uniformly
on the **y-axis**, read across to the curve, then down to the x-axis. Steep parts (dense probability)
compress a big y-range into a small x-range, so many draws land there; flat parts (low density) spread thin.
Reading the curve backwards ($F^{-1}$) turns a uniform into the target distribution. Every continuous
distribution is just a **warped uniform**, and the warp is the inverse CDF.

**Why it matters.** Your computer's RNG produces exactly one thing: $\text{Uniform}(0,1)$. Every sample from a
normal, exponential, or custom distribution is built on this. It is the bridge from "I have a uniform RNG" to
"I can simulate anything."

### Statement A, simulation direction

With $U \sim \text{Uniform}(0,1)$ and $F$ strictly increasing, define $X = F^{-1}(U)$. Then $X \sim F$:

$$P(X \le x) = P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$$

The last equality holds because for a uniform, $P(U \le c) = c$ (probability equals length of the interval
$[0,c]$), and here $c = F(x)$ which is always in $[0,1]$.

### Statement B, flattening direction

If $X \sim F$, then $F(X) \sim \text{Uniform}(0,1)$:

$$P(F(X) \le x) = P(X \le F^{-1}(x)) = F(F^{-1}(x)) = x$$

A CDF equal to $x$ on $[0,1]$ *is* the uniform. **Example of Statement B:** let $F(x)=1-e^{-x}$ (the
Exponential(1) CDF). Feed an exponential variable through its own CDF and it flattens to a uniform:
$1 - e^{-X} \sim \text{Uniform}(0,1)$.

By Statement A, $X = F^{-1}(U)$. Find $F^{-1}$. Set $y = 1 - e^{-x}$ and solve for $x$: $e^{-x} = 1 - y$, so
$-x = \ln(1-y)$ and $x = -\ln(1-y)$.

$$X = -\ln(1-U) \sim \text{Exponential}(1)$$

> **Interview compression.** Inverse CDF turns a uniform into anything (simulation); a variable through its
> own CDF turns into a uniform (flattening). That is why a single uniform RNG is all a computer needs.

## §2 The Normal (Gaussian) Distribution

The bell curve, the most important distribution in probability. The standard normal $\mathcal{N}(0,1)$ has
PDF $f(z) = c\,e^{-z^2/2}$, with $c$ a normalizing constant forcing the area to 1. The general
$\mathcal{N}(\mu,\sigma^2)$ is the same bell shifted to center $\mu$ and stretched to spread $\sigma$.

**Why this shape (build it from scratch).** We want a bell: peaked at center, symmetric, decaying fast.
**Symmetry** forces $z$ to appear as $z^2$ (sign-independent). **Decay** forces a negative exponent. So the
core is $e^{-z^2/2}$: equals 1 at the peak, depends on $z^2$ (symmetric), crashes toward 0 in the tails. The
$\tfrac12$ is a scaling choice that makes variance come out to exactly 1. The constant $c$ is **forced, not
designed**: the bell's area happens to be $\sqrt{2\pi}$, and a PDF must integrate to 1, so $c = 1/\sqrt{2\pi}$.

### Deriving the √(2π) (Gaussian integral)

We need the area $I = \int_{-\infty}^{\infty} e^{-z^2/2}\,dz$. Trick: square it and go polar.

$$I^2 = \int_{-\infty}^{\infty}\!\int_{-\infty}^{\infty} e^{-(z^2+y^2)/2}\,dz\,dy = \int_0^{2\pi}\!\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta$$

In polar, $z^2+y^2 = r^2$ and the area element gains an $r$. Substitute $u=r^2/2,\ du=r\,dr$: the inner
integral is $\int_0^\infty e^{-u}\,du = 1$. The outer gives $\int_0^{2\pi} 1\,d\theta = 2\pi$. So $I^2 = 2\pi$,
hence $I = \sqrt{2\pi}$ and:

$$f(z) = \frac{1}{\sqrt{2\pi}}\,e^{-z^2/2}$$

### Mean and variance of N(0,1)

**Mean $=0$:** $z\,e^{-z^2/2}$ is odd, integrating to 0 over a symmetric range. **Variance $=1$:** since the
mean is 0, $\mathrm{Var}(Z)=E(Z^2)$. By LOTUS and integration by parts ($u=z,\ dv = z e^{-z^2/2}dz$):

$$E(Z^2) = \frac{2}{\sqrt{2\pi}}\Big(\underbrace{[-z e^{-z^2/2}]_0^\infty}_{0} + \underbrace{\int_0^\infty e^{-z^2/2}dz}_{\sqrt{2\pi}/2}\Big) = \frac{2}{\sqrt{2\pi}}\cdot\frac{\sqrt{2\pi}}{2} = 1$$

### Standard to general transform

| Direction | Transform | What it does |
|---|---|---|
| standard to general | $X = \mu + \sigma Z$ | stretch by $\sigma$, slide by $\mu$ |
| general to standard (z-score) | $Z = \dfrac{X-\mu}{\sigma}$ | recenter to 0, rescale spread to 1 |

$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac12\left(\frac{x-\mu}{\sigma}\right)^2}$$

Sums of independent normals: $X_i + X_j \sim \mathcal{N}(\mu_i+\mu_j,\ \sigma_i^2+\sigma_j^2)$. Differences:
means subtract but **variances still add**: $X_i - X_j \sim \mathcal{N}(\mu_i-\mu_j,\ \sigma_i^2+\sigma_j^2)$.

!!! warning "Trap, variances add even on subtraction"
    Subtracting two noisy things makes you *more* uncertain, not less.
    $\mathrm{Var}(X-Y) = \mathrm{Var}(X)+\mathrm{Var}(Y)$ for independents.

### The CDF Φ and the empirical rule

$\Phi(z) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-t^2/2}dt$ has **no closed form** (hence z-tables or
`norm.cdf`). By symmetry $\Phi(-z) = 1 - \Phi(z)$. The handy 68-95-99.7 rule:

$$P(|X-\mu|\le\sigma)\approx 68\%,\quad P(|X-\mu|\le 2\sigma)\approx 95\%,\quad P(|X-\mu|\le 3\sigma)\approx 99.7\%$$

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 620 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Normal distribution bell curve with sigma bands">
  <defs>
    <linearGradient id="g1" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#e0a23b" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#e0a23b" stop-opacity="0.04"/>
    </linearGradient>
  </defs>
  <line x1="40" y1="220" x2="600" y2="220" stroke="#39414f" stroke-width="1.5"/>
  <line x1="320" y1="60" x2="320" y2="220" stroke="#54c7b8" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="227" y1="120" x2="227" y2="220" stroke="#7c8593" stroke-width="0.8" stroke-dasharray="2 3"/>
  <line x1="413" y1="120" x2="413" y2="220" stroke="#7c8593" stroke-width="0.8" stroke-dasharray="2 3"/>
  <line x1="134" y1="185" x2="134" y2="220" stroke="#7c8593" stroke-width="0.6" stroke-dasharray="2 3"/>
  <line x1="506" y1="185" x2="506" y2="220" stroke="#7c8593" stroke-width="0.6" stroke-dasharray="2 3"/>
  <path d="M40,219 C120,217 170,205 227,120 C265,64 290,60 320,60 C350,60 375,64 413,120 C470,205 520,217 600,219 L600,220 L40,220 Z" fill="url(#g1)" stroke="#e0a23b" stroke-width="2"/>
  <text x="320" y="240" fill="#a9b1bd" font-family="monospace" font-size="11" text-anchor="middle">μ</text>
  <text x="227" y="240" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">μ−σ</text>
  <text x="413" y="240" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">μ+σ</text>
  <text x="134" y="240" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">μ−2σ</text>
  <text x="506" y="240" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">μ+2σ</text>
  <text x="320" y="150" fill="#0f1115" font-family="monospace" font-size="11" text-anchor="middle" font-weight="bold">68%</text>
  <text x="320" y="100" fill="#e7e3d8" font-family="monospace" font-size="10" text-anchor="middle">95% within 2σ</text>
</svg>
<figcaption>The bell with sigma bands: about 68% within 1σ, 95% within 2σ, and 99.7% within 3σ.</figcaption>
</figure>

**How to remember the formula (never memorize the symbols).** 1. Bell core: $e^{-(\text{something})^2}$,
square gives symmetry, minus gives decay. 2. The "something" is the z-score $(x-\mu)/\sigma$, with a
$\tfrac12$: exponent $-\tfrac12\big(\tfrac{x-\mu}{\sigma}\big)^2$. 3. Normalize the area: divide by
$\sigma\sqrt{2\pi}$. One-line mnemonic: "one over sigma-root-two-pi, times e to the minus half z-squared."

**Where you see it.** Weight init, Gaussian priors, regression residuals, diffusion-model noise, VAE latents,
Gaussian policies in RL. It is the **max-entropy** distribution for a fixed mean and variance, and the CLT
makes it the default for any sum of many small independent effects. Across populations: heights, test scores,
measurement error.

**When to reach for a Gaussian, and when not to.** **Reach for it** when your quantity is a sum or average of
many small independent contributions (the CLT justifies it), when you only know the mean and variance and want
the least-committal (max-entropy) distribution, or when data empirically looks bell-shaped and symmetric.
**Avoid it** when data is heavy-tailed (financial crashes), bounded or strictly positive (waiting times cannot
go negative), or strongly skewed, since forcing a normal there underestimates extremes and predicts impossible
negative values.

## Interview Questions

**Q1: Explain the two directions of "universality of the uniform" and why they matter.**
Simulation says that if $U$ is uniform on $[0,1]$ and $F$ is a target CDF, then $F^{-1}(U)$ has distribution
$F$, so you can generate any distribution from a uniform. Flattening says that feeding a continuous variable
through its own CDF, $F(X)$, returns a uniform. Together they explain why a single uniform random number
generator is all a computer needs to simulate any distribution.

**Q2: Why does the standard normal PDF have the form $e^{-z^2/2}$, and where does the $\sqrt{2\pi}$ come from?**
The square makes the function symmetric in $z$, and the negative exponent makes it decay in the tails, giving a
bell peaked at zero. The constant is forced, not chosen: the Gaussian integral $\int e^{-z^2/2}dz$ evaluates to
$\sqrt{2\pi}$ via the square-and-go-polar trick, and since a PDF must integrate to one, the normalizer is
$1/\sqrt{2\pi}$.

**Q3: If $X$ and $Y$ are independent, what is $\mathrm{Var}(X-Y)$, and why does it surprise people?**
It is $\mathrm{Var}(X)+\mathrm{Var}(Y)$, the same as for the sum. Variances add on subtraction because the
sign flips inside the variance but the squared deviations do not, so subtracting two noisy quantities makes you
more uncertain, not less. This is a common trap in error propagation.

**Q4: State the empirical rule and one caution about using it.**
About 68 percent of a normal's mass lies within one standard deviation of the mean, 95 percent within two, and
99.7 percent within three. The caution is that this holds only for the normal; heavy-tailed or skewed data put
far more mass in the tails, so the rule overstates how concentrated such data are.
