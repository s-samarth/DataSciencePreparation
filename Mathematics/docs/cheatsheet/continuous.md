# Continuous Distributions

Outcomes on a continuum. Probability lives in a PDF (probability **density** function): height is *density*,
not probability, and the **area** under a stretch of the curve gives the probability. For a single exact
point, probability is 0. This grid lines up the ten real-valued distributions by parameters, story, use
case, formula, and moments.

!!! tip "Rapid Recall"
    The continuous uniform is maximum ignorance on an interval and the raw material of all simulation. The
    Normal is the default for noise and aggregates, justified by the CLT. The exponential is the continuous
    memoryless waiting time, and the gamma generalizes it to the $\alpha$-th event. Beta lives on $[0,1]$ and
    is the conjugate prior for the Bernoulli and binomial; gamma is conjugate for the Poisson. Log-normal
    arises from multiplying many positive factors, Cauchy is the cautionary tale with no mean or variance,
    chi-squared sums squared normals, Student's t handles small samples, and the multivariate normal is the
    correlated bell curve in many dimensions.

## The continuous grid

| Distribution | Parameters | What it models | When and why you use it | PDF / CDF | $E[X]$ | $\mathrm{Var}(X)$ |
|---|---|---|---|---|---|---|
| **Continuous Uniform** <br> $X\sim\text{Unif}(a,b)$ | $a$ = lower bound; $b$ = upper bound | Equally likely anywhere on the interval $[a,b]$. Flat-topped. | Maximum ignorance on an **interval**. Also the **raw material of all simulation** (universality of the uniform: feed it into an inverse CDF to make any distribution). <br> Weight-init ranges, random sampling, dropout thresholds. | $f(x)=\dfrac{1}{b-a},\ a\le x\le b$; $F(x)=\dfrac{x-a}{b-a}$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ |
| **Normal / Gaussian** (ML) <br> $X\sim N(\mu,\sigma^2)$ | $\mu$ = mean / center; $\sigma^2$ = variance; $\sigma$ = std-dev (spread) | The bell curve. The distribution of sums and averages of many small independent effects. | The **default** for noise, errors, and anything that is a sum of many small effects (justified by the CLT). The max-entropy choice given only a mean and variance. <br> Weight init, Gaussian priors, regression residuals, diffusion noise, VAE latents. | $f(x)=\dfrac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$; CDF $\Phi$ has no closed form | $\mu$ | $\sigma^2$ |
| **Exponential** (ML) <br> $X\sim\text{Exp}(\lambda)$ | $\lambda$ = rate (events per unit time) | Waiting time **until the next event** in a Poisson process. The continuous cousin of Geometric. | Time between independent events arriving at a constant rate. The **only** continuous memoryless distribution, elapsed waiting tells you nothing about the remaining wait. <br> Survival analysis, time-to-failure, queueing, hazard models. | $f(x)=\lambda e^{-\lambda x},\ x\ge0$; $F(x)=1-e^{-\lambda x}$ | $\dfrac{1}{\lambda}$ | $\dfrac{1}{\lambda^2}$ |
| **Gamma** <br> $X\sim\text{Gamma}(\alpha,\beta)$ | $\alpha$ = shape (number of events); $\beta$ = rate (alt: scale $\theta=1/\beta$) | Waiting time until the $\alpha$-th event in a Poisson process, a sum of $\alpha$ exponentials. Generalizes Exponential ($\alpha=1$). | Positive, right-skewed waiting times or aggregated amounts. Bayesian workhorse: **conjugate prior** for a Poisson rate and for the precision of a Normal. <br> Priors on rates/variances; modeling positive continuous quantities. | $f(x)=\dfrac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},\ x>0$, where $\Gamma(\alpha)=\int_0^\infty t^{\alpha-1}e^{-t}dt$ extends factorial, $\Gamma(n)=(n-1)!$ | $\dfrac{\alpha}{\beta}$ | $\dfrac{\alpha}{\beta^2}$ |
| **Beta** (ML) <br> $X\sim\text{Beta}(\alpha,\beta)$ | $\alpha>0$, $\beta>0$ = shape parameters | A distribution **over probabilities**, it lives entirely on $[0,1]$. Shape flexes from U-shaped to bell to flat. | The natural way to express belief about an **unknown probability or proportion**. **Conjugate prior** for Bernoulli/Binomial: see $s$ successes, $f$ failures, posterior is $\text{Beta}(\alpha+s,\beta+f)$. <br> Bayesian A/B testing, Thompson sampling in bandits, modeling CTRs. | $f(x)=\dfrac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$, with $B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ the normalizer | $\dfrac{\alpha}{\alpha+\beta}$ | $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| **Log-Normal** <br> $X\sim\text{LogN}(\mu,\sigma^2)$ | $\mu,\sigma^2$ = mean and variance *of* $\ln X$ (NOT of $X$) | A variable whose **logarithm** is normal. Arises from **multiplying** many positive random factors (multiplicative CLT). | Positive, right-skewed quantities spanning orders of magnitude: incomes, file sizes, prices, response times. <br> Log-transform a skewed positive target, then treat it as normal. | $f(x)=\dfrac{1}{x\sigma\sqrt{2\pi}}\,e^{-\frac{(\ln x-\mu)^2}{2\sigma^2}},\ x>0$ | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}\!-1)\,e^{2\mu+\sigma^2}$ |
| **Cauchy** <br> $X\sim\text{Cauchy}(x_0,\gamma)$ | $x_0$ = location (peak / median); $\gamma$ = scale (half-width) | The ratio of two independent standard normals. Bell-ish peak but with **monstrously heavy tails**. | The cautionary tale. Tails so fat that **mean and variance do not exist**, and sample averages never settle (**CLT fails**). Used to model extreme outliers and in physics (resonance). <br> A stress-test for "robust to outliers" claims. | $f(x)=\dfrac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}$; standard $\frac{1}{\pi(1+x^2)}$ | undefined | undefined |
| **Chi-Squared** <br> $X\sim\chi^2(k)$ | $k$ = degrees of freedom (how many squared normals are summed) | Sum of $k$ independent **squared** standard normals: $Z_1^2+\dots+Z_k^2$. A special case of Gamma. | The sampling distribution of a sample variance, and the test statistic for **goodness-of-fit** and **independence** (the chi-squared test). Foundation of frequentist hypothesis testing. <br> Feature-independence tests, variance inference. | $f(x)=\dfrac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2},\ x>0$ | $k$ | $2k$ |
| **Student's t** <br> $X\sim t(\nu)$ | $\nu$ = degrees of freedom | Like a standard normal but with **heavier tails**. Appears when you estimate a mean using the *sample* std-dev from a small sample (true $\sigma$ unknown). | Small-sample inference about a mean, the **t-test**. As $\nu\to\infty$ it converges to the Normal; small $\nu$ means fatter tails (more caution with little data). <br> Robust regression with t-distributed errors; heavy-tailed Bayesian models. | $f(t)\propto\left(1+\dfrac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$ | $0$ (for $\nu>1$) | $\dfrac{\nu}{\nu-2}$ (for $\nu>2$) |
| **Multivariate Normal** (multivariate, ML) <br> $X\sim N(\boldsymbol\mu,\Sigma)$ | $\boldsymbol\mu$ = mean vector (length $d$); $\Sigma$ = $d\times d$ covariance matrix ($\Sigma_{ij}$ = cov of dims $i,j$) | The bell curve in many **correlated** dimensions. Generalizes the Normal to vectors. | Modeling several continuous variables jointly with their correlations. One of the most-used objects in all of ML. <br> Gaussian Mixture Models, Gaussian Processes, LDA/QDA, Kalman filters, VAE priors. | $f(\mathbf{x})=\dfrac{1}{(2\pi)^{d/2}\lvert\Sigma\rvert^{1/2}}e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol\mu)^{\!\top}\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)}$, where $\lvert\Sigma\rvert$ is the determinant, $\Sigma^{-1}$ the inverse, $\top$ transpose, $d$ dimension | $\boldsymbol\mu$ (vector) | $\Sigma$ (matrix) |

For the Normal's derivation and the multivariate covariance structure, see
[Universality of the Uniform & the Normal](../advanced/uniform-normal.md) and
[Joint Distributions & Covariance](../advanced/joint-covariance.md).

## Interview Questions

**Q1: Why is the exponential distribution the continuous analog of the geometric?**
Both model waiting times and both are memoryless, the only members of their respective worlds with that
property. The geometric counts discrete trials until the first success, while the exponential measures
continuous time until the next event in a Poisson process. Memorylessness means elapsed waiting gives no
information about remaining waiting.

**Q2: What makes the Cauchy distribution a cautionary tale?**
Its tails are so heavy that the integrals defining its mean and variance diverge, so neither exists. As a
consequence the law of large numbers and the central limit theorem do not apply, and sample averages never
settle down. It is a reminder that "just average it out" silently assumes finite moments.

**Q3: Name two conjugate-prior pairings and why they matter.**
Beta is conjugate to the Bernoulli and binomial, and gamma is conjugate to the Poisson. Conjugacy means the
posterior stays in the same family as the prior, so Bayesian updating reduces to arithmetic on the parameters,
which is the backbone of tractable Bayesian inference and bandit algorithms.

**Q4: How do the chi-squared, Student's t, and Cauchy distributions relate to the Normal?**
A chi-squared with $k$ degrees of freedom is a sum of $k$ squared standard normals. A standard normal divided
by $\sqrt{\chi^2_\nu/\nu}$ gives Student's t with $\nu$ degrees of freedom, which approaches the Normal as
$\nu\to\infty$. The ratio of two independent standard normals is Cauchy, which is exactly Student's t with one
degree of freedom.
