# OLS as Inference

Regression sits at the exact intersection of everything in this section: it is an estimation problem, it has
a hidden maximum-likelihood identity, its estimators are random variables with sampling distributions, and
its confidence intervals are one-mean inference re-skinned. This page makes that structure explicit and then
ties the whole estimation thread together.

!!! tip "Rapid Recall"
    Ordinary least squares estimates the true intercept and slope by minimizing squared residuals, and under
    Gaussian noise it *is* maximum likelihood, which is why least squares is forced rather than arbitrary. The
    Gauss-Markov theorem says OLS is the Best Linear Unbiased Estimator without needing normality; normality is
    required only for inference. The slope is a random variable with variance $\sigma^2/\sum(x_i-\bar x)^2$, so
    spreading the predictor widely sharpens it. Because $\sigma$ is estimated, coefficients follow a t with
    $n-2$ degrees of freedom, and a confidence interval that contains zero means the predictor cannot be
    distinguished from irrelevant.

## §10 OLS Through the Lens of Estimation & Inference

Regression is inferential statistics wearing a predictive-modeling costume.

### What OLS is, in estimation language

The model says the world generates data as:

$$y_i=\beta_0+\beta_1 x_i+\varepsilon_i$$

$\beta_0,\beta_1$ are **population parameters**, the true, fixed, *unknown* intercept and slope (the regression
analogues of $\mu$). $\varepsilon_i$ is noise. You never see them; from data you compute estimates
$\hat\beta_0,\hat\beta_1$. The fitted line $\hat y=\hat\beta_0+\hat\beta_1 x$ is your *point estimate* of the
true line. OLS is the specific **estimator** that minimizes the sum of squared residuals:

$$\hat\beta_0,\hat\beta_1=\arg\min_{\beta_0,\beta_1}\sum_{i=1}^n(y_i-\beta_0-\beta_1 x_i)^2$$

### The hidden identity, OLS *is* maximum likelihood

Why squared errors? Because **OLS is MLE under Gaussian noise.** Assume $\varepsilon_i\sim N(0,\sigma^2)$; then
each $y_i\sim N(\beta_0+\beta_1 x_i,\sigma^2)$. The log-likelihood has the familiar normal structure:

$$\ell(\beta_0,\beta_1)=-\frac n2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-\beta_0-\beta_1 x_i)^2$$

Maximizing $\ell$, first term constant, $1/(2\sigma^2)$ a positive constant, is identical to **minimizing
$\sum(y_i-\beta_0-\beta_1 x_i)^2$**, the sum of squared residuals. OLS and Gaussian-noise MLE are the same
estimator. Least squares is not an arbitrary choice of "squared," it is what MLE *forces* the moment you assume
normal errors. This is exactly why the inferential apparatus (t-tests, CIs) attaches so cleanly: the
coefficients are MLEs of a normal model, so they inherit normal sampling distributions. See
[Likelihood & MLE](likelihood-mle.md) for the general argument.

### The estimators and Gauss-Markov

$$\hat\beta_1=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}=\frac{\text{Cov}(x,y)}{\text{Var}(x)},\qquad \hat\beta_0=\bar y-\hat\beta_1\bar x$$

The intercept formula forces the line through $(\bar x,\bar y)$. The **Gauss-Markov theorem** (the regression
analogue of "the sample mean is good") states that under its assumptions OLS is **BLUE**, Best Linear Unbiased
Estimator. "Unbiased": $E[\hat\beta_1]=\beta_1$. "Best": lowest variance among linear unbiased estimators.
Notably this needs *no* normality, normality is required only for the *inference* (CIs and tests), not for OLS
being unbiased and minimum-variance.

### The estimators are random variables, sampling distributions

$\hat\beta_1$ is a function of the random $y_i$, so it is a random variable with a sampling distribution. It is
unbiased ($E[\hat\beta_1]=\beta_1$), and its variance is:

$$\text{Var}(\hat\beta_1)=\frac{\sigma^2}{\sum(x_i-\bar x)^2}$$

The slope is more precise (lower variance) when (1) the noise $\sigma^2$ is small, and (2) the x-values are
**spread out** (large $\sum(x_i-\bar x)^2$). If all your x-values are bunched together, you are inferring a
line's tilt from points crammed in a narrow strip, small wiggles swing the slope wildly. Spread x widely and
the slope is anchored firmly. **To estimate a slope well, vary your predictor widely.** Since $\sigma^2$ is
unknown, estimate it from the residuals:

$$\hat\sigma^2=\frac{\sum(y_i-\hat y_i)^2}{n-2}=\frac{\text{SSR}}{n-2}$$

The **$n-2$** is the df correction: 2 parameters ($\hat\beta_0,\hat\beta_1$) were spent before measuring
residual spread. General rule: df $= n$ minus (number of parameters estimated). Then the standard error of the
slope:

$$SE(\hat\beta_1)=\frac{\hat\sigma}{\sqrt{\sum(x_i-\bar x)^2}}$$

### Confidence intervals for regression parameters

Because $\sigma$ was estimated, the standardized coefficient follows a **t-distribution** with $n-2$ df, same
reason as the one-mean case (estimating the noise adds a second source of randomness):

$$\frac{\hat\beta_1-\beta_1}{SE(\hat\beta_1)}\sim t_{n-2}$$
$$\hat\beta_1\;\pm\;t_{\alpha/2,\,n-2}\cdot SE(\hat\beta_1) \qquad\qquad \hat\beta_0\;\pm\;t_{\alpha/2,\,n-2}\cdot SE(\hat\beta_0)$$

This is *literally the same formula* as the CI for a single mean, t critical value times a standard error, with
$\hat\beta_1$ in place of $\bar x$ and $n-2$ df in place of $n-1$. **Regression inference is one-mean inference
re-skinned.** If the CI for $\beta_1$ **contains 0**, the data is consistent with "x has no effect on y" (a
flat line). If it **excludes 0**, you have evidence of a real relationship. This is the same "contains 0 means
no difference" logic as the two-means CI, and it is exactly what the per-coefficient t-test and p-value report,
testing $H_0:\beta_j=0$ via the t-statistic $\hat\beta_j/SE(\hat\beta_j)$. A coefficient whose CI straddles 0
cannot be distinguished from irrelevant. As a worked setup, fit a line on $n=12$ points with $\hat\beta_1=2.5$
and $SE(\hat\beta_1)=0.8$, then build a 95% CI as $2.5\pm t_{0.025,10}\cdot 0.8$.

### Two kinds of intervals in regression (do not conflate)

- **CI for the mean response** at a given x: how precisely you have located the *average* y there (uncertainty about the line itself).
- **Prediction interval** for a new individual observation: always *wider*, because it includes both the line's uncertainty *and* the irreducible noise $\sigma^2$ of a single point scattering around the line.

The parameter CIs above are about the coefficients; these two are about the fitted values, same machinery,
different targets.

## §11 The Whole Thread in One Page

Every concept in this section is a link in one chain.

| # | Concept | The one-line essence |
|---|---|---|
| 01 | Population vs Sample | We want a fixed-but-unknown parameter; we have a known-but-random statistic. Inference bridges them via the sampling distribution (CLT plus LLN). |
| 02 | Estimator vs Estimate | Estimator is the rule (random variable, has properties); estimate is the number (fixed). Bias, variance, consistency belong to the rule. |
| 03 | Probability vs Likelihood | Same $f(x;\theta)$: probability fixes $\theta$ and varies data (sums to 1); likelihood fixes data and varies $\theta$ (does not). Estimation lives on the likelihood side. |
| 04 | Why MLE becomes loss | $\theta$ is the model; pick the $\theta$ making observed data most probable. Log turns product into sum (monotone, peak unmoved). NLL is the loss: Gaussian gives MSE, categorical gives cross-entropy. |
| 05 | Worked MLEs | Bernoulli gives $\hat p=\bar x$; normal mean gives $\hat\mu=\bar x$; normal variance gives $\frac1n\sum(x_i-\bar x)^2$ (biased, divides by $n$). |
| 06 | Bias | Unbiased iff $E[\hat\theta]=\theta$. $\bar X$ unbiased; $\hat\sigma^2_{MLE}$ biased at $\frac{n-1}{n}\sigma^2$, giving the $n{-}1$ correction. MSE = Bias$^2$ + Var. |
| 07 | CIs for means | estimate plus or minus crit times SE. z if $\sigma$ known, t (df $n{-}1$) if estimated, t's fat tails are the price of estimating $\sigma$. CLT rescues non-normal data at large $n$. Two means: SE of a difference; contains 0 means no difference. |
| 08 | CIs: variance & proportion | Variance gives skewed chi-squared, asymmetric. Proportion gives z with $\sqrt{\hat p(1-\hat p)/n}$ (no t, variance fixed by $p$). |
| 09 | Four estimation cases | Mean (t), proportion (z), variance (chi-squared); small finite population multiplies SE by FPC $\sqrt{(N-n)/(N-1)}$ (to 1 for big $N$, to 0 at a census). |
| 10 | OLS & inference | OLS = Gaussian-noise MLE = BLUE. Coefficients are random variables; CI is $\hat\beta\pm t_{\alpha/2,n-2}SE$, one-mean inference re-skinned. Contains 0 means predictor indistinguishable from irrelevant. |

Inferential statistics is the discipline of using a known-but-random sample statistic to make a defensible,
uncertainty-quantified statement about a fixed-but-unknown population parameter: likelihood scores candidate
parameters, MLE picks the best (and becomes your loss), bias checks whether the estimator is centered on the
truth, and confidence intervals wrap the estimate in an honest range whose reliability is a property of the
procedure, not the single interval.

## Interview Questions

**Q1: Why does ordinary least squares minimize squared errors specifically?**
Because under the assumption of Gaussian noise, minimizing the sum of squared residuals is exactly maximum
likelihood. Each $y_i$ is normal around the line, so the log-likelihood contains the negative sum of squared
residuals, and maximizing it is identical to minimizing that sum. Squared error is forced by the normal noise
assumption, not chosen arbitrarily.

**Q2: What does the Gauss-Markov theorem guarantee, and does it require normality?**
It guarantees that ordinary least squares is the Best Linear Unbiased Estimator, meaning unbiased and of lowest
variance among linear unbiased estimators, under assumptions like zero-mean uncorrelated errors with constant
variance. It does not require normality; normality is needed only for the inference layer, the t-tests and
confidence intervals, not for OLS being unbiased and minimum-variance.

**Q3: Why does spreading your predictor values out improve the slope estimate?**
Because the slope's variance is $\sigma^2/\sum(x_i-\bar x)^2$, so a larger spread in the x-values increases the
denominator and shrinks the variance. Intuitively, inferring a line's tilt from points crammed into a narrow
strip is unstable, while widely spread points anchor the slope firmly.

**Q4: Why do regression coefficients use a t distribution with $n-2$ degrees of freedom?**
Because the noise variance $\sigma^2$ is unknown and estimated from the residuals, which adds a second source
of randomness, exactly as in the one-mean case. Two parameters, the intercept and slope, are estimated before
measuring residual spread, so the degrees of freedom are $n-2$, and a coefficient interval that contains zero
means the predictor cannot be distinguished from irrelevant.
