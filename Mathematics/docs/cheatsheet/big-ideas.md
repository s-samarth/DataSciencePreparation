# Big Ideas

The grid of eighteen distributions compresses into a handful of threads. Hold these and you can reconstruct
most of the table from memory, and more importantly know which distribution a new problem is secretly asking
about.

!!! tip "Rapid Recall"
    PMF is mass and PDF is density, so a continuous point has probability zero. Only the geometric and
    exponential are memoryless. Conjugate priors keep the posterior in the prior's family, turning Bayesian
    updates into arithmetic. Cauchy has no mean or variance, so "average it out" assumes finite moments.
    Variance near the mean signals Poisson; variance far above it signals over-dispersion and the negative
    binomial. The Normal is everywhere because of the central limit theorem, not magic.

## PMF versus PDF

Discrete means **mass**: the bar height is the probability. Continuous means **density**: height is not
probability, area is. A single point on a continuous variable has probability 0.

## Memoryless means no aging

Only Geometric (discrete) and Exponential (continuous) are memoryless. Having waited a while gives you zero
information about how much longer you will wait.

$$P(X>s+t\mid X>s)=P(X>t)$$

## Conjugate priors

Pick a prior so the posterior stays in the same family, and updating becomes arithmetic. Beta with Binomial,
Gamma with Poisson, Normal with Normal-mean. The backbone of tractable Bayesian inference and bandits.

## When mean and variance do not exist

Cauchy has tails so heavy that its integrals diverge, so it has no mean, no variance, and the CLT does not
apply. A reminder that "average it out" silently assumes finite moments.

## Mean equals variance is a tell

If a count's variance $\approx$ its mean, think Poisson. If variance noticeably exceeds the mean
(over-dispersion), reach for Negative Binomial instead.

## Why the Normal is everywhere

Not magic, it is the CLT. Anything built from a sum of many small independent contributions ends up
bell-shaped, regardless of the parts. Hence noise, errors, and aggregates default to Normal.

## Interview Questions

**Q1: A colleague says a probability density of 3 is impossible because probabilities cannot exceed 1. Are they right?**
No. A density is not a probability; only the area under it over an interval is. On a narrow support the
density must rise above 1 to enclose unit area, for example a uniform on an interval of width one third has
height 3. The single-point probability for any continuous variable is zero.

**Q2: How do you recognize at a glance whether a count should be modeled as Poisson or negative binomial?**
Compare the sample variance to the sample mean. If they are roughly equal, the Poisson's mean-equals-variance
signature fits. If the variance is noticeably larger, the data are over-dispersed and the negative binomial,
which has a free variance, is the better model.

**Q3: Why does conjugacy matter in practice?**
Conjugacy keeps the posterior in the same parametric family as the prior, so a Bayesian update becomes a
simple parameter adjustment rather than an intractable integral. That is what makes online updating, Thompson
sampling, and many closed-form Bayesian methods feasible.
