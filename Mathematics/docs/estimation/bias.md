# Bias & Unbiased Estimators

An estimator is unbiased if, on average across all possible samples, it lands on the true parameter. The
concept only makes sense once you remember that an estimator is a random variable with a sampling
distribution, so it has an expected value. This page proves where the famous $n-1$ comes from and frames the
bias-variance tradeoff.

!!! tip "Rapid Recall"
    Bias is $E[\hat\theta]-\theta$: an estimator is unbiased when its long-run average equals the truth, which
    is about the center of the sampling distribution, not any single estimate. The sample mean is unbiased for
    any $n$ and any distribution with finite mean. The MLE of variance is biased at $\frac{n-1}{n}\sigma^2$,
    which is exactly why the unbiased sample variance divides by $n-1$, the one degree of freedom spent
    estimating the mean. The fuller measure is mean squared error, $\text{Bias}^2+\text{Variance}$, and
    accepting a little bias to cut a lot of variance is the entire logic of regularization.

## §6 Biased vs Unbiased Estimators

An estimator is unbiased if, on average across all possible samples, it lands on the true parameter. The whole
concept hinges on remembering that the estimator is a random variable with a sampling distribution, so it *has*
an expected value.

$$\text{Bias}(\hat\theta)=E[\hat\theta]-\theta \qquad\Rightarrow\qquad \begin{cases} E[\hat\theta]=\theta & \text{unbiased}\\ E[\hat\theta]\neq\theta & \text{biased}\end{cases}$$

Unbiasedness asks: **is the long-run average of the estimator equal to the truth?** Not whether any single
estimate equals $\theta$ (it almost never does). It asks whether the estimator is *centered* on the truth. An
unbiased estimator is a thrower whose darts are *centered* on the bullseye, even if individual darts miss. A
biased estimator is centered off to one side, consistently pulling left, no matter how many darts. Bias is the
**center of the cluster**, not the spread.

### The procedure

Compute $E[\hat\theta]$ and compare to $\theta$. Three properties of expectation carry almost all the weight:

- **Linearity:** $E[aX+bY]=aE[X]+bE[Y]$, passes through sums and constants. The workhorse.
- **Each point has the population's expectation:** $E[X_i]=\mu$, $\text{Var}(X_i)=\sigma^2$.
- **For squares:** $E[X^2]=\text{Var}(X)+(E[X])^2$.

### Check 1, the sample mean is unbiased

$$\bar X \text{ is unbiased for } \mu \qquad (\text{Bias}=\mu-\mu=0)$$

True for any $n$, any distribution with a finite mean. No correction needed, this is why the sample mean is
beloved.

### Check 2, the MLE of variance is biased (full proof)

We prove $E[\hat\sigma^2_{\text{MLE}}]\neq\sigma^2$ by computing the expectation directly. This is a rite of
passage.

$$E[\hat\sigma^2_{\text{MLE}}]=\frac{n-1}{n}\sigma^2 \;\neq\; \sigma^2 \qquad \text{Bias}=-\frac{\sigma^2}{n}$$

Since $(n{-}1)/n<1$, the MLE **underestimates** the variance, confirming the downward pull. The bias is
negative and shrinks as $n$ grows (bias goes to 0), so the MLE is still **consistent**. The fix falls out
automatically: multiply by $n/(n{-}1)$ to get the unbiased $s^2=\frac{1}{n-1}\sum(X_i-\bar X)^2$. **That is
where the mysterious $n{-}1$ comes from**, the precise correction factor, the one degree of freedom spent
estimating $\mu$.

### The general recipe

- Write $\hat\theta$ as a formula in the random $X_i$.
- Take $E[\hat\theta]$, push the expectation inward via linearity.
- Replace $E[X_i]\to\mu$, $\text{Var}(X_i)\to\sigma^2$, use $E[X^2]=\text{Var}+(E)^2$ for squares.
- Simplify; compare to $\theta$. Equal means unbiased; else the difference is the bias and its sign is the direction.

An unbiased estimator with huge variance can be worse than a slightly biased one with tiny variance. The fuller
measure is **Mean Squared Error**: $\text{MSE}=\text{Bias}^2+\text{Variance}$. Sometimes you accept a little
bias to kill a lot of variance, that is the entire logic of **regularization** in ML (shrinkage estimators are
deliberately biased to reduce variance). "Is it unbiased?" is the right first question; "what is its MSE?" is
often the deciding one.

## Interview Questions

**Q1: What does it mean for an estimator to be unbiased?**
It means its expected value over all possible samples equals the true parameter, $E[\hat\theta]=\theta$, so the
sampling distribution is centered on the truth. Unbiasedness is a statement about the long-run average of the
rule, not about any single estimate, which almost never equals the parameter exactly.

**Q2: Where does the $n-1$ in the sample variance come from?**
The maximum likelihood variance divides by $n$ and is biased at $\frac{n-1}{n}\sigma^2$, underestimating the
truth because it uses $\bar x$ in place of $\mu$. Multiplying by $n/(n-1)$ removes the bias, so the unbiased
estimator divides by $n-1$. That single subtracted degree of freedom is the one spent estimating the mean
before estimating the variance.

**Q3: Is an unbiased estimator always preferable?**
Not necessarily. Mean squared error decomposes as bias squared plus variance, so a slightly biased estimator
with much smaller variance can have lower MSE than an unbiased one with large variance. Deliberately trading a
little bias for a large reduction in variance is exactly what regularization and shrinkage estimators do.

**Q4: The maximum likelihood variance is biased, yet it is still used. Why is that acceptable?**
Because its bias is $-\sigma^2/n$, which shrinks to zero as $n$ grows, so the estimator is consistent and
converges to the true variance with enough data. Maximum likelihood guarantees good asymptotic behavior rather
than finite-sample unbiasedness, and for large samples the distinction between dividing by $n$ and $n-1$ is
negligible.
