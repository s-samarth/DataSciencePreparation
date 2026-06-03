# Population, Samples & Estimators

The entire tension of inferential statistics in one sentence: we care about the population, but we only ever
get to see a sample. This page sets the vocabulary that keeps the reasoning honest, then draws the crucial
line between an estimator (a rule) and an estimate (a number).

!!! tip "Rapid Recall"
    A population parameter is fixed but unknown; a sample statistic is known but random, and estimation
    bridges them. Greek letters denote parameters, hatted Latin letters denote estimates. The sample mean
    $\bar x$ is a realization of a random variable with a sampling distribution, which the central limit
    theorem makes approximately normal around $\mu$. A point estimate is a single best guess, a confidence
    interval attaches honesty as estimate plus or minus a margin. An estimator is the rule, a random variable
    that has bias, variance, and consistency; an estimate is the fixed number it returns, and those properties
    belong to the rule, not the number.

## §1 Population vs Sample & the Estimation Problem

A **population** is the complete set of every entity you care about, every adult voter, every transaction
your system will ever process, every possible user. The population is the truth you want to know. A **sample**
is the small subset you actually collected and measured.

The whole game: you want to know something true about the population, but you cannot measure it (too big, too
expensive, partly in the future, or literally infinite). So you measure a sample and *infer backward* to the
population. That inference is the entire subject. Sometimes cost (calling 900 million voters). Sometimes
physical impossibility (the population includes future transactions that have not happened). Sometimes
measuring destroys the item (you cannot crash-test every car you sell). The population is often a theoretical
object, not a list you could ever hold.

### Parameters versus statistics: the vocabulary that keeps you honest

A **parameter** describes the *population*, the true average height of all adults, the true churn fraction.
Parameters are fixed (a definite real value) but **unknown**. A **statistic** is computed from your *sample*,
it is **known** (you can calculate it) but **random** (a different sample gives a different number).

|  | Population (parameter) | Sample (statistic) |
|---|---|---|
| What it describes | the truth you care about | the data you happened to grab |
| Is it fixed? | Yes, fixed | No, changes sample to sample |
| Do you know it? | No, never exactly | Yes, you can compute it |

The asymmetry is the whole point: **the thing you want is fixed-but-unknown; the thing you have is
known-but-random.** Estimation is the bridge.

**Notation, why the Greek/Latin split exists.** Greek letters for parameters, Latin (often hatted) for
statistics, so you never confuse "the truth" with "your guess at the truth." The hat literally means
"estimate of."

| Quantity | Population parameter | Sample statistic |
|---|---|---|
| Mean | $\mu$ (mu) | $\bar{x}$ ("x-bar") |
| Variance | $\sigma^2$ | $s^2$ |
| Std deviation | $\sigma$ | $s$ |
| Proportion | $p$ | $\hat{p}$ ("p-hat") |

### Why the sample statistic is a random variable

When you compute $\bar{x}$ from 500 people you get one number, say 168 cm, and it feels fixed. It is not. It
is the realization of a **random variable**. The population mean $\mu$ sits there as one fixed number, but
$\bar{x}$ depends on *which* 500 people fell into your sample. Re-sample and you get 169.2, then 167.1, then
168.6. This spread of values $\bar{x}$ *could have taken* is the **sampling distribution of the statistic**,
the hinge everything hangs on.

The **Central Limit Theorem** says the sampling distribution of $\bar{x}$ is approximately normal, centered at
the true $\mu$, with spread that shrinks as $n$ grows. The **Law of Large Numbers** says $\bar{x}\to\mu$ as
$n$ grows. These were not abstract, they are the precise reason a sample can tell you anything about a
population at all. See [Law of Large Numbers & CLT](../advanced/lln-clt.md).

### Point estimates: your single best guess

A **point estimate** is a single number put forward as your best guess for the unknown parameter: "my
estimate of $\mu$ is 168 cm," or $\hat{p}=0.12$. One number. Useful, but it carries a quiet lie by omission:
**it gives no sense of how wrong it might be.** Since $\bar{x}$ is random, the probability it hits a continuous
$\mu$ exactly is essentially zero. That flaw motivates intervals.

### Confidence intervals: a guess with honesty attached

A **confidence interval** replaces the single dart throw with a range plus a statement of confidence: "I am
95% confident $\mu$ lies between 166.4 and 169.6." Almost every CI has the same shape:

$$\text{estimate} \;\pm\; (\text{margin of error})$$

The margin is built from the spread of the sampling distribution (the standard error) and how confident you
want to be (a multiplier). Wider confidence gives a wider interval. More data gives a tighter interval.

!!! warning "Interview trap"
    "95% confidence interval" does **NOT** mean "there is a 95% probability the true $\mu$ is inside this
    particular interval." In the frequentist view $\mu$ is **fixed**; once computed, the interval either
    contains it or does not, no probability left. The 95% refers to the **procedure's long-run hit rate**: if
    many teams each drew a sample and built an interval this way, about 95% of those intervals would contain
    $\mu$. You built one; you do not know if yours is a hit or a miss, only that the factory hits 95% of the
    time.

## §2 Point Estimator vs Point Estimate

A point estimator is the *rule* (a formula, a function of the data). A point estimate is the *number* you get
when you feed actual data into that rule. Estimator is the recipe; estimate is the dish you cooked tonight.

A **point estimator** is a function of the sample, it exists *before* you have seen any data, written in
generic random variables:

$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$$

The $X_i$ are capital, random variables, placeholders for "whatever the sample turns out to be." The estimator
is the *machine*: "add up the observations and divide by $n$." A **point estimate** is what comes out when
real numbers run through it. Plug in 165, 170, 168, 172, 169:

$$\bar{x} = \frac{165+170+168+172+169}{5} = 168.8$$

That **168.8** is the point estimate, a single, concrete, fixed number.

**The estimator is a random variable; the estimate is a fixed number.** The estimator $\bar{X}$ is random
because the $X_i$ are random, it *has a distribution*, an expected value, a variance. The estimate 168.8 is
just a number; the randomness already happened.

The payoff: **all the properties we care about belong to the estimator, not the estimate.** You ask of the
rule:

- Is it **unbiased**? Does $E[\bar{X}]=\mu$? (Does the rule hit the truth on average?)
- What is its **variance**? How much does the output bounce sample to sample?
- Is it **consistent**? As $n$ grows, does the output converge to $\mu$? (This is the LLN as an estimator property.)

You cannot ask whether 168.8 is "unbiased," the question is meaningless for one number. Bias, variance, and
consistency are properties of the *recipe*, evaluated over all the meals it could ever produce.

The estimator is a **throwing strategy**; the estimate is **one dart already in the board**. You can evaluate
a strategy: does it cluster near the bullseye (low bias)? Tightly grouped (low variance)? Once a single dart
has landed, those questions evaporate, the dart is just *there*. You cannot ask whether one dart is "low
variance." Variance was a property of the throwing motion. The hat notation carries this dual life: $\hat{p}$
as a formula "(successes)/n" is the **estimator** (random); $\hat{p}=0.12$ is the **estimate** (fixed). Same
symbol, context tells you which.

## Interview Questions

**Q1: What is the difference between a parameter and a statistic?**
A parameter describes the population and is fixed but unknown, like the true mean $\mu$ or proportion $p$. A
statistic is computed from a sample and is known but random, like $\bar x$ or $\hat p$, because a different
sample yields a different value. Estimation bridges the known-but-random statistic to the fixed-but-unknown
parameter.

**Q2: Why is the sample mean a random variable even though you get one number from it?**
Because the value depends on which units happened to fall into the sample; re-sampling gives a different
number. That distribution of values the sample mean could have taken is its sampling distribution, which the
central limit theorem makes approximately normal around $\mu$ with spread shrinking as $n$ grows. The single
number you compute is one realization of that random variable.

**Q3: Explain the estimator-versus-estimate distinction and why it matters.**
An estimator is a rule, a function of the random sample, so it is itself a random variable with a bias, a
variance, and a consistency property. An estimate is the concrete number that rule produces on a particular
data set. Properties like unbiasedness belong to the estimator, because asking whether a single fixed number
is unbiased is meaningless.

**Q4: What does a 95% confidence level actually refer to?**
It refers to the long-run hit rate of the procedure, not the probability that a particular interval contains
the parameter. In the frequentist view the parameter is fixed, so a computed interval either contains it or
not. If many samples were drawn and intervals built the same way, about 95% of them would contain the true
value.
