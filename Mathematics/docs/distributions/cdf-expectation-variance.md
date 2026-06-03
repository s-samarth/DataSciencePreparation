# CDF, Expectation & Variance

The cumulative distribution function accumulates probability, expectation summarizes a distribution's
center, and variance measures its spread. These three tools work identically across discrete and continuous
worlds, and together they turn a distribution into the numbers that decisions and loss functions ride on.

!!! tip "Rapid Recall"
    The CDF $F_X(x)=P(X\le x)$ is a staircase for discrete variables (jump size equals the PMF) and a smooth
    integral of the density for continuous ones; it is nondecreasing, right-continuous, and limits to 0 and
    1. Expectation $E(X)=\sum x\,P(X=x)$ is a probability-weighted average, and its most important property is
    linearity, $E(X+Y)=E(X)+E(Y)$, which holds even under dependence because the joint structure never
    enters. Variance $\text{Var}(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2$ is the average squared distance from the
    mean, squared so deviations do not cancel, and standard deviation is its square root in the original
    units.

## §7 Cumulative Distribution Function (CDF)

$$F_X(x)=P(X\le x)$$

**Discrete: a staircase.** Flat between values, jumps at each value, jump size equals the PMF there:

$$F_X(x)=\sum_{t\le x}P(X=t)$$

**Continuous: a smooth rising curve.** No jumps; mass is smeared, so the CDF integrates the density:

$$F_X(x)=\int_{-\infty}^{x}f_X(t)\,dt$$

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 300 210" role="img" aria-label="Discrete CDF staircase">
          <text x="150" y="16" fill="#5ad1c4" font-family="sans-serif" font-size="12" text-anchor="middle">Discrete CDF — staircase</text>
          <line x1="30" y1="180" x2="285" y2="180" stroke="#2a323d"/>
          <line x1="30" y1="30" x2="30" y2="180" stroke="#2a323d"/>
          <text x="20" y="34" fill="#7d8a98" font-size="10" font-family="sans-serif">1</text>
          <text x="20" y="184" fill="#7d8a98" font-size="10" font-family="sans-serif">0</text>
          <g stroke="#ffb347" stroke-width="3" fill="none">
            <line x1="30" y1="180" x2="90" y2="180"/>
            <line x1="90" y1="150" x2="150" y2="150"/>
            <line x1="150" y1="95" x2="210" y2="95"/>
            <line x1="210" y1="40" x2="270" y2="40"/>
          </g>
          <g fill="#ffb347"><circle cx="90" cy="150" r="3.5"/><circle cx="150" cy="95" r="3.5"/><circle cx="210" cy="40" r="3.5"/></g>
          <g fill="#161b22" stroke="#ffb347" stroke-width="2"><circle cx="90" cy="180" r="3.5"/><circle cx="150" cy="150" r="3.5"/><circle cx="210" cy="95" r="3.5"/></g>
        </svg>
<svg viewBox="0 0 300 210" role="img" aria-label="Continuous CDF smooth curve">
          <text x="150" y="16" fill="#5ad1c4" font-family="sans-serif" font-size="12" text-anchor="middle">Continuous CDF — smooth</text>
          <line x1="30" y1="180" x2="285" y2="180" stroke="#2a323d"/>
          <line x1="30" y1="30" x2="30" y2="180" stroke="#2a323d"/>
          <text x="20" y="34" fill="#7d8a98" font-size="10" font-family="sans-serif">1</text>
          <text x="20" y="184" fill="#7d8a98" font-size="10" font-family="sans-serif">0</text>
          <path d="M30,178 C90,176 120,150 150,105 C180,60 210,36 270,33" stroke="#9d8cff" stroke-width="3" fill="none"/>
        </svg>
<figcaption>Both CDFs run from 0 (far left) to 1 (far right). Discrete: jumps at each value, where the jump equals the PMF. Continuous: a gradual climb, the integral of the density.</figcaption>
</figure>

**The three properties of a valid CDF.**

- **Monotonically nondecreasing**, a running total only adds mass, never removes it.
- **Right-continuous**, the "$\le$" includes the point $x$, so a jump's mass is counted at $x$ (the step is drawn filled at the top).
- **Limits to 0 and 1**:

$$\lim_{x\to-\infty}F_X(x)=0,\qquad \lim_{x\to+\infty}F_X(x)=1$$

**Independence, now for random variables.**

$$P(X\le x,\,Y\le y)=P(X\le x)\,P(Y\le y)$$

Discrete equivalent: $P(X=x,\,Y=y)=P(X=x)\,P(Y=y)$. Knowing $Y$ tells you nothing about $X$; the joint
factors into the product.

A worked staircase, valid because it is nondecreasing $0\to0.2\to0.7\to1.0$ and ends at 1:

$$F_X(x)=\begin{cases}0 & x<0\\ 0.2 & 0\le x<1\\ 0.7 & 1\le x<2\\ 1.0 & x\ge 2\end{cases}$$

$$P(0<X\le2)=F_X(2)-F_X(0)=1.0-0.2=0.8$$

Cross-check: $P(X=1)+P(X=2)=0.5+0.3=0.8$. (Subtracting $F_X(0)$ removes the $X=0$ mass the strict "$0<$"
excludes.)

## §8 Expected Value & Its Properties

$$E(X)=\sum_{x\in\text{Im}(X)} x\,P(X=x)$$

A probability-weighted average: each value times its probability, summed. $\text{Im}(X)$ is the set of values
$X$ can take.

**The weighted-average intuition.** Average of $1,1,1,1,1,3,3,5$ regrouped by value:

$$\tfrac{5}{8}\cdot1+\tfrac{2}{8}\cdot3+\tfrac{1}{8}\cdot5$$

That *is* the expectation formula, the average with probabilities as weights.

**Bernoulli and the indicator bridge.**

$$X\sim\text{Bern}(p):\quad E(X)=1\cdot p+0\cdot(1-p)=p$$

If $X$ is the indicator of event $A$, then $X\sim\text{Bern}(P(A))$ so $E(X)=P(A)$.

### Linearity of expectation, the most important property

$$E(X+Y)=E(X)+E(Y) \qquad E(cX)=c\,E(X)$$

**Why it is true:**

$$E(X+Y)=\sum_s (X(s)+Y(s))P(\{s\})=\sum_s X(s)P(\{s\})+\sum_s Y(s)P(\{s\})=E(X)+E(Y)$$

Splitting a sum into two needs no assumption about how $X,Y$ relate, the joint structure never enters.

**Payoff: $E[\text{Bin}(n,p)]=np$ in one line.**

$$X=X_1+\dots+X_n,\ X_i\sim\text{Bern}(p)\ \Rightarrow\ E(X)=\underbrace{p+\dots+p}_{n}=np$$

And under dependence, the expected aces in a 5-card hand:

$$E(X)=5\cdot P(\text{a given card is an ace})=5\cdot\tfrac{4}{52}=\tfrac{5}{13}$$

- The one-number summary of a distribution's center; decisions ride on averages (expected revenue, loss, latency).
- **Every ML objective**: training minimizes *expected* loss (risk). Empirical risk equals average loss equals empirical expectation. `loss.mean()` is an expectation.
- Variance, covariance (PCA), entropy and cross-entropy are all expectations.
- **RL**: the value of a state is expected cumulative reward.

## §15 Variance & Standard Deviation

**Why we need it.** "Average return 8%" describes both a steady bond and a wild bet that is usually -50% or
+66%. The mean hides risk; variance exposes it. In ML: the bias-variance tradeoff, gradient noise, prediction
uncertainty, metric volatility across runs.

$$\text{Var}(X)=E\!\left[(X-E[X])^2\right]$$

The average squared distance from the mean.

- **Why not $E[X-E[X]]$?** It is always 0, since positive and negative deviations cancel:

$$E[X-E[X]]=E[X]-E[X]=0$$

Squaring kills the cancellation.

- **Why not $E|X-E[X]|$?** It is valid (mean absolute deviation) but absolute values are not differentiable at 0 and are algebraically ugly. Squaring is smooth and plays beautifully with calculus and linearity.

**The computational form (the one you use).**

$$\text{Var}(X)=E[X^2]-(E[X])^2$$

The mean of the squares minus the square of the mean. Derivation via linearity (treat $E[X]$ as the constant
it is):

$$\text{Var}(X)=E[X^2-2X\,E[X]+(E[X])^2]=E[X^2]-2(E[X])^2+(E[X])^2=E[X^2]-(E[X])^2$$

**Standard deviation.**

$$\text{SD}(X)=\sqrt{\text{Var}(X)}$$

Variance is in *squared* units (dollars squared); the square root returns interpretable original units, the
"typical distance from the mean." Variance for math, standard deviation for human-readable spread.

## Interview Questions

**Q1: State the three properties that define a valid CDF.**
A CDF is monotonically nondecreasing because accumulating probability never removes mass; it is
right-continuous because $P(X\le x)$ includes the point $x$, so a discrete jump's mass is counted at $x$; and
it limits to 0 as $x\to-\infty$ and to 1 as $x\to+\infty$. For a discrete variable it is a staircase whose
jumps equal the PMF, and for a continuous variable it is the integral of the density.

**Q2: Why does linearity of expectation hold even when variables are dependent?**
Because $E(X+Y)=\sum_s (X(s)+Y(s))P(\{s\})$ splits into two separate sums by simple rearrangement, and that
split never references the joint distribution of $X$ and $Y$. Independence is required for products like
$E(XY)$ but not for sums, which is why decomposing a dependent count into indicators and summing their means
always works.

**Q3: Derive the computational form of variance and explain why we square the deviations.**
Starting from $E[(X-E[X])^2]$ and expanding with linearity gives $E[X^2]-2(E[X])^2+(E[X])^2=E[X^2]-(E[X])^2$.
We square rather than take the raw deviation because $E[X-E[X]]$ is identically zero, and we prefer squaring
over absolute value because the square is smooth and differentiable, which suits calculus and optimization.

**Q4: What does standard deviation give you that variance does not?**
Standard deviation is the square root of variance, returning the spread to the original units of the variable
rather than squared units. That makes it directly interpretable as the typical distance of an observation from
the mean, whereas variance is the more convenient quantity for algebra and proofs.
