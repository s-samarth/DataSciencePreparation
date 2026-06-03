# Conditional Expectation & Inequalities

Conditional expectation is the best guess of one variable given another, and it turns out to be the
optimal predictor that every regression model estimates. The inequalities then bound probabilities and
expectations using only partial information, culminating in the tool that proves the law of large numbers.

!!! tip "Rapid Recall"
    $E(Y\mid X)$ is itself a random variable, the function of $X$ that minimizes mean-squared error, which is
    why supervised learning works. Adam's law (iterated expectation) says $E(E(Y\mid X))=E(Y)$, and Eve's law
    splits total variance into unexplained plus explained, the decomposition behind $R^2$. Geometrically,
    $E(Y\mid X)$ is the projection of $Y$ onto functions of $X$. The four inequalities bound the unknown:
    Cauchy-Schwarz gives $|\text{Cor}|\le 1$, Jensen relates $E(g(X))$ to $g(E(X))$ for convex $g$, Markov
    bounds tail mass by the mean, and Chebyshev bounds it by the variance, proving the weak law of large
    numbers.

## §6 Conditional Expectation

Plain $E(Y)$ is one number. $E(Y\mid X)$ asks: knowing $X$, what is the best guess for $Y$? Since the answer
depends on what $X$ turned out to be, **$E(Y\mid X)$ is itself a random variable**, a function of $X$.

$$g(x) = E(Y\mid X=x) \;\Rightarrow\; E(Y\mid X) = g(X)$$

$E(Y\mid X=x)$ (X pinned) is a number; $E(Y\mid X)$ (X free) is random. Sub-rule: a function of X is a known
constant once you condition on X, so $E(h(X)\mid X) = h(X)$.

### Linearity still holds

Conditional expectation is linear like ordinary expectation:

$$E(X+Y\mid X) = E(X\mid X) + E(Y\mid X)$$

For two i.i.d. $\text{Pois}(\lambda)$ variables, this gives $E(X+Y\mid X) = X + E(Y) = X + \lambda$. The
$X\mid X$ term collapses to $X$ (you know it), and the $Y\mid X$ term collapses to $E(Y)=\lambda$ because $Y$
is independent of $X$ (knowing $X$ says nothing about $Y$).

### The two flagship laws

**Adam's Law (iterated expectation).**

$$E\big(E(Y\mid X)\big) = E(Y)$$

Average the conditional average over X and you recover plain $E(Y)$. **Use:** divide-and-conquer for a hard
$E(Y)$, condition on a convenient X, compute the easy inner expectation, average out. Broken stick: break
$X\sim\text{Unif}(0,1)$, then half the rest; $E(Y\mid X)=X/2$, so $E(Y)=E(X/2)=1/4$.

**Eve's Law (total variance).**

$$\text{Var}(Y) = E\big(\text{Var}(Y\mid X)\big) + \text{Var}\big(E(Y\mid X)\big)$$

**Unexplained** (average leftover wobble given X) plus **explained** (how much the conditional mean moves with
X). This is the explained/unexplained split behind $R^2$, bias-variance, and ANOVA.

### Other properties

- $E(h(X)Y\mid X) = h(X)E(Y\mid X)$, functions of X pull out.
- $E(Y\mid X) = E(Y)$ if independent (converse false).
- Residual uncorrelated with any $h(X)$: $\text{Cov}(Y-E(Y\mid X),\ h(X)) = 0$.

**The geometry (the illuminating picture).** Treat random variables as vectors with inner product
$\langle X,Y\rangle = E(XY)$. Then $E(Y\mid X)$ is the **projection of $Y$ onto the space of all functions of
$X$**, the closest approximation to $Y$ you can build from $X$ alone. The residual $Y-E(Y\mid X)$ is orthogonal
to that space. This is *why* $E(Y\mid X)$ is the function of X that minimizes mean-squared error, the single
most important fact here.

### Directional asymmetry

$Z\sim\mathcal{N}(0,1),\ X=Z,\ Y=Z^2$: $E(Y\mid X)=Z^2=Y$ (X pins Y), but $E(X\mid Y)=E(Z\mid Z^2)=0$
($\pm\sqrt{a}$ equally likely). Conditioning is directional.

### Conditional variance

The same idea applied to spread, the variance of $Y$ computed *within* the slice where $X$ is fixed (itself a
function of $X$):

$$\text{Var}(Y\mid X) = E(Y^2\mid X) - \big(E(Y\mid X)\big)^2 = E\big((Y-E(Y\mid X))^2 \mid X\big)$$

This is the ingredient Eve's Law averages over.

**Its real significance.** It is the **optimal MSE predictor**, every regression model is estimating
$E(Y\mid X)$; that is why supervised learning works. Adam's Law powers latent-variable computation (the EM
E-step). Eve's Law is the variance decomposition behind $R^2$. In RL, the value function
$V(s)=E(\text{return}\mid s)$ is a conditional expectation and the Bellman equation is iterated expectation in
disguise.

## §7 The Inequalities

Each bounds a probability or expectation using *limited* information. More info gives a tighter bound.

### Cauchy-Schwarz

$$|E(XY)| \le \sqrt{E(X^2)\,E(Y^2)}$$

The probability version of $|\mathbf{a}\cdot\mathbf{b}| \le |\mathbf{a}||\mathbf{b}|$. **Use:** it is *why*
$|\text{Cor}(X,Y)|\le 1$, the guarantee under every correlation coefficient.

### Jensen's Inequality

$$E(g(X)) \ge g(E(X)) \;\text{ if } g \text{ convex} \quad(\le \text{ if concave})$$

**Intuition.** For a bowl-shaped (convex, $g''>0$) function, the upward bend lifts high points more than it
lowers low ones, so averaging-the-outputs sits above output-of-the-average. Bigger spread in X gives a bigger
gap. Consequences: $E(1/X)\ge 1/E(X)$ and $E(\ln X)\le \ln E(X)$. **Use:** the engine behind the **ELBO** in
variational inference and VAEs (one application of Jensen to the concave log), and why "average of a ratio is
not the ratio of averages" bites in metrics.

### Markov's Inequality

$$P(|X|\ge a) \le \frac{E|X|}{a}, \quad a>0,\ X\ge 0$$

A nonnegative variable cannot put too much mass far above its mean (that mass would drag the mean up).
**Concrete example:** in a group of 100 people it is perfectly possible that at least 95% are younger than the
group's average age, but it is *not* possible for 50% to be older than *twice* the average (that much mass
sitting that high would force a higher mean). Crudest bound, needs only the mean. **Use:** the stepping stone
to Chebyshev and learning-theory tail bounds.

### Chebyshev's Inequality

$$P(|X-\mu|\ge a) \le \frac{\text{Var}(X)}{a^2} \quad\Longleftrightarrow\quad P(|X-\mu|\ge c\,\text{SD}(X)) \le \frac{1}{c^2}$$

At most $1/4$ of any distribution lies beyond $2$ standard deviations, at most $1/9$ beyond $3$. Markov applied
to $(X-\mu)^2$. **Key:** works for *any* finite-variance distribution (no normality), far looser than
68-95-99.7 but universal. It is the tool that **proves the Weak Law of Large Numbers**.

> **Hierarchy.** Markov (mean) to Chebyshev (mean plus variance, tighter) to full distribution (compute
> exactly). The more you know, the tighter you bound.

A worked Jensen example, with $g(x)=1/x$ convex for $x>0$:

$$E\!\left(\tfrac{1}{\hat p}\right) \ge \tfrac{1}{E(\hat p)} = \tfrac{1}{0.5} = 2$$

## Interview Questions

**Q1: Why is $E(Y\mid X)$ called the best predictor of $Y$ given $X$?**
Because, viewing random variables as vectors with inner product $E(XY)$, the conditional expectation is the
orthogonal projection of $Y$ onto the space of all functions of $X$, so it is the function of $X$ closest to
$Y$ in mean-squared error. The residual is orthogonal to every function of $X$. This is exactly why regression,
which estimates $E(Y\mid X)$, minimizes squared error.

**Q2: State Adam's law and Eve's law and what each is used for.**
Adam's law (iterated expectation) says $E(E(Y\mid X))=E(Y)$, useful for computing a hard expectation by
conditioning on a convenient variable and averaging. Eve's law (total variance) says
$\mathrm{Var}(Y)=E(\mathrm{Var}(Y\mid X))+\mathrm{Var}(E(Y\mid X))$, splitting variance into unexplained and
explained parts, the decomposition behind $R^2$ and the bias-variance tradeoff.

**Q3: How does Chebyshev's inequality prove the weak law of large numbers?**
Apply Chebyshev to the sample mean, whose variance is $\sigma^2/n$. Then
$P(|\bar X_n-\mu|>c)\le \sigma^2/(nc^2)$, which goes to zero as $n\to\infty$ for any fixed $c$. That is exactly
the weak law: for each large $n$, the sample mean is probably close to the true mean, and the bound holds for
any finite-variance distribution.

**Q4: Give an example of Jensen's inequality biting in practice.**
For a convex function like $1/x$, $E(1/X)\ge 1/E(X)$, so averaging a ratio is not the same as the ratio of
averages. In machine learning the log is concave, so $E(\log X)\le \log E(X)$, which is the inequality that
yields the evidence lower bound (ELBO) used to train variational autoencoders and other variational models.
