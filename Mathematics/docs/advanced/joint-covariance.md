# Joint Distributions & Covariance

Real problems involve several random variables at once. This page describes their joint behavior through
three lenses (joint, marginal, conditional), then measures whether two variables move together with
covariance and its standardized cousin, correlation.

!!! tip "Rapid Recall"
    A joint distribution is one landscape over $(x,y)$; the marginal flattens it onto one axis and the
    conditional slices it and renormalizes. Independence means the joint factors, $f(x,y)=f_X(x)f_Y(y)$, and
    non-rectangular support forces dependence even with constant density. Covariance
    $E(XY)-E(X)E(Y)$ measures linear co-movement and is bilinear; it drives $\mathrm{Var}(X+Y)=
    \mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)$. Correlation standardizes covariance into $[-1,1]$.
    Independent implies uncorrelated, but never the reverse: $Y=X^2$ with $X$ standard normal has zero
    covariance yet total dependence, because covariance only sees linear structure.

## §4 Joint, Marginal & Conditional Distributions

**Intuition, one landscape, three lenses.** Picture probability spread over a 2D plane: a landscape over
every $(x,y)$. Three questions:

- **Joint**, both values at once: the full landscape.
- **Marginal**, one variable alone: flatten the landscape onto one axis.
- **Conditional**, one variable fixed: take a slice through the landscape.

### Joint definitions

| Object | Discrete | Continuous |
|---|---|---|
| Joint CDF | $F(x,y) = P(X\le x,\ Y\le y)$ | $F(x,y) = P(X\le x,\ Y\le y)$ |
| Joint mass/density | $P(X=x,Y=y)$ | $f(x,y) = \dfrac{\partial}{\partial x\,\partial y}F(x,y)$ |
| Probability of region $B$ | sum over $B$ | $\iint_B f(x,y)\,dx\,dy$ |

Same parallel as 1D: mass heights are probabilities; density heights integrate (now a **volume**) to
probability.

### Marginalizing, erase a variable

$$P(X=x) = \sum_y P(X=x,Y=y) \qquad f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy$$

Account for every value the unwanted variable could take and sum it out. In a table, marginals are the
**row and column sums** in the margins, hence the name.

### Independence, the multiplication test

$$F(x,y) = F_X(x)F_Y(y) \;\Longleftrightarrow\; f(x,y) = f_X(x)f_Y(y)$$

If the joint **factors** into "X-only times Y-only," they are independent.

!!! warning "Trap, non-rectangular support forces dependence"
    If the region where density is nonzero is not a clean box (a disc, a triangle), the variables are
    automatically dependent: the allowed range of one depends on the other's value, even if the density is
    constant.

### Conditional, slice and renormalize

$$f_{Y|X}(y\mid x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

Fix $X=x$ (a vertical slice), then divide by the marginal $f_X(x)$ so the slice integrates to 1 again. Same
"joint over marginal" as $P(A|B)=P(A\cap B)/P(B)$. **Marginalize versus condition:** marginalizing *erases* a
variable (flatten); conditioning *fixes* it to a known value (slice). Independence, restated:
$f_{Y|X}(y\mid x) = f_Y(y)$, conditioning tells you nothing.

### The disc example (support-shape dependence)

Uniform on the unit disc $x^2+y^2\le 1$, density $1/\pi$. Marginal of $X$:

$$f_X(x) = \int_{-\sqrt{1-x^2}}^{\sqrt{1-x^2}} \frac1\pi\,dy = \frac{2}{\pi}\sqrt{1-x^2}$$

Not uniform, a semicircle. Conditional:

$$f_{Y|X}(y\mid x) = \frac{1/\pi}{\frac{2}{\pi}\sqrt{1-x^2}} = \frac{1}{2\sqrt{1-x^2}} \;\Rightarrow\; Y\mid X \sim \text{Unif}\!\left(-\sqrt{1-x^2},\sqrt{1-x^2}\right)$$

Flat in $y$, but its **width depends on $x$**, so $X,Y$ are dependent despite a constant density. The
dependence hides in the support.

### 2D LOTUS and the independence-expectation rule

$$E[g(X,Y)] = \int\!\!\int g(x,y)\,f_{X,Y}(x,y)\,dx\,dy$$

If independent, expectations factor, the fact MGF sums relied on:

$$E(XY) = \int\!\!\int xy\,f_X(x)f_Y(y)\,dx\,dy = E(X)E(Y)$$

(Converse false: $E(XY)=E(X)E(Y)$ means *uncorrelated*, weaker than independent.)

A worked example with joint density 1 on the unit square: $E|X-Y| = \int_0^1\!\int_0^1 |x-y|\,dx\,dy$. The two
triangles ($x>y$ and $x\le y$) are mirror images, so it equals $2\int_0^1\!\int_y^1 (x-y)\,dx\,dy$.

$$\int_y^1 (x-y)\,dx = \left[\tfrac{x^2}{2}-yx\right]_y^1 = \tfrac12 - y + \tfrac{y^2}{2}$$
$$2\int_0^1\!\left(\tfrac12 - y + \tfrac{y^2}{2}\right)dy = 2\left(\tfrac12-\tfrac12+\tfrac16\right) = \tfrac13$$

**Poisson splitting (chicken-and-egg).** With a Poisson count $N=i+j$ split into successes and failures, only
$N=i+j$ survives the sum: $P(X=i,Y=j) = P(X=i\mid N=i+j)\,P(N=i+j)$.

$$= \binom{i+j}{i}p^i q^j \cdot e^{-\lambda}\frac{\lambda^{i+j}}{(i+j)!}, \quad q=1-p$$

The $(i+j)!$ cancels; split $\lambda^{i+j}$ and $e^{-\lambda}=e^{-\lambda p}e^{-\lambda q}$:

$$= \underbrace{\left(e^{-\lambda p}\frac{(\lambda p)^i}{i!}\right)}_{\text{Pois}(\lambda p)}\underbrace{\left(e^{-\lambda q}\frac{(\lambda q)^j}{j!}\right)}_{\text{Pois}(\lambda q)}$$

The joint factors, so the two counts are independent Poissons.

**Where you see it.** *Plain probability:* covariance and correlation are defined through joints; the
chicken-egg problem; order statistics (max and min of a sample) are joint-distribution problems.
*Machine learning,* more than almost any other topic. Supervised learning *is* the joint $p(x,y)$: a
generative model learns the joint directly, a discriminative model learns the conditional $p(y\mid x)$, the
marginal $p(x)$ is the data distribution. Bayes' rule $p(y\mid x)=p(x\mid y)p(y)/p(x)$ is a statement about
joints and conditionals; **Naive Bayes** makes a *conditional independence* assumption (features factor given
the label) to make the joint tractable. Latent-variable models (mixtures, VAEs, topic models) **marginalize**
out the hidden $z$: $p(x)=\int p(x,z)\,dz$. **Graphical models** (Bayesian networks, Markov random fields)
write big joints as products of small conditional and marginal pieces, exploiting independence.
**Autoregressive models and LLMs** factor the joint over a sequence via the chain rule
$p(x_1,\dots,x_n)=\prod_t p(x_t\mid x_{<t})$, every next-token step a conditional slice. The **multivariate
normal** is the most-used continuous joint, and its conditionals and marginals are again Gaussian (the
property powering Gaussian Processes and Kalman filters). The marginalize-versus-condition split maps
directly: discriminative ML **conditions**, generative ML models the **joint** and can **marginalize**.

## §5 Covariance & Correlation

Variance measures one variable's wobble; *covariance* measures whether two wobble together:

$$\text{Cov}(X,Y) = E\big[(X-EX)(Y-EY)\big] = E(XY) - E(X)E(Y)$$

**Intuition, average co-deviation.** $(X-EX)$ is "how far X is from its mean right now." Multiply the two
deviations: same side of their means gives a positive product; opposite sides gives negative; no pattern
cancels to 0. The second formula (mean of product minus product of means) is what you actually compute.

### Properties (covariance is bilinear)

- $\text{Cov}(X,X) = \text{Var}(X)$, variance is a special case.
- $\text{Cov}(X,Y) = \text{Cov}(Y,X)$, symmetric.
- $\text{Cov}(X,Y) = E(XY)-E(X)E(Y)$, the compute formula.
- $\text{Cov}(X,c) = 0$, a constant never wobbles.
- $\text{Cov}(cX,Y) = c\,\text{Cov}(X,Y)$, scaling (the unit problem).
- $\text{Cov}(X,Y+Z) = \text{Cov}(X,Y)+\text{Cov}(X,Z)$, distributes.

$$\text{Cov}\Big(\sum_i a_i X_i,\ \sum_j b_j Y_j\Big) = \sum_{i,j} a_i b_j\,\text{Cov}(X_i,Y_j)$$

### Why it matters, variance of sums

$$\text{Var}(X+Y) = \text{Var}(X) + 2\,\text{Cov}(X,Y) + \text{Var}(Y)$$
$$\text{Var}\Big(\textstyle\sum X_i\Big) = \sum \text{Var}(X_i) + 2\sum_{i<j}\text{Cov}(X_i,X_j)$$

Cross terms vanish only when uncorrelated, that is when "variance of a sum equals sum of variances," the rule
the CLT leans on. Positive covariance inflates the spread of a sum; negative dampens it (portfolio
diversification).

!!! warning "Trap, uncorrelated is not independent"
    Independent implies $\text{Cov}=0$, but **not the reverse**. Let $Z\sim\mathcal{N}(0,1),\ X=Z,\ Y=Z^2$:
    $$\text{Cov}(X,Y) = E(Z^3) - E(Z)E(Z^2) = 0 - 0\cdot 1 = 0$$
    Yet $Y$ is literally a function of $X$. Covariance only sees **linear** structure; the parabola is
    invisible to it.

### Correlation, covariance you can interpret

Covariance's magnitude is unit-dependent. Fix it by standardizing:

$$\text{Cor}(X,Y) = \frac{\text{Cov}(X,Y)}{\text{SD}(X)\,\text{SD}(Y)}, \qquad |\text{Cor}(X,Y)| \le 1$$

It is the covariance of the standardized variables $(X-EX)/\text{SD}(X)$, always in $[-1,1]$, so the magnitude
finally means "strength." Proof of the bound, from $\text{Var}\ge 0$ on standardized $X,Y$ with $\rho=\text{Cor}$:

$$\text{Var}(X+Y) = 2+2\rho \ge 0,\qquad \text{Var}(X-Y) = 2-2\rho \ge 0 \;\Rightarrow\; -1\le\rho\le 1$$

### Computing correlation between two dataset columns

Swap expectations for sample averages over your $n$ rows (Pearson correlation):

$$r = \frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum (x_i-\bar x)^2}\,\sqrt{\sum (y_i-\bar y)^2}}$$

Steps: (1) column means, (2) subtract to get deviations, (3) multiply paired deviations and sum (covariance),
(4) the two standard deviations, (5) divide. In practice:

| Code | Returns |
|---|---|
| `df["x"].corr(df["y"])` | single-pair Pearson correlation |
| `df.cov()` | covariance matrix (raw, scale-dependent) |
| `df.corr()` | correlation matrix in $[-1,1]$, the EDA heatmap |

**Where you see it.** Multicollinearity and feature selection, PCA (eigendecomposition of the covariance
matrix), the multivariate normal (parameterized by $\Sigma$), and decorrelation objectives in representation
learning. Reminder: decorrelating features does **not** make them independent.

> **Interview compression.** Covariance is $E(XY)-E(X)E(Y)$; sign gives direction, magnitude is uninterpretable
> (scale-dependent). Correlation is standardized covariance, always $[-1,1]$. Independent implies uncorrelated,
> never the reverse ($Y=X^2$). $\text{Var}(X+Y)=\text{Var}(X)+\text{Var}(Y)+2\text{Cov}(X,Y)$ is why
> independence makes variance additive.

## Interview Questions

**Q1: What is the difference between marginalizing and conditioning?**
Marginalizing erases a variable by summing or integrating it out, flattening the joint landscape onto the
remaining axis to get $p(x)$. Conditioning fixes a variable to a known value, taking a slice of the landscape
and renormalizing so it integrates to one, giving $p(y\mid x)$. Discriminative models condition, while
generative models learn the joint and can marginalize.

**Q2: Give a concrete example where two variables are uncorrelated but dependent.**
Let $Z$ be standard normal, $X=Z$, and $Y=Z^2$. Then $\text{Cov}(X,Y)=E(Z^3)-E(Z)E(Z^2)=0$, so they are
uncorrelated, yet $Y$ is a deterministic function of $X$, hence completely dependent. Covariance only detects
linear association, so the quadratic relationship is invisible to it.

**Q3: How can two variables be dependent even when their joint density is constant?**
When the support is not a rectangle. For a uniform distribution on the unit disc, the density is the constant
$1/\pi$, but the allowed range of $Y$ depends on $X$ through $\sqrt{1-x^2}$, so the conditional distribution
changes with $x$. The dependence lives in the shape of the support, not in the density values.

**Q4: Why does $\mathrm{Var}(X+Y)$ include a covariance term, and when does it drop out?**
Expanding the variance of a sum gives $\mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)$, where the cross
term measures how the two co-vary. It vanishes exactly when $X$ and $Y$ are uncorrelated, which is when
variance becomes additive, the assumption underlying the law of large numbers and the central limit theorem.
