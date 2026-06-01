# Linear Regression

Linear regression is the entry point to supervised learning and the model every other one is measured against. This page covers what the model is, the squared-error loss and its closed-form solution, the trace derivation of the normal equation, the probabilistic reason the error is squared rather than absolute, and the locally weighted variant that fits a fresh line around every query.

!!! tip "Rapid Recall"
    Fit the line that minimizes total squared vertical miss. The slope is covariance(x,y) over variance(x), and the intercept forces the line through the mean point. In matrix form the closed-form solution is the normal equation \(\mathbf w=(X^\top X)^{-1}X^\top\mathbf y\), which costs \(O(d^3)\) to invert, so huge feature counts fall back to gradient descent. Squared error is not arbitrary: it is the maximum-likelihood fit under Gaussian noise, which is why absolute error (the Laplace MLE) is the more outlier-robust alternative. Locally weighted regression drops the single global line and re-solves a distance-weighted fit at each query, trading fixed model size for keeping all the data.

## §1 The core idea

Linear regression is **fitting the best straight line (or flat plane) through a cloud of points.** "Best" means the line where the total vertical miss-distance, squared, is as small as possible.

Why *vertical* distance, not perpendicular? Because we predict \(y\) *from* \(x\) — the error lives entirely in the \(y\) direction. Why *squared*? Two practical reasons: squaring kills the sign (a \(+3\) and a \(-3\) miss both count as badness and don't cancel), and it punishes big misses disproportionately (off by 10 is *more* than twice as bad as off by 5: 100 vs 25), so the line positions itself to avoid large errors.

> **Cricket analogy.** A coach judging a bowler's line and length over an over does not care about the *direction* of the miss (too wide vs down leg are both bad), and a delivery that is wildly off is disproportionately worse than one slightly off, a full toss is more than twice as punishable as a marginally short ball. So he mentally *squares* the misses. The "best line and length" minimizing those squared misses across six balls is exactly the regression line.

## §2 The math: model, loss, closed form

### The model

One feature: \(\hat{y} = wx + b\), where \(w\) is the slope, \(b\) the intercept, \(\hat y\) the prediction. Many features:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b = \mathbf{w}^\top \mathbf{x} + b$$

### The loss (Mean Squared Error)

Average, over all \(n\) points, of (actual − predicted) squared:

$$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2 = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - (w x_i + b)\big)^2$$

### The closed-form solution

Because MSE is a bowl (a paraboloid in the weights), its minimum is exactly where its slope is zero. Take the derivative, set to zero, solve. For one feature:

$$w = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i}(x_i - \bar{x})^2}, \qquad b = \bar{y} - w\bar{x}$$

Read \(w\): numerator is how \(x\) and \(y\) move together (covariance), denominator is how \(x\) varies alone (variance). So **slope = covariance(x,y) / variance(x)** — literally "co-movement per unit of x-spread." And \(b=\bar y - w\bar x\) forces the line through the mean point \((\bar x,\bar y)\), the centre of mass of the data.

### The matrix form and Normal Equation

Stack one example per row; prepend a column of 1s to absorb the intercept as weight \(w_0\):

$$X \in \mathbb{R}^{n\times(d+1)},\quad \mathbf{w}\in\mathbb{R}^{(d+1)\times1},\quad \mathbf{y}\in\mathbb{R}^{n\times1}$$

Predictions for the whole batch in one matrix multiply: \(\hat{\mathbf{y}} = X\mathbf{w}\) (this *is* the vectorization — no Python loop). Loss: \(L(\mathbf w)=\|\mathbf y - X\mathbf w\|^2\). Setting the gradient to zero gives the *Normal Equation*:

$$\mathbf{w} = (X^\top X)^{-1}X^\top \mathbf{y}$$

Inverting the \((d+1)\times(d+1)\) matrix \(X^\top X\) costs \(O(d^3)\) — which is why, for huge feature counts, we fall back to gradient descent.

## §3 The normal equation via traces (CS229)

Same destination \(\theta=(X^\top X)^{-1}X^\top y\), reached through trace identities. Cost in matrix form (the \(\tfrac12\) cancels the differentiation 2):

$$J(\theta) = \frac{1}{2}(X\theta - y)^\top(X\theta - y)$$

Because \(J\) is a scalar, \(J=\operatorname{tr}J\), which lets us apply trace identities: \(\operatorname{tr}(AB)=\operatorname{tr}(BA)\), \(\nabla_A\operatorname{tr}(AB)=B^\top\), and \(\nabla_A\operatorname{tr}(ABA^\top C)=CAB+C^\top AB^\top\). Expanding and differentiating term by term:

$$\nabla_\theta J(\theta) = X^\top X\theta - X^\top y$$

Set to zero (valid because \(J\) is convex — Hessian \(X^\top X\succeq0\)):

$$X^\top X\theta = X^\top y \;\Rightarrow\; \theta = (X^\top X)^{-1}X^\top y$$

!!! note "Interview one-liner"
    "Write cost as \(\tfrac12\|X\theta - y\|^2\), take the gradient (trace identities or matrix calculus), get \(X^\top X\theta - X^\top y\), set to zero, solve — global min because the cost is convex." The trace machinery is just the rigorous way to differentiate the quadratic form.

## §4 Probabilistic view: why squared error?

Assume \(y^{(i)} = \theta^\top x^{(i)} + \epsilon^{(i)}\) with IID Gaussian noise \(\epsilon^{(i)}\sim\mathcal N(0,\sigma^2)\). Gaussian is *principled*, not arbitrary: total error is a sum of many small independent unmodeled effects, and by the CLT a sum tends to Gaussian regardless of each piece's shape. Then:

$$p(y^{(i)} \mid x^{(i)};\theta) = \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(y^{(i)} - \theta^\top x^{(i)})^2}{2\sigma^2}\right)$$

The likelihood (product, by independence) and its log:

$$L(\theta) = \prod_{i=1}^m p(y^{(i)}\mid x^{(i)};\theta), \qquad \ell(\theta) = m\log\tfrac{1}{\sqrt{2\pi}\sigma} - \frac{1}{\sigma^2}\cdot\frac{1}{2}\sum_{i=1}^m (y^{(i)} - \theta^\top x^{(i)})^2$$

The constant and the positive \(1/\sigma^2\) don't move the argmax, so:

$$\text{maximize } \ell(\theta) \iff \text{minimize } \tfrac{1}{2}\sum_i (y^{(i)} - \theta^\top x^{(i)})^2$$

!!! note "The payoff"
    Maximizing likelihood under Gaussian noise = minimizing squared error. The square comes straight from the Gaussian's exponent. \(L_1\)/absolute error is the MLE under *Laplace* noise (heavier tails) — which is exactly why MAE is more robust to outliers. Choice of loss is secretly a choice of noise distribution. Also: \(\sigma^2\) dropped out — the fitted line is the same whether data is noisy or clean. This same recipe with a Bernoulli gives cross-entropy, which produces [logistic regression](logistic-perceptron.md).

## §5 Locally Weighted Regression (LWR)

One global line systematically misfits a curve, and guessing the right polynomial features is hard. LWR's idea: **fit a fresh line for every query point, using mostly the training points near it.** Zoom in far enough and any smooth curve looks locally straight.

$$\text{minimize}\quad \sum_i w^{(i)}\,(y^{(i)} - \theta^\top x^{(i)})^2, \qquad w^{(i)} = \exp\!\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)$$

Close to the query \(x\) → \(w^{(i)}\to1\) (full attention); far → \(w^{(i)}\to0\) (ignored). It's a bell-shaped attention window (not a probability — no normalizer). **\(\tau\) (bandwidth)** is the bias-variance dial: large \(\tau\) → wide window → approaches global LR → underfit; small \(\tau\) → narrow → wiggly → overfit.

!!! note "Parametric vs non-parametric"
    Plain LR is *parametric*: fit \(\theta\) once, then throw the data away — model size is fixed. LWR is *non-parametric*: every prediction re-solves a weighted regression using the training set itself, so you must keep all data forever. "Non-parametric" precisely means *the stuff you keep grows with the training set* — not "no parameters." The trade: parametric is cheap and rigid; non-parametric is flexible and expensive ([KNN](../core-algorithms/knn.md) is the other classic example).

## Interview questions

**Q1: Why does linear regression square the errors instead of taking absolute values?**
Squaring removes the sign so positive and negative misses do not cancel, and it penalizes large errors disproportionately, pushing the line to avoid big misses. Probabilistically, minimizing squared error is the maximum-likelihood fit under Gaussian noise, with the square coming straight from the Gaussian exponent. Absolute error is the MLE under Laplace noise, which has heavier tails and is therefore more robust to outliers.

**Q2: What is the closed-form solution and when would you not use it?**
The normal equation is \(\mathbf w=(X^\top X)^{-1}X^\top\mathbf y\), found by setting the gradient of the squared-error loss to zero. Inverting \(X^\top X\) costs \(O(d^3)\) in the feature count, so for very high-dimensional problems you switch to gradient descent. The closed form also breaks when \(X^\top X\) is singular, for example under perfect multicollinearity, which is one motivation for Ridge regression.

**Q3: Interpret the single-feature slope.**
The slope equals covariance(x, y) divided by variance(x), so it is the co-movement of x and y per unit of x-spread. The intercept then forces the line through the mean point \((\bar x,\bar y)\), the center of mass of the data.

**Q4: What makes a model parametric versus non-parametric, using LWR as the example?**
A parametric model fits a fixed set of parameters once and discards the data, so its size does not grow with the sample. LWR is non-parametric because every prediction re-solves a distance-weighted regression using the stored training set, so the data must be kept forever. Non-parametric means the thing you retain grows with the training set, not that there are no parameters.
