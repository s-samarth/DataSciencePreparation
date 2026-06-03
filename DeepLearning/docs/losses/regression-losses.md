# Regression Losses

Regression losses score a continuous prediction against a continuous target. The choice among them is really a choice about what noise model you assume and how much you want outliers to matter.

!!! tip "Rapid Recall"
    MSE (L2) is the NLL under Gaussian noise, predicts the conditional mean, is smooth everywhere, but squaring makes it extremely outlier-sensitive. MAE (L1) is the NLL under Laplacian noise, predicts the median, is outlier-robust, but is non-differentiable at zero and has a constant gradient that slows late convergence. Huber is quadratic near zero and linear far out, the best of both, and is the standard for bounding-box regression. Quantile (pinball) loss penalizes over- and under-prediction asymmetrically to predict an arbitrary quantile, the basis of prediction intervals.

## §1 Mean Squared Error (MSE / L2)

| Symbol | Meaning |
| --- | --- |
| $y_i$ | True target for sample $i$. |
| $\hat{y}_i$ | Predicted value for sample $i$. |
| $n$ | Number of samples in the batch. |

$$
L = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \qquad \frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)
$$

**Probabilistic interpretation:** MSE is the negative log-likelihood under Gaussian noise: $y = \hat{y} + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. Maximizing the Gaussian likelihood is exactly equivalent to minimizing MSE.

**Strengths:** smooth and differentiable everywhere; convex for linear models (unique global minimum); strongly penalizes large errors (good when big errors are unacceptable); closed-form solution exists for linear regression (the normal equation).

**Limitations:** extremely sensitive to outliers, squaring amplifies them (one outlier with error 10 contributes 100 to the loss, potentially dominating it); assumes errors are Gaussian with constant variance (homoscedasticity); output is in squared units, hard to interpret directly.

**Use when:** targets are continuous, errors are roughly Gaussian, outliers are rare or important to penalize heavily.

## §2 Mean Absolute Error (MAE / L1)

$$
L = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|, \qquad \frac{\partial L}{\partial \hat{y}} = \mathrm{sign}(\hat{y} - y)
$$

**Probabilistic interpretation:** negative log-likelihood under Laplacian noise. MSE optimizes the conditional mean; MAE optimizes the conditional median.

**Strengths:** robust to outliers (linear penalty, not quadratic); output is in the same units as the target; predicts the median, which is more meaningful for skewed distributions.

**Limitations:** not differentiable at zero (in practice, frameworks define $\mathrm{sign}(0) = 0$); constant gradient means slower convergence near the optimum (the gradient doesn't shrink as you approach the minimum); multiple optimal solutions are possible (anything between two median candidates).

**Use when:** data has outliers you want to ignore, or when you care about median prediction (for example, predicting house prices in a market with extreme values).

## §3 Huber Loss (Smooth L1)

| Symbol | Meaning |
| --- | --- |
| $\delta$ | Threshold parameter where the loss switches from quadratic to linear. Typical 1.0. |

$$
L_\delta(y, \hat{y}) = \begin{cases} \dfrac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\[6pt] \delta\,|y - \hat{y}| - \dfrac{1}{2}\delta^2 & \text{otherwise} \end{cases}
$$

**Intuition:** acts like MSE for small errors (smooth gradients near the minimum) and like MAE for large errors (robust to outliers). $\delta$ is the threshold where it switches.

**Strengths:** the best of both worlds, smooth near zero, robust far from zero; differentiable everywhere (unlike MAE); standard for robust regression and bounding box regression in object detection (Smooth L1 in Faster R-CNN, SSD). **Limitations:** requires tuning $\delta$ (typical 1.0 for standardized targets); not as clean a probabilistic interpretation.

## §4 Quantile Loss (Pinball Loss)

| Symbol | Meaning |
| --- | --- |
| $\tau$ | Target quantile in $(0, 1)$. $\tau = 0.5$ gives the median, $\tau = 0.9$ the 90th percentile. |

$$
L_\tau = \begin{cases} \tau (y - \hat{y}) & \text{if } y \ge \hat{y} \\ (1 - \tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases}
$$

**Intuition:** for $\tau = 0.5$, this reduces to MAE (predicts the median). For $\tau = 0.9$, under-predictions are penalized 9x more than over-predictions, so the model predicts the 90th percentile.

**Strengths:** predicts arbitrary quantiles, not just mean/median; the foundation of **prediction intervals**, train models at $\tau = 0.05$ and $\tau = 0.95$ to get a 90% confidence band; no distributional assumption.

**Use when:** you need uncertainty estimates, risk-aware forecasting (for example, demand forecasting where stockouts cost more than overstock), or asymmetric loss profiles.
