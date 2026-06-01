# Assumptions & Diagnostics

A fitted line can be optimal and still be lying to you. This page hangs the six assumptions of linear regression on one cricket dataset, shows which three you read straight off the residual plot, and decodes the Q-Q plot shape by shape so you can name the failure and its fix in one look.

!!! tip "Rapid Recall"
    The master diagnostic is the residual-versus-fitted plot: an even band passes, a funnel flags non-constant variance, a smile or frown flags a curve forced straight, and a lone far dot flags an influential outlier. Q-Q plots catch normality: a splaying S means a tail problem, a consistent C-shaped curve means skew, and a few stray points mean outliers. Normality is mainly a small-sample concern because the coefficient estimate is a sum over points, so the Central Limit Theorem makes it normal as n grows. The fitted line itself never needed normality; only the significance tests do.

## §1 Assumptions and failure modes

The trick to remembering these: hang all six on **one scene** — a cricket dataset predicting runs from batting average, where each assumption is one way reality fools your straight line.

| Assumption | The character | The tell and consequence | Fix |
| --- | --- | --- | --- |
| **Linearity** | the curve you forced straight | residuals form a **smile/frown**; systematic under/overfit (high bias) | polynomial features, non-linear model |
| **Homoscedasticity** | noisy top-order vs steady tailender | residuals make a **funnel**; line still unbiased but error bars miscalibrated | log-transform y, weighted regression |
| **No multicollinearity** | two scouts, same report | weights **explode and cancel** (+500/−480); VIF > 10; \(X^\top X\) near-singular | drop feature, PCA, Ridge |
| **Normality of errors** | the optional paperwork | Q–Q plot **off the diagonal**; only the significance tests break, not the line | log-transform, robust regression |
| **Independence** | the rain-affected match | errors **clump** (autocorrelation); overconfident SEs | lag features, time-series models |
| **No influential outliers** | the one Bradman | line **tilts to chase** him (MSE squares!); Cook's distance > 1 | Winsorize, Huber loss |

!!! note "Recall anchor"
    The *residual-vs-fitted plot is the master clue board* — three of six (linearity, homoscedasticity, outliers) are read off its *shape* alone (smile, funnel, lone far dot). Interview line: "I'd plot residuals vs fitted first — shape tells me linearity, variance, and outliers in one look — then Q–Q for normality, Durbin-Watson for independence, VIF for collinearity."

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="Three residual plot shapes" role="img" viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg">
<g fill="#4a4438" font-family="monospace" font-size="11">
<!-- panel 1: random (good) -->
<line stroke="#d8cfb8" x1="30" x2="190" y1="100" y2="100"></line>
<circle cx="50" cy="92" fill="#1d4d4f" r="2.5"></circle><circle cx="70" cy="108" fill="#1d4d4f" r="2.5"></circle>
<circle cx="90" cy="95" fill="#1d4d4f" r="2.5"></circle><circle cx="110" cy="106" fill="#1d4d4f" r="2.5"></circle>
<circle cx="130" cy="90" fill="#1d4d4f" r="2.5"></circle><circle cx="150" cy="110" fill="#1d4d4f" r="2.5"></circle>
<circle cx="170" cy="97" fill="#1d4d4f" r="2.5"></circle>
<text text-anchor="middle" x="110" y="150">Random band — good</text>
<!-- panel 2: funnel -->
<line stroke="#d8cfb8" x1="240" x2="400" y1="100" y2="100"></line>
<circle cx="258" cy="98" fill="#7c2d12" r="2.5"></circle><circle cx="278" cy="103" fill="#7c2d12" r="2.5"></circle>
<circle cx="298" cy="94" fill="#7c2d12" r="2.5"></circle><circle cx="318" cy="112" fill="#7c2d12" r="2.5"></circle>
<circle cx="338" cy="84" fill="#7c2d12" r="2.5"></circle><circle cx="358" cy="120" fill="#7c2d12" r="2.5"></circle>
<circle cx="378" cy="75" fill="#7c2d12" r="2.5"></circle>
<path d="M250,100 L390,72 M250,100 L390,128" fill="none" stroke="#7c2d12" stroke-dasharray="3,3" stroke-width="1"></path>
<text text-anchor="middle" x="320" y="150">Funnel — heteroscedastic</text>
<!-- panel 3: smile -->
<line stroke="#d8cfb8" x1="450" x2="610" y1="100" y2="100"></line>
<path d="M460,70 Q530,135 600,70" fill="none" stroke="#9a7b2e" stroke-dasharray="3,3" stroke-width="1.4"></path>
<circle cx="468" cy="74" fill="#9a7b2e" r="2.5"></circle><circle cx="490" cy="100" fill="#9a7b2e" r="2.5"></circle>
<circle cx="510" cy="118" fill="#9a7b2e" r="2.5"></circle><circle cx="530" cy="124" fill="#9a7b2e" r="2.5"></circle>
<circle cx="550" cy="118" fill="#9a7b2e" r="2.5"></circle><circle cx="572" cy="100" fill="#9a7b2e" r="2.5"></circle>
<circle cx="592" cy="74" fill="#9a7b2e" r="2.5"></circle>
<text text-anchor="middle" x="530" y="150">Smile — nonlinearity</text>
</g>
</svg>
<figcaption>Reading the residual-vs-fitted plot by shape: an even band passes; a funnel flags non-constant variance; a smile/frown flags a curve forced straight.</figcaption>
</figure>

### Data diet (falls out of the same scene)

- **Outlier-sensitive?** Very — MSE squares, so one Bradman tilts the line.
- **Normalize features?** Yes — doesn't move the prediction, but unequal scales make GD zig-zag across a stretched-ellipse bowl and make Ridge/Lasso penalize big-scale features unfairly.
- **Categoricals?** One-hot encode. **Missing data?** Impute or drop. **Sample size?** ~10–20 obs per feature for stable coefficients.

## §2 Normality and the Q-Q plot

**Why do tests break without normality?** A coefficient estimate \(\hat{\mathbf w}=(X^\top X)^{-1}X^\top\mathbf y\) is a fixed matrix times \(\mathbf y\), and the only random part of \(\mathbf y\) is the error \(\boldsymbol\epsilon\). So the *shape* of \(\hat{\mathbf w}\)'s sampling distribution = the shape of \(\boldsymbol\epsilon\)'s. If errors are normal, \(\hat{\mathbf w}\) is normal, the t-statistic follows a t-distribution, and a p-value is an area read off that known curve. If errors aren't normal, the p-value reads area off the *wrong* curve — precise and wrong.

- **Heavy tails** → falsely significant results (reject "coef = 0" too often).
- **Skew** → a symmetric CI is mis-centered.

!!! note "The CLT escape hatch"
    \(\hat{\mathbf w}\) is essentially a *sum* over many points, so by the Central Limit Theorem it tends to normal as \(n\) grows *regardless* of the errors' shape. Normality is a small-sample concern. With \(n=50{,}000\), mild non-normality barely dents p-values; with \(n=30\) it wrecks them. And the line itself (Gauss-Markov: best linear unbiased) never needed normality at all.

### Decoding Q-Q plot shapes

Sort residuals, plot against what a perfect normal would produce at those ranks. On the diagonal = normal. *Direction* of deviation = the diagnosis:

- **S-shape** (ends splay from the line) → tail problem. Below-left and above-right = **heavy tails** (the dangerous, false-positive case). Opposite = light tails (conservative).
- **C-shape / single bend** (both ends curve the same way) → **skew**. Up-bend = right skew (log-transform y); down-bend = left skew.
- **A few isolated points** flying off while the middle is fine → **outliers**, not a distribution shape.

Quick decoder: **splaying S = tails**; **consistent curve = skew**. Don't over-read on small \(n\); pair with Shapiro-Wilk if it matters.

## Interview questions

**Q1: Which assumptions can you check from a single residual-versus-fitted plot?**
Three of the six: linearity, homoscedasticity, and influential outliers. A smile or frown shape means a nonlinear relationship forced straight, a funnel means non-constant variance, and a lone far dot means an influential outlier. The remaining checks are the Q-Q plot for normality, Durbin-Watson for independence, and VIF for multicollinearity.

**Q2: Does linear regression require normally distributed errors?**
The fitted line does not; by Gauss-Markov it is the best linear unbiased estimator without any normality assumption. Normality matters only for the significance tests, because the p-value reads an area off the t-distribution that the coefficient follows only when the errors are normal. Even then it is a small-sample concern, since the coefficient estimate is a sum over points and the Central Limit Theorem makes it approximately normal as n grows.

**Q3: How do you read a Q-Q plot?**
Points on the diagonal mean normal errors, and the direction of deviation names the problem. A splaying S-shape signals a tail issue, with below-left and above-right splay meaning the dangerous heavy-tailed case; a consistent C-shaped bend signals skew; and a few isolated points flying off while the middle stays on the line signal outliers rather than a distributional shape.

**Q4: Why is linear regression so sensitive to outliers?**
Because the loss squares the residuals, a single far point contributes a disproportionately large error and the line tilts to chase it. You detect this with a lone far dot on the residual plot or a Cook's distance above 1, and you mitigate it by Winsorizing or switching to a robust loss such as Huber.
