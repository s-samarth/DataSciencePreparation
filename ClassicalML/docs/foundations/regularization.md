# Regularization

Regularization is the single move that fixes both overfitting and the weight-explosion that happens when features outnumber samples: add a penalty on weight size so the model cannot buy large weights for free. This page covers Ridge, Lasso, and ElasticNet, the geometry of why L1 zeros weights while L2 only shrinks them, and the underdetermined regime where coefficients blow up.

!!! tip "Rapid Recall"
    Add a budget on weight size to the loss; lambda sets how strict it is, from plain OLS at zero to all weights crushed at infinity. Ridge (L2) adds lambda to every eigenvalue of \(X^\top X\), guaranteeing invertibility and shrinking all weights toward zero but never to zero. Lasso (L1) has corners on the axes, so the expanding loss ellipse hits a corner first and drives weights to exactly zero, giving automatic feature selection. ElasticNet stacks both to handle correlated feature groups. When features outnumber samples the fit is exact in infinitely many ways, and OLS picks noise-amplifying cancelling weights; Ridge breaks the tie toward the smallest-weight solution.

## §1 Why regularize?

**Why regularize?** (1) *Overfitting*: with many features, OLS grows large weights to thread every point — high variance. (2) *Multicollinearity*: collinear features make \(X^\top X\) singular and weights blow up. Both are fixed by one move: **add a penalty on weight size**, so the model can't make weights large for free.

> **The budget.** OLS is a shopper with an unlimited credit card, it buys huge weights to fit every point. Regularization is a *budget*; \(\lambda\) sets how strict it is. The model must now ask "is this feature worth its weight-cost?" Cheap, low-value features get squeezed out, exactly what stops it chasing noise.

$$L(\mathbf{w}) = \underbrace{\|\mathbf{y} - X\mathbf{w}\|_2^2}_{\text{fit}} + \underbrace{\lambda \cdot \text{penalty}(\mathbf{w})}_{\text{keep weights small}}$$

\(\lambda=0\) → plain OLS (max variance); \(\lambda\to\infty\) → all weights crushed (max bias). Never penalize the intercept; always standardize features first.

### Ridge (L2)

$$L = \|\mathbf y - X\mathbf w\|_2^2 + \lambda\sum_j w_j^2 \qquad\Rightarrow\qquad \mathbf{w} = (X^\top X + \lambda I)^{-1}X^\top \mathbf{y}$$

The \(\lambda I\) adds \(\lambda\) to every eigenvalue of \(X^\top X\), bumping them all strictly positive → guaranteed invertible. Geometrically it **lifts the flat trough back into a strict bowl**. Shrinks all weights toward zero, never exactly to zero.

### Lasso (L1)

$$L = \|\mathbf y - X\mathbf w\|_2^2 + \lambda\sum_j |w_j|$$

Drives many weights to **exactly zero** (automatic feature selection). No closed form (the kink at \(|w|=0\) isn't differentiable) — solved by coordinate descent. The per-coordinate soft-thresholding update reveals the on/off gate:

$$w_j = \mathrm{sign}(\rho_j)\cdot\max\big(|\rho_j| - \lambda,\ 0\big)$$

If a feature's signal \(|\rho_j|\) is below the threshold \(\lambda\), the weight is set to exactly 0.

### Why L1 zeros and L2 does not: the geometry

MSE contours are ellipses centered on the OLS solution. The penalty is a constraint region: **L2 = a smooth circle**, **L1 = a diamond with corners on the axes**. The solution is where the expanding ellipse first touches the region. L1's **corners lie on the axes (where a weight = 0)**, and an ellipse coming in at an angle hits a corner first → sparsity. L2's circle is smooth → it touches at a generic point where both weights are small-but-nonzero. *Corners cause sparsity.*

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="L1 diamond vs L2 circle constraint geometry" role="img" viewBox="0 0 640 280" xmlns="http://www.w3.org/2000/svg">
<g font-family="monospace" font-size="12">
<!-- L1 -->
<g transform="translate(0,0)">
<line stroke="#d8cfb8" x1="40" x2="290" y1="140" y2="140"></line><line stroke="#d8cfb8" x1="165" x2="165" y1="30" y2="250"></line>
<polygon fill="#7c2d12" fill-opacity="0.13" points="165,70 225,140 165,210 105,140" stroke="#7c2d12" stroke-width="2"></polygon>
<ellipse cx="225" cy="80" fill="none" rx="78" ry="46" stroke="#1d4d4f" stroke-width="1.6" transform="rotate(-22 225 80)"></ellipse>
<ellipse cx="225" cy="80" fill="none" rx="52" ry="30" stroke="#1d4d4f" stroke-width="1.6" transform="rotate(-22 225 80)"></ellipse>
<circle cx="165" cy="70" fill="#7c2d12" r="5"></circle>
<text fill="#7c2d12" x="175" y="64">touches a corner → w₁=0</text>
<text fill="#4a4438" text-anchor="middle" x="165" y="270">L1 (Lasso) — diamond</text>
</g>
<!-- L2 -->
<g transform="translate(330,0)">
<line stroke="#d8cfb8" x1="10" x2="270" y1="140" y2="140"></line><line stroke="#d8cfb8" x1="140" x2="140" y1="30" y2="250"></line>
<circle cx="140" cy="140" fill="#1d4d4f" fill-opacity="0.12" r="62" stroke="#1d4d4f" stroke-width="2"></circle>
<ellipse cx="210" cy="80" fill="none" rx="78" ry="46" stroke="#7c2d12" stroke-width="1.6" transform="rotate(-22 210 80)"></ellipse>
<ellipse cx="210" cy="80" fill="none" rx="52" ry="30" stroke="#7c2d12" stroke-width="1.6" transform="rotate(-22 210 80)"></ellipse>
<circle cx="188" cy="98" fill="#1d4d4f" r="5"></circle>
<text fill="#1d4d4f" x="196" y="92">touches the curve → both ≠ 0</text>
<text fill="#4a4438" text-anchor="middle" x="140" y="270">L2 (Ridge) — circle</text>
</g>
</g>
</svg>
<figcaption>The expanding loss ellipse first touches L1's pointy corner (a weight snaps to zero) but L2's smooth circle at a generic point (both weights shrink, neither vanishes).</figcaption>
</figure>

### ElasticNet

$$L = \|\mathbf y - X\mathbf w\|_2^2 + \lambda_1\sum_j|w_j| + \lambda_2\sum_j w_j^2$$

L1 + L2 stacked. Fixes Lasso's failure on *correlated* features (Lasso arbitrarily keeps one, unstable); L2's grouping effect **shares weight across correlated groups**. Use when you want selection but features come in correlated clusters.

|  | Ridge (L2) | Lasso (L1) | ElasticNet |
| --- | --- | --- | --- |
| Penalty | \(\sum w_j^2\) | \(\sum\|w_j\|\) | both |
| Constraint | circle (smooth) | diamond (corners) | rounded diamond |
| Zeros weights? | no, shrinks all | yes (selection) | yes |
| Closed form? | yes (\(+\lambda I\)) | no (coord. descent) | no |
| Collinearity | stabilizes | picks one (poor) | shares (good) |

## §2 Why weights explode when \(p > n\)

With 100 features and 30 samples you're in the **underdetermined** regime: more knobs than constraints. As a linear system, 30 equations and 100 unknowns → infinitely many *exact* solutions. OLS drives training error to literally zero — in infinitely many ways. The question isn't "will it fit" (it fits perfectly) but **which** of the infinite solutions it lands on, and why that's bad.

With more features than constraints, features are guaranteed linearly dependent. Take two near-identical features \(x_5\approx x_6\). The model can fit using \(+1000\,x_5 - 998\,x_6\): the huge weights nearly cancel on every *training* point (so training looks fine) while amplifying the hair's-width noise difference to nail the last residual. Mathematically this is \((X^\top X)^{-1}\) blowing up — inverting a near-zero eigenvalue yields an enormous number; the huge weights *are* that inverse.

!!! note "Why your instinct was half-right"
    "Large weights wouldn't mean it fits well" — true on *test* data, false on *train*. On a test point, \(x_5,x_6\) drift apart differently than in training, the cancellation fails, and a tiny discrepancy × 1000 sends the prediction flying. Weight magnitude is a *sensitivity multiplier*: it memorized, it didn't learn.

> **Stand-up.** A set hyper-tuned to one venue, callbacks only that crowd gets, timing built around that room's echo, a bit that only lands because the seat-5 heckler always interrupts at minute 4. Kills on those 30 nights (perfect fit), collapses in a new city (every laugh depended on a fragile configuration that only existed in training). The huge cancelling weights are those hyper-specific dependencies. Regularization is the comedian who builds material that travels, slightly less optimized for the home crowd (a little bias), but robust everywhere (low variance).

Ridge's \(+\lambda I\) resolves "which of the infinite solutions" by adding a tiebreaker: **pick the smallest-weight one**, which kills the noise-amplifying cancellations.

## Interview questions

**Q1: Why does Lasso produce sparse solutions but Ridge does not?**
The penalty defines a constraint region the loss ellipse expands into: Ridge is a smooth circle, Lasso is a diamond with corners on the axes. An ellipse arriving at an angle touches a corner first, and corners sit where a weight is exactly zero, so Lasso zeros weights and selects features. Ridge's smooth circle is first touched at a generic point where both weights are small but nonzero, so it only shrinks.

**Q2: How does Ridge fix multicollinearity?**
Ridge adds \(\lambda I\) to \(X^\top X\), which adds lambda to every eigenvalue and makes the matrix strictly positive definite and invertible even when features are collinear. Geometrically it lifts the flat trough of the loss back into a strict bowl with a unique minimum. It shrinks all weights toward zero without removing any.

**Q3: Why do coefficients explode when there are more features than samples?**
The system is underdetermined, so there are infinitely many weight vectors that fit the training data exactly, and OLS picks among them via \((X^\top X)^{-1}\), which blows up because a near-zero eigenvalue inverts to an enormous number. Concretely it uses huge cancelling weights on near-identical features that look fine on training but amplify tiny test-time discrepancies. Weight magnitude is a sensitivity multiplier, a sign the model memorized rather than learned.

**Q4: When would you choose ElasticNet over Lasso?**
When you want feature selection but the features come in correlated clusters. Pure Lasso arbitrarily keeps one feature from a correlated group and is unstable, whereas ElasticNet's L2 term has a grouping effect that shares weight across the correlated group while the L1 term still drives selection.
