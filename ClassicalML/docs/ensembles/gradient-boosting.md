# Gradient Boosting

Gradient boosting generalizes AdaBoost's forward stagewise modeling from exponential loss to any differentiable loss, by fitting each new tree to the negative gradient, the ensemble's current error, instead of reweighting points. This page covers residuals as gradients, the regressor/classifier unification, tuning, and Newton's method, which the modern libraries build on.

!!! tip "Rapid Recall"
    Gradient boosting fits each tree to the negative gradient of the loss with respect to the prediction, which for squared error is the residual \(y-F\) and for log loss is \(y-p\), then adds it scaled by a learning rate. It is gradient descent in function space, defined for any differentiable loss, and the trees are always regression trees. It reduces bias, not variance, so over-boosting eventually fits noise and overfits, controlled by small learning rate, early stopping, shallow depth, and subsampling. The golden coupling is that halving the learning rate roughly doubles the number of trees. Newton's method uses curvature to auto-scale the step, which XGBoost exploits.

## §1 Gradient boosting: residuals to gradients

Generalize "forward stagewise additive modeling" from exponential loss (AdaBoost) to *any* differentiable loss.

!!! note "The core leap"
    AdaBoost focuses the next learner by *reweighting* examples (a trick specific to exponential loss). Gradient boosting instead *fits the next tree to the errors directly*: each tree predicts the *residual*, what the current ensemble got wrong, and you add it on. The ensemble is a running sum of corrective trees.

### Squared-error case (build intuition)

\(L(y,F)=\frac{1}{2}(y-F)^2\).

1. **Init:** best constant \(F_0(x)=\bar{y}\) (mean of \(y\)).
2. For \(m=1,\dots,M\): residual \(r_i=y_i-F_{m-1}(x_i)\); fit a regression tree \(h_m\) to the \(r_i\); update \(F_m=F_{m-1}+\nu\,h_m\).

**Trace:** \(y=10\), \(F_0=6\) → residual 4; tree 1 predicts about 3.5 → \(F_1=9.5\) → residual 0.5; tree 2 predicts about 0.4 → \(F_2=9.9\). Residuals shrink each round.

### Why "gradient"

For squared error the residual equals the negative gradient of the loss with respect to the prediction:

$$L=\frac{1}{2}(y-F)^2\Rightarrow\frac{\partial L}{\partial F}=-(y-F)\Rightarrow-\frac{\partial L}{\partial F}=(y-F)=\text{residual}$$

!!! note "Gradient descent in function space"
    "Fit the residual" is secretly "fit the *negative gradient*," which is defined for *any* differentiable loss. Ordinary GD nudges parameters \(\theta\leftarrow\theta-\nu{\nabla}_{\theta}L\); gradient boosting nudges the *function's outputs*, fitting a tree to the desired per-point directions (the "pseudo-residuals") and stepping by \(\nu\). The tree generalizes the gradient step to all of input space.

### General algorithm (any differentiable loss)

1. \(F_0(x)=\arg\min_c\sum_i L(y_i,c)\).
2. Pseudo-residuals \(r_i=-{\big[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\big]}_{F=F_{m-1}}\).
3. Fit a regression tree \(h_m\) to the \(r_i\) (optionally line-search each leaf's value to minimize loss directly).
4. \(F_m=F_{m-1}+\nu\,h_m\).

**Trees are always regression trees**, even for classification, they fit continuous gradients, not classes.

## §2 GBM regressor and classifier

!!! note "Same algorithm, different loss"
    *Regressor (squared error):* residual \(=y-F\). *Classifier (log loss):* ensemble outputs *log-odds*, squashed by sigmoid. \(F_0=\log\frac{p}{1-p}\); pseudo-residual works out to \(r_i=y_i-p_i\) where \(p_i=\sigma(F_{m-1}(x_i))\). Fit a regression tree to \(r_i\), add to log-odds, repeat. Final probability \(\sigma(F_M(x))\). Multiclass: one log-odds per class plus softmax.

So the *only* difference between Regressor and Classifier is the loss plugged into the gradient: squared error → residual \((y-F)\); log loss → residual \((y-p)\). Both have the form **"truth minus current prediction"**, that's the payoff of the gradient framing: one algorithm, swap the loss.

Weak learners here are **shallow trees (depth 3 to 8), not stumps**, depth controls how many feature *interactions* each tree captures (depth-\(d\) → \(d\)-way interactions). The "focus on hard cases" is automatic: wherever the ensemble is most wrong, the gradient is largest, so the next tree devotes its splits there.

## §3 How GBM makes predictions

Run \(x\) through **all** trees, scale each by the learning rate, add on top of the initial constant (you *add*, not average like RF).

**Regressor:**

$$F_M(x)=F_0+\nu\sum_{m=1}^M h_m(x),\quad F_0=\bar{y}$$

That sum *is* the prediction.

**Classifier:**

$$F_M(x)=F_0+\nu\sum_{m=1}^M h_m(x),\quad F_0=\log\frac{p}{1-p}$$

This is a **log-odds**; convert and threshold:

$$p(x)=\sigma(F_M(x)),\quad\hat{y}=[\,p(x)\ge 0.5\,]$$

(Multiclass: one such sum per class, then softmax.)

## §4 GBM tuning, assumptions, weaknesses

### Tuning: three levers

- **LR / n_estimators trade (central):** `learning_rate` (\(\nu\), shrinkage), smaller = each tree corrects less = needs more trees but generalizes much better; the main regularizer. `n_estimators` (\(M\)), more → lower bias, eventually overfits. **Golden coupling:** halve LR is about double \(M\). Best practice: small LR (0.05 to 0.1) plus `n_estimators` via **early stopping** on validation.
- **Per-tree complexity:** `max_depth` (3 to 8, keep shallow), `min_samples_leaf`/`min_samples_split`.
- **Stochasticity:** `subsample` < 1 = *Stochastic Gradient Boosting*, fit each tree on a random row fraction (no replacement), injecting bagging-style decorrelation. `max_features` subsamples features per split.

GBM needs **more careful tuning than RF** (more interacting knobs; overfits if careless).

### Assumptions

- **Differentiable loss**, the whole method is gradient descent on it. The one hard requirement (AdaBoost's reweighting didn't need it; this is what buys generality).
- **Weak learners that underfit individually** (shallow trees), gradual, controllable bias reduction.
- **Low-to-moderate noise**, chases residuals; persistent noise gets fit if over-boosted.

### Bias vs variance

Reduces **bias**: starts from a high-bias constant and every round explicitly reduces training loss (gradient descent on it) → more expressive each round. Does **not** average out variance (dependent sequential chain, no error-cancellation). **Over-boosting → variance grows → overfit** (gradient eventually points at noise), so more trees can hurt, unlike RF. Defense: small LR + early stopping + shallow depth + subsample.

### Weaknesses

- **Sequential → not parallelizable across trees** (tree \(m\) needs \(F_{m-1}\)). Vanilla sklearn GBM is essentially serial.
- Overfits if over-boosted; sensitive to hyperparameters and noise; slower to train than RF; poor regression extrapolation; less robust to outliers under squared loss (use Huber/MAE); less interpretable.

## §5 Newton's method for optimization

Gradient descent uses the slope; Newton's method uses slope *and* curvature, that extra info buys fast convergence.

!!! note "The visual"
    GD feels the downhill tilt and shuffles in fixed-size steps. Newton's method *fits a parabola* matching the slope and curvature at your feet, then *jumps straight to the bottom of that parabola*. If the loss were exactly parabolic, it reaches the minimum in *one step*; smooth losses look parabolic near the minimum, so it converges quadratically.

<figure class="diagram diagram-dark" markdown="0">
<svg aria-label="Newton parabola fit" role="img" viewBox="0 0 560 260" xmlns="http://www.w3.org/2000/svg">
<!-- loss curve -->
<path d="M40,40 C160,260 400,260 520,60" fill="none" stroke="#94a3b5" stroke-width="2"></path>
<text fill="#94a3b5" font-family="monospace" font-size="12" x="60" y="35">L(θ)</text>
<!-- current point -->
<circle cx="180" cy="196" fill="#e0918a" r="5"></circle>
<text fill="#e0918a" font-family="monospace" font-size="12" x="150" y="190">θₜ</text>
<!-- fitted parabola (dashed) matching at the point, min near true min -->
<path d="M120,150 Q250,250 360,150" fill="none" stroke="#6ea8fe" stroke-dasharray="5 4" stroke-width="2"></path>
<text fill="#6ea8fe" font-family="monospace" font-size="12" x="370" y="150">fitted parabola</text>
<!-- jump arrow to parabola min -->
<circle cx="250" cy="232" fill="#7ed3b2" r="5"></circle>
<line stroke="#7ed3b2" stroke-dasharray="3 3" stroke-width="1.5" x1="180" x2="244" y1="196" y2="228"></line>
<text fill="#7ed3b2" font-family="monospace" font-size="11" text-anchor="middle" x="250" y="252">jump to its min</text>
</svg>
<figcaption>Newton's method approximates the loss locally by a parabola and leaps to its vertex.</figcaption>
</figure>

### The math (1-D)

Second-order Taylor (the fitted parabola), minimized over the step \(\Delta\):

$$L({\theta}_t+\Delta)\approx L({\theta}_t)+L'({\theta}_t)\Delta+\frac{1}{2}L''({\theta}_t){\Delta}^2\;\Rightarrow\;\Delta=-\frac{L'({\theta}_t)}{L''({\theta}_t)}$$

$$\boxed{{\theta}_{t+1}={\theta}_t-\frac{L'({\theta}_t)}{L''({\theta}_t)}}$$

Inverse curvature replaces the learning rate. GD: \(\theta-\eta L'\). Newton: \(\theta-L'/L''\), the step is auto-scaled by local geometry. *Steep curvature* (large \(L''\)) → small step (minimum is close, don't overshoot). *Flat region* (small \(L''\)) → big step (minimum is far, stride out). No learning rate to tune.

### Multi-dimensional and cost

$${\theta}_{t+1}={\theta}_t-H^{-1}\nabla L$$

\(H\) = Hessian (\(d\times d\) second partials). Forming and **inverting** it is \(O(d^3)\), the expense of general Newton. (XGBoost escapes this: its Hessian is *diagonal*, so "inversion" is reciprocating \(d\) scalars, \(O(d)\).)

**Why faster:** GD's speed depends on the *condition number* (steepest/flattest curvature ratio); an ill-conditioned ravine makes GD zigzag. Newton rescales each direction by its own curvature, turning the ravine into a round bowl, so it heads straight for the minimum regardless of conditioning. Tradeoff: **GD = cheap steps, many; Newton = expensive steps, few.** Quasi-Newton (L-BFGS) approximates \(H\) cheaply.

## Interview questions

**Q1: In what sense is fitting residuals the same as gradient descent?**
For squared error the residual \(y-F\) is exactly the negative gradient of the loss with respect to the prediction, so fitting a tree to residuals is fitting it to the negative gradient. Generalizing, gradient boosting computes pseudo-residuals as the negative gradient for any differentiable loss and fits a regression tree to them, stepping the function's outputs by a learning rate. It is gradient descent performed in function space rather than parameter space.

**Q2: What is the only difference between a GBM regressor and classifier?**
The loss plugged into the gradient. Squared error gives the pseudo-residual \(y-F\); log loss makes the ensemble output log-odds and gives the pseudo-residual \(y-p\) where p is the sigmoid of the current score. Both are truth minus current prediction, so the same algorithm fits regression trees to that error and, for classification, applies a sigmoid or softmax at the end.

**Q3: Why can adding trees hurt gradient boosting but never a random forest?**
A random forest averages independent trees, so more trees only reduce or flatten variance while bias stays fixed. Gradient boosting adds dependent corrective trees that keep reducing training loss, so once it has fit the signal the gradient starts pointing at noise and the ensemble fits it, raising variance. That is why boosting needs early stopping, a small learning rate, shallow trees, and subsampling.

**Q4: Why is Newton's method faster than gradient descent, and what is its cost?**
It uses curvature as well as slope, fitting a local parabola and jumping to its vertex, which rescales each direction by its own curvature and turns an ill-conditioned ravine into a round bowl, giving quadratic convergence near the minimum with no learning rate to tune. The cost is forming and inverting the Hessian, \(O(d^3)\) in general, which is why it suits low-dimensional convex problems; XGBoost avoids this because its Hessian is diagonal.
