# Boosting & AdaBoost

Boosting flips bagging on its head: instead of averaging many independent strong trees to cut variance, it adds many weak learners sequentially, each built to fix the previous ones' mistakes, to cut bias. This page covers the core contrast, AdaBoost step by step, why its weight is the optimal coefficient under exponential loss, and the multiclass and regression extensions.

!!! tip "Rapid Recall"
    Boosting adds weak learners sequentially, each focused on what the ensemble still gets wrong, reducing bias rather than variance. AdaBoost keeps a weight per example, up-weighting misclassified points, and combines stumps by an \(\alpha\)-weighted vote where \({\alpha}_t=\frac{1}{2}\ln\frac{1-{\epsilon}_t}{{\epsilon}_t}\). That weight is not a heuristic: it is the exact minimizer of exponential loss, making AdaBoost forward stagewise additive modeling. After each update the just-trained stump has weighted error exactly 0.5, forcing the next to find new structure. AdaBoost is extremely sensitive to noise and outliers.

## §1 Boosting: the core contrast

!!! note "Parallel vote vs sequential correction"
    *Bagging:* many *independent strong* trees in *parallel*, averaged, kills variance. *Boosting:* many *weak* learners *sequentially*, each built to fix the previous ones' mistakes, reduces bias. Bagging averages strong-but-noisy; boosting *adds up* weak-but-focused.

A **weak learner** = slightly better than random (e.g. a depth-1 *stump*). Deliberately weak because: (1) high bias / low variance → stable, and boosting's job is to drive bias down by combining many; (2) strong learners would each overfit, and chaining overfitters explodes variance immediately. Bias-reduction only works if each step adds a small controlled amount.

## §2 AdaBoost step by step

Maintain a weight per example = "how much attention the next learner must pay." Misclassified → weight up.

**Init:** every example weight \(w_i=1/n\). **For \(t=1,\dots,T\):**

1. **Train a weak learner** on the weighted data (minimize *weighted* error, high-weight mistakes hurt more).
2. **Weighted error:**
    $${\epsilon}_t=\frac{\sum_{i:\text{wrong}}w_i}{\sum_i w_i}$$
3. **Learner's "say":**
    $${\alpha}_t=\frac{1}{2}\ln\Big(\frac{1-{\epsilon}_t}{{\epsilon}_t}\Big)$$
    - \({\epsilon}_t\to 0\) → \({\alpha}_t\to+\infty\): trust heavily.
    - \({\epsilon}_t=0.5\) → \({\alpha}_t=0\): says nothing.
    - \({\epsilon}_t>0.5\) → \({\alpha}_t<0\): negative vote, flip it and still use it. A reliably-wrong stump is as useful as a reliably-right one.
4. **Reweight:** multiply by \(e^{{\alpha}_t}\) if wrong, \(e^{-{\alpha}_t}\) if right, then renormalize. Misclassified points get heavier → next stump's weighted impurity is dominated by them.

**Final prediction**, an \(\alpha\)-weighted vote:

$$H(x)=\text{sign}\Big(\sum_{t=1}^T{\alpha}_t h_t(x)\Big)$$

How weak becomes strong: each round's reweighting *rotates the problem* so the next stump attacks a currently-hard region. Stump 1 handles the bulk; later stumps pick off residual hard cases. The \(\alpha\)-weighted sum of crude axis-aligned cuts assembles an arbitrarily complex boundary. Not averaging guesses (bagging), additively assembling a complex function from simple pieces.

## §3 AdaBoost intricacies

### Weighted error and "training on weighted data"

Ordinary error counts each point \(1/n\); weighted error sums the weights of the mistakes. The stump's split criterion is unchanged *except* every "count of 1" becomes "weight \(w_i\)": child-node impurity is computed from **weighted class proportions**, and the winning split minimizes **weighted impurity**. So high-weight points literally pull the stump toward getting them right. (Some libraries instead *resample* with probability proportional to \(w_i\), same effect.)

### Where \({\alpha}_t\) comes from

!!! note "Not a heuristic, it is optimal"
    AdaBoost provably does greedy minimization of *exponential loss* \(\sum_i e^{-y_i F(x_i)}\), with \(F=\sum_t{\alpha}_t h_t\), \(y_i\in\{-1,+1\}\). Differentiating with respect to \({\alpha}_t\) and setting to zero yields exactly \({\alpha}_t=\frac{1}{2}\ln\frac{1-{\epsilon}_t}{{\epsilon}_t}\). So \({\alpha}_t\) is the *optimal* coefficient under exponential loss; "competence score" is a consequence. This makes AdaBoost = "forward stagewise additive modeling with exponential loss," the bridge to gradient boosting.

### Why 0.5 is the "random" line, and imbalance

The 0.5 threshold is on the **weighted** error, on the *current reweighted distribution*, not on raw class balance. Key fact most explanations skip:

!!! note "Built-in property"
    After each weight update plus renormalization, the *just-trained stump has weighted error exactly 0.5 on the new distribution.* Multiplying wrong points by \(e^{{\alpha}_t}\) and right by \(e^{-{\alpha}_t}\) (with that specific \({\alpha}_t\)) equalizes the total weight of its correct vs wrong points. This *forces* the next stump to find new structure, not repeat the last one.

**Imbalance objection (legitimate):** on raw imbalanced data, "always predict majority" has low raw error. But weights start at \(1/n\) regardless of class, so in round 1 that trivial stump has weighted error = minority fraction < 0.5, AdaBoost happily uses it (it *is* better than a coin flip). After round 1, reweighting inflates the misclassified minority points until "always majority" hits weighted error about 0.5 and stops being useful, **reweighting actively corrects imbalance** by escalating the neglected class. (Same mechanism that fixes mild imbalance makes it fragile to outliers, it escalates noise too.)

## §4 AdaBoost: multiclass and regression

Classic AdaBoost (sign, \(y\in\{-1,+1\}\)) is **binary by design**, but extends:

| Setting | Method | Key change |
| --- | --- | --- |
| Multiclass | AdaBoost.M1 | Needs every learner > 50% accuracy, hard for \(K\) classes (random = \(1/K\)). Often fails. |
| Multiclass (practical) | **SAMME** (sklearn default) | \({\alpha}_t=\ln\frac{1-{\epsilon}_t}{{\epsilon}_t}+\ln(K-1)\). The \(\ln(K-1)\) rebases the "useless" threshold from 0.5 to \(\frac{K-1}{K}\), learner only needs to beat random \(1/K\). (SAMME.R uses probabilities, faster.) |
| Regression | AdaBoost.R2 (sklearn) | Per-example *loss* from residual magnitude instead of right/wrong; reweight by loss. |

!!! note "Unifying idea"
    The "useless learner" threshold is always "no better than random *on the current problem*." Binary weighted → 0.5; \(K\)-class → \(\frac{K-1}{K}\). That's exactly why SAMME adds \(\ln(K-1)\).

### AdaBoost weaknesses and tuning

- **Extreme noise/outlier sensitivity** (the big one), relentless up-weighting of misclassified points means a mislabeled/outlier point's weight grows every round until it dominates. Noise is amplified, not averaged.
- **Not parallelizable**, round \(t\) needs round \(t-1\)'s weights (sequential chain). Can parallelize *within* a stump's split search, not across rounds.
- **Overfits** with too many rounds on noisy data. **Less interpretable**; superseded by gradient boosting.

**Tuning:** `n_estimators` (\(T\)), main bias/variance dial; `learning_rate` (shrinkage, scales \({\alpha}_t\)), smaller generalizes better, needs more rounds (halve LR is about double \(T\)); `base_estimator` depth, default stump, depth-2/3 only if interactions needed.

## Interview questions

**Q1: How does boosting differ from bagging, and why are the learners deliberately weak?**
Bagging trains many independent strong trees in parallel and averages them to cut variance, while boosting trains weak learners sequentially, each fixing the previous ones' errors, to cut bias. The learners are kept weak, like depth-1 stumps, because they are stable (high bias, low variance) so boosting can drive bias down gradually, and because chaining strong overfitters would explode variance immediately.

**Q2: Walk through one AdaBoost round.**
Train a weak learner to minimize weighted error \({\epsilon}_t=\sum_{i:\text{wrong}}w_i\) on the current weights, compute its say \({\alpha}_t=\frac{1}{2}\ln\frac{1-{\epsilon}_t}{{\epsilon}_t}\), then multiply the weights of misclassified points by \(e^{{\alpha}_t}\) and correct ones by \(e^{-{\alpha}_t}\) and renormalize. The final prediction is the sign of the \(\alpha\)-weighted vote. A stump with error above 0.5 gets a negative \(\alpha\) and is flipped, since reliably wrong is as useful as reliably right.

**Q3: Where does the \(\alpha\) formula come from?**
It is not a heuristic; AdaBoost is greedy forward stagewise minimization of exponential loss \(\sum_i e^{-y_i F(x_i)}\), and differentiating that with respect to the new learner's coefficient and setting it to zero yields exactly \(\frac{1}{2}\ln\frac{1-{\epsilon}_t}{{\epsilon}_t}\). So the competence score is a consequence of optimizing exponential loss, which is the bridge to gradient boosting.

**Q4: Why is AdaBoost so sensitive to noise and outliers?**
Its core mechanism relentlessly up-weights misclassified points, so a mislabeled or outlier point that can never be fit keeps having its weight inflated every round until it dominates the training signal. The same reweighting that helpfully escalates a neglected minority class also escalates noise, so noise is amplified rather than averaged out, and too many rounds on noisy data overfit.
