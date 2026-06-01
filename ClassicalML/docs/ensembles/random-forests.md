# Random Forests

A random forest is more than bagged trees: it adds per-split feature randomness specifically to attack the correlation \(\rho\), the term that sets the hard variance floor. This page covers that fix, the three hyperparameter levers organized by what they control, and where RF still loses to boosting.

!!! tip "Rapid Recall"
    Bootstrapping alone leaves trees correlated, especially when one strong feature dominates the first split everywhere, so RF restricts each node to a random subset of \(m\) features, forcing different trees to split differently and lowering \(\rho\). Defaults are \(m=\sqrt{d}\) for classification and \(d/3\) for regression. More trees never overfit RF, they only lower the cheap variance term, so tune `max_features` first since it lowers the correlation floor. Let RF trees grow deep because bagging handles variance. RF often loses to boosting on tabular accuracy and cannot extrapolate beyond the training target range.

## §1 Random forest = bagging + feature randomness

The thing that makes RF more than "a bunch of bagged trees."

### Bagging's weakness

Bootstrap samples are similar (about 63% of the same data), so trees stay fairly correlated. Worse: if one feature is very strong, *almost every tree splits on it first* regardless of the bootstrap → same root split → high \(\rho\) → the floor \(\rho{\sigma}^2\) stays high. Bootstrapping alone doesn't decorrelate enough.

!!! note "RF's fix: feature subsampling at each split"
    At *every node*, the tree may only consider a *random subset of \(m\) features* (out of \(d\)) when choosing its split. Typically \(m=\sqrt{d}\) (classification), \(m=d/3\) (regression). The dominant feature is now unavailable at many nodes, *forcing different trees to split differently*, directly attacking \(\rho\), the expensive term.

<figure class="diagram diagram-dark" markdown="0">
<svg aria-label="bagging vs RF decorrelation" role="img" viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg">
<text fill="#94a3b5" font-family="Helvetica" font-size="13" text-anchor="middle" x="140" y="22">Plain bagging: all split on F1 first</text>
<text fill="#94a3b5" font-family="Helvetica" font-size="13" text-anchor="middle" x="420" y="22">RF: F1 hidden at many nodes</text>
<!-- left: three identical -->
<g fill="#e7ebf0" font-family="monospace" font-size="11">
<rect fill="#1b2330" height="26" rx="6" stroke="#e0918a" width="60" x="40" y="40"></rect><text text-anchor="middle" x="70" y="57">F1</text>
<rect fill="#1b2330" height="26" rx="6" stroke="#e0918a" width="60" x="120" y="40"></rect><text text-anchor="middle" x="150" y="57">F1</text>
<rect fill="#1b2330" height="26" rx="6" stroke="#e0918a" width="60" x="200" y="40"></rect><text text-anchor="middle" x="230" y="57">F1</text>
<text fill="#e0918a" text-anchor="middle" x="150" y="92">high correlation ρ</text>
</g>
<!-- right: varied -->
<g fill="#e7ebf0" font-family="monospace" font-size="11">
<rect fill="#1b2330" height="26" rx="6" stroke="#7ed3b2" width="60" x="320" y="40"></rect><text text-anchor="middle" x="350" y="57">F3</text>
<rect fill="#1b2330" height="26" rx="6" stroke="#7ed3b2" width="60" x="400" y="40"></rect><text text-anchor="middle" x="430" y="57">F1</text>
<rect fill="#1b2330" height="26" rx="6" stroke="#7ed3b2" width="60" x="480" y="40"></rect><text text-anchor="middle" x="510" y="57">F7</text>
<text fill="#7ed3b2" text-anchor="middle" x="430" y="92">low correlation ρ</text>
</g>
<text fill="#94a3b5" font-family="Helvetica" font-size="13" text-anchor="middle" x="280" y="150">Lower ρ ⟹ lower variance floor ρσ² ⟹ averaging works better</text>
</svg>
<figcaption>Feature subsampling forces trees onto different root splits, breaking the correlation bootstrapping alone cannot.</figcaption>
</figure>

!!! note "Full recipe"
    RF = bootstrap rows (bagging) + random feature subset per split + aggregate by vote/average. Two independent randomness sources, both lowering \(\rho\), so averaging many deep (low-bias) trees crushes variance.

The bargain: restricting to \(m\) features makes each *individual* tree slightly worse (higher bias, can't always pick the best split), but lowers \(\rho\) a lot. Variance reduction dominates. Trade a little per-tree quality for much less correlation.

## §2 Random forest hyperparameters

Organize by three levers, don't memorize a flat list.

### Lever 1: number of trees (\(B\))

`n_estimators`: more trees → lower variance (kills \(\frac{1-\rho}{B}\)), monotonically, diminishing returns. **More trees never overfit RF** (variance only drops or flattens), unlike boosting. Cost is only compute. 200 to 500 usually plenty.

### Lever 2: decorrelation (control \(\rho\))

- `max_features` (\(m\)): *the* signature RF knob. Lower \(m\) → more decorrelation → lower variance, higher per-tree bias. First dial if RF overfits. Defaults \(\sqrt{d}\) / \(d/3\).
- `bootstrap`: whether to bootstrap (True default).
- `max_samples`: bootstrap sample size (less than n adds diversity and speed).

### Lever 3: individual tree complexity

`max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`: same as a single tree. **RF-specific point:** you generally let RF trees grow *deep/unconstrained* (low bias) because bagging handles variance, the opposite of single-tree practice. Only constrain if RF still overfits after tuning `max_features`.

!!! note "Tuning order"
    Set `n_estimators`=500 and forget. Tune `max_features` first (biggest impact on the variance floor). Add depth/leaf limits only if still overfitting. RF is famously robust, far less tuning than XGBoost.

### Weaknesses of random forest

- **Large model / slow inference at scale** — hundreds of deep trees, inference touches all.
- **Often loses to boosting on raw accuracy** for tabular, variance reduction has a ceiling; boosting's bias reduction usually edges it out.
- **Biased toward high-cardinality features** in impurity-based importance.
- **Poor regression extrapolation** — predictions are averages of training leaf values, so RF *cannot* predict outside the training target range; it flatlines beyond the data.
- **Less interpretable** than a single tree; importances can mislead.

## Interview questions

**Q1: Why does a random forest add feature subsampling on top of bagging?**
Because bootstrapping alone leaves trees correlated: the samples overlap heavily, and a single strong feature gets chosen as the first split in almost every tree, keeping \(\rho\) and hence the variance floor \(\rho\sigma^2\) high. Restricting each node to a random subset of m features makes the dominant feature unavailable at many nodes, forcing trees to split differently and directly lowering \(\rho\). Each tree is slightly worse, but the decorrelation more than pays for it.

**Q2: Can adding more trees overfit a random forest?**
No. More trees only shrink the \(\frac{1-\rho}{B}\) term toward zero, so variance monotonically decreases or flattens and bias is unchanged, the cost is just compute. This is the opposite of boosting, where more trees eventually fit noise and raise variance. A few hundred trees is usually plenty.

**Q3: Which hyperparameter do you tune first and why?**
`max_features`, the size of the per-split feature subset, because it controls \(\rho\) and therefore the correlation floor, which is the binding constraint on RF variance. `n_estimators` only chips at the cheap term, so set it to a few hundred and forget it. Depth and leaf limits come last, only if the forest still overfits after tuning `max_features`.

**Q4: Why can a random forest not extrapolate in regression, and where does it lose to boosting?**
Its prediction is an average of training-leaf target values, so it can never exceed the range of targets seen in training and flatlines beyond the data. On tabular accuracy it often loses to boosting because variance reduction has a ceiling at the correlation floor, while boosting keeps reducing bias. Its impurity-based importances are also biased toward high-cardinality features.
