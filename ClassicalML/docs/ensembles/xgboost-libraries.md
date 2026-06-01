# XGBoost & Modern Libraries

XGBoost turned gradient boosting from "fit the negative gradient with a heuristic tree" into an explicit regularized objective optimized with a second-order Taylor expansion, with a closed-form Newton leaf. This page covers that objective, how the libraries parallelize a sequential algorithm and handle missing values, and the differences between XGBoost, LightGBM, and CatBoost.

!!! tip "Rapid Recall"
    XGBoost writes a regularized objective and optimizes it with second-order info: per point it needs only the gradient \(g_i\) and Hessian \(h_i\), and because the loss is a sum over points the Hessian is diagonal, so there is no matrix to invert. The closed-form leaf is \(w_j^*=-\frac{G_j}{H_j+\lambda}\) and the split gain replaces Gini or variance, with built-in pruning when gain falls below \(\gamma\). Boosting cannot parallelize across trees but parallelizes split-finding across features via histograms. XGBoost learns a default direction for missing values. LightGBM grows leaf-wise for speed; CatBoost uses oblivious trees and leakage-safe ordered target encoding.

## §1 XGBoost: regularized second-order objective

Vanilla GBM fits the negative gradient with a heuristic tree criterion. XGBoost writes an explicit regularized objective and optimizes it with a second-order Taylor expansion.

### The diagonal-Hessian insight (clears the cost worry)

!!! note "Why the Hessian is cheap"
    The loss is a *sum over points*, and each point's loss depends only on *its own* prediction \({\hat{y}}_i\), no cross terms. So \(\frac{{\partial}^2 L}{\partial{\hat{y}}_i\,\partial{\hat{y}}_j}=0\) for \(i\neq j\): the Hessian is *diagonal* = just a *vector of per-point scalars*. No matrix, no inversion. Per point you compute two numbers:
    $$g_i=\frac{\partial L(y_i,{\hat{y}}_i)}{\partial{\hat{y}}_i},\quad h_i=\frac{{\partial}^2 L(y_i,{\hat{y}}_i)}{\partial{\hat{y}}_i^2}$$

Squared error: \(g_i={\hat{y}}_i-y_i\), \(h_i=1\). Log loss: \(g_i=p_i-y_i\), \(h_i=p_i(1-p_i)\). Both \(O(n)\), as cheap as the gradient.

### The objective and closed-form leaf

$$L^{(t)}=\sum_i L(y_i,{\hat{y}}_i^{(t-1)}+f_t(x_i))+\Omega(f_t),\quad\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_j w_j^2$$

(\(T\) = number of leaves, \(w_j\) = leaf value.) Second-order Taylor, grouped by leaf with \(G_j=\sum_{i\in j}g_i\), \(H_j=\sum_{i\in j}h_i\), gives a quadratic in each \(w_j\), minimized in closed form:

!!! note "Newton leaf weight and split gain"
    $$w_j^*=-\frac{G_j}{H_j+\lambda},\qquad\text{Gain}=\frac{1}{2}\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma$$

- The leaf value uses *both* derivatives, a Newton step. \(H_j\) in the denominator scales steps by curvature → faster, more stable convergence than GBM's mean-residual step.
- The gain **replaces Gini/entropy/variance** as the split criterion. If the best gain < \(\gamma\), don't split, regularization pruning built into the criterion.

## §2 Parallelization, GPU, and missing values

### Parallelization: boosting is still sequential across trees

!!! note "What is and isn't parallel"
    XGBoost does *not* parallelize across trees (tree \(t\) needs tree \(t-1\)). It parallelizes the work *inside one tree*, split-finding is *embarrassingly parallel across features* (feature A's best split is independent of feature B's). A pre-sorted, compressed columnar "block" format lets split-finding be a parallel scan over columns without re-sorting per node (also cache-efficient).

### GPU: the parallelism is in split-finding, not trees

The GPU accelerates the per-tree gain computation, which is massively data-parallel. The modern **histogram method**: bucket each feature into about 256 bins, then per node compute a **histogram of gradient/Hessian sums per bin**, a giant parallel reduction over millions of rows (both *vectorized* and *parallel* across rows/features/same-depth nodes). The sequential-across-trees nature is irrelevant because the GPU's job is the within-tree histogram math.

### Missing values

| Model | Missing-value handling |
| --- | --- |
| Classic sklearn GBM | None, **manual imputation** required. (Newer `HistGradientBoosting*` does handle it.) |
| **XGBoost** | **Sparsity-aware split finding:** at each split, tries sending all missing points left vs right, picks the higher-gain direction = a *learned default direction*. At inference, NaN flows that way. No imputation; learned from data. |
| LightGBM | Native (default-direction style). |
| CatBoost | Native (treats missing as a special value). |

## §3 XGBoost vs LightGBM vs CatBoost

All three = vanilla GBM + regularized objective, second-order info, histograms, native missing handling, parallel/GPU, early stopping. They differ in growth strategy and categorical handling.

|  | sklearn GBM | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- |
| Tree growth | Depth-wise | Level-wise (or lossguide) | **Leaf-wise** (best-first) | **Symmetric / oblivious** |
| Split finding | Exact, pre-sorted | Exact or histogram | Histogram (GOSS, EFB) | Histogram, oblivious |
| Speed | Slowest | Fast | **Fastest on large data** | Fast |
| Categoricals | Manual encode | Native (recent) | Native (Breiman sort-by-target) | **Native ordered target encoding** |
| Regularization | Minimal | L1/L2 + γ + structure | L1/L2 + leaf constraints | Ordered boosting (anti-leakage) |
| Best at | Baselines | Robust all-rounder | Huge datasets, speed | Many categoricals, less tuning |

- **LightGBM, leaf-wise growth:** splits the single highest-gain leaf anywhere in the tree → deeper, asymmetric trees, faster loss reduction per split, often higher accuracy, but *more overfit-prone* (control with `num_leaves`/`max_depth`). Plus **GOSS** (keep large-gradient points, subsample small-gradient ones) and **EFB** (bundle mutually-exclusive sparse features).
- **CatBoost, oblivious trees:** same split condition across a whole level → balanced, regularized, fast inference, overfit-resistant. Plus **ordered boosting** (the "use only prior rows" anti-leakage principle applied to gradient computation, not just encoding). Headline strength: native, leakage-safe categorical handling with minimal tuning.

!!! note "What to use"
    XGBoost = robust default all-rounder. LightGBM = large data / speed. CatBoost = many categoricals / minimal tuning. All three beat sklearn GBM in speed and usually accuracy; sklearn GBM is now mostly a baseline.

## §4 New hyperparameters (beyond LR / n_estimators / depth / subsample)

### Regularization (the "better regularization")

- `gamma` / `min_split_loss` (\(\gamma\)): **minimum gain to make a split.** Split only if it improves the objective by more than \(\gamma\). Higher = more pruning. Straight from the gain formula.
- `lambda` / `reg_lambda` (\(\lambda\)): **L2 penalty on leaf weights**, the \(\lambda\) in \(w_j^*=-G_j/(H_j+\lambda)\). Shrinks leaf values, smooths predictions. Default about 1.
- `alpha` / `reg_alpha`: **L1 penalty on leaf weights**, drives some to exactly zero (sparsity). Default 0; useful with many irrelevant features.
- `min_child_weight`: **minimum sum of Hessians (\(\sum h_i\)) in a leaf** to split. For squared error (\(h_i=1\)) = min samples; for log loss = min "confidence mass." Higher = more conservative.

### Stochasticity / decorrelation

- `subsample`: row fraction per tree (as in GBM).
- `colsample_bytree` / `bylevel` / `bynode`: **feature fraction** per tree / level / split, RF-style feature subsampling inside boosting.

### Library-specific

- **LightGBM:** `num_leaves` (the key complexity knob for leaf-wise trees, more than `max_depth`), `min_data_in_leaf`.
- **CatBoost:** `cat_features`, `one_hot_max_size`, `depth` (oblivious-tree depth = main capacity knob).

## Interview questions

**Q1: Why is the Hessian in XGBoost cheap to use despite Newton's usual \(O(d^3)\) cost?**
Because the loss is a sum of per-point losses, each depending only on its own prediction, so the mixed second partials vanish and the Hessian is diagonal, just a vector of per-point scalars \(h_i\). There is no matrix to form or invert; per point you compute the gradient \(g_i\) and Hessian \(h_i\), both \(O(n)\). For squared error \(h_i=1\) and for log loss \(h_i=p_i(1-p_i)\).

**Q2: What does the XGBoost objective add over vanilla GBM, and what is the leaf weight?**
It adds an explicit regularizer \(\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_j w_j^2\) and optimizes a second-order Taylor expansion rather than fitting a heuristic tree to the gradient. Grouping by leaf gives the closed-form Newton leaf \(w_j^*=-G_j/(H_j+\lambda)\), and the split gain replaces Gini or variance, with a split rejected if its gain is below \(\gamma\). The curvature term \(H_j\) in the denominator makes the steps better scaled and more stable than GBM's mean-residual step.

**Q3: What in boosting is parallel, given it is sequential across trees?**
The trees themselves cannot be parallelized because tree t needs the predictions of tree t-1, but the split-finding inside one tree is embarrassingly parallel across features, since each feature's best split is independent. XGBoost stores data in a pre-sorted compressed columnar block and uses histograms, bucketing each feature and computing per-bin gradient and Hessian sums as a giant parallel reduction over rows, which is also what the GPU accelerates.

**Q4: How do XGBoost, LightGBM, and CatBoost differ?**
They share the regularized second-order objective, histograms, native missing handling, and early stopping, and differ mainly in growth and categoricals. XGBoost grows level-wise and is the robust all-rounder; LightGBM grows leaf-wise with GOSS and EFB for the fastest training on large data, at higher overfit risk controlled by num_leaves; CatBoost uses balanced oblivious trees with leakage-safe ordered target encoding and ordered boosting, best for many categoricals with minimal tuning.
