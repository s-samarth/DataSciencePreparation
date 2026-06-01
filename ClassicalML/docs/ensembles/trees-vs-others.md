# Trees vs Other Models

Boosted trees still own tabular data in 2026, but knowing exactly where they win, where other models win, and why trees need almost no preprocessing is what separates a recited answer from a real one. This page covers the shared boosting weaknesses, the dominance map, and the preprocessing contrast.

!!! tip "Rapid Recall"
    Boosted trees dominate tabular and structured data with nonlinear relationships, interactions, mixed feature types, and threshold effects, and they need almost no preprocessing because they use only the rank order within each feature. Other models win on linear or extrapolation problems (linear models), very high-dimensional sparse text (linear or SVM), and unstructured images, audio, and sequences (neural nets). Trees do not need scaling, are invariant to monotonic transforms, tolerate outliers and multicollinearity, and handle interactions automatically by depth. The shared weaknesses are high-cardinality categoricals, no extrapolation, and sequential training.

## §1 Boosting weaknesses (shared)

- **High-cardinality categoricals, still a problem** (CatBoost mitigates best via ordered encoding, doesn't solve). Rare levels → leakage/overfit in target encoding; one-hot → thousands of sparse columns trees split poorly. Real fix often domain-level: group rare levels, hash, embeddings.
- Overfits without early stopping/LR tuning; sequential training; hyperparameter/noise sensitivity; no regression extrapolation; weak on high-dim sparse and unstructured data; less interpretable.

## §2 When trees dominate vs when others win

| Trees / boosting dominate | Other models win |
| --- | --- |
| Tabular / structured data (still SOTA 2026) | Linear/near-linear relationships → linear/logistic regression |
| Nonlinear relations + interactions (auto-discovered) | Need to extrapolate beyond training range → linear |
| Mixed feature types and scales (scale-invariant) | Very high-dim sparse (text TF-IDF) → linear/SVM |
| Threshold/step-like effects ("risk jumps after 60") | Unstructured: images/audio/text/sequences → neural nets |
| Moderate sizes, little preprocessing, fast results | Smooth target functions; strong interpretability; tiny datasets |

!!! note "One-line heuristic"
    Tabular + nonlinear + mixed types → boosted trees. Linear / extrapolation / interpretable → linear models. Unstructured (images/text/audio) → neural nets. High-dim sparse → linear. In 2026 the "trees own tabular, nets own perception" split still holds.

## §3 Preprocessing: trees vs classical models

| Step | Trees | Linear / SVM / NN / KNN | Why |
| --- | --- | --- | --- |
| Feature scaling | **Not needed** | **Required** | Trees split on per-feature thresholds, only value *ordering* matters; distance/gradient models are dominated by large-scale features. |
| Monotonic transform (log) | No effect on predictor | Often helpful | Split "x < t" is invariant to monotonic transforms; linear models linearize via log. |
| Outliers | Robust (isolated in a leaf) | Sensitive | Trees partition; linear/distance models get pulled. (Boosting under squared loss does chase outlier residuals.) |
| Categorical encoding | One-hot or native | One-hot/target needed | Linear needs numeric; some tree libs native. |
| Missing values | Native (XGB/LGBM/CatBoost) | Imputation required | Trees route NaN down a learned branch; linear/NN math can't accept NaN. |
| Multicollinearity | Tolerant | Problematic (linear) | Trees don't invert a covariance matrix; correlated features don't destabilize (but confuse importances). |
| Interactions / nonlinear features | Automatic (depth) | Manual (cross/poly terms) | Depth-\(d\) tree → \(d\)-way interactions for free. |

!!! note "Unifying principle"
    Trees are *invariant to monotonic, per-feature transformations*, they use only the rank order of values within each feature and partition one feature at a time. Anything computing a distance, dot product, or cross-feature gradient (KNN, SVM, linear, NN) is scale-sensitive and needs standardization, outlier care, and manual interaction/nonlinear engineering. (One caveat: a skewed *target* can still benefit from log-transform even for trees, it changes the loss landscape, not feature ordering.)

## Interview questions

**Q1: When do boosted trees dominate, and when would you reach for something else?**
Trees dominate tabular and structured data with nonlinear relationships, automatically discovered interactions, mixed feature types and scales, and threshold or step-like effects, and they remain state of the art there in 2026. You reach for linear or logistic models when the relationship is near-linear, when you must extrapolate beyond the training range, or when interpretability matters; for very high-dimensional sparse text you use linear or SVM; and for unstructured images, audio, and sequences you use neural nets.

**Q2: Why do trees need almost no preprocessing while linear, SVM, KNN, and neural nets do?**
Trees split on per-feature thresholds and use only the rank order of values within each feature, so feature scaling and any monotonic transform leave the splits unchanged, outliers get isolated in a leaf, and correlated features do not destabilize them. Anything computing a distance, dot product, or cross-feature gradient is dominated by large-scale features and sensitive to outliers, so it requires standardization, outlier handling, and often manual interaction engineering.

**Q3: Why can a tree handle feature interactions for free but a linear model cannot?**
A depth-d tree conditions each split on the path above it, so a leaf reached after splitting on several features encodes a d-way interaction automatically. A linear model is additive in its inputs, so it can only represent interactions if you manually construct cross or polynomial terms. This is why trees excel at tabular data with rich interactions with little feature engineering.

**Q4: What weaknesses do all boosting methods share?**
High-cardinality categoricals remain a problem: rare levels cause leakage in target encoding and one-hot produces thousands of sparse columns trees split poorly, so the real fix is often domain-level grouping, hashing, or embeddings. They also overfit without early stopping and learning-rate tuning, train sequentially, cannot extrapolate in regression, are weak on high-dimensional sparse and unstructured data, and are less interpretable than a single tree.
