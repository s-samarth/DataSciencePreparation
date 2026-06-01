# Tree Algorithms & Categorical Tricks

Three generations of decision-tree algorithms led to the CART-derived implementations every library ships today. This page covers the ID3, C4.5, and CART lineage, why binary splits won, and the two distinct tricks, Breiman's sort-by-target and CatBoost's ordered target encoding, that defuse the combinatorial blowup of categorical splits.

!!! tip "Rapid Recall"
    ID3 used multi-way entropy splits, C4.5 added gain ratio and continuous and missing-value handling, and CART introduced strictly binary Gini or variance splits and unified classification with regression. Binary won because multi-way splits fragment the data and cannot reuse a feature, while binary splits are data-efficient. The whole modern ecosystem is CART-derived. A categorical with k values has \(2^{k-1}-1\) possible binary partitions; Breiman's theorem sorts categories by positive rate and tests only k-1 contiguous cuts, while CatBoost converts categories to numbers using only prior rows to prevent target leakage.

## §1 ID3 / C4.5 / CART and library reality

Three generations of decision-tree algorithms, a lineage.

| Algorithm | Criterion | Splits | Notes |
| --- | --- | --- | --- |
| **ID3** (1986, Quinlan) | Information Gain (entropy) | Multi-way (one branch per value) | Categorical only. No continuous, no missing, no pruning. IG biased toward high-cardinality features. |
| **C4.5** (1993, Quinlan) | Gain Ratio = IG / split-info | Multi-way | Split-info (entropy of partition sizes) penalizes many-way splits → fixes high-cardinality bias. Adds continuous, missing-value handling, post-pruning. |
| **CART** (1984, Breiman) | Gini / variance reduction | **Strictly binary** | Parallel lineage (statistics community). Unifies classification and regression. Cost-complexity pruning, surrogate splits for missing. |

### Why binary (CART) beat multi-way (C4.5)

Multi-way splits **fragment data fast**, a 10-value feature spends all samples in one split and can't be reused. Binary splits are data-efficient, let you revisit a feature at multiple depths, and unify classification/regression. The cost, apparent \(2^{k-1}\) blowup on categoricals, is defused by Breiman's theorem (below).

### What libraries actually use

| Library | What it is |
| --- | --- |
| **sklearn** | Optimized CART, binary, Gini default (entropy optional). *Does not* natively handle categoricals, you must one-hot / ordinal-encode yourself. |
| **XGBoost** | CART-style binary; own regularized gain (Taylor expansion of loss); histogram-based candidates; native categorical in recent versions. |
| **LightGBM** | CART-style, histogram-based, leaf-wise growth; native categorical via Breiman's sort-by-target. |
| **CatBoost** | CART-style oblivious (symmetric) trees; signature ordered target encoding for categoricals. |

!!! note "What to use"
    Nobody ships ID3/C4.5 in production, the entire modern ecosystem is *CART-derived*. In practice: sklearn trees / RandomForest as baselines, XGBoost / LightGBM / CatBoost for performance.

## §2 Categorical tricks: Breiman and CatBoost

Two *different* tricks for the same problem: CART must split a categorical into two groups, and \(k\) values give \(2^{k-1}-1\) partitions.

### Trick A: Breiman's theorem (sort-by-target shortcut)

For binary classification: compute each category's **fraction of positive class**, sort categories by it, treat the ordering like a continuous feature, and test only the \(k-1\) contiguous cuts.

| City | % positive |
| --- | --- |
| Pune | 0.10 |
| Delhi | 0.35 |
| Chennai | 0.60 |
| Mumbai | 0.85 |

Sorted: Pune → Delhi → Chennai → Mumbai. Test 3 contiguous cuts, not 7 subsets. For \(k=20\): **19 instead of about 500,000.** Blowup becomes linear in \(k\), exactly like the continuous midpoint trick.

!!! note "Why it is provably optimal"
    The impurity-minimizing split groups categories that "behave the same" (similar positive-rate) together. Putting Pune (0.10) and Chennai (0.60) together while separating Delhi (0.35, which sits *between* them) is incoherent, it splits the middle while joining the extremes, always producing messier children. The optimal binary partition is always an *interval in the positive-rate ordering*, so contiguous cuts are the only candidates that can win.

!!! warning "The catch"
    The clean guarantee holds only for *binary classification and regression* (a single scalar to sort by). For multiclass there's no single scalar, libraries fall back to heuristics. Common trap.

### Trick B: CatBoost's ordered target encoding

CatBoost doesn't subset at all, it **converts each category to a number** (roughly the mean target for that category) then treats it as continuous.

!!! warning "The problem: target leakage"
    Naive target encoding computes a category's value from *all* its rows including the current one, so a category appearing once is encoded as exactly its own label. The model "predicts" perfectly on training, catastrophically overfits, and the feature looks far more useful than it is.

**The fix, "ordered" encoding:** impose a random ordering on rows (artificial "time"); to encode a row, use *only rows before it* in that ordering, never the row itself, never future rows, with a smoothing constant and the global prior handling few or zero prior occurrences. Because each row sees only the *past*, its own label can never leak into its own feature. CatBoost averages over several random orderings to cut variance. This "look only at the past" principle is CatBoost's defining idea, applied to both target encoding *and* gradient computation.

### A vs B: hold them apart

|  | Breiman (LightGBM) | CatBoost ordered encoding |
| --- | --- | --- |
| What it does | Keeps category categorical; finds best subset split cheaply via sort-by-target | Converts category to a number, treats as continuous |
| Solves | The \(2^{k-1}\) combinatorial blowup | High-cardinality categoricals + target leakage |
| Key idea | Optimal subset = interval in positive-rate order | Encode using only *prior* rows to prevent leakage |
| Output | A binary partition of categories | A numeric feature |

## Interview questions

**Q1: Trace the ID3, C4.5, CART lineage.**
ID3 used multi-way splits on categorical features with information gain, but had no continuous or missing-value handling and was biased toward high-cardinality features. C4.5 fixed that bias with gain ratio, dividing gain by the split information, and added continuous features, missing values, and post-pruning, still multi-way. CART, from the statistics community, used strictly binary splits with Gini or variance reduction, unified classification and regression, and added cost-complexity pruning, and it is the basis of every modern library.

**Q2: Why did binary splits win over multi-way splits?**
Multi-way splits fragment the data fast, since a many-valued feature consumes all the samples in one split and cannot be reused deeper in the tree. Binary splits are data-efficient, let a feature be revisited at multiple depths, and naturally unify classification and regression. Their apparent cost, the \(2^{k-1}\) ways to partition a categorical, is defused by Breiman's theorem.

**Q3: How does Breiman's theorem make categorical splits tractable, and when does it fail?**
For binary classification or regression it sorts the categories by their mean target (positive rate), then tests only the k-1 contiguous cuts in that order, since the optimal binary partition is provably an interval in that ordering, reducing \(2^{k-1}\) candidates to k-1. Grouping a category with others of similar target while keeping a between-valued category apart always yields messier children, so only contiguous cuts can win. It fails for multiclass, where there is no single scalar to sort by and libraries fall back to heuristics.

**Q4: What problem does CatBoost's ordered target encoding solve, and how?**
It solves target leakage in naive target encoding, where computing a category's encoded value from all its rows, including the current one, lets a row's own label leak into its feature and causes catastrophic overfitting. CatBoost imposes a random ordering and encodes each row using only the rows before it, plus a smoothing constant and global prior, so a row never sees its own label, and it averages over several orderings to reduce variance. This look-only-at-the-past principle also governs its gradient computation.
