# Building Decision Trees

A decision tree is greedy recursive partitioning: at each node, find the single best split, partition, and recurse. This page covers the build loop, how continuous and categorical features generate a finite set of candidate splits, when to stop and how to prune, the small change that turns a classifier into a regressor, and the training and inference complexity.

!!! tip "Rapid Recall"
    Build greedily and recursively: stop if a stopping condition fires, otherwise pick the best (feature, split-point) by impurity reduction over all features, partition, and recurse. Continuous features need only the midpoints between adjacent distinct values, at most \(n-1\) candidates, and gain only changes when the threshold crosses a data point. Without limits a tree grows until every leaf is pure and memorizes the data, so you pre-prune with depth and leaf-size caps or post-prune by cost complexity. Regression trees swap impurity for variance and the leaf vote for the mean. Training is about \(O(d\,n\log^2 n)\); inference is \(O(\text{depth})\), independent of d and n.

## §1 How a decision tree is built

Greedy recursive partitioning. "Recursive" = same procedure per node. "Greedy" = locally best split, no look-ahead.

```
BuildTree(samples S):
    1. Should I stop?  (stopping conditions) -> if yes, make S a LEAF, return.
    2. Otherwise, find the single best (feature, split-point) over ALL features.
    3. Split S into children by that test.
    4. Recurse: BuildTree on each child.
```

### Step 2: finding the best split

```
best_gain = 0; best_split = None
for each feature f:
    for each candidate split-point t on feature f:
        partition S into children using (f, t)
        gain = impurity(S) - weighted_impurity(children)
        if gain > best_gain:
            best_gain = gain; best_split = (f, t)
```

"Evaluate a split" = partition the samples, compute weighted child impurity, subtract from parent. The only open question is *what the set of candidate split-points is* per feature type, covered next.

<figure class="diagram diagram-dark" markdown="0">
<svg aria-label="Recursive partitioning tree" role="img" viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg">
<!-- root -->
<rect fill="#1c222e" height="40" rx="8" stroke="#e8b059" stroke-width="1.5" width="100" x="230" y="20"></rect>
<text fill="#e6e9ef" font-family="monospace" font-size="12" text-anchor="middle" x="280" y="44">S (root)</text>
<!-- level 2 -->
<line stroke="#2a3240" x1="280" x2="150" y1="60" y2="100"></line>
<line stroke="#2a3240" x1="280" x2="410" y1="60" y2="100"></line>
<rect fill="#1c222e" height="40" rx="8" stroke="#5fb3a3" stroke-width="1.5" width="100" x="100" y="100"></rect>
<text fill="#e6e9ef" font-family="monospace" font-size="12" text-anchor="middle" x="150" y="124">child A</text>
<rect fill="#1c222e" height="40" rx="8" stroke="#5fb3a3" stroke-width="1.5" width="100" x="360" y="100"></rect>
<text fill="#e6e9ef" font-family="monospace" font-size="12" text-anchor="middle" x="410" y="124">child B</text>
<!-- level 3 -->
<line stroke="#2a3240" x1="150" x2="90" y1="140" y2="190"></line>
<line stroke="#2a3240" x1="150" x2="210" y1="140" y2="190"></line>
<rect fill="#161b24" height="36" rx="8" stroke="#c97b8e" stroke-width="1.2" width="90" x="45" y="190"></rect>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="middle" x="90" y="212">LEAF</text>
<rect fill="#161b24" height="36" rx="8" stroke="#c97b8e" stroke-width="1.2" width="90" x="165" y="190"></rect>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="middle" x="210" y="212">LEAF</text>
<text fill="#9aa6b8" font-family="Helvetica" font-size="12" text-anchor="middle" x="410" y="215">...recurse until stop</text>
<text fill="#e8b059" font-family="monospace" font-size="10" x="217" y="86">max-gain split</text>
</svg>
<figcaption>Each level partitions the data into disjoint groups; sample counts across siblings at one level sum to n.</figcaption>
</figure>

## §2 Splitting: continuous and categorical features

### Continuous: "won't there be infinite splits?"

!!! note "Key insight"
    Gain only changes when the threshold *crosses an actual data point.* Between consecutive values (say 28 and 35), every threshold (30, 31, 32.7…) yields the *same* partition, so the same gain. So test only *one candidate per gap*, the midpoint between adjacent distinct values. With \(n\) samples: at most \(n-1\) candidates. Finite, not infinite. Splits are always binary ("feature < t?").

Procedure: **sort** by the feature (\(O(n\log n)\)), then candidate thresholds = midpoints between adjacent distinct values. Sorted [22, 28, 35, 40] → candidates {25, 31.5, 37.5}.

#### Speed optimizations

- **Incremental sweep:** after sorting once, sweep left to right updating class counts (move one sample from right child to left at each step) instead of recomputing impurity, each threshold becomes \(O(1)\); the feature is \(O(n\log n)\), dominated by the sort.
- **Histogram binning (XGBoost / LightGBM):** pre-bucket into e.g. 256 bins, only consider bin boundaries. Per-node cost depends on #bins, not \(n\). The single biggest reason modern boosted trees are fast on large tabular data.

### Categorical: "won't permutations blow up?"

| Regime | How it splits | Issue |
| --- | --- | --- |
| **Multi-way** (ID3 / C4.5) | One branch per category value | No combinatorial blowup, but high-cardinality features (ZIP, 40k values) make many tiny near-pure children → fake high IG. Fix: **Information Gain Ratio** = IG / split-info. |
| **Binary subset** (CART) | Partition categories into two subsets | \(2^{k-1}-1\) possible partitions, the real blowup. \(k=20\) → about 500k. Defused by Breiman's shortcut. |

Escapes from the blowup: **Breiman's sort-by-target shortcut** (collapses \(2^{k-1}\) to \(k-1\)) and **target/ordinal encoding** (CatBoost, convert category to a number, treat as continuous). Both detailed in [Tree Algorithms](tree-algorithms.md).

## §3 Stopping, pruning, and leaf predictions

### When to stop expanding (make a leaf)

- **Pure node** — all one class, impurity 0, nothing to gain.
- **No positive gain** — best available IG is about 0; no feature separates further.
- **Pre-pruning / regularization limits** (hyperparameters): `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.

!!! warning "Why limits exist"
    Without them a tree grows until every leaf is pure, a leaf per sample in the worst case, i.e. it *memorizes the training data*, severe overfitting. This is why a lone decision tree almost always overfits, and why these caps exist.

### Pre- vs post-pruning

**Pre-pruning** (the limits above) is greedy and can stop too early, a weak split might enable a great split below it (the XOR problem). **Post-pruning** grows the tree fully then prunes branches that don't help validation (cost-complexity pruning / `ccp_alpha` in sklearn, penalizes tree size, snips subtrees whose accuracy gain doesn't justify complexity). Post-pruning avoids early stopping but costs more compute. In practice: pre-pruning limits plus ensembling.

### Leaf predictions

- **Classification:** majority class of the leaf's samples (or class proportions for predicted probabilities).
- **Regression:** mean of the leaf's target values (sometimes median).

## §4 Classification to regression trees

Same skeleton; two things change.

### Stays identical

The entire structure, greedy recursive partitioning, the candidate-split machinery (continuous sort-and-midpoint, categorical handling), stopping conditions, greedy argmax-over-splits loop.

### What changes

1. **The impurity measure.** Entropy/Gini measure class mixedness, meaningless for a continuous target. Replace with **variance / MSE**: node impurity = variance of its targets. The criterion becomes **variance reduction**:

    $$\text{VarReduction}=\text{Var}(S)-\sum_v\frac{|S_v|}{|S|}\,\text{Var}(S_v)$$

    Same "parent minus size-weighted children" formula, Var swapped for \(H\). (MAE is a robust-to-outliers but slower alternative.)
2. **The leaf prediction.** Class (majority vote) → number (mean of leaf's targets).

!!! note "Summary"
    Same algorithm; swap "how I measure messiness" (entropy/Gini → variance) and "what I output" (vote → average). Everything about gain, weighting, splitting, stopping transfers directly.

## §5 Complexity: training and inference

Notation: \(n\) = number of training samples (rows), \(d\) = number of features (columns).

### Training

**One node, \(k\) samples:** for each feature, sort (\(O(k\log k)\)) then sweep candidate thresholds incrementally. Across \(d\) features: \(O(d\,k\log k)\).

**Sum over nodes, level by level.** At any depth level, nodes partition the data into disjoint groups, so sample counts across siblings sum to \(n\). Sorting cost across one level is \(O(d\,n\log n)\) (the \(k\log k\) terms sum to at most \(n\log n\) since \(\sum k_i=n\)). Multiply by depth = \(O(\log n)\) for a balanced tree:

$$\boxed{\,O(d\cdot n{\log}^2 n)\,}$$

!!! note "Where the two logs come from (don't conflate)"
    One \(\log n\) = *sorting* samples per node. One \(\log n\) = *tree depth* (number of levels).

**Presort optimization:** sort every feature once up front (\(O(d\,n\log n)\)), maintain order while partitioning instead of re-sorting per node. Per-level work becomes \(O(d\,n)\); total \(O(d\,n\log n)\) for a balanced tree. So both \(O(dn{\log}^2 n)\) (re-sort per node) and \(O(dn\log n)\) (presorted) appear as "the" answer, know which assumption each makes.

**Worst case** (degenerate tree, each split peels one sample, depth \(O(n)\)): training \(O(d\,n^2)\), bounded in practice by depth / min-samples limits.

### Inference

Walk one sample root to leaf, one \(O(1)\) comparison per node on the path:

$$\boxed{\,O(\text{depth})=O(\log n)\text{\,}\text{balanced},\,O(n)\text{\,}\text{worst case}\,}$$

The asymmetry (interview point): inference depends on *depth only*, not on \(d\) (you touch only the one feature each node tests, not all features) and not on \(n\) (the tree is already built; training-set size doesn't appear). This is why tree ensembles are viable in latency-sensitive serving.

|  | Balanced (typical) | Worst case |
| --- | --- | --- |
| Training (re-sort per node) | \(O(d\,n{\log}^2 n)\) | \(O(d\,n^2\log n)\) |
| Training (presorted once) | \(O(d\,n\log n)\) | \(O(d\,n^2)\) |
| Inference (per sample) | \(O(\log n)\) | \(O(n)\) |
| Space (store tree) | \(O(n)\) | \(O(n)\) |

Space: number of nodes is at most \(O(n)\) since each leaf holds at least 1 sample, a tree can't have more leaves than samples.

## Interview questions

**Q1: How does a tree avoid infinitely many thresholds on a continuous feature?**
The gain only changes when the threshold crosses an actual data point, since every threshold between two consecutive values produces the identical partition and hence the identical gain. So you sort the feature and test only the midpoints between adjacent distinct values, at most \(n-1\) candidates, and each split is binary. Sorting dominates at \(O(n\log n)\), and an incremental left-to-right sweep makes each threshold evaluation \(O(1)\).

**Q2: Why does a single decision tree overfit, and how do you control it?**
Without limits the tree keeps splitting until every leaf is pure, in the worst case one leaf per sample, which is memorization. You control it by pre-pruning with caps like max depth, min samples per split or leaf, and min impurity decrease, or by post-pruning, growing fully then snipping subtrees whose validation gain does not justify their complexity via cost-complexity pruning. Pre-pruning is greedy and can stop too early on problems like XOR, so in practice people combine caps with ensembling.

**Q3: What changes when you turn a classification tree into a regression tree?**
Only two things. The impurity measure becomes variance, with the split criterion being variance reduction in the same parent-minus-weighted-children form, since class entropy is meaningless for a continuous target. And the leaf prediction becomes the mean of the leaf's targets instead of a majority vote. Everything about the gain, weighting, splitting, and stopping carries over unchanged.

**Q4: Why does inference complexity not depend on the number of features or samples, while training does?**
At inference you walk a single root-to-leaf path and test only the one feature each node splits on, so cost is \(O(\text{depth})\), about \(\log n\) for a balanced tree, with no dependence on d or n because the tree is already built. Training must search all d features at every node to choose the best split, and sorting samples per level plus the tree depth gives the two log factors, roughly \(O(d\,n\log^2 n)\).
