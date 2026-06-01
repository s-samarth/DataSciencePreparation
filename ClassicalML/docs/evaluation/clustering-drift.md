# Clustering, Drift & Explainability

This page closes the metrics section with three production concerns: judging clusters without labels, detecting when production data has drifted from training, and explaining individual predictions. SHAP gets the depth treatment because it is the rare ML tool with a real theorem behind it.

!!! tip "Rapid Recall"
    Silhouette combines a point's intra-cluster tightness and nearest-cluster separation into a score in [-1,1] used to choose K; Davies-Bouldin is a faster relative index where lower is better; ARI compares clusters to ground truth, adjusted for chance. PSI and KS detect input drift, with PSI above 0.2 the retraining trigger, and PSI equals the symmetric sum of the two KL divergences. SHAP attributes a prediction to its features using Shapley values, the unique fair split satisfying efficiency, symmetry, dummy, and linearity, so contributions sum exactly to prediction minus baseline; TreeSHAP makes it exact and polynomial for tree ensembles.

## §1 Clustering metrics

Clustering has no labels, so how do you know if it worked? Intrinsic metrics measure cluster quality without ground truth; extrinsic metrics require ground truth labels.

### Silhouette Score

**Intuition:** For each point, ask "am I in the right cluster?" Compare how close I am to points in my own cluster (\(a\), intra-cluster distance) versus the nearest other cluster (\(b\), nearest-cluster distance).

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Range:** \([-1, 1]\). \(s\approx 1\) means well-matched to its cluster and far from others (good); \(s\approx 0\) means on the border between two clusters; \(s < 0\) means probably in the wrong cluster. The **average silhouette** across all points is used to choose the optimal number of clusters K.

### Davies-Bouldin Index

**Intuition:** For each cluster, find the cluster most "similar" to it (where similarity = high intra-cluster spread plus low inter-cluster distance), and average that worst-case similarity.

$$\text{DB} = \frac{1}{K}\sum_i \max_{j\neq i}\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$$

where \(\sigma_i\) is the average distance of points in cluster \(i\) to its centroid and \(d(c_i,c_j)\) the distance between centroids. **Lower is better.** Unlike Silhouette, DBI has no fixed range.

| | Silhouette | Davies-Bouldin |
|---|-----------|----------------|
| Range | [-1, 1] (interpretable) | [0, ∞) (relative only) |
| Computation | O(n²) pairwise distances | O(n·K) faster |
| Interpretation | Per-point quality available | Only cluster-level |
| Better when | N is small enough for pairwise | N is large |

### Adjusted Rand Index (ARI)

**Intuition:** If you have ground truth labels, ARI measures agreement between true and predicted clusters, adjusted for chance. Without the adjustment, random assignments can score high simply due to many clusters.

$$\text{ARI} = \frac{\text{RI} - \text{Expected RI}}{\text{Max RI} - \text{Expected RI}}$$

where the Rand Index (RI) is the fraction of point pairs that are either together in both assignments or apart in both. **Range:** \([-1, 1]\); ARI = 1 is perfect agreement, about 0 is random, below 0 is worse than random. Used when you have ground truth and want to compare clustering algorithms.

## §2 Data drift metrics

Models degrade in production because the world changes: the data your users generate today may differ from training data. These metrics detect when your model is going stale.

### PSI (Population Stability Index)

**Intuition:** "Has the distribution of my input features shifted since training?" PSI compares the training distribution to the current production distribution, bucket by bucket.

$$\text{PSI} = \sum (\text{Actual\%} - \text{Expected\%})\,\ln\!\frac{\text{Actual\%}}{\text{Expected\%}}$$

**Thresholds (industry standard):** PSI < 0.1 is no significant shift; 0.1 ≤ PSI < 0.2 is a moderate shift to investigate; PSI ≥ 0.2 is significant, the model likely degraded, retrain. Run PSI on key features weekly or monthly and trigger a retraining pipeline when it crosses 0.2.

### KS Test (Kolmogorov-Smirnov)

**Intuition:** "What's the maximum difference between two cumulative distribution functions?" Unlike PSI which sums differences across bins, KS finds the single point of maximum divergence.

$$\text{KS} = \max_x |F_{\text{train}}(x) - F_{\text{prod}}(x)|$$

Plot two CDFs on the same axes; the KS statistic is the largest vertical gap. KS is a formal statistical test with p-values, while PSI is a practical index with rules of thumb; KS is more sensitive to distributional shape, PSI to frequency changes in specific bins. Use both together.

### KL Divergence (Kullback-Leibler)

**Intuition:** "How much information do I lose using distribution Q (the model's assumption) instead of P (reality)?" KL measures the surprise of seeing data from P when you expected Q.

$$D_{\text{KL}}(P\,\|\,Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$$

**Key properties:** KL ≥ 0 always (Gibbs' inequality), zero iff P = Q, **not symmetric** (so not a true distance), and undefined if Q(x) = 0 where P(x) > 0. PSI is actually a symmetric version of KL: \(\text{PSI} = D_{\text{KL}}(P\|Q) + D_{\text{KL}}(Q\|P)\), so PSI "fixes" the asymmetry.

## §3 Explainability: SHAP and Shapley values

SHAP is the rare ML tool with a real theorem behind it. Everything rests on one idea from cooperative game theory.

**The fundamental reframe, a prediction is a payout to divide.** Shapley (1950s) answered: a team cooperates to produce some value; how do you fairly split it, given players contribute differently and interact? He proved there's exactly **one** split satisfying a set of fairness axioms. Not "a good way," the *unique* way. That uniqueness is why SHAP has weight that LIME and permutation importance lack.

| Game theory | SHAP translation |
| --- | --- |
| The game | Making one prediction for one specific example |
| The players | The feature values of that example |
| The payout | \(f(x) - \mathbb{E}[f(x)]\): how far this prediction lands from average |
| A player's Shapley value | Its fair share of that gap |

!!! note "Why SHAP is natively per-feature, per-example"
    The game is defined for *one example* (its feature values are the players); the solution splits the payout *across features*. The math is local by construction. A global explanation is just an aggregate of many local ones.

**Contribution is marginal, but order matters.** The marginal contribution of \(i\) to a coalition \(S\) is \(f(S\cup\{i\}) - f(S)\). The problem: a feature's marginal contribution depends on who's already in the room (features interact), so there's no single "contribution of income," it depends on the order features joined. Shapley's fix: **average the marginal contribution over every possible order.**

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="Income contributes differently depending on join order; Shapley averages over orders" role="img" viewBox="0 0 680 250" width="100%">
<defs><marker id="ar2" markerHeight="6" markerWidth="6" orient="auto-start-reverse" refX="8" refY="5" viewBox="0 0 10 10"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"></path></marker></defs>
<text fill="#23201b" font-family="Fraunces,serif" font-size="16" font-weight="600" x="20" y="24">Order A: income first, then debt</text>
<rect fill="#eceae4" height="44" rx="6" stroke="#888780" stroke-width="1" width="100" x="20" y="34"></rect><text fill="#444" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="70" y="54">baseline</text><text fill="#444" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="70" y="69">0.30</text>
<line marker-end="url(#ar2)" stroke="#564f44" stroke-width="1.5" x1="120" x2="160" y1="56" y2="56"></line>
<rect fill="#dcefe6" height="44" rx="6" stroke="#1d6e56" width="116" x="162" y="34"></rect><text fill="#0f5840" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="220" y="54">+income</text><text fill="#0f5840" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="220" y="69">→0.55 (+.25)</text>
<line marker-end="url(#ar2)" stroke="#564f44" stroke-width="1.5" x1="278" x2="318" y1="56" y2="56"></line>
<rect fill="#e6e3f2" height="44" rx="6" stroke="#3b3170" width="116" x="320" y="34"></rect><text fill="#3b3170" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="378" y="54">+debt</text><text fill="#3b3170" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="378" y="69">→0.80 (+.25)</text>
<text fill="#23201b" font-family="Fraunces,serif" font-size="16" font-weight="600" x="20" y="118">Order B: debt first, then income</text>
<rect fill="#eceae4" height="44" rx="6" stroke="#888780" width="100" x="20" y="128"></rect><text fill="#444" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="70" y="148">baseline</text><text fill="#444" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="70" y="163">0.30</text>
<line marker-end="url(#ar2)" stroke="#564f44" stroke-width="1.5" x1="120" x2="160" y1="150" y2="150"></line>
<rect fill="#e6e3f2" height="44" rx="6" stroke="#3b3170" width="116" x="162" y="128"></rect><text fill="#3b3170" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="220" y="148">+debt</text><text fill="#3b3170" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="220" y="163">→0.40 (+.10)</text>
<line marker-end="url(#ar2)" stroke="#564f44" stroke-width="1.5" x1="278" x2="318" y1="150" y2="150"></line>
<rect fill="#dcefe6" height="44" rx="6" stroke="#1d6e56" width="116" x="320" y="128"></rect><text fill="#0f5840" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="378" y="148">+income</text><text fill="#0f5840" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="378" y="163">→0.80 (+.40)</text>
<rect fill="#dcefe6" height="60" rx="6" stroke="#1d6e56" width="190" x="470" y="40"></rect><text fill="#0f5840" font-family="Fraunces,serif" font-size="14" font-weight="600" text-anchor="middle" x="565" y="64">income φ</text><text fill="#0f5840" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="565" y="84">avg(.25,.40)=.325</text>
<rect fill="#e6e3f2" height="60" rx="6" stroke="#3b3170" width="190" x="470" y="120"></rect><text fill="#3b3170" font-family="Fraunces,serif" font-size="14" font-weight="600" text-anchor="middle" x="565" y="144">debt φ</text><text fill="#3b3170" font-family="JetBrains Mono,monospace" font-size="11" text-anchor="middle" x="565" y="164">avg(.25,.10)=.175</text>
<rect fill="#faf3df" height="38" rx="6" stroke="#9a6a18" width="640" x="20" y="200"></rect><text fill="#7a5212" font-family="JetBrains Mono,monospace" font-size="12" text-anchor="middle" x="340" y="224">additivity: .325 + .175 = .50 = prediction 0.80 − baseline 0.30 ✓</text>
</svg>
<figcaption>Income adds +0.25 joining first but +0.40 joining after debt; the Shapley value averages its marginal contribution over all join orders.</figcaption>
</figure>

$$\phi_i = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\big[f(S\cup\{i\}) - f(S)\big]$$

- \(f(S\cup\{i\}) - f(S)\) is the marginal contribution.
- \(\sum_{S\subseteq N\setminus\{i\}}\) is over every coalition not already containing \(i\) (every room \(i\) could enter).
- \(\frac{|S|!\,(|N|-|S|-1)!}{|N|!}\) is the combinatorial weight, the fraction of the \(n!\) orderings producing this "join \(S\), then add \(i\)" situation, exactly the average over orderings written compactly.

**The four axioms, why this is the answer:**

- **Efficiency (additivity).** Contributions sum exactly to \(f(x) - \mathbb{E}[f(x)]\), a complete, exact decomposition with nothing left over, the killer property for explainability and compliance.
- **Symmetry.** Features contributing identically to every coalition get equal values.
- **Dummy / Null.** A feature contributing 0 to every coalition gets exactly 0.
- **Linearity.** Shapley values of an ensemble equal the sum of per-model values, which makes TreeSHAP work for boosted forests.

The "consistency" property follows: make a feature matter more and its SHAP value can't go down. No competing method satisfies the whole set, so SHAP is the *provably fair* decomposition.

**How "present vs absent" actually works.** A trained model needs all inputs, so SHAP defines \(f(S)\) as the expected prediction, marginalizing absent features over a background distribution:

$$f(S) = \mathbb{E}_{x_{\bar{S}}}\big[f(x_S, x_{\bar{S}})\big]$$

Hold present features at this example's values; for absent ones, average the output over plausible values from a **background/reference dataset**. "Absent" means "replaced by random realistic values and averaged out," which is why SHAP needs a background set, and why the baseline \(\mathbb{E}[f(x)]\) is the prediction when all features are absent.

!!! note "The deepest honest critique"
    Marginalize using the *marginal* distribution of absent features (interventional SHAP, breaks correlations) or the *conditional* distribution given present ones (conditional SHAP, respects correlations but can leak attribution to correlated-but-unused features)? Naming this interventional-versus-observational tension signals real depth.

**The cost, and why TreeSHAP matters.** Exact computation sums over all \(2^n\) subsets, exponential, intractable past a handful of features. In practice: **KernelSHAP** is model-agnostic (samples coalitions, solves a weighted linear regression whose coefficients are the Shapley values), any model but slow; **TreeSHAP** exploits tree structure for *exact* Shapley values in \(O(TLD^2)\) (trees times leaves times depth squared) instead of \(O(2^n)\), which is why SHAP exploded in tabular ML; **DeepSHAP / GradientSHAP** are backprop-style approximations for neural nets.

**Local to global.** Everything above is local, one example. Global views are aggregations: global importance is the mean of \(|\phi_i|\) across examples; the beeswarm/summary plot shows every example's \(\phi_i\) per feature (magnitude and direction); the dependence plot shows \(\phi_i\) versus the feature's value (nonlinear effects and interactions). Per-feature, per-example is the native granularity; global exists only because you can average a fundamentally local quantity.

## Interview questions

**Q1: How do you evaluate clustering without labels, and how do you choose K?**
You use intrinsic metrics. The silhouette score combines each point's intra-cluster tightness and its distance to the nearest other cluster into a value in minus one to one, and you pick the K that maximizes the average silhouette. Davies-Bouldin averages, per cluster, the worst ratio of combined spread to centroid distance, where lower is better and it is faster but only relative. If you do have ground truth, the adjusted Rand index measures pair-agreement corrected for chance.

**Q2: How do PSI, KS, and KL relate for drift detection?**
All three compare a training distribution to a production one. PSI bins the feature and sums the signed percentage differences times their log ratio, with above 0.2 the standard retraining trigger. The KS statistic is the maximum vertical gap between the two CDFs and comes with a formal p-value. KL divergence is the asymmetric information loss from using one distribution for the other, and PSI is exactly the symmetric sum of the two KL directions, which is why PSI fixes KL's asymmetry.

**Q3: What makes SHAP uniquely principled among explanation methods?**
It is grounded in Shapley values from cooperative game theory, which Shapley proved are the unique attribution satisfying efficiency, symmetry, dummy, and linearity. Efficiency is the key one: the feature contributions sum exactly to the prediction minus the baseline, a complete decomposition with nothing left over, which is what debugging and regulatory compliance require. Competing methods like LIME and permutation importance satisfy no such uniqueness theorem.

**Q4: Why is exact SHAP expensive, and how is it made practical?**
Because a feature's contribution depends on which other features are already present, the exact Shapley value averages the marginal contribution over all subsets, which is a sum over two-to-the-n coalitions, exponential in the feature count. KernelSHAP approximates it for any model by sampling coalitions and solving a weighted linear regression, while TreeSHAP exploits tree structure to compute exact values in polynomial time, trees times leaves times depth squared, which is why SHAP became standard on tabular models. Absent features are marginalized over a background dataset, so SHAP always needs a reference distribution.
