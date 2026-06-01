# K-Means Clustering

K-Means partitions data into K groups by one circular principle: each point belongs to the nearest center, and each center sits at the mean of its members. It alternates until those agree, which is provably the hard-assignment special case of Expectation-Maximization. This page covers the objective, why alternating converges, the link to GMM, K-Means++ seeding, and choosing K.

!!! tip "Rapid Recall"
    K-Means minimizes the within-cluster sum of squares by alternating two steps, assigning each point to its nearest centroid and moving each centroid to its members' mean. Each step provably cannot increase the objective and the objective is bounded below, so it converges, but only to a local minimum, which is why initialization matters. It is EM with hard assignments and equal spherical clusters; the soft version is the Gaussian Mixture Model. K-Means++ seeds centroids spread out by sampling proportional to squared distance, and you pick K with the elbow or the silhouette score.

## §1 What it is

**Unsupervised** — no labels. Given data and a number $K$, it partitions points into $K$ groups by one principle: **each point belongs to the cluster whose center it's closest to, and each center sits at the mean of its members.** Those two conditions are circular, so it alternates until they agree.

What it actually *optimizes* is the *within-cluster sum of squares* (inertia / WCSS):

$$J = \sum_{c=1}^{K}\ \sum_{\mathbf{x}_i\in C_c} \lVert \mathbf{x}_i - \boldsymbol{\mu}_c\rVert^2$$

The total squared distance from every point to its own cluster's center. Tight, compact clusters = low $J$. Everything the algorithm does is descent on this one number.

### Step by step

1. **Initialize** $K$ centroids (randomly, or smartly — see K-Means++).
2. **Assignment step:** assign each point to its nearest centroid. Carves space into $K$ regions.
3. **Update step:** move each centroid to the mean of its assigned points.
4. **Repeat** 2–3 until assignments stop changing.

## §2 Why alternating works, and what EM actually means

Each step is **provably the optimal move for one half of the problem, holding the other half fixed:**

- **Assignment step is optimal given centers.** With centers frozen, the best assignment minimizing $J$ is "send each point to its nearest center." Any other increases its distance term.
- **Update step is optimal given assignments.** The single point minimizing the sum of squared distances to a set of points is the **mean** — a calculus fact: $\frac{d}{d\mu}\sum_i(x_i-\mu)^2=0 \Rightarrow \mu=\text{mean}$.

!!! note "The convergence proof"
    Each step cannot increase $J$, and $J$ is bounded below by zero. A quantity that only decreases and can't go below zero *must converge*. That's why K-Means terminates. The catch: it converges to a *local* minimum, not the global one (finding the true global optimum of $J$ is NP-hard). Which valley you land in depends entirely on initialization.

### EM = Expectation-Maximization (clearing the confusion)

"Expectation" here is **not** the statistical mean. EM names the *two alternating steps* of a general algorithm for models with *hidden/latent variables* — here, *which cluster each point came from*. To estimate centers you'd need assignments; to know assignments you'd need centers. EM breaks the cycle:

- **E-step (Expectation):** given current parameters (centers), figure out the hidden variables — which cluster each point probably belongs to.
- **M-step (Maximization):** given those assignments, re-estimate parameters to **maximize the likelihood** of the data — i.e. the centers that best fit the assigned points.

The "maximization" maximizes the *likelihood* (how well the model explains the data), not an expectation. Each round improves the likelihood until convergence.

!!! note "K-Means is a special case of EM"
    E-step = assignment step, but where general EM gives *soft* assignments (70% cluster A, 30% B — a probability), K-Means uses *hard* assignments (100% to nearest). M-step = update step (move to mean). So K-Means is "EM with hard assignments and equal spherical clusters." The fully soft general version is the [Gaussian Mixture Model (GMM)](../unsupervised/gmm-em.md) — soft membership probabilities, elliptical clusters of varying size.

## §3 Why initialization matters and what K-Means++ does

Because K-Means only finds a local minimum, **bad initial centroids → bad final clusters.** Classic failure: two centroids land inside one true cluster while a genuinely separate cluster gets none. Cheap historical fix: run many random starts, keep the lowest $J$ (`n_init`). Wasteful.

**K-Means++** fixes it at the source by choosing initial centroids that are spread out:

1. Pick the first centroid uniformly at random.
2. For every remaining point, compute $D(\mathbf{x})$ = distance to the *nearest already-chosen* centroid.
3. Pick the next centroid randomly, with probability **proportional to $D(\mathbf{x})^2$.**
4. Repeat until $K$ centroids, then run normal K-Means.

Each new seed is pushed *away* from existing ones, so you start with roughly one centroid per natural cluster. Why probability (not deterministically the farthest point)? Robustness to outliers — a lone far-flung outlier just has higher odds, not a guarantee. Gives faster convergence and provably better expected final $J$. It's sklearn's default (`init='k-means++'`).

## §4 Choosing K: elbow and silhouette

$K$ is an input you must supply — awkward for an unsupervised method. Two standard approaches:

- **Elbow method:** plot $J$ (inertia) vs $K$. It always decreases ($K=n\Rightarrow J=0$). Look for the "elbow" where extra clusters stop buying much reduction. Subjective.
- **Silhouette score:** more principled (below).

!!! note "Silhouette score"
    For one point: $a$ = average distance to other points *in its own cluster* (tightness, want small); $b$ = average distance to all points in the *nearest neighboring cluster* (separation, want large).

    $$s = \frac{b-a}{\max(a,b)}$$

    - $s\approx +1$ → much closer to own cluster than the next → well clustered.
    - $s\approx 0$ → on the boundary → ambiguous.
    - $s<0$ → closer to a neighboring cluster than its own → probably misassigned.

    Average $s$ over all points gives one score; pick the $K$ that maximizes it. Unlike the elbow it's a concrete number to maximize, and because it rewards *separation* (not just tightness) it won't keep improving as you add clusters the way inertia does. Caveat: $O(n^2)$ — compute on a sample for big data.

### Assumptions and failure modes

- **Clusters are spherical and equal-sized** — raw Euclidean distance to a center carves round, equal Voronoi blobs. Elongated/crescent/unequal clusters get mangled. (GMM → ellipses; DBSCAN → arbitrary shapes.)
- **Must pick $K$.**
- **Scale-sensitive** — distance-based; `StandardScaler` first.
- **Outlier-sensitive** — centroids are means, dragged by outliers. (K-Medoids resists this.)
- **Hard assignments** — a boundary point is forced fully into one cluster. (GMM gives soft probabilities.)

## §5 When to use or ignore, and 2026 relevance

**Use** for fast simple clustering on large data, roughly round comparable-size clusters, a reasonable guess for $K$; it's $O(n\cdot K\cdot d\cdot \text{iters})$, near-linear in $n$ — scales far better than most clustering. **Ignore** for non-convex/unequal clusters (DBSCAN, spectral), unknown $K$ (DBSCAN infers it), soft membership (GMM), heavy outliers (K-Medoids/DBSCAN), or high-dim raw data (reduce with PCA first).

**Alive in 2026** because clustering is a need LLMs don't remove and K-Means is the cheapest scalable tool:

- **Clustering embeddings** — embed documents/users/images with a transformer, then K-Means the vectors for topic discovery, user segmentation, dedup, organizing a vector store. The embedding step solves K-Means' dimensionality problem.
- **Vector quantization for ANN** — inside FAISS's IVF index, K-Means partitions vector space into cells so search scans only nearby cells. K-Means is literally a component *inside* the vector DBs powering RAG.
- **Classic segmentation** — customer/market segmentation, image color quantization, anomaly detection (far from any centroid).

!!! note "Interview one-liner"
    "K-Means partitions data into $K$ groups by minimizing within-cluster squared distance, alternating between assigning points to the nearest centroid and moving each centroid to its members' mean. It's the hard-assignment special case of EM — assignment is the E-step, mean-update the M-step — and each step provably can't increase the objective, so it converges, but only to a local optimum, which is why K-Means++ seeds centroids spread out by sampling proportional to squared distance. In 2026 it's the go-to cheap clustering for embeddings and a building block inside FAISS's IVF index."

## Interview questions

**Q1: What does K-Means optimize and why is it guaranteed to converge?**
It minimizes the within-cluster sum of squares, the total squared distance from each point to its own centroid. The assignment step is the optimal choice given fixed centers and the update step is the optimal center given fixed assignments, so each step cannot increase the objective, and since the objective is bounded below by zero, a quantity that only decreases must converge. The catch is that it converges to a local minimum, since the global optimum is NP-hard.

**Q2: How is K-Means related to EM and to GMM?**
It is EM specialized to hard assignments and equal spherical clusters: the assignment step is the E-step inferring which cluster each point came from, and the mean update is the M-step re-estimating parameters. General EM gives soft assignments, probabilities of membership, and the fully soft version with elliptical clusters of varying size is the Gaussian Mixture Model. So K-Means is the hard, spherical corner of the same algorithm.

**Q3: Why does initialization matter and how does K-Means++ help?**
Because K-Means only finds a local minimum, poor initial centroids lead to poor final clusters, the classic failure being two centroids inside one true cluster while a separate cluster gets none. K-Means++ seeds centroids spread out: after a random first center, each next center is chosen with probability proportional to the squared distance to the nearest existing center. Using probability rather than the deterministic farthest point keeps it robust to outliers, and it gives faster convergence and provably better expected inertia.

**Q4: How do you choose K, and what are the silhouette score's advantages over the elbow?**
The elbow method plots inertia versus K and looks for where extra clusters stop buying much reduction, but it is subjective. The silhouette score combines tightness a and separation b into \((b-a)/\max(a,b)\) per point, averaged over all points, giving a concrete number to maximize. Because it rewards separation rather than just tightness it does not keep improving as you add clusters the way inertia does, though it costs \(O(n^2)\) so you compute it on a sample for big data.
