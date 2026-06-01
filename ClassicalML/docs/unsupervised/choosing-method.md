# Choosing a Method

With several clustering algorithms and three dimensionality reducers, the skill is matching the method to the question. This page gives the clustering decision logic, compact one-line summaries of each clusterer, and the PCA versus t-SNE versus UMAP comparison with the standard pipeline that uses them together.

!!! tip "Rapid Recall"
    Use K-Means for fast round equal clusters when you know k, GMM for elliptical overlapping clusters or soft membership, DBSCAN for arbitrary shapes and outliers when k is unknown, and never hierarchical on large data. K-Means and GMM are parametric and need k; DBSCAN and hierarchical are nonparametric structure discovery. For dimensionality reduction, PCA is for modeling and is the linear deterministic first move, while t-SNE and UMAP are for seeing, with UMAP the modern default. The pro move is a sequence: PCA to about 50 dims, then UMAP to 2D, and never read cluster sizes or gaps off a t-SNE or UMAP plot.

## §1 Clustering: which one when

### Decision logic, the questions you actually ask

| Question | Answer → algorithm |
| --- | --- |
| Know how many clusters? + big data | **K-Means** (fast, scales) |
| Don't know how many? | **DBSCAN/HDBSCAN** (finds count from density) or **Hierarchical** (build tree, cut later) |
| Round, equal-size blobs? | **K-Means** |
| Elliptical / overlapping / unequal? | **GMM** |
| Arbitrary, snake-like, non-convex? | **DBSCAN** (or single-linkage hierarchical) |
| Need outlier handling? | **DBSCAN** (K-Means and GMM force every point in, and are distorted by outliers) |
| Need soft / probabilistic membership? | **GMM** (only one native) |
| Millions of points? | **K-Means** \(O(n)\), or DBSCAN with spatial index |
| Large data? | **Never hierarchical** (\(O(n^2)\) memory) |
| Assign future/unseen points? | **K-Means** (nearest centroid) or **GMM** (highest responsibility); DBSCAN and hierarchical refit only |
| Need deterministic / reproducible? | **Hierarchical** (no random init) or **DBSCAN** (deterministic up to border ties) |

### Compressed mental map

- **K-Means** — fast, scalable, hard assignment, assumes spherical equal blobs, needs \(k\). The default baseline.
- **GMM** — K-Means' probabilistic upgrade: elliptical clusters, soft assignment, generative, principled \(k\)-selection via BIC. Costs more compute, still needs \(k\).
- **DBSCAN** — density-based: arbitrary shapes, finds \(k\) itself, flags noise. Struggles with varying density and high dimensions.
- **Hierarchical** — full nested tree, no \(k\) up front, deterministic, interpretable structure. \(O(n^2)\), small data only, no out-of-sample assignment.

!!! note "One-liner for an interviewer"
    K-Means and GMM are parametric, centroid/distribution-based methods that need \(k\) and assume cluster shape; DBSCAN and hierarchical are nonparametric structure-discovery methods that don't. GMM *generalizes* K-Means by adding covariance and soft assignment; DBSCAN generalizes nothing, it's a different paradigm built on **density** rather than distance-to-center.

## §2 Dimensionality reduction: PCA vs t-SNE vs UMAP

!!! note "Framing"
    They're not really competitors, they answer different questions and are often used **together**. **PCA is for *modeling*** (and as a first-stage reducer). **t-SNE and UMAP are for *seeing*.**

|  | PCA | t-SNE | UMAP |
| --- | --- | --- | --- |
| Type | Linear | Nonlinear | Nonlinear |
| Preserves | Global variance / large distances | Local neighborhoods only | Local + *some* global |
| Math basis | Eigendecomposition of covariance | KL divergence of neighbor probs | Cross-entropy of fuzzy neighbor graph |
| Speed | Fastest | Slowest \(O(n\log n)\) | Fast ~\(O(n^{1.14})\) |
| Scales to | Millions+ | ~100k | Millions |
| Deterministic | Yes | No | No (default) |
| Transform new points | Yes | **No** | **Yes** |
| Inter-cluster distance meaningful | Yes | No | Partially |
| Cluster size meaningful | Yes | No | Partially |
| Interpretable axes | Yes | No | No |
| Key hyperparameters | n_components | perplexity | n_neighbors, min_dist |
| Primary use | Preprocessing / compression / denoising | Visualization only | Visualization + sometimes reduction |

### Decision logic

- **PCA** — reach for it first, always. Cheap, linear, deterministic, interpretable, reusable. Preprocessing before a model, compression, decorrelation, and the *first stage* before t-SNE/UMAP. If structure is roughly linear, PCA alone suffices.
- **t-SNE** — when you specifically want the crispest *local* cluster separation for a 2D plot, data isn't huge, no need to transform new points. Increasingly legacy, largely superseded.
- **UMAP** — the modern default for nonlinear reduction: fast, scalable, retains more global layout, can project new data, can feed downstream models. In 2026, "reduce this for a 2D plot" → UMAP unless there's a reason otherwise.

!!! note "The pro move: use both in sequence"
    Run PCA first to knock data from about 1000 to about 50 dims, *then* t-SNE/UMAP to 2D. PCA strips noise and cuts dimensionality cheaply, making the nonlinear step faster and often better. Standard practice (sklearn's t-SNE docs recommend it).

!!! warning "The single most-tested misconception"
    **Never read cluster sizes or inter-cluster gaps off a t-SNE / UMAP plot as quantitatively meaningful.**

!!! note "One-liner for an interview"
    PCA is linear, global, deterministic, reusable, used as preprocessing. t-SNE is nonlinear, local, stochastic, fit-once, visualization-only, with meaningless inter-cluster distances. UMAP is the modern default, near-linear, scales to millions, preserves more global structure, and uniquely can transform unseen points. Standard pipeline: PCA to about 50 dims, then UMAP to 2D.

## Interview questions

**Q1: How do you pick a clustering algorithm?**
Match it to the data and the question: K-Means for round equal-size clusters at scale when you know k; GMM when clusters are elliptical, overlapping, or unequal, or when you need soft membership; DBSCAN for arbitrary shapes and built-in outlier flagging when you do not know k; and hierarchical when you want a full interpretable tree on small data. Never use hierarchical on large data because of its \(O(n^2)\) memory, and remember only K-Means and GMM cleanly assign new points.

**Q2: How are K-Means, GMM, DBSCAN, and hierarchical related conceptually?**
K-Means and GMM are parametric methods built on centroids or distributions that require k and assume a cluster shape, with GMM generalizing K-Means by adding covariance and soft assignment. DBSCAN and hierarchical are nonparametric structure-discovery methods that do not need k. DBSCAN is a different paradigm entirely, built on density rather than distance to a center, so it generalizes nothing in the K-Means family.

**Q3: PCA, t-SNE, and UMAP are not really competitors. Explain.**
They answer different questions. PCA is linear, deterministic, reusable, and interpretable, so it is for modeling, compression, and as a first-stage reducer. t-SNE and UMAP are nonlinear and are for seeing: they preserve local neighborhoods for a 2D picture but have meaningless inter-cluster distances and sizes. UMAP is the modern default because it is near-linear, scales to millions, retains more global structure, and can transform new points, which t-SNE cannot.

**Q4: What is the standard dimensionality-reduction pipeline and its main caveat?**
Run PCA first to cut roughly a thousand dimensions down to about fifty, which strips noise cheaply, then run t-SNE or UMAP to two dimensions for visualization, which is faster and often better after the PCA step. The main caveat is epistemic: never read cluster sizes or the gaps between clusters off a t-SNE or UMAP plot as quantitatively meaningful, since both distort size and inter-cluster distance.
