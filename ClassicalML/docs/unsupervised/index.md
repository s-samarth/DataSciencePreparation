# Unsupervised, Text & Anomaly

This section covers learning without labels: density and hierarchical clustering, Gaussian mixtures and the EM algorithm behind them, the nonlinear embeddings t-SNE and UMAP, turning text into features with TF-IDF and LSA, and detecting anomalies by isolation. The unifying threads are EM (which powers GMM, K-Means, and more) and the SVD (which powers PCA and LSA).

!!! tip "Rapid Recall"
    DBSCAN clusters by density and flags noise but breaks on varying density; hierarchical builds a full dendrogram but costs \(O(n^2)\) memory. GMM is K-Means with covariance and soft assignment, fit by EM, which optimizes a lower bound and climbs to a local optimum. t-SNE and UMAP preserve local neighborhoods for visualization, and you must never read cluster sizes or inter-cluster gaps off them. TF-IDF makes a wide sparse word matrix, reduced by Truncated SVD (LSA, PCA without centering) which preserves sparsity. Isolation Forest scores anomalies by how few random cuts isolate a point.

## What each page covers

- **[Density & Hierarchical Clustering](density-hierarchical.md)**: DBSCAN's core/border/noise points and density-reachability, and agglomerative clustering with the linkage choice and the dendrogram.
- **[Gaussian Mixtures & EM](gmm-em.md)**: the generative mixture, soft responsibilities, and the general EM algorithm as a lower-bound climb past the log-of-a-sum.
- **[t-SNE & UMAP](tsne-umap.md)**: neighbor probabilities and the Student-t crowding fix, UMAP's graph layout, and the landmines of reading these plots.
- **[Text Vectorization](text-vectorization.md)**: TF-IDF as distinctive-word weighting, and Truncated SVD / LSA as the centering-free reducer that survives sparse text.
- **[Anomaly Detection](anomaly-detection.md)**: Isolation Forest, why short path length is the signal, and why small subsamples are better.
- **[Choosing a Method](choosing-method.md)**: the clustering decision logic and the PCA vs t-SNE vs UMAP comparison.

## Master cheat-sheet

| Algorithm | Category | Core idea | Complexity | Key gotcha |
| --- | --- | --- | --- | --- |
| **DBSCAN** | Clustering (density) | Dense regions = clusters; chain through core points; flag noise | \(O(n\log n)\) | One global \(\varepsilon\) fails on varying density → HDBSCAN |
| **Hierarchical** | Clustering (tree) | Merge closest clusters; cut dendrogram at the big gap | \(O(n^2)\)–\(O(n^3)\) | \(O(n^2)\) memory; greedy irreversible merges; no new-point assignment |
| **GMM** | Clustering (generative) | Mixture of Gaussians; soft assignment via EM | \(O(nKd^2\cdot\text{it})\) | Local optimum; covariance singularity; choose \(K\) via BIC |
| **EM** | Optimization meta-algo | Guess latent \(Z\) (E), re-fit \(\theta\) as if true (M), repeat; climbs a lower bound | per-problem | Local optimum only; needs tractable posterior (else variational EM) |
| **t-SNE** | Dim-red (visualization) | Match neighbor-probability distributions; Student-t fixes crowding | \(O(n\log n)\) | Inter-cluster distance and size meaningless; no transform; slow |
| **UMAP** | Dim-red (visualization) | Build weighted k-NN graph; lay out via attract/repel + neg sampling | ~\(O(n^{1.14})\) | Partial global reliability only; can tear manifolds |
| **PCA** | Dim-red (linear) | SVD of centered data → max-variance directions | fast | Linear only; centering kills sparsity (use Truncated SVD) |
| **TF-IDF** | Text → features | TF × IDF → distinctive-word weighting; wide sparse matrix | fast | Lexical only (no synonyms); use cosine + L2-normalize |
| **Truncated SVD / LSA** | Dim-red (sparse) | PCA without centering; top-\(k\) singular vectors = latent topics | fast (randomized) | Uncentered → leading component may just be a "size" direction |
| **Isolation Forest** | Anomaly detection | Random cuts; short path length = easy to isolate = anomaly | ~\(O(n)\) | Misses local anomalies (use LOF); axis-aligned (use Extended iForest) |

## The connective tissue

- **K-Means and GMM**: a GMM with spherical equal covariance is K-Means, and K-Means is hard EM.
- **EM** underlies GMM, K-Means, HMMs (Baum-Welch), topic models, probabilistic PCA, and missing-data imputation.
- **PCA = SVD on centered data**; Truncated SVD = SVD uncentered; LSA = Truncated SVD on a doc-term matrix.
- **The text pipeline**: text → TF-IDF → Truncated SVD/LSA (to dense) → K-Means / linear classifier / UMAP visualization.
- **The visualization pipeline**: high-D → PCA (about 50 dims) → t-SNE / UMAP (2D).
- **High dimensions break distance and density methods** (DBSCAN, KNN, Gaussian density); Isolation Forest sidesteps this by avoiding distance entirely.
- **LSA → word2vec → modern embeddings**: the lineage from sparse-lexical to dense-semantic text representation.
