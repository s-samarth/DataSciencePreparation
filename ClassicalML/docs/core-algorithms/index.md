# Core Classical Algorithms

Five algorithms that every classical ML interview circles back to: the support vector machine, Naive Bayes, K-nearest neighbors, K-Means, and PCA. Each gets the full treatment here, intuition, derivations, and the honest answer to where it still earns its place in 2026. The recurring lesson is that none of them died; they became components of the modern stack.

!!! tip "Rapid Recall"
    SVM maximizes the margin and touches data only through dot products, which is the hinge the kernel trick swings on. Naive Bayes is a generative classifier that factorizes the likelihood under conditional independence, trading calibration for a surviving argmax. KNN is lazy, non-parametric, and distance-driven, which is exactly why the curse of dimensionality and scale break it. K-Means is the hard-assignment special case of EM. PCA finds the variance-maximizing directions, the eigenvectors of the covariance matrix and the right singular vectors of the SVD. Three of the five (SVM, KNN, K-Means) require feature scaling.

## What each page covers

- **[Support Vector Machines](svm.md)**: the maximum-margin idea, the dual where only support vectors and dot products appear, the kernel trick (polynomial and RBF), soft margin and C, and why infinite capacity is a tuning burden rather than a free win.
- **[Naive Bayes](naive-bayes.md)**: Bayes' theorem, the law of total probability, the naive conditional-independence assumption, MLE with Laplace smoothing, and the Gaussian, Multinomial, and Bernoulli variants.
- **[KNN & the Curse of Dimensionality](knn.md)**: the lazy instance-based rule, K as the bias-variance dial, KD-trees and why their pruning collapses in high dimensions, and KNN's rebirth as the retrieval engine inside vector databases.
- **[K-Means Clustering](kmeans.md)**: the within-cluster sum of squares, why alternating minimization converges, EM and the link to GMM, K-Means++ seeding, and choosing K with the elbow and silhouette.
- **[Principal Component Analysis](pca.md)**: matrix decomposition, eigendecomposition, the SVD and the bridge to the covariance eigenproblem, the PCA recipe, and why SVD is used for numerical stability.

## Comparison matrix

|  | SVM | Naive Bayes | KNN | K-Means | PCA |
| --- | --- | --- | --- | --- | --- |
| Type | Supervised | Supervised (gen.) | Supervised | Unsupervised (cluster) | Unsupervised (dim. reduce) |
| Training | \(O(n^2\)–\(n^3)\) | \(O(n\cdot d)\) | \(O(1)\) (lazy) | \(O(n K d\,\text{iter})\) | \(O(d^2 n + d^3)\) |
| Scaling needed? | Yes (critical) | No | Yes (critical) | Yes (critical) | Yes (critical) |
| Key hyperparam | \(C\), \(\gamma\), kernel | \(\alpha\) (smoothing) | \(K\), metric | \(K\), init | \(k\) components |
| Curse of dim? | OK (kernel) | Excellent (text) | Fatal | Poor | This is the fix |
| 2026 niche | \(d\gg n\), baseline | cost/latency, baseline | ANN / RAG retrieval | embedding clustering, IVF | embedding compression |

!!! warning "Classic gotcha: K-Means vs KNN"
    Both have \(K\), both use distance, that is where similarity ends. **KNN**: supervised, \(K\) = number of neighbors, no training (lazy), outputs a class/value. **K-Means**: unsupervised, \(K\) = number of clusters, iterative centroid training, outputs a cluster label. Mixing these up is an immediate red flag.

## The thread that ties them together

Classical methods didn't die, they became the components of the modern stack. **SVM** survives where \(d\gg n\) and the margin is strong regularization. **KNN** became approximate nearest-neighbor search, the retrieval engine under every RAG system. **K-Means** clusters embeddings and lives inside FAISS's IVF index. **PCA** still compresses the very transformer embeddings that replaced it elsewhere, and its ordered, truncatable representation is the linear ancestor of Matryoshka embeddings. The eigenvectors and SVD that power PCA, the dot products that power kernels, the distance that powers KNN and K-Means are the same ideas the modern stack is built on. Their survival is almost never about capability; it is about cost, latency, data scarcity, interpretability, and deployment.
