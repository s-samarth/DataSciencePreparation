# Principal Component Analysis

PCA is one sentence sitting on top of linear algebra: find the orthogonal directions along which the data varies most, and re-express the data using only the top few. This page builds the foundation properly, matrix decomposition to eigendecomposition to SVD, derives the bridge that makes PCA the V of the SVD, and ends with why SVD is used for numerical stability.

!!! tip "Rapid Recall"
    PCA finds the orthogonal directions of maximum variance, which are the eigenvectors of the covariance matrix and equivalently the right singular vectors from the SVD of centered data, ranks them by eigenvalue, and projects onto the top k. The eigen-SVD bridge falls out of multiplying X by its transpose: \(X^\top X=V\Sigma^2 V^\top\), so V are the covariance eigenvectors and \(\sigma_i=\sqrt{\lambda_i}\). It is computed via SVD rather than eigendecomposing the covariance because forming \(X^\top X\) squares the condition number. It is linear and unsupervised, so it fails when the signal is low-variance or the structure is nonlinear.

PCA is one sentence on top of linear algebra: **find the orthogonal directions along which the data varies most, and re-express the data using only the top few.** To understand it properly we build up matrix decomposition → eigendecomposition → SVD first.

## §1 Matrix decomposition: what it even means

Decomposing a matrix means factoring it into simpler matrices each with a clean geometric meaning — like $12=2\times2\times3$. A matrix *is* a linear transformation: multiplying a vector rotates, stretches, and reflects it. Decomposition splits that complicated transformation into "rotate, then stretch, then rotate" — simple, understandable pieces.

## §2 Eigenvalue decomposition: the foundation

Multiply most vectors by a matrix $A$ and they get rotated off their direction. For *special* directions, multiplying by $A$ only **stretches or shrinks** the vector — it stays pointing the same way. Those directions are *eigenvectors*; the stretch factor is the *eigenvalue*:

$$A\mathbf{v} = \lambda\mathbf{v}$$

$\mathbf{v}$ is a direction $A$ doesn't rotate; $\lambda$ is how much $A$ scales along it. For **symmetric** matrices (a covariance matrix is symmetric), you can decompose $A$ fully:

$$A = Q\,\Lambda\,Q^{\mathsf T}$$

$Q$ = matrix of eigenvectors (orthogonal for symmetric $A$); $\Lambda$ = diagonal of eigenvalues. Geometrically: $Q^{\mathsf T}$ rotates into the eigenvector coordinate system, $\Lambda$ scales along each axis, $Q$ rotates back. The eigenvectors *are* the natural axes; the eigenvalues say how important each is. **Limitation:** only works on square (nicely, symmetric) matrices. A data matrix $X$ ($n$ samples × $d$ features) is rectangular — which is why SVD exists.

## §3 SVD and the bridge to eigendecomposition

SVD factors *any* matrix, rectangular included:

$$X = U\,\Sigma\,V^{\mathsf T}$$

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="SVD as rotate, stretch, rotate" role="img" viewBox="0 0 680 230" width="100%" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="ar" markerHeight="7" markerWidth="7" orient="auto-start-reverse" refX="8" refY="5" viewBox="0 0 10 10"><path d="M2 1L8 5L2 9" fill="none" stroke="#6b6557" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"></path></marker></defs>
<text fill="#26231d" font-family="Fraunces,serif" font-size="15" font-weight="600" text-anchor="middle" x="80" y="36">Input</text>
<text fill="#6b6557" font-family="Newsreader,serif" font-size="13" font-style="italic" text-anchor="middle" x="80" y="54">unit circle</text>
<circle cx="80" cy="130" fill="none" r="42" stroke="#534AB7" stroke-width="2"></circle>
<line stroke="#534AB7" stroke-width="2.5" x1="80" x2="122" y1="130" y2="130"></line>
<line stroke="#534AB7" stroke-width="2.5" x1="80" x2="80" y1="130" y2="88"></line>
<text fill="#9a3b1f" font-family="JetBrains Mono,monospace" font-size="15" text-anchor="middle" x="210" y="122">Vᵀ</text>
<text fill="#6b6557" font-family="Newsreader,serif" font-size="12" font-style="italic" text-anchor="middle" x="210" y="140">rotate</text>
<line marker-end="url(#ar)" stroke="#6b6557" stroke-width="1.5" x1="148" x2="262" y1="130" y2="130"></line>
<text fill="#26231d" font-family="Fraunces,serif" font-size="15" font-weight="600" text-anchor="middle" x="340" y="36">Rotated</text>
<text fill="#6b6557" font-family="Newsreader,serif" font-size="13" font-style="italic" text-anchor="middle" x="340" y="54">still a circle</text>
<circle cx="340" cy="130" fill="none" r="42" stroke="#1d7a63" stroke-width="2"></circle>
<line stroke="#1d7a63" stroke-width="2.5" x1="340" x2="370" y1="130" y2="160"></line>
<line stroke="#1d7a63" stroke-width="2.5" x1="340" x2="310" y1="130" y2="100"></line>
<text fill="#9a3b1f" font-family="JetBrains Mono,monospace" font-size="15" text-anchor="middle" x="470" y="122">Σ</text>
<text fill="#6b6557" font-family="Newsreader,serif" font-size="12" font-style="italic" text-anchor="middle" x="470" y="140">stretch</text>
<line marker-end="url(#ar)" stroke="#6b6557" stroke-width="1.5" x1="408" x2="512" y1="130" y2="130"></line>
<text fill="#26231d" font-family="Fraunces,serif" font-size="15" font-weight="600" text-anchor="middle" x="592" y="36">Stretched + U</text>
<text fill="#6b6557" font-family="Newsreader,serif" font-size="13" font-style="italic" text-anchor="middle" x="592" y="54">ellipse, rotated</text>
<ellipse cx="592" cy="130" fill="none" rx="50" ry="26" stroke="#b07d24" stroke-width="2" transform="rotate(-25 592 130)"></ellipse>
<line stroke="#b07d24" stroke-width="2.5" x1="592" x2="637" y1="130" y2="109"></line>
<line stroke="#b07d24" stroke-width="2.5" x1="592" x2="580" y1="130" y2="105"></line>
</svg>
<figcaption>Every matrix does three things to space: rotate by V transpose, stretch by Sigma, rotate by U. The long axis of the output ellipse is the largest singular value.</figcaption>
</figure>

Every matrix does exactly three things to space: rotate ($V^{\mathsf T}$), stretch ($\Sigma$), rotate ($U$). $U$ and $V$ are rotations; $\Sigma$ is the only step that changes lengths. The long axis of the output ellipse is $\sigma_1$, the biggest singular value.

- **$V$** (columns = *right singular vectors*): orthogonal directions in the *input/feature* space — the natural axes of your features.
- **$\Sigma$** (diagonal of *singular values* $\sigma_i$, sorted largest first): how much the data stretches along each direction.
- **$U$** (columns = *left singular vectors*): orthogonal directions in the *output/sample* space.

A large $\sigma$ means the data spreads a lot along that direction (informative); a tiny $\sigma$ means it barely varies there (low information). SVD exists for *any* matrix because every matrix has a "rotate-stretch-rotate" form, while eigendecomposition needs the simpler "rotate-stretch-rotate-back-the-same-way" form that only special matrices have.

### Deriving the bridge: multiply X by its transpose

To find $U,\Sigma,V$, multiply $X$ by its own transpose, which collapses one rotation and leaves an eigenvalue problem. Assume $X=U\Sigma V^{\mathsf T}$ and compute:

$$X^{\mathsf T}X = (U\Sigma V^{\mathsf T})^{\mathsf T}(U\Sigma V^{\mathsf T}) = V\Sigma^{\mathsf T}U^{\mathsf T}U\Sigma V^{\mathsf T}$$

The middle is $U^{\mathsf T}U$. Because $U$ is a rotation (orthogonal), $U^{\mathsf T}U=I$ — a rotation followed by its own inverse is nothing. **$U$ vanishes.** That's the point of multiplying by the transpose: it annihilates one rotation. Left with:

$$X^{\mathsf T}X = V\,\Sigma^2\,V^{\mathsf T}$$

Compare to $A=Q\Lambda Q^{\mathsf T}$ — the same form. Therefore:

!!! note "The eigen-SVD bridge"
    - $V$ (right singular vectors) **= eigenvectors of $X^{\mathsf T}X$** (sits in the $Q$ slot).
    - $\Sigma^2$ (singular values squared) **= eigenvalues of $X^{\mathsf T}X$** (sits in the $\Lambda$ slot), so $\sigma_i=\sqrt{\lambda_i}$.

    It didn't have to be assumed — it *dropped out* of multiplying $X$ by its transpose. And $X^{\mathsf T}X$ is (proportional to) the covariance of centered data, which is why PCA — which wants covariance eigenvectors — is secretly just the $V$ of the SVD.

### Where U went: the left singular vectors

Do the mirror move, $XX^{\mathsf T}$. By identical algebra $V^{\mathsf T}V=I$ kills $V$:

$$XX^{\mathsf T} = U\,\Sigma^2\,U^{\mathsf T}$$

- $V$ = eigenvectors of $X^{\mathsf T}X$ — the $d\times d$ *feature-side* matrix ("how features co-vary").
- $U$ = eigenvectors of $XX^{\mathsf T}$ — the $n\times n$ *sample-side* matrix ("how samples co-vary").
- Both share the *same* non-zero eigenvalues $\sigma_i^2$.

Or, without a second eigendecomposition: rearranging $XV=U\Sigma$ gives $\mathbf{u}_i = X\mathbf{v}_i/\sigma_i$ — push each right singular vector through $X$ and normalize. That's how $U$ is computed in practice instead of eigendecomposing the huge $n\times n$ matrix.

!!! note "Why PCA uses SVD instead of eigendecomposing the covariance"
    PCA wants $V$. Route 1: form $X^{\mathsf T}X$ then eigendecompose — but forming it *squares* the singular values ($\sigma\to\sigma^2$). If the largest singular value is $10^6\times$ the smallest, squaring makes the ratio $10^{12}$ and tiny directions get swallowed by floating-point error before you start. Route 2: SVD of $X$ directly gets $V$ and $\sigma$ *without ever forming $X^{\mathsf T}X$*, so the values are never squared and conditioning stays as good as the data allows. Same answer, far more stable — which is why sklearn's PCA runs SVD on the centered data matrix.

## §4 PCA: the steps

PCA's directions are precisely the eigenvectors of the covariance matrix (= right singular vectors of centered data), ranked by eigenvalue (= variance captured). **Why "directions of maximum variance"?** Variance = information: a direction where data spreads a lot distinguishes samples (useful); a near-constant direction tells you almost nothing (drop it).

1. **Center** the data — subtract each feature's mean (PCA is about variance *around the mean*; skip it and the first PC just points at the mean). Often also **standardize** to unit variance — mandatory on different scales, else the large-scale feature dominates by units alone.
2. **Covariance matrix** $C=\frac{1}{n}X^{\mathsf T}X$ (centered $X$) — how features co-vary.
3. **Eigendecompose** $C$ (or SVD the centered $X$). Eigenvectors = PC directions; eigenvalues = variance along each.
4. **Sort** by eigenvalue, descending.
5. **Keep top $k$** — the $k$ directions capturing the most variance.
6. **Project:** $X_{\text{reduced}} = X V_k$. Each point is now $k$ numbers instead of $d$.

Each PC is a linear combination of original features, orthogonal to the others, ordered by variance explained. Pick $k$ by the **cumulative explained-variance ratio** (eigenvalue $i$ / sum of all eigenvalues), keeping enough PCs to reach ~95%. A **scree plot** (eigenvalue vs PC index) shows the elbow where extra PCs stop earning their keep.

## §5 The Matryoshka instinct: where it's right, where it breaks

The shared property is **nested truncation**: with PCA the top $k$ components form a valid representation on their own, and they're a *prefix* of the full set — components are ranked by importance, so "take the top $n$" always gives the most information-dense $n$-dim summary. Drop the tail, keep meaning. That ordered-by-importance, truncate-the-prefix structure is exactly the property *Matryoshka Representation Learning (MRL)* embeddings are engineered to have — where the first 64, 256, 768 dims each work standalone. The instinct is correct on the structural level.

!!! warning "Where it breaks: flag this yourself"
    1. **PCA is linear and unsupervised, found by eigen/SVD. MRL is learned by a neural net** with a loss that explicitly forces every prefix to be useful (trains on multiple truncation lengths at once). PCA's ordering *emerges* from variance; MRL's is *imposed* by the loss.
    2. **PCA orders by variance; MRL by task usefulness.** PCA's top components are highest-variance — not necessarily the most discriminative for a downstream task. MRL prefixes are optimized to be predictive.
    3. **PCA components are orthogonal linear axes; MRL dimensions are entangled learned features** with no orthogonality guarantee.

    Same *interface* (truncatable, importance-ordered), completely different *machinery*. "PCA is the classical, linear, variance-based ancestor of the Matryoshka idea" is fair and sharp; calling them the same thing is wrong.

## §6 When to use or ignore, and 2026 relevance

**Use** for: preprocessing before a model (compress correlated features, speed training, reduce overfitting); multicollinearity (produces uncorrelated components); visualization (2–3 PCs, though UMAP/t-SNE look better); noise reduction/compression (low-variance components are often noise). **Ignore** when: you need interpretability (PCs are blends of all features); relationships are non-linear (Kernel PCA, UMAP, autoencoders); the task-relevant signal is low-variance (PCA is unsupervised and blind to labels — LDA, which maximizes class separation, can beat it); or features are categorical (use MCA).

**Alive in 2026** — cheap, deterministic, still the default first move for linear dimensionality reduction:

- **Compressing embeddings** — squash 768–3072-dim transformer embeddings to a few hundred dims to cut vector-DB storage and speed similarity search, with minimal recall loss. (MRL is the *learned* alternative to this exact problem.)
- **Preprocessing / EDA** — the standard quick look at high-dim tabular data and redundancy removal before classical models.
- **Inside other methods** — denoising, compression, anomaly detection (large reconstruction error = anomaly), PCA → K-Means pipelines.

Displaced for pure visualization (UMAP/t-SNE) and non-linear structure (autoencoders), but for fast linear compression and as a preprocessing default it hasn't moved.

!!! note "Interview one-liner"
    "PCA finds the orthogonal directions of maximum variance — the eigenvectors of the covariance matrix, equivalently the right singular vectors from the SVD of centered data — ranks them by eigenvalue (variance explained), and projects onto the top $k$ to reduce dimensions while keeping the most information. It's computed via SVD rather than eigendecomposing the covariance for numerical stability. It's linear, unsupervised, and assumes high variance means high information — so it fails when the signal is low-variance or the structure is non-linear, where LDA, Kernel PCA, or autoencoders take over."

## Interview questions

**Q1: What does PCA compute, and what is the link between covariance eigenvectors and the SVD?**
PCA finds the orthogonal directions of maximum variance, the eigenvectors of the covariance matrix ranked by eigenvalue. Multiplying X by its transpose gives \(X^\top X=V\Sigma^2 V^\top\), the same form as a symmetric eigendecomposition, so the right singular vectors V are exactly the covariance eigenvectors and the eigenvalues are the singular values squared. Since \(X^\top X\) is proportional to the covariance of centered data, PCA is secretly the V of the SVD.

**Q2: Why use SVD instead of eigendecomposing the covariance matrix?**
Forming \(X^\top X\) squares the singular values, so a condition number of \(10^6\) becomes \(10^{12}\) and tiny but real directions vanish into floating-point error. Running SVD directly on the centered data matrix recovers V and the singular values without ever forming \(X^\top X\), so the values are never squared and conditioning stays as good as the data allows. It is the same answer, far more numerically stable, which is why sklearn does it.

**Q3: Why must you center, and often standardize, before PCA?**
PCA measures variance around the mean, so without centering the first component just points at the mean offset rather than the direction of spread. Standardizing to unit variance is needed when features are on different scales, otherwise a large-unit feature dominates the variance purely by its units rather than its information. You then keep enough components to reach roughly 95% cumulative explained variance.

**Q4: When does PCA fail, and what replaces it?**
It fails when the task-relevant signal is low-variance, because PCA is unsupervised and blind to labels, so LDA which maximizes class separation can beat it; when relationships are nonlinear, where Kernel PCA, UMAP, or autoencoders take over; and when you need interpretability, since components are blends of all features. For categorical features you use MCA instead. Its surviving strength is fast, deterministic, linear compression and preprocessing.
