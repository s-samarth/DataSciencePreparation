# t-SNE & UMAP

Both squash high-dimensional data into 2D or 3D by keeping neighbors together, sacrificing global geometry for crisp local structure. t-SNE matches neighbor-probability distributions; UMAP lays out a neighbor graph and, by 2026, is the default. This page covers both and, crucially, what you must never read off these plots.

!!! tip "Rapid Recall"
    t-SNE converts high-D distances to neighbor probabilities and lays points out in 2D to match them by minimizing KL divergence, with a Student-t low-D kernel that fixes the crowding problem and perplexity setting the effective neighbor count. UMAP builds a weighted k-NN graph and lays it out with attractive and repulsive forces using a cross-entropy that penalizes both near-to-far and far-to-near, so it keeps more global structure, runs near-linearly, and can transform new points. For both, inter-cluster distances and cluster sizes are not quantitatively meaningful, and they are non-deterministic.

## §1 t-SNE

t-distributed Stochastic Neighbor Embedding.

PCA asks "what directions capture the most *variance*?", global, linear, preserves large distances. t-SNE asks "**who are each point's neighbors, and can I lay points out in 2D so neighbors stay neighbors?**" It discards all concern for global geometry and obsesses over **local neighborhood structure**. That's why t-SNE plots show gorgeous, well-separated islands. It is almost exclusively a **visualization** tool (2D/3D for a human eye).

### How it works: distances become probabilities

1. **High-D neighborhoods → probabilities.** For point \(i\): "if I pick a neighbor with probability proportional to closeness, how likely is \(j\)?" Uses a Gaussian; far structure vanishes into negligible probabilities.
2. **Same in low-D.** Place points in 2D, compute neighbor-probabilities \(q_{ij}\) there.
3. **Match the two distributions** by minimizing KL divergence via gradient descent until \(Q\) looks like \(P\).

High-D conditional (asymmetric), then symmetrized:

$$p_{j\mid i}=\frac{\exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\lVert x_i-x_k\rVert^2/2\sigma_i^2)},\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n}$$

Low-D affinities use a Student-t (1 degree of freedom = Cauchy) kernel:

$$q_{ij}=\frac{\left(1+\lVert y_i-y_j\rVert^2\right)^{-1}}{\sum_{k\neq l}\left(1+\lVert y_k-y_l\rVert^2\right)^{-1}}$$

Cost = KL divergence (asymmetric → penalizes putting near points far, not vice versa):

$$C=\mathrm{KL}(P\|Q)=\sum_{i\neq j}p_{ij}\,\log\frac{p_{ij}}{q_{ij}}$$

### The two ideas that make it work

!!! note "1. The Student-t tail (the \"t\") fixes crowding"
    High-D space has vastly more room than 2D. Squashing down, moderately-distant points get crushed together near the center, the **crowding problem**. The heavy-tailed Student-t in low-D lets moderately-far high-D points sit *much* farther apart in 2D without a big probability penalty, giving clusters room to separate cleanly instead of collapsing into one blob. This is the single cleverest piece.

!!! note "2. Perplexity"
    Each point's Gaussian width \(\sigma_i\) is tuned so every point has roughly the same *effective number of neighbors*, that target is **perplexity** (typically 5 to 50). The most important knob; results genuinely change with it.

### The landmines

!!! warning "Most-tested misconceptions"
    - **Inter-cluster distances are meaningless.** Gaps between clusters tell you nothing about dissimilarity.
    - **Cluster sizes are meaningless.** t-SNE expands dense clusters and contracts sparse ones.
    - **Non-deterministic** (random init plus SGD), set a seed.
    - **No `transform` for new points**, it embeds the specific points given; can't project new data.
    - **Can manufacture structure from noise** with the wrong perplexity, vary it and check stability.
    - **Slow**: \(O(n^2)\) naive, \(O(n\log n)\) with Barnes-Hut; struggles past about 100k points.

## §2 UMAP

Uniform Manifold Approximation and Projection.

UMAP solves the same problem as t-SNE, squash high-D to 2D/3D keeping neighbors together, but it's faster, preserves more of the global picture, and can project new points. By 2026 it's the **default**; t-SNE is the older incumbent it displaced. Intuition shift: t-SNE thinks in *neighbor probabilities*; UMAP builds a **graph**, connect each point to its nearest neighbors, then redraw that graph in 2D so connected points stay close.

### The machinery

1. **Build a weighted k-NN graph in high-D.** Connect each point to its \(k\) nearest neighbors; edge weights encode neighbor strength. Per-point local distance scaling (like perplexity) treats dense/sparse regions fairly; each point connects to at least its nearest neighbor (nothing stranded). Graph symmetrized by combining directional weights. (Formally a "fuzzy simplicial set"; operationally a weighted k-NN graph.)
2. **Lay the graph out in low-D.** Smart **spectral initialization** (not random, more stable, more global structure). Optimize with attractive forces (connected points) plus repulsive forces (unconnected), via SGD with **negative sampling** (push apart only a small random sample of non-neighbors each step, borrowed from word2vec, a huge speedup).

Cost = fuzzy-set cross-entropy between high-D graph and low-D graph:

$$C=\sum_{(i,j)}\Big[\,v_{ij}\log\frac{v_{ij}}{w_{ij}}+(1-v_{ij})\log\frac{1-v_{ij}}{1-w_{ij}}\,\Big]$$

\(v_{ij}\) = high-D edge weight, \(w_{ij}\) = low-D edge weight. Unlike t-SNE's KL (penalizes only near to far), cross-entropy penalizes **both** near to far *and* far to near, so UMAP retains more global structure.

### The two key hyperparameters

- **`n_neighbors`** — the perplexity analog. Low (about 5) → fine local detail, fragmented. High (about 50 to 100) → broader, more global structure.
- **`min_dist`** — how tightly points may pack in low-D. Low (about 0 to 0.1) → tight clumps, crisp separation. High (about 0.5+) → spread out, overall topology. No t-SNE equivalent.

### Why UMAP beats t-SNE practically

- **Speed** — approximate NN (NN-Descent) plus negative sampling → about \(O(n^{1.14})\) empirically; handles millions where t-SNE chokes around 100k.
- **Global structure** — spectral init plus cross-entropy preserve more inter-cluster meaning.
- **Out-of-sample `transform`** — learns a reusable mapping; can project new points. t-SNE cannot. Usable in real pipelines.
- **General reduction** — sometimes used to reduce to 10 to 50 dims feeding a downstream model, not only 2D.

!!! warning "Caveats still apply"
    Inter-cluster distances are only *partially* reliable (more than t-SNE, but it's not PCA). It can create false structure / tear manifolds with wrong `n_neighbors`. Non-deterministic by default (seed it). Same epistemic humility as t-SNE.

## Interview questions

**Q1: What problem does the Student-t kernel solve in t-SNE?**
The crowding problem: high-dimensional space has far more room than 2D, so when you squash data down, moderately distant points get crushed together near the center and clusters merge into one blob. The heavy-tailed Student-t kernel in the low-dimensional space lets moderately far high-D points sit much farther apart in 2D without a large probability penalty, giving clusters room to separate cleanly. It is the single cleverest piece of the algorithm.

**Q2: What must you never read off a t-SNE or UMAP plot?**
Inter-cluster distances and cluster sizes as quantitative facts. The gaps between clusters do not encode dissimilarity, and t-SNE expands dense clusters while contracting sparse ones, so size is an artifact. Both are also non-deterministic and can manufacture apparent structure from noise under the wrong perplexity or n_neighbors, so you vary the setting and check stability.

**Q3: Why does UMAP preserve more global structure and run faster than t-SNE?**
Its fuzzy-set cross-entropy penalizes both placing near points far and far points near, whereas t-SNE's KL only penalizes near-to-far, and UMAP starts from a spectral initialization rather than random, both of which retain more inter-cluster layout. It is fast because it uses approximate nearest neighbors and negative sampling, pushing apart only a small random sample of non-neighbors per step, giving roughly \(O(n^{1.14})\) scaling to millions of points.

**Q4: What can UMAP do that t-SNE cannot, and why does it matter?**
UMAP learns a reusable mapping, so it can transform new, unseen points into the existing embedding, while t-SNE only embeds the specific points it was given. This makes UMAP usable in real pipelines, where you fit on training data and project new data later, and it can also reduce to 10 to 50 dimensions to feed a downstream model rather than only producing a 2D picture.
