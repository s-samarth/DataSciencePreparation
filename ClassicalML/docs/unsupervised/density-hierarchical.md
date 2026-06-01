# Density & Hierarchical Clustering

Two clustering paradigms that need no preset number of clusters. DBSCAN asks "is this point in a crowded neighborhood?" and traces arbitrary shapes while flagging noise. Hierarchical clustering builds a full tree of nested groupings you can cut anywhere. This page covers both, including the linkage choice that defines a hierarchical clustering and the failure modes of each.

!!! tip "Rapid Recall"
    DBSCAN labels points as core, border, or noise using a radius eps and a count minPts, then flood-fills clusters through chained core points, which lets it trace non-convex shapes and isolate outliers without a preset k. Its fatal flaw is that one global eps cannot fit both dense and sparse clusters, which is why HDBSCAN exists. Hierarchical clustering merges the two closest clusters repeatedly into a dendrogram, with the linkage rule (single, complete, average, Ward) being the key choice, and you cut the tree at the largest vertical gap. It does not scale: \(O(n^2)\) memory and greedy irreversible merges.

## §1 DBSCAN

Density-Based Spatial Clustering of Applications with Noise.

K-Means asks "which centroid is closest?" DBSCAN asks a fundamentally different question: **"is this point in a crowded neighborhood?"** A cluster is a dense region of points separated from other clusters by sparse regions. A cluster is a crowd; the empty space between crowds is the boundary. This single shift lets DBSCAN find arbitrarily-shaped clusters and label outliers as *noise* instead of forcing every point into a group.

### The two knobs

- **eps (\(\varepsilon\))** — a radius. How far out you look to define a point's neighborhood.
- **minPts** — a count. How many points must lie within \(\varepsilon\) (including the point itself) for the neighborhood to count as "dense."

No `k`. You don't specify the number of clusters, DBSCAN discovers it from the density structure.

### Three kinds of points

- **Core point** — has \(\geq\) minPts neighbors within \(\varepsilon\). Sits in the interior of a dense region.
- **Border point** — fewer than minPts neighbors, but lies within \(\varepsilon\) of a core point. On the edge of a crowd.
- **Noise point** — neither core nor border. Out in the sparse wilderness. This is the outlier.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg">
<!-- dense cluster -->
<circle cx="150" cy="150" fill="rgba(95,182,201,.08)" r="70" stroke="#2a2f3a"></circle>
<!-- core point eps circle -->
<circle cx="150" cy="150" fill="none" opacity="0.7" r="38" stroke="#e0b341" stroke-dasharray="4 4"></circle>
<!-- core points -->
<g fill="#5fb6c9">
<circle cx="150" cy="150" r="7"></circle>
<circle cx="130" cy="125" r="6"></circle><circle cx="170" cy="130" r="6"></circle>
<circle cx="135" cy="172" r="6"></circle><circle cx="175" cy="168" r="6"></circle>
<circle cx="118" cy="150" r="6"></circle><circle cx="160" cy="185" r="6"></circle>
</g>
<!-- border points -->
<g fill="#e0b341">
<circle cx="100" cy="185" r="6"></circle><circle cx="205" cy="120" r="6"></circle>
</g>
<!-- noise -->
<g fill="#eb6f6f">
<circle cx="470" cy="80" r="6"></circle><circle cx="540" cy="200" r="6"></circle><circle cx="430" cy="230" r="6"></circle>
</g>
<text dx="10" dy="-12" fill="#fff" font-family="sans-serif" font-size="11" x="150" y="150">core</text>
<text dx="8" dy="4" fill="#e0b341" font-family="sans-serif" font-size="11" x="205" y="120">border</text>
<text dx="10" dy="4" fill="#eb6f6f" font-family="sans-serif" font-size="11" x="470" y="80">noise</text>
<text fill="#9aa3b2" font-family="sans-serif" font-size="11" text-anchor="middle" x="150" y="240">dashed = ε-radius around a core point (≥ minPts neighbors inside)</text>
</svg>
<figcaption>Core points (blue) chain together into a cluster; border points (gold) attach but do not expand; noise (red) is isolated.</figcaption>
</figure>

### How it runs

1. Pick an unvisited point; count neighbors within \(\varepsilon\).
2. If it's a **core point**, start a new cluster and **expand**: pull in all neighbors. For each neighbor that is *itself* core, pull in *its* neighbors too, flood-fill outward through chained core points until you hit the boundary.
3. If not core, tentatively mark noise (it may later be reclaimed as a border point).
4. Repeat until all points are visited.

!!! note "Why it can trace snakes"
    Density-reachability is **transitive through core points**: core A reaches core B reaches core C, so all join one cluster even if A and C are far apart. This chaining is exactly what lets DBSCAN trace long, curvy, non-convex clusters. K-Means cannot, it's anchored to a single center and assumes spherical, equal-size blobs.

### Choosing the knobs

- **minPts**: rule of thumb \(\geq D+1\) (\(D\) = dimensions), commonly \(2D\). Higher → more conservative, more noise.
- **eps**: the **k-distance plot**. For each point compute the distance to its \(k\)-th nearest neighbor (\(k\) = minPts), sort descending, plot. The *elbow / knee*, where the curve bends sharply upward, is a principled \(\varepsilon\): it marks the transition from "inside a cluster" to "in the sparse gap."

### Complexity

With a spatial index (KD-tree / ball-tree): **\(O(n\log n)\)** average. Naive pairwise: **\(O(n^2)\)**. In high dimensions the index stops helping and you slide back toward \(O(n^2)\).

### Where to use it

- **Good for:** non-globular clusters, unknown number of clusters, built-in outlier flagging, spatial / GPS hotspots.
- **Struggles with:** varying density, high dimensions.

!!! warning "Biggest failure mode"
    A single global \(\varepsilon\) **cannot** be simultaneously tight enough for a dense cluster and loose enough for a sparse one, so DBSCAN breaks on **varying-density** data. This is precisely why **HDBSCAN** exists (it varies the density threshold) and is the 2026 practical default. In high dimensions, *distance concentration* makes all pairwise distances converge, so "dense vs sparse" loses meaning, reduce dimensions first (PCA/UMAP).

## §2 Hierarchical clustering

Instead of one flat partition, hierarchical clustering builds a **tree of nested groupings**, a family tree for your data. At the bottom, every point is its own singleton; at the top, everything is one cluster; in between is every intermediate grouping stacked by similarity. You build the whole hierarchy *once*, then cut it wherever you want however many clusters you need. No \(k\) up front, and you see *structure* (which groups are sub-groups of which), not just flat labels.

### Two directions

- **Agglomerative (bottom-up)** — used about 95% of the time. Start with \(n\) singletons; repeatedly merge the two closest clusters; stop at one cluster.
- **Divisive (top-down)** — start with one cluster, recursively split the most heterogeneous. Rarely used: splitting considers exponentially many partitions and is computationally brutal.

### The crux: linkage, distance between clusters

For points, distance is obvious. Once points become clusters, you need a rule for the distance *between two clusters*, the **linkage criterion**, the single most important choice.

| Linkage | Distance defined as | Behavior |
| --- | --- | --- |
| **Single** | Closest pair across clusters (nearest neighbors) | Long, stringy, "chaining" clusters; traces non-globular shapes like DBSCAN but fragile, one bridge of noise merges two real clusters. Sensitive to noise. |
| **Complete** | Farthest pair across clusters | Compact, roughly spherical, similar-diameter clusters; over-eager to split; sensitive to outliers (one far point dominates the max). |
| **Average** | Average over all cross-cluster pairs | Compromise between single and complete; more robust. |
| **Ward** | Merge that yields smallest increase in total within-cluster variance | The usual default. Balanced, compact clusters; most K-Means-like in spirit. Requires Euclidean distance. |

Ward minimizes the increase in within-cluster sum of squares (SSE) at each merge:

$$\Delta\text{SSE}(A,B)=\frac{|A|\,|B|}{|A|+|B|}\;\lVert\mu_A-\mu_B\rVert^2$$

where \(|A|,|B|\) are cluster sizes and \(\mu_A,\mu_B\) their centroids. The merge with the smallest \(\Delta\text{SSE}\) wins.

### The dendrogram, the actual output

The output isn't labels, it's a **dendrogram**: leaves are points, each internal node is a merge, and the **height** of a merge = the dissimilarity at which the two clusters joined. To get flat clusters, **cut horizontally**: each vertical line your cut crosses becomes one cluster. Cut low → many tiny clusters; cut high → few big ones.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 560 260" xmlns="http://www.w3.org/2000/svg">
<!-- axis -->
<line stroke="#2a2f3a" x1="60" x2="60" y1="20" y2="220"></line>
<text fill="#9aa3b2" font-family="sans-serif" font-size="11" transform="rotate(-90 28,120)" x="20" y="120">merge height</text>
<!-- leaves at y=220 -->
<g fill="none" stroke="#5fb6c9" stroke-width="2">
<!-- A,B merge low -->
<path d="M100,220 L100,180 L150,180 L150,220"></path>
<!-- C,D merge low -->
<path d="M210,220 L210,165 L260,165 L260,220"></path>
<!-- (AB),(CD) merge mid -->
<path d="M125,180 L125,120 L235,120 L235,165"></path>
<!-- E,F merge low -->
<path d="M360,220 L360,175 L410,175 L410,220"></path>
<!-- top merge high -->
<path d="M180,120 L180,55 L385,55 L385,175"></path>
</g>
<!-- cut line -->
<line stroke="#eb6f6f" stroke-dasharray="5 4" x1="60" x2="520" y1="88" y2="88"></line>
<text fill="#eb6f6f" font-family="sans-serif" font-size="11" x="470" y="82">cut → 3 clusters</text>
<!-- labels -->
<g fill="#e7e9ee" font-family="sans-serif" font-size="12" text-anchor="middle">
<text x="100" y="238">A</text><text x="150" y="238">B</text>
<text x="210" y="238">C</text><text x="260" y="238">D</text>
<text x="360" y="238">E</text><text x="410" y="238">F</text>
</g>
<text fill="#9aa3b2" font-family="sans-serif" font-size="10.5" text-anchor="middle" x="290" y="252">cut in the largest vertical gap → respects natural separation</text>
</svg>
<figcaption>Cutting the dendrogram at the largest vertical gap (the longest stretch with no merges) yields clusters that were genuinely distinct.</figcaption>
</figure>

### How it runs

1. Compute the full pairwise distance matrix (\(O(n^2)\) space).
2. Find the two closest clusters; merge.
3. Update the distance matrix via the linkage rule, the **Lance-Williams formula** updates merged-cluster distances algebraically from old distances, without recomputing from raw points.
4. Repeat until one cluster remains, recording each merge height.

### Complexity, the real limitation

Time: **\(O(n^3)\)** naive, **\(O(n^2\log n)\)** with a priority queue, **\(O(n^2)\)** for single linkage (SLINK). Space: **\(O(n^2)\)** for the distance matrix.

!!! warning "Dealbreaker"
    That \(O(n^2)\) memory kills you on large data, at about 50k points you're already at about 2.5 billion matrix entries. Hierarchical clustering does **not** scale. Also: merges are **greedy and irreversible**, an early bad merge propagates up the whole tree. And there is **no model to apply to new points**; you'd have to rebuild.

### Where to use it

- **Good for:** unknown k with exploration of multiple granularities, cases where the hierarchy itself is meaningful (taxonomy, phylogenetics, gene expression), small-to-medium data, deterministic results (no random init).
- **Struggles with:** large data, the need for out-of-sample assignment.

## Interview questions

**Q1: What are the three point types in DBSCAN and how does a cluster form?**
A core point has at least minPts neighbors within radius eps; a border point has fewer but lies within eps of a core point; a noise point is neither and is the outlier. A cluster forms by starting at a core point and flood-filling: pull in its neighbors, and for any neighbor that is itself core, pull in its neighbors too. This density-reachability is transitive through core points, which lets one cluster chain along a long curvy shape.

**Q2: Why does DBSCAN fail on varying-density data, and what fixes it?**
Because it uses a single global eps, which cannot be simultaneously tight enough to separate a dense cluster and loose enough to keep a sparse cluster together, so one setting always mis-handles one of them. HDBSCAN fixes this by varying the density threshold across the data and is the practical default. In high dimensions, distance concentration also makes dense versus sparse meaningless, so you reduce dimensions first.

**Q3: What is the linkage criterion and how do single and Ward differ?**
Linkage is the rule for the distance between two clusters, and it is the most important choice in hierarchical clustering. Single linkage uses the closest pair, producing long stringy chains that trace non-globular shapes but break when a noise bridge merges two real clusters. Ward merges the pair that least increases total within-cluster variance, giving balanced compact clusters and behaving most like K-Means, and it is the usual default but requires Euclidean distance.

**Q4: Why does hierarchical clustering not scale, and what else limits it?**
It builds and updates a full pairwise distance matrix, which is \(O(n^2)\) in memory, so around 50k points already needs billions of entries, and the time is \(O(n^2\log n)\) or worse. Beyond memory, its merges are greedy and irreversible, so an early bad merge propagates up the whole tree, and it produces no model, so assigning a new point requires rebuilding from scratch.
