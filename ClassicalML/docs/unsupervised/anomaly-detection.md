# Anomaly Detection

Most anomaly methods model what "normal" looks like and flag deviations. Isolation Forest inverts that premise: anomalies are few and different, so they are easy to isolate with a handful of random cuts. This page covers how it works, why short path length is the signal, the counterintuitive design choices, and how it compares to other detectors.

!!! tip "Rapid Recall"
    Isolation Forest scores a point by how few random axis-aligned cuts it takes to isolate it: anomalies sit in sparse regions and get fenced off in a couple of cuts, while normal points buried in a crowd need many, so short average path length across many trees means anomaly. It needs no distance, density, or distribution, which makes it fast, near-linear, and friendly to high dimensions. Two counterintuitive choices: subsample only 256 points to avoid swamping and masking, and cap tree depth since you only care that anomalies isolate early. It misses local anomalies (use LOF) and diagonal ones (use Extended iForest).

## §1 How it works

Every other anomaly method *models what "normal" looks like* and flags deviations (fit a Gaussian → flag low-probability; find dense regions → flag sparse stragglers). Isolation Forest **inverts the premise**: anomalies are few and different, so they're *easy to isolate*. Keep making random cuts in feature space, an outlier in a sparse region gets fenced into its own box after just a few cuts; a normal point buried in a crowd needs many. **The anomaly score is: how few cuts did it take to isolate this point?** No notion of distance, density, or distribution required.

### Build one isolation tree (iTree)

1. Take a small random subsample (default **256** points).
2. Pick a **random feature**.
3. Pick a **random split value** between that feature's min and max in the node.
4. Split; recurse until every point is alone (or a depth cap is hit).

The **path length** for a point = number of edges from root to its leaf = number of random cuts needed to isolate it.

### Build a forest

Many such trees, each on a different random subsample with different random splits. **Score** a point by its *average path length* across trees, normalize to \([0,1]\).

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg">
<text fill="#9aa3b2" font-family="sans-serif" font-size="12" text-anchor="middle" x="140" y="24">normal point — many cuts</text>
<rect fill="none" height="150" stroke="#2a2f3a" width="200" x="40" y="40"></rect>
<line opacity=".6" stroke="#5fb6c9" x1="120" x2="120" y1="40" y2="190"></line>
<line opacity=".6" stroke="#5fb6c9" x1="120" x2="240" y1="110" y2="110"></line>
<line opacity=".6" stroke="#5fb6c9" x1="80" x2="80" y1="40" y2="110"></line>
<line opacity=".6" stroke="#5fb6c9" x1="80" x2="120" y1="75" y2="75"></line>
<circle cx="100" cy="92" fill="#5fb6c9" r="5"></circle>
<text fill="#5fb6c9" font-family="sans-serif" font-size="11" x="250" y="180">deep</text>
<text fill="#9aa3b2" font-family="sans-serif" font-size="12" text-anchor="middle" x="420" y="24">anomaly — few cuts</text>
<rect fill="none" height="150" stroke="#2a2f3a" width="200" x="320" y="40"></rect>
<line stroke="#eb6f6f" x1="470" x2="470" y1="40" y2="190"></line>
<line stroke="#eb6f6f" x1="470" x2="520" y1="80" y2="80"></line>
<circle cx="497" cy="60" fill="#eb6f6f" r="5"></circle>
<text fill="#eb6f6f" font-family="sans-serif" font-size="11" x="330" y="180">shallow → isolated fast</text>
</svg>
<figcaption>A normal point (left) requires many random cuts to isolate; an anomaly (right) is fenced off in just a couple, short path length is the signal.</figcaption>
</figure>

Anomaly score (\(h(x)\) = path length, \(c(n)\) = average path length normalizer for \(n\) points):

$$s(x,n)=2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$

Score → 1: isolated very quickly (anomaly). Well below 0.5: many cuts (normal). About 0.5: no clear signal.

## §2 Why it is NOT a decision tree or Random Forest

| Aspect | Decision Tree / Random Forest | Isolation Forest |
| --- | --- | --- |
| Supervision | **Supervised**, needs labels \(y\); splits to separate classes / predict target | **Unsupervised**, no labels; measures a structural property |
| Split choice | **Optimized**, best feature and threshold by information gain / Gini / variance reduction | **Random** feature, **random** threshold; no objective minimized at the split |
| Output | Class label / regression value from the *leaf contents* | Anomaly score from *path length* (how deep), not leaf contents |
| Meaning of depth | No inherent meaning | **Short depth IS the anomaly signal** |
| What's averaged | Predictions (votes / mean) to reduce prediction variance | Path lengths to reduce isolation-depth variance |

It borrows Random Forest's many-random-trees *skeleton* but nothing of its purpose, splitting logic, or output semantics.

!!! note "Two counterintuitive design choices (interview gold)"
    - **Subsample only 256, small is better.** Large samples cause *swamping* (anomalies surrounded by so many normals they stop looking isolated) and *masking* (anomaly clusters hiding each other). Small subsamples keep anomalies sparse and genuinely isolatable, and make it fast and memory-light.
    - **Shallow / depth-capped trees** (about \(\log_2\psi\)). You only care that anomalies isolate *early*; if a point isn't isolated by the cap, it's clearly normal. No loss of signal.

## §3 Complexity

Training is about \(O(t\cdot\psi\log\psi)\) (\(t\) = trees, \(\psi\) = subsample = 256). Since \(\psi\) is a tiny constant, effectively **linear in dataset size and trivially parallelizable**. Scoring \(O(t\log\psi)\) per point. One of the fastest anomaly detectors.

### Where to use it

- **Good for:** high-dimensional data (no distance metric, so it sidesteps the curse of dimensionality), large data (near-linear, parallel), no labels, no distribution assumption, global scattered anomalies.

!!! warning "Watch out for"
    - **Local anomalies in varying-density data** — a point anomalous only relative to its local neighborhood (but not globally sparse) can be missed. **Local Outlier Factor (LOF)** is the density-based alternative for local anomalies.
    - **Axis-aligned cut weakness** — splits on one feature at a time struggle with anomalies that stand out only along a *diagonal* feature combination. **Extended Isolation Forest** uses oblique (random-direction) cuts.
    - **`contamination` parameter** — you usually must specify the expected anomaly proportion to set the threshold; getting it wrong skews how many points get flagged.

### Among anomaly detectors

| Method | Basis | Notes |
| --- | --- | --- |
| **Isolation Forest** | Isolation | Unsupervised, fast, high-D friendly, no distribution assumption, finds *global* anomalies. Default for tabular at scale. |
| Gaussian / GMM density | Distribution | Flags low-probability points; good when normal is Gaussian-ish, bad in high-D. |
| DBSCAN noise | Density | Anomalies as a clustering byproduct; struggles with varying density / high-D. |
| LOF | Local density | Catches *local* anomalies Isolation Forest misses. |
| One-Class SVM | Boundary | Learns a boundary around normal data; powerful but slow, hard to tune, scales poorly. |

## Interview questions

**Q1: How does Isolation Forest score anomalies, and why is that backwards from other methods?**
It makes random axis-aligned cuts and measures the path length, the number of cuts needed to isolate each point, averaged over many trees. Other methods model what normal looks like and flag deviations, whereas Isolation Forest exploits that anomalies are few and different, so they sit in sparse regions and get fenced into their own box in just a couple of cuts, while normal points in a crowd need many. Short average path length is therefore the anomaly signal, with no distance or density needed.

**Q2: Why subsample only 256 points per tree?**
Because large samples cause swamping, where an anomaly is surrounded by so many normal points that it no longer looks isolated, and masking, where dense anomaly clusters hide each other. A small subsample keeps anomalies genuinely sparse and quick to isolate, preserving the signal, and it also makes training fast and memory-light. It is one of the rare cases where less data per learner is deliberately better.

**Q3: How is an isolation tree different from a decision tree?**
A decision tree is supervised and chooses the best feature and threshold to optimize an impurity criterion, outputting a class or value from the leaf contents. An isolation tree is unsupervised, picks a random feature and random split with no objective, and the output is the depth at which a point is isolated rather than anything in the leaf. It borrows the many-random-trees skeleton but none of the purpose, splitting logic, or semantics.

**Q4: What anomalies does Isolation Forest miss, and what addresses them?**
It can miss local anomalies, points that are unusual only relative to their local neighborhood but not globally sparse, for which Local Outlier Factor is the density-based alternative. Because its cuts are axis-aligned, it also struggles with anomalies that stand out only along a diagonal combination of features, which Extended Isolation Forest fixes using oblique random-direction cuts. You also typically set a contamination parameter that must roughly match the true anomaly rate.
