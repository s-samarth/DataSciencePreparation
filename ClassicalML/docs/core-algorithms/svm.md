# Support Vector Machines

The support vector machine picks the one separating hyperplane that sits as far as possible from the closest points of either class, and that single idea, the margin, drives everything else. This page builds from the geometry to the dual, shows why the dual exposes only dot products and so unlocks the kernel trick, covers soft margin and C, and ends with the honest verdict on where SVMs still belong.

!!! tip "Rapid Recall"
    SVM maximizes the margin by minimizing \(\|\mathbf w\|^2\) subject to every point clearing a signed margin of 1, a convex quadratic program with one global optimum. The dual rewrites this so the data appears only as pairwise dot products and the solution is built from support vectors alone. Swapping the dot product for a kernel runs a linear SVM in a richer implicit space without ever visiting it, with RBF giving an infinite-dimensional space. C is the bias-variance dial trading margin width against violations. Infinite capacity is a tuning burden, not a free win, and only LinearSVC really scales.

## §1 What an SVM actually is

A linear classifier draws a hyperplane $\mathbf{w}\cdot\mathbf{x}+b=0$. Infinitely many hyperplanes separate two linearly-separable classes — logistic regression picks one by maximizing likelihood, the perceptron picks whatever it stumbles into first. **SVM picks the one specific hyperplane that sits as far as possible from the closest points of either class.** That "as far as possible" is the entire idea; everything else is machinery serving it.

!!! note "Why the margin generalizes"
    The test points you'll see later are noisy perturbations of training points. A boundary jammed against your training data flips its prediction the moment a test point drifts slightly. A boundary with a wide buffer is robust to that drift. SVM explicitly optimizes for the *worst-case nearest point*, not average fit — that's the geometric argument for why a large margin generalizes.

## §2 How it finds the boundary: margin and the scaling fix

The signed distance from a point $\mathbf{x}$ to the hyperplane is:

$$\text{distance} = \frac{\mathbf{w}\cdot\mathbf{x}+b}{\lVert \mathbf{w}\rVert}$$

We want the closest points as far as possible. There's a *scaling redundancy*: $\mathbf{w}\cdot\mathbf{x}+b=0$ and $2\mathbf{w}\cdot\mathbf{x}+2b=0$ describe the *same plane*. SVM fixes the scale by declaring the closest points satisfy $\lvert \mathbf{w}\cdot\mathbf{x}+b\rvert = 1$. Those closest points are the *support vectors*.

!!! note "Why a scaling redundancy when distance already divides by ‖w‖?"
    The **distance** is scale-free — multiply $(\mathbf{w},b)$ by any $c$ and it cancels top and bottom. That immunity is exactly the problem. The **constraint quantity** $y_i(\mathbf{w}\cdot\mathbf{x}_i+b)$ is *not* scale-free; scaling $(\mathbf{w},b)$ by $c$ scales it by $c$ too. So infinitely many $(\mathbf{w},b)$ describe the same plane with the same margin, differing only by scale — the optimization is underdetermined. Pinning the support vectors to $\lvert\cdot\rvert=1$ nails down that free scale and makes the problem well-posed.

!!! warning "What breaks if you leave everything scale-free"
    The real goal is "maximize $1/\lVert\mathbf{w}\rVert$," i.e. *minimize $\lVert\mathbf{w}\rVert$*. Alone, that has a trivial broken answer: shrink $\mathbf{w}\to 0$, margin $\to\infty$ — meaningless. The constraint $y_i(\mathbf{w}\cdot\mathbf{x}_i+b)\ge 1$ is the anchor that forbids the collapse: the closest points are forced to produce an output of magnitude $\ge 1$, pinning $\lVert\mathbf{w}\rVert$ to a finite minimum. Distance is scale-free, so the optimizer can cheat by shrinking w to fake an infinite margin; the non-scale-free constraint blocks the cheat.

With the closest points at $\lvert\mathbf{w}\cdot\mathbf{x}+b\rvert=1$, the distance from a support vector to the plane is $1/\lVert\mathbf{w}\rVert$ and the full margin is $2/\lVert\mathbf{w}\rVert$. Maximizing $2/\lVert\mathbf{w}\rVert$ = minimizing $\lVert\mathbf{w}\rVert^2$. The hard-margin program:

$$\min_{\mathbf{w},b}\ \tfrac{1}{2}\lVert\mathbf{w}\rVert^2 \quad\text{s.t.}\quad y_i(\mathbf{w}\cdot\mathbf{x}_i+b)\ge 1\ \ \forall i$$

The label $y_i\in\{-1,+1\}$ folds both classes into one clean inequality (positive points $\ge +1$, negative points $\le -1$). This is a **convex quadratic program** — one global optimum, no local minima, solvable reliably. That convexity is a genuine selling point versus neural nets.

## §3 The dual, and what support vectors really are

Solving with Lagrange multipliers $\alpha_i\ge 0$ converts the primal (solve for $\mathbf{w},b$) into the *dual* (solve for the $\alpha_i$). Two results matter enormously:

### Result 1: w is built from support vectors only

$$\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$$

From the KKT conditions, $\alpha_i = 0$ for every point comfortably outside the margin. Only the **support vectors** (points exactly on the margin, or inside it in the soft case) get $\alpha_i>0$.

!!! note "What is a support vector, plainly"
    A training point that lands *exactly on the edge of the margin* (or inside it). It's one of the closest points to the boundary — touching the "pipe wall." The boundary is held in place only by the points pressing against it, like a tent held up only by the poles touching the fabric. Points far away touch nothing, so removing them changes nothing — and mathematically they have $\alpha_i=0$, so they contribute zero to $\mathbf{w}$. That is literally why "delete a non-support-vector and the boundary doesn't move."

### Result 2: data appears only as dot products

$$\max_{\alpha}\ \sum_i \alpha_i - \tfrac{1}{2}\sum_i\sum_j \alpha_i\alpha_j\, y_i y_j\,(\mathbf{x}_i\cdot\mathbf{x}_j)\quad\text{s.t.}\quad 0\le\alpha_i\le C,\ \ \sum_i\alpha_i y_i=0$$

The data never appears alone — only through pairwise dot products $\mathbf{x}_i\cdot\mathbf{x}_j$. The algorithm never needs the *coordinates* of points, only the *dot products between pairs*. **This is the hinge the entire kernel trick swings on.**

## §4 The kernel trick

A dot product is a similarity score: how aligned two vectors are. The trick is one observation:

!!! note "The core idea"
    If the algorithm only ever touches the data through dot products, you can swap $\mathbf{x}_i\cdot\mathbf{x}_j$ for any function $K(\mathbf{x}_i,\mathbf{x}_j)$ that *behaves like* a dot product in some other space — without ever visiting that space. The SVM optimizes happily using $K$, never noticing the swap.

Concretely: map data into a richer space via $\varphi(\mathbf{x})$ and run a linear SVM there. The dual would need $\varphi(\mathbf{x}_i)\cdot\varphi(\mathbf{x}_j)$. For many useful $\varphi$, a function $K$ computes that dot product directly from the original vectors, far cheaper than building $\varphi$.

### Why a degree-2 map of a 2D point becomes 6D

A degree-2 polynomial feature map writes every term up to degree 2:

$$\varphi(\mathbf{x}) = \big(1,\ \sqrt2\,x_1,\ \sqrt2\,x_2,\ x_1^2,\ x_2^2,\ \sqrt2\,x_1 x_2\big)$$

Count them: constant, two linear, two squares, one cross term = **6 features**. The $\sqrt2$ factors are bookkeeping so the dot product comes out clean. Those $x_1^2, x_2^2, x_1x_2$ terms are what let a *linear* boundary in 6D be a *curved* boundary back in 2D — you buy nonlinearity by adding dimensions.

!!! note "Where the advantage is"
    The naive way builds the 6D vector and dots it (6 mults + construction cost). The kernel does the identical thing on the original 2D vectors:
    $$K(\mathbf{x},\mathbf{z}) = (\mathbf{x}\cdot\mathbf{z}+1)^2 = \varphi(\mathbf{x})\cdot\varphi(\mathbf{z})$$
    One 2D dot product, an add, a square. For 2D→6D the saving is small. But degree-$d$ on $n$ features makes roughly $\binom{n+d}{d}$ terms: a 1000-dim input at degree 4 has *billions* of entries in $\varphi$, impossible to build — yet $K=(\mathbf{x}\cdot\mathbf{z}+1)^4$ is still one 1000-dim dot product, an add, a 4th power. The kernel cost stays tied to the original dimension no matter how violently the implicit space explodes.

### The RBF kernel: where K comes from, and what φ is

With polynomials you start from $\varphi$ and find $K$. With RBF you go the other way: start from the similarity you *want* — high when points are close, decaying as they separate:

$$K(\mathbf{x},\mathbf{z}) = \exp\!\big(-\gamma\,\lVert\mathbf{x}-\mathbf{z}\rVert^2\big)$$

You always "know" $K$ — it's a closed-form formula on the original vectors (squared distance, scale by $-\gamma$, exponentiate). That's all the SVM ever needs. **What is $\varphi$ for RBF?** Expand the exponential as a Taylor series and you get polynomial terms of *every* degree — so $\varphi(\mathbf{x})$ has *infinitely many* coordinates. You literally could never write it down, and never have to: $K$ hands you the dot product in that infinite space with a single exponential. The $\gamma$ knob controls how fast similarity decays — large $\gamma$ = influence dies quickly = wiggly boundary.

!!! note "Mercer's condition"
    "Behaves like a dot product" is precise: the kernel matrix must be symmetric positive semi-definite. Naming Mercer's condition signals you understand kernels aren't arbitrary similarity functions.

### The kernels and when to use them

| Kernel | Form | Implicit space | Use when |
| --- | --- | --- | --- |
| **Linear** | $\mathbf{x}\cdot\mathbf{z}$ | original | features ≫ samples (text/TF-IDF), want speed and interpretability. Always the baseline. |
| **RBF (Gaussian)** | $\exp(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2)$ | infinite-dim | default for nonlinear, low-to-medium dim, no strong prior. The workhorse. |
| **Polynomial** | $(\mathbf{x}\cdot\mathbf{z}+c)^d$ | monomials up to degree $d$ | when feature *interactions* matter. Sensitive to $d$, numerically touchy. |
| **Sigmoid** | $\tanh(\kappa\,\mathbf{x}\cdot\mathbf{z}+c)$ | — | mimics a 2-layer net; rarely used, often not a valid Mercer kernel. Historical. |

Discipline: **try linear first.** High-dimensional (text) → usually already separable, stop there. Linear underfits + small/medium data → RBF. Polynomial only with a genuine reason to believe interactions of a known degree drive the label.

### Inference: why kernel SVM is slow but linear SVM collapses to one w

**Linear case.** $f(\mathbf{x})=\text{sign}(\mathbf{w}\cdot\mathbf{x}+b)$ with $\mathbf{w}=\sum_i\alpha_i y_i\mathbf{x}_i$. The $\mathbf{x}_i$ are plain vectors, so you carry out that sum *once at training time* and collapse all support vectors into a single $\mathbf{w}$. Throw the support vectors away; keep $\mathbf{w}$ and $b$. Inference is one dot product, independent of how many support vectors there were.

**Kernel case.** Substituting $K$:

$$f(\mathbf{x}) = \text{sign}\!\Big(\sum_i \alpha_i y_i\, K(\mathbf{x}_i,\mathbf{x}) + b\Big)$$

To collapse, you'd need $\mathbf{w}=\sum_i\alpha_i y_i\varphi(\mathbf{x}_i)$ — but $\varphi(\mathbf{x}_i)$ lives in the high- or infinite-dimensional space. **You can't store $\mathbf{w}$; for RBF it's an infinite vector.** The new point $\mathbf{x}$ is locked *inside* $K(\mathbf{x}_i,\mathbf{x})$, paired with each support vector individually, and only available at prediction time. So you must keep every support vector and loop over all of them per prediction.

!!! note "The contrast in one line"
    Linear SVM sums its support vectors into a single $\mathbf{w}$ once and discards them; kernel SVM can't form $\mathbf{w}$ (it lives in an unbuildable space), so it must keep every support vector and re-evaluate the kernel against each one for every prediction. More support vectors → slower inference and a fatter model. High $C$ or $\gamma$ tend to produce *more* support vectors — a real deployment cost, not just trivia.

## §5 Soft margin and C: where C was hiding

The hard-margin program demands *zero* violations. If even one point is on the wrong side (noise, outlier, non-separable classes), **no $(\mathbf{w},b)$ satisfies all constraints and the problem has no solution at all.** Hard-margin SVM literally can't run on real data. The fix: allow violations but charge a penalty, via slack variables $\xi_i$ and the penalty weight $C$:

$$\min_{\mathbf{w},b,\xi}\ \tfrac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(\mathbf{w}\cdot\mathbf{x}_i+b)\ge 1-\xi_i,\ \ \xi_i\ge 0$$

Each point gets an allowance $\xi_i$ to fall short: $\xi_i=0$ obeys fully; $0<\xi_i<1$ inside margin but still correct; $\xi_i>1$ misclassified. The total slack used is added to the cost, scaled by $C$.

!!! note "What C does: the bias-variance dial"
    $C$ is the *exchange rate* between a wide margin (small $\lVert\mathbf{w}\rVert^2$) and few violations (small $\sum\xi_i$). *Large $C$* → slack is expensive → refuses violations → narrow, contorted margin (high variance; $C\to\infty$ recovers hard margin). *Small $C$* → slack is cheap → tolerates violations for a wider, smoother margin (high bias). In the dual it appears as the cap $0\le\alpha_i\le C$ — it limits how much influence any single point can exert, which is also literally why soft margin is solvable where hard margin wasn't.

!!! warning "C vs γ: the confusion interviewers exploit"
    Both large $C$ and large $\gamma$ push toward overfitting, but for different reasons. *$C$ = error tolerance* (how hard you punish margin violations). *$\gamma$ = boundary flexibility* (how far one RBF point's influence reaches). You grid-search them *together* on a log scale because they interact — the right $C$ depends on the $\gamma$ you picked.

## §6 If RBF builds an infinite polynomial space, why isn't it the best classifier?

Sharp question — and the "infinite space" is exactly the *problem*, not the superpower.

- **More capacity is the central danger.** An infinitely flexible model fits any training set perfectly, noise included. With high $\gamma$ you can *always* find a large-margin separator in the infinite space — but around a boundary contorted into nonsense. Separability in infinite dimensions is trivial and therefore meaningless; *everything* is separable up there. The whole game becomes controlling capacity back down with $C$ and $\gamma$, so RBF-SVM is only as good as your tuning.
- **Scaling.** Training is $O(n^2)$–$O(n^3)$ and inference loops over every support vector. Dead on a million rows. Neural nets train on mini-batches and scale to billions; logistic regression is linear-time.
- **No feature learning.** RBF gives a *fixed, generic* similarity — "close in raw input space = similar." It does not *learn* what features matter. Neural nets learn the representation (edges → faces → separation). On images/audio/language, raw-input similarity is worthless, so no amount of margin helps. The infinite space RBF builds is infinite but *dumb* — polynomial combinations of raw inputs, not learned abstractions.

!!! note "Interview answer"
    "RBF-SVM's infinite capacity means it can fit anything, which means it can overfit anything — so it's not a free win, it's a tuning burden. And it uses a fixed, generic similarity rather than learning features, so it loses to deep nets on perceptual data and to XGBoost on tabular scale, and can't scale past ~100K rows regardless. Capacity was never the bottleneck; controlling capacity, learning representations, and scaling were."

## §7 Do we even care about SVMs in 2026?

Not dead, but pushed into a narrow corner — and in that corner still the right tool, not nostalgia.

- **High-dimensional, low-sample data ($d\gg n$)** — the core survival niche. The margin acts as strong regularization where deep learning starves. *Genomics* (20,000 features, 200 patients) is the textbook case. *Small-data text* (TF-IDF + LinearSVC) is shockingly competitive, sometimes beating fine-tuned transformers on a few thousand labeled docs — cheap, no GPU, inference is one dot product.
- **One-Class SVM** for anomaly/novelty detection (though [Isolation Forest](../unsupervised/anomaly-detection.md) often beats it now).
- **Fast baseline** — LinearSVC alongside logistic regression in the "beat this before reaching for anything heavier" set.

**Genuinely dead for:** tabular at scale (XGBoost/LightGBM win), anything >100K rows ($O(n^2)$ training), images/audio/sequence/language (deep learning), and calibrated probabilities (SVM gives distances; Platt scaling is a bolt-on hack).

!!! note "The one distinction to keep crisp"
    LinearSVC scales and is alive; kernel SVM (RBF) is the part that's mostly gone, because the $O(n^2{+})$ training is what kills it at modern scale. When people say "SVMs are dead," they almost always mean kernel SVMs. Linear SVMs are just regularized linear classifiers and quietly persist.

## Interview questions

**Q1: What does an SVM optimize, and why does the margin help generalization?**
It finds the separating hyperplane that maximizes the margin, the distance to the nearest points of either class, by minimizing \(\|\mathbf w\|^2\) subject to every point clearing a signed margin of 1. A wide buffer is robust to test points that drift from their training positions, whereas a boundary jammed against the data flips easily. SVM explicitly optimizes the worst-case nearest point rather than average fit, which is the geometric argument for generalization.

**Q2: Explain the kernel trick and why it is efficient.**
The dual objective touches the data only through pairwise dot products, so you can replace each dot product with a kernel that equals a dot product in a richer implicit feature space, without ever constructing that space. For a degree-d polynomial the explicit feature map can have billions of terms, yet the kernel \((\mathbf x\cdot\mathbf z+1)^d\) is one original-dimension dot product. The RBF kernel corresponds to an infinite-dimensional space but is still a single closed-form exponential.

**Q3: Why is kernel SVM slow at inference while linear SVM is fast?**
Linear SVM collapses its support vectors into a single weight vector \(\mathbf w=\sum_i\alpha_i y_i\mathbf x_i\) once at training time, so prediction is one dot product regardless of the number of support vectors. Kernel SVM cannot form \(\mathbf w\) because it would live in the high or infinite dimensional space, so it must keep every support vector and evaluate the kernel against each one per prediction. More support vectors mean a slower, fatter model, and high C or gamma produce more of them.

**Q4: What is the difference between C and gamma?**
C is the error tolerance, the exchange rate between a wide margin and few violations: large C punishes violations hard for a narrow high-variance boundary, small C tolerates them for a wider high-bias one. Gamma is RBF boundary flexibility, how far a single point's influence reaches: large gamma gives a wiggly local boundary. Both push toward overfitting but for different reasons, so you grid-search them together on a log scale.

**Q5: Where do SVMs still win in 2026?**
In the high-dimensional low-sample regime where the margin is strong regularization and deep learning starves, the textbook case being genomics with thousands of features and hundreds of patients, plus small-data text with TF-IDF and LinearSVC. LinearSVC also persists as a fast baseline and as regularized linear classification. Kernel SVM is mostly gone because its \(O(n^2)\) or worse training does not scale past roughly 100K rows.
