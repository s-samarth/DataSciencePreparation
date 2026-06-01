# Variance, Bagging & OOB

Everything about bagging follows from one fact: a deep tree has low bias and high variance, and averaging many of them cuts the variance without touching the bias. This page covers why a single tree overfits, the exact variance algebra that bagging exploits, and bootstrapping with its free out-of-bag error estimate.

!!! tip "Rapid Recall"
    A deep tree memorizes the data: near-zero bias but violent variance, because greedy hierarchical splitting means a few changed points near the top flip the first split and cascade. Averaging B models with variance \(\sigma^2\) and pairwise correlation \(\rho\) gives \(\rho\sigma^2+\frac{1-\rho}{B}\sigma^2\), so adding trees only kills the second term and \(\rho\) sets a hard floor. Bagging fakes many datasets by sampling with replacement, training one tree each and aggregating, which decorrelates errors. Each bootstrap leaves out about 37% of points, giving an unbiased out-of-bag error estimate for free.

## §1 Why a single tree overfits

The foundation for everything: a tree is a high-variance, low-bias learner, for structural reasons.

!!! note "Core cause"
    Unconstrained, a tree splits until every leaf is pure, in the limit one leaf per training point. It *memorizes* the data: near-zero bias, but the feature space is carved into tiny regions dictated by individual noisy points.

### Why that is high variance, precisely

Variance = how much the learned model changes if you swap in a different training sample from the same distribution. A deep tree is violently sensitive: because splitting is **greedy and hierarchical**, changing a few points near the top can flip the *first* split, which cascades into a completely different tree below. Two trees on two samples from the same distribution can look totally different, that instability *is* high variance.

Compounding it: greedy splitting means errors propagate rather than average out within one tree, and each split consumes data, so deep nodes decide based on very few points, fitting noise, not signal.

!!! warning "The single-tree dilemma"
    Deep tree = low bias, high variance (overfits). Shallow/pruned tree = high bias, low variance (underfits). Regularization knobs (`max_depth`, `min_samples_leaf`, …) only trade one for the other. *Bagging breaks this tradeoff*, keep deep trees' low bias, kill the variance by averaging.

## §2 Why averaging reduces variance

The exact mechanism, not hand-wavy. This is the single most important math in bagging.

### Case 1: independent models

\(B\) models, each variance \({\sigma}^2\). Variance of their mean (i.i.d.):

$$\text{Var}\Big(\frac{1}{B}\sum_{i=1}^B X_i\Big)=\frac{1}{B^2}\cdot B{\sigma}^2=\frac{{\sigma}^2}{B}$$

Average 100 independent trees → variance drops 100 times. Uncorrelated errors partially cancel (one overshoots, another undershoots). **Bias unchanged** (averaging unbiased models stays unbiased) → variance reduced *for free*.

### Case 2: correlated models (the realistic case)

Trees on the same data make correlated errors. With pairwise correlation \(\rho\):

!!! note "The master equation of bagging / RF"
    $$\text{Var}(\text{average})=\rho{\sigma}^2+\frac{1-\rho}{B}{\sigma}^2$$

- **Second term** \(\frac{1-\rho}{B}{\sigma}^2\): vanishes as \(B\to\infty\). Adding trees kills it. Cheap/free.
- **First term** \(\rho{\sigma}^2\): **does not depend on \(B\) at all.** Correlation sets a hard variance floor, no number of trees gets below it.

The design consequence: to reduce variance you must reduce \(\rho\), make trees as uncorrelated as possible. Adding trees only helps the cheap term; everything Random Forest does beyond plain bagging is an attack on \(\rho\).

## §3 Bagging and bootstrapping (OOB)

### The problem and the fake

You'd want many independent training sets to average, but you have only one, and more data isn't an option. So you *fake* multiple datasets.

!!! note "Bootstrapping"
    Create a new dataset of size \(n\) by *sampling with replacement* from the original \(n\) points. Some points repeat, others are omitted. Do it \(B\) times for \(B\) different "views." Bagging = Bootstrap AGGregating: train one tree per bootstrap sample, then aggregate (average for regression, majority vote for classification).

Different bootstrap samples → different trees → partially decorrelated errors → \(\rho<1\) → averaging buys real variance reduction.

### Out-of-bag (OOB) estimate

Probability a given point is never picked in one size-\(n\) bootstrap:

$${\Big(1-\frac{1}{n}\Big)}^n\;\xrightarrow{\,n\to\infty\,}\;\frac{1}{e}\approx 0.368$$

So **about 37% of points are left out** of each sample. Predict each point using only the trees that did *not* train on it → an unbiased error estimate **without a separate holdout or cross-validation**. A genuine free perk of bagging.

## Interview questions

**Q1: Why is a single decision tree a high-variance model?**
Because splitting is greedy and hierarchical, so swapping in a slightly different training sample can flip the first split near the top, which cascades into an entirely different tree below. Each split also consumes data, so deep nodes decide on very few points and fit noise. Unconstrained, the tree grows one leaf per point and memorizes, giving near-zero bias but extreme sensitivity to the data.

**Q2: Write the variance of an average of B correlated trees and interpret it.**
It is \(\rho\sigma^2+\frac{1-\rho}{B}\sigma^2\), where \(\sigma^2\) is the per-tree variance and \(\rho\) the pairwise correlation. The second term vanishes as B grows, so adding trees is a cheap, monotone variance reduction, but the first term does not depend on B, so the correlation sets a hard floor. This is why reducing \(\rho\) matters more than adding trees.

**Q3: What is bagging and why does it reduce variance without raising bias?**
Bagging trains one model per bootstrap sample, drawn with replacement, then aggregates by averaging or majority vote. Different bootstraps yield partially decorrelated trees, so their errors partially cancel when averaged, cutting variance. Averaging unbiased models stays unbiased, so the bias is unchanged and the variance reduction comes for free.

**Q4: What is the out-of-bag estimate and where does the 37% come from?**
In a size-n bootstrap, the chance a given point is never drawn is \((1-1/n)^n\), which tends to \(1/e\approx 0.368\) as n grows, so about 37% of points are left out of each sample. You evaluate each point using only the trees that did not train on it, which yields an unbiased error estimate without a separate holdout or cross-validation, a free perk of bagging.
