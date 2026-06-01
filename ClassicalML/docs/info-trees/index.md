# Information Theory & Decision Trees

This section runs from what information fundamentally is, through entropy, cross-entropy, and KL and the losses they generate, down to how a decision tree is actually built, optimized, and made tractable. The thread is that the same quantity, surprise, underlies the loss functions of every classifier and the split criteria of every tree.

!!! tip "Rapid Recall"
    Information is the number of optimal yes/no questions to pin down an outcome, and the surprise of a probability-p outcome is \(-\log_2 p\). Entropy is expected surprise; cross-entropy is the surprise you actually pay when reality is p but you encoded for q; KL is the wasted bits, cross-entropy minus entropy. Classifier training minimizes cross-entropy, which equals maximizing likelihood and minimizing KL. Decision trees split greedily on the feature with the largest impurity reduction, where Gini is a first-order Taylor approximation of entropy, and information gain is exactly the mutual information between feature and label.

## What each page covers

- **[Information & Entropy](information-entropy.md)**: what information is, surprise as one quantity seen before and after the event, the Cauchy derivation that forces the logarithm, and the shape and units of entropy.
- **[Cross-Entropy, KL & Log-Loss](cross-entropy-kl.md)**: cross-entropy as the classifier loss, KL divergence and its asymmetry, why the logistic loss is cross-entropy and not entropy, and knowledge distillation.
- **[Impurity & Splitting Criteria](impurity-splitting.md)**: Gini impurity and its mathematical link to entropy, and information gain and Gini gain as the split-selection rule.
- **[Building Decision Trees](building-trees.md)**: greedy recursive partitioning, continuous and categorical splitting, stopping and pruning, regression trees, and training/inference complexity.
- **[Tree Algorithms & Categorical Tricks](tree-algorithms.md)**: the ID3, C4.5, and CART lineage, what libraries actually ship, and the Breiman and CatBoost tricks for categoricals.

## One-page formula strip

- **Surprise / Information:** \(I(p)=-\log_2 p\) (same number, before vs after the event).
- **Entropy:** \(H(S)=-\sum_i p_i\log_2 p_i\) (expected surprise; binary peak = 1 bit at \(p=0.5\)).
- **Cross-entropy:** \(H(p,q)=-\sum_i p_i\log q_i\) (one-hot label gives \(-\log q_c\) = log-loss = NLL).
- **KL:** \(D_{KL}(p\parallel q)=H(p,q)-H(p)=\sum_i p_i\log\frac{p_i}{q_i}\ge 0\), asymmetric.
- **Decomposition:** \(\text{CE}=H(p)+D_{KL}(p\parallel q)\), so minimize CE = minimize KL = MLE.
- **KD loss:** \(\alpha\,H(p_{\text{true}},q_S)+(1-\alpha)T^2 D_{KL}(q_T^{(T)}\parallel q_S^{(T)})\).
- **Gini:** \(1-\sum_i p_i^2\) (about \(\frac{1}{\ln 2}\) times entropy, first-order Taylor; binary peak 0.5).
- **Information Gain:** \(H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)=H(S)-H(S\mid A)\) = mutual information.
- **Variance reduction (regression):** \(\text{Var}(S)-\sum_v\frac{|S_v|}{|S|}\text{Var}(S_v)\).
- **Tree complexity:** train \(O(d\,n\log^2 n)\) (re-sort) or \(O(d\,n\log n)\) (presort); infer \(O(\log n)\), independent of \(d\) and \(n\).
- **Continuous splits:** at most \(n-1\) midpoints. **Categorical blowup:** \(2^{k-1}-1\) reduced to \(k-1\) via Breiman sort-by-target.
