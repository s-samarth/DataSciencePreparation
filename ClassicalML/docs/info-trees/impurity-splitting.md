# Impurity & Splitting Criteria

A tree needs to measure how mixed a node is and how much a split cleans it up. This page covers Gini impurity and its mathematical link to entropy (it is a first-order Taylor approximation), then information gain and Gini gain, the parent-minus-weighted-children rule that selects every split.

!!! tip "Rapid Recall"
    Gini impurity \(1-\sum_i p_i^2\) is the expected error of guessing by class frequencies, and it is a first-order Taylor approximation of entropy, which is why the two agree on splits about 99% of the time. Gini is the sensible default because it is in the currency of classification error and needs no logarithm. Information gain is parent impurity minus the size-weighted child impurity, and size-weighting stops a tiny clean offshoot from faking a good split. Information gain equals \(H(S)-H(S\mid A)\), the mutual information between feature and label, and it can never be negative.

## §1 Gini impurity and its link to entropy

Same goal as entropy (measure mixedness), different imaginary game.

### Derivation

Pick a random element (class \(i\) w.p. \(p_i\)). Guess its label by sampling from the same distribution (guess \(i\) w.p. \(p_i\)). Probability wrong for that element: \(1-p_i\). Averaged over the true class:

$$\text{Gini}(S)=\sum_i p_i(1-p_i)=\sum_i p_i-\sum_i p_i^2=1-\sum_i p_i^2$$

!!! note "Meaning"
    Gini = *expected misclassification rate of the dumbest classifier* (random guessing matched to class frequencies). Lower = purer. Range for binary: [0, 0.5], peak 0.5 at 50/50.

### Two games measuring one disease

|  | The game | "Cost" it measures |
| --- | --- | --- |
| **Entropy** | Identify the class via optimal yes/no questions | Uncertainty / description length (bits) |
| **Gini** | Guess the class by random proportional stab, how often wrong? | Error probability |

Both are 0 at a pure node, maximal at uniform, monotonic in between, that's *why* they agree on splits. "Hard to identify" and "easy to misclassify" are two symptoms of the same disease (mixedness).

### Mathematical link

First-order Taylor of \(\log\) around \(p=1\): \(\,-{\log}_2 p\approx\frac{1}{\ln 2}(1-p)\). Substitute into entropy:

$$H=-\sum_i p_i{\log}_2 p_i\approx\frac{1}{\ln 2}\sum_i p_i(1-p_i)=\frac{1}{\ln 2}\,\text{Gini}$$

So **Gini is literally a first-order approximation of entropy**, the reason the curves overlap and they pick the same split about 99% of the time.

### Why Gini is the sensible default

- **Aligned with the task:** a tree ultimately *classifies* and is judged on errors. Gini is already in the currency of classification error; entropy is in the currency of information (a slightly more roundabout proxy).
- **Speed:** no logarithm, and impurity is computed millions of times across thresholds, features, and nodes.
- **They rarely disagree**, so the cheaper one costs nothing. When they do: entropy is marginally more sensitive near the pure end (log blows up faster as \(p\to 0\)), occasionally preferring more balanced splits. Academic for nearly all problems.

## §2 Information gain and Gini gain

Impurity measures one node's mess. Gain is the decision rule, how much the split reduced average mess.

### Symbols

| Symbol | Meaning |
| --- | --- |
| \(S\) | Set of samples at the current node (e.g. 100 emails, 60 spam / 40 not). |
| \(A\) | A candidate feature/question to split on (e.g. "contains the word 'free'?"). |
| \(H(S)\) | Entropy of the node before splitting. |
| \(S_v\) | Subset of samples whose feature \(A\) takes value \(v\) (e.g. \(S_{\text{yes}}\), \(S_{\text{no}}\)). |

!!! note "Definition"
    $$IG(S,A)=\underbrace{H(S)}_{\text{messiness before}}-\underbrace{\sum_v\frac{|S_v|}{|S|}\,H(S_v)}_{\text{weighted messiness after}}$$
    Measure mess before, measure mess after, subtract. Big drop implies the question sorted samples into cleaner groups. Compute for every feature, *split on the max*.

### The weighting matters (the trap)

The "after" term is a *size-weighted* average; weights \(\frac{|S_v|}{|S|}\) are the fraction of samples in each branch (sum to 1). Worked example, 100 emails split on "contains 'free'?":

```
S_yes: 30 emails, entropy 0.3
S_no : 70 emails, entropy 0.9
after = (30/100)(0.3) + (70/100)(0.9) = 0.09 + 0.63 = 0.72
IG    = H(S) - 0.72 = 0.97 - 0.72 = 0.25 bits
```

If a split sent 990 samples into a still-90%-messy node and 10 into a pure node, an *unweighted* average would look decent, but almost all data is still messy. Size-weighting makes the big messy node dominate, so IG comes out low. **Weighting stops a tiny clean offshoot from faking a good split.**

### Two readings (same arithmetic)

- **Operational:** how much average impurity dropped after splitting on \(A\). (What you compute.)
- **Information-theoretic:** the "after" term *is* \(H(S\mid A)\), so \(IG=H(S)-H(S\mid A)\) = how much knowing \(A\) reduces uncertainty about the label = the **mutual information** between feature and label. (The high-value interview answer to "what is IG fundamentally?")

!!! warning "IG ≥ 0 always (the trap)"
    A real theorem, not a convention: from concavity of entropy via Jensen, conditioning never increases expected entropy, \(H(S)\ge H(S\mid A)\). Worst case the feature is useless, so IG = 0. It can never go negative, the split can never *increase* average impurity.

### Gini gain: the exact analogue

The "gain" idea is impurity-measure-agnostic. Reuse the identical skeleton with Gini swapped in:

$$\text{Gini Gain}(S,A)=\text{Gini}(S)-\sum_v\frac{|S_v|}{|S|}\,\text{Gini}(S_v)$$

Same parent-minus-weighted-children structure; same greedy argmax procedure. CART (sklearn default) runs this with Gini. Note: IG has the clean mutual-information identity; Gini Gain is "reduction in expected proportional-guess error," no equally elegant named identity.

## Interview questions

**Q1: What does Gini impurity measure, and how is it related to entropy?**
Gini is the expected misclassification rate of a classifier that guesses labels by the class frequencies, \(1-\sum_i p_i^2\), so lower is purer. It is a first-order Taylor approximation of entropy around purity, \(H\approx\frac{1}{\ln 2}\text{Gini}\), which is why the two impurity curves nearly overlap and select the same split about 99% of the time. Both are zero at a pure node and maximal at the uniform split.

**Q2: Why is Gini the default split criterion over entropy?**
Because it is in the currency of classification error, which is what the tree is judged on, while entropy is in the currency of information, a slightly more roundabout proxy. Gini also needs no logarithm, which matters because impurity is evaluated millions of times across thresholds, features, and nodes. They rarely disagree, so the cheaper one costs nothing.

**Q3: Why is the post-split impurity size-weighted?**
Because the goal is to reduce the average impurity over all the data, not per branch. Weighting each child by its fraction of samples makes a large still-messy node dominate, so a split that sends almost everything into a messy node and a handful into a pure one correctly scores a low gain. Without weighting, a tiny clean offshoot could fake a good split.

**Q4: What is information gain fundamentally, and can it be negative?**
The "after" term is the conditional entropy \(H(S\mid A)\), so information gain is \(H(S)-H(S\mid A)\), which is exactly the mutual information between the feature and the label, how much knowing the feature reduces uncertainty about the class. It can never be negative: by Jensen's inequality and the concavity of entropy, conditioning never increases expected entropy, so the worst case is a useless feature with zero gain.
