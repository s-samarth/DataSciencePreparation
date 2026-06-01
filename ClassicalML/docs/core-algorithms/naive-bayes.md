# Naive Bayes

Naive Bayes asks the probability question directly: given the observed features, which class is most probable? It is a generative classifier that models what each class looks like, inverts with Bayes, and survives on a deliberately false independence assumption that corrupts the probabilities but preserves the ranking. This page covers the theorem, the naive assumption, MLE with smoothing, and the variants.

!!! tip "Rapid Recall"
    Naive Bayes models \(P(\mathbf x\mid y)P(y)\) and picks the argmax, dropping the evidence denominator because it is constant across classes. The naive assumption factorizes the intractable joint likelihood into a product of per-feature terms, which is false but usually preserves the argmax even as it ruins calibration. Training is counting frequencies by MLE, with Laplace smoothing to stop a single unseen feature from zeroing a class. Pick the variant by feature type: Multinomial for counts, Bernoulli for present/absent, Gaussian for continuous. It survives in 2026 as a cost, latency, and tiny-data play and as the baseline to beat.

!!! warning "Two corrections for interview hygiene"
    Naive Bayes is *not* a large-margin classifier (that's SVM) and it has *no kernels* (also SVM). It's a *probabilistic generative classifier*. The NB analog of "different kinds" is its variants — Gaussian, Multinomial, Bernoulli — chosen by feature type.

## §1 What it is

A classifier that asks a probability question directly: **given the observed features, which class is most probable?** It computes $P(y\mid\mathbf{x})$ for each class and picks the largest. It's called *generative* because, instead of learning a boundary between classes (like logistic regression or SVM — *discriminative*), it learns what each class's data *looks like* — a model of $P(\mathbf{x}\mid y)$ — then uses Bayes' theorem to flip that into $P(y\mid\mathbf{x})$. It models how data was generated per class, then asks which generator most likely produced this point.

## §2 Bayes' theorem and the law of total probability

We want $P(y\mid\mathbf{x})$ but can't estimate it directly — too many feature combinations to ever see enough labeled examples. We *can* estimate the reverse $P(\mathbf{x}\mid y)$ by counting (within spam, how often does "free" appear?). Bayes' theorem is the bridge:

$$P(y\mid\mathbf{x}) = \frac{P(\mathbf{x}\mid y)\,P(y)}{P(\mathbf{x})}$$

- $P(y\mid\mathbf{x})$ — **posterior**: what we want, belief in class $y$ after seeing data.
- $P(\mathbf{x}\mid y)$ — **likelihood**: how probable the features are if the class were $y$. Estimated from training data.
- $P(y)$ — **prior**: how common class $y$ is before looking at features.
- $P(\mathbf{x})$ — **evidence**: how probable the features are overall.

### Where the law of total probability comes in

The denominator is computed by the law of total probability — sum over every class of the within-class probability weighted by class frequency:

$$P(\mathbf{x}) = \sum_k P(\mathbf{x}\mid y_k)\,P(y_k)$$

The classes are mutually exclusive and exhaustive (a partition), so this normalizer guarantees the posteriors sum to 1.

!!! note "The key practical move"
    $P(\mathbf{x})$ is the *same constant for every class* — it doesn't depend on $y$. Since we only want the *argmax* over classes, we drop the denominator entirely:
    $$\hat y = \arg\max_y\ P(\mathbf{x}\mid y)\,P(y)$$
    You only compute $P(\mathbf{x})$ when you want actual calibrated posterior numbers.

## §3 The naive assumption: the part that makes it work

The wall: $P(\mathbf{x}\mid y) = P(x_1,x_2,\dots,x_n\mid y)$ is the joint probability of *all features together*. For a 50,000-word vocabulary that's the probability of one specific full word-combination — you'd need astronomically more data than exists. Intractable.

The naive assumption smashes through it: **assume features are conditionally independent given the class.** The joint factorizes:

$$P(x_1,\dots,x_n\mid y) = \prod_i P(x_i\mid y)$$

Now each $P(x_i\mid y)$ is trivial to estimate. One impossible joint estimation becomes $n$ easy marginal estimations.

!!! note "Why a blatantly false assumption still classifies well"
    "Free" and "money" co-occur in spam — they're correlated, not independent. But NB doesn't need accurate probabilities, only the right *argmax*. Correlated features make the estimates *overconfident* (too close to 0/1, double-counting correlated evidence), but the *ranking* of classes usually survives. The independence assumption corrupts the calibration but usually preserves the argmax, so accuracy holds even though the probabilities are garbage.

Decision rule, moved to **log space** (multiplying thousands of small probabilities underflows to zero):

$$\hat y = \arg\max_y\ \Big[\log P(y) + \sum_i \log P(x_i\mid y)\Big]$$

## §4 The ML math: MLE, smoothing, and the variants

"Training" is estimating $P(y)$ and each $P(x_i\mid y)$ by **maximum likelihood estimation**, which for the common cases reduces to counting frequencies — one pass, $O(n\cdot d)$.

**Prior** — MLE of a class probability is its observed frequency:

$$P(y=k) = \frac{\#\{\text{examples in class }k\}}{\#\{\text{total examples}\}}$$

### Multinomial NB (counts) and the zero-frequency catastrophe

$$P(x_i\mid y=k) = \frac{\text{count}(x_i,k)+\alpha}{\sum_j \text{count}(x_j,k)+\alpha\,\lvert V\rvert}$$

Without smoothing, an unseen word has MLE probability exactly 0 — and since the decision is a product (or sum of logs, $\log 0=-\infty$), **one unseen word zeroes out the entire class** no matter how strongly everything else voted. *Laplace (additive) smoothing* adds a pseudocount $\alpha$ (usually 1) to every count, as if each word were seen $\alpha$ times in each class before training. $\lvert V\rvert$ is the vocabulary size; the denominator adds $\alpha$ per possible word so it still sums to 1. The single most important NB implementation detail.

### Gaussian NB (continuous features)

You can't count a real number, so assume each feature is normally distributed within each class and estimate mean and variance by MLE:

$$P(x_i\mid y=k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}}\exp\!\Big(-\frac{(x_i-\mu_{ik})^2}{2\sigma_{ik}^2}\Big)$$

Failure mode: if a feature isn't actually bell-shaped within the class, this density is wrong and Gaussian NB degrades.

### When to use which

| Variant | Feature type | Canonical use |
| --- | --- | --- |
| **Multinomial** | counts / frequencies | text classification with word counts or TF-IDF — the NLP default |
| **Bernoulli** | binary present/absent | short text, where *whether* a word appears matters more than how often; penalizes absent words too |
| **Gaussian** | continuous real-valued | numeric features roughly normal within each class |
| **Complement** | counts, imbalanced | imbalanced text; usually beats Multinomial when classes are skewed |

Picking the variant = matching the likelihood model to your feature type. That choice *is* the modeling decision; there's nothing like a kernel because NB never computes similarities between points — it estimates per-class distributions.

## §5 When does Naive Bayes survive the LLM era?

In a pure accuracy contest it loses to embeddings-plus-a-classifier almost everywhere. It survives where **constraints favor a near-free model**:

- **Brutal latency / volume / cost floors** — spam and abuse filtering at email-provider scale: billions of messages, microsecond budgets, no GPU. NB is a dot-product-free lookup of precomputed log-probabilities. An embedding per message at that volume is economically absurd.
- **The "beat this before you spend money" baseline** — TF-IDF + NB in thirty seconds. If it hits 92%, the expensive approach must justify the gap. Its most durable role; reaching for it signals maturity.
- **Tiny labeled data** — with dozens-to-hundreds of examples, NB's strong bias (the independence assumption is a regularizer) often *beats* a fancier model that overfits.
- **Edge / offline / no-infra** — a few KB of count tables; runs where a transformer can't deploy.
- **Interpretability/audit** — point at exact per-word log-probabilities that drove a decision.

**Genuinely finished** wherever *meaning* matters and you have data + compute. NB sees "not good" as {not, good} voting independently — it structurally cannot grasp negation, sarcasm, word order, or context. Embeddings capture exactly that.

!!! note "Interview framing"
    "Naive Bayes survives as a cost/latency play and as the baseline you must beat — high-volume spam filtering, tiny-data cold starts, edge devices, the thirty-second sanity check before anyone spends on embeddings or LLMs. It's never chosen for accuracy anymore; it's chosen when compute, data, or latency constraints make a near-free, transparent model rational. The moment semantics and context matter and you have data, embeddings replace it — because the independence assumption that makes it cheap is also what makes it semantically blind."

!!! note "The meta-point"
    The survival of classical models in the LLM era is almost never about *capability*. It's about cost, latency, data scarcity, interpretability, and deployment constraints. That lens applies to SVM, NB, logistic regression — all of them.

## Interview questions

**Q1: Why is Naive Bayes called generative, and how does it classify?**
It is generative because it models \(P(\mathbf x\mid y)\), what each class's data looks like, plus the prior \(P(y)\), rather than learning a decision boundary directly like discriminative models. It then uses Bayes' theorem to flip these into the posterior \(P(y\mid\mathbf x)\) and picks the largest. Because the evidence \(P(\mathbf x)\) is constant across classes, it drops out and prediction is \(\arg\max_y P(\mathbf x\mid y)P(y)\).

**Q2: What is the naive assumption and why does it work despite being false?**
It assumes features are conditionally independent given the class, which factorizes the intractable joint likelihood into a product of easy per-feature estimates. Real features are correlated, so the assumption is false and makes the probability estimates overconfident by double-counting correlated evidence. But classification needs only the correct argmax, not calibrated probabilities, and the ranking of classes usually survives, so accuracy holds.

**Q3: What is the zero-frequency problem and how is it fixed?**
If a feature value never appears in a class during training, its MLE probability is exactly zero, and since the decision multiplies probabilities, that single zero wipes out the entire class no matter how strongly the other features voted. Laplace smoothing fixes it by adding a pseudocount alpha to every count, as if each value were seen alpha times per class, with the denominator adding alpha per vocabulary item so it still normalizes. It is the most important NB implementation detail.

**Q4: How do you choose among the Naive Bayes variants?**
Match the likelihood model to the feature type: Multinomial for counts or frequencies such as word counts and TF-IDF, Bernoulli for binary present-or-absent features in short text, Gaussian for continuous features assumed normal within each class, and Complement for imbalanced text. That choice is the entire modeling decision, since Naive Bayes estimates per-class distributions rather than computing similarities between points.
