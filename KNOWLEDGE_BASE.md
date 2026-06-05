# Knowledge Base — Master Index for AI Agents

> **What this file is.** A single, machine-readable map of everything in this repository's
> seven study sites: every **topic → section → page**, what each page covers, the exact
> concepts inside it, and the precise file path to cite. It exists so an AI agent (Codex,
> Claude Code, or any custom agent) can **plan a retrieval before reading**, jump straight to
> the right page, and **attribute every claim** to a source. Read this file first; then open
> only the pages you actually need.

This is a **knowledge base for knowledge tasks** (explaining, comparing, deriving, interview
prep), not a codebase guide. The prose lives in Markdown under each `<Topic>/docs/**`. The
built websites are just a rendered view of the same Markdown — **always read the `.md` source,
never the generated `*/site/*.html`.**

---

## How an agent should use this file

1. **Plan from the index.** Read the topic map and the relevant section overviews below to
   decide *which* pages answer the question. Don't grep blindly — the section overviews tell
   you where a concept lives and what its prerequisites are.
2. **Open the page(s) by path.** Each page entry gives its exact path (e.g.
   `LLM/docs/alignment/dpo.md`) and its key concepts (the page's `##` section headings). Match
   the question to a concept, open that page, and read the matching `##` section.
3. **Quote with a citation.** When you use something, cite the **path** (and heading if
   relevant). See the attribution protocol below — it is mandatory.
4. **Stay in scope.** This corpus is *curated interview-prep notes*. It is deep but not
   exhaustive, reflects the author's framing, and some "latest landscape" material ages. It is
   not ground truth — cite it as "our knowledge base," not as authority.

---

## Attribution protocol (mandatory) ⚠️

Any agent using this repository as a knowledge base **must keep a hard boundary between what
came from this knowledge base and what came from anywhere else.** This lets the reader instantly
tell "we already have notes on this" from "this is new and may need study."

- **From the knowledge base →** explicitly attribute it. Quote or paraphrase, then mark it,
  e.g.: *"(from our knowledge base: `LLM/docs/alignment/dpo.md` → §"The DPO loss")."* Prefer
  citing the **path** and, when useful, the `##` **section heading**.
- **From anywhere else →** explicitly flag it as external. If a claim comes from web search,
  the model's own reasoning/training, a tool result, or general knowledge, say so, e.g.:
  *"(not in our knowledge base — sourced from web search)"* or *"(not in our knowledge base —
  my own reasoning)."*
- **When you extend a KB idea with outside information,** split the sentence: state the KB part
  with its citation, then the added part with its external flag. Never silently blend them.
- **If you cannot find support in the KB, say so** ("our knowledge base does not cover this")
  rather than implying it does.

The goal is a clean, auditable line: **KB-sourced** content is content we already understand
and can trust at our own level; **externally-sourced** content is flagged so we know to verify
or study it.

---

## Citation & path conventions

- **Source of truth:** the Markdown at `<Topic>/docs/<section>/<page>.md`. Cite this path.
- **Live URL mapping (for humans):** a page at `<Topic>/docs/<section>/<page>.md` is published
  at `https://s-samarth.github.io/DataSciencePreparation/<Topic>/<section>/<page>/`. A `##`
  heading becomes a URL anchor by lowercasing and hyphenating its text
  (e.g. `## The p-value` → `#the-p-value`).
- **Section index pages** are `<section>/index.md` — they hold the section overview and the
  mental model for that group of pages. Read them when planning.
- **"Key concepts"** listed per page below are that page's literal `##` headings — they are the
  most reliable retrieval anchors and the best target for an in-page jump.

---

## Topic map

| # | Topic | Site folder | One-line scope |
|---|-------|-------------|----------------|
| 1 | **Mathematics** | `Mathematics/` | Probability, inferential statistics, and linear algebra for ML |
| 2 | **Classical ML** | `ClassicalML/` | Linear models, trees, ensembles, unsupervised learning, evaluation metrics |
| 3 | **Deep Learning** | `DeepLearning/` | NN mechanics, gradient flow, optimization, losses, CNNs, vision/generative, sequences & attention |
| 4 | **LLM Study Notes** | `LLM/` | Transformer internals, training, alignment, PEFT, inference, serving, 2026 landscape |
| 5 | **Agentic AI** | `AgenticAI/` | RAG, agent loops, LangGraph, protocols (MCP/A2A), eval, serving, agent system design |
| 6 | **ML Case Studies** | `MLCaseStudies/` | 20 worked ML/GenAI system-design case studies with interview transcripts |
| 7 | **Productionizing ML** | `ProductionizingML/` | Production ML system design, data/training/serving, the production loop, infra, implementation |

> A separate `RawNotes/` folder holds original source material (HTML/notebooks). It is **not**
> part of this index and is **not** a retrieval target — ignore it for knowledge tasks.

---

## 1. Mathematics

**Site:** `Mathematics/` · **Scope:** the mathematics behind ML interviews, in three tracks —
probability, inferential statistics, and linear algebra. Every page leads with intuition, keeps
the formalism beside it, and ends with an **Interview Questions** block (so each page also
doubles as a question bank for its topic).
**Retrieve from here when** the question is about probability theory, distributions, estimators,
hypothesis tests, or the linear-algebra machinery (rank, eigen, SVD, matrix calculus) under
regression/PCA/optimization. **Prerequisite arrows:** Foundations → Random Variables → Advanced
Probability → (Cheat Sheet, Estimation); Estimation → Testing; Linear Algebra supports
Estimation and Advanced Probability.

### 1.1 Probability Foundations — `Mathematics/docs/foundations/`
Counting, the language of events, conditioning, and the classic paradoxes. The base everything
else stands on: how to count outcomes, translate word problems into set/probability language,
condition correctly, and invert with Bayes.
- **Overview** — `foundations/index.md` — what the section covers and how the ideas connect.
- **Counting & the Sampling Table** — `foundations/counting.md` — *concepts:* the sampling table (how many ways) · Vandermonde's identity · inclusion–exclusion principle.
- **Events, Independence & Conditioning** — `foundations/events-conditioning.md` — *concepts:* translating English into set language · independence of events · conditional probability · multiplication & chain rule · law of total probability.
- **Bayes' Theorem** — `foundations/bayes.md` — *concepts:* Bayes' theorem · conditional independence.
- **Paradoxes & Random Walks** — `foundations/paradoxes-walks.md` — *concepts:* Monty Hall · Simpson's paradox · random walk & gambler's ruin · walks in 1D/2D/3D and beyond.

### 1.2 Random Variables & Distributions — `Mathematics/docs/distributions/`
From a single idea (the random variable) to the discrete and continuous families, with PMF/PDF,
CDF, expectation, and variance machinery.
- **Overview** — `distributions/index.md` — the discrete family as one idea; what the section covers.
- **Random Variables & Discrete Families** — `distributions/random-variables.md` — *concepts:* random variables & random processes · Bernoulli · Binomial · PMF · indicator random variables · Hypergeometric.
- **CDF, Expectation & Variance** — `distributions/cdf-expectation-variance.md` — *concepts:* cumulative distribution function · expected value & its properties · variance & standard deviation.
- **More Discrete Distributions** — `distributions/discrete-families.md` — *concepts:* Geometric · Negative Binomial · Poisson · Binomial→Poisson limit · three famous worked examples.
- **Continuous Random Variables** — `distributions/continuous.md` — *concepts:* probability density function · continuous expectation · Uniform · LOTUS (law of the unconscious statistician) · master cheat sheet.

### 1.3 Advanced Probability — `Mathematics/docs/advanced/`
Normal theory, transforms, multivariate structure, limit theorems, and Markov chains — the
bridge from basic probability into inference.
- **Overview** — `advanced/index.md` — what the section covers and the arc of the section.
- **Universality of the Uniform & the Normal** — `advanced/uniform-normal.md` — *concepts:* universality of the uniform · the Normal (Gaussian) distribution.
- **Moment-Generating Functions** — `advanced/mgf.md` — *concepts:* moment-generating functions (MGF) and their uses.
- **Joint Distributions & Covariance** — `advanced/joint-covariance.md` — *concepts:* joint, marginal & conditional distributions · covariance & correlation.
- **Conditional Expectation & Inequalities** — `advanced/conditional-inequalities.md` — *concepts:* conditional expectation · the probability inequalities (Markov, Chebyshev, Jensen, etc.).
- **Law of Large Numbers & CLT** — `advanced/lln-clt.md` — *concepts:* sample statistics & the law of large numbers · the central limit theorem.
- **Markov Chains** — `advanced/markov-chains.md` — *concepts:* Markov chains (states, transitions, stationary behavior).

### 1.4 Distribution Cheat Sheet — `Mathematics/docs/cheatsheet/`
A standalone side-by-side lookup of ~18 distributions plus the "big ideas" that connect them —
the fastest reference ten minutes before a screen.
- **Overview** — `cheatsheet/index.md` — what the section covers and how the distributions connect.
- **Discrete Distributions** — `cheatsheet/discrete.md` — *concepts:* the discrete grid (parameters, PMF, mean/variance per family).
- **Continuous Distributions** — `cheatsheet/continuous.md` — *concepts:* the continuous grid.
- **Big Ideas** — `cheatsheet/big-ideas.md` — *concepts:* PMF vs PDF · memorylessness · conjugate priors · when mean/variance don't exist · mean = variance as a tell · why the Normal is everywhere.

### 1.5 Estimation & Inference — `Mathematics/docs/estimation/`
Turning probability into estimators: population vs sample, likelihood and MLE, bias, confidence
intervals, and OLS viewed as inference.
- **Overview** — `estimation/index.md` — what the section covers and the reasoning chain.
- **Population, Samples & Estimators** — `estimation/population-estimators.md` — *concepts:* population vs sample & the estimation problem · point estimator vs point estimate.
- **Likelihood & MLE** — `estimation/likelihood-mle.md` — *concepts:* probability vs likelihood · why θ, why likelihood, why maximize, why it becomes the loss · worked MLE problems.
- **Bias & Unbiased Estimators** — `estimation/bias.md` — *concepts:* biased vs unbiased estimators.
- **Confidence Intervals** — `estimation/confidence-intervals.md` — *concepts:* CIs for one/two means · CIs for variance & proportions · the four-case estimation reference.
- **OLS as Inference** — `estimation/ols-inference.md` — *concepts:* OLS through the lens of estimation & inference · the whole thread in one page.

### 1.6 Hypothesis Testing & Bayesian Inference — `Mathematics/docs/testing/`
The decision pipeline: hypotheses, p-values, error types and power, the standard test families
(t/z/χ²/F, ANOVA, non-parametric), Bayesian estimation/MAP, and practical inference concepts.
- **Overview** — `testing/index.md` — what the section covers and the decision pipeline.
- **Hypothesis Testing & p-values** — `testing/hypothesis-pvalues.md` — *concepts:* hypothesis testing · the p-value.
- **Errors, FDR & Power** — `testing/errors-power.md` — *concepts:* Type I/II errors, FDR, precision/recall · statistical power & power analysis.
- **t, z, Chi-Squared & F Tests** — `testing/t-z-chi-f.md` — *concepts:* t-distribution & t-test · z-test and z-vs-t · chi-squared distribution & tests · F-distribution.
- **Variance Tests, ANOVA & Non-Parametric** — `testing/variance-anova-nonparam.md` — *concepts:* tests for variances & ANOVA · order statistics · Wilcoxon & Kolmogorov–Smirnov.
- **Bayesian Estimation & MAP** — `testing/bayesian-map.md` — *concepts:* Bayesian estimation & MAP.
- **Practical Concepts** — `testing/practical-concepts.md` — *concepts:* sample size vs effective sample size · SD vs SE · bootstrapping · population vs sample variance (why n−1) · odds & log-odds · CLT & LLN in inference.

### 1.7 Linear Algebra for ML — `Mathematics/docs/linear-algebra/`
The computational language under regression, covariance, PCA, and optimization: matrices as
maps, eigenstructure, rank, special/definite matrices, decompositions, and matrix calculus.
- **Overview** — `linear-algebra/index.md` — what the section covers and the single thread.
- **Matrices as Machines** — `linear-algebra/matrices-machines.md` — *concepts:* matrices as machines (linear maps) · matrix multiplication, both methods.
- **Eigenvalues, Pivots & Rank** — `linear-algebra/eigen-rank.md` — *concepts:* eigenvalues & eigenvectors · pivots · rank & low rank.
- **Special Matrices & Definiteness** — `linear-algebra/special-definite.md` — *concepts:* special matrices · positive definite & positive semidefinite.
- **Decompositions (LU, Eigen, SVD)** — `linear-algebra/decompositions.md` — *concepts:* why decompose at all · LU · diagonalization / eigenvalue decomposition · singular value decomposition.
- **Basis, Inverse & Matrix Calculus** — `linear-algebra/basis-inverse-calculus.md` — *concepts:* basis · inverse & invertibility · pseudo-inverse · null space · matrix calculus cheat sheet · the single thread.

---

## 2. Classical ML

**Site:** `ClassicalML/` · **Scope:** classical (pre-deep-learning) machine learning as one
continuous arc — linear models → information theory & single trees → tree ensembles →
unsupervised learning → evaluation metrics. Every page has a **Rapid Recall** callout (the
compressed version for the 10 minutes before a screen), the full derivation beneath it,
diagrams, and an **Interview Questions** block. Many pages also include a "**2026 relevance**"
take on whether the method still matters.
**Retrieve from here when** the question is about a named classical algorithm (SVM, Naive
Bayes, KNN, K-Means, PCA, decision trees, Random Forest, AdaBoost, GBM, XGBoost, DBSCAN, GMM,
t-SNE/UMAP, TF-IDF/LSA, Isolation Forest), regularization, the information theory behind splits,
or **any evaluation metric** (classification/regression/ranking/LLM/vision/RL/clustering/drift).
**Prerequisite arrows:** Linear Models → Core Algorithms → Info & Trees → Ensembles; Core →
Unsupervised; Ensembles + Unsupervised → Evaluation.

### 2.1 Foundations & Linear Models — `ClassicalML/docs/foundations/`
The supervised optimization and probabilistic backbone: least squares, gradient descent,
regularization, logistic regression/perceptron, GLMs, softmax, and generative (GDA) models.
- **Overview** — `foundations/index.md` — what each page covers; the whole arc in one paragraph.
- **Linear Regression** — `foundations/linear-regression.md` — *concepts:* the core idea · model/loss/closed form · the normal equation via traces (CS229) · probabilistic view (why squared error) · locally weighted regression (LWR).
- **Gradient Descent & the LMS Rule** — `foundations/gradient-descent.md` — *concepts:* gradient descent and its three flavors · the LMS / Widrow–Hoff rule · convexity, the Hessian, positive semi-definite.
- **Assumptions & Diagnostics** — `foundations/assumptions-diagnostics.md` — *concepts:* assumptions and failure modes · normality and the Q–Q plot.
- **Regularization** — `foundations/regularization.md` — *concepts:* why regularize · why weights explode when p > n (L1/L2 intuition).
- **Logistic Regression & Perceptron** — `foundations/logistic-perceptron.md` — *concepts:* logistic regression · Newton's method · the perceptron.
- **GLMs, Softmax & Generative** — `foundations/glms-generative.md` — *concepts:* generalized linear models (the keystone) · softmax regression (multiclass) · generative learning & GDA · GDA vs logistic regression.

### 2.2 Core Classical Algorithms — `ClassicalML/docs/core-algorithms/`
The five canonical algorithms every interview probes — SVM, Naive Bayes, KNN, K-Means, PCA —
each with mechanics, the key math, when to use/ignore, and a 2026 relevance check.
- **Overview** — `core-algorithms/index.md` — what each page covers · comparison matrix · the connecting thread.
- **Support Vector Machines** — `core-algorithms/svm.md` — *concepts:* what an SVM is · margin & the scaling fix · the dual & support vectors · the kernel trick · soft margin and C · why RBF (infinite polynomial space) isn't always best · SVMs in 2026.
- **Naive Bayes** — `core-algorithms/naive-bayes.md` — *concepts:* what it is · Bayes' theorem & total probability · the naive assumption · MLE, smoothing, and the variants · surviving the LLM era.
- **KNN & Curse of Dimensionality** — `core-algorithms/knn.md` — *concepts:* what it is · step-by-step · K as the bias–variance dial · KNN in 2026 · the curse of dimensionality · KD-trees and why they suffer the same curse.
- **K-Means Clustering** — `core-algorithms/kmeans.md` — *concepts:* what it is · why alternating works / what EM means · why init matters & K-Means++ · choosing K (elbow, silhouette) · when to use/ignore + 2026.
- **Principal Component Analysis** — `core-algorithms/pca.md` — *concepts:* matrix decomposition · eigenvalue decomposition · SVD and the bridge to eigendecomposition · the PCA steps · the "Matryoshka" instinct (where it's right/breaks) · 2026 relevance.

### 2.3 Information Theory & Decision Trees — `ClassicalML/docs/info-trees/`
The entropy/impurity foundation under splits, and how a single tree is actually built.
- **Overview** — `info-trees/index.md` — what each page covers · one-page formula strip.
- **Information & Entropy** — `info-trees/information-entropy.md` — *concepts:* what information is · surprise & information · why the entropy formula is what it is · entropy definition/shape/units.
- **Cross-Entropy, KL & Log-Loss** — `info-trees/cross-entropy-kl.md` — *concepts:* cross-entropy · KL divergence · MLE, log-loss, "are we optimizing entropy?" · knowledge-distillation (KD) loss.
- **Impurity & Splitting Criteria** — `info-trees/impurity-splitting.md` — *concepts:* Gini impurity & its link to entropy · information gain & Gini gain.
- **Building Decision Trees** — `info-trees/building-trees.md` — *concepts:* how a tree is built · splitting continuous vs categorical · stopping, pruning, leaf predictions · classification → regression trees · training/inference complexity.
- **Tree Algorithms & Categorical Tricks** — `info-trees/tree-algorithms.md` — *concepts:* ID3 / C4.5 / CART and library reality · categorical tricks (Breiman, CatBoost).

### 2.4 Tree Ensembles — `ClassicalML/docs/ensembles/`
From a single overfitting tree to bagging, random forests, boosting, GBM, and modern gradient
boosting libraries — plus when trees beat (or lose to) everything else.
- **Overview** — `ensembles/index.md` — what each page covers · one-page formula strip.
- **Variance, Bagging & OOB** — `ensembles/bagging-oob.md` — *concepts:* why a single tree overfits · why averaging reduces variance · bagging & bootstrapping (OOB).
- **Random Forests** — `ensembles/random-forests.md` — *concepts:* RF = bagging + feature randomness · RF hyperparameters.
- **Boosting & AdaBoost** — `ensembles/boosting-adaboost.md` — *concepts:* boosting's core contrast · AdaBoost step by step · AdaBoost intricacies · multiclass & regression.
- **Gradient Boosting** — `ensembles/gradient-boosting.md` — *concepts:* residuals → gradients · GBM regressor & classifier · how GBM predicts · tuning/assumptions/weaknesses · Newton's method.
- **XGBoost & Modern Libraries** — `ensembles/xgboost-libraries.md` — *concepts:* regularized second-order objective · parallelization, GPU, missing values · XGBoost vs LightGBM vs CatBoost · the new hyperparameters.
- **Trees vs Other Models** — `ensembles/trees-vs-others.md` — *concepts:* shared boosting weaknesses · when trees dominate vs when others win · preprocessing: trees vs classical models.

### 2.5 Unsupervised, Text & Anomaly — `ClassicalML/docs/unsupervised/`
Clustering beyond K-Means, mixtures and EM, embeddings/visualization, text vectorization, and
anomaly detection — with a "which method when" decision guide.
- **Overview** — `unsupervised/index.md` — what each page covers · master cheat-sheet · the connective tissue.
- **Density & Hierarchical Clustering** — `unsupervised/density-hierarchical.md` — *concepts:* DBSCAN · hierarchical clustering.
- **Gaussian Mixtures & EM** — `unsupervised/gmm-em.md` — *concepts:* the generative model · fitting via EM (specific case) · Expectation–Maximization (general algorithm).
- **t-SNE & UMAP** — `unsupervised/tsne-umap.md` — *concepts:* t-SNE · UMAP.
- **Text Vectorization** — `unsupervised/text-vectorization.md` — *concepts:* TF-IDF · Truncated SVD and LSA.
- **Anomaly Detection** — `unsupervised/anomaly-detection.md` — *concepts:* how Isolation Forest works · why it is NOT a decision tree / Random Forest · complexity.
- **Choosing a Method** — `unsupervised/choosing-method.md` — *concepts:* clustering — which one when · dimensionality reduction — PCA vs t-SNE vs UMAP.

### 2.6 Evaluation Metrics — `ClassicalML/docs/evaluation/`
The cross-cutting scoring layer for every model type. The most reusable reference in the repo
for "which metric, and what does it actually measure."
- **Overview** — `evaluation/index.md` — how to use the section · which metric for which task · interview power moves · rapid-fire reflexes.
- **Classification & Threshold** — `evaluation/classification.md` — *concepts:* confusion matrix · accuracy · precision · recall · F1 · AUC-ROC · AUC-PR (under imbalance) · log loss · specificity · MCC · comparison table.
- **Regression Metrics** — `evaluation/regression.md` — *concepts:* MSE · RMSE · MAE · MAPE · R² · Huber loss.
- **Ranking & Recommendation** — `evaluation/ranking.md` — *concepts:* NDCG · Precision@k, AP, MAP · MRR · online/business metrics.
- **LLM & Generative AI** — `evaluation/llm-generative.md` — *concepts:* perplexity (effective branching) · ROUGE · BERTScore · BLEU · RAG metrics (RAGAS) · human & advanced evaluation.
- **Vision & RL Metrics** — `evaluation/vision-rl.md` — *concepts:* object detection · image classification · segmentation · image generation (GANs/diffusion) · RL metrics.
- **Clustering, Drift & Explainability** — `evaluation/clustering-drift.md` — *concepts:* clustering metrics · data-drift metrics · explainability (SHAP & Shapley values).

---

## 3. Deep Learning

**Site:** `DeepLearning/` · **Scope:** a math-first reference for neural-network mechanics and
the architectures built on them. The mental model: every technique fits one of **five buckets** —
predicting (forward pass), assigning blame (backprop), taking the step (optimizers), staying
stable (norm/clipping/residuals/warmup), and not overfitting (dropout/weight decay/early
stopping/label smoothing). A recurring theme is the **gradient highway** (LSTM cell state through
time, residual connection through depth), and the unifying idea that **almost every loss is the
negative log-likelihood of an assumed distribution**. Most pages have a Rapid Recall + Interview
Questions; several include runnable PyTorch.
**Retrieve from here when** the question is about how networks compute/train (forward pass,
backprop, the training loop/pipeline), gradient flow (activations, vanishing/exploding,
residuals/LSTM), optimizers (SGD→AdamW, Lion/LAMB/Sophia/Shampoo), normalization &
regularization, the **loss-function taxonomy**, CNNs, vision/generative models (GANs, diffusion,
segmentation/detection), or the pre-Transformer sequence arc (tokenization, embeddings, RNNs,
attention build-up). **Note:** attention is built *up to* the Transformer here; full Transformer
internals live in Topic 4 (LLM).

### 3.1 Foundations — `DeepLearning/docs/foundations/`
What a network computes and how it learns: the forward pass, why activations matter, the
training loop, backprop with full local rules, a complete worked 2-layer derivation, and an
end-to-end PyTorch training pipeline.
- **Overview** — `foundations/index.md` — *concepts:* what a neural network is mechanically · the five buckets.
- **Forward Pass** — `foundations/forward-pass.md` — *concepts:* what a linear layer does · why activations are essential (the proof) · complete forward pass of a 2-layer classifier.
- **The Training Loop** — `foundations/training-loop.md` — *concepts:* the six steps · train mode vs eval mode.
- **Backpropagation** — `foundations/backpropagation.md` — *concepts:* the problem · local rules for each operation.
- **Full Backprop Derivation** — `foundations/backprop-derivation.md` — *concepts:* step-by-step gradients (∂L/∂z₂, W₂, b₂, a₁, through ReLU, W₁/b₁) · weight update · numerical verification.
- **The Training Pipeline** — `foundations/training-pipeline.md` — *concepts:* synthetic data · split before normalization · Dataset class · DataLoaders · model def · loss/optimizer/scheduler · training loop with shape tracking · final eval · saving/serving · tensor-shape & complexity references.

### 3.2 Gradient Flow — `DeepLearning/docs/gradient-flow/`
Why gradients vanish or explode and every fix for it — activations, clipping, and the gradient
highways.
- **Overview** — `gradient-flow/index.md` — *concepts:* the core problem · the fixes mapped to pages.
- **Activation Functions** — `gradient-flow/activation-functions.md` — *concepts:* Sigmoid · Tanh · ReLU · LeakyReLU · GELU · SiLU/Swish · comparison.
- **Vanishing Gradients** — `gradient-flow/vanishing-gradients.md` — *concepts:* the core equation · the sigmoid disaster (numerically) · why it's worse in RNNs · the three fixes.
- **Exploding Gradients & Clipping** — `gradient-flow/exploding-gradients.md` — *concepts:* numerical demonstration · gradient clipping · when to use it.
- **Gradient Highways** — `gradient-flow/gradient-highways.md` — *concepts:* the LSTM cell · residual connections.

### 3.3 Optimization — `DeepLearning/docs/optimization/`
Turning gradients into updates: the batch-size spectrum, SGD/momentum, the adaptive family, and
modern optimizers.
- **Overview** — `optimization/index.md` — *concepts:* the batch-size spectrum · the optimizer family.
- **SGD & Momentum** — `optimization/sgd-and-momentum.md` — *concepts:* vanilla SGD · SGD with momentum.
- **Adaptive Methods** — `optimization/adaptive-methods.md` — *concepts:* AdaGrad · RMSProp · Adam · AdamW.
- **Modern Optimizers** — `optimization/modern-optimizers.md` — *concepts:* Lion · LAMB · Sophia (second-order) · Shampoo/Distributed Shampoo · hyperparameter cheat sheet · decision tree.

### 3.4 Normalization & Regularization — `DeepLearning/docs/regularization/`
Everything that keeps training stable and generalizing: the norm family, classic regularizers,
and LR schedules.
- **Overview** — `regularization/index.md` — *concepts:* what is in this section.
- **Normalization** — `regularization/normalization.md` — *concepts:* BatchNorm · LayerNorm · RMSNorm · the "norm" comparison.
- **Regularization** — `regularization/regularization.md` — *concepts:* weight decay (L2) · dropout · early stopping · label smoothing.
- **Learning Rate Schedules** — `regularization/lr-schedules.md` — *concepts:* step decay · cosine annealing · warmup+cosine (LLM standard) · ReduceLROnPlateau · cyclical/one-cycle · recommendation matrix · PyTorch implementation.

### 3.5 Loss Functions — `DeepLearning/docs/losses/`
The full loss taxonomy under one frame: every loss as the negative log-likelihood of an assumed
distribution, across regression, classification, ranking/metric learning, and LLM training.
- **Overview** — `losses/index.md` — *concepts:* the big picture · the unifying (MLE) frame · the distribution-to-loss mapping · the families.
- **Regression Losses** — `losses/regression-losses.md` — *concepts:* MSE/L2 · MAE/L1 · Huber (Smooth L1) · Quantile (Pinball).
- **Classification Losses** — `losses/classification-losses.md` — *concepts:* BCE/log loss · categorical/softmax cross-entropy · focal loss · hinge · label-smoothing CE · KL divergence · Jensen–Shannon divergence.
- **Ranking & Metric Learning** — `losses/ranking-losses.md` — *concepts:* triplet loss · contrastive (pairwise) · InfoNCE/NT-Xent · pairwise ranking (LambdaRank, RankNet).
- **LLM & Generation Losses** — `losses/llm-losses.md` — *concepts:* next-token cross-entropy · masked LM loss · RLHF/PPO losses · DPO · GRPO.

### 3.6 Convolutional Networks — `DeepLearning/docs/cnns/`
Why CNNs exist, the mechanics of convolution, the output-size formula and pooling, and the
architectural lineage through ResNet.
- **Overview** — `cnns/index.md` — *concepts:* why CNNs exist at all · what is in this section.
- **Convolution Mechanics** — `cnns/convolution-mechanics.md` — *concepts:* what a convolution is · fully worked numeric example · channels & the weight tensor · kernel sizes & the 1×1 convolution.
- **Output Size, Pooling & Forward Pass** — `cnns/output-pooling-forward.md` — *concepts:* output-size master formula · pooling and the full forward pass.
- **CNN Architectures** — `cnns/architectures.md` — *concepts:* architectural evolution & skip connections · post-ResNet efficiency directions · data augmentation · a CNN in PyTorch.

### 3.7 Vision Tasks & Generative Models — `DeepLearning/docs/vision-generative/`
Beyond classification: dense prediction, and the two generative families.
- **Overview** — `vision-generative/index.md` — *concepts:* what is in this section.
- **Segmentation & Detection** — `vision-generative/segmentation-detection.md` — *concepts:* image segmentation · object detection and YOLO.
- **GANs** — `vision-generative/gans.md` — *concepts:* forger vs detective intuition · the minimax game · the problems with GANs (interview heart).
- **Diffusion Models** — `vision-generative/diffusion.md` — *concepts:* core intuition · forward noising (closed form) · training objective · reverse sampling · the U-Net · text-to-image (DALL·E / Stable Diffusion).

### 3.8 Sequences & Attention — `DeepLearning/docs/sequences/`
The sequence-modeling arc from text representation up to attention — the bridge into the
Transformer (which Topic 4 continues).
- **Overview** — `sequences/index.md` — *concepts:* why sequence modeling, why text · what is in this section.
- **Tokenization & BPE** — `sequences/tokenization-bpe.md` — *concepts:* levels of tokenization · Byte Pair Encoding (greedy merge) · special tokens.
- **Embeddings** — `sequences/embeddings.md` — *concepts:* intuition · Word2Vec · GloVe (count-based) · the static-embedding limitation · sentence embeddings today.
- **RNNs & Gated Cells** — `sequences/rnns-and-gates.md` — *concepts:* sequence modeling & vanilla RNNs · LSTMs and GRUs.
- **Architectures & Attention** — `sequences/architectures-attention.md` — *concepts:* encoder-only / decoder-only / encoder-decoder · attention build-up (stopping before the Transformer) · why this is the bridge to Transformers.

---

## 4. LLM Study Notes

**Site:** `LLM/` · **Scope:** deep dives across the modern LLM stack for AI-engineering / AI-PM
roles, framed for 2026. The arc: Transformer architecture → pretraining → SFT → (PEFT, alignment)
→ inference & serving → 2026 landscape. Every page has a **Rapid Recall** TL;DR, full canonical
prose with H2/H3 anchors, and **Interview Questions** (hard ones flagged as traps). Three reading
paths reuse the same pages: *Build From Scratch* (hands-on PyTorch), *Interview Explainer*
(concept-first), and *HTML Deep-Dive* (longest derivations).
**Retrieve from here when** the question is about Transformer internals (attention, multi-head,
normalization, positional encodings, residual/block structure), pretraining a decoder-only model,
supervised fine-tuning, PEFT (LoRA/QLoRA/NF4/quantization), alignment (reward models,
PPO/RLHF, DPO, GRPO, RLVR, Constitutional AI/RLAIF), inference architecture (prefill/decode,
KV-cache, GQA/MQA/MLA, Flash Attention, MoE), serving (vLLM, SGLang, quantization formats,
speculative decoding, Docker/FastAPI deploy), or the 2026 model/benchmark/pricing landscape.
**This is the canonical Transformer/LLM source** — for pre-Transformer sequence basics see Topic 3.

### 4.1 Foundations (Transformer Architecture) — `LLM/docs/foundations/`
The Transformer from first principles: attention, multi-head, normalization, positional
encodings, the block/residual structure, failure modes, and the non-decoder architecture family.
- **Overview** — `foundations/index.md` — *concepts:* what to read in what order · the five things to never forget.
- **Attention** — `foundations/attention.md` — *concepts:* the Python-dict analogy · why three projections (Q/K/V) · the math with the √d_k derivation · the causal mask · minimum-viable implementation.
- **Multi-Head and Softmax** — `foundations/multi-head-and-softmax.md` — *concepts:* the split is at the dot-product not the weights · softmax can only pick one ranking · compute/memory accounting · why attention is O(N²) not O(N³) · textual visualization of a layer.
- **Normalization** — `foundations/normalization.md` — *concepts:* what normalization does · LayerNorm vs BatchNorm vs RMSNorm · RMSNorm as the 2026 default.
- **Block and Residual** — `foundations/block-and-residual.md` — *concepts:* the block in one line · pre-norm vs post-norm · two sub-blocks per layer · weight tying · scaled init · assembling one block.
- **Positional Encodings** — `foundations/positional-encodings.md` — *concepts:* why positional encoding exists · sinusoidal (2017) · RoPE (rotary) · ALiBi.
- **Failure Modes** — `foundations/failure-modes.md` — *concepts:* gradient flow through a deep stack · softmax attention dilution at long N · assumptions & failure-mode table.
- **Architecture Families** — `foundations/architecture-families.md` — *concepts:* ViT · BERT (encoder-only) · CLIP (contrastive VL) · Whisper (speech seq2seq) · self vs cross attention, bi vs cross encoders · task→architecture decision map · contrastive-learning losses.

### 4.2 Build From Scratch (hands-on) — `LLM/docs/build-from-scratch/`
Pretrain a small decoder-only Transformer, fine-tune it, then align it — all in pure PyTorch on a
free Colab T4. The runnable counterpart to the explainer pages.
- **Overview** — `build-from-scratch/index.md` — *concepts:* why a toy task for alignment · the pipeline · reading order · what you can claim afterward.
- **Pretraining (TinyStories)** — `build-from-scratch/pretraining-tinystories.md` — *concepts:* config dataclass · custom BPE tokenizer · packed sequences + memory-mapped dataloader · sinusoidal AND RoPE · multi-head causal self-attention · RMSNorm + SwiGLU MLP · transformer block & full GPT · AdamW param groups · warmup→cosine schedule · mixed precision · the training loop.
- **SFT Walkthrough** — `build-from-scratch/sft-walkthrough.md` — *concepts:* the five SFT-specific concepts · Dolly-15K · chat-template formatting · loss masking · dynamic padding collator · shared training loop · full SFT on Qwen2.5-0.5B · LoRA SFT on TinyLlama-1.1B · save/reload/merge · `trl.SFTTrainer` shortcut.
- **Alignment Walkthrough** — `build-from-scratch/alignment-walkthrough.md` — *concepts:* the toy task · a ~3M-param transformer · synthetic preference pairs · Method 1 Reward Model (Bradley-Terry) · Method 2 PPO · Method 3 DPO · Method 4 GRPO · Method 4b RLVR · side-by-side comparison.

### 4.3 Post-Training (SFT) — `LLM/docs/sft/`
Supervised fine-tuning: the three-stage pipeline, chat templates, loss masking, and the MLE/MAP
backbone that explains why the loss is what it is.
- **Overview** — `sft/index.md` — *concepts:* pages in this section.
- **Three-Stage Pipeline** — `sft/three-stage-pipeline.md` — *concepts:* the funnel (pretrain→SFT→align) · what each stage does to the model · pretraining briefly · 2026 reality check.
- **Chat Templates** — `sft/chat-templates.md` — *concepts:* two ways to feed instructions · the three mechanical truths · why chat templates won · practical usage.
- **Loss Masking** — `sft/loss-masking.md` — *concepts:* the core idea · implementation · finding the assistant span · two common bugs · verification · packing · the SFT loss.
- **MLE and MAP Backbone** — `sft/mle-map-backbone.md` — *concepts:* probability vs likelihood · worked 10-flips example · the log-likelihood trick & universal recipe · MAP and L2 = Gaussian-prior · where it shows up downstream.

### 4.4 PEFT (LoRA, QLoRA) — `LLM/docs/peft/`
Parameter-efficient fine-tuning end to end: LoRA mechanics, why low-rank works, initialization,
target modules, quantization/NF4, QLoRA assembled, and memory tricks.
- **Overview** — `peft/index.md` — *concepts:* how the pieces connect · pages in this section.
- **LoRA Mechanics** — `peft/lora-mechanics.md` — *concepts:* the sticky-note analogy · the decomposition · parameter saving · forward pass with α/r scaling · merging (free-inference) · LoRA from scratch in 25 lines · hyperparameters · the PEFT library version.
- **Why Low-Rank** — `peft/why-low-rank.md` — *concepts:* rank ≠ sparsity · empirical evidence · redirection not relearning · one-sentence answer · when the assumption breaks.
- **Initialization and Gradient Flow** — `peft/initialization-gradient-flow.md` — *concepts:* the setup · why A can't also be zero · the asymmetry · what the LoRA paper does · steps 1-3.
- **Target Modules** — `peft/target-modules.md` — *concepts:* the anatomy · why Q and V are default · the decision dial · naming conventions across libraries · the PEFT config.
- **Quantization and NF4** — `peft/quantization-nf4.md` — *concepts:* number formats · uniform quantization & info loss · NF4 · when to use what · PTQ vs QAT.
- **QLoRA Assembled** — `peft/qlora-assembled.md` — *concepts:* component table · 70B on one 48 GB GPU · α/r scaling · standard 2026 code · ecosystem notes · what you give up vs full FT.
- **Memory Tricks** — `peft/memory-tricks.md` — *concepts:* paged optimizers · gradient accumulation · gradient checkpointing · the three side by side.

### 4.5 Alignment — `LLM/docs/alignment/`
The alignment stack from reward modeling through the policy-optimization family, with a calibrated
take on the GRPO/R1 hype.
- **Overview** — `alignment/index.md` — *concepts:* the 2026 pipeline · the alignment problem · pages in this section.
- **Reward Models and Bradley-Terry** — `alignment/reward-models-bradley-terry.md` — *concepts:* why a neural net · Bradley-Terry assumption · what the gradient does · shift-invariance trick · pooling strategy · the dimensionality trap · RM training in five lines.
- **RL from Zero and PPO** — `alignment/rl-from-zero-and-ppo.md` — *concepts:* RL from zero · mapping RL onto LLMs · why naive REINFORCE fails · the four models in PPO · PPO's three machineries · one PPO iteration end to end · why RLHF is hard.
- **DPO** — `alignment/dpo.md` — *concepts:* the core insight · RLHF closed-form solution · invert reward in terms of policy · plug into Bradley-Terry · the DPO loss · what it's doing · 10-line implementation · the β hyperparameter · 2026 limitations · DPO variants.
- **GRPO** — `alignment/grpo.md` — *concepts:* the core trick · the advantage formula · the GRPO loss · why it shines for reasoning · where it breaks · implementation skeleton · refinements (Dr. GRPO, DAPO) · hyperparameters.
- **RLVR** — `alignment/rlvr.md` — *concepts:* one-line definition · the trivial code change · why it's the 2026 hype · verifiable domains · failure modes · where it fits in the pipeline.
- **The GRPO Hype, Decoded** — `alignment/grpo-hype-decoded.md` — *concepts:* what R1-Zero proved / did NOT prove · why pure GRPO can't replace SFT or pretraining · the calibrated table · the real revolution · interview-grade phrasing.

### 4.6 Inference Architecture — `LLM/docs/inference-arch/`
The hardware-aware view of inference: the memory hierarchy is the boss, and the four tricks that
follow from it.
- **Overview** — `inference-arch/index.md` — *concepts:* the four-trick summary · the memory hierarchy is the boss · inference's two phases.
- **Prefill vs Decode** — `inference-arch/prefill-vs-decode.md` — *concepts:* the two governing hardware numbers · prefill (compute-bound) · decode (memory-bound) · prefill/decode disaggregation · why the framing matters.
- **KV-Cache** — `inference-arch/kv-cache.md` — *concepts:* why the cache exists · the "2·L·H·D·N·B bytes" formula · worked LLaMA-3 70B example · storage/dimensionality · bandwidth vs capacity · the inference-loop sketch · the ways out.
- **GQA, MQA, MLA** — `inference-arch/gqa-mqa-mla.md` — *concepts:* the MHA→MQA→GQA spectrum · the asymmetric intuition · implementation · the memory math · MLA (DeepSeek) · comparison table · the meta-pattern.
- **Flash Attention** — `inference-arch/flash-attention.md` — *concepts:* what's wasted in naive attention · the one-sentence idea · online softmax rescaling · the tiling picture · three wins · 2026 version landscape · how to use it.
- **Mixture of Experts** — `inference-arch/mixture-of-experts.md` — *concepts:* the tension it solves · origin · it replaces the FFN not attention · routing math · routing collapse & load balancing · fine-grained + shared experts · implementation sketch · train vs inference nuance · why MoE makes sense at scale.
- **Frontier Techniques** — `inference-arch/frontier-techniques.md` — *concepts:* NTK-aware RoPE scaling · CSA+HCA sparse attention at 1M tokens · QAT · multi-token prediction (MTP) · the unifying meta-pattern.

### 4.7 Serving and Optimization — `LLM/docs/serving/`
Production serving: quantization formats, decoding/speculative, the major engines, and a minimal
Docker/FastAPI deploy.
- **Overview** — `serving/index.md` — *concepts:* the fundamental problem · pages in this section · the decision tree.
- **Quantization Formats** — `serving/quantization-formats.md` — *concepts:* the tradeoff dial · key formats · three smart 4-bit methods · 2026 decision tree · library landscape · memory cheat sheet · quality comparison · the trap question.
- **Decoding and Speculative** — `serving/decoding-and-speculative.md` — *concepts:* basic decoding strategies · why generation is slow · speculative decoding core idea · why output distribution is unchanged · four speculative variants · acceptance-rate math · implementation sketch · when it hurts.
- **vLLM** — `serving/vllm.md` — *concepts:* what vLLM is · PagedAttention · continuous batching · tensor parallelism · other features · minimal example.
- **SGLang and Alternatives** — `serving/sglang-and-alternatives.md` — *concepts:* SGLang/RadixAttention · TensorRT-LLM · TGI · Ollama/llama.cpp/GGUF · LMDeploy & the Chinese ecosystem · decision table · disaggregated serving.
- **Docker and FastAPI Deploy** — `serving/docker-fastapi-deploy.md` — *concepts:* the mental model · minimal FastAPI service · the Dockerfile · build & run · production upgrades · 2026 deployment targets · minimal-viable production setup.

### 4.8 2026 Landscape — `LLM/docs/landscape/`
The moving parts of the 2026 ecosystem: models, distillation, scalable alignment, hallucination,
caching/routing, multimodal, voice (India), and benchmarks. (Most time-sensitive section — treat
"latest" claims as as-of-writing.)
- **Overview** — `landscape/index.md` — *concepts:* what this section covers · one unifying framing.
- **Model Landscape** — `landscape/model-landscape.md` — *concepts:* the four axes · closed/proprietary frontier · open-weight frontier · India-specific (Sarvam AI) · the 2026 decision tree · the pricing collapse · April 2026 pricing · quick picks · post-training recipe stack.
- **Distillation** — `landscape/distillation.md` — *concepts:* classic distillation (Hinton 2015) · two flavors · why the classic playbook is dead for frontier APIs · reasoning models · DeepSeek-R1 distillation play · the real attack pipeline · defense layers · the KL framing · legitimate pipelines.
- **Constitutional AI and RLAIF** — `landscape/constitutional-ai-and-rlaif.md` — *concepts:* the simple story · mental model · the two phases · the constitution itself · why it works (and doesn't) · 2026 relevance · RLAIF (general case) · scalable alignment.
- **Hallucination Mitigation** — `landscape/hallucination-mitigation.md` — *concepts:* the fundamental fact · the layered mitigation stack · the stack in practice · industry specifics · interview framing.
- **Caching and Routing** — `landscape/caching-and-routing.md` — *concepts:* semantic caching · prompt cache vs semantic cache · model routing · the combined stack.
- **Multimodal Models** — `landscape/multimodal-models.md` — *concepts:* the simple story · three architectures · training recipe · key design decisions · limitations · the CLIP backbone · for an AI PM in 2026.
- **Voice AI (India)** — `landscape/voice-ai-india.md` — *concepts:* the two architectures · the voice-AI stack in Indian interviews · Indian voice-AI startup landscape · interview framing.
- **Benchmarks 2026** — `landscape/benchmarks-2026.md` — *concepts:* the big picture · the 2026 benchmark cheat sheet · picking benchmarks for your use case · the "always verify" principle · why MMLU died · the contamination problem · the trap answer.

---

## 5. Agentic AI

**Site:** `AgenticAI/` · **Scope:** building and operating agentic systems — retrieval (RAG),
agent loops, LangGraph internals, agent protocols, evaluation, production serving/infra, and
end-to-end agent system design. Strictly scoped to *agentic* AI (LLM internals appear only where
they serve an agentic concept). Every page opens with a **Rapid Recall** callout; many sections
end with a **Layer Checklist** and **Interview Questions**.
**Retrieve from here when** the question is about RAG (chunking, embeddings, vector stores, ANN
algorithms like HNSW/IVF/PQ, hybrid retrieval, reranking, query transformation, advanced/agentic
RAG, GraphRAG), agent design (the loop, ReAct, reflection, plan-and-execute, memory taxonomy,
routing, HITL, multi-agent topologies), **LangGraph** mechanics (Pregel/BSP, state/reducers,
Command/Send, checkpointers, interrupts, streaming), **protocols** (MCP, A2A, coding agents),
agent/RAG **evaluation** (Ragas, trajectory eval, LLM-as-judge, Elo), agent **serving** (cost,
inference stack, serving modes, observability, guardrails), or whiteboard **agent system design**.
**Note:** the Seller Copilot case study was intentionally removed from this site.

### 5.1 RAG & Retrieval — `AgenticAI/docs/rag/`
Retrieval-augmented generation from the naive pipeline to agentic RAG, plus the ANN algorithms
under vector search (including a from-scratch HNSW).
- **Overview** — `rag/index.md` — *concepts:* why RAG exists · two kinds of memory · when to pick each · the four pains RAG removes · the naive pipeline end to end · "search done well + an LLM" · 2026 production default stack.
- **Foundations** — `rag/foundations.md` — *concepts:* document loading · chunking (the most important knob) · embedding models · vector stores · tables in RAG.
- **Retrieval Mechanics** — `rag/retrieval-mechanics.md` — *concepts:* similarity metrics · hybrid retrieval (BM25 + dense, fused with RRF).
- **ANN Algorithms** — `rag/ann-algorithms.md` — *concepts:* the three families · HNSW · IVF · Product Quantization (PQ) · LSH & KD-tree (casualties of high dimensions) · complexity table · FAISS recipes · the decision rule.
- **Reranking** — `rag/reranking.md` — *concepts:* cross-encoder reranking · query rewriting · query transformation (HyDE, multi-query, step-back, decomposition).
- **Advanced RAG** — `rag/advanced-rag.md` — *concepts:* self-querying retrievers & contextual compression · agentic RAG (CRAG, Self-RAG, Adaptive) · GraphRAG & hierarchical summarization · multimodal RAG · how citations work · pipeline-as-loop · web search / deep research as RAG · long-context vs RAG.
- **HNSW From Scratch** — `rag/hnsw-from-scratch.md` — *concepts:* simplifications vs the paper · function-call graph · the full implementation · interview takeaways.

### 5.2 Agents & Orchestration — `AgenticAI/docs/agents/`
The agent loop and the design space around it: tools, memory, routing, planning, HITL, and
multi-agent topologies.
- **Overview** — `agents/index.md` — *concepts:* the agent loop · the agent-design decision tree · section guide · layer checklist · numbers worth remembering.
- **Foundations** — `agents/foundations.md` — *concepts:* from chatbots to agents (the loop) · plan-and-execute & reflection · tool design (schemas, descriptions, parallel calls) · tool failures & retry logic.
- **Memory, Routing, Planning, HITL** — `agents/memory-routing-planning.md` — *concepts:* memory taxonomy (working/short/long/episodic/semantic/procedural) · consolidation, retention, privacy isolation · routing (model/tool/agent) · planning (write_todos as context engineering) · human-in-the-loop with LangGraph interrupts.
- **Sync, Async & Multi-Turn State** — `agents/sync-async-state.md` — *concepts:* sync vs async (when to reach for each) · multi-turn conversations & keeping state across turns.
- **Multi-Agent Topologies** — `agents/multi-agent-topologies.md` — *concepts:* why multi-agent · supervisor pattern · swarm (peer handoffs) · hierarchical & scatter-gather + Anthropic's research system · DeepAgents harness · handoffs (state transfer, context compression, custom tools).

### 5.3 LangGraph — `AgenticAI/docs/langgraph/`
LangGraph internals from the Pregel/BSP engine up to multi-agent subgraphs — the code-forward
view, with a customizability ladder and primitive reference.
- **Overview** — `langgraph/index.md` — *concepts:* the complete picture · the Pregel/BSP loop · the customizability ladder · primitive reference · 2026 versions · framework comparison · layer checklist.
- **Graph & Pregel Model** — `langgraph/graph-pregel.md` — *concepts:* what a "graph" is in LangGraph · the Pregel/BSP engine · reading any LangGraph codebase with confidence.
- **State & Reducers** — `langgraph/state-reducers.md` — *concepts:* state, channels, three ways to define state · reducers (how updates merge) · nodes in depth.
- **Control Flow (Edges, Command, Send)** — `langgraph/control-flow.md` — *concepts:* edges (normal, conditional, the agent loop) · `Command` and `Send` primitives · the Functional API.
- **ReAct & create_agent** — `langgraph/react-create-agent.md` — *concepts:* a ReAct agent from scratch (no abstractions) · the prebuilt `create_agent`, tools, and middleware.
- **Persistence, Interrupts, Streaming** — `langgraph/persistence-streaming.md` — *concepts:* checkpointers and the store · human-in-the-loop `interrupt`/resume · streaming and time travel.
- **CRAG & Multi-Agent Subgraphs** — `langgraph/crag-multiagent.md` — *concepts:* RAG as a graph (naive → agentic CRAG) · multi-agent networks in LangGraph.

### 5.4 Protocols & Coding Agents — `AgenticAI/docs/protocols/`
The interop layer: MCP (tool/host portability), A2A (agent-to-agent), and what makes coding
agents structurally different.
- **Overview** — `protocols/index.md` — *concepts:* the N+M insight · two protocols, two scopes · section guide · layer checklist.
- **MCP** — `protocols/mcp.md` — *concepts:* the host-portability problem · MCP architecture (host/client/server, three primitives) · transport, lifecycle, OAuth 2.1, a minimal server · MCP vs REST/GraphQL · failure modes · context bloat (MCP vs CLI) · registry & discovery · when to build one.
- **A2A** — `protocols/a2a.md` — *concepts:* A2A v1.0 (agent cards, task lifecycle, signed discovery) · MCP vs A2A vs function calling (decision matrix & composition) · adoption check.
- **Coding Agents** — `protocols/coding-agents.md` — *concepts:* what makes coding agents different · the four architectures · the toolkit · filesystem as working memory · failure modes · evaluation for coding agents.

### 5.5 Evaluation & Monitoring — `AgenticAI/docs/eval/`
Evaluating RAG and agents where "it looks right" isn't enough: metrics, trajectory eval,
LLM-as-judge, and dataset design.
- **Overview** — `eval/index.md` — *concepts:* the eval landscape · why eval matters more than metrics · section guide · layer checklist.
- **RAG Eval** — `eval/rag-eval.md` — *concepts:* the eval problem · Ragas (the four metrics) · DeepEval/TruLens/Phoenix · standard offline metrics · ranking metrics (MRR, nDCG@k) · BLEU/ROUGE/beam · regression test set & CI for RAG.
- **Agent Eval** — `eval/agent-eval.md` — *concepts:* why agent eval is hard · agentic functional correctness · 2026 benchmarks & gaming · trajectory evaluation · tool-call eval (right tool/args/order) · parallel vs sequential · LLM-as-judge for trajectories · Elo · LangSmith + agentevals · online evaluators.
- **Dataset Design** — `eval/dataset-design.md` — *concepts:* dataset design · composition checklist for a new eval set.
- **LLM-as-Judge** — `eval/llm-as-judge.md` — *concepts:* LLM-as-judge · hallucination rate · groundedness scoring · the labelling pyramid · three production realities of judges · when it's the wrong tool.

### 5.6 Serving, Infra & Guardrails — `AgenticAI/docs/serving/`
Running agents in production: token cost, the inference stack, execution modes, observability,
and guardrails/safety.
- **Overview** — `serving/index.md` — *concepts:* the production stack at a glance · section guide · layer checklist.
- **Cost & Token Economics** — `serving/cost.md` — *concepts:* token economics · the seven levers · quantization for self-hosted · cost dashboard · output-token math · caching for consistency.
- **Inference Stack** — `serving/inference-stack.md` — *concepts:* prefill vs decode · throughput vs latency & batching · continuous batching · vLLM · parallelism strategies · alternative engines · autoscaling · 2026 hardware cost benchmark · inference optimization stack.
- **Serving Modes** — `serving/serving-modes.md` — *concepts:* the four execution modes · why async matters for agents · when sync is fine · streaming · background jobs · state stores & durable execution · the 2026 pattern · multi-provider failover.
- **Observability** — `serving/observability.md` — *concepts:* tracing · 2026 tooling · what to alert on · feedback loops · shadow traffic · P50/P95/P99 latency (why averages lie) · latency budget for RAG.
- **Guardrails** — `serving/guardrails.md` — *concepts:* defense-in-depth layers · input/output/execution guardrails · jailbreak defenses · data privacy & compliance · RAG-specific failure modes · the 2026 attack landscape.

### 5.7 End-to-End System Design — `AgenticAI/docs/system-design/`
Five whiteboard scenarios run through a universal framework (clarify → requirements →
architecture → deep-dives → gotchas).
- **Overview** — `system-design/index.md` — *concepts:* the universal framework · the five scenarios · layer checklist.
- **A. Customer Service Agent** — `system-design/a-customer-service.md` — *concepts:* clarify · functional requirements · architecture · component choices · critical deep-dives · gotchas.
- **B. LLM Eval Pipeline** — `system-design/b-llm-eval-pipeline.md` — *concepts:* clarify · three eval planes · metrics · architecture · Elo + agentic correctness · CI/CD integration · gotchas.
- **C. Sub-800ms Voice Agent** — `system-design/c-voice-agent.md` — *concepts:* the hard latency constraint · architecture · key optimizations · complexity tradeoffs · credit-union specifics · where it breaks.
- **D. Enterprise Knowledge Assistant** — `system-design/d-enterprise-ka.md` — *concepts:* clarify · architecture · ACL propagation · source-specific handling · Graph RAG layer · citation & trust · cost & scaling · gotchas.
- **E. Code Review Agent** — `system-design/e-code-review.md` — *concepts:* clarify · architecture · key decisions · evaluation · cost · gotchas.

---

## 6. ML Case Studies

**Site:** `MLCaseStudies/` · **Scope:** 20 worked ML/GenAI **system-design** case studies for
interviews. Every case is driven to the floor through the **same reusable scaffold**, so the
real differentiator per case is its *intellectual core* — the one data/label or framing problem
unique to that domain. Read one end to end, internalize the scaffold, and the rest are the same
skeleton with different nouns.
**Retrieve from here when** the question is a *system-design* prompt for a specific product
(recommendation, feed ranking, search, ads, RAG copilot, LLM eval/serving platform, fraud,
moderation, spam, model deployment, drift/retraining, etc.), or when you need a worked
end-to-end design, a junior-vs-senior contrast, an interview transcript, or a whiteboard cheat
sheet for one of these problems. For the underlying algorithms/metrics, cross-reference Topics
2–5.

**The shared scaffold (present on every case page, as `## Section N` / `## N.` headings):**
how to use the doc → the reusable scaffold → clarify requirements (pin down numbers) → frame as
an ML problem → **data, features & the domain's label problem** → baseline → why it breaks →
one architecture explained to the floor → evaluation (and the offline/online gap) → deployment
& serving (three paths) → monitoring, retraining, incident response → full one-hour interview
transcript → junior-vs-senior contrast → one-page whiteboard cheat sheet → follow-up questions →
common mistakes → transfer (what this case unlocks) → sources. When citing, name the case path
and the section, e.g. `…/05-production-rag-enterprise-search.md` → "§7 Evaluation (RAGAS
quadrant)."

### 6.1 Recommendation & Personalization — `MLCaseStudies/docs/recommendation/`
How systems decide what to show, and how feedback loops bias the labels you train on.
- **Recommendation System** — `recommendation/01-recommendation-system.md` — *core:* the label-bias problem in a retrieval+ranking recommender.
- **News Feed / Personalized Ranking** — `recommendation/02-news-feed-personalized-ranking.md` — *core:* the engagement trap; value model + integrity.
- **Conversational Recommender** — `recommendation/15-conversational-recommender.md` — *core:* ask vs recommend; in-session vs cross-session, off-policy logs.
- **People You May Know / Graph** — `recommendation/16-people-you-may-know-graph-recommendation.md` — *core:* candidate generation as the game; temporal leakage + endogenous labels.
- **Autocomplete / Typeahead** — `recommendation/18-autocomplete-typeahead-personalization.md` — *core:* completing a query (not ranking a doc); the bias problem.
- **Notification Optimization / Bandit** — `recommendation/19-notification-optimization-bandit-system.md` — *core:* incrementality, delayed reward, fatigue budget; the label is a counterfactual.

### 6.2 Search, Ranking & Ads — `MLCaseStudies/docs/search-ads/`
Ranking under a relevance-label problem, calibration, and the auction/experimentation layer.
- **Search Ranking System** — `search-ads/03-search-ranking-system.md` — *core:* the relevance-label problem; learning-to-rank; interleaving.
- **Ads CTR / Ranking / Experimentation** — `search-ads/11-ads-ctr-ranking-experimentation.md` — *core:* bias/delay in labels; calibration; model + auction.

### 6.3 LLM & GenAI Systems — `MLCaseStudies/docs/llm-systems/`
Grounding, retrieval, agents, serving, safety, and evaluating systems with no clean ground truth.
- **Enterprise AI Copilot** — `llm-systems/04-enterprise-ai-copilot.md` — *core:* grounding, permissions, and why answers go wrong.
- **Production RAG / Enterprise Search** — `llm-systems/05-production-rag-enterprise-search.md` — *core:* data & chunking; the RAGAS evaluation quadrant.
- **LLM Evaluation & Monitoring Platform** — `llm-systems/06-llm-evaluation-monitoring-platform.md` — *core:* label scarcity; you evaluate the eval platform itself.
- **AI Agent for Customer Support** — `llm-systems/07-ai-agent-customer-support-ticket-resolution.md` — *core:* the human-handoff flywheel; containment/wrong-action operating curve.
- **Multi-Tenant LLM Serving Platform** — `llm-systems/08-multi-tenant-llm-serving-platform.md` — *core:* workload characterization & routing; the latency/throughput frontier.
- **LLM Safety Gateway** — `llm-systems/13-llm-safety-gateway.md` — *core:* the threat model; security eval against an adaptive attacker.
- **Document Intelligence Pipeline** — `llm-systems/14-document-intelligence-pipeline.md` — *core:* the cascade & compounding errors; span ground truth.

### 6.4 Trust, Safety & Anomaly — `MLCaseStudies/docs/trust-safety/`
Imbalanced, adversarial, contested-label problems where the cost of a mistake is asymmetric.
- **Fraud / Anomaly Detection** — `trust-safety/09-fraud-anomaly-detection-system.md` — *core:* the censored-label problem; imbalanced, cost-weighted, adversarially honest eval.
- **Content Moderation / Policy Enforcement** — `trust-safety/10-content-moderation-policy-enforcement.md` — *core:* contested ground truth; evaluate on prevalence, not flagged-precision.
- **Spam / Bot Detection** — `trust-safety/17-spam-bot-detection-system.md` — *core:* coordination as signal; adversarially-polluted labels + extreme imbalance.

### 6.5 ML Platform & Ops — `MLCaseStudies/docs/platform-ops/`
The infrastructure under every model: safe rollout, train-serve skew, drift, and retraining.
- **ML Model Deployment Platform** — `platform-ops/12-ml-model-deployment-platform.md` — *core:* a model is not code; train-serve skew & reproducibility; shadow→canary→ramp→rollback.
- **ML Monitoring / Drift / Retraining** — `platform-ops/20-ml-monitoring-drift-retraining.md` — *core:* a ladder of proxies; the label-delay problem; evaluating the monitor.

---

## 7. Productionizing ML

**Site:** `ProductionizingML/` · **Scope:** taking ML from a notebook to a production system. A
conceptual system-design curriculum (a running **checkout-fraud** scenario threads through it)
plus a hands-on **Implementation Masterclass** of the Python/FastAPI/Docker/observability
patterns that close the gap between "works on my laptop" and "serves 1000 req/s reliably." Every
page has a Rapid Recall + Interview Questions; the first six sections are concepts, the last is
runnable code.
**Retrieve from here when** the question is about **generic** system-design building blocks
(load balancers, queues, sharding, consistent hashing, rate limiting), ML-specific data
engineering (events→examples, prediction time, label windows, leakage, lakehouse), the
production training lifecycle (pipelines, tracking/versioning, distributed training, eval gates &
CI/CD, feature stores), model serving (modes, latency budget, batching/caching, compression, LLM
serving), the post-deployment loop (decay, logging, delayed labels, drift, canary/A-B, playbooks),
infrastructure (storage/compute, orchestration, K8s/Ray, governance, cost, build-vs-buy), or the
**runnable implementation** (GIL/concurrency, asyncio, FastAPI serving, batching, Docker,
observability). **Disambiguation:** this is the *production-engineering* view (overlaps Topic 6's
system design and Topic 4/5's serving, but from an infra/implementation angle).

### 7.1 System Design Foundations — `ProductionizingML/docs/sysdesign/`
How to approach a system-design interview *before* ML complexity enters.
- **Overview** — `sysdesign/index.md` — *concepts:* what system-design interviews actually test.
- **The Standard Request Path** — `sysdesign/request-path.md` — *concepts:* why each boundary exists · three request shapes · what the boxes mean.
- **The 12 Building Blocks** — `sysdesign/building-blocks.md` — *concepts:* load balancer · API gateway · caching · SQL vs NoSQL · message queues · CDN · replication vs sharding · consistent hashing · rate limiting · batch vs stream · microservices vs monolith · monitoring/observability.
- **Latency and Capacity** — `sysdesign/latency-capacity.md` — *concepts:* why the tail dominates · capacity estimation.
- **Worked Case Studies** — `sysdesign/case-studies.md` — *concepts:* URL shortener · rate limiter · notification system · key-value store · practice rubric.

### 7.2 ML Data Foundations — `ProductionizingML/docs/data/`
Why ML data differs from software data: the event→example→label chain and where it leaks.
- **Overview** — `data/index.md` — *concepts:* running fraud scenario · why ML requirements differ from software requirements.
- **Events to Training Examples** — `data/events-to-examples.md` — *concepts:* what the product logs · prediction time (the key idea) · observation window & label window.
- **Labels and Ground Truth** — `data/labels.md` — *concepts:* labels are engineered not given · label delay changes system design · label bias.
- **Features, Freshness, Leakage** — `data/features-leakage.md` — *concepts:* what a feature is · leakage explained slowly · a leakage detector.
- **Formats and Storage** — `data/formats-storage.md` — *concepts:* data formats & physical layout · data models, OLTP/OLAP, ACID/BASE.
- **Pipelines and Lakehouse** — `data/pipelines-lakehouse.md` — *concepts:* ETL/ELT, batch, streaming, CDC · lake/warehouse/lakehouse · passing data between components.

### 7.3 Training in Production — `ProductionizingML/docs/training/`
The production training lifecycle from pipeline anatomy through CI/CD gates and feature stores.
- **Overview** — `training/index.md` — *concepts:* running fraud candidate · training-pipeline anatomy.
- **Baselines and Model Choice** — `training/baselines-models.md` — *concepts:* why baselines come first · model choice as a system decision.
- **Tracking and Versioning** — `training/tracking-versioning.md` — *concepts:* experiment tracking · versioning & reproducibility.
- **Distributed Training** — `training/distributed.md` — *concepts:* data parallelism · model parallelism · checkpointing.
- **Evaluation Gates and CI/CD** — `training/evaluation-cicd.md` — *concepts:* evaluation gates · CI/CD for ML.
- **Feature Stores and LLMs** — `training/feature-stores.md` — *concepts:* why a feature store exists · LLM & post-training reality · interview synthesis.

### 7.4 Serving the Model — `ProductionizingML/docs/serving/`
Inference after promotion: modes, latency, batching/caching, compression, and LLM serving.
- **Overview** — `serving/index.md` — *concepts:* running fraud-at-checkout · what happens when a model serves a prediction.
- **Serving Modes** — `serving/modes.md` — *concepts:* batch · online · streaming · async · edge inference.
- **Online Path and Latency** — `serving/online-path-latency.md` — *concepts:* the online serving path, slowly · the latency budget.
- **Batching and Caching** — `serving/batching-caching.md` — *concepts:* batching, queues, backpressure · caching in ML serving.
- **Model Compression** — `serving/compression.md` — *concepts:* quantization · pruning · distillation · low-rank factorization.
- **LLM Serving** — `serving/llm-serving.md` — *concepts:* prefill · decode · KV cache · paged attention & continuous batching · speculative decoding · LLM serving metrics.
- **Runtimes and Failures** — `serving/runtimes-failures.md` — *concepts:* cloud vs edge · serving runtimes in 2026 · failure handling.

### 7.5 The Production Loop — `ProductionizingML/docs/loop/`
What happens after deployment: decay, logging, delayed labels, drift, rollout, experiments.
- **Overview** — `loop/index.md` — *concepts:* running fraud-after-launch · why models decay.
- **Logging and Labels** — `loop/logging-labels.md` — *concepts:* what to log at prediction time · delayed labels & proxy metrics.
- **Production Evaluation** — `loop/production-evaluation.md` — *concepts:* why it differs from offline eval · calibration · slice evaluation · robustness & invariance.
- **Drift and Monitoring** — `loop/drift.md` — *concepts:* the three kinds of drift · detection methods · monitoring vs observability.
- **Deployment Strategies** — `loop/deployment-strategies.md` — *concepts:* shadow · canary · A/B testing · interleaving.
- **Experiments and Playbooks** — `loop/experiments-playbooks.md` — *concepts:* A/B testing & bandits · response playbooks · interview synthesis.

### 7.6 Infrastructure and Build-vs-Buy — `ProductionizingML/docs/infra/`
Platform thinking: the four layers, the compute/storage/orchestration stack, K8s, and the
build-vs-buy decision (with an end-to-end fraud-platform capstone).
- **Overview** — `infra/index.md` — *concepts:* running fraud ML platform · what infrastructure means in ML · the four infrastructure layers.
- **Storage and Compute** — `infra/storage-compute.md` — *concepts:* where each kind of state lives · CPU/GPU/TPU, training vs serving.
- **Orchestration** — `infra/orchestration.md` — *concepts:* from scripts to reliable workflows · backfills.
- **ML Platform and Kubernetes** — `infra/platform-k8s.md` — *concepts:* ML platform anatomy · Kubernetes, Ray, GPU scheduling.
- **Governance and Cost** — `infra/governance-cost.md` — *concepts:* security & governance · cost governance.
- **Build vs Buy and Capstone** — `infra/build-vs-buy.md` — *concepts:* build vs buy · end-to-end fraud-platform capstone.

### 7.7 Implementation Masterclass — `ProductionizingML/docs/impl/`
The runnable code layer — the Python/serving patterns behind the concepts above.
- **Overview** — `impl/index.md` — *concepts:* the mental model — where everything fits.
- **Concurrency and the GIL** — `impl/concurrency.md` — *concepts:* I/O-bound vs CPU-bound · the GIL · the decision table · concurrency vs parallelism.
- **Asyncio Deep Dive** — `impl/asyncio.md` — *concepts:* the event loop · coroutines/async/await · blocking the event loop (the cardinal sin) · gather, create_task, to_thread.
- **FastAPI and Model Serving** — `impl/fastapi-serving.md` — *concepts:* FastAPI, WSGI vs ASGI · Pydantic validation · sync vs async endpoints · serving a real scikit-learn model · running it for real.
- **Serving and Batching** — `impl/serving-batching.md` — *concepts:* latency & throughput · batching (the throughput lever) · the serving-framework landscape · how the pieces fit.
- **Docker and Deployment** — `impl/docker-deploy.md` — *concepts:* images vs containers · Dockerfile & layer-caching · multi-stage builds & image hygiene · the four ways to run in the cloud.
- **Observability** — `impl/observability.md` — *concepts:* the three pillars · structured logging · metrics (RED & USE) · tracing · ML-specific drift monitoring · interview synthesis.

---

## Cross-topic disambiguation (for retrieval)

Several concepts intentionally appear in more than one site, at different depths/angles. Pick the
source that matches the *intent* of the question:

- **Attention / Transformers** → DL `sequences/architectures-attention.md` builds *up to*
  attention; **LLM `foundations/`** is the canonical Transformer internals.
- **LLM inference (prefill/decode, KV-cache, paged attention, speculative)** → **LLM
  `inference-arch/` + `serving/`** is deepest; AgenticAI `serving/inference-stack.md` and
  ProductionizingML `serving/llm-serving.md` are the applied/serving views.
- **Evaluation metrics** → **ClassicalML `evaluation/`** is the canonical metric reference;
  AgenticAI `eval/` is RAG/agent-specific; Mathematics `testing/` is the statistical-testing view.
- **RAG** → **AgenticAI `rag/`** is canonical; MLCaseStudies `llm-systems/05…` is the system-design view.
- **Drift / monitoring / canary / A-B** → ProductionizingML `loop/` and MLCaseStudies
  `platform-ops/` overlap; ProductionizingML is concept-first, MLCaseStudies is interview-design-first.
- **Losses (DPO/GRPO/RLHF)** → **LLM `alignment/`** is deepest; DL `losses/llm-losses.md` is the loss-function-taxonomy view.
- **System design** → ProductionizingML `sysdesign/` = generic building blocks; MLCaseStudies = full ML product designs; AgenticAI `system-design/` = agent-specific scenarios.

---

*End of index. When you cite anything above, follow the [Attribution protocol](#attribution-protocol-mandatory-️): mark knowledge-base content with its path, and flag anything sourced elsewhere as external.*
