# Ranking & Metric Learning Losses

For tasks where the output is not a class or a number but a learned embedding space, you need losses that operate on distances or similarities between samples.

!!! tip "Rapid Recall"
    Triplet loss pulls an anchor toward a positive and pushes it from a negative with a margin, the basis of face recognition, but triplet mining is hard. Contrastive (pairwise) loss does the same for pairs. InfoNCE/NT-Xent is cross-entropy over "which item in the batch is the true match," the engine of self-supervised learning (SimCLR, MoCo, CLIP), and it loves large batches and a small temperature. Pairwise ranking losses (RankNet, LambdaRank) optimize the probability that item $i$ ranks above item $j$, used in search and recommendation.

## §1 Triplet Loss

| Symbol | Meaning |
| --- | --- |
| $a$ | Anchor sample. |
| $p$ | Positive, a sample of the same class/identity as the anchor. |
| $n$ | Negative, a sample of a different class. |
| $d(\cdot,\cdot)$ | Distance function. Usually L2 or cosine. |
| $m$ | Margin, the minimum desired separation between positive and negative. |

$$
L = \max\big(0, \ d(a, p) - d(a, n) + m\big)
$$

**Intuition:** pull the anchor close to the positive, push the anchor far from the negative, with a buffer (margin). If the negative is already far enough away ($d(a, n) \ge d(a, p) + m$), the loss is zero, the gradient becomes inactive.

**Strengths:** the foundation of face recognition (FaceNet), metric learning, and embedding-based retrieval; learns useful representations without needing explicit class labels. **Limitations:** triplet mining is hard, most random triplets are too easy (loss = 0), so you need hard or semi-hard negative mining; slow convergence; sensitive to the margin hyperparameter.

## §2 Contrastive Loss (Pairwise)

$$
\text{with } y = 1 \text{ if the pair is similar, } 0 \text{ if different:} \qquad L = (1 - y)\cdot\frac{1}{2}d^2 + y\cdot\frac{1}{2}\max(0, m - d)^2
$$

Similar pairs: pull together. Dissimilar pairs: push apart only until the margin is satisfied.

## §3 InfoNCE / NT-Xent (Contrastive Learning)

| Symbol | Meaning |
| --- | --- |
| $z_i, z_j$ | Embeddings of a positive pair (for example, two augmentations of the same image). |
| $\mathrm{sim}(\cdot,\cdot)$ | Similarity function. Usually cosine. |
| $\tau$ | Temperature. Typically 0.05-0.1. Lower = sharper distribution. |
| $N$ | Batch size (number of negatives drawn from the batch). |

$$
L = -\log\frac{\exp(\mathrm{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{N} \exp(\mathrm{sim}(z_i, z_k) / \tau)}
$$

**Intuition:** cross-entropy where the "classes" are which item in the batch is the true match. Pulls the embeddings of two augmentations of the same example together; pushes against everyone else in the batch.

**Strengths:** the foundation of self-supervised learning (SimCLR, MoCo, CLIP); no need for class labels, supervision comes from data augmentations or paired modalities (image-text in CLIP); scales well with batch size (more negatives means better signal). **Limitations:** requires very large batch sizes (typically 4096+) for good negatives, OR a memory bank (MoCo); sensitive to temperature $\tau$ (typically 0.05-0.1); performance degrades if augmentations are too weak (collapse) or too strong (lose semantics).

**Use when:** self-supervised pretraining, multimodal learning (CLIP-style), retrieval systems.

## §4 Pairwise Ranking Losses (LambdaRank, RankNet)

For learning-to-rank: compute pairwise probabilities that item $i$ should rank above item $j$, optimize cross-entropy on these pairs. LambdaRank weights pairs by their impact on NDCG (Normalized Discounted Cumulative Gain). Used in search and recommendation ranking, the cross-encoder reranking family used in Microsoft Bing, Seller Copilot reranking, and similar systems.
