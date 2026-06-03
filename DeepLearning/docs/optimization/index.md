# Optimization

Once backprop has handed you a gradient for every weight, the optimizer decides how to turn those gradients into a weight update. This is the "taking the step" bucket: vanilla SGD subtracts the gradient, and everything fancier (momentum, adaptive rates, Adam, AdamW) is a smarter update rule layered on top. Before the optimizers themselves, settle the batch-size question that frames all of them.

!!! tip "Rapid Recall"
    Batch size lives on a spectrum: true SGD (one sample) is too noisy, full-batch GD won't fit in memory, so mini-batch SGD (32 to 1024) is the universal default, and "SGD" in modern usage means mini-batch. Larger batches give less gradient noise and allow a higher learning rate (roughly linear scaling). GPUs make a batch of 256 nearly as cheap in wall-clock as a batch of 1, which is why batch size 1 is wasteful. LLMs reach huge effective batches through gradient accumulation.

## §1 The batch size spectrum

Three modes exist on paper; only mini-batch is used in practice.

| Variant | Batch size | Gradient quality | GPU use | In practice |
| --- | --- | --- | --- | --- |
| True SGD | 1 sample | Very noisy | Wasteful, GPU sits idle | Theoretical only |
| **Mini-batch SGD** | 32–1024 | Good estimate | Excellent, designed for this | **The default everywhere** |
| Full batch GD | Whole dataset | Exact | Won't fit in memory | Theoretical only for large data |

### What "SGD" means in 2026

When people say "SGD" in modern deep learning, they almost always mean *mini-batch SGD*, not single-sample SGD. The "stochastic" refers to sampling mini-batches from the dataset, not to the batch size being 1.

### Why mini-batch wins

- One sample is too noisy, each update is a wild guess at the true gradient direction.
- Full batch wastes information, early in training a coarse gradient estimate is enough; you don't need a perfect gradient to know which way is generally downhill.
- GPUs are designed for parallel matrix operations. A batch of 256 samples runs in roughly the same wall-clock time as 1 sample on a GPU. The hardware is wasted with batch size 1.

### Batch size as a hyperparameter

Larger batch → less gradient noise → can usually use a higher learning rate (linear scaling rule: 2x batch ≈ 2x LR, roughly). Smaller batch → more noise → some regularization benefit but slower per epoch.

Standard range: 32 to 512 for most tasks. LLMs use much larger effective batches (millions of tokens) via *gradient accumulation*, process several small batches, accumulate gradients without stepping, then step once. This simulates a large batch without large memory.

## §2 The optimizer family

The rest of this section builds up the optimizer family in the order it was historically invented, because each one fixes a flaw in the last:

- [SGD and Momentum](sgd-and-momentum.md): the baseline subtract-the-gradient rule, plus the heavy-ball idea that smooths it.
- [Adaptive methods](adaptive-methods.md): AdaGrad, RMSProp, Adam, and AdamW, which give each parameter its own effective learning rate.
- [Modern optimizers](modern-optimizers.md): Lion, LAMB, Sophia, Shampoo, plus the cheat sheet and decision tree for picking one.

## Interview Questions

**Q1: What is the difference between batch SGD, mini-batch SGD, and full batch GD?**

Batch sizes lie on a spectrum. True SGD uses one sample at a time, very noisy gradients, wastes GPU. Mini-batch SGD uses 32 to 1024 samples, a good gradient estimate with excellent GPU utilization. This is what's used in practice. Full batch GD uses the entire dataset, an exact gradient, but won't fit in memory for any real dataset. When people say "SGD" in modern deep learning, they almost always mean mini-batch SGD. The "stochastic" refers to sampling mini-batches randomly, not to the batch size being 1.
