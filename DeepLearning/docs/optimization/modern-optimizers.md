# Modern Optimizers & Choosing One

For interviews, AdamW + SGD+Momentum is plenty. But naming a few modern variants signals you're current, and the cheat sheet and decision tree at the end are what you actually reach for when starting a project.

!!! tip "Rapid Recall"
    Lion (2023) uses only the sign of a momentum estimate and drops the second moment, so it is 2× memory instead of Adam's 3× and often matches AdamW on transformers. LAMB adds per-layer LR adaptation for very large batch sizes (it trained BERT at batch 32K). Sophia uses a diagonal Hessian estimate; Shampoo uses a Kronecker-factored preconditioner. For picking: AdamW for tabular and transformers, SGD+Momentum for CNNs, and the learning rate is always the first knob to tune.

## §1 Lion (EvoLved Sign Momentum), 2023

From Google. Uses just the sign of a momentum estimate, no second moment. **Memory: 2× instead of 3×.**

$$
\begin{aligned}
\text{update} &= \mathrm{sign}\big(\beta_1 \cdot m + (1 - \beta_1) \cdot g\big) \\
\theta &\leftarrow \theta - \eta \cdot (\text{update} + \lambda \cdot \theta) \\
m &\leftarrow \beta_2 \cdot m + (1 - \beta_2) \cdot g
\end{aligned}
$$

Often matches or beats AdamW on transformers with lower memory cost. Worth knowing as the modern challenger to AdamW for LLM training.

## §2 LAMB (Layer-wise Adaptive Moments for Batch)

AdamW with per-layer learning rate adaptation. Designed for huge batch sizes, used to train BERT with batch size 32K. Probably overkill unless you are training very large models on many GPUs.

## §3 Sophia (Second-order Optimizer)

Uses a diagonal Hessian estimate. Faster convergence on LLM pretraining in benchmarks. Adds complexity. Used in some recent training runs.

## §4 Shampoo / Distributed Shampoo

Approximates a per-layer preconditioner via Kronecker products. Used in some large-scale training. Complex but better convergence at scale.

## §5 Hyperparameter cheat sheet

| Hyperparameter | Typical value | What it does | When to tune |
| --- | --- | --- | --- |
| Learning rate $\eta$ | $10^{-3}$ (Adam), $3\times10^{-4}$ (AdamW for LLMs), 0.1 (SGD) | Step size | Always, this is THE knob |
| Batch size | 32-512 | Gradient noise | Maximize within GPU memory |
| $\beta_1$ (momentum) | 0.9 | Smoothness of grad EMA | Rarely (0.95 for very long training) |
| $\beta_2$ (squared grad EMA) | 0.999 | Stability of adaptive LR | Rarely (0.95 for short training) |
| Weight decay $\lambda$ | 0.01-0.1 (AdamW), $10^{-4}$ (SGD) | Regularization strength | Always for transformers |
| $\varepsilon$ | $10^{-8}$ | Numerical safety | Almost never |
| LR warmup steps | ~1% of total steps | Stabilize early training | If early loss spikes |

## §6 The decision tree

- **Tabular / small data:** AdamW with lr $= 10^{-3}$.
- **CNN / image classification:** SGD + Momentum with lr $= 0.1$, cosine decay, weight decay $= 10^{-4}$. Adam works but generalization is sometimes worse.
- **Transformers / NLP / LLMs:** AdamW with lr $= 3 \times 10^{-4}$, weight decay 0.01-0.1, warmup + cosine schedule.
- **RNNs (rare in 2026):** Adam or RMSProp.
- **Memory-constrained large model training:** Lion, 8-bit AdamW, or sharded optimizer state.

The [learning rate schedules](../regularization/lr-schedules.md) page covers the warmup + cosine recipe these recommendations assume.
