# Regularization

A network with enough parameters can memorize the training data perfectly while failing on new data. Regularization trades a small amount of training fit for better generalization. The four workhorses are weight decay, dropout, early stopping, and label smoothing.

!!! tip "Rapid Recall"
    Weight decay adds an L2 penalty so each step shrinks weights by $(1 - \eta\lambda)$ before the gradient update; do not apply it to biases or norm parameters. Dropout zeros random units during training (scaled by $1/(1-p)$ so inference needs no rescale) and behaves like averaging an exponential ensemble of sub-networks. Early stopping halts at the minimum validation loss and restores the best checkpoint, the sweet spot before the model memorizes noise. Label smoothing replaces one-hot targets with $1-\varepsilon$ for the true class, curbing overconfidence and improving calibration.

## §1 Weight Decay (L2 Regularization)

Add a penalty on large weights. The gradient gains a shrinkage term, so every step pulls weights slightly toward zero before applying the gradient update. Net effect: simpler models, better generalization.

| Symbol | Meaning |
| --- | --- |
| $L_{\text{task}}$ | The original task loss (cross-entropy, MSE, etc.). |
| $L_{\text{total}}$ | The total loss including the regularization penalty. |
| $\lambda$ | Weight decay coefficient. Controls regularization strength. |
| $\theta$ | The parameters being regularized. |

$$
\begin{aligned}
\textbf{Add an L2 penalty:}\quad & L_{\text{total}} = L_{\text{task}} + \frac{\lambda}{2}\lVert\theta\rVert^2 \\[4pt]
\textbf{Gradient gains an extra term:}\quad & \frac{\partial L_{\text{total}}}{\partial \theta} = \frac{\partial L_{\text{task}}}{\partial \theta} + \lambda \cdot \theta \\[4pt]
\textbf{Plug into SGD update:}\quad & \theta \leftarrow \theta - \eta\left(\frac{\partial L_{\text{task}}}{\partial \theta} + \lambda \theta\right)
= \underbrace{\theta(1 - \eta\lambda)}_{\text{multiplicative shrinkage IS "weight decay"}} - \eta\frac{\partial L_{\text{task}}}{\partial \theta}
\end{aligned}
$$

The factor $(1 - \eta\lambda)$ shrinks weights every step before the gradient update. The name "weight decay" comes from this multiplicative shrinkage.

### Three lenses for understanding why it works

- **Bayesian.** L2 regularization is MAP estimation with a Gaussian prior on weights. You are saying "I expect weights to be small a priori; the data has to convince me otherwise."
- **Geometric.** Among all models that fit the training data equally well, you're picking the one with smallest weights. Smaller weights produce smoother decision boundaries, less overfitting.
- **Implicit regularization.** For overparametrized networks, infinitely many weight configs achieve zero training loss. Weight decay biases the optimizer toward simpler configurations.

### The Adam vs AdamW issue (revisited)

In Adam, adding $\lambda\theta$ to the gradient means it gets scaled by $1/\sqrt{\hat{v}}$, the adaptive factor based on squared gradient history:

$$
\textbf{Adam (broken):}\quad \text{update} = \frac{g + \lambda\theta}{\sqrt{\hat{v}}} \quad \text{(weight decay scaled by adaptive factor)}
$$

Parameters with large gradient history (large $\hat{v}$) get *less* weight decay. Not what we want. AdamW decouples:

$$
\textbf{AdamW (correct):}\quad \theta \leftarrow \theta(1 - \eta\lambda) - \eta\frac{\hat{m}}{\sqrt{\hat{v}}}
$$

Uniform decay across all parameters. This is why [AdamW](../optimization/adaptive-methods.md) is standard for transformer training.

### Typical values

- **CNNs with SGD + Momentum:** $\lambda \approx 10^{-4}$ to $5 \times 10^{-4}$
- **Transformers with AdamW:** $\lambda \approx 0.01$ to $0.1$ (much larger because of the decoupled formulation)
- **LLMs:** $\lambda = 0.1$ is common

!!! warning "What NOT to decay"
    Standard practice (and a crucial detail): **do not apply weight decay to biases, LayerNorm/BatchNorm parameters ($\gamma$, $\beta$), or embedding layers**. These are typically excluded via parameter groups:

    ```python
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
    optimizer_params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_params)
    ```

    Why? Biases and norm parameters are 1D and serve specific shift/scale roles. Shrinking them toward zero distorts the network. People get this wrong constantly.

Complexity is trivial: just $\theta \leftarrow \theta(1 - \eta\lambda)$ during the optimizer step. Adds zero memory and approximately one elementwise multiply per parameter.

## §2 Dropout

Randomly zero out some neurons during training. Forces the network to be robust, it cannot rely on any single neuron because that neuron might be dropped. Equivalent to averaging an ensemble of exponentially many sub-networks.

| Symbol | Meaning |
| --- | --- |
| $p$ | Dropout rate. Probability of dropping (zeroing) each neuron. Typical 0.1-0.5. |
| mask | Bernoulli random vector, 1 with probability $(1-p)$, 0 with probability $p$. |
| $a$ | The activation we apply dropout to. |

$$
\begin{aligned}
\textbf{During training:}\quad & \text{mask} \sim \mathrm{Bernoulli}(1 - p), \qquad a_{\text{dropped}} = \frac{\text{mask} \odot a}{1 - p} \\[4pt]
\textbf{During inference:}\quad & a_{\text{dropped}} = a \quad \text{(dropout disabled, pass through)}
\end{aligned}
$$

### Why the 1/(1−p) scaling matters

Without scaling, the expected magnitude of the output drops by factor $(1-p)$ during training. Then at inference (no dropout), magnitudes suddenly jump back up by $1/(1-p)$, a distribution shift between train and inference. The scaling during training ensures expected output magnitude is preserved: $\mathbb{E}[\text{mask} \cdot a / (1-p)] = (1-p) \cdot a / (1-p) = a$. This is "inverted dropout", scale during training so no scaling is needed at inference.

### The ensemble interpretation

Each training step uses a different sub-network (different mask). The network cannot rely on any single neuron, it is forced to distribute information across many paths. Equivalent to training an ensemble of $2^N$ networks (where $N$ is the number of neurons) and averaging them at inference. The averaging happens implicitly through weight sharing.

### Typical rates

- **0.1**, transformers (applied after attention and FFN layers)
- **0.2-0.3**, moderate dense layers
- **0.5**, original dense layers in image classification heads
- **0**, modern LLMs often use no dropout at all, relying on weight decay and large datasets instead

!!! warning "Do not forget eval mode"
    Dropout is disabled by `model.eval()`. If you forget this, your inference will be stochastic, different predictions for the same input on different calls. A common bug, debuggable by checking that two inference calls give identical outputs.

## §3 Early Stopping

Monitor validation loss during training. If it has not improved for some number of steps, stop training and restore the best checkpoint. No loss-function change, no hyperparameter beyond patience. Use it always (with one exception below).

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 5

for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = evaluate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        restore_checkpoint()
        break
```

### Why it works

Training loss almost always keeps decreasing, the model is fitting more of the training data, including its noise. Validation loss decreases for a while, then starts increasing as the model overfits. The minimum validation loss is the sweet spot where the model has learned the signal but not yet memorized the noise. Early stopping captures this point automatically.

### When NOT to use it

LLM pretraining is the main exception. Training is done for a fixed number of tokens with a planned LR schedule. There is no validation set to monitor, and the schedule assumes training continues to completion. Do not early-stop these.

The patience hyperparameter: higher patience (say 10) means you wait longer before stopping, fewer false stops due to noise. Lower patience (say 3) means you stop earlier, faster training but risk premature stopping. Typical: 5-10 epochs.

## §4 Label Smoothing

Replace one-hot targets with smoothed versions. Tells the model "this is 90% a cat" instead of "this is 100% a cat." Discourages overconfidence, improves calibration, mild regularization. Standard in modern image classification and transformer training.

$$
y_c^{\text{smooth}} = \begin{cases} 1 - \varepsilon & \text{if } c \text{ is the correct class} \\ \dfrac{\varepsilon}{C - 1} & \text{otherwise} \end{cases}
$$

where $C$ = number of classes, $\varepsilon \approx 0.1$ typical. Apply standard cross-entropy with these smoothed labels. (The [label-smoothing cross-entropy](../losses/classification-losses.md) loss page restates this from the loss-function angle.)

### Why it helps

- **Better calibration.** Without smoothing, models become arbitrarily overconfident, predicted probability 99% for things that are right only 80% of the time. With smoothing, predicted probabilities match true accuracies more closely.
- **Regularization effect.** The model cannot simply predict 1.0 for the correct class to drive loss to zero. There is always a floor of expected loss, which discourages overfitting on training labels.
- **Robustness to label noise.** If some labels are wrong, smoothing reduces the penalty for not perfectly matching them.

Tradeoffs: it slightly hurts top-1 accuracy in some cases (the model "hedges" more). The hyperparameter $\varepsilon$ rarely needs tuning, 0.1 works for most cases.

## Interview Questions

**Q1: What types of parameters should NOT have weight decay applied?**

Biases, LayerNorm/BatchNorm parameters ($\gamma$ and $\beta$), and embedding layers should be excluded from weight decay. These are 1D parameters serving specific shift/scale roles, and shrinking them toward zero distorts the network's ability to learn proper offsets and scales. This is typically implemented via parameter groups in the optimizer, pass two groups, one with the decay coefficient and one with `weight_decay=0`. People get this wrong constantly; it can meaningfully impact training stability.

**Q2: How does label smoothing affect calibration?**

It improves calibration. Without label smoothing, models become arbitrarily overconfident, they learn to predict probability 0.99 for cases where they're actually right only 80% of the time. The loss is minimized by driving probability to 1.0 for the correct class, even when the model doesn't have enough evidence. With label smoothing, the target is 0.9 for the correct class (with $\varepsilon = 0.1$) instead of 1.0. The model can never drive loss to zero by becoming arbitrarily confident, there's always a floor of expected loss. This forces it to express genuine uncertainty, leading to predicted probabilities that match true accuracies more closely.
