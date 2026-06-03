# Normalization

Another major attack on unstable gradients: keep activations in a well-behaved range at every layer regardless of weight initialization or input distribution. The three normalizers differ in *what axis* they normalize over, which is exactly what decides where each one works.

!!! tip "Rapid Recall"
    BatchNorm normalizes each channel across the batch, so it is great for CNNs but breaks at small batch sizes, on variable-length sequences, and needs running averages for inference (hence `model.eval()`). LayerNorm normalizes across features within each sample, so it is batch-size and sequence-length independent and behaves identically at train and inference, which is why every transformer uses it. RMSNorm is LayerNorm without mean subtraction or the shift parameter, about 25% cheaper, and standard in modern LLMs. All three apply a learnable scale (and BN/LN a shift) so the network can undo normalization if needed.

## §1 Batch Normalization

BatchNorm normalizes across the batch dimension. Standard for CNNs.

| Symbol | Meaning |
| --- | --- |
| $z$ | Input activations to BN, shape $(B, C)$ for MLP or $(B, C, H, W)$ for CNN. We normalize per channel. |
| $\mu_B$ | Per-channel mean across the batch (and spatial dims for CNN). |
| $\sigma_B^2$ | Per-channel variance across the batch. |
| $\hat{z}$ | Normalized activations, zero mean, unit variance per channel. |
| $\gamma, \beta$ | Learnable scale and shift. Per channel. Allow the network to "undo" normalization if needed. |
| $\varepsilon$ | Tiny constant for numerical stability. |

$$
\begin{aligned}
\mu_B &= \frac{1}{B}\sum_{i=1}^{B} z_i && \text{mean across batch} \\
\sigma_B^2 &= \frac{1}{B}\sum_{i=1}^{B} (z_i - \mu_B)^2 && \text{variance across batch} \\
\hat{z}_i &= \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}} && \text{normalize} \\
y_i &= \gamma \cdot \hat{z}_i + \beta && \text{scale and shift (learnable)}
\end{aligned}
$$

### Inference mode, running averages

At inference, you cannot use batch statistics, batches might be size 1, or have a different distribution. So during training, BN maintains running averages of $\mu$ and $\sigma^2$:

$$
\mu_{\text{running}} \leftarrow 0.9 \cdot \mu_{\text{running}} + 0.1 \cdot \mu_B, \qquad
\sigma^2_{\text{running}} \leftarrow 0.9 \cdot \sigma^2_{\text{running}} + 0.1 \cdot \sigma_B^2
$$

At inference, use $\mu_{\text{running}}$ and $\sigma^2_{\text{running}}$ instead of batch statistics. This is why `model.eval()` matters, it switches BN to use running stats.

### Why BN helps

- **Stabilizes activations.** Keeps them in the well-behaved range of activation functions, preventing saturation and vanishing gradients.
- **Allows higher learning rates.** Gradient updates are less sensitive to weight scale because the next layer's input is normalized anyway.
- **Mild regularization.** Batch statistics add noise, different mini-batches give different normalizations, a small stochastic effect.
- **Smooths the loss landscape.** Later research suggests this is the real benefit, not the originally-claimed "internal covariate shift reduction."

### Failure modes

| Problem | What happens | Why |
| --- | --- | --- |
| Small batches (< 8) | Training unstable, accuracy drops | Batch statistics are noisy estimates |
| Train/inference distribution shift | Silent accuracy drop | Running averages do not match actual data |
| Variable-length sequences | Does not make sense | Different positions are semantically different |
| Online learning / RL | Can't compute batch stats | No batch to average over |

### Parallelizability

Highly parallel across the batch dimension on a single GPU. For multi-GPU training: **SyncBN** gathers statistics across all GPUs (correct but adds a communication round), or **Local BN** uses per-GPU stats (faster but biased for small per-GPU batch). Choose based on tolerance for the bias.

## §2 Layer Normalization

The fix for BatchNorm's batch-size dependency. Normalize across the feature dimension within each sample, not across the batch. Standard for all transformers. For each sample independently:

$$
\begin{aligned}
\mu_i &= \frac{1}{D}\sum_{d=1}^{D} z_{i,d} && \text{mean across features (this sample)} \\
\sigma_i^2 &= \frac{1}{D}\sum_{d=1}^{D} (z_{i,d} - \mu_i)^2 \\
\hat{z}_{i,d} &= \frac{z_{i,d} - \mu_i}{\sqrt{\sigma_i^2 + \varepsilon}} \\
y_{i,d} &= \gamma_d \cdot \hat{z}_{i,d} + \beta_d && \text{learnable per-feature scale and shift}
\end{aligned}
$$

### Shape for transformers

Input shape $(B, T, D)$, batch, sequence length, model dim. Statistics computed per $(b, t)$ pair across the $D$ dimension. Output same shape.

### Why this works for transformers

- **No batch dependence**, works with any batch size, even batch size 1.
- **No sequence dependence**, works for variable-length sequences.
- **Same behavior at training and inference**, no running averages needed.

### Why BatchNorm does not work for transformers

Different positions in a sequence are *semantically different*. Position 5 might be a noun, position 50 might be a verb. Normalizing across the batch *at each position* assumes those positions have similar statistics across samples, they don't. LayerNorm normalizes within each token's own representation, which is the right thing for sequences. It is trivially parallel: each $(b, t)$ pair is independent, no cross-sample or cross-position communication.

## §3 RMSNorm

LayerNorm minus the mean-centering. Used in LLaMA, Mistral, DeepSeek, and most modern LLMs. About 25% cheaper than LayerNorm with comparable or better empirical performance. Just divide by root-mean-square magnitude:

$$
\mathrm{RMS}(z) = \sqrt{\frac{1}{D}\sum_d z_d^2}, \qquad \hat{z}_d = \frac{z_d}{\mathrm{RMS}(z)}, \qquad y_d = \gamma_d \cdot \hat{z}_d
$$

(no $\beta$ shift parameter either.)

### What changed from LayerNorm

- No mean subtraction (no computation of $\mu$).
- No shift parameter $\beta$ (saves $D$ parameters).
- Otherwise identical structure.

Empirically, the re-centering step (subtracting the mean) in LayerNorm does not help much on transformers but adds computation. RMSNorm is faster (no mean computation, no shift parameter) and slightly simpler. Works as well or better in practice for large models.

## §4 The "norm" comparison

|  | BatchNorm | LayerNorm | RMSNorm |
| --- | --- | --- | --- |
| Normalizes over | Batch (per channel) | Features (per sample) | Features (no mean) |
| Batch size 1? | No | Yes | Yes |
| Variable sequences? | No | Yes | Yes |
| Train/inference modes? | Yes (running stats) | No | No |
| Compute | Moderate | Moderate | Lowest |
| Parameters per layer | $2C$ + $2C$ buffers | $2D$ | $D$ |
| Standard for | CNNs | BERT, GPT-2 | Modern LLMs (LLaMA, Mistral) |

**Decision rule:** BatchNorm for CNNs. LayerNorm or RMSNorm for transformers/sequences. RMSNorm if you want the fastest option and are training LLM-scale models. The placement of LayerNorm within a residual block (pre-LN vs post-LN) is covered on the [gradient highways](../gradient-flow/gradient-highways.md) page.

## Interview Questions

**Q1: BatchNorm vs LayerNorm, when does each break?**

BatchNorm normalizes across the batch dimension for each feature. It breaks when batch size is small (less than 8) because the batch statistics become noisy estimates of the true statistics. It also breaks for variable-length sequences because different positions in a sequence don't share semantics, so normalizing across them is meaningless. LayerNorm normalizes across the feature dimension for each individual sample, so it works with any batch size and any sequence length. This is why every transformer uses LayerNorm. BatchNorm is still the standard for CNNs where batch sizes are typically large and fixed-size spatial features are consistent across samples.
