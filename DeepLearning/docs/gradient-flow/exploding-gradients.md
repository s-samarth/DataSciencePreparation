# Exploding Gradients & Clipping

Vanishing's mirror: if each Jacobian has magnitude greater than 1, the gradient grows exponentially through depth. With poorly initialized weights or deep RNNs unrolled over long sequences, early-layer gradients can hit numerical overflow. NaN territory. Loss becomes infinity. Weights become NaN. Training is dead.

!!! tip "Rapid Recall"
    A modest per-layer amplification compounds: 1.5x over 20 layers is already ~3000x, and a little more is overflow. The standard fix is clip-by-global-norm: if the total L2 norm of all gradients exceeds a threshold (usually 1.0), rescale the whole gradient vector to that norm, preserving direction and capping magnitude. Clip between `backward()` and `step()`. It is mandatory for RNN/LSTM training, standard for transformer pretraining, and cheap insurance everywhere else.

## §1 Numerical demonstration

Same 20-layer simulation, weights initialized slightly large (scale 1.5), ReLU active so $\sigma' = 1$:

$$
\begin{aligned}
\text{Layer 20:}\quad & 1.5^{1}  = 1.5 \\
\text{Layer 15:}\quad & 1.5^{6}  = 11.4 \\
\text{Layer 10:}\quad & 1.5^{11} = 86.5 \\
\text{Layer 5:}\quad  & 1.5^{16} = 657 \\
\text{Layer 1:}\quad  & 1.5^{20} = 3{,}325
\end{aligned}
$$

A modest 1.5x amplification per layer becomes 3000x by layer 1. With more layers or larger weights you hit overflow fast.

## §2 The fix: gradient clipping

Two flavors. The first is naive and rarely used; the second is the standard.

### Clipping by value (naive)

```python
for param in model.parameters():
    param.grad.clamp_(-clip_val, clip_val)
```

Cap each element. Simple but distorts the gradient direction (different elements get clamped differently).

### Clipping by norm (the standard)

If the total gradient norm exceeds the threshold, rescale:

$$
\text{if } \lVert g \rVert_2 > \text{max\_norm}: \quad g \leftarrow g \cdot \frac{\text{max\_norm}}{\lVert g \rVert_2}
$$

Rescale the entire gradient vector so its L2 norm is at most max_norm. **Direction is preserved; magnitude is capped.** This is the right fix because exploding gradients are about magnitude, not direction.

### Worked example

$$
\begin{aligned}
\text{Before:}\quad & g = [3, 4], \quad \lVert g \rVert = 5, \quad \text{max\_norm} = 1 \\
\text{After:}\quad  & g = [3, 4] \cdot (1/5) = [0.6, 0.8], \quad \lVert g \rVert = 1
\end{aligned}
$$

Direction unchanged, magnitude capped.

## §3 When to use it

- **Always for RNN/LSTM training.** Gradients compound multiplicatively over time; clipping is mandatory.
- **Standard for transformer pretraining.** max_norm = 1.0 is the default for LLMs (GPT, LLaMA, Claude all use it).
- **Cheap insurance for everything else.** Cost is $O(P)$, negligible. One pass to compute the global norm, one to rescale.

!!! note "Implementation in PyTorch"
    ```python
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ```
    Always between `backward()` and `step()`. Clipping after step does nothing; clipping before backward gives empty gradients.

## Interview Questions

**Q1: How does gradient clipping work and when do you use it?**

Gradient clipping bounds the magnitude of the gradient vector before the optimizer step. The standard version is clip-by-global-norm: if the L2 norm of all gradients concatenated exceeds a threshold (typically 1.0), rescale the entire gradient by the ratio $\text{max\_norm} / \lVert g \rVert$. Direction is preserved, magnitude is capped. This prevents catastrophic updates from rare bad batches, especially in RNNs (where gradients can compound multiplicatively over time) and transformer training (where occasional batches have huge attention gradients). Standard for any RNN training and transformer pretraining; cheap insurance for everything else.
