# Full Backprop Derivation: 2-Layer Network

Now derive backpropagation for the 2-layer classifier from the [Forward Pass](forward-pass.md) page step by step. This is what gets asked in interviews. After working through it once, you should reproduce it on a whiteboard in under 5 minutes.

!!! tip "Rapid Recall"
    The whole derivation hangs off one elegant result: for softmax + cross-entropy, $\partial L/\partial z_2 = (\hat{y} - Y)/B$, the prediction error. From there every other gradient is the matmul backward rule plus shape-matching: $\partial L/\partial W_2 = (\partial L/\partial z_2)^{\top} a_1$, pass the error back with $\partial L/\partial a_1 = (\partial L/\partial z_2) W_2$, gate it through ReLU with $\mathbb{1}[z_1 > 0]$, then $\partial L/\partial W_1 = (\partial L/\partial z_1)^{\top} X$. Biases sum the upstream gradient over the batch. Verify the whole thing against central-difference finite differences; a relative error near $10^{-9}$ means correct, near $10^{-1}$ means a bug.

## §1 The setup

$$
\begin{aligned}
z_1 &= X \cdot W_1^{\top} + b_1 && \text{shape } (B, H) \\
a_1 &= \mathrm{ReLU}(z_1) && \text{shape } (B, H) \\
z_2 &= a_1 \cdot W_2^{\top} + b_2 && \text{shape } (B, C) \text{ — logits} \\
\hat{y} &= \mathrm{softmax}(z_2) && \text{shape } (B, C) \text{ — probabilities} \\
L &= -\frac{1}{B}\sum_i \log \hat{y}_{i, y_i} && \text{scalar}
\end{aligned}
$$

$B$ = batch size, $H$ = hidden dimension, $C$ = number of classes. We want $\partial L/\partial W_1$, $\partial L/\partial b_1$, $\partial L/\partial W_2$, $\partial L/\partial b_2$.

## §2 Step 1: Derivative of loss w.r.t. logits z₂

The famously elegant result. When softmax and cross-entropy are combined and you take the derivative w.r.t. $z_2$, the exp and log cancel out:

$$
\frac{\partial L}{\partial z_2} = \frac{\hat{y} - Y}{B}
$$

Where $Y$ is one-hot labels of shape $(B, C)$: 1 at the correct class, 0 elsewhere. $\hat{y} - Y$ is just the prediction error.

Why this cancels: softmax involves exp, cross-entropy involves log, they are inverses. The algebra is tedious but cancels to this form. **Memorize this result.**

## §3 Step 2: Gradient with respect to W₂

Apply the matmul backward rule. Since $z_2 = a_1 \cdot W_2^{\top} + b_2$:

$$
\frac{\partial L}{\partial W_2} = \left(\frac{\partial L}{\partial z_2}\right)^{\top} \cdot a_1 \quad \text{shape } (C, H)
$$

Plain English: gradient on $W_2$ is the upstream error times the activations feeding this layer. Verify shapes: $\partial L/\partial z_2$ is $(B, C)$, $a_1$ is $(B, H)$, $W_2$ is $(C, H)$. The combination producing $(C, H)$ is $(\partial L/\partial z_2)^{\top} \cdot a_1$.

## §4 Step 3: Gradient with respect to b₂

Bias adds to every sample in the batch. Addition passes gradient through, so we sum across the batch dimension:

$$
\frac{\partial L}{\partial b_2} = \sum_{\text{batch}} \frac{\partial L}{\partial z_2} = \left(\frac{\partial L}{\partial z_2}\right)\text{.sum(axis=0)} \quad \text{shape } (C,)
$$

## §5 Step 4: Gradient with respect to a₁ (pass error back through W₂)

$$
\frac{\partial L}{\partial a_1} = \left(\frac{\partial L}{\partial z_2}\right) \cdot W_2 \quad (B, C) \cdot (C, H) \to (B, H)
$$

Plain English: how much each $a_1$ entry contributed to the loss, weighted by the $W_2$ connections it fed.

## §6 Step 5: Gradient through ReLU

ReLU passes gradient through where input was positive, blocks it where input was non-positive. Elementwise:

$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \mathbb{1}[z_1 > 0] \quad (\odot = \text{elementwise multiply})
$$

## §7 Step 6: Gradient with respect to W₁ and b₁

$$
\begin{aligned}
\frac{\partial L}{\partial W_1} &= \left(\frac{\partial L}{\partial z_1}\right)^{\top} \cdot X && (H, B) \cdot (B, D_{in}) \to (H, D_{in}) \\
\frac{\partial L}{\partial b_1} &= \left(\frac{\partial L}{\partial z_1}\right)\text{.sum(axis=0)} && \text{shape } (H,)
\end{aligned}
$$

## §8 Step 7: Update the weights

$$
W_1 \leftarrow W_1 - \eta \frac{\partial L}{\partial W_1}, \quad
b_1 \leftarrow b_1 - \eta \frac{\partial L}{\partial b_1}, \quad
W_2 \leftarrow W_2 - \eta \frac{\partial L}{\partial W_2}, \quad
b_2 \leftarrow b_2 - \eta \frac{\partial L}{\partial b_2}
$$

$\eta$ (eta) is the learning rate, a small positive number like $10^{-3}$. We subtract because we want to reduce loss, and the gradient points in the direction of increase.

!!! note "The realization"
    If you understand this derivation, you understand backpropagation through any architecture. Convolutions, attention, batch normalization, residual connections, all the same pattern. Each operation has a local backward rule. Chain them in reverse. The shapes force the formula. The complexity of architecture doesn't change the algorithm; you just have more local backward rules to compose.

## §9 Numerical verification

The way to truly believe backprop works is to verify it against finite differences. Implement both, compare, see the agreement to roughly 9 decimal places, internalize.

For each parameter: perturb by $+\varepsilon$, compute loss; perturb by $-\varepsilon$, compute loss; central difference gives the numerical gradient. Compare to analytical from backprop. Compute relative error.

```python
import numpy as np

def numerical_gradient(param, forward_fn, eps=1e-5):
    """Compute gradient by central differences. The ground truth."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + eps
        L_plus = forward_fn()

        param[idx] = original - eps
        L_minus = forward_fn()

        grad[idx] = (L_plus - L_minus) / (2 * eps)
        param[idx] = original   # restore!
        it.iternext()
    return grad

# Compare against analytical gradient from backprop:
rel_error = np.max(np.abs(grad_analytical - grad_numerical) /
                  (np.abs(grad_analytical) + np.abs(grad_numerical) + 1e-12))
# Expect rel_error around 1e-9. If you get 1e-1, you have a bug.
```

### Diagnostic ranges

| Relative error | Diagnosis |
| --- | --- |
| $10^{-7}$ to $10^{-9}$ | Correct. Residual is just floating-point noise from finite differences. |
| $10^{-4}$ to $10^{-6}$ | Likely floating-point precision issues. Tighten $\varepsilon$ or use higher precision. |
| $10^{-1}$ to $10^{-2}$ | Real bug. Usually a missing transpose, wrong sign, or activation derivative at wrong point. |

### Tiny scalar example (verifiable by hand)

```python
import numpy as np
np.random.seed(0)

x, y = 0.7, 1.5
w1, w2 = 1.2, -0.8

def forward(w1, w2):
    z1 = w1 * x
    a1 = max(0, z1)            # ReLU
    y_hat = w2 * a1
    loss = (y_hat - y) ** 2
    return loss, (z1, a1, y_hat)

# Analytical backprop
loss_val, (z1, a1, y_hat) = forward(w1, w2)
dL_dyhat = 2 * (y_hat - y)
dL_dw2 = dL_dyhat * a1
dL_da1 = dL_dyhat * w2
dL_dz1 = dL_da1 * (1 if z1 > 0 else 0)
dL_dw1 = dL_dz1 * x

# Numerical verification
eps = 1e-6
loss_p, _ = forward(w1 + eps, w2)
loss_m, _ = forward(w1 - eps, w2)
dL_dw1_num = (loss_p - loss_m) / (2 * eps)

print(f"Analytical dL/dw1 = {dL_dw1:.6f}")
print(f"Numerical  dL/dw1 = {dL_dw1_num:.6f}")
# Both match to ~6 decimal places.
```

!!! note "Why this matters for you"
    Once you have done this verification by hand, you *know*, not just believe, that backprop works. You can extend the pattern to any architecture. Add another layer? Just chain more local backwards. Different activation? Just change one elementwise rule. The algorithm is robust to architecture changes; only the local backward rules change.

## Interview Questions

**Q1: Walk me through backpropagation for a 2-layer network.**

Forward pass: input → linear+activation twice → loss. Backward pass: compute $\partial L/\partial z_2$ first, for softmax + cross-entropy this simplifies to $(\hat{y} - y) \cdot a_1^{\top}$, the prediction error times the previous activations. Then pass back through $W_2$ to get $\partial L/\partial a_1 = W_2^{\top} \cdot \partial L/\partial z_2$, multiply elementwise by the activation derivative for $\partial L/\partial z_1$, and finally $\partial L/\partial W_1 = \partial L/\partial z_1 \cdot x^{\top}$. Each weight update is just $W \leftarrow W - \eta \cdot \partial L/\partial W$. The chain rule is the core mechanism, each layer's gradient is the downstream gradient times the local derivative, organized so each intermediate is computed once and reused.

**Q2: Why does cross-entropy + softmax give the clean ŷ − y gradient?**

Softmax involves exp, cross-entropy involves log, and these are inverse functions. When you compose them and take the derivative w.r.t. the logits, the exp and log cancel algebraically, leaving just the difference between predicted probability and true label. Formally: $\partial L/\partial z_j = \hat{y}_j - y_j$. This is the "prediction error", a clean, intuitive gradient signal that makes training efficient.

**Q3: Why is finite differences a good way to verify backprop?**

Finite differences computes the gradient directly from the definition of a derivative: $(L(w+\varepsilon) - L(w-\varepsilon)) / (2\varepsilon)$. It's mathematically unambiguous and doesn't depend on any complex algorithm, just run the forward pass twice. So it's the ground truth. You compare your backprop implementation's analytical gradient to finite differences. Relative error around $10^{-9}$ means you're correct (the residual is just floating-point noise). Relative error around $10^{-1}$ means you have a bug, most commonly a missing transpose, wrong sign, or activation derivative evaluated at the wrong point.
