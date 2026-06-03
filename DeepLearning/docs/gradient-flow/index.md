# Gradient Flow

Deep networks have a fundamental gradient-flow problem: the gradient that reaches an early layer is a product of many per-layer factors, and a product of many numbers either collapses toward zero or blows up toward infinity. This is the "keeping it stable" bucket, and almost every architectural trick you know exists to keep that product near one.

!!! tip "Rapid Recall"
    The gradient at layer 1 is a chain of $L$ per-layer Jacobians. If each factor is consistently below 1 the product vanishes geometrically; above 1 it explodes. Sigmoid caps its derivative at 0.25, so deep sigmoid stacks vanish; ReLU and GELU keep the derivative near 1. Exploding gradients are handled with clip-by-norm. For sequences, the LSTM cell state creates a gradient highway where $\partial c_t/\partial c_{t-1} = f_t$; for deep feedforward and transformers, residual connections add an identity bypass $\partial y/\partial x = I + \partial F/\partial x$.

## §1 The core problem

For an $L$-layer network, the gradient at layer 1's weights is a chain of $L$ matrix products. Each layer-to-layer Jacobian is approximately $\partial a_k/\partial a_{k-1} \approx W_k \cdot \sigma'(z_k)$. If the magnitude of each factor is consistently less than 1, the product collapses geometrically (vanishing). If consistently greater than 1, it explodes geometrically. Both are catastrophic, and both stop early layers from learning.

## §2 The fixes, mapped to pages

Each fix gets its own page in this section:

- **Better activations and initialization.** [Activation functions](activation-functions.md): ReLU has derivative 1 in the active region, avoiding sigmoid's 0.25 cap; He initialization keeps activation variance stable across layers.
- **Diagnosing the failure.** [Vanishing gradients](vanishing-gradients.md) shows the sigmoid disaster numerically and why it is especially bad in RNNs, and [exploding gradients](exploding-gradients.md) shows the mirror failure and the standard clip-by-norm fix.
- **Gradient highways.** [LSTM cell state and residual connections](gradient-highways.md): both create a path along which gradients flow unattenuated, the LSTM through time and the residual through depth.
