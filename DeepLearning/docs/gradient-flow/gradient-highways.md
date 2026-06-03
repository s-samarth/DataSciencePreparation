# Gradient Highways: LSTM and Residual Connections

Both the vanishing and exploding problems come from gradients being repeatedly multiplied through depth or time. The two great fixes both work the same way: they build a path along which the gradient flows *without* being multiplied by a potentially small or large factor. LSTMs build that highway through time; residual connections build it through depth.

!!! tip "Rapid Recall"
    An LSTM adds a cell state with the update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$; the key fact is $\partial c_t/\partial c_{t-1} = f_t$, just the forget gate, with no saturating-activation derivative and no repeated weight matrix. When long-range memory matters the network learns $f_t \approx 1$ and gradients flow unattenuated (the constant error carousel). A residual block computes $y = x + F(x)$, so $\partial y/\partial x = I + \partial F/\partial x$; the identity term is a bypass that carries the gradient even when $F$'s gradient is tiny. This is why ResNets reach 152+ layers and every transformer block is wrapped in residuals.

## §1 The LSTM cell

Vanilla RNNs cannot learn long-range dependencies because gradients vanish or explode through time. LSTMs fix this with a cleverly designed cell state that creates a gradient highway, a path through time where gradients flow without being repeatedly multiplied by potentially-small numbers.

| Symbol | Meaning |
| --- | --- |
| $c_t$ | Cell state at time $t$, the "memory" that flows through time. |
| $h_t$ | Hidden state at time $t$, output of the cell. |
| $f_t$ | Forget gate at time $t$, values in $(0, 1)$ controlling how much of $c_{t-1}$ to keep. |
| $i_t$ | Input gate at time $t$, how much of new info to add. |
| $\tilde{c}_t$ | Candidate cell state, the proposed update. |
| $o_t$ | Output gate at time $t$, controls what the cell exposes as $h_t$. |
| $\odot$ | Elementwise multiplication. |

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) && \text{forget gate} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) && \text{input gate} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) && \text{output gate} \\
\tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) && \text{candidate} \\[4pt]
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{the crucial cell-state update} \\
h_t &= o_t \odot \tanh(c_t) && \text{hidden state output}
\end{aligned}
$$

### The gradient highway

The magic is in the cell-state update. Compute the gradient of $c_t$ with respect to $c_{t-1}$:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

That is it. Just the forget gate. No derivative of a saturating activation function. No repeated weight matrix.

If the network learns to keep $f_t \approx 1$ for time steps where long-range information matters, the gradient flows through unattenuated across many time steps. This is the *constant error carousel*, a path through time where gradients ride without being repeatedly squished or amplified.

### Why this works

Compare to the vanilla RNN, where the gradient through time involves $\partial h_t/\partial h_{t-1} = W_h^{\top} \cdot \mathrm{diag}(\sigma'(h_t))$ at every step. Two problems:

- The matrix $W_h$ is repeated. If its spectral norm differs from 1, gradients explode or vanish.
- The activation derivative $\sigma'$ caps at 1 (tanh) or 0.25 (sigmoid). Multiplied across many steps, it vanishes.

LSTM's cell-state path has *neither* of these. $\partial c_t/\partial c_{t-1} = f_t$, just a number in $(0, 1)$ that the network can learn to set near 1 when it needs to preserve information.

### GRU, a simpler variant

Gated Recurrent Unit. Two gates instead of four (update gate and reset gate). Combines cell state and hidden state. Roughly comparable to LSTM in quality, fewer parameters, simpler to implement. In practice, LSTM and GRU give very similar results; LSTM is more historically dominant.

!!! note "What to say in an interview"
    "LSTM introduces a cell state $c_t$ with the update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The gradient $\partial c_t/\partial c_{t-1}$ equals just the forget gate $f_t$, which is a learnable value in $(0, 1)$. When the network needs long-range memory, it learns to keep $f_t$ near 1, and gradients flow through many time steps unattenuated. This is called the constant error carousel and is what allows LSTMs to capture dependencies that vanilla RNNs cannot."

## §2 Residual connections

For feedforward networks, the analog of LSTM's gradient highway is the residual connection. Stacking 100+ layers became practical only after this idea. Every modern deep network (ResNets, transformers, ViTs) uses it.

### The structure

A residual block computes:

$$
y = x + F(x)
$$

where $F$ is some learned transformation (for example, two conv layers, or attention + FFN). The output is the input plus a learned correction. $F(x)$ is the "residual", what to add to $x$ to improve it.

### The gradient highway

$$
\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}
$$

That identity matrix $I$ is the magic. Even if $F$'s gradient $\partial F/\partial x$ becomes tiny (or zero), the gradient still flows through unchanged via the identity path. **The gradient highway runs around $F$, parallel to it.**

### Why this enables deep networks

Without residuals: the gradient at an early layer is a product of $L$ Jacobians. Each one $\le 1$ means vanishing. Each one $\ge 1$ means exploding.

With residuals: the gradient at an early layer is a sum over paths. There is a "shortcut path" that bypasses any individual block. Even if 95% of layers' $F$ gradients are tiny, the identity bypasses get the signal through.

This is why ResNets enabled networks with 152+ layers, when plain CNNs topped out around 20 layers. The same fix applies to transformers, every transformer block is structured as $y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x))$ and $y = x + \mathrm{FFN}(\mathrm{LayerNorm}(x))$.

!!! note "Mental model"
    Without residuals, the gradient walks through a long, dark tunnel and gets weaker at every step. With residuals, there's a parallel express lane that bypasses each layer's tunnel. The express lane preserves the gradient at full strength. If any block's local gradient is small, the network can effectively learn to "skip" it without losing the upstream signal.

### Pre-LN vs post-LN in transformers

Two arrangements of LayerNorm within a residual block:

$$
\begin{aligned}
\text{Post-LN (original Transformer, 2017):}\quad & y = \mathrm{LayerNorm}(x + \mathrm{Attention}(x)) \\
\text{Pre-LN (modern, more stable):}\quad & y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x))
\end{aligned}
$$

Pre-LN is now standard (GPT, LLaMA, most modern transformers) because the residual path is "clean", gradients flow through pure identity, not modulated by LayerNorm. Easier to train deep models, less sensitive to LR. More on the normalization layers themselves is on the [normalization](../regularization/normalization.md) page.

## Interview Questions

**Q1: How does LSTM solve the vanishing gradient problem?**

The LSTM cell state update is $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The gradient $\partial c_t/\partial c_{t-1}$ equals just the forget gate $f_t$, which is a learnable value in $(0, 1)$. No derivative of a saturating activation function, no repeated weight matrix. When the network needs long-range memory, it learns to keep $f_t$ near 1, and gradients flow through many time steps unattenuated. This is called the constant error carousel and is what allows LSTMs to capture long-range dependencies that vanilla RNNs cannot.

**Q2: How do residual connections enable training very deep networks?**

A residual block computes $y = x + F(x)$, so the gradient $\partial y/\partial x = I + \partial F/\partial x$. That identity matrix $I$ is a gradient highway, even if $F$'s gradient $\partial F/\partial x$ is tiny or zero, the gradient still flows through unchanged via the identity path. This means the network can effectively learn to "skip" any block by setting its $F$ output to near zero, without losing the upstream gradient signal. This is what enabled ResNets to go to 152+ layers when plain CNNs topped out around 20. Every transformer block uses the same pattern: $y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x))$ and $y = x + \mathrm{FFN}(\mathrm{LayerNorm}(x))$.
