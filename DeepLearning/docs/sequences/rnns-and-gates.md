# RNNs & Gated Cells

To process a sequence you need to remember what you've seen, that memory is the hidden state, updated every step with the same shared weights. Vanilla RNNs do this but their gradients vanish over long ranges; LSTMs and GRUs fix it with a gated, additive memory.

!!! tip "Rapid Recall"
    An RNN carries a hidden state $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$, reusing the same weights every step, trained by backprop-through-time. Because BPTT repeatedly multiplies by $W_{hh}$ and $\tanh'$, gradients vanish (< 1) or explode (> 1); clipping patches exploding, but vanishing is the structural flaw. The LSTM adds a cell state $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ edited by forget/input/output gates; because the cell update is *additive*, when $f_t \approx 1$ the gradient multiplies by ~1, a highway across time (the same idea as ResNet). GRU merges the gates and the states, fewer params, comparable quality.

## §1 Sequence modeling and vanilla RNNs

An MLP processes a fixed input in one shot, no "before"/"after." For a sequence you need to process elements **one at a time while remembering what you've seen**. That memory is the **hidden state** $h_t$, a vector summarizing the sequence so far, updated every step. The defining idea: apply the **same weights at every timestep** (weight sharing across time, like CNNs share across space) and feed the previous output back in. It's a loop.

```
        ┌──────────────────────────────────┐
        │            (same weights)          │
        ▼                                    │
x_t ──►[ RNN cell ]──► h_t ──► output y_t    │
            ▲                                │
            └────────── h_{t-1} ─────────────┘
Unrolled:   x1     x2     x3     x4
             │      │      │      │
            [A]─►──[A]─►──[A]─►──[A]      A = same cell reused
             │      │      │      │
            h1     h2     h3     h4   (each h depends on all earlier x)
```

The equations:

$$
h_t = \tanh\!\big(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b_h\big), \qquad y_t = W_{hy}\,h_t + b_y
$$

The new hidden state is a squashed mix of the **current input** ($W_{xh}x_t$) and the **entire past compressed into $h_{t-1}$** ($W_{hh}h_{t-1}$). $W_{hh}$ is the recurrent weight, applied every step. $h_0$ is usually zeros. For language modeling, $y_t$ becomes logits then softmax then next-token probs.

It is trained by **Backprop Through Time (BPTT)**: unroll the loop into a deep feedforward net (one layer per timestep, all sharing weights), then normal backprop. Since the same $W_{hh}$ is used every step, its gradient is the **sum** over all timesteps.

!!! warning "The fatal flaw: vanishing / exploding gradients"
    Backprop through many timesteps repeatedly multiplies by $W_{hh}$ (and $\tanh$ derivatives):

    $$
    \frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} \approx \prod \big(W_{hh}^\top \cdot \text{diag}(\tanh')\big)
    $$

    Multiply a number by itself 100 times: if < 1 it shrinks to **0** (gradient **vanishes**, can't learn dependencies past ~10 to 20 steps, "I grew up in France … I speak fluent ___" loses the link); if > 1 it blows up to **∞** (gradient **explodes**, NaNs). Exploding is patchable with [gradient clipping](../gradient-flow/exploding-gradients.md); vanishing is the structural problem, the entire reason LSTMs were invented.

## §2 LSTMs and GRUs

The vanilla RNN rewrites its hidden state every step ($h_t = \tanh(\dots)$), so old info is repeatedly mangled and decays. The LSTM adds a separate **cell state** $C_t$, a memory conveyor belt running straight through with only minor, gated, **mostly-additive** edits. Information rides the belt across many steps almost untouched. Small neural gates decide what to **erase**, **write**, and **read**. The key word is **additive**, addition doesn't vanish the way repeated multiplication does.

```
  cell state C: ═══►(×)═══►(+)═══════►  C_t   (the conveyor belt)
                     ▲      ▲
                  forget  write
                   gate   gate
  hidden state h ──────────────────────► h_t  (the read-out)
```

The gates ($\sigma$ = sigmoid, output in [0,1] = "how much to let through"). Forget gate (what to erase):

$$
f_t = \sigma(W_f\,[h_{t-1}, x_t] + b_f)
$$

Input gate + candidate (what to write):

$$
i_t = \sigma(W_i\,[h_{t-1}, x_t] + b_i), \qquad \tilde{C}_t = \tanh(W_C\,[h_{t-1}, x_t] + b_C)
$$

Cell update (the heart, additive):

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

Output gate + read-out:

$$
o_t = \sigma(W_o\,[h_{t-1}, x_t] + b_o), \qquad h_t = o_t \odot \tanh(C_t)
$$

$\odot$ = elementwise multiply. $C_{t-1}$ is *added* in, not pushed through a weight matrix, which is why the gradient survives. $C_t$ is long-term memory; $h_t$ is the filtered short-term read-out.

Why it fixes vanishing gradients: along the cell state $C_t = f_t \odot C_{t-1} + \dots$, when the forget gate $f_t \approx 1$, $\partial C_t/\partial C_{t-1} \approx 1$, so backprop multiplies by **~1 repeatedly** instead of by < 1. A near-constant gradient highway across time, structurally the *same idea* as [ResNet's](../cnns/architectures.md) $C_t \approx C_{t-1} + \text{stuff}$. Long-range dependencies survive (hundreds of steps vs ~10 to 20).

**GRU (the lighter sibling):** merges forget + input into a single **update gate**, and merges cell + hidden state into one. Two gates, fewer params, about as good, trains faster. "GRU = LSTM with fewer gates, no separate cell state; comparable performance, less compute."

!!! note "Where the special tokens bite in RNN/LSTM land"
    Pad batches to equal length with `<PAD>` and use `pack_padded_sequence`/`pad_packed_sequence` so the LSTM **skips pad steps** (the final hidden state reflects the real last token, not padding). Mask `<PAD>` out of the loss (`CrossEntropyLoss(ignore_index=pad_id)`) so padding contributes zero gradient. Seed generation with `<SOS>`, stop at `<EOS>`.
