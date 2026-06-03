# Adaptive Methods: AdaGrad → RMSProp → Adam → AdamW

The adaptive family gives each parameter its own effective learning rate based on its gradient history. The lineage is a chain of fixes: AdaGrad introduces the idea but decays to zero, RMSProp fixes the decay with an EMA, Adam adds momentum on top, and AdamW fixes Adam's broken weight decay. AdamW is the default for essentially all transformer training in 2026.

!!! tip "Rapid Recall"
    AdaGrad divides by the square root of the running sum of squared gradients, so its effective LR monotonically decays to zero (its fatal flaw). RMSProp replaces the sum with an EMA so old gradients are forgotten. Adam = momentum + RMSProp: it tracks a first moment $m$ and second moment $v$, applies bias correction (because both start at zero), and updates with $\theta \leftarrow \theta - \eta\, \hat{m}/(\sqrt{\hat{v}} + \varepsilon)$. Adam stores $m$ and $v$, so 3× parameter memory. AdamW decouples weight decay, applying it directly to the weights instead of through the adaptive denominator, which is why every modern LLM uses it.

## §1 AdaGrad

Included not because you'll use it (you won't) but because it's the conceptual ancestor of RMSProp and Adam. It introduces the *adaptive learning rate* idea, give each parameter its own effective step size based on its own gradient history.

Different parameters need different learning rates. A weight that consistently gets large gradients should take small steps. A weight that gets small gradients should take large steps. So divide each gradient by the square root of the sum of all past squared gradients.

| Symbol | Meaning |
| --- | --- |
| $G$ | Running sum of squared gradients, same shape as $\theta$. Initialized to zero. |
| $\varepsilon$ | Tiny constant ($10^{-8}$) to prevent division by zero. |

$$
G \leftarrow G + g^2 \quad \text{(elementwise; accumulate squared grads)}, \qquad \theta \leftarrow \theta - \eta \cdot \frac{g}{\sqrt{G} + \varepsilon}
$$

### The fatal flaw

$G$ only ever grows. It never shrinks. So the effective learning rate $\eta / \sqrt{G}$ monotonically decreases to zero. After enough iterations, the optimizer freezes, every parameter is dividing by an enormous number. Useless for long training runs. Memory overhead is **2×** (one extra tensor $G$).

Almost never used anymore. Sometimes used for very sparse problems (NLP with one-hot features) where many parameters get gradients infrequently and you actually want rare-update parameters to take big steps. But Adam/AdamW have largely replaced it.

## §2 RMSProp

Fixes AdaGrad's decay-to-zero problem. Instead of summing all past squared gradients (which grows forever), use an exponential moving average. Old gradients are forgotten; the optimizer can keep moving.

| Symbol | Meaning |
| --- | --- |
| $v$ | EMA of squared gradients. Same shape as $\theta$. Initialized to zero. |
| $\beta$ | Decay rate. Typically 0.99 (slow forgetting). |

$$
v \leftarrow \beta \cdot v + (1 - \beta) \cdot g^2, \qquad \theta \leftarrow \theta - \eta \cdot \frac{g}{\sqrt{v} + \varepsilon}
$$

Each parameter gets a personalized learning rate that adapts to its own gradient history. Parameters with consistently large gradients get smaller effective LRs. Parameters with small gradients get larger effective LRs. The EMA means old history is forgotten gradually, the optimizer adapts to non-stationary problems.

!!! note "Important clarification"
    RMSProp does **not** use momentum in the way SGD+Momentum does. It tracks squared gradients (a variance-like statistic), not gradients themselves. Adam adds proper momentum on top of this.

Hyperparameters: $\eta$ typically $10^{-3}$ to $10^{-4}$, $\beta$ (decay rate) 0.99 default, $\varepsilon = 10^{-8}$. Memory overhead **2×** (one extra tensor $v$). RMSProp was the go-to for RNNs and LSTMs in the 2014-2016 era. Now mostly replaced by Adam/AdamW. Still occasionally used for RL.

## §3 Adam

The optimizer of modern deep learning. Default for transformers, default for most new projects. **Adam = Momentum + RMSProp.** It combines smoothed gradients with per-parameter adaptive learning rates.

| Symbol | Meaning |
| --- | --- |
| $m$ | First moment estimate, EMA of gradients (momentum). Shape of $\theta$. |
| $v$ | Second moment estimate, EMA of squared gradients (RMSProp). Shape of $\theta$. |
| $\beta_1$ | Decay rate for first moment. Default 0.9. |
| $\beta_2$ | Decay rate for second moment. Default 0.999. |
| $\hat{m}, \hat{v}$ | Bias-corrected first and second moment estimates. |
| $t$ | Step counter, starts at 1. |
| $\varepsilon$ | $10^{-8}$, numerical safety. |

$$
\begin{aligned}
\textbf{Update moment estimates:}\quad
m &\leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g && \text{first moment, momentum} \\
v &\leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2 && \text{second moment, RMSProp} \\[4pt]
\textbf{Bias correction:}\quad
\hat{m} &= \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}} \\[4pt]
\textbf{Update:}\quad
\theta &\leftarrow \theta - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}
\end{aligned}
$$

### Why bias correction matters

Both $m$ and $v$ are initialized to zero. At step 1, with $\beta_1 = 0.9$:

$$
m_1 = 0.9 \cdot 0 + 0.1 \cdot g_1 = 0.1 \cdot g_1
$$

So $m_1$ is only 10% of $g_1$, heavily biased toward zero. Without correction, early training takes tiny steps. The bias correction $\hat{m} = m / (1 - \beta_1^{\,t})$ exactly undoes this:

$$
\hat{m}_1 = \frac{0.1 \cdot g_1}{1 - 0.9} = \frac{0.1 \cdot g_1}{0.1} = g_1
$$

After many steps, $\beta_1^{\,t} \to 0$, so $\hat{m} \approx m$. The correction becomes negligible after the first few hundred steps. Easy to ignore, but it matters for the early phase of training.

### Default hyperparameters

Do not change these unless you have a reason. They are the empirically validated defaults.

- **$\beta_1 = 0.9$** (momentum decay)
- **$\beta_2 = 0.999$** (squared gradient decay)
- **$\varepsilon = 10^{-8}$**
- **$\eta = 3 \times 10^{-4}$** ("Karpathy's constant" for safe defaults). $10^{-3}$ often works too.

### Memory overhead, the 3× problem

Per parameter, Adam stores the parameter itself (1×), $m$ the first moment (1×, same shape), and $v$ the second moment (1×, same shape). **Total: 3× the parameter memory.**

!!! warning "The 7B model concrete example"
    A 7B parameter model in fp32 (4 bytes per parameter): model weights $= 7\text{B} \times 4 = 28$ GB, Adam $m$ state $= 28$ GB, Adam $v$ state $= 28$ GB, for a **total optimizer memory of 84 GB**. And that's before adding gradients (another 28 GB) and activations for backward (often the largest of all). This is why training large LLMs uses memory-efficient variants: 8-bit Adam (bitsandbytes), sharded optimizers (ZeRO, FSDP), or alternatives like [Lion](modern-optimizers.md) (2× memory).

**Strengths:** Works well across most architectures and tasks with minimal tuning. Combines the best of momentum and adaptive rates. Handles sparse gradients well. Robust to hyperparameter choices. **Weaknesses:** 3× memory overhead. Sometimes generalizes worse than SGD+Momentum (especially on CNNs). Weight decay is broken, which AdamW fixes.

## §4 AdamW

Adam with a critical bug fix: decoupled weight decay. Standard for all transformer training in 2026, GPT, LLaMA, Claude, every modern LLM uses AdamW or a close variant.

### The bug AdamW fixes

In Adam, the "weight decay" hyperparameter is implemented as L2 regularization, add $(\lambda/2)\lVert\theta\rVert^2$ to the loss. The gradient gains an extra term: $g + \lambda\theta$. Adam then takes this combined gradient and scales it by $1/\sqrt{\hat{v}}$, which means **the effective weight decay also gets scaled by the adaptive factor**. Parameters with large gradient history get less weight decay than parameters with small gradient history. That wasn't what regularization was supposed to do.

### AdamW's fix

Apply weight decay *directly to the parameters*, not to the gradient. Decouple it from the adaptive scaling.

$$
\begin{aligned}
m &\leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g \\
v &\leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2 \\
\hat{m} &= m / (1 - \beta_1^{\,t}), \qquad \hat{v} = v / (1 - \beta_2^{\,t}) \\[4pt]
\theta &\leftarrow \theta - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon} - \underbrace{\eta \cdot \lambda \cdot \theta}_{\text{decay applied directly, NOT through } \hat{v}}
\end{aligned}
$$

The weight decay term $\eta \cdot \lambda \cdot \theta$ is applied independently of the gradient-based update. Every parameter gets uniformly shrunk by $(1 - \eta\lambda)$ each step, regardless of gradient history. See the [weight decay](../regularization/regularization.md) page for the three lenses on why this regularizes.

### Default hyperparameters

- Same as Adam: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$
- **$\eta$**, $10^{-4}$ to $10^{-3}$ (often $3 \times 10^{-4}$)
- **$\lambda$ (weight decay)**, 0.01 to 0.1 for transformers. *Much larger* than the equivalent L2 reg in Adam because the decoupled formulation makes it stronger.

Memory overhead is the same as Adam, **3× parameter memory**. Use AdamW almost always for transformer training, and almost always for any new project where you want weight decay. **It is the default in 2026.**

!!! warning "PyTorch gotcha"
    `torch.optim.Adam(weight_decay=0.01)` is **NOT AdamW**, it is the broken L2-style Adam. To get AdamW, use `torch.optim.AdamW` explicitly. This bug appears in plenty of published code.

## Interview Questions

**Q1: Why use AdamW instead of Adam + L2 regularization?**

In Adam, the gradient gets scaled by $1/\sqrt{\hat{v}}$, the adaptive term based on squared gradient history. When you add L2 regularization, the weight decay term $\lambda\theta$ gets scaled by the same factor, which reduces its effectiveness for parameters with large gradient history. AdamW decouples the weight decay from the gradient update: it applies $\theta \leftarrow (1 - \eta\lambda)\theta$ separately from the Adam step. This means every parameter gets the same effective weight decay, which is the intended behavior of L2 regularization. This is why all modern LLM training uses AdamW.

**Q2: Why does Adam have 3× memory overhead?**

Adam stores two extra tensors per parameter: $m$ (first moment estimate, EMA of gradients) and $v$ (second moment estimate, EMA of squared gradients). Both have the same shape as the parameter. So total optimizer memory is 3× the parameter count: the parameter itself plus $m$ plus $v$. For a 7B parameter model in fp32, that's 84GB just for optimizer state. This is why memory-efficient variants like 8-bit Adam (bitsandbytes), sharded optimizers (ZeRO, FSDP), and alternatives like Lion (2× memory) exist.

**Q3: What does bias correction in Adam do, and why is it needed?**

Adam's first and second moment estimates $m$ and $v$ are initialized to zero. At step 1 with $\beta_1 = 0.9$, $m_1 = 0.9 \cdot 0 + 0.1 \cdot g_1 = 0.1 \cdot g_1$, only 10% of the true gradient. Without correction, early training takes tiny steps. Bias correction $\hat{m} = m / (1 - \beta_1^{\,t})$ exactly undoes the initialization bias. At step 1: $\hat{m}_1 = 0.1 \cdot g_1 / (1 - 0.9) = g_1$, exactly the gradient. After many steps, $\beta^t \to 0$ and the correction becomes negligible. Easy to miss but it matters for the first few hundred steps.
