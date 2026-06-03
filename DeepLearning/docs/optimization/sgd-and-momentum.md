# SGD & Momentum

The baseline. Every other optimizer is just a more sophisticated update rule on top of vanilla SGD, and momentum is the first and most important improvement.

!!! tip "Rapid Recall"
    SGD is one line, $\theta \leftarrow \theta - \eta g$, with zero extra memory and the learning rate as its only knob. It is slow, very LR-sensitive, and oscillates in narrow ravines, but it sometimes generalizes better than adaptive methods and is still the standard CNN recipe. Momentum moves along an EMA of recent gradients (a heavy ball), so consistent directions accelerate and oscillations cancel; it costs one extra tensor (2× memory) and $\beta$ is almost always 0.9. The standard vision recipe is $\eta = 0.1$, $\beta = 0.9$, weight decay $10^{-4}$, with a step or cosine schedule.

## §1 SGD (vanilla)

| Symbol | Meaning |
| --- | --- |
| $\theta$ | The parameter being updated (a single weight, or the whole weight vector, same form). |
| $\eta$ | Learning rate. A small positive number controlling step size. |
| $g$ | The gradient $\nabla L(\theta)$ computed on the current mini-batch. |

$$
\theta \leftarrow \theta - \eta \cdot g
$$

Subtract a scaled gradient. That's it. No state, no memory. The whole algorithm fits in one line.

### Memory overhead per parameter

**Zero extra.** You only need to store the parameter itself. SGD is the most memory-efficient optimizer.

### Hyperparameters

- **$\eta$ (learning rate)**, the only knob. Typical range: $10^{-4}$ to $10^{-1}$. Wrong LR is the most common cause of training failure. Too high: training diverges or oscillates. Too low: training is glacially slow.

### Strengths and weaknesses

**Strengths:** Zero memory overhead. Sometimes generalizes *better* than adaptive optimizers, there is evidence that SGD finds "flatter" minima which generalize better, especially in CNNs. Still standard for training ResNets and many vision models.

**Weaknesses:** Converges slowly. Same LR for every parameter, bad for problems where parameters have very different gradient magnitudes (embeddings vs final layer in NLP). Very sensitive to LR choice. Oscillates in narrow ravines of the loss landscape (steep in one direction, shallow in another).

### When to use

CNN training where you have time to tune carefully. Fine-tuning where you want stability over speed.

## §2 SGD with momentum

The first improvement on SGD. Instead of moving in the direction of the current gradient, move in the direction of an exponential moving average (EMA) of recent gradients. Think of a heavy ball rolling down a hill, it accumulates speed in consistent directions and ignores small fluctuations.

| Symbol | Meaning |
| --- | --- |
| $v$ | Velocity, accumulated direction of motion. Same shape as $\theta$. Initialized to zero. |
| $\beta$ | Momentum coefficient. Typically 0.9. Larger = more smoothing, slower to react. |

$$
\begin{aligned}
\textbf{PyTorch's formulation:}\quad & v \leftarrow \beta \cdot v + g, \qquad \theta \leftarrow \theta - \eta \cdot v \\[4pt]
\textbf{"True EMA" formulation:}\quad & v \leftarrow \beta \cdot v + (1 - \beta) \cdot g, \qquad \theta \leftarrow \theta - \eta \cdot v
\end{aligned}
$$

(The two are mathematically equivalent up to LR rescaling.)

### Why it works

The velocity is a running average of gradients. Where gradients consistently point the same way (for example, a gentle slope toward the minimum), velocities *accumulate* and the step size grows. Where gradients oscillate (for example, across a narrow ravine), velocities *cancel out* and the step stays small. Net effect: faster convergence, less oscillation.

### Memory overhead

One extra tensor $v$ of the same shape as $\theta$. So **2× the memory** of vanilla SGD.

### Hyperparameters

- **$\eta$ (learning rate)**, same role as before. Tune.
- **$\beta$ (momentum coefficient)**, almost always 0.9. Sometimes 0.99 for very long training runs. Rarely worth tuning.

### Nesterov momentum (variant)

A slight tweak that "looks ahead", compute the gradient at the position you are *about to* be at, not where you currently are:

$$
v \leftarrow \beta \cdot v + \nabla L(\theta - \eta \cdot \beta \cdot v), \qquad \theta \leftarrow \theta - \eta \cdot v
$$

Usually 1-2% better than vanilla momentum. Available as `nesterov=True` in PyTorch's SGD. Rarely makes or breaks a project.

### When to use

Default choice for vision tasks. SGD + momentum still trains ImageNet-scale models competitively in 2026. The combination of $\eta = 0.1$, $\beta = 0.9$, weight decay $= 10^{-4}$, and a step or cosine schedule is the standard CNN recipe. The next page, [adaptive methods](adaptive-methods.md), shows what changes when each parameter gets its own learning rate.
