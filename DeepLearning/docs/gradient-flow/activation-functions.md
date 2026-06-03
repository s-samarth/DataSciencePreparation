# Activation Functions

Activations are the whole reason depth is useful. Remove them and a 100-layer network collapses to a single linear layer. The choice of activation also directly controls whether gradients survive depth.

!!! tip "Rapid Recall"
    The derivative of the activation is what gets multiplied into the gradient at each layer, so its shape decides whether gradients survive. Sigmoid and tanh saturate (derivative goes to 0 at the tails), which kills deep stacks; sigmoid's peak derivative of 0.25 is especially bad. ReLU has derivative exactly 1 where active, so it does not shrink gradients, but neurons can "die." GELU and SiLU are smooth, non-saturating, and standard in transformers and modern LLMs. Use sigmoid only at a binary output, ReLU/LeakyReLU for CNNs and MLPs, GELU/SiLU for transformers.

<figure class="diagram diagram-dark" markdown="0">
  <img src="../../assets/img/activations.png" alt="Activation functions and their derivatives plotted over the input range">
  <figcaption>Activation shapes and their derivatives (dashed). Sigmoid and tanh flatten out at the tails so their derivatives go to zero, while ReLU stays at slope 1 wherever it is active. The derivative is what gets multiplied into the gradient at each layer during backprop.</figcaption>
</figure>

## §1 Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1 - \sigma(z)), \qquad \text{max derivative} = 0.25 \text{ (at } z = 0)
$$

Output bounded in $(0, 1)$. Useful as the final activation for binary classification (probability output). Catastrophic in hidden layers: a peak derivative of 0.25 means gradients shrink by at least 4x per layer. After 10 layers: $0.25^{10} \approx 10^{-6}$. Gradients vanish. The full story is on the [vanishing gradients](vanishing-gradients.md) page.

## §2 Tanh

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh'(z) = 1 - \tanh(z)^2, \qquad \text{max derivative} = 1.0 \text{ (at } z = 0)
$$

Output bounded in $(-1, 1)$, zero-centered. Slightly better than sigmoid for hidden layers (peak derivative is 1, not 0.25), but still saturates at large $|z|$. Used in old RNNs; mostly obsolete now.

## §3 ReLU (Rectified Linear Unit)

$$
\mathrm{ReLU}(z) = \max(0, z), \qquad \mathrm{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & \text{otherwise} \end{cases}
$$

The fix for vanishing gradients in CNNs. The derivative is exactly 1 in the active region, so gradients do not shrink through ReLU. No exp computation either, the fastest activation. Standard for CNNs and MLPs.

!!! warning "Dying ReLU problem"
    If a neuron's input is always negative, ReLU outputs 0, and the gradient through it is 0. The weights feeding into it never update. The neuron is permanently dead. Caused by a too-large learning rate, bad initialization, or unlucky updates pushing weights to a region where the unit is always inactive. Diagnose by tracking the fraction of zero activations, if more than 50% of a layer is always dead, you have a problem. Fixes: lower learning rate, He initialization, LeakyReLU.

## §4 LeakyReLU

$$
\mathrm{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases} \quad (\alpha \approx 0.01), \qquad \mathrm{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & \text{otherwise} \end{cases}
$$

A small slope ($\alpha \approx 0.01$) in the negative region. Never truly dies because the gradient is $\alpha$, not 0, even for negative inputs. Use if you see dying ReLU problems with vanilla ReLU.

## §5 GELU (Gaussian Error Linear Unit)

$$
\mathrm{GELU}(z) \approx z \cdot \Phi(z) \quad (\Phi = \text{standard normal CDF})
$$

$$
\mathrm{GELU}(z) \approx 0.5 \cdot z \cdot \left(1 + \tanh\!\left(\sqrt{2/\pi}\,\big(z + 0.044715\,z^3\big)\right)\right)
$$

Smooth, non-monotonic. Stochastically gates based on input magnitude, large positive $z$ passes through, large negative $z$ is suppressed, the transition is smooth. Empirically outperforms ReLU on transformers. **Standard for BERT, GPT, Claude, and most modern transformer architectures.**

## §6 SiLU / Swish

$$
\mathrm{SiLU}(z) = z \cdot \sigma(z) \quad \text{(sigmoid-weighted linear unit)}
$$

Similar to GELU. Used in LLaMA and some modern architectures. Slightly cheaper than GELU because it only requires sigmoid, not the more complex normal CDF approximation.

## §7 Comparison

| Activation | Form | Vanishing gradient? | Dead neurons? | Used in |
| --- | --- | --- | --- | --- |
| Sigmoid | $1/(1+e^{-z})$ | Yes, max $\sigma' = 0.25$ | No | Binary output layer only |
| Tanh | $(e^z - e^{-z})/(e^z + e^{-z})$ | Yes, saturates | No | Old RNNs |
| ReLU | $\max(0, z)$ | No (active region) | Yes (dying ReLU) | CNNs, MLPs |
| LeakyReLU | $z$ or $\alpha z$ | No | No | CNN alternative |
| GELU | $z \cdot \Phi(z)$ | No | No (smooth) | Transformers (BERT, GPT) |
| SiLU/Swish | $z \cdot \sigma(z)$ | No | No | LLaMA, some LLMs |

## Interview Questions

**Q1: What's the dying ReLU problem and when does it matter?**

A ReLU neuron "dies" when its pre-activation $z$ is always negative, the gradient is 0 and the neuron never updates regardless of the training signal. This happens when weights are initialized poorly or the learning rate is too high and a large negative update permanently pushes a neuron into the negative region. In practice it matters most in deep networks or networks with high learning rates. Diagnostic: track the fraction of zero activations. If more than 50% of a layer's neurons are always 0, you have a problem. Fixes: LeakyReLU, careful initialization (He init for ReLU), lower LR. GELU and other smooth activations don't have this problem.
