# Generative Adversarial Networks (GANs)

A GAN is a forger versus a detective: two networks in competition, where one network learns the loss for the other. The genius is that no human writes a "make it look real" loss, the discriminator learns it and adapts as the generator improves.

!!! tip "Rapid Recall"
    Generator G turns noise into fakes; discriminator D outputs the probability an image is real. They train adversarially via the minimax game; at equilibrium G's fakes are indistinguishable and D guesses 0.5. In practice G is trained to maximize $\log D(G(z))$ (the non-saturating loss) because the original objective gives vanishing gradients when G is losing. The classic failures are mode collapse, training instability, vanishing gradients to G, no loss-to-quality correlation, and hard evaluation (use FID). WGAN-GP is the big fix. GAN = fast, one-shot, unstable; diffusion = slow, iterative, stable.

## §1 Intuition, a forger vs a detective

Two networks in competition. **Generator G** (the forger) takes random noise $z$ and produces a fake image $G(z)$; its goal is to fool the discriminator. **Discriminator D** (the detective) takes an image and outputs the probability it's real; its goal is to tell real from fake. They train adversarially, each forces the other to improve. At equilibrium, G's fakes are indistinguishable and D is reduced to guessing (0.5 everywhere). The genius: **no human writes a "make it look real" loss**, D *learns* the loss for G and adapts as G improves.

```
   noise z ─► [Generator G] ─► fake image ─┐
                                            ├─► [Discriminator D] ─► real? (0..1)
            real image ───────────────────┘
   D's goal: real→1, fake→0.   G's goal: make D output 1 on fakes.
```

## §2 The minimax game

$$
\min_G \max_D \; V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]
$$

**D maximizes:** push $D(x) \to 1$ on real (first term → 0, the max), push $D(G(z)) \to 0$ on fakes (second term → 0). D is just a binary classifier; its loss is BCE on real-vs-fake. **G minimizes:** G touches only the second term and wants $D(G(z)) \to 1$ (fool D), driving $\log(1 - D(G(z))) \to -\infty$.

!!! note "The practical fix you must mention"
    Early on, fakes are obviously bad, $D(G(z)) \approx 0$, and the gradient of $\log(1 - D(G(z)))$ there is nearly flat, so G learns nothing (vanishing gradient). In practice G is trained to **maximize $\log D(G(z))$** instead. Same fixed point, much stronger gradients when G is losing, the **non-saturating loss**. Training alternates: update D (real + fake batch, BCE), then update G (push D toward calling fakes real).

## §3 The problems with GANs (the heart of the interview)

- **Mode collapse.** G finds one (or few) outputs that reliably fool D and produces only those, great-looking, zero-diversity samples. Cause: G optimizes "fool D," not "cover the data." Mitigations: minibatch discrimination, unrolled GANs, Wasserstein loss.
- **Training instability / non-convergence.** A two-player Nash-equilibrium search, not a single loss minimization. Losses oscillate; one net can overpower the other. If **D gets too good too fast**, $D(G(z)) \to 0$, G's gradient vanishes, G stops learning.
- **Vanishing gradients to G**, the saturation problem; the non-saturating loss helps but a too-strong D still starves G.
- **No loss-to-quality correlation.** Because D and G move together, loss values don't indicate sample quality, you can't early-stop on loss.
- **Hard evaluation.** No likelihood. Standard metrics: **Inception Score** and **Fréchet Inception Distance (FID)**, FID compares the mean/covariance of real vs generated images in a pretrained Inception feature space; lower = better. FID is the field standard.
- **Hyperparameter sensitivity.** LRs, architecture, D:G update ratio, all finicky.

| Fix | What it does |
| --- | --- |
| **DCGAN** | Architectural recipe that made GANs train on images: all-convolutional, BatchNorm, strided convs instead of pooling. |
| **WGAN / WGAN-GP** | Most important theoretical fix. Replaces JS-divergence with the **Wasserstein (earth-mover) distance** → smooth, non-vanishing gradients; largely cures mode collapse + instability. WGAN-GP adds a gradient penalty (replacing crude weight clipping). |
| **Conditional GAN** | Feed a class label to G and D → control what's generated. Basis for pix2pix, CycleGAN (unpaired translation). |
| **Progressive / StyleGAN** | Grow resolution gradually, style-based generator → the photorealistic-face era. |

GAN vs diffusion: GAN is one-shot, fast, unstable, and mode-collapse-prone; [diffusion](diffusion.md) is iterative, slow, stable, and high-diversity. GANs have largely been overtaken by diffusion for text-to-image quality, but still win where single-pass inference speed matters. (The Wasserstein fix replaced the [JS divergence](../losses/classification-losses.md) used in the original objective.)
