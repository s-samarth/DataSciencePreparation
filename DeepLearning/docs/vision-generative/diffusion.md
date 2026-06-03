# Diffusion Models

Destroying an image is easy, so learn to reverse the destruction. Diffusion replaces the adversarial game with a single stable regression: predict the noise. That one change buys stable training, full mode coverage, and the highest image quality, at the cost of slow iterative inference.

!!! tip "Rapid Recall"
    Forward: add Gaussian noise ~1000 times until the image is pure noise (fixed, closed-form). Reverse: train a network to predict the noise so you can subtract it, then denoise step by step from pure noise. The objective is plain MSE between the true and predicted noise, no minimax, so no mode collapse and full diversity, but inference is slow (many passes). The denoiser is a U-Net with a timestep embedding and self-attention. Text-to-image conditions via cross-attention; latent diffusion (Stable Diffusion) runs the whole process in a compressed VAE latent for efficiency.

## §1 Core intuition

Destroying an image is easy, so **learn to reverse the destruction.** **Forward:** add a tiny bit of Gaussian noise ~1000 times until the image dissolves into pure noise, fixed, no learning. **Reverse:** train a network to undo *one* noising step (predict the noise added so you can subtract it). Repeat from pure noise to get a clean image. Generation = denoise step by step. The adversarial game is gone, replaced by a single stable regression: **predict the noise.**

### Why they're so good at images

- **Stable training objective**, plain MSE regression. No minimax, no equilibrium, so no mode collapse, no oscillation. Train longer, it keeps improving.
- **Full mode coverage**, the objective rewards reconstructing all training data, so high diversity.
- **Many easy steps**, noise→image in one shot (GAN) is brutal; diffusion decomposes it into ~1000 tiny refinements. Difficulty amortized, so high quality. *This is the key reason.*
- **Coarse-to-fine by construction**, early reverse steps (high noise) fix global layout; late steps (low noise) fix fine texture.

The price: **slow inference**, many forward passes vs a GAN's single pass. The central tradeoff.

## §2 Forward (noising), closed form

With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$, you can jump to any noise level $t$ in one shot:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

The noisy image is the original scaled down plus noise scaled up. As $t \to T$, $\bar\alpha_t \to 0$, so pure noise. This closed form is what makes training cheap.

## §3 Training objective

A network $\epsilon_\theta$ takes the noisy image and timestep and predicts the noise. The loss is MSE:

$$
\mathcal{L} = \mathbb{E}_{x_0,\,\epsilon,\,t}\big[\;\|\,\epsilon - \epsilon_\theta(x_t,\, t)\,\|^2\;\big]
$$

That's the whole loss. (This is DDPM noise-prediction; equivalent parameterizations predict $x_0$ or velocity $v$.)

## §4 Sampling (reverse)

Start from $x_T \sim \mathcal{N}(0, I)$. For $t = T \dots 1$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big) + \sigma_t z
$$

Subtract the predicted noise (scaled), add a little fresh noise $z$. Repeat to get a clean image. **DDIM** makes this deterministic and lets you skip steps (20 to 50 instead of 1000).

## §5 Architecture, a U-Net

The network $\epsilon_\theta$ is almost always a **U-Net** (the same encoder-decoder + skip connections from [segmentation](segmentation-detection.md)), since predicting per-pixel noise is a dense image-to-image task. Components:

- **Encoder-decoder + skip connections**, global structure + preserved fine detail.
- **Residual blocks (ResNet-style)** at each level, stable deep training.
- **Timestep embedding**, the net must behave differently at high vs low noise. The integer $t$ becomes a sinusoidal embedding, then a small MLP, then is injected into every residual block ("how noisy is this input").
- **Self-attention** at lower-resolution levels, global coherence (both eyes match, left relates to right). Convolution is local; attention adds global consistency. Modern diffusion U-Nets are conv + attention hybrids.

```
noisy x_t ─┐                                  ┌─► predicted noise ε
           │                                   │
      [enc block]──────skip──────────►[dec block]
           │                              ▲
      [enc + attn]──────skip────►[dec + attn]
           │                          ▲
      [bottleneck + self-attention]───┘
           ▲
 timestep t → [sinusoidal embed → MLP] → injected into every block
```

## §6 Text-to-image (DALL·E / Stable Diffusion)

- **Conditioning on text.** Encode the prompt (CLIP / T5), inject via **cross-attention**, image features (queries) attend to text tokens (keys/values), steering every step. **Classifier-free guidance** amplifies the prompt: run with and without text, extrapolate in the text direction (the "guidance scale" knob trades adherence vs diversity).
- **Latent diffusion (Stable Diffusion's breakthrough).** Running 1000 steps on full 512×512×3 is brutal. Use a pretrained **VAE** to compress to a small latent (for example 64×64×4), run the *entire* diffusion in that space, and decode back once at the end. ~48× less spatial data per step, so high-res on consumer GPUs.

```python
# Conceptual DDPM training loop
for x0 in dataloader:                          # real images (or VAE latents)
    t = torch.randint(0, T, (x0.size(0),))     # random timestep per image
    eps = torch.randn_like(x0)                 # the noise we add
    xt = sqrt_abar[t]*x0 + sqrt_one_minus_abar[t]*eps   # closed-form forward
    eps_pred = unet(xt, t)                     # (+ text embedding if conditional)
    loss = F.mse_loss(eps_pred, eps)           # predict the noise
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

In practice use HuggingFace `diffusers`: `UNet2DModel` / `UNet2DConditionModel` (text), a scheduler (`DDPMScheduler` / `DDIMScheduler`) for the $\beta_t$ schedule + sampling math; train the U-Net while the VAE + text encoder stay frozen.
