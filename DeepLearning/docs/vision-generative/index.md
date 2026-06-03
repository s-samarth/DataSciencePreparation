# Vision Tasks & Generative Models

Beyond classification, vision branches into dense prediction (which pixels are what, where are the objects) and generation (produce new images). This section covers segmentation and detection, then the two great generative families, GANs and diffusion.

!!! tip "Rapid Recall"
    Segmentation labels every pixel; it uses an encoder-decoder with skip connections (U-Net), trains with cross-entropy + Dice (Dice handles class imbalance), and is scored by mIoU. Detection finds multiple objects as boxes: two-stage (Faster R-CNN, accurate) vs one-stage (YOLO, fast, grid regression in one pass with anchors and NMS), scored by mAP. GANs are a minimax forger-vs-detective game: fast but unstable and mode-collapse-prone (WGAN-GP is the big fix). Diffusion learns to reverse noising by predicting the noise with an MSE objective: stable and diverse but slow. GAN = fast/unstable, diffusion = stable/diverse/slow.

## §1 What is in this section

- [Segmentation and detection](segmentation-detection.md): per-pixel labeling with U-Net, the two detection paradigms, how YOLO works, anchors, NMS, and the mAP/mIoU metrics.
- [GANs](gans.md): the minimax game, the non-saturating fix, the classic failure modes, and the lineage from DCGAN to StyleGAN.
- [Diffusion models](diffusion.md): the noise-prediction objective, the U-Net architecture, and text-to-image with latent diffusion.

These build directly on the [convolutional networks](../cnns/index.md) section, the U-Net is an encoder-decoder CNN, and the diffusion denoiser is a U-Net with attention.
