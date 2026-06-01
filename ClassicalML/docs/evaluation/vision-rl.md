# Vision & RL Metrics

Computer vision and reinforcement learning each evaluate something the earlier families do not: spatial overlap and image quality for vision, and behavior over time for RL. This page collects the standard metrics for object detection, classification, segmentation, and image generation, then the core RL metrics including the offline estimators.

!!! tip "Rapid Recall"
    Detection uses IoU for box overlap and mAP (averaged over classes and IoU thresholds for COCO) as the headline. Segmentation uses Dice, which is literally the F1 of pixels and monotonically related to IoU. Image generation uses FID, the Frechet distance between Inception feature distributions of real and generated images (lower is better), with SSIM for perceptual and PSNR for pixel-level quality. RL maximizes discounted return; success rate and episode length track behavior, and offline RL estimates a new policy from logged data via inverse propensity scoring and the more stable doubly robust estimator.

## §1 Object detection

### IoU (Intersection over Union)

**Intuition:** You predicted a bounding box around a dog; the ground truth is another box. How much do they overlap? IoU measures the overlap area as a fraction of the total area covered by both boxes.

$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

A perfect prediction has IoU = 1.0; no overlap gives 0. **Thresholds in practice:** IoU ≥ 0.5 is the PASCAL VOC "loose" match, IoU ≥ 0.75 is "strict," and IoU @ [0.5:0.95] is the COCO standard (averaged over multiple thresholds). Why IoU and not distance between centers? Two boxes can have the same center distance but very different overlap if one is much larger; IoU captures both position and size accuracy.

### mAP (Mean Average Precision for detection)

**Intuition:** For each object class, compute AP (the precision-recall curve area) at various IoU thresholds, then average across all classes. This is the standard for object detection benchmarks.

$$\text{mAP} = \frac{1}{C}\sum_c \text{AP}_c$$

where \(C\) is the number of classes and \(\text{AP}_c\) is computed at a specific IoU threshold (or averaged across thresholds for COCO). **PASCAL VOC mAP** is AP at IoU ≥ 0.5 (single threshold); **COCO mAP** averages AP across IoU thresholds [0.5, 0.55, ..., 0.95] (much stricter).

## §2 Image classification

**Top-1 Error** = (predictions where argmax ≠ true label) / N. The model's highest-probability prediction is wrong; this is just 1 minus accuracy for single-label classification.

**Top-5 Error** = (predictions where true label is not in the top 5) / N. The correct class doesn't appear in the model's 5 highest-confidence predictions. Used in ImageNet because with 1000 classes, many visually similar (different dog breeds), getting the exact answer is hard but getting it in the top 5 is more reasonable. Top-5 measures "does the model at least narrow it down to the right neighborhood?"

## §3 Image segmentation

### Dice Coefficient (Dice-Sorensen)

**Intuition:** The F1 score, but for pixels. For segmentation you classify every pixel as object or not; Dice measures the overlap between your predicted pixel mask and the ground truth mask.

$$\text{Dice} = \frac{2|A\cap B|}{|A| + |B|} = \frac{2\,TP}{2\,TP + FP + FN}$$

where \(A\) is the set of pixels predicted foreground and \(B\) the actual foreground. This is literally the F1 formula, so Dice = F1 for binary pixel classification. **Why Dice over IoU for segmentation?** Dice = 2·IoU / (1 + IoU), so they are monotonically related and rank models identically, but Dice is more commonly used as a *loss* function (Dice Loss) because it directly optimizes overlap, and the medical imaging community strongly prefers it.

**Pixel Accuracy** = correctly classified pixels / total pixels. Same gotcha as classification accuracy: if 95% of the image is background, predicting "all background" gives 95% pixel accuracy but captures nothing.

## §4 Image generation (GANs / diffusion)

### FID (Frechet Inception Distance)

**Intuition:** "How similar is the *distribution* of generated images to the distribution of real images?" FID doesn't compare individual images. It passes real and generated images through a pre-trained Inception-v3, extracts feature vectors, fits a multivariate Gaussian to each set, and measures the distance between the two Gaussians.

$$\text{FID} = \lVert\mu_r - \mu_g\rVert^2 + \text{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big)$$

**Lower FID is better;** FID = 0 means identical distributions. The Inception features capture high-level semantic content, so similar FID means similar diversity, quality, and semantic content. **Gotchas:** biased toward ImageNet-like images, needs a large sample (about 10K minimum for stable estimates), and measures distributional similarity rather than individual image quality.

### SSIM (Structural Similarity Index)

**Intuition:** Compares two images the way a human perceives similarity, by examining luminance, contrast, and structure separately, then combining them.

$$\text{SSIM}(x,y) = [l(x,y)]^\alpha\,[c(x,y)]^\beta\,[s(x,y)]^\gamma$$

with luminance \(l = (2\mu_x\mu_y + C_1)/(\mu_x^2 + \mu_y^2 + C_1)\), contrast \(c = (2\sigma_x\sigma_y + C_2)/(\sigma_x^2 + \sigma_y^2 + C_2)\), and structure \(s = (\sigma_{xy} + C_3)/(\sigma_x\sigma_y + C_3)\). Range \([-1,1]\); SSIM = 1 means identical. PSNR measures raw pixel-level error while SSIM measures perceptual similarity, so two images can share the same PSNR but very different SSIM if the errors are structured (blur) versus random (noise).

### PSNR (Peak Signal-to-Noise Ratio)

**Intuition:** "How much signal versus noise?" Higher PSNR means less noise and better reconstruction. Based on MSE but expressed in decibels.

$$\text{PSNR} = 10\log_{10}\!\frac{\text{MAX}^2}{\text{MSE}}$$

where MAX is the maximum pixel value (255 for 8-bit). Higher is better; typical "good" values are 30 to 50 dB. Used in compression, super-resolution, and denoising; simple and widely reported but poorly correlated with human perception, so use SSIM for that.

## §5 Reinforcement learning metrics

RL agents are evaluated on actions and outcomes over time, not predictions. The fundamental question: "Did the agent learn to behave well?"

### Cumulative reward (return)

**Intuition:** The total reward the agent collects during one episode, the thing RL maximizes.

$$G_t = \sum_{k=0}^{\infty}\gamma^k\,r_{t+k+1},\qquad 0 < \gamma \le 1$$

The discount factor \(\gamma\) controls how much the agent cares about future versus immediate rewards: \(\gamma = 0.99\) is patient (values future rewards almost as much as current), \(\gamma = 0.5\) is myopic (strongly prefers immediate rewards).

### Other behavioral metrics

- **Average episode length / survival time:** in tasks like CartPole, longer survival means a better policy.
- **Success rate** = episodes where the goal was achieved / total episodes. For goal-conditioned RL with a clear binary outcome (reached the target, solved the maze).
- **Value loss / policy loss:** internal training metrics. Value loss measures how well the value network estimates future rewards; policy loss measures how well the policy is being optimized (algorithm-dependent: PPO, A2C, REINFORCE). If value loss stops decreasing, the agent's understanding of "what's good" has plateaued.

### Offline RL metrics

**The problem:** in offline RL you have a dataset collected by some old policy (a human doctor's treatment decisions) and want to estimate how a *new* policy would perform *without* deploying it, which is counterfactual reasoning.

**IPS (Inverse Propensity Score):**

$$V_{\text{IPS}}(\pi) = \frac{1}{N}\sum_i \frac{\pi(a_i\mid s_i)}{\pi_{\text{old}}(a_i\mid s_i)}\,r_i$$

Reweights historical outcomes by how likely the new policy would have taken the same action. High variance because the importance weights can be extreme.

**Doubly Robust Estimator:** combines IPS with a value-function estimate. More stable than IPS alone because if *either* the importance weights *or* the value-function estimate is correct, the overall estimate is unbiased.

## Interview questions

**Q1: What is mAP and how does COCO differ from PASCAL VOC?**
mAP averages the average precision, the area under the precision-recall curve, across all classes, where a detection counts as correct only if its IoU with the ground-truth box exceeds a threshold. PASCAL VOC uses a single IoU threshold of 0.5, a loose match, while COCO averages mAP over ten IoU thresholds from 0.5 to 0.95, which is much stricter because it rewards tight localization. IoU itself is intersection over union of the boxes, capturing both position and size.

**Q2: Why is Dice preferred over IoU in segmentation despite ranking models the same?**
Dice equals 2·IoU/(1+IoU), so they are monotonically related and order models identically, but Dice is the F1 of pixels, two times TP over two TP plus FP plus FN, and it makes a better loss function because it directly and smoothly optimizes overlap, whereas IoU is harder to differentiate. The medical imaging community standardized on it for that reason. Pixel accuracy is avoided because, like classification accuracy, it is inflated by a large background class.

**Q3: What does FID measure and why is it distributional?**
FID embeds real and generated images with a pretrained Inception network, fits a Gaussian to each set of feature vectors, and computes the Frechet distance between those two Gaussians, so lower is better and zero means identical distributions. It is distributional because it compares the whole population of generated images to real ones rather than pairing individual images, capturing diversity and quality together. Its caveats are an ImageNet bias and the need for around ten thousand samples for stability.

**Q4: How do you evaluate a new RL policy from logged data without deploying it?**
You use off-policy estimation. Inverse propensity scoring reweights each logged reward by the ratio of the new policy's action probability to the logging policy's, estimating what the new policy would have earned, but the importance ratios can be extreme so it has high variance. The doubly robust estimator combines this with a learned value-function estimate and is unbiased if either the importance weights or the value model is correct, making it far more stable.
