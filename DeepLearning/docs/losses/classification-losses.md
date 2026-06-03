# Classification Losses

Classification losses score predicted class probabilities against discrete labels. Cross-entropy is the workhorse, and the variants exist to handle imbalance, calibration, margins, and distribution matching.

!!! tip "Rapid Recall"
    Binary cross-entropy is the NLL under a Bernoulli; categorical cross-entropy is the NLL under a categorical, and softmax + CE cancel to the clean $\partial L/\partial z = \hat{y} - y$ gradient. Sparse CE is the same with integer labels. Focal loss multiplies CE by $(1-\hat{y})^\gamma$ to downweight easy examples and fix extreme imbalance. Hinge loss optimizes a margin (the SVM loss). Label smoothing softens targets for calibration. KL divergence measures distribution mismatch (and equals CE up to a constant when the target is fixed), is asymmetric, and JS is its symmetric bounded cousin.

## §1 Binary Cross-Entropy (BCE / Log Loss)

| Symbol | Meaning |
| --- | --- |
| $y_i$ | True label in $\{0, 1\}$. |
| $\hat{y}_i$ | Predicted probability that $y_i = 1$ (after sigmoid). In $(0, 1)$. |
| $z_i$ | Raw logit before sigmoid. |

$$
L = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\big], \qquad \frac{\partial L}{\partial z} = \hat{y} - y
$$

**Probabilistic interpretation:** negative log-likelihood under a Bernoulli distribution. The model is saying "the label is a coin flip with probability $\hat{y}$; what is the likelihood of the observed labels under my model?"

**Strengths:** probabilistically principled; the gradient is well-behaved everywhere when paired with sigmoid; heavily penalizes confident wrong predictions ($\log(0) \to \infty$); the right loss for binary classification. **Limitations:** sensitive to class imbalance (the majority class dominates the loss); assumes labels are correct (label noise breaks it); can be numerically unstable if you compute sigmoid then log separately, use `BCEWithLogitsLoss` in PyTorch, which combines them stably.

**Use when:** binary classification, multi-label classification (apply BCE per label independently).

## §2 Categorical Cross-Entropy / Softmax Cross-Entropy

| Symbol | Meaning |
| --- | --- |
| $y_c$ | One-hot encoded target. 1 at the correct class, 0 elsewhere. |
| $\hat{y}_c$ | Predicted probability for class $c$ (after softmax). |
| $C$ | Number of classes. |

$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) = -\log(\hat{y}_{\text{correct}}), \qquad \frac{\partial L}{\partial z} = \hat{y} - y
$$

**Probabilistic interpretation:** negative log-likelihood under a categorical (multinomial with $n=1$) distribution. The exp in softmax and log in cross-entropy cancel to give the beautifully clean gradient (derived in full on the [backprop derivation](../foundations/backprop-derivation.md) page).

**Strengths:** standard for multi-class classification; the clean gradient form makes backprop clean; outputs are calibrated probabilities (mostly); the de facto loss for image classification, language modeling, and more. **Limitations:** assumes mutually exclusive classes (use BCE for multi-label); modern deep networks are **overconfident**, softmax probabilities are not well-calibrated (the model says 99% when it should say 70%); sensitive to class imbalance; doesn't handle label smoothness (treats "cat" and "dog" as equally wrong when the true label is "kitten").

!!! warning "Implementation note"
    In PyTorch, `nn.CrossEntropyLoss` combines softmax and NLL internally and takes raw logits (not probabilities). Don't apply softmax before it, that gives mathematically incorrect double-softmax results.

### Sparse Categorical Cross-Entropy

Mathematically identical to categorical cross-entropy, but the target is an integer class index instead of a one-hot vector. Just a memory/compute optimization for large class counts (for example, language model vocabularies of 50K+ tokens). No new theory.

## §3 Focal Loss

| Symbol | Meaning |
| --- | --- |
| $\gamma$ | Focusing parameter. Typical 2.0. Higher $\gamma$ = more focus on hard examples. |
| $\alpha$ | Class weighting parameter for imbalance. |
| $\hat{y}$ | Predicted probability of the correct class. |

$$
L = -\alpha (1 - \hat{y})^\gamma \log(\hat{y})
$$

**Intuition:** standard cross-entropy is $-\log(\hat{y})$. Focal loss multiplies by $(1 - \hat{y})^\gamma$, when the model is already confident and correct ($\hat{y} \to 1$), this factor $\to 0$, downweighting easy examples. Hard examples ($\hat{y}$ near 0) keep full weight.

**Strengths:** solves extreme class imbalance (the original motivation was object detection where 99% of anchors are background); forces the model to focus learning on hard examples; standard in object detection (RetinaNet, modern YOLO variants). **Limitations:** extra hyperparameters ($\gamma$, $\alpha$) to tune; can be unstable if data is mostly noise (the model fixates on un-learnable examples); for balanced or mildly imbalanced data, it doesn't help and can hurt.

**Use when:** severe class imbalance (more than 1:100), especially in dense prediction tasks like [object detection](../vision-generative/segmentation-detection.md) or semantic segmentation.

## §4 Hinge Loss

$$
\text{Binary, } y \in \{-1, +1\}: \qquad L = \max(0, 1 - y \cdot \hat{y})
$$

**Intuition:** zero loss if the prediction is correct AND confident (margin $\ge 1$). Linear penalty otherwise. The "1" defines the margin, you want correct predictions to be confidently correct, not just barely correct.

**Strengths:** the foundation of SVMs (it directly optimizes the margin); produces sparse solutions (only "support vectors" with margin < 1 contribute to the gradient); more robust to outliers than cross-entropy (no log, so no extreme penalty). **Limitations:** not probabilistic (outputs are not calibrated probabilities); not differentiable at the kink (uses subgradient methods); largely superseded by cross-entropy for deep learning; doesn't extend cleanly to multi-class (requires variants like Crammer-Singer).

## §5 Label Smoothing Cross-Entropy

Covered in full from the [regularization](../regularization/regularization.md) angle. Brief recap of the math:

$$
y_c^{\text{smooth}} = \begin{cases} 1 - \varepsilon & \text{if } c = \text{correct class} \\ \dfrac{\varepsilon}{C - 1} & \text{otherwise} \end{cases}
$$

Apply standard cross-entropy with these. Better calibration. Standard for image classification and transformer training.

## §6 KL Divergence

| Symbol | Meaning |
| --- | --- |
| $P$ | Target distribution. |
| $Q$ | Predicted distribution. |

$$
D_{\text{KL}}(P \,\Vert\, Q) = \sum_i P_i \log\frac{P_i}{Q_i}
$$

**Relationship to cross-entropy:** $\mathrm{CE}(P, Q) = H(P) + \mathrm{KL}(P \Vert Q)$. When $P$ is fixed (the target), KL and CE have identical gradients, they differ only by a constant.

**Asymmetry:** $\mathrm{KL}(P \Vert Q) \ne \mathrm{KL}(Q \Vert P)$.

- **Forward KL ($P \Vert Q$):** zero-avoiding, $Q$ must cover everywhere $P$ has mass. Tends to produce broad, mode-covering $Q$.
- **Reverse KL ($Q \Vert P$):** zero-forcing, $Q$ can ignore regions where $P$ is small. Tends to produce a mode-seeking $Q$ (focuses on one mode).

**Use cases:** knowledge distillation (match a student's output distribution to a teacher's soft predictions); VAEs (KL between the encoder's $q(z\mid x)$ and prior $p(z)$ regularizes the latent space); RLHF / policy optimization (a KL penalty keeps the new policy close to a reference); variational inference generally. **Limitations:** undefined when $Q$ has zero probability where $P$ does not (use $\varepsilon$ smoothing); not a true distance (not symmetric, doesn't satisfy the triangle inequality); the choice of direction (forward vs reverse) is task-specific and important.

## §7 Jensen-Shannon (JS) Divergence

$$
M = \frac{P + Q}{2}, \qquad \mathrm{JS}(P, Q) = \frac{1}{2}\mathrm{KL}(P \,\Vert\, M) + \frac{1}{2}\mathrm{KL}(Q \,\Vert\, M)
$$

A symmetric version of KL. Bounded between 0 and $\log 2$. Used in original [GANs](../vision-generative/gans.md) (which then largely got replaced by Wasserstein loss because JS gives near-zero gradients when distributions don't overlap).

## Interview Questions

**Q1: What's wrong with using MSE for classification?**

Vanishing gradients when the sigmoid (or softmax) saturates. The combined derivative of sigmoid + MSE involves the sigmoid derivative, which shrinks to 0 when the model is confidently wrong. So a model that's badly mistaken doesn't get a strong gradient signal to correct itself. Cross-entropy was specifically designed to keep gradients large when the model is wrong. The log term in cross-entropy exactly cancels out the saturating activation derivative, giving the clean $\hat{y} - y$ gradient that doesn't vanish.

**Q2: KL divergence is asymmetric, when do you use forward vs reverse?**

Forward KL ($P \Vert Q$) is "zero-avoiding" or "mode-covering", $Q$ must cover everywhere $P$ has mass, otherwise the log-ratio blows up. Tends to produce broad $Q$ distributions. Used in VAEs to encourage the encoder to cover the latent space. Reverse KL ($Q \Vert P$) is "zero-forcing" or "mode-seeking", $Q$ can ignore regions where $P$ is small. Tends to produce $Q$ that focuses on one mode. Used in variational inference where you want a tractable $Q$ to capture the main mode of an intractable $P$. The choice depends on whether you want the approximating distribution to be conservative (cover everything) or focused (commit to the most likely region).
