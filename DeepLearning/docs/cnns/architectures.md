# CNN Architectures

One problem, repeated: we want deeper networks for more abstraction, but depth keeps breaking training. Each landmark architecture is a different fix, and the line runs straight from LeNet to the residual connection that powers everything after 2015, including the Transformer.

!!! tip "Rapid Recall"
    LeNet proved the conv→pool→dense template; AlexNet scaled it (ReLU, dropout, GPUs); VGG showed depth via uniform 3×3 stacks but ballooned to 138M params and degraded past ~19 layers. The degradation problem is an optimization failure, not overfitting: a 56-layer plain net had higher *training* error than a 20-layer one. ResNet's fix is the residual block $y = x + F(x)$, whose gradient $\partial L/\partial x = \partial L/\partial y(1 + \partial F/\partial x)$ keeps a multiply-free path back, a gradient highway that enabled 152+ layers. After ResNet: DenseNet concatenates, MobileNet uses depthwise-separable convs for ~9× cheaper compute, EfficientNet scales depth/width/resolution jointly.

## §1 Architectural evolution and skip connections

| Net | Year | Contribution |
| --- | --- | --- |
| **LeNet-5** | 1998 | Proof of concept. conv→pool→conv→pool→dense template. ~60K params, digits. |
| **AlexNet** | 2012 | "Works at scale." 8 layers, 60M params, won ImageNet. **ReLU**, **dropout**, GPU training, heavy augmentation. Big 11×11 kernels (soon abandoned). |
| **VGG** | 2014 | Depth via uniform 3×3 stacks (every conv 3×3/s1/p1, every pool 2×2). Showed depth drives accuracy. But 138M params (most in the dense layers) and degraded past ~19 layers. |
| **ResNet** | 2015 | Residual connections. Trained 152+ layers. The spine of all modern deep nets. |

### The degradation problem (the setup for ResNet)

A deeper net should never be worse than a shallow one, the extra layers could just learn identity and copy the input forward. In practice the opposite happened: a 56-layer plain net had **higher training error** than a 20-layer one. This is **not overfitting** (training error went up). It is an **optimization failure**, the gradient signal degrades through many layers and SGD can't even find the identity mapping that would make depth harmless.

### ResNet's fix, the single most important idea here

Instead of asking a block to learn a target $H(x)$, ask it to learn the **residual** $F(x) = H(x) - x$ and add the input back: $y = F(x) + x$. If the best thing is "do nothing," the block just drives $F(x) \to 0$ (push weights to zero, easy). Learning identity becomes trivial, so adding depth can never hurt.

Backprop through $y = x + F(x)$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(1 + \frac{\partial F}{\partial x}\right) = \frac{\partial L}{\partial y} + \frac{\partial L}{\partial y}\frac{\partial F}{\partial x}
$$

The **"1"** means the gradient has a path back to earlier layers that is *never* multiplied down by weights. Even if $\partial F/\partial x \to 0$ (vanishing), the $\partial L/\partial y \cdot 1$ term survives. The skip connection is a **gradient highway**. This is what let them train 152 layers when plain nets choked at ~20. (The same idea, viewed from the [gradient flow](../gradient-flow/gradient-highways.md) side, powers the LSTM cell state.)

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg">
  <style>
    .nb{fill:#222736;stroke:#2c3140;stroke-width:1.5;rx:8}
    .nt{fill:#e8e6e0;font-family:'Spline Sans Mono',monospace;font-size:13px;text-anchor:middle}
    .fl{stroke:#5ea8c4;stroke-width:2;fill:none;marker-end:url(#a2)}
    .sk{stroke:#e0b341;stroke-width:2.5;fill:none;marker-end:url(#a2);stroke-dasharray:6 4}
    .st{fill:#e0b341;font-family:'Spline Sans Mono',monospace;font-size:12px}
  </style>
  <defs><marker id="a2" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
    <path d="M0,0 L9,4.5 L0,9 z" fill="#5ea8c4"/></marker></defs>
  <text x="60" y="110" class="nt">x</text>
  <path d="M 75 105 L 140 105" class="fl"/>
  <rect x="140" y="80" width="130" height="50" class="nb"/><text x="205" y="110" class="nt">Conv-BN-ReLU</text>
  <path d="M 270 105 L 330 105" class="fl"/>
  <rect x="330" y="80" width="110" height="50" class="nb"/><text x="385" y="110" class="nt">Conv-BN</text>
  <path d="M 440 105 L 500 105" class="fl"/>
  <circle cx="520" cy="105" r="20" fill="#222736" stroke="#7fae6f" stroke-width="2"/><text x="520" y="111" class="nt" style="fill:#7fae6f">+</text>
  <path d="M 540 105 L 600 105" class="fl"/>
  <text x="615" y="110" class="nt">y</text>
  <path d="M 75 105 Q 90 30 300 30 Q 510 30 518 82" class="sk"/>
  <text x="250" y="22" class="st">identity shortcut (gradient highway)  y = x + F(x)</text>
</svg>
<figcaption>ResNet block: the dashed amber path carries x untouched to the addition, guaranteeing gradient flow back to earlier layers.</figcaption>
</figure>

When $F(x)$ changes channel count or spatial size, $x$ and $F(x)$ can't be added directly, fix with a **1×1 conv on the shortcut** (the "projection shortcut"). Deeper ResNets use the **bottleneck block**: $1\times1$ reduce $\to 3\times3 \to 1\times1$ restore, the 1×1s shrink channels so the expensive 3×3 runs cheaply.

## §2 After ResNet, three efficiency directions

**DenseNet (2016), connect everything.** Where ResNet *adds*, DenseNet **concatenates**: each layer receives all preceding layers' feature maps:

$$
x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])
$$

(brackets = channel-wise concat). Maximal feature reuse, very strong gradient flow, surprisingly **few parameters** (each layer adds only a small "growth rate" of channels, for example 12 or 32). Cost moves to memory.

**MobileNet (2017), depthwise separable convolutions.** Factorize a normal conv (spatial + channel mixing) into two cheap steps. **Depthwise**: one $K\times K$ filter per input channel, independently, spatial only, cost $K^2 C_{in}$. **Pointwise**: a 1×1 conv across channels, channel mixing, cost $C_{in} C_{out}$.

$$
\frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}
$$

For K=3 that's $\approx \frac{1}{9}$, about a **9× compute reduction**. That is how CNNs run on phones. MobileNetV2 added an **inverted residual** with linear bottleneck (ResNet's skip between thin bottlenecks).

**EfficientNet (2019), compound scaling.** Scale depth, width, and resolution **together** in a fixed ratio (more resolution needs more layers for receptive field, and more channels for detail):

$$
\text{depth}=\alpha^\phi,\quad \text{width}=\beta^\phi,\quad \text{resolution}=\gamma^\phi,\qquad \alpha\cdot\beta^2\cdot\gamma^2 \approx 2
$$

(width and resolution squared because each costs quadratically). The base net B0 (found by NAS, built from MobileNet inverted-residuals + **squeeze-and-excitation** channel attention) scales to B0…B7 just by raising $\phi$. Matched best-in-class accuracy with up to ~10× fewer params.

The throughline: **VGG** shows depth helps, use small 3×3. **ResNet** fixed depth breaking training; $+x$ is the gradient highway, so arbitrarily deep. **DenseNet** concatenates all prior features, params down. **MobileNet** uses depthwise+pointwise for ~9× cheaper. **EfficientNet** scales depth/width/resolution jointly. Residual connections are the spine of everything after 2015, including the Transformer (same $x + \text{sublayer}(x)$).

## §3 Data augmentation, enriching the dataset for free

Augmentation generates new, label-preserving variants so the model sees more "effective" data and learns invariances you care about. The core principle: **only apply transforms under which the label is genuinely invariant.**

**Geometric.** Horizontal flip is safe for most natural images (a flipped cat is a cat), but **not** where orientation carries meaning: text, digits (a flipped 3 isn't a 3), medical lateralization; vertical flip is rarely safe except aerial/microscopy. Random crop (+ resize) is the single most effective augmentation for ImageNet-style training, it teaches scale/position invariance. Rotation: small angles (±15°) for natural images; full 360° only with no canonical orientation. Translation / scale / shear / affine give viewpoint variation.

**Photometric.** Color jitter (brightness, contrast, saturation, hue) for lighting/camera robustness. Grayscale / channel drop forces reliance on shape over color. Gaussian noise / blur for sensor-noise and focus robustness. Cutout / Random Erasing blacks out a random rectangle so the model uses the whole object, not one patch.

**Multi-image / label-mixing.** Mixup blends two images and labels: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$, $\lambda \sim \text{Beta}(\alpha,\alpha)$, which smooths decision boundaries. CutMix pastes a patch from image B into A, with the label mixed by patch area, combining Cutout's localization with Mixup's label blending.

!!! warning "The non-negotiable rules"
    1. Augment **training only**, never val/test (except deliberate test-time augmentation). 2. **Normalize after augmenting**, using fixed dataset statistics (ImageNet mean/std). 3. Choose transforms that respect your domain's invariances, the flipped-digit / mirrored-medical-scan break is the classic gotcha. 4. Augmentation substitutes for data: most valuable on small datasets, the primary defense against overfitting.

```python
import torchvision.transforms as T
train_tf = T.Compose([
    T.RandomResizedCrop(224),            # random crop + scale
    T.RandomHorizontalFlip(),            # safe for natural images
    T.ColorJitter(0.2,0.2,0.2,0.1),    # brightness, contrast, sat, hue
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],   # ImageNet mean
                [0.229,0.224,0.225]),  # ImageNet std
    T.RandomErasing(p=0.25),           # cutout, applied on the tensor
])
val_tf = T.Compose([                     # NO augmentation
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
```

## §4 A CNN in PyTorch

The network from building blocks:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SmallCNN(nn.Module):
    # Mini-VGG classifier for 32x32x3 images (CIFAR-10)
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv args: (in_ch, out_ch, kernel, stride, padding); p=1,k=3,s=1 => "same"
        self.conv1 = nn.Conv2d(3,   32,  3, 1, 1)   # 32x32x3  -> 32x32x32
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64,  3, 1, 1)   # -> 32x32x64
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128,  3, 1, 1)   # -> 16x16x128
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)        # halves spatial dims
        self.gap   = nn.AdaptiveAvgPool2d(1)   # global avg pool -> 1x1xC
        self.fc    = nn.Linear(128, num_classes)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):                       # x: [B, 3, 32, 32]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B,32,16,16]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B,64,8,8]
        x = F.relu(self.bn3(self.conv3(x)))             # [B,128,8,8]
        x = self.gap(x)                                # [B,128,1,1]
        x = torch.flatten(x, 1)                       # [B,128]
        return self.fc(self.drop(x))                     # [B,num_classes] raw logits
```

Things to internalize: **Conv → BN → ReLU → (Pool)** is the canonical block order (BN before ReLU). **Spatial down, channels up** (32→16→8; 3→32→64→128). **No softmax in forward**, `nn.CrossEntropyLoss` expects raw logits and applies log-softmax internally; applying softmax yourself is the double-softmax bug. **GAP** instead of flatten→huge-FC eliminates the millions of dense params VGG wasted.

The training loop (the part interviews probe):

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallCNN(10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

for epoch in range(30):
    model.train()
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()        # grads accumulate by default; clear them
        logits = model(imgs)         # forward
        loss = criterion(logits, labels)
        loss.backward()              # autograd fills p.grad for every param
        optimizer.step()             # p <- p - lr * p.grad (+ momentum/Adam)
    scheduler.step()
```

The four-line core is `zero_grad → forward/loss → backward → step`, with `zero_grad` first because PyTorch **accumulates** gradients (a feature, it lets you simulate big batches by summing several backward passes); forget it and your gradient is the sum of all batches seen so far. See the [training loop](../foundations/training-loop.md) page for the full six steps.

In practice, don't hand-roll, use a pretrained backbone:

```python
import torchvision.models as models
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for p in model.parameters(): p.requires_grad = False   # freeze backbone
model.fc = nn.Linear(model.fc.in_features, 10)        # new trainable head
```

Early conv layers learn generic features (edges, textures) that transfer across tasks; late layers are task-specific. **Freeze early, retrain the head.** With more data, fine-tune later layers at a low LR. This is why transfer learning works, and why you rarely train from scratch.
