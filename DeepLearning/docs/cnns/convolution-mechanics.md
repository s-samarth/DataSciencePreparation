# Convolution Mechanics

A convolution is the single operation a CNN repeats billions of times. Understanding exactly what it computes, how channels stack into the weight tensor, and why the parameter count is independent of image size is the foundation for everything else about CNNs.

!!! tip "Rapid Recall"
    A filter (kernel) is a small grid of learnable numbers, a pattern template, slid over the image; at each position the dot product (elementwise multiply then sum) gives one output number, and the collection is a feature map. With $C_{in}$ input channels, one filter is $K \times K \times C_{in}$ and sums across channels to produce one map; $C_{out}$ filters give $C_{out}$ maps, so the weight tensor is $C_{out} \times C_{in} \times K \times K$, independent of image size. Stacking small kernels (two 3×3 = one 5×5 receptive field) is cheaper and more expressive. A 1×1 convolution mixes channels at a single pixel, used for cheap dimensionality change.

## §1 What a convolution actually is

A filter (kernel) is a small grid of learnable numbers, a **pattern template**. You slide it over the image. At each position you ask: *"how much does the patch underneath me look like my pattern?"* The answer is one number. Collect these over the whole image and you get a **feature map**: a heatmap of where the pattern appeared. The "looks like" operation is just **elementwise multiply, then sum**, a dot product.

With input patch and kernel both $K \times K$:

$$
y_{i,j} = b + \sum_{m=0}^{K-1}\sum_{n=0}^{K-1} w_{m,n}\, x_{\,i+m,\; j+n}
$$

where $w$ = kernel weights, $b$ = one bias for the whole filter, $x$ = input. One dot product per output cell.

!!! note "Interview note: convolution vs cross-correlation"
    What deep learning calls "convolution" is technically **cross-correlation**, true convolution flips the kernel first ($x_{i-m,j-n}$). Since the kernel is *learned*, the flip is irrelevant, the network just learns the flipped weights. If asked: "it's cross-correlation; the flip doesn't matter because the weights are learned."

## §2 Fully worked numeric example

Input 4×4, kernel 3×3, stride 1, no padding. Output is 2×2.

```
Input X            Kernel W (edge-ish), bias b = 0
 1  2  0  1         1  0 -1
 0  1  2  3         1  0 -1
 1  0  1  2         1  0 -1
 2  1  0  1
```

Top-left cell, overlay W on the top-left 3×3 block:

$$
(1)(1)+(2)(0)+(0)(-1)+(0)(1)+(1)(0)+(2)(-1)+(1)(1)+(0)(0)+(1)(-1) = -1
$$

Slide one right (cols 1 to 3): $= -3$. Slide down (rows 1 to 3, cols 0 to 2): $= 0$. Bottom-right: $= -4$.

```
Output feature map:
-1  -3
 0  -4
```

That is a complete convolution. Every CNN is this, billions of times, with learned W.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gl{fill:none;stroke:#2c3140;stroke-width:1.5}
    .gt{fill:#b6b3aa;font-family:'Spline Sans Mono',monospace;font-size:13px;text-anchor:middle}
    .ker{fill:rgba(224,179,65,.18);stroke:#e0b341;stroke-width:2.5}
    .lbl{fill:#8a8778;font-family:'Spline Sans Mono',monospace;font-size:12px}
    .out{fill:rgba(94,168,196,.16);stroke:#5ea8c4;stroke-width:2}
    .ot{fill:#5ea8c4;font-family:'Spline Sans Mono',monospace;font-size:13px;text-anchor:middle}
    .arr{stroke:#c97a5a;stroke-width:2;fill:none;marker-end:url(#ah)}
  </style>
  <defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
    <path d="M0,0 L9,4.5 L0,9 z" fill="#c97a5a"/></marker></defs>
  <text x="120" y="28" class="lbl">Input 4×4</text>
<rect x="40" y="42" width="52" height="52" class="gl"/><text x="66.0" y="73.0" class="gt">1</text><rect x="92" y="42" width="52" height="52" class="gl"/><text x="118.0" y="73.0" class="gt">2</text><rect x="144" y="42" width="52" height="52" class="gl"/><text x="170.0" y="73.0" class="gt">0</text><rect x="196" y="42" width="52" height="52" class="gl"/><text x="222.0" y="73.0" class="gt">1</text><rect x="40" y="94" width="52" height="52" class="gl"/><text x="66.0" y="125.0" class="gt">0</text><rect x="92" y="94" width="52" height="52" class="gl"/><text x="118.0" y="125.0" class="gt">1</text><rect x="144" y="94" width="52" height="52" class="gl"/><text x="170.0" y="125.0" class="gt">2</text><rect x="196" y="94" width="52" height="52" class="gl"/><text x="222.0" y="125.0" class="gt">3</text><rect x="40" y="146" width="52" height="52" class="gl"/><text x="66.0" y="177.0" class="gt">1</text><rect x="92" y="146" width="52" height="52" class="gl"/><text x="118.0" y="177.0" class="gt">0</text><rect x="144" y="146" width="52" height="52" class="gl"/><text x="170.0" y="177.0" class="gt">1</text><rect x="196" y="146" width="52" height="52" class="gl"/><text x="222.0" y="177.0" class="gt">2</text><rect x="40" y="198" width="52" height="52" class="gl"/><text x="66.0" y="229.0" class="gt">2</text><rect x="92" y="198" width="52" height="52" class="gl"/><text x="118.0" y="229.0" class="gt">1</text><rect x="144" y="198" width="52" height="52" class="gl"/><text x="170.0" y="229.0" class="gt">0</text><rect x="196" y="198" width="52" height="52" class="gl"/><text x="222.0" y="229.0" class="gt">1</text><rect x="40" y="42" width="156" height="156" class="ker"/><text x="118.0" y="220" class="lbl" text-anchor="middle">3×3 kernel here → one output</text><path d="M 260 120.0 L 326 120.0" class="arr"/><text x="395" y="28" class="lbl">Output 2×2</text><rect x="343" y="52" width="52" height="52" class="out"/><text x="369.0" y="83.0" class="ot">-1</text><rect x="395" y="52" width="52" height="52" class="gl"/><text x="421.0" y="83.0" class="ot">-3</text><rect x="343" y="104" width="52" height="52" class="gl"/><text x="369.0" y="135.0" class="ot">0</text><rect x="395" y="104" width="52" height="52" class="gl"/><text x="421.0" y="135.0" class="ot">-4</text><text x="395" y="182" class="lbl" text-anchor="middle">each kernel position → 1 cell</text></svg>
<figcaption>The kernel slides across every valid position; each placement produces one output number. 4×4 input, 3×3 kernel, stride 1 → 2×2 output.</figcaption>
</figure>

## §3 Channels and the weight tensor

Real images aren't flat. An RGB image is $H \times W \times 3$. After the first conv layer you have many feature maps (channels). So convolution is almost never 2D in practice, it is **3D over the channel dimension**, and this is where the parameter math lives.

A single filter spans **all** input channels. If the input has $C_{in}$ channels, one filter is not $K \times K$, it is $K \times K \times C_{in}$, with a separate slice per channel. You convolve each slice with its channel, then **sum across all channels** into a single number. So one filter produces **one** output feature map.

$$
y_{i,j} = b + \sum_{c=0}^{C_{in}-1}\sum_{m=0}^{K-1}\sum_{n=0}^{K-1} w_{c,m,n}\, x_{c,\,i+m,\,j+n}
$$

Want $C_{out}$ feature maps? Use $C_{out}$ independent filters, each $K \times K \times C_{in}$. Stack their outputs along the channel axis. The weight tensor shape is:

$$
\underbrace{C_{out}}_{\#\,\text{filters}} \times \underbrace{C_{in}}_{\text{depth each}} \times \underbrace{K \times K}_{\text{spatial}}
$$

The weight tensor is *not* decided by image size. It is decided by three hyperparameters: kernel size $K$, input channels $C_{in}$ (fixed by the previous layer), and output channels $C_{out}$ (your design choice). The image's height and width never enter the parameter count, that is the whole point of weight sharing.

```
Input: 3 channels (RGB)        One filter has 3 slices (one per channel)
  R 5×5   G 5×5   B 5×5          W_R 3×3  W_G 3×3  W_B 3×3
    │       │       │               │        │        │
    └─conv──┴─conv──┴─conv──────────┘        │        │
                  │                            │        │
            sum all three ◄────────────────────┴────────┘
                  │
              ONE output feature map (+ bias)
  Want 64 output maps? → 64 such 3-slice filters, stacked.
```

## §4 Kernel sizes and the 1×1 convolution

A $K \times K$ kernel over $C_{in}$ channels producing $C_{out}$ has $K^2 C_{in} C_{out}$ weights. Cost grows with $K^2$. A 5×5 is $25/9 \approx 2.8\times$ more expensive than a 3×3.

Stacking small kernels beats one big kernel. Two stacked 3×3 convs have the same **receptive field** as one 5×5; three 3×3s ≈ one 7×7, but with fewer parameters and more non-linearities. One 7×7 filter: $7^2 = 49$ params. Three 3×3 filters: $3 \times 3^2 = 27$ params. Same receptive field, **45% fewer parameters** and two extra ReLUs, so more expressive. This is the entire design philosophy of VGG.

The receptive field grows as:

$$
r_{\ell} = r_{\ell-1} + (K_\ell - 1)\prod_{i=1}^{\ell-1} S_i
$$

With all stride-1, K=3 layers, the field grows $3 \to 5 \to 7 \to 9$, +2 per layer. Add stride or pooling and it grows multiplicatively, that is how a final-layer neuron ends up seeing the whole image with few layers.

### 1×1 convolutions, genuinely useful

A 1×1 kernel looks at a single spatial position but **all channels**. It does nothing spatially, it is a per-pixel fully-connected layer across channels. Its job is **channel mixing and dimensionality change**: squeeze 256→64 channels cheaply before an expensive 3×3 (ResNet's bottleneck), or expand, or add a non-linearity across channels at zero spatial cost. Params: $C_{in}C_{out}$. Used everywhere from Inception to MobileNet. The [architectures](architectures.md) page shows where the bottleneck uses them.
