# Output Size, Pooling & the Forward Pass

The output-size formula is the single most grilled CNN fact in interviews, and pooling plus the full forward pass show how spatial size shrinks while channel depth grows. Get these and you can size any layer by hand.

!!! tip "Rapid Recall"
    Output size is $O = \lfloor (W - K + 2P)/S \rfloor + 1$. Padding preserves spatial size so you can go deep; "same" padding sets $P = (K-1)/2$ (which is why odd kernels are preferred), "valid" is $P=0$. Stride is the step size and is learnable downsampling. Pooling downsamples with no parameters (2×2 max-pool keeps the strongest activation, giving local translation invariance), and global average pooling collapses each map to one number to replace the giant flatten→dense. The CNN signature: spatial size shrinks, channel depth grows, "where" becomes "what."

## §1 Output size, the master formula

$$
O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

$W$ = input size, $K$ = kernel, $P$ = padding each side, $S$ = stride, $\lfloor \cdot \rfloor$ = floor (the kernel can't hang off the edge).

**Derive it so you never forget:** after padding, the input is $W + 2P$ wide. The kernel's left edge sits at $0, S, 2S, \dots$ while its right edge $(+K-1)$ stays inside. The last valid start is $W + 2P - K$; the number of starts is $\frac{W + 2P - K}{S} + 1$, floored.

### Why padding exists

- **Without it the output shrinks every layer** ($W \to W - K + 1$). With K=3 you lose 2 px/layer; stack 10 layers and 32×32 → 12×12. Padding preserves size so you can go deep.
- **Edge pixels are under-sampled**, a corner is touched far fewer times than the center. Padding gives edges more chances to be seen.

**"Same" padding** keeps output = input (stride 1): set $P = \frac{K-1}{2}$. So K=3→P=1, K=5→P=2, K=7→P=3. This is why **odd kernel sizes are preferred**, even kernels give fractional padding. **"Valid"** = P=0. Padding is almost always **zeros**; reflection/replication padding is used in style-transfer/super-resolution to avoid border artifacts.

### Why stride exists

Stride is the step size of the slide. S=1 touches every position; S=2 skips every other, so output roughly halves per dimension. Strided convolution is **learnable downsampling**, modern nets often replace max-pool with stride-2 convs so the downsampling itself is learned.

### Worked output-size problems (the exact interview format)

| Problem | Computation | Output |
| --- | --- | --- |
| 32×32, K=5, S=1, P=0 | $\lfloor(32-5+0)/1\rfloor+1 = 28$ | 28×28 |
| 224×224, K=7, S=2, P=3 (ResNet stem) | $\lfloor(224-7+6)/2\rfloor+1 = \lfloor111.5\rfloor+1 = 112$ | 112×112 |
| 28×28, K=3, S=1, P=1 ("same") | $\lfloor(28-3+2)/1\rfloor+1 = 28$ | 28×28 |
| 56×56, K=3, S=2, P=1 (downsample) | $\lfloor(56-3+2)/2\rfloor+1 = \lfloor27.5\rfloor+1 = 28$ | 28×28 |

!!! note "Full layer with channels, the complete answer"
    Input $224\times224\times3$, layer = 64 filters, K=7, S=2, P=3:

    - Spatial: $\lfloor(224-7+6)/2\rfloor+1 = 112$
    - Channels out = number of filters = 64 → output tensor **112×112×64**
    - Params: $K^2 C_{in}C_{out} + C_{out} = 49\cdot3\cdot64 + 64 = 9472$
    - MACs: $(112\cdot112\cdot64)\times(7\cdot7\cdot3) \approx 1.18\times10^8$

    **Spatial size drives compute; channels + kernel drive parameters.** Keep these two separate in your head.

Reusable checklist for any such problem: (1) spatial output via the formula (do H and W separately if non-square); (2) output channels = number of filters, always; (3) params $= K^2 C_{in} C_{out} + C_{out}$; (4) FLOPs = (output spatial × output channels) × $K^2 C_{in}$.

## §2 Pooling and the full forward pass

Pooling downsamples a feature map with **no learnable parameters**. A 2×2 max-pool, stride 2, takes each non-overlapping 2×2 block and keeps the max.

```
 1  3 | 2  4       max(1,3,5,6)=6   max(2,4,7,8)=8
 5  6 | 7  8   →
-------+------       max(2,1,0,3)=3   max(1,0,4,5)=5
 2  1 | 1  0
 3  0 | 4  5      Result:  6  8
                           3  5
```

**Why pool:** downsampling means less compute and a larger downstream receptive field; and **local translation invariance**, if the bright feature shifts one pixel within the window, the max is unchanged. Max-pool keeps the strongest activation; **global average pooling** at the end collapses each feature map to one number ($H\times W\times C \to 1\times1\times C$), replacing the giant flatten→dense layer and killing huge parameter counts.

```
Signature of a CNN: spatial dimensions shrink, channel depth grows.
Input 224×224×3
 → Conv 7×7, 64, s2, p3  → 112×112×64 → ReLU
 → MaxPool 3×3, s2, p1   → 56×56×64
 → Conv 3×3, 128, s1, p1 → 56×56×128 → ReLU
 → MaxPool 2×2, s2       → 28×28×128
 ... channels ↑  spatial ↓ ...
 → Global Average Pool   → 1×1×C → Flatten → Linear → classes
Information moves from "where" (high-res, few channels) to "what" (low-res, many channels).
```
