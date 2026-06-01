# Quantization and NF4

The core problem: if you can only store 16 distinct numbers (4-bit), which 16 do you pick, and how do you map weights onto them? The instinct that "uniform spacing is wasteful" is correct — NN weights cluster near zero, so uniform INT4 wastes most of its 16 levels on empty tails. NF4 fixes this by placing levels at the quantiles of a normal distribution.

!!! tip "Rapid Recall"
    Number formats matter because the bottleneck for LLM inference is memory bandwidth, not compute. **BF16** replaced FP16 for training because BF16 keeps FP32's 8-bit exponent (same range, no overflow); FP16's 5-bit exponent overflows past 65504. **Uniform INT4** fails on NN weights because weights are ~Gaussian (dense near zero, thin tails) and uniform spacing wastes 16 levels evenly. **NF4** (Normal Float 4-bit) places its 16 levels at the *quantiles* of a standard normal — dense near zero, sparse at the tails — information-theoretically optimal for normally-distributed data. Block-wise quantization (block of 64 weights, store absmax as the scale) plus double quantization (quantize the per-block constants themselves) saves ~26 GB at 70B scale.

## §1 The number formats

| Format | Bits | Exp / Mantissa | Range | Note |
|---|---|---|---|---|
| Float32 | 32 | 8 / 23 | ±3.4e38 | Max precision; optimizer states. |
| Float16 | 16 | 5 / 10 | ±65504 | More precision but **overflows** to inf. |
| BFloat16 | 16 | 8 / 7 | ±3.4e38 | FP32's range, less precision — **no overflow**. |
| INT8 | 8 | integer | −128 to 127 | 256 values; 2× shrink. |
| INT4 / NF4 | 4 | integer | 16 values | 4× shrink; NF4 is the smart variant. |

!!! abstract "Why BF16 replaced FP16 for training"
    BF16 keeps Float32's exponent (same range), trading away mantissa bits. You never get overflow on gradients, even though each value is less precise. FP16 can overflow (>65504 → inf), which wrecks training.

For an inference-side decision tree on FP8 vs AWQ vs GPTQ vs GGUF, see [Quantization formats](../serving/quantization-formats.md).

## §2 Uniform quantization, and why it loses info

Find min/max, divide into equal buckets, snap each weight to the nearest. Simple, but **NN weights are ~Gaussian** — a tall spike near 0, thin tails. Uniform spacing wastes most of its 16 levels on the empty tails, leaving only 3 to 4 levels for the dense near-zero region where thousands of weights actually live. Massive precision loss exactly where it matters.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 820 260" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<path d="M60,200 Q230,200 380,40 Q530,200 700,200" fill="none" stroke="#6f6d68" stroke-width="1.5"/>
<text x="380" y="225" fill="#6f6d68" font-size="10" text-anchor="middle">weight distribution (~Gaussian)</text>
<text x="60" y="30" fill="#ff6b6b" font-size="10" font-weight="600">UNIFORM (wasteful)</text>
<g stroke="#ff6b6b" stroke-width="1.5">
<line x1="80" y1="195" x2="80" y2="210"/><line x1="120" y1="195" x2="120" y2="210"/>
<line x1="160" y1="195" x2="160" y2="210"/><line x1="200" y1="195" x2="200" y2="210"/>
<line x1="240" y1="195" x2="240" y2="210"/><line x1="280" y1="195" x2="280" y2="210"/>
<line x1="320" y1="195" x2="320" y2="210"/><line x1="360" y1="195" x2="360" y2="210"/>
<line x1="400" y1="195" x2="400" y2="210"/><line x1="440" y1="195" x2="440" y2="210"/>
<line x1="480" y1="195" x2="480" y2="210"/><line x1="520" y1="195" x2="520" y2="210"/>
<line x1="560" y1="195" x2="560" y2="210"/><line x1="600" y1="195" x2="600" y2="210"/>
<line x1="640" y1="195" x2="640" y2="210"/><line x1="680" y1="195" x2="680" y2="210"/>
</g>
<text x="380" y="247" fill="#ff6b6b" font-size="9" text-anchor="middle">evenly spaced — most levels in empty tails</text>
<text x="60" y="70" fill="#6ee7a0" font-size="10" font-weight="600">NF4 (quantile-spaced)</text>
<g stroke="#6ee7a0" stroke-width="1.5">
<line x1="120" y1="178" x2="120" y2="192"/><line x1="200" y1="178" x2="200" y2="192"/>
<line x1="270" y1="178" x2="270" y2="192"/><line x1="320" y1="178" x2="320" y2="192"/>
<line x1="350" y1="178" x2="350" y2="192"/><line x1="368" y1="178" x2="368" y2="192"/>
<line x1="380" y1="178" x2="380" y2="192"/><line x1="392" y1="178" x2="392" y2="192"/>
<line x1="410" y1="178" x2="410" y2="192"/><line x1="440" y1="178" x2="440" y2="192"/>
<line x1="490" y1="178" x2="490" y2="192"/><line x1="560" y1="178" x2="560" y2="192"/>
<line x1="640" y1="178" x2="640" y2="192"/>
</g>
<text x="380" y="120" fill="#6ee7a0" font-size="9" text-anchor="middle">dense near 0 — levels where weights actually live</text>
</svg>
<figcaption>NF4 places its 16 levels at the quantiles of a normal distribution — clustered near zero, sparse in the tails.</figcaption>
</figure>

## §3 NF4 — the smart solution

NF4 (Normal Float 4-bit) spaces its levels at the **quantiles of a standard normal**: each level represents 1/16 of the probability mass, not 1/16 of the number line. More levels near zero, fewer at the extremes. This is **information-theoretically optimal** for normally-distributed data.

### 3.1 Block-wise quantization

One scale factor per layer is too coarse. Quantize in small blocks:

```
Block size = 64 weights
For each block:
  1. absmax = max(|weights|)              # the block's scale
  2. normalized = weights / absmax        # map into N(0,1) space
  3. quantize via fixed NF4 levels        # store 4-bit indices
  4. store absmax as a float32 constant

Dequantize:  recovered = NF4_lookup[index] × absmax
```

Cost: 64 weights × 4 bits + 1 FP32 constant = 36 bytes vs 128 bytes original → ~3.6× compression.

### 3.2 Double quantization

Those per-block scale constants add up: 70B / 64 ≈ 1.1B constants × 4 bytes = **4.4 GB just for constants**. Double quantization quantizes *the constants themselves* (FP32 → FP8, in blocks of 256), saving ~0.37 bits per parameter ≈ **~26 GB at 70B scale**.

## §4 When to use what

| Format | Use case | Why |
|---|---|---|
| FP32 | Optimizer states, small-model training | Max precision, no overflow. |
| BF16 | Large-model training, LoRA adapters | FP32 range, half memory, no overflow. |
| FP16 | Inference on pre-Ampere GPUs | Fast, but overflow risk. |
| INT8 | Inference quant for medium models | 2× shrink, fast integer math, minimal loss. |
| NF4 | QLoRA frozen base, extreme compression | 4× shrink, matches weight distribution. |
| INT4 uniform | (avoid for LLMs) | Loses info near zero where weights cluster. |

!!! abstract "Quantization in one line"
    Pick a smarter set of representative values (matching the actual weight distribution, not uniform), snap each weight to its nearest representative, store only the index plus a scale factor. NF4 wins because its representatives sit where NN weights actually live.

## §5 PTQ vs QAT — when each makes sense

**Post-Training Quantization (PTQ)** is what AWQ, GPTQ, and NF4-for-QLoRA do: take a trained model and quantize the weights without further training. Works to ~INT8 / FP8 nearly losslessly; at INT4 the quality drop varies (3-8% drop with naive PTQ, ~1% with AWQ/GPTQ which use calibration data).

**Quantization-Aware Training (QAT)** makes the model experience quantization *during* training. Forward pass rounds to INT4 then dequantizes back to BF16; backward uses Straight-Through Estimator (pretend round() has gradient 1). The optimizer learns weights robust to their own rounding. Quality drop at INT4: <1%. Cost: full retraining compute, and you commit to one quantization target. See [Frontier techniques](../inference-arch/frontier-techniques.md) for QAT in DeepSeek V4 and Kimi K2.6.

## Interview Questions

**Q1: Why NF4 instead of uniform INT4?**

NN weights are approximately Gaussian — dense near zero, thin tails. Uniform INT4 spaces its 16 levels evenly, wasting most of them on near-empty tails and starving the near-zero region where weights live. NF4 places levels at the quantiles of a normal distribution — dense near zero, sparse at the extremes — which is information-theoretically optimal for normally-distributed data.

**Q2: Why does BF16 beat FP16 for training?**

Both are 16-bit, but BF16 keeps FP32's 8 exponent bits (same range) at the cost of mantissa precision, while FP16 has more mantissa but only 5 exponent bits — so FP16 overflows to inf past 65504, which wrecks gradients. BF16 never overflows, so it is the safe default for large-model training.

**Q3: What is double quantization and is it worth it?**

Block-wise quantization stores one FP32 scale constant per 64-weight block. At 70B that is ~1.1B constants = ~4.4 GB. Double quantization quantizes those constants too (FP32 → FP8, blocks of 256), saving ~0.37 bits per parameter ≈ ~26 GB at 70B scale. Small per-parameter, meaningful at scale.

**Q4: Why does QLoRA use NF4 instead of INT8 for the frozen base?**

Memory. INT8 gives 2× shrink (70B → 70 GB at INT8 vs 35 GB at NF4). To fit a 70B model on a single 48 GB GPU plus the LoRA adapter plus activations plus a paged optimizer state, you need that 4× shrink. NF4 gives 4× shrink with nearly the same quality as INT8 because of its distribution-aware level placement, so it is the right choice for the QLoRA recipe specifically.

**Q5 (Trap): If NF4 is information-theoretically optimal, why not use it everywhere?**

NF4 is optimal *for normally-distributed data*. For inference serving on NVIDIA GPUs, AWQ INT4 with the Marlin kernel runs much faster than NF4 because the kernel is hardware-optimized for INT4 matmul; NF4 is a training-time format. For inference at scale, AWQ wins on speed even though NF4 has a slight quality edge. See [Quantization formats](../serving/quantization-formats.md).
