# Normalization, the Honest Story

The textbook "internal covariate shift" framing is sloppy. The honest mechanism is that normalization forces each layer's activations to a controlled scale regardless of the weights below, which smooths the loss landscape and decouples layer scales so deep stacks do not see activations or gradients explode or vanish.

!!! tip "Rapid Recall"
    At any training step, normalization forces each layer's activations to a controlled scale (mean ≈ 0, var ≈ 1) regardless of the weights below. This smooths the loss landscape, bounds gradient magnitudes, and decouples layer scales so an 80-layer model does not see compounding 1.1× or 0.9× factors explode or vanish gradients. Transformers use LayerNorm (not BatchNorm) because variable sequence lengths, autoregressive decode, and small batches all break batch statistics. RMSNorm drops mean centering and bias, is ~7% cheaper, more stable in fp16/bf16, and is the 2026 default.

## §1 What normalization actually does

At any training step it forces each layer's activations to a controlled scale (mean ≈ 0, var ≈ 1) regardless of the weights below. This smooths the loss landscape (bounds gradient magnitudes) and decouples layer scales, so deep stacks do not see activations or gradients explode or vanish.

The real problem in deep nets: small weight updates at an early layer cascade and amplify downstream. A per-layer factor of 1.1 over 50 layers compounds to a 117× explosion; 0.9 compounds to 0.005×, a vanishing. Normalization breaks the cascade. Layer 6 always sees a normalized input no matter what layer 5 did.

## §2 LayerNorm vs BatchNorm vs RMSNorm

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 620 200" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace" font-size="12">
<text x="155" y="20" text-anchor="middle" font-weight="700" fill="#1f5e5b">BatchNorm (↓ across batch)</text>
<text x="465" y="20" text-anchor="middle" font-weight="700" fill="#b5462a">LayerNorm (→ across features)</text>
<g transform="translate(60,35)">
<rect x="0" y="0" width="190" height="130" fill="none" stroke="#1a1410" stroke-width="1.3"/>
<rect x="0" y="0" width="38" height="130" fill="rgba(31,94,91,0.25)"/>
<rect x="76" y="0" width="38" height="130" fill="rgba(31,94,91,0.25)"/>
<rect x="152" y="0" width="38" height="130" fill="rgba(31,94,91,0.25)"/>
<line x1="38" y1="0" x2="38" y2="130" stroke="#d4c4ab"/>
<line x1="76" y1="0" x2="76" y2="130" stroke="#d4c4ab"/>
<line x1="114" y1="0" x2="114" y2="130" stroke="#d4c4ab"/>
<line x1="152" y1="0" x2="152" y2="130" stroke="#d4c4ab"/>
<line x1="0" y1="43" x2="190" y2="43" stroke="#d4c4ab"/>
<line x1="0" y1="86" x2="190" y2="86" stroke="#d4c4ab"/>
<text x="-8" y="70" text-anchor="end" font-size="10" transform="rotate(-90,-8,70)">batch →</text>
</g>
<text x="155" y="185" text-anchor="middle" font-size="10">stats per feature, across examples</text>
<g transform="translate(370,35)">
<rect x="0" y="0" width="190" height="130" fill="none" stroke="#1a1410" stroke-width="1.3"/>
<rect x="0" y="0" width="190" height="43" fill="rgba(181,70,42,0.25)"/>
<rect x="0" y="86" width="190" height="44" fill="rgba(181,70,42,0.25)"/>
<line x1="38" y1="0" x2="38" y2="130" stroke="#d4c4ab"/>
<line x1="76" y1="0" x2="76" y2="130" stroke="#d4c4ab"/>
<line x1="114" y1="0" x2="114" y2="130" stroke="#d4c4ab"/>
<line x1="152" y1="0" x2="152" y2="130" stroke="#d4c4ab"/>
<line x1="0" y1="43" x2="190" y2="43" stroke="#d4c4ab"/>
<line x1="0" y1="86" x2="190" y2="86" stroke="#d4c4ab"/>
</g>
<text x="465" y="185" text-anchor="middle" font-size="10">stats per token, across its features</text>
</svg>
<figcaption>BatchNorm normalizes columns (per feature, across the batch). LayerNorm normalizes rows (per token, across its features).</figcaption>
</figure>

**Why transformers use LayerNorm, not BatchNorm:** variable sequence lengths (padding would pollute batch stats), train/inference mismatch (BN needs running averages; autoregressive decode has no batch), and small-batch instability. LayerNorm is purely per-token, identical at train and inference, and handles variable length cleanly.

## §3 RMSNorm, the 2026 default

RMSNorm (used by LLaMA, DeepSeek, Qwen, Mistral) drops mean-centering and the bias term. It only rescales by the root-mean-square. About 7% cheaper, more stable in fp16/bf16, same quality.

```
LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
RMSNorm(x)   = γ · x      / √(mean(x²) + ε)        # no mean, no bias
```

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in float32 for stability, cast back
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)
```

## Interview Questions

**Q1: Why do transformers use layer normalization and not batch normalization?**

Batch norm computes stats across the batch dimension, which creates a train/inference mismatch and couples examples in a batch. For sequence models where sequence lengths vary and batch sizes are often small (especially at inference), this is fragile. Layer norm (and RMSNorm) normalizes across the feature dimension of a single token, so each token is self-contained. No batch coupling, works identically at train and inference, handles variable length cleanly.

**Q2: What is the difference between LayerNorm and RMSNorm, and why did the field switch?**

LayerNorm centers (subtracts mean) and scales (divides by std), then applies learned gain and bias. RMSNorm drops the mean centering and the bias term: only `x / sqrt(mean(x²))` times a learned gain. It is about 7% cheaper, slightly more stable at fp16/bf16, and matches LayerNorm in quality. Llama, DeepSeek, Qwen, and most modern open LLMs use it.

**Q3: Why does normalization help training, in one sentence?**

It forces each layer's activations to a controlled scale regardless of the layers below, which breaks the per-layer cascade of small multiplicative deviations that otherwise explode or vanish gradients over deep stacks.
