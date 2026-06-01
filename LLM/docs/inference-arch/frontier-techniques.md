# Frontier Techniques: NTK RoPE, CSA+HCA, QAT, MTP

The 2026 model cards are surprisingly conservative on the base transformer (everyone uses MoE + SwiGLU + RMSNorm + pre-norm + RoPE + GQA / MLA + Flash Attention + PagedAttention + continuous batching). Innovation moved up the stack to attention sparsity, optimizer mechanics, native low-bit, and multi-token prediction. This page covers the four most important.

!!! tip "Rapid Recall"
    **NTK-aware RoPE** extends context beyond training length by changing the RoPE *base* (stretching low frequencies a lot, high frequencies barely) — preserves local syntax while making room for longer phrases. Position Interpolation slows everything uniformly and degrades local detail. **CSA + HCA** (DeepSeek V4) does 1M-context sparse attention: CSA is foveal (compress ÷4 + top-1024 select), HCA is peripheral (compress ÷128 + dense over all). Interleaved across layers gives both granularities through depth. **QAT** (Quantization-Aware Training) makes the model experience INT4/FP4 rounding *during* training via fake-quant + Straight-Through Estimator; <1% quality drop at INT4 vs 3-8% with PTQ. Kimi K2.6 ships native INT4 via QAT. **MTP** (Multi-Token Prediction) adds auxiliary heads predicting tokens N+1, N+2, N+3 with combined loss; denser training signal, and at inference the heads emit K tokens per forward pass, which is built-in speculative decoding (1.5 to 2× speedup, no separate draft model).

## §1 NTK-aware RoPE scaling

**Problem:** a model trained at 4K has only seen specific `position × frequency` rotations. At 32K it meets unfamiliar angles → attention noise → perplexity blowup.

RoPE uses many frequencies across the head dim:

\[ \text{freq}_i = \text{base}^{-2i / d_{head}} \]

- **Low dims → high freq → short wavelength →** local relationships.
- **High dims → low freq → long wavelength →** global relationships.

### 1.1 Three context-extension methods

| Method | What it does | Problem |
|---|---|---|
| Position Interpolation | `new_pos = pos / scale` — squish all positions uniformly | compresses high freqs too → local syntax degrades |
| NTK-aware | change the **base**, stretching low freqs a lot, high freqs barely | preserves local detail; needs tuned α |

!!! note "Musical analogy"
    PI slows the whole song uniformly — you lose the crisp high notes. NTK-aware only slows the bass while keeping the treble sharp: the melody stays intact, you just make room for longer phrases. The name comes from Neural Tangent Kernel theory — networks are most sensitive to high-frequency components, so do not disturb them.

Refinements: **NTK-by-parts** and **YaRN** get surgical about which frequencies to scale by how much; **Dynamic NTK** scales with current sequence length.

## §2 CSA + HCA — sparse attention at 1M tokens

Even MLA does *dense* attention over compressed K/V. At 1M tokens you still attend to 1M positions. DeepSeek V4's bet: **you do not need all tokens, and the ones you keep can be compressed.**

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg">
<text x="20" y="26" class="svg-title" fill="#f4a847">TIERED MEMORY — recent → compressed-selective → heavily-compressed-global</text>
<text x="20" y="62" class="svg-ink" fill="#e7e9ee" style="font-weight:600">CSA · foveal</text>
<text x="20" y="80" class="svg-lab" fill="#6f7686">compress ÷4, then top-1024 select (Lightning Indexer)</text>
<g>
<rect x="20" y="92" width="14" height="34" rx="2" fill="#5ad1c5" opacity="0.9"/>
<rect x="38" y="92" width="14" height="34" rx="2" fill="#2a2f3d"/>
<rect x="56" y="92" width="14" height="34" rx="2" fill="#5ad1c5" opacity="0.9"/>
<rect x="74" y="92" width="14" height="34" rx="2" fill="#2a2f3d"/>
<rect x="92" y="92" width="14" height="34" rx="2" fill="#5ad1c5" opacity="0.9"/>
<rect x="110" y="92" width="14" height="34" rx="2" fill="#2a2f3d"/>
<rect x="128" y="92" width="14" height="34" rx="2" fill="#5ad1c5" opacity="0.9"/>
<text x="155" y="114" class="svg-lab" fill="#6f7686">selected blocks only</text>
</g>
<text x="20" y="160" class="svg-ink" fill="#e7e9ee" style="font-weight:600">HCA · peripheral</text>
<text x="20" y="178" class="svg-lab" fill="#6f7686">compress ÷128, then dense attention over all ~7,800 entries</text>
<g>
<rect x="20" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="31" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="42" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="53" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="64" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="75" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="86" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="97" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="108" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<rect x="119" y="190" width="9" height="28" rx="1.5" fill="#a78bda" opacity="0.85"/>
<text x="140" y="210" class="svg-lab" fill="#6f7686">all entries, coarse</text>
</g>
<text x="430" y="100" class="svg-soft" fill="#aab0be">CSA = precise but selective</text>
<text x="430" y="122" class="svg-soft" fill="#aab0be">HCA = coarse but exhaustive</text>
<text x="430" y="150" class="svg-lab" fill="#6f7686">Interleaved across layers →</text>
<text x="430" y="168" class="svg-lab" fill="#6f7686">both granularities through depth</text>
<text x="430" y="200" class="svg-soft" fill="#7fc88a">Result: 1M context at ~10% of</text>
<text x="430" y="218" class="svg-soft" fill="#7fc88a">V3's KV cache</text>
</svg>
<figcaption>Foveal / peripheral vision for context: sharp focus on what matters, fuzzy awareness of everything else.</figcaption>
</figure>

### 2.1 CSA — compress AND select

- Compress every `m = 4` tokens → one entry (1M → 250K entries).
- Per query, attend only top-1024 entries (scored cheaply by the **Lightning Indexer** — a learned retrieval mechanism).
- Plus 128-token sliding window of raw recent tokens for local detail.
- Net: ~1,100 attention positions instead of 1,000,000 (~900× reduction).

### 2.2 HCA — compress harder, attend everywhere

- Compress every `m' = 128` tokens → one entry (1M → ~7,800 entries).
- Small enough for **dense** attention — no selection needed.
- Catches distant dependencies CSA's top-K might miss.

!!! abstract "The conceptual jump"
    The MoE intuition ported into attention: just as MoE says "not every parameter activates per token," CSA says "not every past token gets attended per query." Sparsity is the through-line.

## §3 Quantization-Aware Training (QAT)

Post-training quantization (AWQ / GPTQ / GGUF; see [Quantization formats](../serving/quantization-formats.md)) works to ~INT8 / FP8. At **INT4** (16 values) it struggles — the model never saw rounding error in training, so errors compound. QAT fixes this by making the model experience quantization *during* training.

```
FORWARD:   W_fake = dequantize(quantize(W))   # round to INT4, then back to BF16
           y = X @ W_fake                     # forward sees the rounding error
BACKWARD:  Straight-Through Estimator: pretend round() has gradient 1
           gradients flow as if quantization weren't there
```

The optimizer learns weights **robust to their own rounding**: a weight wanting 0.273 (rounds to 0.25) may shift to 0.31 if that rounds to what the network needs. After training, real quantization is near-lossless.

| | PTQ @ INT4 | QAT @ INT4 |
|---|---|---|
| Quality drop | 3 to 8% | <1% |
| Cost | cheap, post-hoc | retraining compute |
| Flexibility | any target later | committed to one target |

**Payoff at INT4:** 4× memory, ~4× faster decode (decode is bandwidth-bound). **Kimi K2.6** ships INT4 natively via QAT (~594 GB, ~2× speedup). **DeepSeek V4** applies FP4 (MXFP4) QAT to MoE experts plus the indexer's QK path — exactly the parameter-heavy parts.

!!! warning "Why not always QAT?"
    (1) It costs full retraining compute. (2) You commit to one quantization target — a QAT-INT4 model is optimized for INT4 and cannot flexibly switch to FP8 later. PTQ stays the flexible default; QAT is the all-in deployment-spec move.

## §4 Multi-Token Prediction (MTP)

A standard transformer trains one head: predict N+1. But the hidden state at position N is rich — it could predict further. MTP adds **auxiliary heads**: head 0 → N+1, head 1 → N+2, head 2 → N+3, with a combined loss:

\[ L_{\text{total}} = L_{N+1} + \lambda \cdot ( L_{N+2} + L_{N+3} + \dots ) \]

### 4.1 Training benefit — denser signal

Auxiliary-head gradients flow back through the shared body, forcing each hidden state to encode enough to support multiple future predictions. Every training token contributes multiple loss terms → richer representations, better data efficiency.

### 4.2 Inference benefit — built-in speculative decoding

Speculative decoding (see [Decoding and speculative](../serving/decoding-and-speculative.md)) normally needs a separate small "draft" model to guess ahead, verified in parallel by the big model. **MTP makes the model its own draft model**: the heads emit N+1 to N+K in one forward pass; the next pass verifies them in parallel. Matches are accepted → K tokens per weight-streaming cycle, directly attacking the decode bandwidth floor. ~1.5 to 2× speedup, no second model to train or serve.

!!! abstract "One mechanism, two payoffs"
    Training time → richer gradient signal → smarter base model. Inference time → free self-speculation → faster decode. DeepSeek V3 introduced it; V4 retains it.

## §5 The unifying meta-pattern

!!! note "Stop treating the model as a monolith"
    **Engineer different parts for different purposes; co-design training and inference instead of optimizing them separately.**

    - NTK-aware → different RoPE frequencies need different scaling.
    - CSA + HCA → different context distances need different attention granularity.
    - GQA / MLA → Q and K/V deserve different treatment.
    - MoE → not every parameter should activate per token.
    - QAT → the model should know its deployment constraints while training.
    - MTP → one mechanism serving both training and inference.
    - Prefill/decode disaggregation → opposite bottlenecks deserve separate hardware.

The recurring decode-time currency: **trade compute (surplus) for memory bandwidth (scarce)** — Flash Attention, MLA, speculative/MTP decoding, quantization all run that play.

## Interview Questions

**Q1: What is NTK-aware scaling and why does it preserve local syntax?**

RoPE encodes position via rotations whose frequencies span from high (low dims, local) to low (high dims, global). Position Interpolation uniformly squishes positions, which compresses high-frequency rotations the model had memorized and degrades local syntax. NTK-aware instead changes the *base* of the frequency formula, which stretches the low frequencies a lot but barely touches the high frequencies — local rotations stay close to training, global rotations stretch to accommodate longer context. The musical version: PI slows the whole song; NTK only slows the bass.

**Q2: What do CSA and HCA buy you and how do they compose?**

CSA gives precise but selective attention (compress ÷4, then top-1024 select via Lightning Indexer) — sharp focus on the parts that matter. HCA gives coarse but exhaustive attention (compress ÷128, dense over all ~7,800 entries) — fuzzy awareness of everything else. Interleaved across layers, every layer gets both granularities. Result: DeepSeek V4 serves 1M context at ~10% of V3's KV cache. The conceptual jump is MoE for attention: not every past token needs to be attended.

**Q3: How does QAT differ from PTQ and when is each preferred?**

PTQ quantizes a trained model post-hoc with calibration data; cheap, flexible, ~3 to 8% drop at INT4. QAT makes the model experience fake-quantization during training via Straight-Through Estimator gradients; costs full retraining compute and commits to one target (INT4 vs FP8), but gets <1% drop at INT4. PTQ is the flexible default for deployment; QAT is the all-in move for a model you plan to ship at one fixed precision (Kimi K2.6 INT4, DeepSeek V4 FP4 for MoE experts).

**Q4: What does MTP give you at training vs inference?**

Training: denser gradient signal — every token contributes multiple loss terms (N+1, N+2, N+3, ...), forcing the shared body to learn representations that support multiple-step prediction. Better data efficiency. Inference: built-in speculative decoding — the heads emit K candidate tokens in one forward pass, the next pass verifies them in parallel. ~1.5 to 2× decode speedup, no separate draft model to train or serve. Same mechanism, two payoffs.

**Q5 (Trap): If MTP gives you speculative decoding for free, why does anyone still train a separate draft model?**

MTP heads share the trunk, so the draft predictions are constrained to what the trunk can encode for a near-future step — typically 2 to 4 tokens of meaningful lookahead, not 8 or 16. A separate small draft model (like EAGLE-3 or vanilla speculative) can predict further ahead and on more diverse distributions. Production servers running long-form generation often still prefer EAGLE-3 style for higher acceptance rates over MTP's free 2 to 4 tokens, especially for non-MTP-trained models.
