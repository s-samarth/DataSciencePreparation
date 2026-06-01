# Memory Tricks: Paged Optimizers, Gradient Accumulation, Gradient Checkpointing

Three orthogonal memory techniques that complete the QLoRA story. Paged optimizers swap optimizer state to CPU RAM via NVIDIA Unified Memory. Gradient accumulation fakes a large batch by splitting it into micro-batches. Gradient checkpointing trades compute for memory by recomputing activations. In practice you run all three together.

!!! tip "Rapid Recall"
    Three orthogonal memory tricks. **Paged optimizer** uses NVIDIA Unified Memory to spill Adam's `m, v` state to CPU RAM and page back to GPU just-in-time at the optimizer step; the optimizer step is the only moment that state is touched, so it can live in CPU most of the time. Cost: 5 to 40% slowdown depending on paging. **Gradient accumulation** processes a large batch as K micro-batches, summing `.grad` across them before one `optimizer.step()`. Mathematically identical to one big batch with LayerNorm (not with BatchNorm). Trades time for memory. **Gradient checkpointing** stores activations only at checkpoint layers and recomputes the gaps during backward. Trades compute for memory; typical 25 to 40% slowdown for 4× activation memory savings.

## §1 Paged optimizers

### 1.1 The memory crisis: Adam stores 3× the model

Each parameter under Adam needs the weight itself, the first moment `m`, and the second moment `v`. Both `m` and `v` are kept in FP32 for stability.

```
7B model + Adam:

Model weights (BF16):    7B × 2  = 14 GB
Gradients (BF16):        7B × 2  = 14 GB
Adam m (FP32):           7B × 4  = 28 GB
Adam v (FP32):           7B × 4  = 28 GB
Master weights (FP32):   7B × 4  = 28 GB
                                  ───────
TOTAL:                            112 GB     # model itself is only 14 GB
```

Optimizer state alone (m + v) is **4× the model size**. For 70B: ~1.1 TB. Impossible on any single GPU.

### 1.2 The idea: GPU memory virtual-memory

!!! note "OS analogy"
    Your laptop has 16 GB RAM but runs programs wanting 32 GB — the OS keeps hot data in RAM and pages cold data to disk, swapping transparently. Paged optimizers do exactly this between **GPU memory (fast, small)** and **CPU RAM (slow, large)**, using NVIDIA Unified Memory.

### 1.3 The key timing insight

Optimizer state is **only touched during the optimizer step**. During forward and backward (most of training time), you only need weights + activations on GPU. So optimizer state can sit in CPU RAM and get pulled to GPU just-in-time for the step. Pages move at 4 KB granularity; an LRU policy keeps hot pages resident.

### 1.4 Paged vs manual CPU offloading

| | Manual offloading | Paged optimizer |
|---|---|---|
| Who decides placement | You, ahead of time | CUDA driver, on demand |
| Granularity | Whole tensors | 4 KB pages |
| OOM on memory spikes | Likely (must size for peak) | Avoided — pages evict during spikes |
| Code changes | Significant | Drop-in optimizer swap |

### 1.5 Using it

```python
import bitsandbytes as bnb

# Instead of torch.optim.AdamW(...)
optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=2e-4)
# PagedAdamW8bit = 8-bit quantized state + paging. QLoRA default.
# Variants: PagedAdamW32bit, PagedLion8bit
```

!!! abstract "Cost and payoff"
    CPU RAM is ~50 to 100× slower than GPU memory. Slowdown: ~5 to 10% if state mostly fits in GPU, 20 to 40% with real paging, up to 2 to 3× if thrashing. But a 30%-slower run that *completes* beats a faster run that **OOMs**. Paged optimizers are the unsung third hero of QLoRA — they prevent OOM during memory spikes.

## §2 Gradient accumulation

### 2.1 Why batch size matters

Each batch gives an estimate of the true gradient. Small batch → noisy estimate → unstable training. Large batch → smoother → more stable. But each sample eats activation memory, so a 24 GB GPU may only fit batch 4 when you want batch 32.

### 2.2 The core trick

Compute the gradient over 8 micro-batches of 4 **without updating weights between them**, sum the gradients, then do one update. Mathematically identical to one batch of 32.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 820 200" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<defs>
<marker id="gaarr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
<path d="M0,0 L9,4.5 L0,9 Z" fill="#6f6d68"/>
</marker>
</defs>
<text x="20" y="30" fill="#a4a29c" font-size="11">8 × micro-batch (size 4) — gradients accumulate, no update</text>
<g>
<rect x="20" y="44" width="78" height="40" rx="6" fill="#16181d" stroke="#5ec8ff"/><text x="59" y="68" fill="#5ec8ff" font-size="10" text-anchor="middle">mb 1</text>
<rect x="106" y="44" width="78" height="40" rx="6" fill="#16181d" stroke="#5ec8ff"/><text x="145" y="68" fill="#5ec8ff" font-size="10" text-anchor="middle">mb 2</text>
<rect x="192" y="44" width="78" height="40" rx="6" fill="#16181d" stroke="#5ec8ff"/><text x="231" y="68" fill="#5ec8ff" font-size="10" text-anchor="middle">mb 3</text>
<text x="300" y="68" fill="#6f6d68" font-size="14" text-anchor="middle">· · ·</text>
<rect x="340" y="44" width="78" height="40" rx="6" fill="#16181d" stroke="#5ec8ff"/><text x="379" y="68" fill="#5ec8ff" font-size="10" text-anchor="middle">mb 8</text>
</g>
<text x="219" y="108" fill="#6f6d68" font-size="9" text-anchor="middle">.backward() ADDS into .grad each time</text>
<path d="M430,64 L470,64" stroke="#d4ff5e" stroke-width="1.5" marker-end="url(#gaarr)"/>
<rect x="478" y="44" width="150" height="40" rx="6" fill="#16181d" stroke="#d4ff5e"/>
<text x="553" y="62" fill="#d4ff5e" font-size="10" text-anchor="middle" font-weight="600">optimizer.step()</text>
<text x="553" y="76" fill="#6f6d68" font-size="9" text-anchor="middle">one update / 8 mb</text>
<text x="553" y="108" fill="#6ee7a0" font-size="9" text-anchor="middle">≡ one batch of 32</text>
</svg>
<figcaption>PyTorch's .backward() accumulates into .grad — 8 calls without zero_grad() = the sum, which equals a true batch-32 gradient.</figcaption>
</figure>

### 2.3 Why it is mathematically exact

Batch loss is an average; the gradient of an average is the average of gradients (linearity). Splitting N into K micro-batches, each micro-batch gradient is K× its share, so you divide each micro-batch loss by K before backward:

\[ \nabla L_{\text{batch}} = \frac{1}{N} \sum_i \nabla L(x_i, y_i) \]

That is why the code uses `loss / accumulation_steps` — to recover the correct average.

```python
accum = 8
for i, micro_batch in enumerate(dataloader):
    loss = model(micro_batch) / accum     # normalize
    loss.backward()                        # gradients accumulate into .grad
    if (i + 1) % accum == 0:
        optimizer.step()
        optimizer.zero_grad()
```

!!! warning "The gotcha: BatchNorm breaks"
    BatchNorm computes statistics over the batch dimension. With micro-batches of 4, it only sees 4 samples at a time, so its running stats differ from true batch-32 training. With **LayerNorm** (used in all modern LLMs), gradient accumulation is *exactly* identical to large-batch training. One reason transformers moved to LayerNorm. It plays nicely with every memory trick.

## §3 Gradient checkpointing

### 3.1 Why activations dominate

To compute `∂L/∂x` for `y = f(x)`, you need `x` from the forward pass, so PyTorch caches every intermediate. For L layers:

\[ \text{Activation memory} \approx L \times B \times S \times d \times \text{bytes} \]

For a 32-layer transformer (d = 4096, batch = 4, seq = 2048, BF16), activations can hit **~67 GB**, far more than the 14 GB of weights.

### 3.2 The core trick: store checkpoints, recompute the gaps

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 820 220" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<defs>
<marker id="gcarr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
<path d="M0,0 L9,4.5 L0,9 Z" fill="#ff8b5e"/>
</marker>
</defs>
<text x="20" y="26" fill="#a4a29c" font-size="11">Forward — store only at checkpoints (★), discard the rest</text>
<g font-size="9" text-anchor="middle">
<rect x="30" y="40" width="36" height="30" rx="4" fill="#16181d" stroke="#d4ff5e"/><text x="48" y="60" fill="#d4ff5e">L4★</text>
<rect x="76" y="40" width="36" height="30" rx="4" fill="#1c1f26" stroke="#2a2d35"/><text x="94" y="60" fill="#6f6d68">L3</text>
<rect x="122" y="40" width="36" height="30" rx="4" fill="#1c1f26" stroke="#2a2d35"/><text x="140" y="60" fill="#6f6d68">L2</text>
<rect x="168" y="40" width="36" height="30" rx="4" fill="#1c1f26" stroke="#2a2d35"/><text x="186" y="60" fill="#6f6d68">L1</text>
<rect x="214" y="40" width="36" height="30" rx="4" fill="#16181d" stroke="#d4ff5e"/><text x="232" y="60" fill="#d4ff5e">L0★</text>
</g>
<text x="140" y="92" fill="#6f6d68" font-size="9" text-anchor="middle">only ★ stored → ~4× less activation memory</text>
<text x="20" y="135" fill="#a4a29c" font-size="11">Backward — need L2's activation, don't have it → recompute from L0★</text>
<g font-size="9" text-anchor="middle">
<rect x="214" y="150" width="36" height="30" rx="4" fill="#16181d" stroke="#ff8b5e"/><text x="232" y="170" fill="#ff8b5e">L0★</text>
<path d="M252,165 L286,165" stroke="#ff8b5e" stroke-width="1.3" marker-end="url(#gcarr)"/>
<rect x="290" y="150" width="40" height="30" rx="4" fill="#1c1f26" stroke="#ff8b5e" stroke-dasharray="3 2"/><text x="310" y="170" fill="#ff8b5e">recompute L1</text>
<path d="M332,165 L360,165" stroke="#ff8b5e" stroke-width="1.3" marker-end="url(#gcarr)"/>
<rect x="364" y="150" width="40" height="30" rx="4" fill="#1c1f26" stroke="#ff8b5e" stroke-dasharray="3 2"/><text x="384" y="170" fill="#ff8b5e">L2 ✓</text>
</g>
</svg>
<figcaption>Checkpoint a few layers; when backward needs a discarded activation, re-run forward from the nearest checkpoint.</figcaption>
</figure>

### 3.3 The tradeoff

| Setting | GPU memory | Training time |
|---|---|---|
| No checkpointing | 70 GB | 1.00× |
| Checkpoint every 4 layers | 30 GB | 1.25× |
| Checkpoint every layer | 18 GB | 1.40× |

For LLMs, the sweet spot is checkpointing at the transformer-block boundary.

```python
# Manual
from torch.utils.checkpoint import checkpoint
x = checkpoint(self.attention, x, use_reentrant=False)

# Hugging Face one-liner
model.gradient_checkpointing_enable()
```

## §4 The three tricks side by side

| | Gradient Accumulation | Gradient Checkpointing | Paged Optimizer |
|---|---|---|---|
| Problem solved | Batch too big for memory | Activations too big for memory | Optimizer state too big for memory |
| Mechanism | Split batch, sum gradients, update once | Store few activations, recompute rest in backward | Page optimizer state to CPU RAM |
| Trades | Time for memory | Compute for memory | Some speed for OOM resilience |
| Enables | Larger effective batch | Larger model / longer sequences | Adam on huge models |
| Caveat | BatchNorm stats differ (LayerNorm fine) | 25 to 40% slower | 5 to 40% slower depending on paging |

In practice you run all three at once: `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `gradient_checkpointing=True`, and `PagedAdamW8bit` as the optimizer.

## Interview Questions

**Q1: What does a paged optimizer do, and what does it cost?**

It treats CPU RAM as overflow for GPU memory, automatically paging optimizer state (m, v) in and out — like OS virtual memory — using NVIDIA Unified Memory. Optimizer state is only needed during the optimizer step, so it can live in CPU RAM most of the time. Cost is speed (~5 to 40% slowdown depending on paging), but it prevents OOM during memory spikes. It is the third hero of QLoRA alongside NF4 and LoRA.

**Q2: Is gradient accumulation perfectly equivalent to large-batch training?**

For the gradient, yes — with LayerNorm. The trap is BatchNorm: it computes statistics over the batch dimension, so micro-batches of 4 give different running stats than a true batch of 32. Modern LLMs use LayerNorm (no batch-dimension dependency), so accumulation is mathematically identical. One reason transformers adopted LayerNorm.

**Q3: Gradient accumulation vs gradient checkpointing — what is the difference?**

Different problems. Accumulation solves "batch too big" by splitting it into micro-batches and summing gradients before one update — trades time for memory, enables larger effective batch. Checkpointing solves "activations too big" by storing only a few activations and recomputing the rest in backward — trades compute for memory, enables larger models or longer sequences. You usually run both together.

**Q4: Why is the standard advice to checkpoint at the transformer-block boundary?**

Because activations within a block (after attention, after FFN, the residual sums) are correlated in size and recompute together cleanly, but cross-block boundaries are natural cut points where the residual stream is the only thing flowing forward. Storing the residual at every block boundary and recomputing the inside of each block on backward gives ~4× activation savings for ~25% extra compute, which is the best tradeoff for most LLM training runs.

**Q5: You enable gradient checkpointing and your training run gets a 40% slower. Why?**

Backward now has to redo a forward pass through each checkpointed segment to recompute the dropped activations. The naive estimate is "1 forward + 1 backward + 1 recompute = 3/2 the original work," which matches the empirical ~40% slowdown for whole-block checkpointing. Speed up by checkpointing less aggressively (every 2 blocks instead of every block), or use `torch.utils.checkpoint(..., use_reentrant=False)` for a slight speedup and better autograd compatibility.
