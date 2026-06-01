# Quantization Formats for Inference

The #1 lever for cutting inference cost. In 2026 this is table stakes — unquantized serving is how you burn money. A neural network weight stored as FP16 (16-bit float) is 2 bytes. INT8 (1 byte) cuts memory in half. INT4 (4 bits) cuts it by 4×. With 4× less memory, you fit bigger models, serve bigger batches, and burn less HBM bandwidth per token — faster AND cheaper.

!!! tip "Rapid Recall"
    Three smart 4-bit methods, all weight-only quant. **GPTQ** minimizes layer-wise output error via second-order info (Hessian); 2-4h for 7B; near-lossless. **AWQ** identifies the ~1% of "salient" weight channels (those connected to activation outliers), scales them up before quantization to give them more INT4 levels, then scales back at inference; 10× faster to apply than GPTQ and slightly better quality; the 2026 default for GPU serving on NVIDIA with Marlin kernels. **NF4** is QLoRA training only (see [Quantization and NF4](../peft/quantization-nf4.md)); not for serving. **FP8** is hardware-native on Hopper/Blackwell, near-lossless, 2× FP16. **2026 decision tree:** FP8 if H100+, AWQ INT4 if VRAM-tight, INT8 if safe-default, GGUF Q4_K_M for local/CPU. **Trap:** AWQ without the Marlin kernel is actually *slower* than FP16 because dequant overhead exceeds memory savings; always benchmark on your hardware.

## §1 The tradeoff dial

```
Precision:   FP32  →  FP16/BF16  →  FP8  →  INT8  →  INT4  →  INT2/binary
Memory:      4x        2x           1x      1x       0.5x     0.25x (vs INT8)
Quality:     full      ~99.5%       ~99%    ~99%     95-98%   crashes
```

Each step down the dial trades quality for memory and speed. **The game is picking the right step for your hardware and task.**

## §2 The key formats

### 2.1 INT8 — the safe default

- **Memory savings:** 2× vs FP16.
- **Quality loss:** <1% on most benchmarks.
- **When to use:** production workhorse. When in doubt, INT8.
- **How it works:** for each weight group, compute `scale = (max - min) / 255`, then `q = round(w / scale)`. At inference, dequantize: `w ≈ q * scale`.
- **Gotcha:** activation outliers. Transformers have ~0.1% of activations that are 100× larger than the rest. Naive INT8 on these breaks quality. Solved by **LLM.int8()** (bitsandbytes) which keeps outlier activations in FP16 and quantizes the rest.

### 2.2 INT4 — the aggressive default (with smart methods)

- **Memory savings:** 4× vs FP16.
- **Quality loss:** 2-5% with naive methods, ~1% with GPTQ/AWQ.
- **When to use:** fit a 70B model on a single H100, or a 7B on a laptop GPU.
- **Critical:** naive INT4 does NOT work. You MUST use GPTQ, AWQ, or NF4.

### 2.3 FP8 — the new kid (2025+)

- **Memory savings:** 2× vs FP16 (same as INT8).
- **Quality:** better than INT8 because floating-point preserves dynamic range.
- **When to use:** on Hopper (H100) or Blackwell (B100/B200) hardware with native FP8 tensor cores. The frontier default in 2026 for labs that can afford H100s.

### 2.4 NF4 — for QLoRA training

Covered in [Quantization and NF4](../peft/quantization-nf4.md). Not typically used for production inference (use AWQ/GPTQ instead).

## §3 The three smart methods for 4-bit

This is the interview-critical section. These are all **weight-only** quantization methods — they quantize weights but keep activations in FP16/BF16.

### 3.1 GPTQ (Frantar et al., ICLR 2023)

**Intuition:** quantize weights layer by layer, using approximate second-order information (Hessian) to minimize the change in the layer's output, not just the weights themselves.

**Recipe:**

1. Collect a small calibration dataset (~128 samples).
2. For each layer: quantize one column of weights at a time, and **update the remaining unquantized columns** to compensate for the introduced error.
3. Uses the Hessian of the reconstruction loss (cheaply approximated) to decide compensation.

**Math:** for a single layer with input X, minimize \(\|WX - \hat{Q}X\|^2_F\) (Frobenius norm of the output difference). The Hessian H = 2XX^T captures how each weight affects output; after quantizing column q, update remaining: `W_updated = W - (W_q - Q_q) · H_q^{-1} · H_remaining`.

**Result:** near-lossless INT4 weight quantization. Widely supported (vLLM, HF Transformers, TGI, text-generation-webui). Most open-source community quants (TheBloke, etc.) are GPTQ.

### 3.2 AWQ (Lin et al., MLSys 2024 Best Paper)

**Intuition:** not all weights are equally important. A small fraction (~1%) of weight channels are "salient" — they correspond to activation outliers. Protect these channels by **scaling them up before quantization** so they get more bins in the INT4 range, then compensate at inference by scaling the activations down.

**Recipe:**

1. Use calibration data to find activation outlier magnitudes per channel: `s = max(|X_channel|)`.
2. Compute per-channel scaling factors.
3. Replace W with `W * s^α` (α ≈ 0.5) and inputs with `x / s^α` (mathematically equivalent).
4. Quantize the scaled weights — the important channels now span more of the INT4 range.

**Result:** often beats GPTQ by a small margin on perplexity. **Hardware-friendly** — the quantized format runs well on GPUs because the scaling is per-channel (not per-group), enabling efficient INT4 matmul kernels with the **Marlin** kernel. AWQ is typically the right answer for production serving on NVIDIA GPUs as of 2026.

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)

quant_config = {
    "zero_point": True,    # asymmetric quantization
    "q_group_size": 128,   # weights per quantization group
    "w_bit": 4,            # target bit-width
    "version": "GEMM",     # use Marlin kernel: version="Marlin"
}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama-3.1-8b-awq-int4")
# Takes 10-30 min for 8B, 1-3 hours for 70B on single GPU.
```

## §4 The 2026 decision tree

```
Do you have Blackwell / Hopper (H100, H200, B100)?
├── YES: Use FP8 (native tensor-core support, near-lossless)
└── NO: Is VRAM tight?
         ├── YES: Use AWQ INT4 (best quality at 4-bit on GPU)
         └── NO: Use INT8 (safest, ~99% quality)

Edge cases:
  - CPU/hybrid deployment → GGUF Q5_K_M (llama.cpp format)
  - Training then deploying a LoRA → QLoRA NF4 for training,
    then merge + re-quantize to AWQ for deploy
  - Code/math/reasoning model → Prefer INT8 or FP8.
    INT4 hurts these more than general chat.
```

## §5 Library landscape

| Library | Best for | Format |
|---|---|---|
| **bitsandbytes** | Fast iteration, QLoRA training | NF4, LLM.int8() |
| **AutoAWQ** | Production GPU serving | AWQ INT4 |
| **AutoGPTQ / GPTQModel** | Community compat, broad model support | GPTQ INT4/3 |
| **llama.cpp / GGUF** | CPU + consumer GPU hybrid | Q2-Q8 |
| **TensorRT-Model-Optimizer** | NVIDIA production inference | FP8, INT8, INT4 |
| **vLLM** (supports all) | Full serving stack | FP8, INT8, AWQ, GPTQ, GGUF |

## §6 Memory cheat sheet

For a model of P billion parameters:

| Precision | Bytes/param | Memory for 7B | Memory for 70B |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16/BF16 | 2 | 14 GB | 140 GB |
| INT8/FP8 | 1 | 7 GB | 70 GB |
| INT4 (AWQ) | 0.5 | 3.5 GB | 35 GB |

**Plus add 15-25% for KV cache, activations, and framework overhead at typical batch sizes.**

This is the math that lets a 70B model run on a single 80 GB H100 at AWQ INT4 with room for a reasonable KV cache.

## §7 Quality comparison

| | AWQ | GPTQ | GGUF (Q4_K_M) | FP8 |
|---|---|---|---|---|
| Quality retention | ~95-99% | ~90-95% | ~92% | ~99% |
| Inference speed (vLLM+Marlin) | 741 tok/s (fastest GPU) | 712 tok/s | N/A (CPU) | ~2× FP16 |
| Quantization time (7B) | 10-30 min | 2-4 hours | N/A (download) | Built-in |
| Hardware | NVIDIA GPU | NVIDIA GPU | CPU + any GPU | H100/Hopper+ |
| Best for | Production GPU serving | Pre-quant model availability | Local dev / CPU | H100 production |

**Decision rule:** FP8 on H100 first (if hardware supports it). Otherwise AWQ for new quantization. GPTQ only if you need a pre-quantized model that does not exist in AWQ format. GGUF for local / Ollama.

## §8 The trap question

!!! warning "Does quantization always speed up inference?"
    **No.** AWQ without an optimized kernel like Marlin is actually slower than FP16, because the hardware has to dequantize INT4 weights back to FP16 before matrix multiplication, and that overhead exceeds the memory bandwidth savings. Quantization speeds up inference when (a) you are memory-bandwidth bound (typical for LLM decode) AND (b) your hardware has kernel support for efficient quantized ops (Marlin on NVIDIA, hardware-native FP8 on H100). Always benchmark on your actual hardware before assuming quantization = faster.

## Interview Questions

**Q1: What is the difference between GPTQ and AWQ?**

Both are post-training INT4 quantization methods. GPTQ minimizes the layer-wise output error using second-order information (the Hessian of each layer) — mathematically principled but expensive to run (2-4 hours for 7B). AWQ takes a different approach: identifies the ~1% of weights that are "salient" (connected to high-magnitude activations) and protects them during quantization by scaling them up before quantization and scaling back after. AWQ is 10× faster to apply, produces better quality models, and the resulting format runs faster with the Marlin kernel. In 2026, AWQ is the recommended INT4 format for production GPU serving.

**Q2: When would you choose GPTQ over AWQ, or vice versa?**

Both achieve near-lossless quality. AWQ typically has a slight quality edge because it protects salient channels via activation-aware scaling, and the quantized format is more hardware-friendly on NVIDIA GPUs — faster INT4 kernels. GPTQ has much broader community support — nearly every open-source model on HuggingFace has a GPTQ version available. In 2026, default to AWQ for new production deployments; GPTQ is the fallback when AWQ is not available for a specific model.

**Q3: Why does AWQ's activation-aware scaling preserve quality at INT4?**

Activation outliers (the ~1% of weight channels connected to large activations) dominate the layer's output. Naive INT4 spaces 16 levels evenly across all weight channels, giving the salient channels the same resolution as everything else and crushing their precision. AWQ multiplies the salient weights by a per-channel scaling factor before quantization, giving them more INT4 levels' worth of range; at inference the activation is divided by the same factor so the math is identical, but the salient channels' quantization error is much smaller. The 99% of unimportant channels lose precision, but no one cares about those.

**Q4 (Trap): Does quantizing a model make it faster on any hardware?**

No. AWQ without an optimized kernel like Marlin is actually slower than FP16 because the hardware has to dequantize INT4 weights back to FP16 before matrix multiplication, and that overhead exceeds the memory bandwidth savings. Quantization speeds up inference when (a) you are memory-bandwidth bound (typical for LLM decode) AND (b) your hardware has kernel support for efficient quantized ops. Always benchmark on your actual hardware before assuming quantization = faster.

**Q5: When does quantization NOT make sense?**

For reasoning-heavy, math, and code generation tasks at very low bit-widths. Quality degrades nonlinearly below 4-bit — INT2 is usually unusable. Even at 4-bit, GPTQ degrades noticeably on coding benchmarks (~10% below baseline Pass@1). For tasks where quality is non-negotiable (medical, legal), FP8 on Hopper hardware is the better tradeoff — minimal quality loss with 2× throughput vs FP16. Also: do not quantize if you are planning to fine-tune. QLoRA (bitsandbytes NF4) is the exception — it trains adapters around frozen 4-bit weights.

**Q6: Name the 2026 quantization landscape in one breath.**

Three tiers. (1) PTQ stack for GPU: AWQ INT4 (production default with Marlin), GPTQ INT4 (broader model coverage), GGUF for CPU. (2) Hardware-native: FP8 on Hopper, NVFP4 on Blackwell — software work is minimal. (3) Trained-from-scratch ternary: BitNet b1.58 (research bet, 100B-class models at 5-7 tok/s on CPU, 55-82% energy reduction). TGI entered maintenance mode in December 2025; HuggingFace officially recommends vLLM or SGLang now.
