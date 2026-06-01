# QLoRA, Fully Assembled

QLoRA = quantize the base to 4-bit NF4, run LoRA adapters in BF16 on top, let paged optimizers handle the optimizer state. Result: fine-tune a 70B model on a single 48 GB GPU. This page is the memory budget walk-through and the standard production code.

!!! tip "Rapid Recall"
    QLoRA's three tricks. (1) Base model in NF4 (4-bit): 70B × 0.5 bytes ≈ 35 GB, frozen. (2) LoRA adapters in BF16, trainable, ~0.5 GB. (3) Paged optimizer keeps Adam's `m, v` state in CPU RAM and pages into GPU just-in-time at the optimizer step. Compute happens in BF16: NF4 weights are dequantized just-in-time for each matmul, then the dequantized copy is discarded. The α/r scaling factor stays the same as LoRA; **RSLoRA** (use_rslora=True) changes scaling to α/√r and is more stable at high ranks (r > 64).

## §1 The component table

| Component | Precision | Trained? | Where it lives |
|---|---|---|---|
| Base model weights | NF4 (4-bit) | frozen | GPU (dequantized just-in-time for matmul) |
| LoRA adapters (A, B) | BF16 | yes | GPU |
| Optimizer state (m, v) | FP32 / 8-bit | — | Paged → CPU RAM, pulled to GPU at step |

## §2 The 70B memory budget on one 48 GB GPU

```
Base (NF4, 4-bit):    70B × 0.5 bytes  ≈ 35 GB
LoRA adapters (BF16): <<1B × 2 bytes   ≈ 0.5 GB
Adapter gradients:                      ≈ 0.5 GB
Optimizer state:      paged to CPU      ≈ ~0 GB resident
Activations + overhead:                 ≈ 8 to 10 GB
                                        ──────────
Total resident:                         ≈ 44 to 46 GB  → fits in 48 GB
```

The compute happens in BF16: NF4 weights are dequantized **just-in-time** for each matrix multiply, then the dequantized copy is discarded. You never hold the full BF16 base model in GPU memory — only the 4-bit version plus the small chunk currently in flight.

## §3 The α/r scaling factor

LoRA's effective contribution magnitude depends on `r` — by random-matrix arguments, roughly proportional to `1/√r` at init. So changing `r` implicitly changes how aggressively the adapter updates:

- r = 8 → BA has some scale → effective learning rate is X.
- r = 64 → BA has a different scale → effective learning rate is ~8X.

Without correction, every time you sweep r during hyperparameter search you would silently change the learning rate too. You would have to re-tune LR for every r. Annoying and error-prone.

!!! abstract "What α/r does"
    It normalizes BA's contribution so the effective scale is approximately constant regardless of r. Tune **α once**, then sweep r freely. The two effects of raising r — more capacity, but smaller per-step contribution — roughly balance.

### Picking values

| Setting | Scale (α/r) | Behaviour |
|---|---|---|
| α = 16, r = 8 | 2.0 | Common default — adapter contributes strongly. |
| α = 16, r = 16 | 1.0 | More capacity, gentler per-step update. |
| α = 2r (rule of thumb) | 2.0 always | Keeps scale fixed while you change r. |
| RSLoRA (2023) | α/√r | Argued more stable at high ranks; `use_rslora=True`. |

High α = adapter updates more aggressively (risk of overfit on small data). Low α = adapter stays closer to the base model. Having α as a separate knob lets you control adapter *strength* independently of its *expressiveness* (rank).

## §4 The standard 2026 code

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16
    bnb_4bit_use_double_quant=True,         # double quantization
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16, lora_alpha=32,                    # scale = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable: ~0.05%
```

## §5 2026 ecosystem notes

The PEFT + bitsandbytes + Unsloth stack is the open-source default in 2026.

- **Hugging Face PEFT** — official adapter library.
- **bitsandbytes** — provides NF4 quantization and 8-bit / paged optimizers.
- **Unsloth** — custom CUDA kernels giving ~2× faster QLoRA at the same quality.
- **DoRA** (`use_dora=True` in PEFT) — decomposes weight into magnitude + direction. Consistently ~0.5 to 1% better than vanilla LoRA at the same rank. ~10% more memory.
- **RSLoRA** (`use_rslora=True`) — α/√r scaling, more stable at high ranks.
- **TRL `SFTTrainer`** — wraps the whole thing with chat templates, loss masking, and PEFT injection in one trainer.

## §6 What you give up vs full fine-tuning

QLoRA is ~30 to 40% slower than vanilla LoRA on FP16 due to dequantization overhead. Worth it any time you are GPU-constrained.

QLoRA is a *training* technique, not typically a *serving* technique. After QLoRA training, you usually merge the BF16 adapter back into the BF16 base, then re-quantize to your deployment format (AWQ INT4 or FP8) for production serving. Deploying directly with bitsandbytes 4-bit runtime quant is slower than AWQ/GPTQ, so it is OK for dev but not production throughput. See [Quantization formats](../serving/quantization-formats.md).

## Interview Questions

**Q1: How does QLoRA fit a 70B model on a single 48 GB GPU?**

Three tricks. NF4 quantizes the frozen base to 4-bit (140 GB → ~35 GB). LoRA adapters run in BF16 but are tiny. Paged optimizers spill optimizer state to CPU RAM. ~35 GB base + ~10 GB overhead fits in 48 GB. Compute happens in BF16 by dequantizing NF4 weights just-in-time, then discarding the dequantized copy.

**Q2: What is the α/r scaling factor and why does it matter?**

BA's magnitude depends on r, so changing r implicitly changes the adapter's effective learning rate — going from r = 8 to r = 64 would scale it ~8×. α/r normalizes this so you tune α once and sweep r freely. Common default α = 2r (scale 2). RSLoRA uses α/√r, argued to be more stable at high ranks. High α = stronger / more aggressive adapter; low α = stays closer to base.

**Q3: Is QLoRA a training technique or a serving technique?**

Training, primarily. After QLoRA training, you typically merge the BF16 adapter into the BF16 base, then re-quantize to AWQ INT4 or FP8 for production serving — those are faster at inference than the bitsandbytes 4-bit runtime quant QLoRA uses during training. The 4-bit base in QLoRA is about getting the model into VRAM to train it, not about deployment speed.

**Q4 (Trap): If QLoRA quantizes the base to 4-bit and the adapter is in BF16, where does the actual matmul happen and in what precision?**

The matmul happens in BF16. At each forward step the NF4 base weights are dequantized just-in-time to BF16, the matmul runs in BF16, and the dequantized copy is discarded before the next layer. The 4-bit format is purely for storage; the compute path stays in BF16 so gradients and adapter updates remain numerically well-behaved.

**Q5: What is DoRA and when is it worth turning on?**

DoRA (Weight-Decomposed LoRA, 2024) decomposes each weight matrix into magnitude and direction components and trains both with LoRA-like efficiency. Consistently 0.5 to 1% better than vanilla LoRA on most benchmarks. About 10% more memory and a similar small speed cost. Available in PEFT via `use_dora=True`. Worth trying if you are squeezing the last bit of performance from a fixed compute budget.
