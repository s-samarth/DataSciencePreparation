# LoRA Mechanics

Instead of updating a giant weight matrix directly, learn two small matrices whose product approximates the update. Freeze the original, train only the add-on. After training, merge for zero inference overhead. This page is the mechanics: the decomposition, the forward pass, the parameter math, the merge trick, and the standard hyperparameters.

!!! tip "Rapid Recall"
    For a frozen weight `W ∈ ℝ^(d×d)`, LoRA learns `ΔW = BA` with `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×d)`, `r ≪ d`. Forward becomes `h = Wx + (α/r) · BAx`. At d = 4096, r = 8: full update has 16.77M params; LoRA has 65,536 — a **99.6% reduction**. After training, merge: `W_new = W + (α/r)·BA`. Single matrix, zero inference overhead. Keep adapters unmerged when you want multi-task serving (one base + many adapters hot-swapped per request) or continued training (merging is a one-way door).

## §1 The sticky-note analogy

!!! note "Mental model"
    The pretrained model is an expert who already knows 90% of the job. Full fine-tuning replaces their brain (expensive, destructive). LoRA hands them a **sticky note** of domain corrections — tiny, non-destructive, and you can staple it into the brain afterward (merge) so there is zero inference overhead.

## §2 The decomposition

For a weight matrix \(W \in \mathbb{R}^{d \times d}\), rather than learning the full update \(\Delta W\) (which has d² entries), learn:

\[ \Delta W \approx B \cdot A \quad \text{where } B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, \; r \ll d \]

Two thin matrices, squeezed through a rank-r bottleneck. The forward pass runs the frozen path and the adapter path in parallel, then adds them:

\[ h = Wx + \Delta W x = Wx + (\alpha/r) \cdot BAx \]

`W` is frozen (no gradient). Only `A` and `B` are trained.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 820 280" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<rect x="40" y="60" width="150" height="150" rx="6" fill="#1c1f26" stroke="#6f6d68" stroke-width="1.5"/>
<text x="115" y="138" fill="#a4a29c" font-size="13" text-anchor="middle" font-weight="600">W</text>
<text x="115" y="156" fill="#6f6d68" font-size="10" text-anchor="middle">d × d frozen</text>
<text x="115" y="40" fill="#e8e6e1" font-size="11" text-anchor="middle">16.7M params, frozen</text>
<line x1="40" y1="60" x2="190" y2="210" stroke="#2a2d35" stroke-width="0.8"/>
<line x1="190" y1="60" x2="40" y2="210" stroke="#2a2d35" stroke-width="0.8"/>
<text x="225" y="140" fill="#d4ff5e" font-size="22" text-anchor="middle">+</text>
<rect x="270" y="60" width="34" height="150" rx="5" fill="#16181d" stroke="#5ec8ff" stroke-width="1.5"/>
<text x="287" y="140" fill="#5ec8ff" font-size="13" text-anchor="middle" font-weight="600">B</text>
<text x="287" y="232" fill="#6f6d68" font-size="9" text-anchor="middle">d×r</text>
<text x="322" y="140" fill="#a4a29c" font-size="16" text-anchor="middle">×</text>
<rect x="340" y="120" width="150" height="34" rx="5" fill="#16181d" stroke="#5ec8ff" stroke-width="1.5"/>
<text x="415" y="142" fill="#5ec8ff" font-size="13" text-anchor="middle" font-weight="600">A</text>
<text x="415" y="176" fill="#6f6d68" font-size="9" text-anchor="middle">r×d</text>
<text x="525" y="140" fill="#d4ff5e" font-size="22" text-anchor="middle">→</text>
<rect x="570" y="60" width="150" height="150" rx="6" fill="#16181d" stroke="#d4ff5e" stroke-width="1.5"/>
<text x="645" y="132" fill="#d4ff5e" font-size="12" text-anchor="middle" font-weight="600">BA = ΔW</text>
<text x="645" y="150" fill="#6f6d68" font-size="9.5" text-anchor="middle">d×d, rank ≤ r</text>
<text x="645" y="232" fill="#e8e6e1" font-size="11" text-anchor="middle">65K params, trained</text>
</svg>
<figcaption>BA reconstructs a full d×d shape but is constrained to rank r. Dense, not sparse, but living in a tiny subspace.</figcaption>
</figure>

## §3 The parameter saving

| Quantity | Count | Example (d=4096, r=8) |
|---|---|---|
| Full update ΔW | \(d^2\) | 16,777,216 params |
| LoRA (A + B) | \(2dr\) | 65,536 params |
| Reduction | \(1 - 2r/d\) | **99.6% fewer** |

In general, parameter reduction is \(1 - 2r/d\). At d = 4096 and r = 8 that is 99.6%. At r = 16 it is 99.2%. Even at r = 64 you are saving over 96%.

## §4 The forward pass with `α/r` scaling

LoRA uses a scaling factor `α`:

\[ h = Wx + \frac{\alpha}{r} \cdot BAx \]

Why? When you change `r`, the magnitude of `BA` changes. Scaling by `α/r` makes the effective learning rate insensitive to `r`. You can tune α once and sweep r. Typical: `α = 2r` (so effective scale = 2). See [α/r scaling discussion in QLoRA Assembled](qlora-assembled.md) for the rank-stabilized variant.

## §5 Merging — the free-inference trick

After training, fold the adapter into the base weight. There is then a **single matrix** and inference is identical in speed to the original model:

\[ W_{\text{new}} = W + (\alpha/r) \cdot BA \]

Now `W_new` is a single matrix. Zero additional inference compute vs the base model.

!!! abstract "Why ever keep it unmerged?"
    Two reasons. **Multi-task serving**: 10 domain adapters on one base model = 1 base + 10 tiny adapters, hot-swapped per request, instead of 10 full model copies. vLLM, LoRAX, and SGLang all support this natively. **Continued training**: once merged, you cannot disentangle the adapter from the base weights. If you need to update the adapter with new data later, you would be fine-tuning `W_merged`, which is no longer LoRA-compatible in the original sense.

## §6 LoRA from scratch in 25 lines

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False        # freeze original weights

        d_out, d_in = base_layer.weight.shape
        self.r = r
        self.scale = alpha / r                         # the α/r scaling factor

        # LoRA matrices: random A, zero B
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        # Original frozen path + LoRA path, added.
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self):
        """Fold adapter into base weight for zero-overhead inference."""
        with torch.no_grad():
            self.base.weight.data += self.scale * (self.B @ self.A)
        self.base.weight.requires_grad = False
        # After merge: just use self.base(x) directly.
```

That is the whole soul of LoRA. Everything else in the section explains how to pick the hyperparameters and stack quantization on top.

## §7 Typical hyperparameters

| Knob | Typical | Effect |
|---|---|---|
| `r` (rank) | 8 to 64 | Capacity of the adapter. Higher = more expressive, more params. |
| `α` (alpha) | 16 to 32 (often 2r) | Strength of the adapter contribution via α/r scaling. |
| target modules | Q, V minimum, expand to all attention + FFN | Which linear layers get adapters. |
| learning rate | 1e-4 to 3e-4 | Higher than full FT, because only a tiny subset of params updates. |

For deeper picks see [Target modules](target-modules.md). For the production stack with PEFT and bitsandbytes, see [QLoRA assembled](qlora-assembled.md).

## §8 The PEFT library version

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-hf")

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 3,407,872 || all params: 6,742,609,920 || trainable%: 0.0506
```

## Interview Questions

**Q1: Explain LoRA in two sentences.**

Weight updates during fine-tuning have low intrinsic rank, so instead of updating W directly you learn `ΔW = BA` where B and A are thin matrices with rank r ≪ d. Forward pass becomes `h = Wx + (α/r) BA x`; after training you merge for zero inference overhead, and parameters drop from d² to 2dr — at d = 4096, r = 8 that is a 99.6% reduction.

**Q2: Is BA sparse?**

No. BA is fully dense — every one of the d² entries is nonzero. It is **low-rank**, not sparse. Rank counts linearly-independent directions, not nonzeros. ΔW touches every weight but is constrained to a rank-r subspace of all possible changes.

**Q3: When would you NOT use LoRA and just do full fine-tuning?**

When the task requires the model to learn genuinely new capabilities, not redirection of existing ones (a new domain with totally different structure). Or when you have the compute budget and need maximum accuracy: full FT consistently beats LoRA by 1 to 3% on most benchmarks. LoRA is a nudge; it shines when the base already "understands" the domain.

**Q4 (Trap): If merging is free, why ever keep adapters unmerged?**

Two reasons. Multi-task serving: 10 adapters on one base model = 1 base + 10 tiny adapters hot-swapped per request, vs 10 full model copies. And continued training: once merged you cannot disentangle the adapter from W to keep training it as LoRA. Merging is a one-way door.

**Q5: What is the parameter count reduction for a given d and r?**

For a single `d × d` weight matrix, going from full FT to LoRA drops params from `d²` to `2dr`, a factor of `d / (2r)`. At d = 4096, r = 8 that is 256× smaller (99.6% reduction). The same scaling holds across the model — LoRA on all linear layers typically lands at 0.05% to 0.5% of total params trainable.
