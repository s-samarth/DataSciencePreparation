# PEFT: LoRA and QLoRA

Parameter-Efficient Fine-Tuning is the standard 2026 stack for adapting large LLMs to a specific domain with limited data and compute. LoRA gives you ~0.1% of trainable parameters; QLoRA combines LoRA with 4-bit NF4 quantization to put a 70B model on a single 48GB GPU; paged optimizers, gradient accumulation, and gradient checkpointing handle the remaining memory pressure.

!!! tip "Rapid Recall"
    Fine-tuning a huge model fully is wasteful because the useful weight change ΔW is empirically low-rank — it lives in a tiny subspace. LoRA exploits this by learning `ΔW = BA` with thin matrices, training ~0.1% of params and merging at the end for zero inference cost. QLoRA stacks **4-bit NF4 quantization** of the frozen base on top, and **paged optimizers + gradient accumulation + gradient checkpointing** handle the remaining memory pressure so a 70B model fine-tunes on a single 48GB GPU.

## How the pieces connect

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 900 360" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<defs>
<marker id="peftarr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
<path d="M0,0 L9,4.5 L0,9 Z" fill="#6f6d68"/>
</marker>
</defs>
<rect x="350" y="150" width="200" height="60" rx="12" fill="#16181d" stroke="#d4ff5e" stroke-width="1.5"/>
<text x="450" y="176" fill="#d4ff5e" font-size="15" text-anchor="middle" font-weight="600">PEFT goal</text>
<text x="450" y="195" fill="#a4a29c" font-size="10" text-anchor="middle">adapt big model, tiny cost</text>
<rect x="330" y="20" width="240" height="52" rx="10" fill="#1c1f26" stroke="#c98bff" stroke-width="1"/>
<text x="450" y="42" fill="#c98bff" font-size="12" text-anchor="middle" font-weight="600">ΔW is low-rank</text>
<text x="450" y="59" fill="#a4a29c" font-size="9.5" text-anchor="middle">the assumption that makes it valid</text>
<line x1="450" y1="72" x2="450" y2="148" stroke="#6f6d68" stroke-width="1.3" marker-end="url(#peftarr)"/>
<rect x="40" y="120" width="220" height="52" rx="10" fill="#1c1f26" stroke="#5ec8ff" stroke-width="1"/>
<text x="150" y="142" fill="#5ec8ff" font-size="12" text-anchor="middle" font-weight="600">LoRA = BA</text>
<text x="150" y="159" fill="#a4a29c" font-size="9.5" text-anchor="middle">init · scaling · target layers</text>
<line x1="260" y1="160" x2="348" y2="172" stroke="#6f6d68" stroke-width="1.3" marker-end="url(#peftarr)"/>
<rect x="640" y="120" width="220" height="52" rx="10" fill="#1c1f26" stroke="#ff8b5e" stroke-width="1"/>
<text x="750" y="142" fill="#ff8b5e" font-size="12" text-anchor="middle" font-weight="600">QLoRA</text>
<text x="750" y="159" fill="#a4a29c" font-size="9.5" text-anchor="middle">+ 4-bit NF4 frozen base</text>
<line x1="640" y1="160" x2="552" y2="172" stroke="#6f6d68" stroke-width="1.3" marker-end="url(#peftarr)"/>
<rect x="120" y="285" width="200" height="52" rx="10" fill="#1c1f26" stroke="#6ee7a0" stroke-width="1"/>
<text x="220" y="307" fill="#6ee7a0" font-size="11.5" text-anchor="middle" font-weight="600">Paged optimizer</text>
<text x="220" y="324" fill="#a4a29c" font-size="9.5" text-anchor="middle">optimizer state ↔ CPU</text>
<rect x="350" y="285" width="200" height="52" rx="10" fill="#1c1f26" stroke="#6ee7a0" stroke-width="1"/>
<text x="450" y="307" fill="#6ee7a0" font-size="11.5" text-anchor="middle" font-weight="600">Grad accumulation</text>
<text x="450" y="324" fill="#a4a29c" font-size="9.5" text-anchor="middle">big batch in pieces</text>
<rect x="580" y="285" width="200" height="52" rx="10" fill="#1c1f26" stroke="#6ee7a0" stroke-width="1"/>
<text x="680" y="307" fill="#6ee7a0" font-size="11.5" text-anchor="middle" font-weight="600">Grad checkpoint</text>
<text x="680" y="324" fill="#a4a29c" font-size="9.5" text-anchor="middle">recompute activations</text>
<line x1="400" y1="210" x2="300" y2="283" stroke="#6f6d68" stroke-width="1.1" marker-end="url(#peftarr)"/>
<line x1="450" y1="210" x2="450" y2="283" stroke="#6f6d68" stroke-width="1.1" marker-end="url(#peftarr)"/>
<line x1="500" y1="210" x2="600" y2="283" stroke="#6f6d68" stroke-width="1.1" marker-end="url(#peftarr)"/>
</svg>
<figcaption>The dependency graph: low-rank assumption underpins LoRA; QLoRA adds 4-bit base; three memory tricks make a 70B model fit on one GPU.</figcaption>
</figure>

## Pages in this section

- **[LoRA mechanics](lora-mechanics.md)** — the `ΔW = BA` decomposition, the forward pass, the parameter saving, and the merge trick.
- **[Why low-rank](why-low-rank.md)** — the two papers that prove ΔW is low-rank, plus the chef analogy that makes it intuitive.
- **[Initialization and gradient flow](initialization-gradient-flow.md)** — why B = 0 and A = random, and why everyone gets the "no gradient flows" reasoning wrong.
- **[Target modules](target-modules.md)** — the six attachable matrices (Q, K, V, O, W_up, W_gate, W_down) and the decision dial for which to use.
- **[Quantization and NF4](quantization-nf4.md)** — number formats, why uniform INT4 fails for NN weights, how NF4 wins, double quantization, BF16 vs FP16.
- **[QLoRA assembled](qlora-assembled.md)** — the full memory budget for 70B on 48GB and the production code.
- **[Memory tricks](memory-tricks.md)** — paged optimizers, gradient accumulation, gradient checkpointing.

For the runnable LoRA SFT walkthrough on TinyLlama-1.1B, see [SFT Walkthrough Part B](../build-from-scratch/sft-walkthrough.md).
