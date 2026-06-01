# Target Modules: Q, K, V, O, and FFN

LoRA can attach to any linear layer in the transformer. The original paper's ablation said Q and V; modern recipes start at Q + K + V + O, and add FFN layers when the task demands new factual knowledge rather than just behavior change. This page maps the six attachable matrices and gives a decision dial for picking which ones.

!!! tip "Rapid Recall"
    A transformer block has six attachable linear matrices: **W_Q, W_K, W_V, W_O** in attention; **W_up, W_gate, W_down** in the SwiGLU FFN. The LoRA paper's ablation picked **Q + V** as the best perf-per-param. Modern default is **Q + K + V + O** for domain adaptation (medical, legal, code). Add **FFN layers** when the task needs new factual knowledge, since FFN is where knowledge is believed to live; W_up is d × 4d, much larger than the attention projections, so this raises adapter param count significantly. **Practical rule:** if you are underperforming with Q + K + V + O, add FFN before raising rank — it often helps more than doubling r.

## §1 The anatomy

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 860 420" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<defs>
<marker id="tarr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#6f6d68"/></marker>
</defs>
<text x="430" y="26" fill="#e8e6e1" font-size="13" text-anchor="middle" font-weight="600">One Transformer Block</text>
<rect x="370" y="44" width="120" height="30" rx="6" fill="#1c1f26" stroke="#6f6d68"/>
<text x="430" y="64" fill="#a4a29c" font-size="11" text-anchor="middle">input x</text>
<line x1="430" y1="74" x2="430" y2="92" stroke="#6f6d68" marker-end="url(#tarr)"/>
<rect x="120" y="96" width="620" height="130" rx="10" fill="rgba(94,200,255,0.04)" stroke="#5ec8ff" stroke-width="1.2"/>
<text x="135" y="116" fill="#5ec8ff" font-size="11" font-weight="600">MULTI-HEAD SELF-ATTENTION</text>
<rect x="150" y="128" width="80" height="34" rx="6" fill="#16181d" stroke="#d4ff5e"/><text x="190" y="150" fill="#d4ff5e" font-size="12" text-anchor="middle" font-weight="600">W_Q</text>
<rect x="250" y="128" width="80" height="34" rx="6" fill="#16181d" stroke="#5ec8ff"/><text x="290" y="150" fill="#5ec8ff" font-size="12" text-anchor="middle" font-weight="600">W_K</text>
<rect x="350" y="128" width="80" height="34" rx="6" fill="#16181d" stroke="#d4ff5e"/><text x="390" y="150" fill="#d4ff5e" font-size="12" text-anchor="middle" font-weight="600">W_V</text>
<text x="190" y="178" fill="#6f6d68" font-size="9" text-anchor="middle">what to seek</text>
<text x="290" y="178" fill="#6f6d68" font-size="9" text-anchor="middle">advertise</text>
<text x="390" y="178" fill="#6f6d68" font-size="9" text-anchor="middle">retrieve</text>
<rect x="460" y="128" width="120" height="34" rx="6" fill="#1c1f26" stroke="#6f6d68"/>
<text x="520" y="150" fill="#a4a29c" font-size="10" text-anchor="middle">softmax(QKᵀ/√d)V</text>
<rect x="610" y="128" width="100" height="34" rx="6" fill="#16181d" stroke="#ff8b5e"/><text x="660" y="150" fill="#ff8b5e" font-size="12" text-anchor="middle" font-weight="600">W_O</text>
<text x="660" y="178" fill="#6f6d68" font-size="9" text-anchor="middle">mix back</text>
<text x="430" y="208" fill="#a4a29c" font-size="9.5" text-anchor="middle">4 matrices: Q, K, V, O — each d×d</text>
<line x1="430" y1="226" x2="430" y2="248" stroke="#6f6d68" marker-end="url(#tarr)"/>
<rect x="220" y="252" width="420" height="120" rx="10" fill="rgba(255,139,94,0.04)" stroke="#ff8b5e" stroke-width="1.2"/>
<text x="235" y="272" fill="#ff8b5e" font-size="11" font-weight="600">FEED-FORWARD NETWORK</text>
<rect x="250" y="286" width="90" height="34" rx="6" fill="#16181d" stroke="#c98bff"/><text x="295" y="308" fill="#c98bff" font-size="11" text-anchor="middle" font-weight="600">W_up</text>
<rect x="360" y="286" width="90" height="34" rx="6" fill="#16181d" stroke="#c98bff"/><text x="405" y="308" fill="#c98bff" font-size="11" text-anchor="middle" font-weight="600">W_gate</text>
<rect x="470" y="286" width="90" height="34" rx="6" fill="#16181d" stroke="#c98bff"/><text x="515" y="308" fill="#c98bff" font-size="11" text-anchor="middle" font-weight="600">W_down</text>
<text x="430" y="345" fill="#6f6d68" font-size="9.5" text-anchor="middle">expand → activate → compress · where factual knowledge is believed to live</text>
<text x="430" y="360" fill="#6f6d68" font-size="9" text-anchor="middle">W_up is d×4d — LARGER than attention matrices</text>
<line x1="430" y1="372" x2="430" y2="392" stroke="#6f6d68" marker-end="url(#tarr)"/>
<rect x="370" y="394" width="120" height="24" rx="6" fill="#1c1f26" stroke="#6f6d68"/><text x="430" y="411" fill="#a4a29c" font-size="10" text-anchor="middle">output</text>
</svg>
<figcaption>The six attachable matrices. "O" = output projection, the 4th matrix inside attention — not a separate level.</figcaption>
</figure>

A transformer block has two sub-components, each containing linear layers. LoRA can attach to any of them.

- **Attention block:** W_Q (search), W_K (advertise), W_V (retrieve), W_O (mix back). All four are typically d × d. (Modern attention uses GQA where W_K and W_V are d × d_kv with d_kv < d, but the LoRA target list does not change.)
- **FFN block:** W_up, W_gate, W_down. In SwiGLU FFN, W_up is d × 4d (the "expansion"), W_gate is d × 4d (the gating projection), W_down is 4d × d (the projection back). **W_up alone is 4× the size of any attention matrix**, so adding FFN to your LoRA targets raises adapter param count significantly even at the same rank.

## §2 Why Q and V are the default

From the LoRA paper's ablations, Q + V gave the best performance per parameter:

- **W_Q** controls what each token *searches for*. Task adaptation often means learning new attention patterns. High leverage.
- **W_V** controls what information gets *retrieved*. Even with good attention patterns, wrong V = wrong output. High leverage.
- **W_K** is what tokens *advertise*. Adding it on top of Q + V gave diminishing returns in the original paper — if Q already searches better, K matters less. (Debated; many now include it.)
- **W_O** mixes multi-head output back. Less critical than Q + V originally, but most modern recipes include it.

## §3 The decision dial

| Task type | Targets | r / α | Why |
|---|---|---|---|
| Style / format / instruction | Q, V | 8 / 16 | Capable base + little data; minimize overfit. |
| Domain adaptation (medical, legal, code) | Q, K, V, O | 16 / 32 | Must shift attention + recombination. The common default. |
| New factual knowledge / new language | + FFN (up, down) | 32-64 / 64 | FFN stores knowledge — needed when learning new things, not just behaving differently. |
| Continued pretraining | Everything (+embed, LM head) | high | Heavy adaptation — consider full FT instead. |

!!! abstract "The non-obvious shortcut"
    FFN layers are **larger** than attention layers (W_up is d × 4d). So adding FFN raises your adapter param count a lot even at the same rank. The practical rule: if you are underperforming with Q + K + V + O, **add FFN before increasing rank** — it often helps more than doubling r.

## §4 Naming conventions across libraries

The module names vary across model families. PEFT lets you pass either a list or the `"all-linear"` keyword.

- **Llama / Mistral / Qwen:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **GPT-2 / GPT-J:** `c_attn` (fused QKV), `c_proj` (output), `c_fc` (FFN up), `c_proj` (FFN down).
- **PEFT shortcut:** `target_modules="all-linear"` attaches to every `nn.Linear` in the model, which is the modern best-practice default for serious training runs.

## §5 The PEFT config

```python
from peft import LoraConfig, TaskType

# Minimal default: Q and V only.
config_min = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Standard default: all attention projections.
config_std = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Heavy: attention + FFN (the all-linear pattern).
config_heavy = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)
```

## Interview Questions

**Q1: What is "O" and which layers should LoRA target?**

O is the output projection — the 4th linear matrix inside attention (after Q, K, V), which mixes multi-head output back into the residual stream. Default is Q + V (best perf-per-param in the LoRA ablations): Q controls what tokens search for, V controls what is retrieved. Add K + O for domain adaptation; add FFN (up, gate, down) when the task needs new factual knowledge, since FFN is where knowledge is believed to live.

**Q2: Underperforming with Q + K + V + O — raise rank or add FFN?**

Add FFN before raising rank — it often helps more. But note FFN matrices are larger (W_up is d × 4d), so this adds more adapter params than the same rank bump on attention. Spend the budget where the task needs it: behavior shift → attention; new knowledge → FFN.

**Q3: Why are Q and V the "high leverage" pair, not Q and K?**

Q and K together determine the attention pattern, so you might think Q + K would be the obvious pair. But Q already shifts attention patterns by itself if you let it learn new search directions; meanwhile V controls the *content* that gets pulled in — even with perfect attention, wrong V means wrong output. Q + V covers "where to look" plus "what to deliver," which empirically gives the cleanest perf-per-param.

**Q4: What is the difference between targeting individual modules and `"all-linear"`?**

Naming individual modules gives you precise control and slightly fewer trainable params (you skip the lm_head, embedding, and maybe O). `"all-linear"` is the modern best-practice default for serious training runs — it attaches to every `nn.Linear` and typically squeezes out 2 to 3% more performance over Q + V at the cost of more adapter parameters and slightly more memory. Use it when you have the VRAM and want max quality.
