# Positional Encodings: Sinusoidal → RoPE → ALiBi

Attention is permutation-invariant: "cat sat mat" and "mat sat cat" look identical. Without position info, you have a bag of words. The field tried four generations of fixes; the winner in 2026 is RoPE because relative position falls out of the dot product for free, and the clean math lets you extend context after training (PI, NTK-aware, YaRN, LongRoPE).

!!! tip "Rapid Recall"
    Attention is permutation-invariant; you must inject order. Sinusoidal (2017) adds a fixed sine/cosine fingerprint to embeddings; cheap, parameter-free, but position info dilutes through the stack. **RoPE (2021)** *rotates* Q and K by an angle proportional to position; the dot product `q_m · k_n` then depends only on `m − n`, so relative position emerges naturally. RoPE is applied at every layer, has zero parameters, and extrapolates well with NTK/YaRN scaling tricks. ALiBi adds a linear distance penalty to scores; clean idea but caused head collapse and lost adoption in 2024.

## §1 Why positional encoding exists at all

Attention is permutation-invariant. Without position info, "dog bites man" and "man bites dog" look identical to the model. You must inject order somehow.

| Method | Where applied | Params | Extrapolation | 2026 status |
|---|---|---|---|---|
| **Sinusoidal** | Added to embeddings (input) | 0 | Poor (spikes past train len) | Original Transformer, T5 |
| **Learned absolute** | Added to embeddings | O(max_len · d) | None | BERT, ViT |
| **Relative (T5 bias)** | Bias inside attention | Small | Medium | T5 family |
| **RoPE** | Rotates Q, K in attention | 0 | Good (with NTK/YaRN) | **LLaMA, DeepSeek, Mistral, Qwen** |
| **ALiBi** | Linear distance penalty on scores | 0 | Excellent | BLOOM, MPT (lost adoption) |

## §2 Sinusoidal — the 2017 original

Add a fixed sinusoidal pattern to token embeddings. Each position gets a unique sine/cosine "fingerprint." Pros: simple, parameter-free, can extrapolate slightly. Cons: position info gets diluted through layers, attention scores do not directly encode relative distance.

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

```python
class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal PE added to token embeddings. Vaswani 2017."""
    def __init__(self, n_embd, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * -(math.log(10000.0) / n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)
```

## §3 RoPE — Rotary Position Embedding

Instead of adding to embeddings, **rotate** the query and key vectors by an angle proportional to position. The dot product `q · k` then naturally encodes relative position.

Key insight: the inner product of rotated vectors `R(m)·q` and `R(n)·k` depends only on `m − n`, so the model sees relative offsets. You encode absolute, you get relative.

```python
def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device='cuda'):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)   # [T, head_dim/2]
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """x: [B, n_head, T, head_dim], cos/sin: [T, head_dim/2]. Rotate each (x_even, x_odd) pair by theta_pos."""
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos = cos[:x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(-2)].unsqueeze(0).unsqueeze(0)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd  = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
```

In interviews: RoPE is the answer for "what positional encoding would you use today?" It is the dominant choice in 2024-2026 open LLMs.

### Why RoPE won

- **Does not pollute embeddings.** Position lives in attention geometry, not added to token values.
- **Applied every layer.** Positional info does not decay through the stack.
- **Relative position emerges for free.** Rotate Q by θ_m and K by θ_n; the dot product depends on `cos(θ_m − θ_n)`, i.e., on `m − n`.
- **Clean math enables context extension.** Position Interpolation, NTK-aware scaling, YaRN, and LongRoPE all manipulate RoPE's frequencies post-training to stretch 4K to 128K. See [Frontier Techniques](../inference-arch/frontier-techniques.md) for the NTK-aware derivation.

!!! warning "Trap question"
    "RoPE only rotates Q and K, not V — isn't position lost in the output?" No. Position is needed to *decide what to attend to* (computed from Q·K). The delivered content (V) should not be position-tagged; putting position in V would pollute the value content. Attention weights have already baked in position when they were computed.

## §4 ALiBi — clean idea that lost adoption

Skip positional embeddings entirely. Just add a linear penalty to attention scores based on distance: `−m · |i − j|` where `m` is a per-head slope. Simple, extrapolates the best on paper. But it causes 30 to 44% of heads to collapse onto BOS in deep models, and RoPE plus YaRN ecosystems erased its advantage. Dead since 2024.

## Interview Questions

**Q1: Why RoPE over sinusoidal positional encoding?**

Three reasons. First, RoPE encodes relative position naturally: the dot product of rotated q and k depends only on `m − n`, so the model reasons about distance, not absolute index. Second, it is applied inside attention at every layer, not just added at the bottom, so positional info does not decay through the stack. Third, it extrapolates better beyond training length, especially with scaling tricks like NTK and YaRN. Sinusoidal perplexity spikes hard past the training window; RoPE degrades gracefully.

**Q2 (Trap): RoPE only rotates Q and K, not V — is position lost in the output?**

No. Position is needed to *decide what to attend to*, which is the Q·K computation. The delivered content (V) should not be position-tagged; baking position into V would pollute the value content. Attention weights have already absorbed position when they were computed.

**Q3: Why did ALiBi lose to RoPE?**

ALiBi adds a linear distance penalty to attention scores per head. Clean and extrapolates well in principle, but in practice it caused a meaningful fraction of heads to collapse onto the BOS token in deep models, and the RoPE plus YaRN/NTK ecosystem closed the extrapolation gap with cleaner math. ALiBi was largely abandoned after 2024.

**Q4: What is the relationship between RoPE base, head_dim, and context length?**

RoPE uses many frequencies across the head dim: `freq_i = base^(−2i/d_head)`. Low dimensions get high frequencies (short wavelengths, good for local patterns); high dimensions get low frequencies (long wavelengths, good for global). To extend a 4K-trained model to 32K without retraining, NTK-aware scaling raises the base (which stretches low frequencies a lot and high frequencies barely), preserving local syntax while making room for longer phrases. PI just uniformly compresses positions and degrades local detail.
