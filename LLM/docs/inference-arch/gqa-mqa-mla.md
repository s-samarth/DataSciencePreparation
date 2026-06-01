# GQA, MQA, MLA — Shrinking the KV-Cache

The 2017 transformer had as many K and V heads as Q heads (Multi-Head Attention). At long context that KV cache became the inference bottleneck. The fix is to keep many query heads but fewer KV heads: MQA collapses to one, GQA groups them (4 or 8), MLA compresses to a low-rank latent. Every modern model picks somewhere on this spectrum.

!!! tip "Rapid Recall"
    **MHA**: one K and V head per query head (e.g., 32 of each); huge cache. **MQA**: 1 shared K and V head across all query heads; max reduction but noticeable quality drop and instability at scale. **GQA**: group query heads (e.g., 4 per group) and share K, V within groups; near-MHA quality at MQA-like cache savings; **the open-source default since Llama 2 70B**. **MLA** (DeepSeek V2/V3): cache a small compressed latent per token and up-project to K/V on the fly via matrix algebra that folds the up-projection into the Q projection; ~93% cache reduction vs MHA. The meta-pattern: decode-time optimizations almost always **trade compute for memory bandwidth** — compute is in surplus at decode, bandwidth is scarce.

## §1 The MHA → MQA → GQA spectrum

| Scheme | K/V heads | Cache | Quality |
|---|---|---|---|
| MHA | one per query head (e.g., 64) | huge | full |
| MQA | 1 shared | ÷64 | noticeable drop |
| GQA | grouped (e.g., 8 groups of 8) | ÷8 | ≈ MHA |

For a 70B Llama-style model with 64 query heads, GQA with 8 groups means 8 KV heads (each shared by 8 Q heads). The cache shrinks 8× vs MHA; quality drop is negligible in practice.

## §2 The asymmetric intuition

!!! note "Intuition"
    Q = "what am I looking for?" K, V = "what is in the library to be found and retrieved?" Sharing K, V across query heads says "different queries read the same library" — a small quality hit. Sharing Q would mean "different heads all ask the same question" — obviously bad. That asymmetry is why GQA is essentially free.

This is the simplest way to explain why GQA works: you only lose if you blur the *questions*, not if you blur the *answers*. Sharing KV across query heads compresses what is being retrieved, but each query head still asks its own thing.

## §3 The implementation

```python
class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads          # queries per kv head
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim)   # fewer
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim)   # fewer
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.wq(x).view(B, N, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, N, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.n_kv_heads, self.head_dim)

        # Repeat KV to match Q head count (conceptually; good kernels fuse this).
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.wo(out)
```

The W_K and W_V matrices are smaller than W_Q; this is the *only* architectural change vs MHA. Everything else (causal mask, SDPA, output projection) is identical.

## §4 The memory math

For H query heads, H_kv KV heads (where H_kv divides H, g = H / H_kv queries per group):

- **MHA** (H_kv = H): KV-cache is `H × d_k` per token per layer.
- **MQA** (H_kv = 1): KV-cache is `d_k` per token per layer. H× reduction.
- **GQA** (H_kv = H/g): KV-cache is `(H/g) × d_k`. g× reduction.

LLaMA 3 70B chooses g = 8: 8× smaller KV-cache than MHA equivalent, with negligible quality loss.

## §5 MLA — DeepSeek's bet

Instead of caching K and V directly, cache a small **compressed latent** per token and up-project it into K and V on the fly. Through matrix algebra the up-projection folds into the Q projection, so you effectively attend in the latent space. DeepSeek V3 reports ~93% cache reduction vs MHA.

The cost is complexity: you need separate rotary-position-embedded and non-RoPE parts of the K vector because you cannot easily apply RoPE to a compressed latent. If you can stomach the engineering, MLA wins on the Pareto frontier of quality vs memory.

## §6 The comparison table

| | MHA | GQA | MQA | MLA |
|---|---|---|---|---|
| KV heads | H | H/g | 1 | latent (≪ H·d_k) |
| KV-cache size | 1× | 1/g× | 1/H× | ~1/10× |
| Quality | Best | Near-best | Worst | Best |
| Training stability | Best | Good | Unstable at scale | Good |
| Used by | Old models, some specialty | LLaMA, Mistral, Gemma | Falcon, some inference-optimized | DeepSeek V2/V3 |

**Decision rule:** for a new dense model in 2026, use GQA with g = 4 to 8. For extreme inference efficiency at scale, consider MLA if you can afford the implementation complexity.

## §7 The meta-pattern (carry this everywhere)

!!! abstract "Compute for bandwidth"
    Decode-time optimizations almost always **trade compute for memory bandwidth** — compute is in surplus at decode, bandwidth is scarce. Flash Attention (recompute in SRAM vs store in HBM), MLA (decompress vs cache), speculative decoding (draft compute vs target loads) all run the same play. Whenever you see a new decode optimization, ask "what is it spending more of (compute) to save (bandwidth)?" The answer is almost always the same.

## Interview Questions

**Q1: How does GQA differ from MQA, and why did MQA lose?**

MQA shares a single K and V head across all query heads — maximum KV-cache savings (H× reduction). GQA groups query heads (e.g., 4 per group) and shares one K, V head per group, giving H/g× reduction. MQA is more aggressive but at scale it causes training instability and meaningful quality regression. GQA hits a sweet spot: near-MHA quality, near-MQA inference cost, and it is robust during training. Every major open-source LLM from LLaMA 2 70B onward uses GQA.

**Q2: Why is MLA replacing GQA in some frontier models?**

MLA compresses K and V into a low-rank latent vector before caching. Instead of storing H_kv × d_k values per token per layer, you store just the latent dim (much smaller). On decode, you reconstruct K and V from the latent — but the math folds the up-projection into the Q projection, so you effectively attend in latent space. DeepSeek V2's MLA achieves ~90% KV-cache reduction vs MHA, better than GQA's typical 8× reduction, while preserving quality better than MQA. The cost is added complexity: you need separate rotary-position-embedded and non-RoPE parts of the K vector because you cannot easily apply RoPE to a compressed latent. If you can stomach the engineering, MLA wins on the Pareto frontier of quality vs memory.

**Q3: Why is sharing K and V across query heads cheaper than sharing Q?**

The asymmetry. Q is "what am I looking for"; K and V are "the library being searched." Different query heads asking different things is what gives multi-head its power. Sharing the library across heads costs you a little (different queries get the same retrieved content) but is mostly fine. Sharing the questions across heads would collapse multi-head into single-head — disastrous.

**Q4: In the KV-cache formula `2 × L × H_kv × d_k × N × B × bytes`, why does GQA save memory but not MQA-style training stability?**

Cache savings come from H_kv shrinking; training stability is about how diverse the K, V representations need to be to support all the Q heads downstream. MQA collapses to H_kv = 1 which is too much sharing — many query heads get only one global retrieved-content channel, and training has trouble distributing useful information into that single channel. GQA with H_kv = 4 to 8 keeps enough diversity in K, V to support the query heads without forcing a single bottleneck.

**Q5: Decision rule for a new dense model in 2026?**

GQA with g = 4 to 8 (so H_kv = 4 to 8 if you have 32 query heads, 8 if you have 64). MLA if you have the engineering budget and need extreme inference efficiency. Avoid MHA for new models; it is wasteful. Avoid MQA at scale; it is unstable.
