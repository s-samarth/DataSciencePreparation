# KV-Cache

Without a cache, generating token 2001 would re-run attention over tokens 1 to 2000 to recompute their K and V. Decode would scale **quadratically** with sequence length. The fix rests on one observation: in causal attention, past K and V never change. Compute them once, cache them, attend the new query against the cache. This page is the memory formula every inference engineer should be able to write from memory.

!!! tip "Rapid Recall"
    The KV-cache formula: `KV cache = 2 × L × H_kv × d_head × N × B × bytes`. Walk down the transformer, multiply every dimension you meet: **2** (K and V) · **L**ayers · **H**_kv heads (post-GQA count, not query head count) · **D**im per head · **N** tokens · **B**atch · **bytes** (2 = FP16, 1 = FP8, 0.5 = INT4). LLaMA-3 70B at 32K context, batch 1, FP16: ~10.5 GB. The KV-cache scales **linearly with N and B** — the two knobs that blow up serving memory. At long context the KV load can dominate the weight load, which is the whole reason GQA, MLA, KV-quant, and PagedAttention are load-bearing, not optional.

## §1 Why the cache exists

In causal attention, K and V for tokens 1 to N never change once computed: token t cannot attend to t+1, so when we extend the sequence by one token we only need K and V for the new token. The K and V of all previous tokens are reusable.

Without caching, generating token N+1 re-runs the full attention computation over tokens 1 to N from scratch, re-projecting each token to K and V. That is O(N²) work per generated token and O(N³) for a full sequence. With caching, it is O(N) per generated token and O(N²) total.

## §2 The one formula — "2 LHDN B-bytes"

\[ \text{KV cache} = 2 \times L \times H_{kv} \times d_{head} \times N \times B \times \text{bytes} \]

Walk down the transformer, multiply every dimension you meet:

- **2** — K and V both cached.
- **L** — layers.
- **H_kv** — number of KV heads (post-GQA count, NOT the query head count).
- **d_head** — dimension per head.
- **N** — tokens (current sequence length).
- **B** — batch (concurrent sequences).
- **bytes** — 2 for FP16/BF16, 1 for FP8/INT8, 0.5 for INT4.

## §3 Worked example — LLaMA-3 70B at 32K, batch 1, FP16

```
2 × 80 layers × 8 kv_heads × 128 d_head × 32000 × 1 × 2 bytes  ≈ 10.5 GB

Shortcut: collapse H_kv × d_head into d_model_kv (= 8 × 128 = 1024)
   2 × 80 × 1024 × 32K × 2 bytes  ≈ 10.5 GB

Per-token-per-request constant ≈ 328 KB/token
```

That 10.5 GB is **per sequence**. Serving a batch of 16: 168 GB of KV-cache alone, before activations or model weights. This is why long-context serving is hard.

For comparison, the same model with MHA (64 query heads, no GQA) would have 64 KV heads, multiplying the cache by 8× to 84 GB per sequence at 32K — infeasible. **This is why GQA exists.** See [GQA, MQA, MLA](gqa-mqa-mla.md).

## §4 Storage and dimensionality

Per layer: a K tensor and a V tensor, each shaped `[B, H_kv, N, d_head]`. PagedAttention (vLLM) chops the N dimension into 16-token blocks stored non-contiguously — the per-token shape is identical, only the memory layout is paged. See [vLLM](../serving/vllm.md).

## §5 Why it is a bandwidth problem, not just capacity

!!! abstract "KV bandwidth at long context"
    That 10.5 GB is streamed from HBM **every decode step**, on top of the weights. It scales **linearly with N and B** — the two knobs that blow up serving memory. At long context the KV load can dominate the weight load. For LLaMA-70B at 128K context, KV cache is 42 GB per sequence vs 140 GB of FP16 weights; at batch 4, the cache is bigger than the model. This is the whole reason GQA / MLA / KV-quant / PagedAttention are load-bearing, not optional.

## §6 The KV-cache inference loop sketch

```python
kv_cache = {}   # {layer_idx: (K_cache, V_cache)}

def generate_step(x, kv_cache, layer_idx):
    """x is a single new token: (B, 1, d)"""
    q, k_new, v_new = project_qkv(x)

    # Append new K, V to cache.
    if layer_idx in kv_cache:
        K_full = torch.cat([kv_cache[layer_idx][0], k_new], dim=1)
        V_full = torch.cat([kv_cache[layer_idx][1], v_new], dim=1)
    else:
        K_full, V_full = k_new, v_new
    kv_cache[layer_idx] = (K_full, V_full)

    # Attention: q attends to all cached K, V.
    # No causal mask needed — q is the last token, all cached K/V are valid past.
    out = F.scaled_dot_product_attention(q, K_full, V_full, is_causal=False)
    return out
```

Real production servers use PagedAttention (vLLM) rather than a contiguous Python dict; the cache lives in 16-token blocks managed by a page table for fragmentation-free allocation.

## §7 The ways out

You cannot avoid the KV cache (without recomputing, which is worse), but you can make it smaller.

- **Use fewer KV heads (GQA, MQA, MLA).** See [GQA, MQA, MLA](gqa-mqa-mla.md).
- **Quantize KV.** INT8 KV cache halves traffic; INT4 quarters it with manageable quality loss. KVQuant is a notable method.
- **Page the cache (PagedAttention).** Does not shrink the cache but eliminates the 60-80% fragmentation waste of static allocation. See [vLLM](../serving/vllm.md).
- **Compress via latent (MLA).** DeepSeek V2/V3's approach: cache a low-rank latent per token and up-project to K/V on the fly. ~90% reduction vs MHA.
- **Sliding window attention.** Drop tokens beyond a fixed window. Works for some tasks; breaks for others.

## Interview Questions

**Q1: Estimate the KV-cache memory for LLaMA-3 70B at 32K context.**

LLaMA-3 70B: 80 layers, 8 KV heads (GQA), head dim 128, BF16 = 2 bytes. Formula: `2 × L × H_kv × d_k × N × bytes`. Compute: `2 × 80 × 8 × 128 × 32768 × 2 = 10.7 GB` per sequence. That is per sequence — if you are serving a batch of 16, KV-cache alone is ~170 GB, which is why you need tensor parallelism or KV quantization at production scale. With MHA instead of GQA (64 heads), this would have been 86 GB per sequence — infeasible.

**Q2: Why is the KV cache a bandwidth problem and not just a capacity problem?**

That cache has to be streamed from HBM every single decode step, on top of the weights. As context grows, the per-token KV load eventually catches up to and exceeds the weight load. At LLaMA-70B 128K context, the cache is 42 GB per sequence vs 140 GB of weights — at batch 4 the cache dominates. Bandwidth, not just VRAM, is what gets exhausted.

**Q3: What is the formula and how do you derive it?**

`KV = 2 × L × H_kv × d_head × N × B × bytes`. Each layer has its own K and V (so 2 × L), each layer has H_kv heads (which is post-GQA, *not* the query head count), each head has d_head dimensions, you have N tokens cached, batch B concurrent sequences, and `bytes` per element. The trap is using H (query heads) instead of H_kv — every modern model uses GQA where H_kv < H, and that difference is the whole memory benefit.

**Q4: How does PagedAttention change the KV-cache formula?**

It does not change the formula; it changes the memory layout. The total bytes are the same. PagedAttention allocates the cache in 16-token blocks linked via a page table, eliminating the 60 to 80% fragmentation waste of static contiguous allocation. The net effect is you can serve 2 to 4× more concurrent sequences in the same VRAM, but the per-sequence per-token cache cost is unchanged.

**Q5: Why does the KV-cache formula use H_kv and not the query head count H?**

Because with GQA, multiple query heads share one KV head. The cache only stores K and V, not Q (which is recomputed each step), so the cache size scales with H_kv. The query heads are computed fresh at each decode step from the residual stream and do not need caching. Using H instead of H_kv overestimates the cache by a factor of H / H_kv (typically 8× for LLaMA-3 70B).
