# Attention as Soft Retrieval

Attention is the operation that turned sequence modeling from "process one token at a time" into "let every token directly talk to every other token in one matmul." It is the soft, differentiable, learned version of a dictionary lookup, and every other transformer trick exists to make this operation stable, efficient, or position-aware.

!!! tip "Rapid Recall"
    Attention is a soft dictionary lookup. Each token produces three vectors: a query (what am I looking for?), a key (how do I advertise myself?), and a value (what info do I hand over?). Compute similarity of query against every key, divide by sqrt(d_k) so the dot products do not explode, softmax to weights, then take a weighted average of values. That is `softmax(QKᵀ/√d_k)·V`. The sqrt(d_k) scaling is the most common trap question.

## §1 The Python-dict analogy

A Python dict does `d[key] = value`: exact match, one result. Attention relaxes every part of that. "Match" becomes a similarity score, "exact" becomes a softmax-weighted blend over all entries, and the whole thing is differentiable so gradients flow.

```python
# Hard lookup: exact key match, one value
result = d[query]

# Soft lookup (attention): similarity match, weighted blend
result = sum(similarity(query, k_i) * v_i for k_i, v_i in zip(keys, values))
```

## §2 Why three separate projections

Each token's embedding is projected into three subspaces, each optimized for a different job. The cleanest analogy is a two-tower retrieval system.

- **Query (Q)** = `X @ W_Q`. What this token is searching for, like a user-query embedding.
- **Key (K)** = `X @ W_K`. What this token can be matched against, like an item embedding in a retrieval index.
- **Value (V)** = `X @ W_V`. What this token returns if matched, the payload.

The features useful for *being found* are not the same as features useful for *being returned*. Same reason two-tower recommenders keep user and item towers separate. They cannot collapse into one matrix because (a) you would lose capacity, (b) `X·Xᵀ` is symmetric but real attention is *asymmetric* (A attending to B is not the same as B attending to A), and (c) you want to decouple "what I match on" from "what I return."

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace" font-size="13">
<rect x="20" y="100" width="80" height="48" rx="4" fill="#ebe0cf" stroke="#1a1410" stroke-width="1.5"/>
<text x="60" y="120" text-anchor="middle" font-weight="700">X</text>
<text x="60" y="138" text-anchor="middle" font-size="10">(N, d)</text>
<rect x="180" y="30" width="90" height="42" rx="4" fill="#b5462a" stroke="#1a1410" stroke-width="1.5"/>
<text x="225" y="55" text-anchor="middle" fill="#f4ede2" font-weight="700">Q = X·Wq</text>
<rect x="180" y="103" width="90" height="42" rx="4" fill="#1f5e5b" stroke="#1a1410" stroke-width="1.5"/>
<text x="225" y="128" text-anchor="middle" fill="#f4ede2" font-weight="700">K = X·Wk</text>
<rect x="180" y="176" width="90" height="42" rx="4" fill="#c08a2d" stroke="#1a1410" stroke-width="1.5"/>
<text x="225" y="201" text-anchor="middle" fill="#1a1410" font-weight="700">V = X·Wv</text>
<line x1="100" y1="118" x2="180" y2="51" stroke="#1a1410" stroke-width="1.3"/>
<line x1="100" y1="124" x2="180" y2="124" stroke="#1a1410" stroke-width="1.3"/>
<line x1="100" y1="130" x2="180" y2="197" stroke="#1a1410" stroke-width="1.3"/>
<rect x="340" y="60" width="120" height="46" rx="4" fill="#fbf6ee" stroke="#b5462a" stroke-width="1.5"/>
<text x="400" y="80" text-anchor="middle" font-weight="700" font-size="11">scores =</text>
<text x="400" y="96" text-anchor="middle" font-size="11">Q·Kᵀ / √dk</text>
<line x1="270" y1="51" x2="340" y2="78" stroke="#b5462a" stroke-width="1.3"/>
<line x1="270" y1="124" x2="340" y2="90" stroke="#1f5e5b" stroke-width="1.3"/>
<rect x="340" y="125" width="120" height="40" rx="4" fill="#fbf6ee" stroke="#1a1410" stroke-width="1.5"/>
<text x="400" y="150" text-anchor="middle" font-weight="700" font-size="11">softmax(scores)</text>
<line x1="400" y1="106" x2="400" y2="125" stroke="#1a1410" stroke-width="1.3"/>
<rect x="540" y="105" width="150" height="46" rx="4" fill="#1a1410" stroke="#1a1410" stroke-width="1.5"/>
<text x="615" y="125" text-anchor="middle" fill="#f4ede2" font-weight="700" font-size="11">output =</text>
<text x="615" y="141" text-anchor="middle" fill="#f4ede2" font-size="11">weights · V</text>
<line x1="460" y1="145" x2="540" y2="130" stroke="#1a1410" stroke-width="1.3"/>
<line x1="270" y1="197" x2="540" y2="140" stroke="#c08a2d" stroke-width="1.3" stroke-dasharray="4 3"/>
</svg>
<figcaption>The attention operation, end to end. X is projected into Q, K, V; scores come from Q·Kᵀ, softmax produces weights, and the output is the weighted blend of values.</figcaption>
</figure>

!!! note "Intuition"
    If you have ever computed `cosine_similarity(query_emb, doc_embs)` then taken a top-k, attention is exactly that, except it is a *soft* top-k (softmax), the "documents" are other tokens in the same sequence, and Q, K, V are learned projections rather than fixed embeddings. The N×N attention matrix is literally a learned pairwise similarity matrix over the sequence.

## §3 The math, with the sqrt(d_k) derivation

For each query vector q, all key vectors K, and all value vectors V:

\[ \text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{QK^T}{\sqrt{d_k}} \right) V \]

Plain English: compute similarity of each query with each key, scale it down so gradients do not explode, normalize to probabilities, use as weights to average the values.

**Why divide by sqrt(d_k)?** If q and k are independent vectors with mean 0 and variance 1, each of dimension d_k, then q·k has mean 0 and variance d_k. As d_k grows, the dot products grow in magnitude, softmax saturates toward one-hot, and gradients through softmax vanish. Dividing by sqrt(d_k) normalizes variance back to 1 and keeps softmax in a healthy regime. Without it, training becomes unstable at d_k > 64. This is a classic interview trap.

## §4 The causal mask

For autoregressive (decoder-only) models, before softmax we add a mask M where `M[i,j] = -inf if j > i else 0`. After softmax, future positions have weight 0. Token i can only attend to tokens 1..i. This is what makes a decoder-only model autoregressive: there is no peeking at the future.

## §5 The minimum-viable implementation

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: (batch, heads, seq_len, d_k)
    d_k = Q.size(-1)

    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (B, H, N, N)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn  # (B, H, N, d_k)
```

That is the whole soul of a transformer. Everything else is normalization, residuals, FFN, and positional encodings wrapped around this.

## Interview Questions

**Q1: Why do we divide attention scores by sqrt(d_k)?**

Because dot products of two independent d_k-dim vectors have variance d_k. As d_k grows, scores grow, softmax saturates toward one-hot, and gradients through softmax vanish. Dividing by sqrt(d_k) normalizes the variance back to 1 and keeps softmax in a healthy regime. Without it, training becomes unstable at d_k > 64.

**Q2 (Trap): If I set d_k to 1, can I skip the sqrt(d_k) scaling?**

Yes, mathematically the scaling factor is 1 so it is a no-op. But d_k = 1 is useless: you have collapsed the key space to a scalar and lost all expressive power. The question tests whether you understand the scaling is a function of dimension, not a magic constant. Related trap: some people think the scaling factor is d_k (not sqrt). It is sqrt because the variance of a sum of d_k products is d_k, so the std is sqrt(d_k).

**Q3: Why three separate projections instead of using the raw embeddings as Q, K, V?**

Two reasons. First, you would lose capacity, since `X·Xᵀ` is symmetric and real attention is asymmetric (A attending to B is not the same as B attending to A). Second, you want to decouple "what I match on" from "what I return"; the features useful for being found are not the same as the features useful for being delivered.

**Q4: How does the causal mask work and where is it applied?**

You add `-inf` to the upper triangle of the score matrix before softmax. After softmax, those positions have weight exactly 0, so position i can only attend to positions 1..i. In PyTorch's SDPA you just pass `is_causal=True` and the kernel handles it efficiently without materializing the mask tensor.
