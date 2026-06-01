# Multi-Head Attention and the Softmax Bottleneck

The textbook answer for "why multi-head?" is "different heads learn different patterns." Right, but incomplete. The deeper reason is that a single softmax can only express one ranking, and a token usually needs information from multiple unrelated sources at once. Splitting into heads gives each head a clean sharp signal in its own subspace.

!!! tip "Rapid Recall"
    Multi-head attention does NOT split the weight matrices. You have one `W_Q` of shape `(d_model, d_model)`, project to full `d_model`, then **reshape** the last dim into `(heads, head_dim)`. Each head computes an independent attention over its own 64-dim subspace; heads cannot mix until the final concat and `W_O`. The parameter count is identical to single-head. The real reason multi-head wins is that softmax is a sharpening function and one softmax can only produce one ranking, but tokens often need to attend to syntactically and semantically distinct things at the same time.

## §1 The split is at the dot-product, not at the weights

You have one `W_Q` of shape `(d_model, d_model)`. You project to full `d_model`, then reshape the last dimension into `(heads, head_dim)`. Each head computes an *independent* attention over its own 64-dim subspace. They cannot mix until the final concat plus `W_O`.

```python
Q = (X @ W_Q).reshape(N, 8, 64)   # ONE matmul, then split last dim
for h in range(8):
    scores_h = Q[h] @ K[h].T / sqrt(64)   # 8 INDEPENDENT similarity matrices
    out_h    = softmax(scores_h) @ V[h]
```

## §2 The real reason: softmax can only pick one ranking

Softmax is a sharpening function. It picks winners and crushes losers. With one softmax per query, you produce *one* distribution over the sequence. But a token usually needs info from multiple *unrelated* sources at once: its syntactic head, its semantic referent, the paragraph topic.

| Single head | 8 heads |
|---|---|
| One softmax must spread weight across cat AND mat AND structure. To get any one signal sharply, it must crush the others. Result is a muddled average. | Head 0 puts 0.9 on subject "cat". Head 1 puts 0.85 on object "mat". Head 2 captures the relative-clause structure. Each delivers a clean, sharp signal in its own subspace. |

!!! warning "Trap question"
    "Why not 512 heads of dim 1?" Each head needs enough dimensions to represent the *aspect* it matches on. Below ~32 dims you cannot form useful subspace projections; you have turned attention into a noisy per-feature gate. Sweet spot is head_dim 64 to 128.

## §3 Compute and memory accounting

Total FLOPs are identical to single-head (8 × N²·64 = N²·512). The attention-matrix memory scales with the number of heads, because every head caches its own K and V. **This is exactly why GQA exists** — see [GQA, MQA, MLA](../inference-arch/gqa-mqa-mla.md).

## §4 Why attention is O(N²), not O(N³)

The confusion comes from mixing up two different dimensions. `Q @ Kᵀ` has shapes `(N, d) @ (d, N) → (N, N)`. Matmul cost for `(A,B)@(B,C)` is `O(A·B·C)`, so here it is **O(N · d · N) = O(N²·d)**.

The N³ rule you remembered is for multiplying two *square* N×N matrices. Q and K are tall-skinny (N tokens, only d features each). The inner dimension is `d`, not N. As long as d ≪ N (true for long context: d ≈ 4096, N ≈ 128K), the bottleneck is the N² growth.

!!! abstract "Why people drop the d"
    Saying "attention is O(N²)" treats `d` as a fixed constant and tracks only how cost scales with *sequence length*. For a fixed model, doubling context from 64K → 128K quadruples cost and memory. The full complexity is O(N²·d); the casual phrasing tracks only N.

| Context N | Compute (N²·d_head) | Attention matrix memory (fp16) |
|---|---|---|
| 1K | 128 M ops | 2 MB |
| 8K | 8 B ops | 128 MB |
| 32K | 130 B ops | 2 GB |
| 128K | 2 T ops | 32 GB *(one head, one layer)* |

Both compute (N²·d) and memory (N²) scale quadratically. The memory side is what [Flash Attention](../inference-arch/flash-attention.md) fixes. By tiling so the N×N matrix is never written to HBM, it keeps the same FLOPs but drops memory from O(N²) to O(N).

## §5 The textual visualization of a layer

Picture a grid. Write your N tokens along the top row and also along the left column. Now fill in the N×N grid: cell (i, j) is "how much does token i attend to token j?" This is the attention matrix. Row i sums to 1 (softmax). For each token i:

1. Dot-product i's query vector with every token's key vector → N raw scores.
2. Divide by sqrt(d_k) so the scores do not explode.
3. Apply softmax → N weights that sum to 1.
4. Use those weights to average all N value vectors → new representation for token i.

The full layer: input → LayerNorm → Attention → residual add → LayerNorm → FFN → residual add → output. Stack 32 to 80 of these. Add input embeddings plus positional info at the bottom, add a final linear+softmax head at the top for next-token prediction.

## Interview Questions

**Q1: Why multi-head attention instead of one big attention?**

Different heads learn to capture different types of relationships: syntax, coreference, long-range semantics. One big head of the same total dim would have to mix all these signals into one representation. Splitting into heads gives the model multiple independent "perspectives" that get combined by the output projection. In practice, ablation studies show you can prune heads significantly, but starting with multi-head converges better and lets the model discover specialized circuits.

**Q2: What is the parameter cost of going from 1 head to 8 heads?**

Zero. The split happens after the projection, by reshape. Total compute is also identical. Only the attention-matrix memory scales with head count, because each head holds its own attention scores and KV.

**Q3 (Trap): Why not use 512 heads of dim 1?**

Each head needs enough dimensions to project queries and keys into a useful subspace. Below ~32 dimensions per head, the subspace collapses and you are effectively attending on noisy per-feature signals. The sweet spot is head_dim of 64 to 128, which is why models choose head counts as `d_model / 64` or `d_model / 128`.

**Q4: Where does the O(N²) come from and why is it a memory problem, not just a compute problem?**

Q @ Kᵀ has shape (N, d) @ (d, N) → (N, N). Compute is O(N²·d). Memory is O(N²) for the scores plus probabilities. At N = 8K and d_head = 128 in fp16, the scores matrix is 128 MB per head per layer. At 128K it is 32 GB. Compute and memory both scale quadratically with N; Flash Attention fixes the memory side by tiling so the N×N matrix never materializes in HBM.
