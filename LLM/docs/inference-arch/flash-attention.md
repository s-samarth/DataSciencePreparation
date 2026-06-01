# Flash Attention and the Online-Softmax Trick

Standard attention writes the N×N attention matrix to HBM, reads it back for softmax, writes the probability matrix, reads it again for the value matmul. At least 4 round trips through the slow lane. Flash Attention's idea, in one sentence: **never materialize the N×N matrix.** Tile the computation so attention scores live and die inside fast SRAM, only the final output ever touches HBM. The blocker is softmax — it is non-local, needing the whole row's sum to normalize. The fix is **online softmax**.

!!! tip "Rapid Recall"
    Standard attention round-trips the N×N scores and probabilities through HBM at least 4 times. Flash Attention **tiles Q, K, V into chunks that fit in SRAM** and runs **online softmax**: at each new tile, update `running_max = max(m, m_j)`, then rescale the running sum and partial output using `exp(m_old − m_new)` — exact same result as batch softmax, but streaming. HBM traffic drops from O(N²) to O(N·d), the theoretical floor. 2 to 4× speedup, exact attention (not approximate, unlike Linformer / Performer / Longformer). Each FA version is co-designed for one GPU generation: FA2 attacked parallelism, FA3 exploited Hopper's WGMMA + TMA async (~75% util), FA4 attacks Blackwell's tensor-core vs SFU asymmetry via polynomial `exp()` and conditional rescaling.

## §1 What is actually wasted in naive attention

Standard attention computes `S = QKᵀ` → `P = softmax(S)` → `O = P@V`. The `S` and `P` matrices are **N×N** (1 billion floats per head at 32K). The real cost is not capacity, it is that you **write S to HBM, read it back for softmax, write P, read it again** for the value matmul. At least 4 round-trips through the slow lane.

| Memory tier | Size (H100) | Bandwidth |
|---|---|---|
| HBM ("GPU RAM") | ~80 GB | ~3 TB/s |
| SRAM (on-chip / SM) | ~228 KB × 132 SMs | ~19 TB/s (~6×) |

SRAM is ~6× faster but ~350,000× smaller. Anything that keeps data in SRAM wins big; anything that bounces to HBM loses.

## §2 The idea in one sentence

**Never materialize the N×N matrix.** Tile the computation so attention scores live and die inside fast SRAM; only the final `O` ever touches HBM. The blocker is softmax — it is non-local, needing the whole row's sum to normalize. The fix is **online softmax**.

## §3 Online softmax — the rescaling trick

```
True max over A∪B:   m = max(m_A, m_B)
True sum over A∪B:   s = exp(m_A − m)·s_A + exp(m_B − m)·s_B
```

Because `exp(a − m) = exp(a − m_A) · exp(m_A − m)`, the old running sum just needs a single scalar rescale when a new chunk arrives. Stream chunks, carry two scalars (running max, running sum) plus a partial output. Never see the full row.

The key math insight: softmax is invariant to adding a constant to all inputs. `softmax(x − c) = softmax(x)`. That is what lets you combine partial softmaxes by rescaling.

## §4 The tiling picture

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 760 220" xmlns="http://www.w3.org/2000/svg">
<text x="20" y="26" class="svg-title" fill="#f4a847">TILED ATTENTION — Q tile fixed, K/V tiles stream through SRAM</text>
<rect x="20" y="50" width="70" height="120" rx="6" fill="#1e2331" stroke="#f4a847" stroke-width="1.5"/>
<text x="55" y="115" text-anchor="middle" class="svg-soft" fill="#aab0be">Q tile</text>
<text x="55" y="186" text-anchor="middle" class="svg-lab" fill="#6f7686">in SRAM</text>
<g>
<rect x="140" y="50" width="60" height="120" rx="6" fill="#1e2331" stroke="#5ad1c5" stroke-width="1.2"/>
<rect x="215" y="50" width="60" height="120" rx="6" fill="#1e2331" stroke="#5ad1c5" stroke-width="1.2"/>
<rect x="290" y="50" width="60" height="120" rx="6" fill="#1e2331" stroke="#5ad1c5" stroke-width="1.2"/>
<text x="245" y="115" text-anchor="middle" class="svg-soft" fill="#aab0be">K/V tiles →</text>
</g>
<text x="380" y="100" class="svg-ink" fill="#e7e9ee">running max ↻</text>
<text x="380" y="125" class="svg-ink" fill="#e7e9ee">running sum ↻</text>
<text x="380" y="150" class="svg-lab" fill="#6f7686">rescale each new tile</text>
<line x1="540" y1="110" x2="585" y2="110" stroke="#f4a847" stroke-width="1.5" marker-end="url(#fa)"/>
<rect x="590" y="80" width="80" height="60" rx="6" fill="#1e2331" stroke="#7fc88a" stroke-width="1.5"/>
<text x="630" y="108" text-anchor="middle" class="svg-soft" fill="#aab0be">O tile</text>
<text x="630" y="124" text-anchor="middle" class="svg-lab" fill="#6f7686">→ HBM once</text>
<defs><marker id="fa" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#f4a847"/></marker></defs>
</svg>
<figcaption>HBM traffic drops from O(N²) to O(N·d), the theoretical floor. The scores never leave the chip.</figcaption>
</figure>

For each Q tile, loop over K, V tiles. Load the Q tile and one K, V tile pair into SRAM. Compute the partial attention block (B_r × B_c), do a running softmax using the online softmax algorithm, accumulate into output. Move to next K, V tile. Update output incrementally. The N×N matrix never materializes.

## §5 Three distinct wins

- **Speed (main win).** Crushes HBM traffic, the real bottleneck. ~2 to 4× over naive attention.
- **Memory (famous but secondary).** N² footprint vanishes → enables long-context training.
- **Exact, not approximate.** Identical output to vanilla attention. This is why it ate Linformer / Performer / Longformer — strictly better, no quality tradeoff.

## §6 2026 version landscape

| Version | Target GPU | Attacks |
|---|---|---|
| FA2 | Ampere / Ada (A100, 4090) | parallelism |
| FA3 | Hopper (H100, H200) | async MMA + warp specialization (~75% util) |
| FA4 | Blackwell (B200, GB200) | tensor-core vs SFU asymmetry (~1605 TFLOPS FP16, 71%) |

FA4's twist: Blackwell doubled tensor-core throughput but left the SFUs (which compute `exp()`) flat, so softmax became the bottleneck. FA4 responds by (1) **software-emulated exp via polynomials** on abundant FMA units, and (2) **conditional rescaling** — skip the online-softmax rescale unless the max shift threatens stability (~10× fewer rescales).

!!! abstract "The meta-pattern"
    Each FA version is **co-designed for one GPU generation's bottleneck**. FA1 attacked HBM traffic, FA2 attacked parallelism, FA3 attacked Hopper async, FA4 attacks Blackwell's compute asymmetry. The algorithm keeps evolving because the hardware keeps moving what is slow. Same exact attention output; the kernel engineering changes.

## §7 How to actually use it

In modern PyTorch, Flash Attention is the default backend for `scaled_dot_product_attention`:

```python
import torch.nn.functional as F

# PyTorch 2+ automatically picks Flash Attention when the inputs qualify.
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

For maximum control, the `flash-attn` package exposes the kernel directly:

```python
from flash_attn import flash_attn_func
out = flash_attn_func(Q, K, V, causal=True)   # Q, K, V: (B, N, H, d)
```

In vLLM, SGLang, and TensorRT-LLM, Flash Attention is on by default for supported shapes (head_dim ≤ 256, supported GPU arch).

## Interview Questions

**Q1: Walk me through Flash Attention's tiling strategy and why it is faster.**

Standard attention computes the full N×N score matrix in HBM, does softmax, multiplies by V, writes back to HBM. That is 3 round trips through a huge matrix. Flash Attention splits Q into row-tiles and K, V into column-tiles. For each Q tile, it iterates over K, V tiles, computes a partial attention block entirely in SRAM, runs online softmax to combine partial results, and only writes the final output tile to HBM. The N×N matrix never materializes in HBM. Savings: 2 to 4× speedup and O(N) memory instead of O(N²). The online softmax math exploits that softmax is shift-invariant — you can combine local max and sum statistics as you go.

**Q2: Why is online softmax mathematically equivalent to batch softmax?**

Softmax is invariant to adding a constant: `softmax(x − c) = softmax(x)`. So when you process tiles and a new tile reveals a larger max, you rescale the already-accumulated partial sum and output by `exp(old_max − new_max)`. The numerator and denominator each pick up the same multiplicative correction, and the final softmax is unchanged. It is exact, not approximate.

**Q3 (Trap): If Flash Attention is exact attention, just faster, why did FA2 and FA3 have to exist?**

Flash Attention 1 was algorithmically correct but did not fully saturate H100. FA2 improved work partitioning across GPU thread blocks to reduce non-matmul ops. FA3 exploits Hopper-specific hardware: WGMMA asynchronous matmul and TMA asynchronous memory. It warp-specializes producer and consumer work so memory loads overlap with compute, runs a 2-stage pipeline overlapping GEMM with softmax, and supports FP8 with incoherent processing (Hadamard randomization to reduce outlier quantization error). The algorithm stays the same exact attention; the kernel engineering changes. FA2 hit ~35% of H100 peak; FA3 hits ~75%.

**Q4: What is FA4 doing differently on Blackwell?**

Blackwell roughly doubled tensor-core throughput but did not scale the SFUs (special function units that compute `exp()`). Softmax suddenly became the bottleneck rather than the matmul. FA4 responds with (1) software-emulated `exp()` via polynomials on the abundant FMA units, freeing the SFUs, and (2) conditional rescaling — skip the online-softmax rescale unless the running max shifts enough to threaten stability, which cuts rescale ops ~10×. Same exact attention; different kernel optimized for different hardware proportions.

**Q5: When does Flash Attention NOT apply?**

Head dimensions larger than ~256 do not fit comfortably in SRAM tiles. Variable-length sequences and complex attention masks (e.g., custom block-sparse patterns) need careful kernel handling and may not be supported in all variants. For non-causal attention or specific masked patterns you sometimes have to fall back to xformers or a custom Triton kernel. But for standard causal LLM attention with head_dim ≤ 128, Flash Attention is essentially always on.
