# Prefill vs Decode — Two Workloads on One GPU

The single most important frame for understanding LLM inference. The GPU, the weights, and the matmul kernels never change between these phases. What changes is the **shape of the work**, and that shape flips the bottleneck entirely. Prefill saturates compute; decode is memory-bandwidth-bound. Every optimization in this section maps to which phase it accelerates.

!!! tip "Rapid Recall"
    H100 has ~1000 TFLOPS compute and ~3 TB/s HBM bandwidth. The ratio gives **arithmetic intensity**: you need ~330 FLOPs per byte loaded to keep the compute units busy. **Prefill** processes all prompt tokens in parallel: matmul is `[2000, d] @ [d, d]`, weight loaded once and reused across 2000 rows, arithmetic intensity ~2000, compute-bound, looks like training. **Decode** processes one token: matmul is `[1, d] @ [d, d]`, weight loaded once for one row, arithmetic intensity ~1, memory-bound, the GPU sits idle waiting for HBM. **Decode latency floor:** `model_bytes / HBM_bandwidth`. For LLaMA-70B FP16 on H100: `140 GB / 3 TB/s ≈ 47 ms/token`. This is exactly why FP8/FP4 quantization helps decode so much — halving weight bytes halves the actual bottleneck.

## §1 The two hardware numbers that govern everything

- **Compute throughput** — ~1000 TFLOPS FP16 on an H100. How fast it multiplies.
- **Memory bandwidth** — ~3 TB/s from HBM. How fast it reads weights.

The ratio gives **arithmetic intensity**: you need to do ~330 FLOPs per byte loaded to keep the compute units busy. Below that, the GPU sits idle waiting on memory — an expensive memory controller.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 760 280" xmlns="http://www.w3.org/2000/svg">
<text x="20" y="30" class="svg-title" fill="#f4a847">PREFILL — compute-bound</text>
<text x="430" y="30" class="svg-title" fill="#f4a847">DECODE — memory-bound</text>
<rect x="20" y="55" width="120" height="120" rx="6" fill="#1e2331" stroke="#5ad1c5" stroke-width="1.5"/>
<text x="80" y="120" text-anchor="middle" class="svg-soft" fill="#aab0be">2000 × d</text>
<text x="80" y="138" text-anchor="middle" class="svg-lab" fill="#6f7686">prompt rows</text>
<text x="155" y="120" class="svg-ink" fill="#e7e9ee">×</text>
<rect x="175" y="55" width="120" height="120" rx="6" fill="#1e2331" stroke="#5ad1c5" stroke-width="1.5"/>
<text x="235" y="118" text-anchor="middle" class="svg-soft" fill="#aab0be">d × d</text>
<text x="235" y="136" text-anchor="middle" class="svg-lab" fill="#6f7686">weight (loaded once)</text>
<text x="20" y="205" class="svg-soft" fill="#aab0be">≈ 2000 FLOPs / byte loaded</text>
<text x="20" y="225" class="svg-lab" fill="#7fc88a">Compute units SATURATED, looks like training</text>
<text x="20" y="245" class="svg-lab" fill="#6f7686">Latency scales ~linearly with prompt length</text>
<line x1="385" y1="50" x2="385" y2="260" stroke="#2a2f3d" stroke-width="1"/>
<rect x="430" y="55" width="120" height="14" rx="4" fill="#1e2331" stroke="#ef7b8e" stroke-width="1.5"/>
<text x="490" y="50" text-anchor="middle" class="svg-lab" fill="#6f7686">1 × d (one token)</text>
<text x="565" y="66" class="svg-ink" fill="#e7e9ee">×</text>
<rect x="585" y="55" width="120" height="120" rx="6" fill="#1e2331" stroke="#ef7b8e" stroke-width="1.5"/>
<text x="645" y="118" text-anchor="middle" class="svg-soft" fill="#aab0be">d × d</text>
<text x="645" y="136" text-anchor="middle" class="svg-lab" fill="#6f7686">SAME full weight load</text>
<text x="430" y="205" class="svg-soft" fill="#aab0be">≈ 1 FLOP / byte loaded</text>
<text x="430" y="225" class="svg-lab" fill="#ef7b8e">Compute IDLE, waiting on HBM</text>
<text x="430" y="245" class="svg-lab" fill="#6f7686">Floor = model_bytes / HBM_bandwidth</text>
</svg>
<figcaption>Same weight matrix loaded in both phases; decode does 1/2000th the math per load.</figcaption>
</figure>

## §2 Prefill — the compute-bound regime

You know all prompt tokens up front, so you process them in parallel. Q/K/V projections become fat matmuls (`[2000, d] @ [d, d]`): the weight is loaded **once** and reused across 2000 rows. Arithmetic intensity sits far above 330. This is why prefill **looks like training** — same parallel-over-sequence compute graph, minus the backward pass.

Prefill cost scales linearly with prompt length. Doubling the prompt doubles the prefill time. Tokens per second during prefill on H100 with a 70B model in BF16 is in the thousands.

## §3 Decode — the memory-bound regime

Autoregressive generation is one token at a time; you cannot parallelize across the time dimension. The Q projection becomes `[1, d] @ [d, d]` — you load the entire weight matrix to compute a single token's vector. The math finishes in nanoseconds, then the GPU waits hundreds of nanoseconds for the next weights to arrive.

!!! abstract "Intuition lock"
    Decode latency per token ≈ `total_params × bytes / HBM_bandwidth`. For LLaMA-70B FP16 on H100: `140 GB / 3 TB/s ≈ 47 ms/token` — a hard floor. You cannot decode faster than you can read the weights once. This is exactly **why FP8/FP4 quantization helps decode so much**: halving weight bytes halves the actual bottleneck.

Tokens per second during decode for a single sequence on H100 with a 70B model in BF16 is roughly 20 to 25. Batch the decode across many sequences and you amortize the weight load, raising throughput. This is what continuous batching exploits; see [vLLM](../serving/vllm.md).

## §4 Serving consequence: prefill/decode disaggregation

Opposite bottlenecks want opposite hardware and batching. New prefill requests block ongoing decode (a "decode stall"). Frontier stacks (DeepSeek deploy, SGLang disaggregated, NVIDIA Dynamo) run **separate GPU pools**: prefill nodes compute the KV-cache and ship it to decode nodes. Each pool is right-sized for its own bottleneck — the system-level analog of GQA.

The KV-cache produced by a prefill node has to be transferred to the decode node, which is itself a non-trivial cost (~100s of MB to several GB per request at long context), so disaggregation only pays off for high-throughput workloads where the prefill stall would otherwise dominate.

## §5 Why this framing matters

Every optimization in this section maps to one regime.

| Technique | Helps which phase? | Why |
|---|---|---|
| KV-cache | Decode | Avoids O(N²) recompute per token. |
| GQA / MLA | Decode | Cuts KV-cache memory traffic. |
| Flash Attention | Both, but mostly prefill | Tiles the N×N matrix so HBM traffic drops. |
| FP8 / INT4 weight quant | Decode | Halves or quarters weight bytes, the actual bottleneck. |
| Continuous batching | Decode | Amortizes weight load across concurrent sequences. |
| Speculative decoding | Decode | Generates K tokens per weight-streaming cycle. |
| Prefix caching | Prefill | Skips redundant attention over shared system prompts. |
| MoE | Both | Activates fewer parameters per token. |
| MTP | Decode | Produces K tokens per forward pass via auxiliary heads. |

The most expensive optimization to engineer (Flash Attention rewrites the kernel) helps the cheapest phase (prefill); the cheapest optimizations (quantization, KV-cache) help the expensive phase (decode). That asymmetry is exactly what you would expect once you see decode is the bandwidth-bound bottleneck.

## Interview Questions

**Q1: Why is LLM decoding memory-bound but prefill is compute-bound?**

During prefill, you process hundreds or thousands of prompt tokens in parallel through each layer — big matmuls that saturate tensor cores. During decode, you process one token at a time, so the matmuls are thin (matrix × vector). You still have to load the full model weights and full KV-cache from HBM for each token generated, but the compute per byte loaded is tiny. Result: the GPU sits idle waiting for memory. Every decode optimization — GQA, MLA, speculative decoding — targets memory bandwidth, not FLOPS.

**Q2: What is the theoretical floor on decode latency for LLaMA-70B FP16 on H100?**

`model_bytes / HBM_bandwidth = 140 GB / 3 TB/s ≈ 47 ms/token`. You cannot decode faster than you can read the weights once. Quantizing to FP8 halves the bytes and halves the floor to ~23 ms; INT4 quarters it to ~12 ms. This is the single most important number for understanding why quantization matters for inference cost.

**Q3: Why does prefill/decode disaggregation help and when does it not pay off?**

Different bottlenecks want different hardware and batching. Prefill is compute-bound (fewer, beefier GPUs with high TFLOPS are ideal); decode is memory-bound (cheaper GPUs with high HBM bandwidth and lots of VRAM are ideal). Running them on the same pool means decode stalls every time a new prefill arrives. Disaggregation right-sizes each pool. Cost: KV-cache transfer between pools is non-trivial (100s of MB to several GB per request at long context), so it only pays off above a throughput threshold where the prefill stall would otherwise dominate.

**Q4 (Trap): If decode is memory-bound, why does quantization that touches only weights help so much?**

Because in decode, the entire weight matrix gets loaded once per token (for matrix-vector), so weight bytes are the dominant data moved from HBM. Halve weight bytes (FP16 → FP8) and you halve the actual bandwidth pressure, which directly halves decode latency. KV-cache and activations matter too at long context, but for short to moderate sequences the weight read dominates and weight quantization is the highest-leverage decode optimization.
