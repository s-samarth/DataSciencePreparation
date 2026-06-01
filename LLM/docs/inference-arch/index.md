# Modern Inference Architecture

The 2017 transformer is beautiful but inefficient at inference. Every modern LLM (LLaMA 3, DeepSeek V4, Mistral, Claude, GPT-5) adds a stack of optimizations to make training and inference tractable. This section covers the six that matter most: prefill vs decode framing, KV-cache, GQA/MLA, Flash Attention, MoE, and frontier techniques (NTK-aware RoPE, CSA+HCA sparse attention, QAT, MTP).

!!! tip "Rapid Recall"
    At training time, transformers are compute-bound (big matmuls). At inference time, when generating text token by token, they become **memory-bound** — most time is spent moving weights and past context from memory, not computing. Every modern architecture trick is about cutting memory bandwidth or cutting compute per active parameter. **KV-cache:** do not recompute attention over the past; store it. **Flash Attention:** do not write the N×N attention matrix to slow memory; compute it in tiles that fit in fast memory. **GQA / MLA:** do not keep N key-value heads when you can share or compress them. **MoE:** do not activate all parameters for every token; route to a sparse subset.

## The four-trick summary

- **KV-cache:** do not recompute attention over the past; store it.
- **Flash Attention:** do not write the N×N attention matrix to slow memory; compute it in tiles that fit in fast memory.
- **GQA:** do not keep N key-value heads when you can share them across query heads.
- **MoE:** do not activate all parameters for every token; route to a sparse subset.

## Pages in this section

- **[Prefill vs decode](prefill-vs-decode.md)** — the two regimes on one GPU; the arithmetic-intensity argument; prefill/decode disaggregation as the system-level analog of GQA.
- **[KV-cache](kv-cache.md)** — the "2 LHDN B-bytes" formula and why KV bandwidth dominates weight bandwidth at long context.
- **[GQA, MQA, MLA](gqa-mqa-mla.md)** — the spectrum from MHA to MLA, decode-time tradeoff = compute for memory bandwidth.
- **[Flash Attention](flash-attention.md)** — the online-softmax tiling trick; FA2 / FA3 / FA4 evolution co-designed for Ampere / Hopper / Blackwell.
- **[Mixture of Experts](mixture-of-experts.md)** — routing, auxiliary-loss-free balancing (DeepSeek V3), fine-grained experts plus a shared expert.
- **[Frontier techniques](frontier-techniques.md)** — NTK-aware RoPE scaling, CSA+HCA sparse attention at 1M context, QAT for native INT4, Multi-Token Prediction.

## The memory hierarchy is the boss

A modern H100 GPU has:

- Tensor cores that do ~1000 TFLOPS of FP16 matmul.
- ~80 GB of HBM (high-bandwidth memory) at ~3 TB/s.
- ~50 MB of on-chip SRAM per streaming multiprocessor at ~20 TB/s.

SRAM is ~10× faster than HBM. If your algorithm keeps data in SRAM, it flies; if it bounces to HBM, it crawls. **Every modern LLM optimization is about respecting this hierarchy.**

## Inference has two phases

1. **Prefill (processing the prompt):** compute-bound. Batched matmul over all prompt tokens in parallel. Looks like training.
2. **Decode (generating output tokens):** memory-bound. One token at a time. You are not computing much; you are loading weights and KV-cache from HBM. This phase is where KV-cache, GQA, and MLA matter.

For the serving stack built on top of this hardware (vLLM, SGLang, quantization formats, speculative decoding), see [Serving and Optimization](../serving/index.md).
