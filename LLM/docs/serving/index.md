# Serving and Optimization

Most people think training is the hard part. In production, **inference is where the money burns**. A trained model sits on disk; an inference system has to serve thousands of concurrent users, each wanting different outputs of different lengths, at latencies under a second, without melting your GPU budget. This section covers the stack: quantization formats, decoding tricks, vLLM, SGLang, and a minimal Docker + FastAPI deployment recipe.

!!! tip "Rapid Recall"
    LLM inference is **memory-bandwidth-bound, not compute-bound** (see [Prefill vs decode](../inference-arch/prefill-vs-decode.md)). Every optimization is one of three things: **reduce how much memory you move** (quantization), **generate more tokens per memory load** (continuous batching, speculative decoding), or **manage memory more efficiently** (PagedAttention). The 2026 production default is **vLLM with AWQ INT4 (or FP8 on H100) plus continuous batching plus prefix caching plus EAGLE-3 speculative decoding**. For multi-turn chatbots and RAG with shared prefixes, **SGLang's RadixAttention** wins by ~29% over vLLM. TGI is EOL; TensorRT-LLM is max-perf at high cost; Ollama / llama.cpp is dev-only.

## The fundamental problem

LLM inference is memory-bandwidth-bound, not compute-bound. Every token generation requires loading the entire model's weights from GPU memory, doing one matmul, and writing back. Modern GPUs can do 10¹⁴ FLOPs/sec, but HBM memory bandwidth is "only" ~3 TB/s. For a 70B model at FP16 (140 GB), you cannot load the weights more than ~20 times per second, capping your token generation rate. **The GPU is sitting idle most of the time, waiting on memory.**

Every technique in this section is fundamentally about one of three things:

1. **Reduce how much memory you move** (quantization).
2. **Generate more tokens per memory load** (continuous batching, speculative decoding).
3. **Manage memory more efficiently** (PagedAttention).

Keep that frame in mind as you go through each technique.

## Pages in this section

- **[Quantization formats](quantization-formats.md)** — INT8 vs INT4 vs FP8 vs NF4 vs GGUF; the 2026 decision tree; GPTQ vs AWQ math.
- **[Decoding and speculative](decoding-and-speculative.md)** — sampling (greedy, top-k, top-p, temperature); speculative decoding's rejection-sampling argument; EAGLE-3, Medusa, PLD.
- **[vLLM](vllm.md)** — PagedAttention deep, continuous batching, tensor and pipeline parallelism, prefix caching.
- **[SGLang and alternatives](sglang-and-alternatives.md)** — RadixAttention, when SGLang beats vLLM, TGI's end of life, TensorRT-LLM, Ollama.
- **[Docker and FastAPI deploy](docker-fastapi-deploy.md)** — from a Python script to an HTTP service in a container; production upgrades.

## The decision tree, before the details

```
Do you have Blackwell / Hopper (H100, H200, B100)?
├── YES: Use FP8 (native tensor-core support, near-lossless)
└── NO: Is VRAM tight?
         ├── YES: Use AWQ INT4 (best quality at 4-bit on GPU)
         └── NO: Use INT8 (safest, ~99% quality)
```

For the full decoding stack on top of that, see [Decoding and speculative](decoding-and-speculative.md). For the serving framework on top, see [vLLM](vllm.md) for diverse workloads and [SGLang and alternatives](sglang-and-alternatives.md) for shared-prefix workloads.
