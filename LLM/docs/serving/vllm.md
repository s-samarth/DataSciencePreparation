# vLLM — PagedAttention and Continuous Batching

vLLM is an open-source inference server from UC Berkeley. In 2026 it is the **de facto standard** for serving open-source LLMs in production. Understanding vLLM = understanding how modern serving works. The two big innovations are **PagedAttention** (KV-cache memory management borrowed from OS virtual memory) and **continuous batching** (iteration-level scheduling). The rest is engineering.

!!! tip "Rapid Recall"
    vLLM = OpenAI-compatible Python server wrapping a model + PagedAttention + continuous batching + quantization (FP8/AWQ/GPTQ/INT8/GGUF) + speculative decoding + tensor and pipeline parallelism + prefix caching + chunked prefill. **PagedAttention** allocates KV-cache in small fixed-size blocks (16 tokens) linked via a page table, like OS virtual memory. Eliminates the 60-80% fragmentation waste of static allocation; lets you pack 2-4× more concurrent requests; enables prefix sharing (copy-on-write for shared system prompts). **Continuous batching** schedules at the iteration level: after each forward pass, kick out any request that hit EOS and admit a new one from the queue. The GPU never idles waiting for the longest request in the batch. ~20× throughput vs static batching for typical workloads. Both required custom CUDA kernels.

## §1 What vLLM actually is

A Python server that wraps a model and handles:

- HTTP API (OpenAI-compatible).
- Request queueing and scheduling.
- KV cache management (PagedAttention).
- Continuous batching.
- Tensor parallelism, pipeline parallelism, expert parallelism.
- Speculative decoding.
- Quantization support (FP8, AWQ, GPTQ, INT8, INT4, GGUF).
- Prefix caching.
- Chunked prefill.

The big-ticket innovations are PagedAttention and continuous batching. Rest is engineering.

## §2 PagedAttention — the key insight

### 2.1 The problem

During generation, each request maintains a KV cache (see [KV-cache](../inference-arch/kv-cache.md)) — stored keys and values for every previous token, so you do not recompute attention over history. KV cache grows with sequence length and is **unpredictable** (you do not know how long the output will be).

Old approach: **pre-allocate a contiguous block of GPU memory** for each request based on max sequence length (e.g., 4096 tokens). Problems:

- **Internal fragmentation:** if the output is only 50 tokens, you wasted the memory for the other 4046 slots.
- **External fragmentation:** you cannot fit a new request in the gap between two existing ones.
- **Studies show 60-80% of KV cache memory is wasted** with static allocation.

### 2.2 The solution (borrowed from OS virtual memory)

PagedAttention allocates KV cache in **small fixed-size blocks** (typically 16 tokens each), linked together via a page table. Each request has a "logical" view of its cache as contiguous, but physically the blocks are scattered in GPU memory.

```
Request A's logical KV cache: [tok1, tok2, tok3, tok4, tok5, ...]
                                  ↓      ↓      ↓      ↓      ↓
                              Block42 Block17 Block5 Block99 Block31 (physical)
```

### 2.3 Results

- **~0% fragmentation** (you allocate exactly what you use).
- **Can pack 2-4× more concurrent requests** into the same VRAM.
- **Enables prefix sharing:** two requests with the same system prompt share the same physical blocks (copy-on-write). Huge win for chat apps where every request has the same system prompt.

### 2.4 Why it required a custom kernel

Standard attention kernels assume contiguous K and V tensors. PagedAttention required rewriting the attention CUDA kernel to index through a page table. This is the vLLM paper's main technical contribution (Kwon et al., SOSP 2023).

## §3 Continuous batching

### 3.1 The problem with static batching

Traditional batching waits for all N requests in a batch to finish before starting a new batch. Problem: **output lengths vary wildly.** Request A generates 50 tokens, Request B generates 500 tokens. The slots for A, C, D sit idle for 450 tokens after A finishes, waiting for B to catch up.

Measured in production: **SM utilization hovers at 30-40% with static batching**, even at moderate request rates.

### 3.2 The solution

Schedule at the **iteration level**, not the batch level. After each forward pass:

1. If a request finished (generated EOS), kick it out.
2. If a new request is waiting, add it to the batch.
3. Continue.

```
Time →
Static batching:  [A B C D][finished A,C,D, still waiting for B..................]
Continuous:       [A B C D][E B C D][E B F D][E B F G]...
                   A done     C done   D done
                   E joins    F joins  G joins
```

Result: **~20× throughput improvement** over static batching for typical workloads. GPU stays busy. This is the single biggest win from vLLM.

## §4 Tensor parallelism (TP)

For models that do not fit in a single GPU (70B+), split each weight matrix across GPUs **column-wise** or **row-wise**. Each GPU holds a slice, does its portion of the matmul, then they **all-reduce** the results.

```
Single-GPU:   Wx (one GPU does full matmul)
TP=4:         GPU0: W[:,  0: d/4] * x → partial
              GPU1: W[:, d/4: d/2] * x → partial
              GPU2: W[:, d/2:3d/4] * x → partial
              GPU3: W[:,3d/4:   d] * x → partial
              all_reduce → full Wx
```

**Tradeoffs:**

- Communication cost per layer (all-reduce). Needs NVLink or fast interconnect (InfiniBand); PCIe chokes.
- Model must be divisible by TP size (pad if not).
- Standard pairings: 70B at TP=4 on 4×A100, 405B at TP=8 on 8×H100.

### 4.1 Pipeline parallelism (PP)

Split the model **vertically** by layers. GPU0 runs layers 1-10, GPU1 runs 11-20, etc. Outputs flow through sequentially. Lower comm cost than TP, but introduces pipeline bubbles (idle time) unless carefully scheduled. Usually combined with TP for very large models (TP within node, PP across nodes).

## §5 Other vLLM features

- **Prefix caching.** Cache KV for shared prompt prefixes. With system prompts, this is often 80%+ of your prefill tokens.
- **Chunked prefill.** Split long prefill into chunks that interleave with decode. Improves TTFT (time-to-first-token) tail latency.
- **FlashAttention.** Memory-efficient attention kernel. Default on modern vLLM. See [Flash Attention](../inference-arch/flash-attention.md).
- **Disaggregated prefill/decode.** Run prefill on one set of GPUs and decode on another. Frontier technique as of 2026. See [Prefill vs decode §4](../inference-arch/prefill-vs-decode.md).

## §6 Minimal vLLM example

```bash
# Install
pip install vllm

# Serve a model with OpenAI-compatible API
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching

# That's it. Now curl http://localhost:8000/v1/chat/completions
# like it's the OpenAI API.
```

With AWQ quant + spec decoding:

```bash
vllm serve TheBloke/Llama-3.1-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 4 \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5
```

Python API for batch offline inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.92,
)
params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Explain quantum entanglement"], params)
print(outputs[0].outputs[0].text)
```

Key hyperparameters:

- `gpu_memory_utilization`: default 0.90. Raise to 0.95 for more KV cache. OOM if too high.
- `tensor_parallel_size`: number of GPUs to split across. Must divide number of attention heads.
- `max_model_len`: cap context window to reduce KV cache memory pressure.

## Interview Questions

**Q1: Explain PagedAttention in one minute.**

In normal LLM serving, the KV cache is allocated as a contiguous block per request, sized for the max possible sequence length. This wastes 60-80% of memory due to internal fragmentation (short outputs) and external fragmentation (gaps). PagedAttention borrows from OS virtual memory: allocate the KV cache in small fixed-size blocks (typically 16 tokens) linked via a page table. Each request has a logical contiguous view, but physical blocks are scattered. Eliminates fragmentation, allows prefix sharing between requests, and lets you pack 2-4× more concurrent requests into the same VRAM. Required a custom CUDA attention kernel that walks the page table.

**Q2: What is continuous batching and why is it 20× faster than static batching?**

Static batching groups N requests, processes them to completion, then starts the next batch. Problem: output lengths vary wildly. A request generating 50 tokens sits in an idle slot for 450 iterations waiting for a 500-token request to finish. With LLM output distributions, this means GPU utilization of 30 to 40%. Continuous batching schedules at the iteration level instead — after each forward pass, kick out any request that generated EOS and admit a new one from the queue. The GPU stays busy; throughput goes up 20×+. This is vLLM's biggest single optimization.

**Q3: I am running a 70B model and can barely fit it on a single H100 (80GB). What is my optimization playbook?**

First, quantize. AWQ INT4 gets the 70B model from ~140 GB to ~35 GB, freeing up room for the KV cache and enabling bigger batches. Second, use vLLM with PagedAttention — the 45 GB remaining VRAM fits several concurrent KV caches, and continuous batching keeps the GPU utilized. Third, enable speculative decoding with a small draft model (Llama-3.2-1B) for 2-3× decode speedup. Fourth, turn on prefix caching if your workload has repeated system prompts. With all four, you are looking at 10-20× throughput vs naive FP16 serving on the same GPU.

**Q4: Explain tensor parallelism vs pipeline parallelism.**

Tensor parallelism splits each weight matrix across GPUs — column-wise or row-wise — and each GPU does a partial matmul followed by an all-reduce. Fine-grained, communicates every layer, needs fast interconnect (NVLink). Pipeline parallelism splits the model by layers — GPU 0 runs layers 1-10, GPU 1 runs 11-20, etc. — and activations flow through sequentially. Coarser, less communication, but suffers from pipeline bubbles unless micro-batching is scheduled carefully. In practice: TP within a node (NVLink), PP across nodes (InfiniBand). For a 405B model on 2 nodes of 8×H100, typical setup is TP=8 within each node, PP=2 across them.

**Q5: Name vLLM's five key optimizations.**

PagedAttention (KV-cache fragmentation eliminated), continuous batching (iteration-level scheduling), quantization support (FP8/AWQ/GPTQ/INT8/GGUF native), speculative decoding (EAGLE-3 / Medusa / vanilla), prefix caching (shared system-prompt KV reuse). Plus the engineering: chunked prefill, FlashAttention by default, OpenAI-compatible API, TP/PP, expert parallelism for MoE.

**Q6 (Trap): vLLM is open-source — why has anyone else still been in business?**

vLLM excels at diverse-prompt, throughput-oriented workloads. Three competitors win in specific niches. (1) SGLang for shared-prefix workloads (multi-turn chat, RAG, agents) — its RadixAttention gives ~29% more throughput on those. (2) TensorRT-LLM for absolute peak performance on NVIDIA hardware when you can afford a 10-30 min compilation step and only serve one model long-term. (3) Ollama / llama.cpp for laptops and CPU+GPU hybrid — vLLM does not run on CPU. The 2026 default for production GPU serving is vLLM unless your workload specifically fits one of the alternatives.
