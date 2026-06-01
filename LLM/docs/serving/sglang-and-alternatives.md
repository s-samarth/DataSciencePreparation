# SGLang and the Other Serving Frameworks

vLLM is the default in 2026, but four other frameworks each win a specific niche. SGLang wins on shared-prefix workloads (multi-turn chat, RAG, agents) via RadixAttention. TensorRT-LLM wins on absolute peak performance for a single long-lived production model on NVIDIA. TGI is end-of-life as of December 2025. Ollama and llama.cpp win on laptops and CPU+GPU hybrid dev.

!!! tip "Rapid Recall"
    **SGLang** = vLLM + **RadixAttention**: keeps completed KV computations in an LRU radix tree, retrieves them when new requests share a prefix. For multi-turn chatbots (system prompt reuse) and RAG (shared document context), this is ~29% higher throughput than vLLM (16,200 vs 12,500 tok/s on H100, Llama 8B). **TensorRT-LLM** (NVIDIA): max performance with 10-30 min ahead-of-time compilation; NVIDIA-only, painful to set up, worth it for single-model long-term production. **TGI**: HuggingFace Text Generation Inference. Entered maintenance mode December 2025; HuggingFace officially recommends vLLM or SGLang. **Ollama / llama.cpp / GGUF**: CPU+GPU hybrid via 4-8 bit GGUF quantization, starts in seconds, great for dev and single-user. Flatlines at ~22 req/s — do not use in production.

## §1 SGLang — RadixAttention

SGLang's key innovation: keep the KV cache in an **LRU radix tree**. When a new request comes in, find the longest prefix already in the tree and reuse its KV cache. Two requests with the same system prompt share the same physical blocks, just like vLLM's prefix caching but tracked at a finer granularity and persisting across requests.

### 1.1 When SGLang beats vLLM

- **Multi-turn chatbots.** Every turn shares the conversation history. Cache hit rates often 80%+.
- **RAG pipelines.** Many queries reference the same documents. Document KV reused.
- **Agentic workloads.** Tool definitions and system prompts repeat per call.
- **DeepSeek models.** SGLang is the officially preferred framework — 3.1× faster than vLLM on V3.

**Production numbers (2026, H100, Llama 8B):**

- SGLang: 16,200 tok/s.
- vLLM: 12,500 tok/s.
- Gap: 29% in SGLang's favor on shared-prefix workloads.

### 1.2 When SGLang LOSES to vLLM

For unique-prompt workloads (batch offline processing with no shared prefixes), RadixAttention's LRU cache adds overhead without benefit. Cache miss rate ~100%, but you still pay the tree-maintenance cost. Use vLLM or LMDeploy instead.

### 1.3 The command

```bash
# Start server (OpenAI-compatible API).
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.85

# Query.
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello"}]}'
```

SGLang supports the same quantization formats as vLLM (AWQ, GPTQ, FP8, GGUF) and the same speculative decoding methods (EAGLE-3, Medusa, vanilla). The differentiator is RadixAttention.

## §2 TensorRT-LLM (NVIDIA)

Max performance on NVIDIA hardware. Painful to use. Compile-ahead graphs.

The tradeoff: you spend 10-30 minutes compiling the model into an optimized inference graph for your specific GPU and sequence length. The compiled artifact is then ~5-15% faster than vLLM/SGLang for the same workload. Worth it when:

- You are deploying ONE model long-term in production.
- Throughput matters more than developer iteration speed.
- You are on NVIDIA hardware (only NVIDIA supported).

Not worth it when:

- You iterate on which model to deploy frequently (each switch costs 10-30 min).
- You serve multiple models on the same cluster.
- You need anything other than NVIDIA hardware.

In 2026 the gap between TensorRT-LLM and vLLM has narrowed substantially — vLLM with all optimizations on lands within ~10% of TensorRT-LLM for most workloads.

## §3 TGI (HuggingFace Text Generation Inference)

**End of life.** TGI entered maintenance mode in December 2025. HuggingFace now officially recommends vLLM or SGLang.

If you have an existing TGI deployment, migrate. The replacement path is straightforward: TGI's OpenAI-compatible endpoint maps directly to vLLM's, the model loading is HuggingFace-native in both, and you typically see a 2-3× throughput increase by switching.

## §4 Ollama, llama.cpp, GGUF

The CPU+GPU hybrid stack. GGUF (GPT-Generated Unified Format) is the quantization format optimized for llama.cpp, supporting Q2 to Q8 with various compression schemes. Ollama wraps llama.cpp with a friendly CLI and OpenAI-compatible API.

**When to use:**

- Local development on a laptop.
- Single-user apps (one request at a time).
- Quick experimentation before deciding on a serving stack.

**When NOT to use:**

- Production serving. Ollama flatlines at ~22 req/s regardless of concurrency. It is not built for high-throughput multi-tenant serving.
- Anything where you need vLLM-level features (continuous batching, PagedAttention, AWQ kernel performance).

Migration path: prototype on Ollama, ship on vLLM or SGLang.

## §5 LMDeploy and the Chinese ecosystem

LMDeploy (open-source, originally from Shanghai AI Lab) is a strong alternative for Qwen / DeepSeek models, with first-class support for those families. Comparable throughput to vLLM on those specific models. Worth knowing exists; not the default for Western-trained models.

## §6 The decision table

| Framework | Throughput (H100, Llama 8B) | Best workload | Setup complexity | Hardware |
|---|---|---|---|---|
| vLLM | 12,500 tok/s | Batch, diverse prompts | Low | NVIDIA, AMD, TPU, Gaudi |
| SGLang | 16,200 tok/s | Multi-turn, RAG, agents | Low | NVIDIA, AMD |
| TensorRT-LLM | ~15-18k tok/s (compiled) | Single model, max throughput | High (compile step) | NVIDIA only |
| Ollama | ~500 tok/s | Local dev | Very low | CPU + any |
| TGI | (EOL) | (migrate) | — | — |

**Decision rule:**

- **Shared-prefix workloads** (chatbots, RAG, agents) → SGLang.
- **Everything else in production** → vLLM.
- **TGI** is EOL — migrate now.
- **Ollama** for laptop-only dev.
- **TensorRT-LLM** only if you have a single long-lived model and the engineering budget.

## §7 The 2026 frontier — disaggregated serving

Frontier stacks (DeepSeek deploy, SGLang disaggregated, NVIDIA Dynamo) split prefill and decode onto separate GPU pools. Each pool is right-sized for its bottleneck — prefill nodes for compute, decode nodes for bandwidth. The KV cache produced by prefill is shipped to decode over fast interconnect.

This is overkill for most workloads. It pays off when:

- High request rate (>100 req/s).
- Long contexts (KV transfer cost is amortized over many decode steps).
- You can afford the engineering complexity of two GPU pools.

For typical 2026 production, single-pool vLLM or SGLang is the right answer.

## Interview Questions

**Q1: When would you choose SGLang over vLLM?**

When your workload has shared prefixes — multi-turn conversations, RAG pipelines where many queries reference the same documents, or agentic systems that prepend the same tool definitions to every request. SGLang's RadixAttention keeps the KV cache in an LRU radix tree and reuses it when new requests share a prefix. The 2026 benchmark gap is 29% higher throughput (16,200 vs 12,500 tok/s on H100). For unique prompts with no overlap, the cache adds overhead without benefit, and vLLM is preferable.

**Q2: Why has TGI been end-of-lifed?**

HuggingFace's Text Generation Inference was a strong serving framework from 2023-2024 but had design choices that did not keep pace with vLLM's PagedAttention and SGLang's RadixAttention. By late 2025, vLLM and SGLang were 2-3× faster on most workloads, and HuggingFace officially recommended migrating. TGI is in maintenance mode as of December 2025; existing deployments should switch to vLLM (for general workloads) or SGLang (for shared-prefix workloads).

**Q3: When is TensorRT-LLM worth the engineering pain?**

When you are deploying ONE model long-term in production on NVIDIA hardware and you need every last ~10% of throughput. The 10-30 min ahead-of-time compile step pays off only if you do not switch models often. For most teams that iterate on which open-weight model to use, vLLM's quicker turnaround is worth more than TensorRT-LLM's slight performance edge. The gap has also narrowed in 2026 — vLLM with all optimizations on lands within ~10% of TensorRT-LLM for most workloads.

**Q4: Can you serve a production LLM with Ollama?**

No, not at any meaningful scale. Ollama flatlines at ~22 req/s regardless of concurrency. It is designed for single-user local development and quick experimentation, not multi-tenant serving. Prototype on Ollama, ship on vLLM or SGLang. The migration is straightforward — both support GGUF if you want to keep the same quantization format, and both have OpenAI-compatible APIs.

**Q5: What is disaggregated serving and when does it pay off?**

Split prefill and decode onto separate GPU pools. Each pool is right-sized for its bottleneck: prefill nodes have high compute (FLOPS) and fewer GPUs; decode nodes have high HBM bandwidth and lots of VRAM. The KV cache produced by prefill is shipped to decode over fast interconnect. Pays off when request rate is high enough (>100 req/s) that the decode stall from new prefills would otherwise dominate, and when KV transfer cost amortizes over many decode steps (long contexts). Frontier technique; overkill for typical production. Single-pool vLLM or SGLang is right for most teams.
