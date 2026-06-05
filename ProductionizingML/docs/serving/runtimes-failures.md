# Runtimes and Failure Handling

The last serving decisions: where the model runs (cloud or edge), which runtime drives it, and what happens when something breaks. A serving design without a fallback story is not production-ready, so this page ends on the synthesis answer.

!!! tip "Rapid Recall"
    Cloud serving centralizes control (update, monitor, secure, scale) and is the default; edge serving moves inference close to the user for latency, offline use, or privacy, under severe constraints. A runtime is the engine that loads the model, schedules work, chooses kernels, and exposes inference: the correct interview move is to state the requirement first, then name the runtime. Failures (invalid input, missing or stale features, timeouts, GPU OOM, cold start, overloaded queue, bad version) each need a policy, and the mitigations are validation, timeouts, circuit breakers, budgeted retries, fallback features, smaller fallback models, rules fallback, manual review, shadow, canary, rollback, and version pinning. Always state what happens when the model or feature store fails.

## §1 Cloud vs Edge

Cloud serving centralizes control. Edge serving moves inference close to the user.

Cloud is easier to update, monitor, secure, and scale. It is the default for many products because model versions can be changed centrally and logs are available. Edge inference is useful when network latency, offline operation, privacy, or bandwidth dominates. Examples include wake-word detection, keyboard suggestions, camera models, mobile personalization, and local LLMs.

Edge constraints are severe: memory, compute, battery, thermals, device diversity, update rollout, and weak observability. Runtimes include TensorRT for NVIDIA devices, ONNX Runtime for portability, TFLite for mobile/embedded, Core ML for Apple devices, and GGUF/llama.cpp for local open-weight LLMs.

## §2 Serving Runtimes in 2026

A runtime is the engine that loads the model, schedules work, chooses kernels, and exposes inference.

| Runtime/platform | Use when | Conceptual role |
|---|---|---|
| Triton Inference Server | You need multi-framework CPU/GPU serving | Model repository, dynamic batching, concurrent model instances, multiple backends |
| ONNX Runtime | You want portable optimized inference | Runs ONNX graphs with execution providers such as CPU, CUDA, TensorRT, ROCm/MIGraphX |
| TensorRT / TensorRT-LLM | You are optimizing NVIDIA GPU inference | Graph optimization, kernel fusion, quantization, LLM-specific execution |
| vLLM | You serve high-throughput LLM APIs | PagedAttention, continuous batching, OpenAI-compatible serving |
| SGLang | You serve structured LLM/agent workloads | Runtime optimizations for prompt programs, prefix/KV reuse patterns |
| KServe / BentoML / Ray Serve | You need deployment, autoscaling, packaging, routing | Wrap runtimes into production services |

The correct interview move is to explain the requirement first, then name a runtime. "For LLM serving with high concurrency, I need continuous batching and KV-cache management, so vLLM or TensorRT-LLM behind KServe/Triton could fit" is much stronger than "use vLLM."

## §3 Failure Handling

A serving system must fail in a way the product understands.

Failures include invalid input, missing features, stale features, feature-store timeout, model-server timeout, GPU out-of-memory, cold start, overloaded queue, bad model version, dependency outage, and logging failure. Each needs a policy.

Common mitigations include request validation, timeouts, circuit breakers, retries with budgets, fallback features, smaller fallback models, rules-based fallback, manual review, shadow deploys, canaries, rollback, and model version pinning. For fraud, a feature timeout may route the transaction to manual review rather than approve blindly. For recommendations, fallback to popular items may be acceptable. The fallback depends on product risk.

!!! note "Serving design close"
    Always state what happens when the model or feature store fails. A design without fallback is not production-ready.

## §4 Interview Synthesis

> "I would choose serving mode from freshness and latency. For checkout fraud, I need online serving. The product service sends request fields to a model service, which validates input, fetches online features, preprocesses, runs a low-latency model, applies thresholds, logs prediction evidence, and returns approve/decline/review within the p99 budget. I would define a latency budget, monitor p95/p99 and feature freshness, use batching only if it does not hurt tail latency, use caching only where correctness allows, and deploy model versions through shadow and canary with rollback and fallback policies."

## Interview Questions

**Q1: When is edge serving worth its constraints?**
When network latency, offline operation, privacy, or bandwidth dominates: wake-word detection, keyboard suggestions, camera models, mobile personalization, local LLMs. You accept severe constraints (memory, compute, battery, thermals, device diversity, hard update rollout, weak observability) in exchange for locality. Otherwise cloud is the default because it is easier to update, monitor, secure, and scale.

**Q2: How should you talk about serving runtimes in an interview?**
State the requirement first, then name a runtime. Saying "for high-concurrency LLM serving I need continuous batching and KV-cache management, so vLLM or TensorRT-LLM behind KServe or Triton fits" is far stronger than "use vLLM." The runtime is the engine that loads the model, schedules work, picks kernels, and exposes inference, and the right one follows from the workload's needs.

**Q3: What failure modes must a serving design account for?**
Invalid input, missing or stale features, feature-store and model-server timeouts, GPU out-of-memory, cold starts, overloaded queues, bad model versions, dependency outages, and logging failures. Each needs an explicit policy, because a serving system must fail in a way the product understands rather than silently approving or hanging.

**Q4: What does a production-ready fallback look like for fraud versus recommendations?**
For fraud, a feature-store timeout should route the transaction to manual review rather than blindly approve, because the risk of a wrong approval is high. For recommendations, falling back to popular items is usually acceptable. The fallback depends on product risk, and stating it explicitly is what makes the design production-ready.
