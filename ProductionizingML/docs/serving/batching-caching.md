# Batching and Caching

Two of the biggest serving levers, and two of the easiest to misuse. Batching trades latency for throughput; caching trades freshness for speed. Both are only safe within limits set by the product's correctness needs.

!!! tip "Rapid Recall"
    Batching improves throughput by running several requests together, but it can hurt latency because requests wait for the batch to form, so in strict online serving the max batch wait must be tiny or adaptive. Queue depth tells you whether demand exceeds capacity: a growing queue is a warning, and backpressure means saying "slow down" before collapse. For LLMs, continuous batching lets new requests join as old ones finish, which is why vLLM-style serving matters. Caching is only safe when the cached value still means the same thing: feature caches can serve stale risk signals, prediction caches are dangerous when inputs include time or permissions, and LLM semantic caching must respect that similar language is not equivalent meaning, authorization, or freshness.

## §1 Batching, queues, and backpressure

Batching improves throughput by doing several requests together, but it can hurt latency because requests wait for the batch to form.

GPUs like large parallel work. If you send one tiny request at a time, the GPU may be underused. Dynamic batching waits for a short time, combines several requests, and runs them together. This can greatly improve throughput. But the waiting time counts against latency. If your p99 budget is strict, aggressive batching can make the system fail the product SLO.

The batch-wait tradeoff is direct: more waiting can increase batch size and GPU utilization, but it also adds latency before inference begins. In strict online serving, the max batch wait must be tiny or adaptive. A small wait of a few milliseconds may be invisible to users while still filling useful batches; a large wait can quietly blow a tight checkout budget.

Queue depth tells you whether demand exceeds serving capacity. A small queue can smooth bursts. A growing queue is a warning sign. If the queue keeps growing, you need autoscaling, traffic shedding, prioritization, fallback, or a cheaper model. Backpressure means the system says "stop" or "slow down" before it collapses.

In LLM serving, batching is more complicated because each request generates a different number of tokens. Continuous batching allows new requests to enter as old requests finish. This keeps GPUs busy during token generation and is one reason vLLM-style serving became important.

## §2 Caching in ML serving

Caching saves repeated work, but ML caching is only safe when the cached value still means the same thing.

A feature cache stores recently fetched features. It helps when the same user or entity is requested repeatedly. But if the model needs very fresh risk signals, stale feature cache can cause wrong predictions.

A prediction cache stores outputs for identical inputs. This can work for deterministic, stable requests. It is dangerous when inputs include time, user permissions, or rapidly changing state. A fraud score from 30 minutes ago may be unsafe if the account just had 20 failed logins.

An embedding cache stores expensive representations. If product text or documents do not change often, embeddings can be precomputed or cached. For RAG or semantic search, caching embeddings and retrieval results can reduce cost.

LLM prompt caching reuses work for repeated prompt prefixes, such as a long system prompt. Semantic caching reuses an answer for a similar prompt. Semantic caching must be treated carefully: similar language is not the same as equivalent meaning, authorization, or freshness.

## Interview Questions

**Q1: Why can batching hurt a strict online SLO?**
Because batching raises throughput by waiting to combine several requests, and that wait counts against latency. On a GPU, larger batches improve utilization, but if the p99 budget is tight, the time spent forming the batch can push the path past the SLO. That is why online batch wait must be tiny or adaptive rather than aggressive.

**Q2: What does a growing queue tell you, and what is backpressure?**
A small queue smooths bursts, but a steadily growing queue means demand exceeds serving capacity. The response is autoscaling, traffic shedding, prioritization, fallback, or a cheaper model. Backpressure is the system signaling "stop" or "slow down" to upstream callers before it collapses under load.

**Q3: Why is continuous batching important for LLM serving?**
Because LLM requests generate different numbers of tokens, so a fixed batch would stall waiting for the slowest sequence. Continuous batching lets new requests join as old sequences finish, keeping the GPU busy throughout token generation. That utilization gain is a major reason vLLM-style serving became important.

**Q4: When is a prediction cache dangerous?**
When the inputs include time, user permissions, or rapidly changing state. A cached fraud score from 30 minutes ago can be unsafe if the account just had 20 failed logins. Caching is only safe when the cached value still means the same thing, which is also why LLM semantic caching is risky: similar language is not equivalent meaning, authorization, or freshness.
