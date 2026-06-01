# Caching and Routing

The two operational levers that cut LLM API costs by 70-80% in 2026. Semantic caching serves "similar but not identical" queries from a vector cache; model routing sends each request to the cheapest model that can handle it. Both pay back in weeks. Critical distinction: **semantic cache** (application-level, you build it) is not the same as **prompt cache** (provider-level, automatic KV-cache reuse on identical prefixes).

!!! tip "Rapid Recall"
    **Semantic cache:** embed the incoming query, vector-search the cache (Redis, Pinecone, pgvector), return cached response if top-1 cosine > threshold (~0.92). Stack: embed (10-50 ms) → vector search → similarity check → serve cached (~30 ms total) or forward to LLM (~1-3 s) + store. Customer-support FAQs: 40-70% hit rate. Creative tasks: <5%. **Prompt cache** is provider-level (OpenAI, Anthropic): caches the KV cache of repeated system prompts on their side, gives 50-90% discount on cached input tokens. Different mechanism. **Routing:** classify request difficulty → cheap (Haiku, Flash) → medium (Sonnet, Gemini Pro) → expensive (Opus, GPT-5.4 Pro) → reasoning (R1, Deep Think). **Reported savings: 50-80% cost reduction with similar quality on mixed workloads.** Routing + semantic cache stack to 70-80%.

## §1 Semantic caching

### 1.1 The simple story

Your users keep asking similar questions. "What is your refund policy?" "How do I return something?" "Can I get my money back?" — all the same question dressed differently. Instead of sending each to the LLM (paying full freight), **compute an embedding of the incoming question, look it up in a vector cache, and return the previous answer if similarity is above a threshold**.

### 1.2 Mental model

- Traditional cache: exact-string match. "What is your refund policy?" = cache hit only for the exact same string.
- Semantic cache: vector similarity match. "refund policy" / "get money back" / "return item for refund" all map to the same cache entry.

### 1.3 The stack

1. Request comes in.
2. Compute embedding of the request (fast — 10-50 ms for a small embedding model).
3. Nearest-neighbor search in vector DB (Redis, Pinecone, Weaviate, pgvector).
4. If top-1 similarity > threshold (e.g., 0.92 cosine): return cached response.
5. Otherwise: forward to LLM, compute response, store (query_embedding, response) in cache.

### 1.4 Typical gains

- **Customer support FAQs:** 40-70% cache hit rate.
- **Code assistant:** 20-30% cache hit rate.
- **Creative tasks:** <5% cache hit rate (do not bother caching).
- **Cost reduction:** proportional to hit rate × LLM cost.
- **Latency:** cache hits return in ~30 ms vs 1-3 s LLM calls.

### 1.5 The hard parts (interview gotchas)

1. **Similarity threshold calibration.** Too low → wrong answers served from cache. Too high → no hits. Needs per-domain tuning.
2. **Personalization.** Different users get different answers even for the same query. Solution: include user attributes in the cache key or maintain per-user caches.
3. **Temporal invalidation.** Product policies change. Need TTL + versioning on cache entries.
4. **Long-tail queries.** 80% of unique queries get 0 hits. Cache only helps the head.
5. **Adversarial inputs.** Someone could probe to find cache collisions and extract other users' data.

### 1.6 2026 tools

- **GPTCache** (open-source, most popular).
- **Redis Vector Search** with semantic caching extensions.
- **Anthropic's prompt caching** (different thing — see below).
- **OpenAI's prompt caching** (also a different thing).

## §2 Prompt cache vs semantic cache

!!! warning "Two different concepts with the same word in the name"
    **Semantic cache (application-level, you build it):** embed incoming queries, vector-match against past responses. You implement this with GPTCache or roll your own. Returns full responses.

    **Prompt cache (provider-level, automatic):** OpenAI, Anthropic, and others cache the KV cache of repeated system prompts on their side. 50-90% cost discount on cached *input* tokens. Different mechanism, different layer of the stack, complementary not substitute.

Most production systems use both. Prompt cache reduces the cost of every request that shares a system prompt; semantic cache reduces the count of requests that hit the LLM at all.

## §3 Model routing

### 3.1 The simple story

Route every user request to the cheapest model that can handle it acceptably. Simple queries go to Haiku 4.5 at $0.25/M tokens. Medium queries go to Sonnet 4.6 at $3/M. Hard queries escalate to Opus 4.7 at $15/M. The routing decision itself is made by a small classifier model or a heuristic.

### 3.2 Mental model

You are building a hierarchical decision system where each tier has better quality but higher cost. The goal is to classify the request into the cheapest tier that will succeed.

Three routing strategies:

1. **Pre-routing (cascade).** Classify query → pick model → generate → done. Fastest. Risk: classifier is wrong.
2. **Post-routing (fallback).** Try cheap model → if confidence low / output validation fails → escalate to expensive. Higher latency on escalations. Safer.
3. **Hybrid.** Pre-route + post-validate with fallback. Most production systems in 2026.

### 3.3 Typical routing architecture

```
User Request
     ↓
[Classifier: "how hard is this?"]
     ↓
├── Simple (1-3):    → Haiku 4.5 / Gemini Flash / Phi-4 ($0.05-1/M)
├── Medium (4-6):    → Sonnet 4.6 / GPT-5 mini / Gemini 3.1 Pro ($3-10/M)
├── Complex (7-9):   → Opus 4.7 / GPT-5.4 / Gemini 2.5 Deep Think ($5-25/M)
└── Reasoning-heavy: → o3 / DeepSeek-R1 / Claude Opus 4.7 thinking
     ↓
[Confidence check]
     ↓
If confidence < threshold → escalate to next tier
Else → return response
```

### 3.4 Reported savings

Routing vs "always use top-tier": **50-80% cost reduction** with similar quality on mixed workloads. This is the #1 infrastructure optimization for AI-first products in 2026.

### 3.5 When routing fails

- **Legal / medical contexts** where getting it "mostly right" is unacceptable — use top tier every time.
- **Multi-turn conversations** — routing mid-conversation causes tone shifts.
- **Agent chains** — errors compound across hops; one bad route poisons downstream reasoning.

### 3.6 Tools for routing

- **Martian, OpenRouter, Portkey, LiteLLM** — routing as a service.
- **Claude's native routing.** GPT-5.4 also does internal routing now — the model decides fast vs thinking automatically.
- **Custom**: embedding + classifier → model dispatch.

## §4 The combined stack

```
Incoming query
     ↓
[Semantic cache lookup]  → hit  → return (skip LLM entirely)
     ↓ (miss)
[Router classifier]
     ↓
[Selected model with prompt-cache enabled]
     ↓
Response → store in semantic cache → return
```

Provider-level prompt caching is on by default for system prompts (just send them). Semantic cache is something you implement. Routing is something you decide on.

Reported aggregate savings:

- Routing alone: 50-70%.
- Plus semantic cache: 70-80%.
- Plus prompt caching: 75-85%.

## Interview Questions

**Q1: If asked "how would you reduce our AI API costs?", what is the answer?**

Semantic caching + routing before anything else. Routing alone often captures 50%+ savings. Adding semantic caching captures another 20-40%. Provider-level prompt caching (50-90% off on repeated prefixes) is free if you structure system prompts consistently. Tertiary optimization: moving high-volume workloads to self-hosted vLLM with quantized open-weight models.

**Q2: What is the difference between semantic cache and prompt cache?**

Semantic cache is application-level — embed the incoming query, vector-search against past queries, serve the cached response if similarity exceeds a threshold. You build this with GPTCache or roll your own. Returns full responses. Prompt cache is provider-level — OpenAI/Anthropic cache the KV cache of repeated system prompts on their side, giving you 50-90% discount on cached input tokens. Different layers of the stack, complementary not substitute. Most production stacks use both.

**Q3: When would semantic caching fail in production?**

Five places. (1) Similarity threshold too low → wrong answers served. (2) Personalization missed → user A gets user B's cached answer. (3) Stale entries → product policy changed but cache did not invalidate. (4) Long-tail queries → 80% of unique queries get 0 hits. (5) Adversarial probing → someone discovers a collision and extracts other users' data. Mitigations: per-domain threshold tuning, include user attributes in the cache key, aggressive TTL on policy-sensitive content, accept that the tail will not cache, and monitor for adversarial query patterns.

**Q4: Architect a 3-tier routing system and explain expected cost savings.**

Three tiers: simple (Haiku 4.5 / Gemini Flash, $0.25-1/M output), medium (Sonnet 4.6 / Gemini 3.1 Pro, $3-15/M), complex (Opus 4.7 / GPT-5.4, $25-180/M). A small classifier (BERT-tier or a cheap LLM call) labels each request 1-9; route to the corresponding tier. Add a confidence check: if the chosen tier returns low confidence, escalate. For a typical chat workload with 60% simple / 30% medium / 10% complex queries, this drops the average per-query cost from $15/M (all-Opus) to ~$5/M, a 67% reduction with negligible quality loss because the easy questions never needed Opus.

**Q5: Why is routing risky in multi-turn conversations or agent chains?**

Two failure modes. Multi-turn: switching from Haiku to Opus mid-conversation causes tone shifts and the user perceives the model as inconsistent or even hallucinating context. Agent chains: errors compound across hops, so one bad route in the middle poisons the downstream reasoning that depends on its output. Mitigation: route at the conversation level (entire session uses the same tier based on initial classification) or build agent chains that re-check at each hop. For agentic workloads in 2026, the trend is "use the most capable model for the agent loop" since the cost of one wrong hop exceeds the savings.

**Q6: What is the lasting tradeoff between semantic caching and personalization?**

Semantic cache assumes similar queries should get the same answer. Personalization assumes different users should get different answers to the same query. These are in direct tension. Three resolutions. (1) Per-user cache (no cross-user sharing) — preserves personalization, kills most cache benefit. (2) Cache the public part, personalize the wrap (cache "general refund policy," personalize the greeting and account-specific bits at the edge). (3) Skip caching entirely for personalized domains; only cache where the answer is genuinely user-independent (FAQs, general knowledge). Most production systems do (2).
