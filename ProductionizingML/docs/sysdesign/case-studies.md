# Worked Case Studies

These are not complete production designs. They are interview-grade skeletons: requirements, high-level design, deep dive, and bottleneck discussion. The point is to practice explaining invariants and tradeoffs.

!!! tip "Rapid Recall"
    Four classic designs, each drilled the same way: requirements, high-level design, deep dive, bottlenecks. URL shortener is read-heavy, so cache the redirect, keep metadata correct in SQL, and queue analytics. A rate limiter needs atomic token-bucket state, often a Redis Lua script, with local buckets plus global reconciliation. A notification system keeps user-facing writes durable and pushes slow delivery to idempotent queue consumers. A key-value store leans on consistent hashing, replication, and tunable consistency. The whiteboard habit: always show read path, write path, storage choice, scale bottleneck, and failure fallback.

## §1 URL Shortener

**Requirements:** create short links, redirect quickly, prevent collisions, and record analytics. Reads dominate writes; redirect p99 matters more than creation latency.

**High-level design:** POST /urls writes original_url, code, owner, expiry to SQL with a unique code index. GET /{code} checks Redis, falls back to SQL, then redirects. Click analytics goes to Kafka/SQS so the redirect path is not blocked.

**Deep dive:** generate random base62 codes or hash+salt; enforce uniqueness with a database constraint. Cache popular codes with TTL. Analytics consumers aggregate per code into OLAP storage such as ClickHouse/BigQuery/Snowflake.

**Bottlenecks:** hot links, cache stampede, DB read pressure, malicious link creation, and analytics fan-out. Interview close: use cache for read-heavy redirect, SQL for metadata correctness, queue for analytics.

## §2 Rate Limiter

**Requirements:** enforce per-user or per-tenant quotas with bursts, low latency, and correctness good enough to protect downstream APIs.

**High-level design:** gateway calls a limiter before forwarding. Token bucket state stores remaining tokens and last refill time. Redis Lua script makes refill+spend atomic for distributed requests.

**Deep dive:** local in-memory buckets reduce Redis calls, but need periodic global reconciliation. Multi-region systems choose between strict global quota and low-latency approximate quota.

**Bottlenecks:** hot tenants, Redis outage, clock skew, retry storms. Interview close: rate-limit by the resource that matters, such as requests for REST APIs or tokens/GPU seconds for LLM APIs.

## §3 Notification System

**Requirements:** send push/email/SMS reliably, respect user preferences, handle priority, retries, batching, and provider failures.

**High-level design:** API persists notification intent, enqueues work by channel and priority, workers fan out to providers, and status updates return asynchronously.

**Deep dive:** idempotency keys prevent duplicate sends; dead-letter queues isolate poison messages; provider-specific rate limits require per-channel throttles; batching reduces cost for low-priority email.

**Bottlenecks:** fan-out storms, provider outages, preference lookups, retry amplification. Interview close: keep user-facing writes durable, move slow delivery to queues, and make consumers idempotent.

## §4 Key-Value Store

**Requirements:** low-latency get/put by key, large scale, replication, partitioning, and tunable consistency.

**High-level design:** client or router hashes keys to partitions, replicas store copies, writes go to a leader or quorum, reads choose leader, follower, or quorum depending on consistency requirement.

**Deep dive:** consistent hashing minimizes movement; virtual nodes balance load; compaction reclaims storage; hinted handoff or repair handles temporary replica failures.

**Bottlenecks:** hot keys, uneven partitions, replica lag, compaction stalls. Interview close: this is the backbone of distributed caches and online feature stores in the [Training](../training/index.md) and [Infrastructure](../infra/index.md) sections.

!!! note "Interview note"
    Whiteboard habit: always show the read path, write path, storage choice, scale bottleneck, and failure fallback. That is the difference between a memorized component list and a system design answer.

## §5 Practice Rubric

Use these prompts to drill the framework. Each one stresses a different invariant.

**Twitter / News Feed.** Drills fan-out, caching, ranking, eventual consistency, hot users, and queue-backed writes. Good answer explains fan-out-on-write vs fan-out-on-read and why celebrity users are special.

**Search Autocomplete.** Drills tries or prefix indexes, cache, popularity ranking, freshness, and memory footprint. Good answer separates online query serving from offline index building.

**Distributed Cache.** Drills consistent hashing, eviction, replication, hot keys, cache stampede, and failure fallback. Good answer admits that a cache miss path must still work, just slower.

**Uber / ETA.** Drills geo-indexing, streams, matching, freshness, and ML predictions. Good answer connects plain system design to the ML serving path used in later sections.

## Interview Questions

**Q1: Design a URL shortener. What is the core tradeoff?**
Reads dominate writes, so optimize the redirect path: GET /{code} checks Redis, falls back to SQL, then redirects, while creation writes metadata to SQL with a unique code constraint. Click analytics goes through Kafka or SQS so the redirect is never blocked. The main risks are hot links and cache stampede on popular codes.

**Q2: How do you make a distributed rate limiter correct?**
Store token-bucket state (remaining tokens and last refill time) and make refill-plus-spend atomic, typically with a Redis Lua script, so two gateway replicas cannot both spend the last token. Local in-memory buckets cut Redis calls but need periodic global reconciliation, and multi-region systems trade a strict global quota against a low-latency approximate one.

**Q3: What keeps a notification system reliable under provider failures?**
Persist the notification intent durably, then enqueue delivery by channel and priority so slow providers do not block the API. Idempotency keys prevent duplicate sends, dead-letter queues isolate poison messages, per-channel throttles respect provider limits, and batching reduces cost for low-priority email.

**Q4: What is the one whiteboard habit that signals a strong candidate?**
Always show the read path, write path, storage choice, scale bottleneck, and failure fallback. Walking that chain, rather than listing components, is what turns a memorized glossary into a defended design.
