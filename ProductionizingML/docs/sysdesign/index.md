# System Design Foundations

System design is not memorizing Redis, Kafka, and Cassandra. It is the discipline of deciding where computation, state, latency, and failure live so a product remains correct, fast, and operable as scale changes. This section is the foundation layer the rest of the site builds on, and it deliberately goes deeper than a glossary because interviewers reward a defended chain of reasoning, not component name-dropping.

!!! tip "Rapid Recall"
    The interview is a simulation of engineering judgment under incomplete requirements. Work in four moves: separate functional from non-functional requirements, do back-of-envelope scale math, draw the read and write paths, then deep-dive the riskiest subsystem and name its metrics and failure fallback. A strong answer optimizes the path that matters first, justifies each component by the access pattern, and closes on the main risk plus its mitigation. Cyan marks the plain software mechanism; amber marks the ML extension later pages build on.

## §1 What System Design Interviews Actually Test

The interview is a simulation of engineering judgment under incomplete requirements. You are not expected to design the perfect real production system in 45 minutes. You are expected to ask the right questions, choose a reasonable architecture, explain tradeoffs, and notice where the design can fail.

When you finish this section, you should be able to draw a non-ML service, explain why each component exists, estimate rough scale, name the likely bottleneck, and describe the fallback when a dependency fails. This page deliberately goes deeper than a glossary because interviewers do not reward component name-dropping. They reward a defended chain of reasoning.

### 1. Requirements

Separate functional requirements from non-functional requirements. Functional: shorten a URL, send a notification, store a key. Non-functional: p99 latency, QPS, durability, privacy, availability, freshness, cost, regional constraints. If you skip this, every later component choice is ungrounded.

**Good question:** "Is the read path or write path more important?"

### 2. Back-of-envelope scale

Estimate enough to choose shape: QPS, read/write ratio, payload size, storage growth, peak multiplier, cache hit-rate target, and retention. You do not need perfect math; you need numbers that expose whether one database is enough or whether partitioning, queues, and caches matter.

**Example:** 10k writes/sec at 1 KB is about 864 GB/day before replication and indexes.

### 3. High-level design

Draw the read path and write path. The boxes are less important than the flows: synchronous request/response, asynchronous side effects, storage writes, cache lookups, and observability events. A clean simple diagram beats a pile of distributed systems buzzwords.

### 4. Deep dive and bottlenecks

Pick the riskiest subsystem and go deeper: cache invalidation, rate limiting correctness, queue retries, shard key choice, data consistency, failure recovery, or p99 latency. Close by naming metrics and runbooks.

!!! note "Interview note"
    A strong answer sounds like: "Given the product goal and QPS, I would optimize the read path first, use a cache because the access pattern is skewed, keep writes durable in SQL, push analytics to a queue, and monitor p99 latency plus cache hit rate. The main risk is stale cache entries, so I would use TTL plus event-driven invalidation for critical updates."

## Where to go next

- [The Standard Request Path](request-path.md) walks one request across every boundary it crosses.
- [The 12 Building Blocks](building-blocks.md) explains each component from the problem it solves.
- [Latency and Capacity](latency-capacity.md) covers tail behavior and capacity math.
- [Worked Case Studies](case-studies.md) drills the framework on four classic designs.

## Interview Questions

**Q1: What is a system design interview actually testing?**
Engineering judgment under incomplete requirements. You are not expected to design a perfect production system in 45 minutes. You are expected to ask the right questions, choose a reasonable architecture, explain tradeoffs, and notice where the design can fail.

**Q2: What is the first thing you do before choosing any component?**
Separate functional requirements (what the system does) from non-functional requirements (p99 latency, QPS, durability, privacy, availability, freshness, cost, regional constraints). If you skip this, every later component choice is ungrounded. A good clarifying question is whether the read path or the write path matters more.

**Q3: Why do back-of-envelope numbers matter if they are not exact?**
You do not need perfect math; you need numbers that expose the shape of the problem, whether one database is enough or whether partitioning, queues, and caches matter. For example, 10k writes/sec at 1 KB is about 864 GB/day before replication and indexes, which immediately tells you a single unpartitioned store will struggle.

**Q4: How should you close a deep dive?**
Pick the riskiest subsystem, go deeper on its correctness and failure behavior, then name the metrics and runbooks that would catch a regression. Naming p99 latency, cache hit rate, and the fallback when a dependency fails is what separates a system design answer from a memorized component list.
