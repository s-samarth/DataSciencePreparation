# 08. Multi-Tenant LLM Serving Platform

**Company tags:** Cloud providers, model platforms (Together, Fireworks, Anyscale), AI infra startups, large enterprises
**Interview frequency:** High for LLM / platform / infra roles
**Why it matters:** This is the case that separates "I call the OpenAI API" candidates from engineers who understand what happens *inside* the GPU. The whole interview lives in three facts: **tokens are the unit of cost, the KV cache is the memory bottleneck, and batching is what turns idle GPUs into throughput.** Get those and you can derive the rest.

---

## 0. How to use this doc

Built two ways; read it twice.

1. **As a thinking guide.** The headers are the whiteboard order. This case is *systems-first*, so the "model" section is about inference mechanics, not training.
2. **As a worked transcript.** Section 11 is a full timestamped hour. Cover the `YOU:` lines and answer from memory.

The one idea to carry out: **LLM serving is a throughput-vs-latency scheduling problem under a hard memory constraint (the KV cache), not a modeling problem. The two latencies you control — time-to-first-token (prefill) and time-per-output-token (decode) — come from two physically different phases, and continuous batching + paged KV cache are how you keep the GPU busy without blowing the memory budget.** Say that and the interviewer knows you've operated inference, not just consumed an API.

A note of honesty that scores points: there is very little "ML" here. The only learned decision is *routing* (pick the cheapest model that meets the quality bar). The rest is distributed systems and GPU memory management. Naming that reframing up front is a senior signal.

Scaffold (adapted for an infra case):

```
Clarify -> Frame -> Workload/Routing -> Baseline -> Serving internals -> Eval/Load -> Deploy -> Monitor
```

---

## 1. The reusable scaffold, stated once

| Phase | The question |
|---|---|
| Clarify | What models, what tenants, what SLOs, what isolation guarantees? |
| Frame | What's actually learned (routing) vs engineered (everything else)? |
| Workload / Routing | What does the traffic look like, and how do we route it? |
| Baseline | Simplest shippable serving stack, and what breaks it? |
| Serving internals | KV cache, batching, prefill/decode — explained to the floor. |
| Eval / Load | Latency/throughput frontier; load testing; the offline/online gap. |
| Deploy | Three paths; rollout; capacity planning. |
| Monitor | What pages someone; the fallback. |

---

## 2. Clarify requirements (scripted)

| Question | Why it changes the design |
|---|---|
| "How many distinct models and sizes? Open-weights we host, or third-party APIs too?" | Hosting our own (Llama-70B on our GPUs) makes KV cache and batching *our* problem. Proxying an API makes it a routing/quota/caching problem. Most platforms do both; the hard part is the self-hosted models. |
| "Interactive (chat, streaming) or batch (offline summarization)?" | Interactive optimizes TTFT and per-token latency. Batch optimizes pure throughput and can tolerate huge batches. They have opposite scheduling goals, so I'd run them on separate pools. |
| "What isolation between tenants — soft (fair-share) or hard (dedicated capacity / data isolation)?" | Hard isolation (a regulated tenant who can't share GPUs) changes capacity planning entirely. Soft isolation is a fairness-scheduling problem. And data isolation (no cross-tenant prompt/cache leakage) is a non-negotiable security boundary regardless. |
| "What are the latency SLOs, and are responses streamed?" | Streaming means TTFT is what users feel; full-response latency matters less. This sets whether I optimize prefill or total tokens. |
| "Is there a quality bar per task, and do we have eval data to route on?" | Routing to a cheaper model is only safe if I can measure that it still meets the task's quality bar. No eval data → no safe dynamic routing, only static routing. |
| "What's the cost target and the GPU budget?" | GPUs are the scarce resource. The whole design is "maximize useful tokens per GPU-second subject to SLOs." This sets utilization targets. |

**Numbers I'll commit to and carry through:**

- **Tenants:** ~30 internal product teams sharing the platform.
- **Models:** a small (8B), a large (70B) self-hosted, plus a frontier third-party API for the hardest tasks.
- **Traffic:** ~50K requests/min peak, mixed prompt sizes (200-8000 tokens) and output sizes (50-1000 tokens).
- **Hardware:** H100-80GB class GPUs.
- **SLOs (interactive):** TTFT p95 < 500ms, time-per-output-token (TPOT) < 50ms (≈ 20 tok/s, faster than reading speed).
- **Utilization target:** > 60% GPU utilization (idle GPUs are the main cost leak).
- **Isolation:** soft fair-share by default + hard data isolation always; dedicated pools for a couple of regulated tenants.

### The KV-cache memory budget, derived out loud (the heart of this case)

This is the calculation that proves you understand serving. KV cache per token:

```
bytes/token = 2 (K and V) x n_layers x n_kv_heads x head_dim x precision_bytes
```

For a 70B model (≈80 layers, GQA with 8 KV heads, head_dim 128, fp16):

```
2 x 80 x 8 x 128 x 2 = ~320 KB per token
```

A single 4,000-token conversation therefore holds **~1.3 GB** of KV cache. Now the budget on one 80GB H100:

```
70B weights in fp16 = ~140 GB  -> doesn't fit on one GPU; needs 2 (tensor parallel)
2 x H100 = 160 GB total; weights eat 140 -> ~20 GB left for KV cache
20 GB / 1.3 GB per 4K sequence  -> ~15 concurrent long sequences
```

That number — **~15 concurrent sequences** — is the entire serving problem in one figure. The KV cache, not compute, caps concurrency. Everything (quantization, PagedAttention, GQA, batching) is in service of fitting more sequences into that 20GB. State this and you've won the technical credibility battle.

### Throughput / cost back-of-envelope

50K req/min × avg ~400 output tokens = 20M output tokens/min ≈ **330K tok/s** aggregate. If one 2-GPU 70B worker does ~2-3K tok/s at good batch sizes, that's ~100+ worker-pairs at the 70B tier alone — which is exactly why routing cheap traffic to the 8B model and caching repeats is a cost necessity, not a nicety.

---

## 3. Frame the problem

- **Framing:** a request scheduler over a pool of model workers, maximizing useful tokens per GPU-second subject to per-tenant SLOs, quotas, isolation, and a quality bar — with one learned component: the router.
- **What's learned vs engineered:** the **router** ("which model is the cheapest that still meets this task's quality bar?") is the only ML-ish decision, and even it is often a classifier or rule set over prompt features. Everything else — batching, KV management, quotas, fallback — is systems engineering.
- **Why this framing wins:** it stops you from over-claiming ML and focuses you on the real bottleneck (GPU memory and scheduling). Interviewers for this role want an engineer, not a model-namer.
- **Non-ML / non-platform baseline:** a single shared model endpoint behind a rate limiter. Ships in a day. Breaks the moment two tenants have different SLOs or one floods the queue.

---

## 4. Workload characterization and routing

You can't schedule what you haven't characterized. The "data" here is the **traffic shape**, not labels.

- **Key request features:** tenant, prompt length, expected output length (estimated), task type, SLA tier, safety class, cache key. Prompt and output length are the most important — they predict KV-cache footprint and latency, which drive scheduling.
- **The routing decision:** map each request to the cheapest model that meets its quality bar.
  - **Static routing:** by tenant/task config — "team X's summarization always uses the 8B." Simple, predictable, can overpay.
  - **Dynamic routing:** a small classifier on prompt features predicts difficulty and routes easy → 8B, hard → 70B, hardest → frontier API. Needs per-task eval data to set the thresholds safely, and a quality monitor to catch mis-routes.
- **Routing labels:** come from offline eval (the case-06 platform) — run a sample of each task's traffic through every model, score quality, and learn the cheapest model that clears the bar. Without that eval data, you cannot route dynamically without risking silent quality drops.
- **Bias/risk to name:** routing optimizes a *measured* quality proxy; if the proxy diverges from real user value, you'll cheerfully route everyone to the cheap model and quietly degrade. Monitor live quality per route.

---

## 5. Baseline -> why it breaks -> next rung

| Rung | What it is | Why it breaks -> trigger to climb |
|---|---|---|
| 0 | Single shared model endpoint + global rate limit. | No tenant isolation, one noisy tenant starves everyone, one model can't fit all SLO/quality needs. Trigger: second tenant with a different SLO. |
| 1 | Gateway + auth/quota + **static** routing to a few model endpoints. | Naive request batching wastes GPU; long prompts block short ones (head-of-line); overpays by sending easy traffic to big models. Trigger: low GPU utilization and tail-latency complaints. |
| 2 | **Continuous batching + PagedAttention pooled serving (vLLM/TGI-style) + semantic cache + dynamic routing + priority/fair-share scheduling.** | Production default. Trigger to extend: extreme latency needs or huge scale. |
| 3 | Disaggregated prefill/decode, speculative decoding, multi-LoRA serving, prefix-cache sharing, KV-cache offload/tiering. | More operational complexity. Trigger: measured bottleneck that rung 2 can't fix (e.g., prefill starving decode, or many fine-tunes of one base model). |

Earn rung 2 by explaining *why* naive batching wastes the GPU (next section). Rung 3 items are the "I've actually tuned this" extensions you name when pushed.

---

## 6. Serving internals, explained to the floor

### Prefill vs decode — two physically different phases

A generation request has two phases with opposite hardware profiles:

- **Prefill:** process the whole prompt in parallel, build the KV cache, emit the first token. This is **compute-bound** (big matrix multiplies, GPU-saturating) and its cost scales with *prompt* length. Prefill latency ≈ TTFT.
- **Decode:** generate output tokens one at a time, each attending to the whole KV cache. This is **memory-bandwidth-bound** (you stream the whole KV cache + weights per token, doing little compute), and its cost scales with *output* length. Decode latency ≈ TPOT per token.

Why this matters: a long prompt hurts TTFT (prefill); a long answer hurts total time (decode). And because prefill is compute-heavy while decode is bandwidth-heavy, mixing them naively means a big prefill can stall the steady stream of decodes for other users — head-of-line blocking. (Rung 3's prefill/decode *disaggregation* puts them on separate pools to fix exactly this.)

### Continuous (in-flight) batching — why naive batching wastes the GPU

Naive "static" batching waits to collect N requests, runs them together, and can't return until the *longest* one finishes — so a batch of mostly-short answers idles while one 1000-token answer drains, and new requests wait for the whole batch. **Continuous batching** instead schedules at the *token* level: as soon as any sequence in the batch finishes, its slot is freed and a waiting request joins mid-flight. This keeps the GPU saturated and is the single biggest throughput win in modern serving (the vLLM contribution). Throughput can rise several-fold over static batching at the same latency.

### PagedAttention — why the KV cache needs an OS-style allocator

The KV cache is the scarce resource (section 2: ~15 sequences fit). Naive serving pre-allocates a contiguous max-length buffer per sequence, so a request that *might* produce 2000 tokens reserves 2000 tokens of cache even if it stops at 50 — massive internal fragmentation, and you can't fit nearly as many sequences as the math allows. **PagedAttention** treats KV cache like virtual memory: store it in fixed-size *blocks* (pages), allocated on demand as the sequence grows, with a block table mapping logical positions to physical blocks. This near-eliminates fragmentation, lets you pack far more concurrent sequences into the same 20GB, and enables **prefix sharing** — two requests with the same system prompt can share the physical KV blocks for that prefix, a big win when every tenant prepends a long instruction.

### The levers, and what each trades

- **Quantization (fp16 → int8/fp8/int4 weights):** shrinks weights, freeing GPU memory for more KV cache (more concurrency) and speeding bandwidth-bound decode — at a small, *measurable* quality cost. Always validate quality, never assume it's free.
- **GQA (grouped-query attention):** fewer KV heads → smaller KV cache per token (it's already in the 70B math above) → more concurrency. A model-architecture lever.
- **Speculative decoding:** a small draft model proposes several tokens, the big model verifies them in one parallel pass; accepted tokens come "for free," cutting TPOT. Helps latency, costs complexity and some GPU.
- **Semantic cache:** cache responses keyed by prompt embedding (not just exact match) so near-duplicate requests skip generation entirely. Huge cost saver on repetitive enterprise traffic; needs careful invalidation and a similarity threshold tuned to avoid serving a stale/wrong cached answer.
- **Multi-LoRA serving:** serve many fine-tuned adapters over one shared base model in memory, instead of one full copy per tenant — critical when 30 tenants each want "their" model.

### Multi-tenancy — fairness, isolation, and the noisy neighbor

- **Fair-share scheduling + priority queues:** without it, one tenant blasting long requests consumes all the KV slots and starves everyone (the noisy-neighbor problem). Weighted fair queueing per tenant, with priority tiers for latency-critical traffic, bounds each tenant's share.
- **Quotas and admission control:** per-tenant token/req-rate limits, enforced at the gateway, with backpressure (429s) rather than letting the queue explode. Set a **max output token** cap so one runaway request can't hold a KV slot forever.
- **Data isolation (non-negotiable):** no prompt, completion, or **cache** crosses tenant boundaries. The semantic cache must be partitioned per tenant (or per data-classification) or you leak one tenant's data to another via a cache hit — a classic and serious bug. Dedicated pools for tenants who require hard capacity isolation.

### Canonical references (verified)

- PagedAttention / vLLM — Kwon et al., 2023: https://arxiv.org/abs/2309.06180
- Orca, continuous (iteration-level) batching — Yu et al., 2022: https://www.usenix.org/conference/osdi22/presentation/yu
- Speculative decoding — Leviathan et al., 2022: https://arxiv.org/abs/2211.17192
- GQA (grouped-query attention) — Ainslie et al., 2023: https://arxiv.org/abs/2305.13245
- vLLM docs (Hugging Face inference engines): https://huggingface.co/docs/inference-endpoints/en/engines/vllm

---

## 7. Evaluation — the latency/throughput frontier and load testing

Serving isn't evaluated with accuracy; it's evaluated with a **frontier curve**: throughput (tokens/s per GPU) on one axis, latency (TTFT, TPOT) on the other. Higher batch size → more throughput but worse per-request latency. The job is to operate at the knee that meets SLOs at lowest cost.

- **Offline / pre-launch:** load tests that *replay realistic traffic shape* (the mix of prompt/output lengths matters more than raw QPS). Measure TTFT p50/p95/p99, TPOT, throughput/GPU, GPU utilization, and the saturation point where tail latency explodes.
- **Online:** SLO compliance per tenant and per tier, GPU utilization, cost per 1M tokens, cache hit rate, routing accuracy (did dynamic routing keep quality up?), error/timeout rate.
- **Quality guardrail:** per-route quality monitored live (via the case-06 eval platform) so quantization or aggressive routing can't silently degrade output.
- **Safety/isolation guardrails:** zero cross-tenant leakage (audited), quota-violation rate, unsafe-output rate.

### The offline-to-online gap, the serving flavor of the trap

**"Load tests passed at 50K req/min, but production fell over at 30K."** Causes, ordered:

1. **Synthetic traffic shape was wrong.** The load test used uniform 500-token prompts; real traffic has a fat tail of 8000-token prompts whose prefill stalls everything. *Workload shape, not average QPS, determines saturation.*
2. **Bursty arrivals.** Load tests send smooth traffic; real traffic spikes, and the KV cache fills during bursts, triggering preemption/recompute.
3. **Cache hit rate was inflated offline.** Synthetic prompts repeated more than real ones, so offline throughput assumed a cache hit rate production doesn't have.
4. **Tenant mix shifted.** One tenant's long-context feature launched and changed the average sequence length, shrinking concurrency.
5. **Routing drift.** Live prompts are harder than the routing classifier's training sample, so more traffic escalates to the expensive 70B than planned, exhausting capacity.

### One fully specified A/B test (or rather, canary)

Serving changes are validated by **canary + shadow**, but here's a crisp experiment:

- **Hypothesis:** enabling continuous batching + PagedAttention raises throughput/GPU without breaching TTFT/TPOT SLOs or quality.
- **Unit:** route a % of *traffic* (not users) to the new serving stack; shadow-mirror real requests first to compare outputs.
- **Arms:** control = static-batching workers; treatment = continuous-batching + paged KV workers.
- **Primary:** tokens/s per GPU (throughput) at fixed SLO.
- **Guardrails (auto-stop):** TTFT p95, TPOT p95, error/timeout rate, per-route quality, zero cross-tenant leakage.
- **Ramp:** 1 → 5 → 25 → 50% of traffic, watching tail latency at each step (averages lie; tails tell the truth in serving).
- **Runtime:** long enough to span a peak-traffic window and a couple of tenant batch jobs.
- **Rollback:** any SLO/quality/leakage breach → shift traffic back to control instantly (the gateway makes this a routing-table change).

---

## 8. Deployment — three paths

- **Serving path:** gateway (auth, quota, admission control) → router → scheduler/batcher → model workers (paged KV, continuous batching) → streaming response. Plus the semantic cache in front. Each layer scales independently.
- **Data path:** request/response telemetry — tenant, token counts, latencies, model used, cache hit, cost — into a metrics store powering per-tenant cost dashboards, capacity planning, and routing-classifier retraining. (No raw cross-tenant prompt sharing.)
- **Feedback path:** quality signals per route + load-test results feed routing thresholds, autoscaling policy, and capacity forecasts. Cost dashboards feed chargeback to tenants, which shapes their behavior (they stop over-requesting the 70B when they're billed for it).

### Rollout & capacity discipline

- **Model version / engine upgrades:** shadow → canary by traffic % → ramp, with instant rollback via the gateway routing table. Never flip an engine version for all traffic at once.
- **Autoscaling:** scale worker pools on queue depth and KV-cache pressure, not just CPU. GPU cold-start (loading 140GB of weights) is slow, so keep warm headroom and scale *ahead* of demand using forecasts — reactive autoscaling alone leaves you cold during a spike.

### Monitoring and fallback

- **What pages someone:** TTFT/TPOT SLO breach, KV-cache pressure → preemption/recompute spikes, GPU OOM, queue depth runaway, cross-tenant leakage (sev-0), cost-per-token spike, cache hit-rate collapse, a tenant exceeding quota and not being throttled.
- **Fallback ladder:** under overload, **shed gracefully** — serve from semantic cache, downgrade to the smaller model, return 429 with backpressure to low-priority tenants to protect high-priority SLOs, and degrade non-streaming batch jobs first. If a model worker pool is down, route to the frontier API as an expensive-but-available fallback. Protect the SLO of paying/critical tenants over best-effort ones.
- **Incident response:** the gateway routing table is the master kill-switch — shift traffic off a bad worker/version instantly. Diff engine/model/router versions, inspect per-tenant metrics to find the noisy neighbor, and roll back the offending layer.

---

## 9. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Core framing | "Host the model behind an API." | Throughput-vs-latency scheduling under a KV-cache memory cap; mostly systems, one learned router. |
| Bottleneck | "Need more GPUs / compute." | KV cache is the memory bottleneck; derives ~15 concurrent sequences from first principles. |
| Latency | "Make it fast." | Distinguishes TTFT (prefill, compute-bound) from TPOT (decode, bandwidth-bound). |
| Batching | "Batch requests." | Continuous/in-flight batching vs static; explains why static wastes the GPU. |
| Memory | "Use vLLM." | PagedAttention as virtual-memory for KV; fragmentation + prefix sharing. |
| Cost | "Use a smaller model." | Dynamic routing to cheapest-model-meeting-quality + semantic cache + quantization, each with its trade. |
| Multi-tenancy | "Add API keys." | Fair-share scheduling, quotas/admission control, and *cache partitioning* to prevent leakage. |
| Eval | "Measure latency." | Latency/throughput frontier; load-test with realistic traffic shape; tails not averages. |
| Offline/online | "Load test predicts prod." | Workload shape, bursts, cache-rate inflation, routing drift break the prediction. |
| Deploy | "Deploy and autoscale." | Scale on KV pressure, scale ahead via forecasts, gateway as kill-switch, shed load by priority. |

---

## 10. One-page whiteboard cheat sheet

```
NUMBERS: 30 tenants | 50K req/min | models 8B / 70B / frontier-API
         H100-80GB | TTFT p95<500ms | TPOT<50ms (20tok/s) | util>60%

KV-CACHE MATH (the money slide):
  bytes/tok = 2 x layers x kv_heads x head_dim x precision
  70B: 2x80x8x128x2 = ~320KB/tok -> 4K ctx = ~1.3GB/seq
  2xH100=160GB; weights 140 -> ~20GB KV -> ~15 concurrent seqs
  => KV CACHE, not compute, caps concurrency. Everything serves this.

TWO PHASES:
  prefill = parallel, COMPUTE-bound, scales w/ prompt len -> TTFT
  decode  = 1 tok/step, BANDWIDTH-bound, scales w/ output len -> TPOT

LADDER: single endpoint -> gateway+static route
        -> CONTINUOUS BATCHING + PAGEDATTENTION + sem-cache + dynamic route + fair-share
        -> prefill/decode disagg, spec decoding, multi-LoRA, prefix sharing

KEY MECHANICS:
  continuous batching: token-level scheduling, free slot mid-flight (vLLM)
  PagedAttention: KV as paged virtual memory -> no fragmentation + prefix sharing
  quantization: smaller weights -> more KV room + faster decode (validate quality!)
  GQA: fewer KV heads -> smaller cache | spec decoding: draft+verify -> lower TPOT

MULTI-TENANT: fair-share + priority queues (noisy neighbor)
              quotas/admission + max-output cap | DATA+CACHE isolation (leak=sev0)

EVAL: throughput-vs-latency FRONTIER | load-test REALISTIC SHAPE | tails not averages
OFFLINE-OK/ONLINE-BAD: wrong traffic shape | bursts | inflated cache rate
                       | tenant-mix shift | routing drift

DEPLOY: shadow->canary by traffic% | scale on KV pressure + forecast ahead
        gateway routing table = kill switch | shed load by priority
```

---

## 11. Full interview transcript (one hour)

**[00:00] INTERVIEWER:** Design a multi-tenant LLM serving platform for internal product teams.

**[00:30] YOU:** Let me clarify a few things that change the design. Are we hosting open-weights models on our own GPUs, proxying third-party APIs, or both? Interactive streaming traffic or batch? What isolation do tenants need — fair-share, or hard dedicated capacity and data isolation? And do we have eval data to route on?

**[01:00] INTERVIEWER:** Both — you host a Llama-70B and an 8B, and you can fall back to a frontier API. Mostly interactive chat. Soft isolation is fine but no data can leak between tenants. Assume you have some eval data.

**[01:20] YOU:** Good. I'll say up front: this is mostly a systems problem, not an ML one. The only learned decision is routing — pick the cheapest model that still meets the task's quality bar. Everything else is GPU memory management and scheduling. And the one fact that drives the whole design is the KV cache. Can I do the memory math?

**[01:45] INTERVIEWER:** Please.

**[01:55] YOU:** KV cache per token is 2 — for K and V — times layers, times KV heads, times head dim, times precision bytes. For a 70B with about 80 layers, 8 KV heads with grouped-query attention, head dim 128, fp16, that's roughly 320 KB per token. So a 4,000-token conversation holds about 1.3 GB of KV cache. Now the budget: the 70B in fp16 is about 140 GB, which doesn't fit on one 80GB H100, so I need two GPUs just for weights. Two H100s give 160 GB; weights eat 140, leaving about 20 GB for KV cache. Twenty divided by 1.3 is roughly 15 concurrent long sequences. That number is the entire serving problem — the KV cache, not compute, caps how many users I can serve at once. Everything else — quantization, paged attention, GQA, batching — exists to fit more sequences into that 20 GB.

**[03:30] INTERVIEWER:** Nice. So how do you fit more?

**[03:40] YOU:** Three big levers. First, continuous batching. Naive static batching collects N requests, runs them together, and can't return until the longest finishes — so short answers idle while one long answer drains, and new requests wait for the whole batch. Continuous batching schedules at the token level: the moment any sequence finishes, its slot frees and a waiting request joins mid-flight. That keeps the GPU saturated and is the biggest throughput win in modern serving. Second, PagedAttention. Naive serving pre-allocates a contiguous max-length KV buffer per request, so a request that *might* emit 2000 tokens reserves all of it even if it stops at 50 — huge fragmentation. PagedAttention treats KV like virtual memory: fixed-size blocks allocated on demand with a block table. Near-zero fragmentation, way more concurrency, and it lets two requests with the same system prompt share the physical KV blocks for that prefix. Third, quantization — drop weights to int8 or fp8, freeing memory for more KV and speeding up the bandwidth-bound decode phase, at a small measured quality cost.

**[05:30] INTERVIEWER:** You mentioned a "decode phase." Explain the phases.

**[05:40] YOU:** A request has two physically different phases. Prefill processes the whole prompt in parallel to build the KV cache and emit the first token — it's compute-bound, saturates the GPU, and its cost scales with prompt length. Prefill latency is essentially time-to-first-token. Decode then generates output one token at a time, each attending to the full KV cache — it's memory-bandwidth-bound, does little compute, and scales with output length. Decode determines time-per-output-token. This matters operationally: a long prompt hurts TTFT, a long answer hurts total latency, and a big prefill can stall the steady decode stream for other users — head-of-line blocking. At extreme scale I'd disaggregate prefill and decode onto separate pools so they stop interfering.

**[07:10] INTERVIEWER:** How do you route between the 8B, 70B, and the API?

**[07:20] YOU:** Cheapest model that meets the quality bar. Static routing by tenant/task config is the safe default — predictable, but it overpays when easy traffic hits the big model. Dynamic routing uses a small classifier on prompt features to send easy prompts to the 8B, hard ones to the 70B, hardest to the frontier API. But dynamic routing is only safe if I have per-task eval data to set the thresholds and a live quality monitor per route — otherwise I'll happily route everyone to the cheap model and silently degrade quality. And I'd add a semantic cache in front keyed on prompt embedding, so near-duplicate requests skip generation entirely — a big saver on repetitive enterprise traffic.

**[08:40] INTERVIEWER:** The semantic cache — any risk there?

**[08:50] YOU:** Two. One, correctness: a too-loose similarity threshold serves a cached answer to a subtly different question, so the threshold needs tuning and the cache needs invalidation when source data changes. Two, and more serious, isolation: the cache must be partitioned per tenant or per data classification. If tenant A's answer is served to tenant B via a cache hit, that's a cross-tenant data leak — a sev-0. Data isolation is non-negotiable even though we said soft compute isolation is fine.

**[09:50] INTERVIEWER:** Speaking of tenants — one team starts blasting huge requests. What happens?

**[10:00] YOU:** The noisy-neighbor problem, and without protection that one tenant fills all ~15 KV slots and starves everyone. So: weighted fair-share scheduling per tenant with priority queues, per-tenant token and rate quotas enforced at the gateway with backpressure — 429s rather than letting the queue explode — and a max-output-token cap so one runaway request can't hold a KV slot forever. Critical tenants get priority tiers so their SLOs are protected when capacity is tight. And I'd bill tenants for their usage via cost dashboards — chargeback changes behavior; they stop over-requesting the 70B when they pay for it.

**[11:30] INTERVIEWER:** How do you evaluate this platform before launch?

**[11:40] YOU:** It's not accuracy, it's a latency-versus-throughput frontier. Higher batch size buys throughput but costs per-request latency; I operate at the knee that meets SLOs at lowest cost. Pre-launch, I load-test with *realistic traffic shape* — the mix of prompt and output lengths matters far more than raw QPS — and measure TTFT p50/p95/p99, TPOT, tokens-per-second per GPU, utilization, and the saturation point where tail latency blows up. I watch tails, not averages; in serving the averages lie.

**[12:50] INTERVIEWER:** Your load test passes at 50K req/min but production falls over at 30K. Why?

**[13:00] YOU:** Ordered. Most likely the synthetic traffic shape was wrong — I tested uniform 500-token prompts but real traffic has a fat tail of 8000-token prompts whose prefill stalls everything; workload shape, not average QPS, sets saturation. Second, bursty arrivals — load tests send smooth traffic, real traffic spikes and fills the KV cache, triggering preemption and recompute. Third, my offline cache hit rate was inflated because synthetic prompts repeated more than real ones, so my throughput assumed a cache rate prod doesn't have. Fourth, a tenant launched a long-context feature and shifted the average sequence length, shrinking concurrency. Fifth, routing drift — live prompts are harder than my classifier's training sample, so more traffic escalates to the 70B than planned and exhausts capacity.

**[14:40] INTERVIEWER:** How do you roll out a new serving engine version safely?

**[14:50] YOU:** Shadow first — mirror real requests to the new stack and diff outputs and latencies without serving them. Then canary by traffic percentage, 1, 5, 25, 50, watching tail latency and per-route quality at each step. The gateway routing table is my master control: rollback is just shifting traffic back, instantly. Guardrails that auto-stop the ramp: TTFT and TPOT p95, error and timeout rate, per-route quality, and zero cross-tenant leakage.

**[16:00] INTERVIEWER:** Autoscaling — anything special for GPUs?

**[16:10] YOU:** Yes — I scale on queue depth and KV-cache pressure, not CPU. And GPU cold-start is brutal: loading 140 GB of weights takes a while, so purely reactive autoscaling leaves me cold during a spike. I keep warm headroom and scale *ahead* of demand using traffic forecasts. Under genuine overload I shed gracefully — serve from cache, downgrade to the smaller model, and 429 low-priority tenants with backpressure to protect the critical ones' SLOs. If a whole worker pool dies, I fail over to the frontier API — expensive, but available.

**[17:30] INTERVIEWER:** What dominates cost and how do you drive it down?

**[17:40] YOU:** GPU-seconds. The biggest lever is utilization — idle GPUs are pure waste — which is exactly why continuous batching matters; it can multiply throughput per GPU several-fold over static batching at the same latency. After that: route cheap traffic to the 8B, semantic-cache repeats, quantize to fit more concurrency per GPU, and for the many-fine-tunes case, multi-LoRA serving so 30 tenants' adapters share one base model in memory instead of 30 full copies. I'd track cost per million tokens as a first-class metric and chargeback to tenants.

**[18:50] YOU:** This connects to the reliability and cost-control work I did on enterprise copilots — though I'd keep this answer infra-focused. The lesson that transferred was that the perceived-latency win came from streaming and TTFT, and the cost win came from ruthlessly routing the easy majority of traffic to a smaller model and caching repeats, rather than from any single model choice. The KV-cache budget is what made those trade-offs concrete instead of hand-wavy.

**[19:40] INTERVIEWER:** That's a strong systems answer.

### Why this transcript works

- **Names the reframing** (systems problem, one learned router) instead of over-claiming ML.
- **Derives the KV-cache concurrency limit from first principles** — the credibility moment.
- **Separates prefill/decode and TTFT/TPOT** with the right hardware reasoning (compute vs bandwidth bound).
- **Explains why static batching wastes the GPU** and what PagedAttention actually fixes.
- **Treats cache isolation as a security boundary**, not just a perf feature.
- **Handles multi-tenant fairness** (noisy neighbor, quotas, priority, chargeback).
- **Uses the latency/throughput frontier** and the load-test-vs-prod trap with workload-shape reasoning.
- **Closes on real cost/reliability experience**, kept infra-focused as the prompt requests.

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x scale?** More worker pools, regional sharding, smarter forecast-driven autoscaling, prefill/decode disaggregation, and harder isolation tiers. The KV-cache math per worker doesn't change — you just need more workers and better packing.
- **How do you handle a 200K-token context request?** It blows the KV budget alone, so route it to a long-context-specialized pool or model, cap it, use KV-cache offloading/tiering to CPU/NVMe for cold blocks, and charge appropriately — long context is expensive and should be priced.
- **How do you pick batch size / max concurrency?** From the frontier curve — raise batch size until TTFT/TPOT p95 hit the SLO ceiling, then back off. It's workload-dependent, so it's tuned per model pool and re-checked as traffic shifts.
- **Offline OK, online bad — what do you check?** Traffic shape, burstiness, inflated cache rate, tenant-mix shift, routing drift. (Section 7.)
- **How do you debug a latency spike?** Per-tenant metrics to find the noisy neighbor, KV-pressure and preemption counters, prefill-queue depth (a big prompt blocking decodes), and diff the engine/model/router versions; shift traffic off the bad layer via the gateway.
- **How do you prevent cross-tenant leakage?** Partition the semantic cache and any prefix-cache by tenant/classification, scope auth at the gateway, never log raw prompts across tenant boundaries, and audit for it as a sev-0 guardrail.
- **Self-host vs API per request?** Self-host when volume amortizes the GPU cost and you need control/latency/data-residency; use the API for spiky, low-volume, or hardest-quality traffic where you can't justify dedicated GPUs.

---

## 13. Common mistakes

- Treating it as an **ML modeling** problem instead of a **systems/scheduling** problem.
- Not knowing the **KV cache** is the memory bottleneck, or being unable to estimate concurrency.
- Conflating **TTFT and TPOT**, or not knowing prefill is compute-bound and decode bandwidth-bound.
- Proposing **static batching** (or not knowing the difference from continuous batching).
- Saying "use vLLM" without explaining **what PagedAttention solves**.
- Forgetting **cache/data isolation** between tenants (a leakage sev-0).
- No **fairness/quota/admission control**, so a noisy neighbor starves everyone.
- Load-testing with **unrealistic traffic shape** and trusting averages over tails.
- Reactive-only **autoscaling** that ignores slow GPU cold-start.

---

## 14. Transfer — what this case unlocks

- **04 / 07 Agents & Copilots:** the model-tiering and cost-control levers here are what make those application systems affordable; this case is their substrate.
- **06 LLM Eval & Monitoring:** the per-route quality monitor that makes dynamic routing safe is built on that eval platform.
- **12 ML Model Deployment Platform:** shares rollout discipline (shadow/canary/gateway kill-switch) and autoscaling, generalized beyond LLMs.
- **05 Production RAG:** TTFT/streaming reasoning and caching apply directly to the RAG serving path.
- **13 LLM Safety Gateway:** sits in the same gateway layer; safety filtering is another stage in this request path.
- **20 ML Monitoring & Drift:** routing drift and quality-per-route monitoring are a drift-detection problem.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- vLLM inference engines (Hugging Face): https://huggingface.co/docs/inference-endpoints/en/engines/vllm
- vLLM PagedAttention paper: https://arxiv.org/abs/2309.06180

Added (verified canonical):
- PagedAttention / vLLM (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
- Orca, continuous batching (Yu et al., OSDI 2022): https://www.usenix.org/conference/osdi22/presentation/yu
- Speculative decoding (Leviathan et al., 2022): https://arxiv.org/abs/2211.17192
- Grouped-query attention (Ainslie et al., 2023): https://arxiv.org/abs/2305.13245
