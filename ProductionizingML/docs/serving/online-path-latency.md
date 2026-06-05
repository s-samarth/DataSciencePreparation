# Online Path and Latency Budget

Online serving is hard because it combines live product latency with model correctness. This page separates the online responsibilities so you know where bugs live, then treats latency as a spending plan for milliseconds.

!!! tip "Rapid Recall"
    A clean online design separates the product service, feature service or store, model server, decision policy, and logging pipeline, because a wrong decision can come from stale features, a model regression, a bad threshold, missing logging, or a timeout. A latency budget is a spending plan for milliseconds: if checkout allows 100 ms, that includes network, feature lookup, inference, postprocess, and logging, so you can hit 90 ms before tail spikes. Latency has percentiles, and at scale even 1% slow requests is constant pain. Good serving uses timeouts so one slow dependency routes to fallback features, a simpler model, or manual review rather than hanging the whole path.

## §1 Online Serving Path, Slowly

Online serving is hard because it combines live product latency with model correctness.

For fraud, the product service calls the model service during checkout. That model service may depend on an online feature store. The feature store may depend on streaming pipelines that keep counters fresh. The model service may also call a device-risk service. Each dependency adds latency and failure risk.

A clean online design separates responsibilities:

- **Product service:** owns checkout flow and user-facing behavior.
- **Feature service/store:** owns low-latency feature retrieval and freshness.
- **Model server:** owns model artifact loading, preprocessing, inference, and postprocessing.
- **Decision policy:** owns thresholds, manual review routing, and business rules.
- **Logging pipeline:** owns prediction logs and later label joins.

You can colocate some of these in simple systems, but conceptually they are distinct. This distinction helps in interviews because it shows you understand where bugs live. A wrong decision could come from stale features, model regression, bad threshold, missing logging, or a serving timeout.

!!! note "Beginner mental model"
    The model server is not the brain alone. It is the whole nervous system around the brain: inputs, preprocessing, runtime, output interpretation, and memory of what happened.

## §2 Latency Budget

A latency budget is a spending plan for milliseconds.

If checkout allows 100 ms for fraud decisioning, that 100 ms includes everything in the path. Suppose network and gateway overhead cost 15 ms, feature lookup costs 30 ms, model inference costs 35 ms, postprocessing costs 5 ms, and logging adds 5 ms. You are already at 90 ms before rare tail spikes. If one dependency has a p99 of 80 ms, the whole path may violate the user experience even if the model itself is fast.

Latency also has percentiles. p50 is the median request. p95 is slower than 95% of requests. p99 is slower than 99% of requests. Product users feel tail latency because a small fraction of slow requests still happens constantly at scale. If you process 1 million requests per day, 1% is 10,000 slow requests.

Good serving design uses timeouts. If the feature store does not respond within its budget, the model service may use fallback features, a simpler model, or manual review. Without timeouts, one dependency can hang the whole checkout path.

The budget is just a sum of the path components against the SLO. A few worked examples:

| Network + gateway | Feature lookup | Model inference | Postprocess + log | Total | Verdict against 100 ms |
|---|---|---|---|---|---|
| 15 ms | 30 ms | 35 ms | 10 ms | 90 ms | Fits on paper; reserve headroom for p95/p99, retries, cold starts. |
| 15 ms | 60 ms | 35 ms | 10 ms | 120 ms | Exceeds budget before tail spikes; cut feature latency or route to manual review. |
| 10 ms | 20 ms | 20 ms | 5 ms | 55 ms | Comfortable; room for a richer model or stricter retries. |

!!! warning "Trap"
    "Our model inference is 20 ms" is not a serving latency answer. You need end-to-end p95/p99, including feature retrieval, queueing, serialization, cold starts, and retries.

## Interview Questions

**Q1: How would you decompose an online serving design, and why?**
Into product service, feature service or store, model server, decision policy, and logging pipeline. Separating them shows where bugs live, because a wrong decision can come from stale features, a model regression, a bad threshold, missing logging, or a serving timeout. Even if you colocate them in a simple system, keeping the responsibilities distinct makes the failure modes addressable.

**Q2: What is a latency budget and what goes into it?**
A spending plan for the milliseconds the product allows, covering everything in the path: network and gateway, feature lookup, model inference, postprocessing, and logging. If checkout allows 100 ms and those sum to 90 ms, you are nearly out of budget before any tail spike, so the budget forces you to account for the whole path, not just inference.

**Q3: Why do timeouts matter so much in online serving?**
Because without them a single slow or hung dependency, like the feature store, can stall the entire checkout path. With a timeout, the model service can fall back to cached or default features, a simpler model, or manual review and still respond within budget. Timeouts turn a dependency failure into a degraded-but-alive response instead of an outage.

**Q4: Why care about p99 if p50 is fast?**
Because users feel the tail, and at scale the tail is constant: a million requests per day with a 1% slow rate is 10,000 slow requests every day. A single dependency with an 80 ms p99 can blow a 100 ms budget even when the median path is quick, so SLAs and budgets are written on p95 and p99.
