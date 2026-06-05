# Build vs Buy and Capstone

Build-vs-buy is not about pride. It is about which operational burden your team should own. This page gives the decision framework and then the capstone answer that ties every section of the site into one end-to-end fraud platform.

!!! tip "Rapid Recall"
    Managed platforms (SageMaker, Vertex AI, Databricks, Azure ML) win when speed, governance, and low operational overhead matter, but can be expensive or restrictive at scale. Self-hosted stacks win with platform engineers, many models, custom runtimes, strict cost pressure, vendor-neutral needs, or deep LLM serving, but you then own upgrades, incidents, patches, drivers, autoscaling, and support. Hybrid is often best, splitting along data gravity, team skill, compliance, cost, and latency. The common mistake is undercounting engineering time: a "free" open-source stack is not free if two engineers spend half their time keeping it alive.

## §1 Build vs Buy

Build-vs-buy is not about pride. It is about which operational burden your team should own.

Managed platforms are strong when speed, governance, and low operational overhead matter. SageMaker fits AWS-native teams. Vertex AI fits GCP-native teams, especially with BigQuery gravity. Databricks fits lakehouse-centric organizations. Azure ML fits Microsoft/Azure environments. Managed platforms give integrated IAM, artifact storage, endpoints, pipelines, monitoring, and approval workflows, but they can be expensive or restrictive at scale.

Self-hosted stacks are strong when you have platform engineers, many models, custom runtimes, strict cost pressure, vendor-neutral needs, unusual GPU scheduling, or deep LLM serving requirements. But self-hosting means you own upgrades, incidents, security patches, drivers, autoscaling, documentation, and user support.

Hybrid is often best. You might use a managed warehouse, MLflow for tracking, Feast for features, KServe/vLLM for inference, and a managed monitoring tool. Or you might use SageMaker training but custom Kubernetes inference. The right split follows data gravity, team skill, compliance, cost, and product latency.

### A decision rule

A simple way to reason about it: count the pressures that push toward owning the platform, a dedicated platform team, twenty or more models, strict data residency, and custom LLM or GPU runtime needs. The more of these are true, the more it makes sense to build or self-host selectively; the fewer, the more a managed platform wins.

| Team | Scale | Compliance | Runtime | Recommendation |
|---|---|---|---|---|
| Small | Few models | Standard | Standard | Buy managed first |
| Small | 20+ models | Standard | Standard | Buy managed, watch cost |
| Platform team | 20+ models | Strict residency | Standard | Build/self-host selectively |
| Platform team | 20+ models | Strict residency | Custom LLM/GPU | Build/self-host selectively |

When the constraints justify owning some platform layers, still buy the commodity parts where they do not reduce control. When they do not, managed services reduce operational burden while the team learns its workload shape; revisit when scale, cost, or custom runtime needs grow.

!!! warning "Common mistake"
    Undercounting engineering time. A "free" open-source stack is not free if two engineers spend half their time keeping it alive.

## §2 Capstone: End-to-End Fraud Platform

A complete ML systems answer ties every prior section together.

For checkout fraud, the application writes transaction events. Kafka or pub/sub carries event streams. Object storage and the lakehouse retain history. Batch and streaming jobs compute features. Offline feature history builds point-in-time training sets. An online feature store serves current values at checkout. A training pipeline validates data, trains candidates, logs experiments, evaluates slices, registers artifacts, and promotes through gates. The serving system fetches features, runs the model, applies thresholds, logs predictions, and falls back safely. Monitoring joins delayed labels, watches drift, tracks business KPIs, and triggers rollback, rethresholding, or retraining.

Infrastructure is the shared platform beneath this: storage, compute, orchestration, registry, serving, monitoring, security, governance, and cost controls.

!!! note "Final interview answer"
    "I would separate product logic from platform capabilities. The platform should provide reproducible pipelines, versioned artifacts, low-latency serving, feature consistency, monitoring, secure access, and cost visibility so teams can ship models safely without rebuilding the lifecycle each time."

## Interview Questions

**Q1: How do you frame a build-vs-buy decision?**
Not as pride but as which operational burden the team should own. Managed platforms win when speed, governance, and low overhead matter; self-hosting wins with platform engineers, many models, custom runtimes, strict cost pressure, vendor-neutral needs, or deep LLM serving. The decision follows data gravity, team skill, compliance, cost, and latency, and hybrid splits are often best.

**Q2: What pushes a team toward building or self-hosting?**
The accumulation of pressures: a dedicated platform team, many models (say twenty or more), strict data residency, and custom LLM or GPU runtime needs. The more of these hold, the more owning some platform layers pays off. With few of them, a managed platform reduces operational burden while the team learns its workload, and you revisit as scale and cost grow.

**Q3: What is the most common mistake in build-vs-buy reasoning?**
Undercounting engineering time. A "free" open-source stack is not free if two engineers spend half their time keeping it alive through upgrades, incidents, patches, drivers, and support. The true comparison is managed cost versus the fully loaded cost of the people who operate a self-hosted stack, not license price alone.

**Q4: Give the capstone answer for an end-to-end fraud platform.**
Separate product logic from platform capabilities. Events flow through streaming into a lakehouse; batch and streaming jobs build offline feature history and materialize online features; a training pipeline validates, trains, evaluates slices, registers, and gates; serving fetches features, runs the model, thresholds, logs, and falls back safely; and monitoring joins delayed labels, watches drift, tracks business KPIs, and triggers rollback, rethresholding, or retraining, all on shared infrastructure for storage, compute, orchestration, registry, serving, monitoring, security, governance, and cost.
