# Governance and Cost

ML infrastructure controls sensitive data, expensive compute, and automated decisions, so access and auditability matter. And cost is a system behavior, not just a cloud bill, because the levers that cut it usually trade against reliability or quality.

!!! tip "Rapid Recall"
    Security and governance exist because ML infrastructure controls sensitive data, expensive compute, and automated decisions: IAM governs who can read training data, launch GPU jobs, register models, and access PII; secrets management, encryption, and audit logs round it out; governance adds classification, retention, deletion, approvals, model cards, and lineage, and regulated domains may need to prove which model decided, on what data, with whose approval. Cost is a system behavior: serving often costs more than training because endpoints run continuously, and every cost control (scale-to-zero, batching, quantization, caching, spot training, log sampling) has a reliability or quality tradeoff.

## §1 Security and Governance

IAM controls who can read training data, launch GPU jobs, register models, approve production deployment, view prediction logs, and access PII. Secrets management keeps database credentials, provider keys, and tokens out of notebooks and code. Encryption protects data at rest and in transit. Audit logs record who accessed data and who promoted a model.

Governance includes data classification, retention, deletion workflows, approval processes, model cards, lineage, and policy checks. For regulated domains, you may need to prove which model made a decision, which data trained it, what features it used, and who approved it.

Supply-chain security matters too. Containers need vulnerability scanning. Dependencies need pinning. Model artifacts and base images should be trusted. Prompt templates and evaluation datasets may also need version control and approval.

## §2 Cost Governance

ML cost is a system behavior, not just a cloud bill.

Training costs are visible because GPU jobs are expensive. Serving costs can be larger because endpoints run continuously. Feature computation, storage, logs, labels, vector indexes, observability, and idle notebooks also cost money. LLM systems add token-dependent cost and GPU memory pressure.

Cost controls include autoscaling, scale-to-zero for non-critical endpoints, batching, quantization, distillation, caching, smaller-model routing, spot training with checkpointing, storage lifecycle policies, log sampling, deleting unused artifacts, and shutting down idle notebooks. But every cost control has a reliability or quality tradeoff.

The intuition shifts with scale. At low scale, early cost is experimentation and data storage, and managed defaults plus simplicity usually win. At production scale, always-on serving, GPU utilization, feature computation, logs, vector indexes, and idle endpoints often dominate, and utilization, routing, quantization, caching, and platform ownership become worth discussing.

## Interview Questions

**Q1: What does IAM govern in an ML platform, and why does it matter so much?**
IAM controls who can read training data, launch GPU jobs, register models, approve production deployment, view prediction logs, and access PII. It matters because the platform handles sensitive data, expensive compute, and automated decisions, so loose access risks data leaks, runaway cost, or an unreviewed model reaching production. Secrets management, encryption, and audit logs complete the picture by keeping credentials safe and recording who did what.

**Q2: What does governance need to prove in a regulated domain?**
Which model made a given decision, which data trained it, what features it used, and who approved it. That requires data classification, retention and deletion workflows, approval processes, model cards, and lineage so the whole chain is auditable. Without that record you cannot defend an automated decision after the fact.

**Q3: Why is serving often more expensive than training?**
Because training is a bounded GPU job, while serving endpoints run continuously to stay warm and responsive. On top of always-on compute, feature computation, storage, logs, labels, vector indexes, observability, and idle notebooks all accrue cost, and LLMs add token-dependent cost and GPU memory pressure. The continuous nature of serving makes it a steady drain rather than a one-time spend.

**Q4: Why does every cost control come with a tradeoff?**
Because the levers that save money usually reduce reliability or quality: scale-to-zero adds cold starts, aggressive batching adds latency, quantization and distillation risk accuracy, caching risks staleness, spot training risks interruption, and log sampling reduces observability. So cost governance is not free optimization; it is balancing spend against the reliability and quality the product needs.
