# ML Data Foundations

An ML system is not only code plus a model. It is a decision-making system whose behavior depends on how real-world events are recorded, converted into labels, transformed into features, stored, joined, validated, and served back to the model. If the data layer is wrong, the model can look brilliant offline and fail silently in production.

!!! tip "Rapid Recall"
    A normal service fails loudly when it crashes; an ML service often fails quietly by returning a plausible but wrong answer. So ML reliability is not just SLOs, it is whether predictions stay accurate, calibrated, fair, and valid for today's population. ML adds three new requirements beyond ordinary software: reliability now includes statistical correctness, scalability spans traffic plus model complexity plus number of models, maintainability adds data and model lineage, and adaptability (monitoring, retraining, rollback, feature evolution) is part of the design, not an afterthought. The running fraud-at-checkout scenario forces every data question: what was known at payment time, when the label arrives, which features must be fresh, and where each kind of data lives.

## Running Scenario: Fraud Detection at Checkout

Throughout this section, imagine you work at an e-commerce company. When a user attempts payment, you need to predict whether the transaction is fraudulent. The prediction must happen in under 100 ms because the checkout flow cannot wait. The true label may arrive days or weeks later through chargebacks, manual review, customer complaints, or bank reports.

This scenario is useful because it forces nearly every ML data question: What was known at payment time? When does the label arrive? Which features must be fresh? Which data belongs in a product database, warehouse, stream, feature store, or lakehouse? How do you avoid training on future information?

## §1 Why ML System Requirements Differ From Software Requirements

A normal service fails loudly when it crashes. An ML service often fails quietly by returning a plausible but wrong answer.

In ordinary software, reliability usually means the system keeps responding correctly under infrastructure failure: server crash, database failover, bad deploy, network timeout, disk full, traffic spike. These are still important in ML systems. But ML adds a second meaning of reliability: the prediction itself must remain trustworthy under changing data.

In the fraud scenario, the API could be perfectly healthy. It may return HTTP 200 in 40 ms. But if fraudsters change behavior, the model may approve bad transactions. No exception is thrown. The business discovers the problem later through chargebacks. That is the key mental shift: ML correctness is partly statistical and delayed.

### Reliability

Software reliability asks: does the service return a response, persist data safely, recover from failures, and meet SLOs? ML reliability asks: are predictions still accurate, calibrated, fair enough for the use case, robust on edge cases, and valid for today's population?

For fraud, software reliability monitors checkout latency, API errors, database errors, and queue lag. ML reliability monitors approval rate, score distribution, feature missingness, chargeback rate once labels arrive, and performance by country, payment method, device type, and account age.

### Scalability

Software scalability usually focuses on traffic and storage. ML scalability has at least three axes. First, traffic volume: more prediction requests per second. Second, model complexity: a bigger model may require more CPU, GPU, memory, or latency budget. Third, number of models: a company may serve fraud, ranking, churn, recommendations, ETA, moderation, and LLM models at the same time.

A system serving one logistic regression model from memory can scale with ordinary API replicas. A system serving a large transformer may need GPU scheduling, batching, model warmup, and queue-based backpressure. A platform serving hundreds of models needs model registry, version routing, monitoring per model, and cost governance.

### Maintainability

Software maintainability asks whether code is modular, tested, and deployable. ML maintainability adds data and model lineage. If a model version caused a regression, you need to answer: which code commit trained it, which data snapshot was used, which feature definitions were active, which labels were included, which preprocessing code ran, and which evaluation report approved it?

### Adaptability

Adaptability is the ML-specific requirement that gets overlooked. Fraud patterns change. User behavior changes after a UI redesign. A new payment method launches. A holiday sale changes transaction distribution. The system must adapt without shutting down checkout. That means monitoring, retraining, rollback, and feature evolution are part of the system design, not afterthoughts.

!!! note "Interview note"
    If an interviewer asks "what makes ML systems different from normal systems?", do not say only "data drift." Say: "The API can be operationally healthy while the model is statistically wrong. So I need to design for software SLOs and ML quality signals: labels, feature freshness, drift, calibration, slice metrics, and retraining."

## Where to go next

- [From Raw Events to Training Examples](events-to-examples.md) builds point-in-time correct rows.
- [Labels and Ground Truth](labels.md) treats the label as a product with its own bias.
- [Features, Freshness, and Leakage](features-leakage.md) is the leakage discipline.
- [Formats and Storage](formats-storage.md) covers Parquet, OLTP/OLAP, and store selection.
- [Pipelines and the Lakehouse](pipelines-lakehouse.md) covers ETL, streaming, CDC, and the lakehouse.

## Interview Questions

**Q1: What makes ML system requirements different from ordinary software requirements?**
Ordinary software fails loudly; ML can fail silently by returning a plausible but wrong answer while the API stays healthy. So reliability gains a statistical, delayed meaning: predictions must stay accurate, calibrated, and valid for today's population. ML also adds scalability across model complexity and number of models, lineage-based maintainability, and adaptability through monitoring and retraining.

**Q2: Give a concrete example of an ML system being healthy and wrong at the same time.**
In fraud detection the checkout API can return HTTP 200 in 40 ms with no errors, yet approve fraudulent transactions because fraudster behavior shifted. No exception is thrown; the business only discovers it later through chargebacks. That is why ML monitoring must watch approval rate, score distribution, and slice performance, not just latency and error rate.

**Q3: Why is adaptability a design requirement rather than an afterthought?**
Because the world the model learned from keeps changing: fraud patterns evolve, a UI redesign shifts behavior, a new payment method launches, a holiday sale changes the transaction distribution. The system must adapt without shutting down checkout, so monitoring, retraining, rollback, and feature evolution have to be built in from the start.
