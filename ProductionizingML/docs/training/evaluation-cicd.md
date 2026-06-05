# Evaluation Gates and CI/CD

Evaluation decides whether a trained artifact deserves the risk of production traffic. CI/CD for ML is the machinery that enforces that decision automatically, adding data and model gates on top of the familiar code gates.

!!! tip "Rapid Recall"
    Offline metrics depend on the task, but global metrics are never enough: a fraud model can improve overall AUC while hurting a small country, new users, or high-value customers, which slice metrics reveal. Evaluation must include serving constraints, since a model that is 1% more accurate but 10x slower, or that uses an offline-only feature, cannot ship. ML CI/CD adds data gates before training (schema, ranges, nulls, freshness, duplicates, label availability, point-in-time, leakage) and model gates after (metrics, slices, fairness, calibration, robustness, latency, memory, cost) on top of the usual code gates. Deployment gates then stage the rollout: register, shadow, canary at 1%, 5%, 25%, full.

## §1 Evaluation Gates

Evaluation decides whether a trained artifact deserves the risk of production traffic.

Offline metrics depend on task. Classification uses precision, recall, ROC-AUC, PR-AUC, log loss, calibration, confusion matrices. Ranking uses NDCG, MAP, MRR, recall@k. Regression uses MAE, RMSE, quantile loss. LLM systems use task-specific evals, human preference, judge models, safety metrics, refusal behavior, tool success, factuality checks, and regression test sets.

Global metrics are not enough. A fraud model can improve AUC overall while hurting a small country, new users, a payment method, or high-value customers. Slice metrics reveal this. If the system makes consequential decisions, fairness and compliance slices may be mandatory.

Evaluation must include serving constraints. A model that is 1% more accurate but 10x slower may be impossible for checkout. A model that uses a feature unavailable online cannot ship. A model that needs a GPU for every request may not fit budget. This is why evaluation gates include latency, memory, cost, and feature availability.

!!! note "Promotion question"
    Would you ship a model with better global AUC, worse recall for one protected segment, and p99 latency above the product SLO? A mature answer is no, not directly. You investigate the slice regression, maybe adjust thresholds, retrain, or run a limited shadow/canary, but you do not blindly promote.

## §2 CI/CD for ML

Software CI/CD proves code can be built, tested, and deployed. ML CI/CD also proves data and model behavior are acceptable.

The code gate remains: unit tests, integration tests, type checks, packaging, dependency scanning, container build, and API contract tests. But ML adds data gates before training: schema, ranges, null rates, freshness, duplicates, label availability, point-in-time correctness, and leakage checks.

After training, model gates decide whether the candidate can proceed. They compare against thresholds and the current production model. They check global metrics, slice metrics, fairness, calibration, robustness, latency, memory, cost, and explainability requirements where relevant.

Deployment gates decide how the model reaches users. A candidate might first be registered, then deployed in shadow mode, then canary to 1%, then 5%, then 25%, then full rollout. The [Production Loop](../loop/index.md) section covers this loop deeply.

### Promotion gates

A candidate passes through a sequence of gates before it carries real traffic:

1. Code tests and package build pass.
2. Training data schema, freshness, nulls, and ranges pass.
3. Point-in-time joins and leakage checks pass.
4. Candidate beats current production on primary metric.
5. Critical slices do not regress beyond threshold.
6. Calibration, fairness, latency, memory, and cost pass.
7. Artifact is registered with lineage and approved for staged rollout.

## Interview Questions

**Q1: Why are global metrics insufficient for an evaluation gate?**
Because a model can improve overall AUC while quietly regressing a small country, new users, a payment method, or high-value customers. Slice metrics surface that, and for consequential decisions fairness and compliance slices may be mandatory. A gate that only checks the global number can approve a model that is worse where it matters most.

**Q2: What serving constraints belong in an evaluation gate?**
Latency, memory, cost, and feature availability. A model that is slightly more accurate but far slower may be impossible for checkout, a model that depends on an offline-only feature cannot serve online, and a model that needs a GPU per request may bust the budget. Accuracy alone does not make a model shippable.

**Q3: How does ML CI/CD differ from software CI/CD?**
It keeps the code gates (tests, type checks, packaging, container build, API contracts) and adds two ML-specific layers: data gates before training (schema, ranges, nulls, freshness, duplicates, label availability, point-in-time correctness, leakage) and model gates after training (global and slice metrics, fairness, calibration, robustness, latency, memory, cost). Then deployment gates stage the rollout.

**Q4: Describe the staged path from a trained candidate to full traffic.**
The candidate is registered, then deployed in shadow mode to observe it without affecting users, then promoted to a small canary such as 1%, then 5%, 25%, and finally full rollout, each step gated on metrics and slice behavior. Staging the exposure limits blast radius if the model misbehaves under real traffic.
