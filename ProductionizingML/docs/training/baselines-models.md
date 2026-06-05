# Baselines and Model Choice

The best production model is not always the most complex model. It is the simplest model that meets quality, latency, cost, and maintainability requirements. This page is about why baselines come first and why model choice is a system decision, not just an accuracy contest.

!!! tip "Rapid Recall"
    A baseline is a deliberately simple solution used as a reference point: a rule, then logistic regression, then XGBoost or LightGBM, before any deep model. Baselines catch data bugs: impossibly high AUC suggests leakage, and a complex model that barely beats a baseline may not be worth its serving cost. Model choice is a system decision, because a neural model can improve accuracy while demanding GPU serving, longer p99, harder explanations, larger artifacts, and more complex monitoring, whereas a tree model may be cheaper on CPU, easier to explain, and good enough.

## §1 Why baselines come first

A baseline is a deliberately simple solution used as a reference point. For fraud, a baseline might be a rule: block transactions above a threshold from newly created accounts. A stronger baseline might be logistic regression. A common production baseline might be XGBoost or LightGBM. Only after these baselines are understood should you justify deep models or transformers.

Baselines catch data bugs. If a simple model gets impossibly high AUC, you may have leakage. If a complex model barely beats a baseline, the extra serving cost may not be worth it. If the baseline performs better on a critical slice, the new model may be learning spurious correlations.

## §2 Model choice is a system decision

Model choice is a system decision. A neural model may improve accuracy but require GPU serving, longer p99 latency, harder explanations, larger artifacts, and more complex monitoring. A tree model may be easier to serve on CPU, easier to explain, and good enough. In interviews, you should show that you understand this tradeoff.

| Model family | Why use it | Production cost |
|---|---|---|
| Rules / heuristics | Fast, explainable, good fallback | Low accuracy ceiling, manual maintenance |
| Logistic regression | Strong baseline, calibrated-ish, simple | Needs feature engineering, limited nonlinearity |
| GBDT / XGBoost / LightGBM | Excellent tabular performance | Can be harder to calibrate and monitor than linear models |
| Deep neural models | Useful for embeddings, text, images, large interactions | More serving cost, more tuning, harder debugging |
| Transformers / LLMs | Language, multimodal, reasoning-like tasks | High latency/cost, evaluation and safety complexity |

## Interview Questions

**Q1: Why start with a baseline instead of the best model you can build?**
Because the best production model is the simplest one that meets quality, latency, cost, and maintainability requirements, and a baseline tells you whether the complexity is even buying anything. A rule, logistic regression, or XGBoost gives a reference point; if the fancy model barely beats it, the extra serving cost is not justified.

**Q2: How do baselines help catch bugs?**
A baseline that scores impossibly high AUC is a leakage warning. A complex model that barely beats the baseline signals the added cost may not be worth it. And a baseline that beats the new model on a critical slice suggests the new model is learning spurious correlations rather than the real signal.

**Q3: Why is model choice a system decision rather than an accuracy decision?**
Because a more accurate model can be infeasible to operate: a neural model may need GPU serving, longer p99 latency, larger artifacts, harder explanations, and more complex monitoring, while a tree model may serve cheaply on CPU and be good enough. The right answer weighs accuracy against serving cost, latency, explainability, and maintainability.
