# ML Platform & Ops

Two case studies on the infrastructure underneath every model. The deployment platform is about why a model is not just code: it carries data dependencies and reproducibility concerns, so safe rollout means shadow, canary, ramp, and rollback rather than a single deploy. The monitoring system is about the label delay problem, the ladder of proxy signals you watch while real labels are still pending, and how you decide when to retrain.

These two are the closing arc of the set: once you can serve and rank, these are the cases that keep a system alive in production.

- [ML Model Deployment Platform](12-ml-model-deployment-platform.md)
- [ML Monitoring / Drift / Retraining System](20-ml-monitoring-drift-retraining.md)
