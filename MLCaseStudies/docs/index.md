# ML System Design Case Studies

Twenty worked ML and GenAI system design case studies, built for interview prep. Each one drives a single problem to the floor through the same reusable scaffold: clarify, frame as ML, data and labels, baseline, model, evaluation, deployment, monitoring. Every case study includes a full one hour interview transcript, a junior versus senior contrast, and a one page whiteboard cheat sheet.

Read one case study end to end, internalize the scaffold, and the rest become the same skeleton with different nouns.

## Recommendation & Personalization

How systems decide what to show a user, and how feedback loops quietly bias the labels you train on.

- [Recommendation System](recommendation/01-recommendation-system.md)
- [News Feed / Personalized Ranking](recommendation/02-news-feed-personalized-ranking.md)
- [Conversational Recommender](recommendation/15-conversational-recommender.md)
- [People You May Know / Graph Recommendation](recommendation/16-people-you-may-know-graph-recommendation.md)
- [Autocomplete / Typeahead Personalization](recommendation/18-autocomplete-typeahead-personalization.md)
- [Notification Optimization / Bandit System](recommendation/19-notification-optimization-bandit-system.md)

## Search, Ranking & Ads

Ranking under a relevance label problem, calibration, and the auction and experimentation layer that sits on top.

- [Search Ranking System](search-ads/03-search-ranking-system.md)
- [Ads CTR / Ranking / Experimentation](search-ads/11-ads-ctr-ranking-experimentation.md)

## LLM & GenAI Systems

Grounding, retrieval, agents, serving, safety, and how you evaluate systems that have no clean ground truth.

- [Enterprise AI Copilot](llm-systems/04-enterprise-ai-copilot.md)
- [Production RAG / Enterprise Search](llm-systems/05-production-rag-enterprise-search.md)
- [LLM Evaluation & Monitoring Platform](llm-systems/06-llm-evaluation-monitoring-platform.md)
- [AI Agent for Customer Support](llm-systems/07-ai-agent-customer-support-ticket-resolution.md)
- [Multi-Tenant LLM Serving Platform](llm-systems/08-multi-tenant-llm-serving-platform.md)
- [LLM Safety Gateway](llm-systems/13-llm-safety-gateway.md)
- [Document Intelligence Pipeline](llm-systems/14-document-intelligence-pipeline.md)

## Trust, Safety & Anomaly

Imbalanced, adversarial, and contested label problems where the cost of a mistake is asymmetric.

- [Fraud / Anomaly Detection System](trust-safety/09-fraud-anomaly-detection-system.md)
- [Content Moderation / Policy Enforcement](trust-safety/10-content-moderation-policy-enforcement.md)
- [Spam / Bot Detection System](trust-safety/17-spam-bot-detection-system.md)

## ML Platform & Ops

The infrastructure underneath every model: safe rollout, train serve skew, drift, and retraining.

- [ML Model Deployment Platform](platform-ops/12-ml-model-deployment-platform.md)
- [ML Monitoring / Drift / Retraining System](platform-ops/20-ml-monitoring-drift-retraining.md)
