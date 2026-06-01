# LLM & GenAI Systems

Seven case studies on building production systems around large language models. The thread that connects them is that there is rarely a clean ground truth label, so evaluation becomes the hard part. The copilot and RAG cases focus on grounding answers in retrieved context and respecting permissions. The agent case adds tool use and a human handoff flywheel. The evaluation platform makes judging quality the product itself. The serving platform is the infrastructure question of latency and throughput under multi tenant load. The safety gateway treats the model as a target for an adaptive attacker. The document intelligence pipeline is a cascade where early errors compound.

Master the copilot and RAG first, then the rest extend the same grounding, retrieval, and evaluation ideas.

- [Enterprise AI Copilot](04-enterprise-ai-copilot.md)
- [Production RAG / Enterprise Search](05-production-rag-enterprise-search.md)
- [LLM Evaluation & Monitoring Platform](06-llm-evaluation-monitoring-platform.md)
- [AI Agent for Customer Support](07-ai-agent-customer-support-ticket-resolution.md)
- [Multi-Tenant LLM Serving Platform](08-multi-tenant-llm-serving-platform.md)
- [LLM Safety Gateway](13-llm-safety-gateway.md)
- [Document Intelligence Pipeline](14-document-intelligence-pipeline.md)
