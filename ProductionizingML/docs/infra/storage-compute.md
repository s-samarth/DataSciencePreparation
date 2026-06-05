# Storage and Compute

The bottom layer of the stack: where state lives and what runs the math. ML systems use many stores because "data" is not one thing, and they stress compute in different ways, so training and serving should not be scheduled the same way.

!!! tip "Rapid Recall"
    Raw events need cheap durable storage, curated tables need analytical scans, online features need low-latency lookups, embeddings need vector search, artifacts need versioned storage, and metadata needs transactional records. Forcing one database to do all of this gives poor performance or poor correctness. On compute, data validation and ETL run well on CPUs, deep learning training needs GPUs or TPUs, and LLM serving needs GPUs for memory as much as compute. Training and serving have different shapes: training can tolerate checkpoint and retry, while serving must stay warm, respond quickly, and scale with traffic, so a spot interruption is fine for training but disastrous for an endpoint without redundancy.

## §1 Storage: where each kind of state lives

ML systems use many stores because "data" is not one thing.

Raw events need durable, cheap storage. Curated training tables need analytical scans. Online features need low-latency key lookups. Embeddings need vector search. Model artifacts need versioned artifact storage. Metadata needs transactional records. Logs need queryable retention. Forcing one database to do all of this creates either poor performance or poor correctness.

| Storage layer | What it stores | Why it exists |
|---|---|---|
| Object storage | raw events, Parquet files, artifacts, logs | cheap durable storage for large files |
| Warehouse/lakehouse | curated tables, labels, feature history | large scans, SQL, governance, training data |
| Online store | latest feature values | low p99 lookup during serving |
| Vector store/index | embeddings and nearest-neighbor indexes | semantic retrieval, RAG, recommendations |
| Metadata DB | runs, model versions, schemas, permissions | transactional system of record |
| Artifact store | model files, tokenizers, preprocessors, reports | versioned deployable outputs |

In the fraud platform, the checkout database is the source of truth for transactions, but training should not run huge scans on it. Events flow into a lakehouse. Feature histories are built there. Latest features are materialized to an online store. Model artifacts go to a registry/artifact store. Prediction logs go to analytical storage for monitoring and labels.

## §2 Compute: CPU, GPU, TPU, training, serving

Different ML jobs stress compute differently, so they should not all be scheduled the same way.

Data validation and many ETL jobs run well on CPUs. Classical ML and small batch inference may also be CPU-friendly. Deep learning training often needs GPUs or TPUs for dense matrix operations. LLM serving needs GPUs not only for compute but for memory: model weights and KV cache must fit.

Training and serving have different compute shapes. Training jobs may run for hours on many GPUs and can often tolerate checkpoint/retry. Serving jobs must stay warm, respond quickly, and scale with traffic. A spot interruption is acceptable for a checkpointed training job but disastrous for a production endpoint if there is no redundancy.

Compute choices also affect software design. A CPU model can run inside ordinary web replicas. A GPU model may need separate model-serving pods, batching, warmup, autoscaling by queue depth, and model loading time management. A giant LLM may require tensor parallelism across GPUs and careful routing.

## Interview Questions

**Q1: Why do ML systems use so many different storage layers?**
Because "data" is not one thing and each access pattern wants a different store: raw events want cheap durable object storage, curated tables want analytical warehouse or lakehouse scans, online features want low-latency key lookups, embeddings want vector search, artifacts want versioned storage, and metadata wants transactional records. Forcing one database to serve all of these yields either poor performance or poor correctness.

**Q2: Why does LLM serving need GPUs for memory, not just compute?**
Because the model weights and the KV cache for every active sequence must physically fit in GPU memory, and that memory pressure often limits concurrency before raw compute does. Unlike a small classifier that fits anywhere, an LLM's growing KV cache makes memory the binding constraint, which is why GPU choice for LLM serving is as much about capacity as speed.

**Q3: Why should training and serving be scheduled differently?**
Because they have different shapes. Training runs for hours on many GPUs and tolerates checkpoint and retry, so a spot interruption is acceptable. Serving must stay warm, respond within a latency budget, and scale with traffic, so the same spot interruption is disastrous without redundancy. Treating them identically either wastes money on training or risks outages on serving.

**Q4: How does compute choice ripple into software design?**
A CPU model can run inside ordinary web replicas, but a GPU model often needs separate serving pods, batching, warmup, autoscaling by queue depth, and management of model load time, and a giant LLM may need tensor parallelism across GPUs with careful routing. So the hardware decision is not isolated; it dictates the serving architecture around the model.
