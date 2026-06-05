# The 12 Building Blocks

This section intentionally avoids the flashcard style. Each building block is taught by starting with the simple problem it solves, then walking through how it works in a real request or data path. If a term appears, it is explained before it is used as an interview shortcut.

!!! tip "Rapid Recall"
    Twelve components, each justified by the problem it solves: load balancer (stable address over many replicas), API gateway (policy front door), cache (remember expensive answers), SQL vs NoSQL (match data shape and access pattern), message queue (decouple slow side effects), CDN (files near users), replication vs sharding (copies for reads, splits for writes), consistent hashing (rebalance without reshuffling), rate limiting (fairness and protection), batch vs stream (freshness vs complexity), microservices vs monolith (independence vs coordination), and monitoring vs observability (known failures vs investigating the unknown). Each block has an amber ML version layered on top.

## §1 Load Balancer

A load balancer is the traffic desk in front of many copies of the same service. The user thinks there is one website; internally, the load balancer chooses one healthy server to handle each request.

Start with the simplest system: one client sends a request to one server. That works until the server becomes overloaded, crashes, needs a deploy, or sits too far from some users. The natural next step is to run multiple identical application servers. But now the client needs one stable address. You do not want the mobile app to know that today there are 8 servers and tomorrow there are 15. The load balancer solves that indirection problem. Clients call the load balancer; the load balancer calls one backend server.

Request by request, the load balancer does three things. First, it accepts the network connection from the client. Second, it chooses a backend from its pool of healthy servers. Third, it forwards the request and returns the response. The important word is **healthy**. A load balancer is not useful if it keeps sending traffic to a dead server, so it repeatedly checks health endpoints such as `/healthz`. If a server stops responding, the load balancer removes it from rotation. When the server recovers, it can be added back.

The routing algorithm is the rule for choosing a backend. **Round-robin** means server A, then B, then C, then A again. It is simple and good when requests are similar. **Least-connections** sends the next request to the server currently handling the fewest active requests, which is better when some requests take longer than others. **Consistent hashing** uses something about the request, such as user id, session id, or key, to choose a server in a stable way. That matters when the server has local state or warm cache.

There are two common levels of load balancing. **L4 load balancing** works at the transport layer: it mostly sees IPs, ports, and TCP/UDP connections. It is fast and generic. **L7 load balancing** works at the HTTP/application layer: it can inspect paths, headers, cookies, and methods. That allows routing `/api/payments` to one service and `/api/search` to another, but it does more work per request.

Sticky sessions are a special case. Suppose server A stores a user's session in local memory. If request 1 goes to server A and request 2 goes to server B, server B may not know who the user is. A load balancer can keep that user stuck to server A using a cookie or hash. But sticky sessions are usually a smell: they make autoscaling, failover, and deployments harder. A more robust design stores shared state in Redis, a database, or a session service so any app server can handle any request.

!!! warning "Failure mode"
    If all requests for one hot key are consistently routed to one server, that server becomes a hotspot. Consistent routing helps cache locality but can hurt load balance. This is the type of tradeoff interviewers care about.

**ML version:** model-serving load balancing is harder than ordinary web serving. A model replica might need 30 seconds to load weights into GPU memory. LLM serving may have KV-cache state for active conversations. Some requests are short and cheap; others generate thousands of tokens. A good model router may consider model version, GPU memory, queue length, tenant budget, and whether a replica is warm. That is why [Serving the Model](../serving/index.md) revisits load balancing as model routing.

## §2 API Gateway

An API gateway is the controlled front door to your backend. It decides whether a request is allowed, where it should go, and what global rules apply before product code runs.

Imagine a company with separate services for users, payments, search, notifications, and recommendations. If every client talked to every service directly, the mobile app would need to know internal service names, auth rules, versions, and rate limits. That makes clients fragile. The gateway gives clients one stable entry point while hiding internal service layout.

The gateway usually handles cross-cutting concerns: TLS termination, authentication, authorization, request routing, rate limiting, API versioning, request size limits, throttling, and sometimes response aggregation. These are not business features; they are policies every service needs. Centralizing them avoids copying the same code into every service.

Do not confuse an API gateway with a load balancer. A load balancer asks, "Which replica of this service should receive the request?" A gateway asks, "Is this request valid, who is making it, what API is it calling, and which internal service owns that route?" In many systems both exist: load balancer first, gateway second, service replicas behind it.

The trap is putting too much logic in the gateway. If the gateway starts computing order totals, ranking feeds, or applying business rules, it becomes a giant central service that every team must coordinate through. Keep it focused on policy and routing.

**ML version:** AI gateways often add prompt length limits, model access control, tenant spend limits, safety filters, and routing rules such as "simple requests go to the small model, hard requests go to the large model." This is not just rate limiting by request count; it may limit tokens, GPU seconds, or dollars.

## §3 Caching

A cache remembers expensive answers so you do not recompute or refetch them for every request.

Suppose 10,000 users ask for the same product page in one minute. Without a cache, every request hits the database. But the product title, image URL, and price may not change every second. A cache stores that result in fast memory, usually Redis or Memcached, so repeated reads return quickly and the database is protected.

The most common pattern is **cache-aside**. The service first checks the cache. If the value is present, that is a cache hit and the service returns it. If the value is absent, that is a cache miss: the service queries the database, stores the result in cache, and returns it. The cache is not the source of truth; the database is.

Caching has two hard questions: what do you cache, and how do you keep it correct? A **TTL** says "this value expires after 5 minutes." TTL is simple and bounds staleness, but it means users may see old data for a while. **Event-driven invalidation** says "when the product changes, publish an event that deletes or updates the cache key." That is fresher but requires reliable events. Many real systems use both: event invalidation for important updates and TTL as a safety net.

A **cache stampede** happens when a hot key expires and many requests miss at the same time. Suddenly all of them hit the database and the database suffers. Mitigations include jittered TTLs, single-flight locking, background refresh, and serving slightly stale data while refreshing.

Memory is not free. Caches need eviction policies such as LRU, LFU, or TTL-based expiration. A cache with a poor hit rate adds complexity without much benefit. In an interview, mention the metric: cache hit rate. If the hit rate is 95%, the database sees only 5% of reads. If the hit rate is 20%, your cache may not be worth it.

**ML version:** ML systems cache features, embeddings, model outputs, and sometimes LLM responses. Semantic caching for LLMs is trickier than normal caching because two prompts can be similar but not equivalent. The cached answer must still respect user permissions, current facts, safety policy, and personalization context.

## §4 Databases: SQL vs NoSQL

A database choice is really a choice about shape of data, access pattern, consistency, and scale.

Relational databases such as Postgres and MySQL store data in tables with rows and columns. They are excellent when relationships matter: users own orders, orders contain items, payments reference orders. SQL gives joins, constraints, indexes, and transactions. If money or inventory is involved, this correctness is valuable.

NoSQL is not one thing. A document database stores JSON-like documents, useful when records have flexible nested shape such as user profiles or content metadata. A key-value store retrieves a value by exact key, useful for sessions, caches, and online features. A wide-column store such as Cassandra or Bigtable stores partitioned rows at huge write scale, useful for telemetry and time-series style access. A graph database optimizes relationship traversal, useful for fraud rings, social graphs, or knowledge graphs.

The right question is not "SQL or NoSQL?" The right question is "What are the top reads and writes?" If you need "find order by id and update payment status atomically," SQL is natural. If you need "write millions of device events per second and query by device id and time range," a wide-column design may fit better. If you need "lookup latest features for user 123 in under 10 ms," a key-value store is natural.

Interviews often reward saying "I would start with Postgres unless the access pattern proves otherwise." That shows you are not overengineering. Then explain what would force a different choice: write throughput, schema flexibility, global distribution, massive storage, or low-latency key lookups.

**ML version:** ML systems usually use multiple stores. Model metadata may live in SQL. Training data may live in a lakehouse or warehouse. Online features may live in Redis, DynamoDB, Bigtable, or Cassandra. Embeddings may live in a vector database. One database rarely serves the whole lifecycle well.

## §5 Message Queues

A queue is a waiting room for work. Producers drop work off; consumers pick it up when they have capacity.

Without a queue, a user request may have to wait for every side effect: write database row, send email, update analytics, call partner API, generate notification, and refresh search index. That makes the user-facing path slow and fragile. With a queue, the service can perform the critical durable write, enqueue side effects, and return quickly.

Queues decouple time and ownership. The producer does not need to know how many workers exist. Consumers can be scaled independently. If traffic spikes, the queue depth grows instead of dropping every request immediately. This is useful, but it is not magic. If consumers are permanently slower than producers, the queue becomes an ever-growing backlog. Queue lag is a key metric.

Delivery semantics matter. **At-most-once** means a message may be lost but not duplicated. **At-least-once** means a message will be delivered but may be duplicated. Most practical systems choose at-least-once and make consumers idempotent. Idempotent means processing the same message twice has the same effect as processing it once. For example, use an idempotency key before charging a credit card or sending a notification.

Kafka behaves like a durable append-only log with offsets and replay. RabbitMQ is often used for routing work queues. SQS is managed and simple. The exact tool matters less than the design questions: can you replay, how do retries work, where do poison messages go, and what happens when consumers fall behind?

**ML version:** queues carry training jobs, batch inference jobs, feature materialization events, label events, drift checks, and human-review tasks. They are the connective tissue between data collection, training, serving, and monitoring.

## §6 CDN

A CDN caches files near users so the origin server does not have to serve every byte from one place.

If a user in India downloads an image from a server in Virginia, physics adds latency. A content delivery network places copies of static or cacheable content at edge locations around the world. The first request may go to origin; later nearby requests are served from the edge.

CDNs are strongest for static assets: JavaScript, CSS, images, videos, downloads, documentation, and public files. They reduce latency, origin traffic, and bandwidth cost. They also absorb traffic spikes better than your application servers.

The hard part is freshness. Cache headers such as `Cache-Control` tell edges how long to keep content. If you publish a broken JavaScript file with a one-day TTL, users may keep getting the broken file unless you purge it. A common pattern is content-addressed filenames, such as `app.8f3a.js`, so new content gets a new URL and old cached content does not conflict.

**ML version:** CDNs usually do not serve live model predictions. They serve model artifacts, tokenizer files, WebAssembly/WebGPU bundles, static demos, and edge model packages. If a mobile app downloads an on-device model, CDN behavior becomes part of ML deployment.

## §7 Database Scaling: Replication vs Sharding

Replication makes copies; sharding splits ownership. Copies help reads and availability. Splits help when one machine cannot own all writes or data.

**Replication** usually means one primary database accepts writes and one or more replicas copy its data. Reads can go to replicas, which increases read capacity. If the primary fails, a replica can be promoted. The cost is replication lag: a user may write data and then read from a replica that has not caught up yet.

**Sharding** means splitting rows across multiple databases. For example, users with ids 1-1M live on shard A, 1M-2M on shard B, and so on. Now writes are distributed. Storage is distributed. But queries become harder. If you need "all orders for user 123," shard by user id works well. If you need "top products across all users," you may need to query many shards or maintain a separate analytics system.

The shard key is the most important decision. A bad shard key creates hotspots. If you shard by country and 80% of traffic is from one country, that shard is overloaded. If you shard by user id, a celebrity user may still be hot. Many systems use hashing to spread load, but hashing makes range queries harder.

In an interview, prefer replication before sharding because replication is simpler. Sharding is a major operational commitment: migrations, resharding, cross-shard transactions, global indexes, and backups become harder.

**ML version:** online feature stores, telemetry stores, and embedding stores often need sharding because they ingest huge volumes. But the serving path needs low p99 latency, so shard choice directly affects model latency.

## §8 Consistent Hashing

Consistent hashing is a way to assign keys to servers so adding or removing a server does not reshuffle everything.

Imagine a cache cluster with four nodes. A simple approach is `hash(key) % 4`. That works until you add a fifth node. Now the formula becomes `hash(key) % 5`, and most keys move to different nodes. For a cache, that means a massive cache miss storm. For storage, that means huge data movement.

Consistent hashing maps both keys and nodes onto a logical ring. A key belongs to the first node clockwise from its position. When a node is added, it only takes over a slice of the ring from neighboring nodes. When a node is removed, only its slice moves. Most keys stay where they were.

Virtual nodes improve balance. Instead of placing each physical server once on the ring, place it many times. That smooths out unlucky hash placement and makes it easier to weight stronger servers more heavily.

Consistent hashing solves rebalancing pain, not every scaling problem. It does not automatically fix hot keys. If one key receives 20% of traffic, the node owning that key still suffers. Hot-key mitigation may require replication, request coalescing, key splitting, or special-case routing.

**ML version:** consistent hashing can keep users, feature keys, or conversation sessions routed to stable serving nodes. For LLMs, this can preserve warm cache locality, but it must be balanced against GPU utilization.

## §9 Rate Limiting

Rate limiting is a fairness and protection mechanism. It prevents one caller from consuming the capacity everyone depends on.

The simplest limiter is "100 requests per minute." But implementation matters. A fixed window counter resets every minute, which allows bursts at boundaries: 100 requests at 12:00:59 and 100 more at 12:01:00. Sliding windows smooth that but cost more state.

A **token bucket** is easier to reason about. Each user has a bucket. Tokens refill at a steady rate, up to a maximum. Each request spends a token. If the bucket is empty, reject or delay the request. The bucket size controls burst allowance; the refill rate controls sustained throughput.

Distributed rate limiting is harder because requests for the same user may hit many gateway replicas. You need shared state, often Redis, or a hybrid approach: local fast limits for speed plus global reconciliation for fairness. Atomic updates matter. If two gateways read "one token left" at the same time and both allow a request, the limit is violated. Redis Lua scripts or atomic counters help.

Choose the limiting key carefully. IP-based limits can punish many users behind one NAT. User-based limits require authentication. Tenant-based limits fit B2B products. Endpoint-specific limits protect expensive APIs.

**ML version:** request count is not enough. One LLM request asking for 10 output tokens is cheap; another asking for 10,000 tokens is expensive. AI systems often rate-limit by input tokens, output tokens, GPU seconds, model class, or dollar budget.

## §10 Batch vs Stream Processing

Batch processing answers questions later using a pile of data. Stream processing reacts continuously as events arrive.

Batch jobs are natural when freshness requirements are loose. If you retrain a model every night, compute daily revenue, or backfill a year of features, a batch engine like Spark can scan large files and produce outputs. Batch is simpler because the input is bounded: the job has a beginning and an end.

Stream processing is for unbounded event flows: clicks, transactions, sensor events, logs, impressions, messages. A stream processor such as Flink, Kafka Streams, or Spark Streaming maintains state over windows. For example, "number of failed login attempts by user in the last 10 minutes" is a streaming feature.

Streams introduce time complexity. Event time is when something happened; processing time is when your system saw it. Late events happen. Watermarks tell the system when it believes a window is mostly complete. Replay matters because you may need to recompute state after a bug.

The interview decision is freshness vs complexity. If a fraud signal must update in seconds, stream. If a recommendation model refreshes daily, batch may be enough. Do not choose streaming just because it sounds advanced.

**ML version:** training often uses batch historical data, while serving may need streaming features. This split is one root of training-serving skew, which the [ML Data Foundations](../data/index.md) and [Training in Production](../training/index.md) sections cover deeply.

## §11 Microservices vs Monolith

A monolith deploys one application. Microservices split the system into independently deployable services. The split buys independence but costs coordination.

A monolith is not automatically bad. It is often the fastest way to build a new product because function calls are local, transactions are simpler, and debugging is easier. Many early systems should start as modular monoliths: clean internal boundaries, one deployable unit.

Microservices make sense when teams, scaling needs, or reliability boundaries diverge. Payments may need stricter compliance and deployment controls than profile pictures. Search may need different infrastructure from account settings. Notifications may scale workers independently from the main API.

The cost is distributed systems complexity. A local function call becomes a network call. Network calls fail, time out, duplicate, and return partial results. You need API versioning, tracing, retries, circuit breakers, service discovery, and deployment coordination. A microservice architecture without observability becomes harder to debug than a monolith.

In an interview, do not split everything immediately. Explain the initial simple design, then name the pressure that would force extraction: independent scaling, ownership, compliance, failure isolation, or release cadence.

**ML version:** ML systems naturally split over time because data ingestion, feature computation, training, model serving, evaluation, and monitoring have very different resource needs and release cycles. But splitting too early can hide data contracts and make reproducibility harder.

## §12 Monitoring and Observability

Monitoring tells you that something known is wrong. Observability gives you enough evidence to investigate something you did not predict.

The basic software signals are latency, traffic, errors, and saturation. Latency asks how long requests take. Traffic asks how many requests arrive. Errors ask how many fail. Saturation asks how close a resource is to being full: CPU, memory, disk, DB connections, queue lag, thread pools, GPU memory.

Metrics are numeric time series, such as p99 latency or error rate. Logs are event records, such as "payment provider timed out." Traces connect the path of one request across services. If a request touches gateway, user service, feature service, model server, and database, a trace shows which span was slow.

Alerts should map to action. "CPU above 60%" is often not actionable. "Checkout p99 above 2 seconds for 10 minutes and error rate above 1%" is closer to user harm. A good alert has an owner, severity, dashboard, and runbook.

Observability also shapes architecture. If every request has a request id, logs and traces can be joined. If every queue message has an idempotency key and source event id, retries can be debugged. If deploys annotate dashboards, regressions are easier to correlate.

**ML version:** ML adds silent failure. The API can return 200 OK while predictions become worse. So you also monitor feature distributions, prediction distributions, calibration, drift, label delay, model version, data freshness, and business metrics by slice. The [Production Loop](../loop/index.md) section is dedicated to this.

## Interview Questions

**Q1: What is the difference between a load balancer and an API gateway?**
A load balancer asks which replica of a service should receive the request and removes unhealthy replicas from rotation. A gateway asks whether the request is valid, who is making it, what API it is calling, and which internal service owns that route. In many systems both exist: load balancer first, gateway second, service replicas behind it.

**Q2: How do you keep a cache correct, and what is a cache stampede?**
Bound staleness with a TTL and use event-driven invalidation when important data changes; many systems use both, with TTL as a safety net. A stampede happens when a hot key expires and many requests miss at once, all hitting the database together. Mitigate with jittered TTLs, single-flight locking, background refresh, and serving slightly stale data while refreshing.

**Q3: Replication or sharding first, and why?**
Replication first, because it is simpler: copies add read capacity and availability, with replication lag as the main cost. Sharding splits ownership across machines to distribute writes and storage, but it makes queries, migrations, cross-shard transactions, and the shard-key choice into major operational commitments. Reach for sharding only when one machine cannot own all the writes or data.

**Q4: Why does `hash(key) % N` break when you add a node, and how does consistent hashing fix it?**
Changing N reshuffles most keys, causing a cache-miss storm or massive data movement. Consistent hashing maps keys and nodes onto a ring so a new node only takes a slice from its neighbors and most keys stay put. It does not fix hot keys, which still need replication, coalescing, or special routing.

**Q5: How is rate limiting different for an LLM API?**
Request count is not enough, because one request asking for 10 output tokens is cheap while another asking for 10,000 is expensive. AI systems rate-limit by input tokens, output tokens, GPU seconds, model class, or dollar budget, and distributed limiting needs atomic shared state so two gateways do not both spend the last token.

**Q6: What does observability add beyond monitoring for an ML system?**
Monitoring catches known failures; observability gives evidence to investigate the unpredicted. ML adds silent failure, where the API returns 200 OK while predictions degrade, so you also watch feature distributions, prediction distributions, calibration, drift, label delay, model version, data freshness, and business metrics by slice.
