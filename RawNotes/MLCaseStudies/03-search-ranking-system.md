# 03. Search Ranking System — ML System Design Case Study

**Company tags:** Google, Amazon, Flipkart, Microsoft (Bing), Airbnb, Etsy, Shopify
**Interview frequency:** Very high
**Why it matters:** Search shares the retrieve-then-rank funnel with recsys, but the query changes everything: you have a strong intent signal, your labels come from a different and equally biased place (human relevance judgments *plus* clicks), and the ranking stage has its own deep machinery — **learning-to-rank**, where the metric you care about (NDCG) is non-differentiable and you need a trick (LambdaMART) to optimize it. Search is also the literal substrate of every RAG interview, so the retrieval half transfers directly to cases 04 and 05. This document goes to the floor on LTR, relevance labels, and hybrid retrieval.

---

## How to use this document

Built two ways: a **thinking guide** (Sections 0–8) and a **worked one-hour interview transcript** (Section 9). Read the guide, then watch every transcript line map back to it. Internalize *the route, not the destination.*

> **The single most important habit:** every decision traces back to a number you stated up front (corpus size, QPS, latency) or a constraint the interviewer gave you.

> **The search-specific habit:** never say "I'd use embeddings" as if that solves search. The senior answer is "BM25 nails exact constraints, dense retrieval nails semantic intent, so I fuse them; then I rank with learning-to-rank, and I optimize NDCG via LambdaMART because NDCG itself is non-differentiable." Lexical-vs-semantic and how-you-actually-optimize-the-rank-metric are the depth axes.

---

## Section 0: The reusable scaffold (learn this ONCE)

```text
1. Clarify        -> turn an ambiguous product goal into a scoped problem + numbers
2. Frame as ML    -> what exactly are we predicting? what is the label? non-ML baseline?
3. Data & labels  -> where do labels come from, and WHY are they biased?
4. Baseline       -> simplest shippable thing, then name what breaks
5. Model          -> climb the ladder; explain ONE thing to the floor
6. Evaluation     -> offline metrics, online experiment, and the gap between them
7. Deploy         -> serving path / data (indexing) path / feedback path
8. Monitor        -> drift, delayed labels, retraining, fallback, incident response
```

**The three-path mantra, search flavor:** search is unusually clean about its paths. **Serving path** (query → understanding → retrieve → rank → rerank → results; tight latency). **Indexing/data path** (docs → clean/enrich → tokens + embeddings → inverted index + vector index; offline, batch). **Feedback path** (clicks/dwell/conversions → debiased relevance labels → next training set; where click bias enters). Draw all three; it signals production maturity.

---

## Section 1: Clarify requirements (and pin down NUMBERS)

### Interview prompt
> "Design a search ranking system for an e-commerce marketplace or enterprise knowledge base."

### The clarifying questions that actually change the design

| Question | Why it changes the design |
|---|---|
| **What's the corpus and query type?** E-commerce products, web docs, enterprise KB? | E-commerce has structured attributes (brand, price, size) and conversion as a label. Enterprise KB has ACLs (access control) and no clicks at first. Web has billions of docs. Different retrieval and labels. |
| **What's the success definition?** Click, purchase, "issue solved," dwell? | This is your label. E-commerce: purchase/add-to-cart (sparse, high-signal) vs click (abundant, biased). Support KB: "issue resolved." The objective defines the relevance label. |
| **How big is the corpus and what's QPS?** | 1M docs vs 100M products vs billions of web pages sets whether you can BM25-scan or must shard heavily, and how big the ANN index is. |
| **Latency budget?** | Search tolerates a bit more than typeahead but is still tight (p99 ~300ms), and that budget is split — the cross-encoder reranker is the hog, which is why it runs last on a tiny set. |
| **Personalized or not?** | Web/e-commerce personalize (user history as features). Legal/medical search often must *not* personalize (everyone sees the same authoritative result). |
| **Are there hard constraints?** In-stock, price range, ACLs, freshness, safety? | These are non-negotiable filters that ride alongside relevance — a perfect-relevance out-of-stock item is useless. Filters interact with ranking. |

> **Junior move:** "I'll embed queries and docs and do nearest-neighbor."
> **Senior move:** "What's the corpus size, the success label, and do I have hard constraints like in-stock or ACLs? Because pure semantic retrieval misses exact SKUs and constraints — I'll want hybrid lexical + dense, and the label decides whether I optimize for clicks or purchases."

### Pin the numbers (carry these through the ENTIRE answer)

```text
Corpus:               100M products (e-commerce marketplace)
Search QPS:           ~50,000 average, ~150,000 peak
Latency SLO:          p99 < 300ms end-to-end
Funnel:               100M docs --retrieval (BM25 + dense ANN)--> ~1,000
                      --LTR ranker--> ~100 --cross-encoder rerank--> top 10-20
```

**The latency budget split** (derive out loud):
```text
300ms total, roughly:
  query understanding (rewrite, intent, entity/attribute extraction)  ~25ms
  retrieval: BM25 + dense ANN (100M -> ~1,000)                          ~40ms
  feature fetch                                                          ~20ms
  LTR ranker (LambdaMART/GBDT over ~1,000 candidates)                    ~50ms
  cross-encoder rerank (top ~100 -> top 10)                             ~120ms  <- the hog
  merge / dedupe / business filters / format                            ~20ms
  overhead / network                                                     ~25ms
```
**The key latency argument (say it):** a cross-encoder jointly encodes (query, doc) and is the most accurate ranker, but that means one forward pass *per candidate* — you cannot run it on 1,000 docs in budget, only on ~100. That's exactly why the architecture is staged: cheap retrieval → mid LTR → expensive cross-encoder on a tiny set.

**Storage back-of-envelope:** 100M docs × 768-dim × 4B ≈ 300GB for the dense index → shard the ANN index and/or quantize (PQ/int8 → ~75GB). The inverted index for BM25 is separate and also sharded.

---

## Section 2: Frame it as an ML problem

- **Framing:** **query understanding → hybrid retrieval → learning-to-rank → cross-encoder rerank.** Say all four; each is a stage with a different job.
- **Prediction target:** **graded relevance** of a (query, document, user-context) triple — not binary. Search relevance is graded (e.g., Perfect / Excellent / Good / Fair / Bad), which is what NDCG consumes.
- **What is a positive label?** Two very different sources, and naming both is the signal: **(a) human relevance judgments** (raters following guidelines — the gold for offline eval, expensive, sparse) and **(b) implicit feedback** (clicks/dwell/purchases — cheap, abundant, *badly biased*). The whole label section is about reconciling these.
- **Non-ML baseline (always name one):** BM25 over an inverted index, plus filters (in-stock, category), synonyms, and popularity sort. This is genuinely strong for head queries and is your fallback. Do not skip it; lexical search is a real, hard-to-beat baseline.

> **Why this framing matters (say it):** separating retrieval (recall) from ranking (precision) from reranking (top-K precision) lets each stage use the right cost/feature budget, and separating human-judged relevance from click-relevance lets you *evaluate* honestly even though you *train* mostly on biased clicks.

---

## Section 3: Data, labels, and the relevance-label problem (the intellectual core)

Recsys's core was click bias. Search has the same click bias **plus** a second, subtler issue: clicks are not relevance, and human judgments don't match user intent. This reconciliation is where the case is won.

### Data sources
- **Query logs:** the query, reformulations, clicks, dwell, skips, purchases, the SERP shown and at what positions.
- **Documents/items:** title, description, attributes (brand, price, category), embeddings, quality/popularity, freshness, inventory.
- **Human judgments:** graded relevance labels from raters following guidelines (the offline gold standard).
- **Context:** user history (if personalized), device, location, session.

### Query understanding (search-specific, often skipped)
The query is structured signal you must extract:
- **Spelling/normalization, query rewriting** ("running shoes" ≈ "trainers").
- **Intent classification** (navigational vs informational vs transactional).
- **Entity / attribute extraction:** "red nike running shoes size 10" → {brand: nike, color: red, category: running-shoes, size: 10}. These become **hard filters**, not just ranking features. Missing this is why "I'll just embed the query" fails on e-commerce.

### The two label problems you MUST confront

**Problem 1 — clicks are biased, and not just by position.** Search has *more* click biases than recsys:
- **Position bias:** top results clicked because they're on top.
- **Trust/attention bias:** users trust the top of an authoritative engine and click position 1 even when 2 is better.
- **Caption/snippet bias:** an attractive title/snippet draws clicks regardless of landing-page relevance — a click can be on a *good caption for a bad page*.

**The fix — click models, to the floor.** A **click model** separates *examination* from *relevance*. The **Position-Based Model (PBM):**
```text
P(click | q, d, position) = P(examine | position) * P(relevant | q, d)
```
You estimate the examination propensity per position (e.g., via result randomization or an EM fit), then **divide it out** to recover an unbiased relevance signal. This feeds **unbiased learning-to-rank** (Joachims 2017): weight each click in the LTR loss by **inverse propensity** `1 / P(examine | position)`, so a click at position 10 (rarely examined) counts more than a click at position 1. Name PBM and IPW; it's the search version of recsys's position-feature trick, and it's more principled.

**Problem 2 — human judgments ≠ user satisfaction.** Raters judge "is this topically relevant" by guidelines; users care about price, availability, trust, freshness. So offline NDCG on human labels can rise while users are unhappier. The fix: maintain *both* label streams, eval offline on human judgments for stability, but let online behavior (debiased) be the arbiter.

> **Junior says:** "I'll train on clicks."
> **Senior says:** "Clicks are biased by position, trust, and caption. I'd fit a position-based click model to separate examination from relevance, train LTR with inverse-propensity weighting, and keep human graded judgments as the offline gold — because clicks and human labels each lie in different ways."

### Cold start
- **New item:** content embedding + attribute metadata gives a retrievable representation with zero query history; exploration surfaces it.
- **New query (tail):** dense retrieval shines here (semantic match) where BM25 has no co-occurrence; this is the main reason to add dense retrieval at all.

---

## Section 4: Baseline first — then name exactly what breaks

```text
RUNG 0: BM25 lexical search + filters + popularity sort
   -> inverted index; great for exact entities, SKUs, acronyms, constraints.
   BREAKS: no semantics -> "trainers" misses "running shoes"; weak on tail/ambiguous queries.
   TRIGGER: we need semantic recall for synonyms and intent.

RUNG 1: Dense (bi-encoder) retrieval
   -> embed query & doc, ANN nearest neighbor; handles synonyms, tail queries.
   BREAKS: misses EXACT constraints (a specific SKU, size, in-stock); needs ANN upkeep;
           can "hallucinate" topical-but-wrong matches.
   TRIGGER: we need BOTH lexical precision AND semantic recall.

RUNG 2: Hybrid retrieval (BM25 + dense, fused) + learning-to-rank   <-- PRODUCTION DEFAULT
   -> fuse lexical + dense candidates (RRF or as features), then a LambdaMART/GBDT
      ranker over rich features (BM25 score, dense sim, attributes, behavior).
   FIXES: precision + recall + a real ranking objective (NDCG).
   BREAKS / costs: needs relevance labels + click debiasing; two indices to maintain.
   TRIGGER (if pushed): top-K precision still not good enough on hard queries.

RUNG 3: Cross-encoder (or LLM) reranker on the top ~100   (advanced, last-stage only)
   -> jointly encode (query, doc) for max precision; or listwise LLM reranker (RankGPT-style).
   USE WHEN: top-K precision is the bottleneck and you can afford ~100 forward passes.
   COST: latency (the 120ms hog) -> ONLY on a tiny candidate set; consider distillation.
```

> **Say this:** "I'd ship BM25 day one — it's a strong, explainable baseline and my fallback. Production is hybrid retrieval plus LambdaMART, because that's the first rung with both lexical precision and semantic recall *and* a real ranking objective. I add a cross-encoder reranker only on the top ~100, because it costs a forward pass per candidate and I can't afford it earlier."

---

## Section 5: Learning-to-rank — ONE design, to the floor

Depth on one thing beats naming four. The thing to explain to the floor in search is **learning-to-rank, specifically why you can't just gradient-descend NDCG and how LambdaMART gets around it.** This is the search interview's signature mechanism.

### 5.1 The three LTR paradigms (know the difference cold)

```text
POINTWISE:  predict relevance grade per (q,d) independently (regression/classification).
            Problem: ranking is RELATIVE; pointwise ignores that doc A only needs to
            beat doc B for THIS query. Misweights easy vs hard queries.

PAIRWISE (RankNet): for a pair (d_i, d_j), model P(d_i > d_j) and minimize cross-entropy
            on the correct ordering. Optimizes # of inversions. Better, but treats all
            inversions equally -- swapping ranks 1&2 == swapping 99&100, which is wrong
            for a top-heavy metric like NDCG.

LISTWISE (LambdaRank/LambdaMART): optimize the LIST metric (NDCG) directly. THIS is the answer.
```

### 5.2 LambdaMART — the trick, to the floor

**The core problem:** NDCG involves a *sort*, so it's piecewise-constant and non-differentiable — you cannot take its gradient and backprop. This stumps people. The LambdaRank insight is to **skip the loss and define the gradient directly.**

```text
Start from the RankNet pairwise gradient for a pair (i, j): a force that pushes the
better document up and the worse document down.

LambdaRank's trick: SCALE that pairwise gradient by |ΔNDCG_ij| -- the change in NDCG
you'd get if you swapped documents i and j in the current ranking.

   lambda_ij = (RankNet gradient) * |ΔNDCG_ij|

So a swap that moves a relevant doc from rank 100 to rank 1 (huge ΔNDCG) gets a large
gradient; a swap deep in the tail (tiny ΔNDCG) gets a tiny one. You never differentiate
NDCG -- you weight pairwise forces by how much each swap matters to NDCG.

LambdaMART = these lambda gradients used inside gradient-boosted trees (MART).
LambdaLoss (2018) later showed LambdaRank optimizes a well-defined probabilistic loss,
giving it a proper theoretical footing.
```
This is the whole answer to "how do you optimize a non-differentiable ranking metric." Say it cleanly and you've demonstrated the depth most candidates lack. LambdaMART (GBDT) is still a top production ranker; the neural equivalent uses the same lambda-weighted or LambdaLoss objectives.

### 5.3 The features the LTR model eats
- **Relevance:** BM25 score, dense cosine similarity, field-level matches (title vs description), exact-match flags.
- **Document quality:** popularity, conversion rate, ratings, freshness, in-stock.
- **Query-doc:** attribute-match (did the extracted brand/size match?), historical CTR for this (query, doc) debiased.
- **Personalization (if allowed):** user-history affinity.

### 5.4 Hybrid fusion (how lexical + dense combine)
Two candidate lists (BM25, dense) must merge. Two standard ways:
- **Reciprocal Rank Fusion (RRF):** `score(d) = Σ_lists 1/(k + rank_list(d))`. Simple, robust, no tuning, no score normalization needed. A great default.
- **As features into the LTR model:** feed BM25 score and dense sim as features and let LambdaMART learn the combination. More powerful, needs labels.

### 5.5 Bi-encoder vs cross-encoder (the retrieval/rerank split, to the floor)
```text
BI-ENCODER (retrieval):  encode query and doc SEPARATELY -> two vectors -> cosine.
   Doc vectors PRECOMPUTED + ANN-indexed. One query encode at serve. Fast, scales to 100M.
   Weak: no token-level interaction between query and doc.

CROSS-ENCODER (rerank):  encode (query, doc) TOGETHER in one transformer -> relevance score.
   Sees full token interaction -> most accurate. But ONE forward pass PER candidate ->
   O(candidates) cost -> only affordable on the final ~100.

(Late-interaction, e.g. ColBERT, is the middle ground: token-level vectors with a cheap
 MaxSim operator -- more accurate than bi-encoder, cheaper than cross-encoder.)
```
This is the same factorization argument as recsys two-tower, now for text: the thing that scales must factorize; the thing that's most accurate can't, so it goes last on a small set.

### 5.6 The full architecture diagram (draw this)

```text
                         QUERY
                           |
        [ Query understanding: rewrite, intent, entity/attribute extraction ]
                           |
            +--------------+--------------+
            |                             |
   [ BM25 inverted index ]      [ Dense bi-encoder ANN ]   (100M docs)
            |                             |
            +-------- merge / RRF --------+
                           |
                  ~1,000 candidates (RECALL)
                           |
        [ LambdaMART LTR ranker ]  (BM25, dense sim, attrs, behavior features)
                           |
                  ~100 candidates (PRECISION)
                           |
        [ Cross-encoder / LLM reranker ]  (top-K precision)
                           |
        [ Filters: in-stock, ACL, safety, dedupe ] -> top 10-20
                           |
        (clicks/dwell/purchases + shown positions logged)
                           |
   [ Feedback path: click model (PBM) -> IPW-debiased labels -> retrain ]
```

---

## Section 6: Evaluation — offline metrics, interleaving, and the gap

### Offline metrics
- **Retrieval:** **Recall@K** (did the relevant docs survive retrieval?), coverage.
- **Ranking:** **NDCG@K** (graded, position-discounted — the headline metric), **MRR** (first relevant result, good for navigational), **MAP**. Computed on **human judgments** for stability.
- **Always slice:** head vs tail queries, by intent type, by locale. Tail queries are where semantic retrieval earns its keep; aggregate NDCG hides tail failures. Keep a **bank of bad SERPs**.

### Online metrics + the search-special method
- **Product:** search success rate, click-through to a satisfied action, **reformulation rate** and **zero-result rate** (lower is better), purchase/conversion.
- **Guardrails:** p99 latency, unsafe-result rate, duplicate rate, freshness, query coverage.
- **Interleaving (the search-specific online eval — name it):** instead of (or before) a full A/B, **Team-Draft Interleaving** mixes ranker A's and ranker B's results into a *single* SERP and attributes each click to whichever ranker contributed it. It's far more **sensitive** than A/B (same user sees both, controlling for query/user variance), so you need ~10–100x less traffic to detect a ranking difference. Mention interleaving as the first online filter, with A/B for the business-metric confirmation.

### The offline→online gap (THE talking point)

> **Classic question: "Offline NDCG improved but online success dropped. Why?"**

1. **Human-label vs user-intent mismatch** — you got better at topical relevance, but users care about price/availability/freshness the raters ignored.
2. **Click-label bias** — offline trained on position-biased clicks; you got better at predicting the *old* ranking's click pattern.
3. **Query distribution shift** — you optimized head queries; the tail (where most volume cumulatively lives) regressed.
4. **Caption/snippet effect** — the right doc ranked higher but its snippet doesn't convey relevance, so users don't click.
5. **Freshness/inventory** — offline corpus snapshot is stale vs live inventory.

**The lesson:** offline NDCG on human labels is a stable *filter*; interleaving is the sensitive online arbiter for ranking quality; A/B on conversion confirms business impact.

### A concrete A/B test (fully specified)

```text
Hypothesis:        The hybrid+LambdaMART ranker raises search-driven conversion
                   and lowers reformulation rate without hurting latency.
Pre-filter:        Team-Draft INTERLEAVING first -- cheap, sensitive ranking signal
                   to decide if the new ranker is even worth a full A/B.
Unit of randomization: user (consistent experience).
Control / Treatment:   current ranker / new ranker.
Primary metric:    search-driven conversion (or satisfied-click rate).
Guardrails:        p99 latency, zero-result rate, reformulation rate, unsafe-result
                   rate, revenue. Any breach = auto-rollback.
Ramp:              1% -> 5% -> 20% -> 50%, holding to check guardrails.
Sample size / MDE: power for ~1% relative conversion lift; slice by head/tail because
                   tail volume is large in aggregate and easy to regress.
Duration:          >= 1-2 weeks for weekly seasonality + query-mix variation.
Decision/rollback: ship if conversion up and guardrails hold.
```

Mention **interleaving** explicitly — it's the single best "I've actually evaluated search" signal.

---

## Section 7: Deployment & serving (the three paths)

### Serving path (latency-critical)
```text
query -> understanding -> BM25 + dense ANN (parallel) -> RRF merge
      -> LambdaMART rank -> cross-encoder rerank (top 100) -> filters -> top 10-20
```
- Run BM25 and dense retrieval **in parallel**, merge, then rank — keep them as separate services.
- Shard/quantize the dense ANN index (the 300GB estimate). Cache hot-query results.
- The cross-encoder is the latency hog → cap candidates hard, batch, and consider distilling it into the LTR features.

### Indexing / data path (offline, batch)
```text
docs/items -> clean / normalize / enrich -> tokenize (inverted index)
           + embed (vector index) -> attach metadata + ACLs
           -> incremental index refresh (freshness/inventory)
```
- Two indices to keep in sync (inverted + vector). Incremental refresh for freshness/inventory.
- **Log the served features and positions** (kills skew, enables click-model debiasing).

### Feedback path (where click bias enters)
```text
clicks + dwell + purchases + shown positions -> fit position-based click model
   -> IPW-debiased relevance labels -> next LTR training set -> retrain
```

### Rollout discipline
```text
shadow -> interleaving (sensitive ranking check) -> canary -> A/B (ramp)
```

---

## Section 8: Monitoring, retraining, incident response

- **Monitor:** zero-result rate and reformulation rate (the canaries of broken retrieval), latency p95/p99, NDCG on a held-out judged set, **query distribution drift** (new trending queries the index doesn't cover), index freshness/inventory staleness, and conversion KPI.
- **Retraining:** ranker retrained on a schedule (weekly-ish) plus triggered on drift; embeddings re-trained less often; the **index refreshed continuously** for freshness/inventory (a different cadence from the model).
- **Fallback:** on ranker/ANN failure, degrade to BM25 + popularity (Rung 0) → cached results. Never zero results; a zero-result page is the worst search outcome.
- **Incident response:** freeze the ranker, diff against last release, inspect a bank of bad SERPs, check whether retrieval (recall) or ranking (precision) broke, roll back on guardrail breach.

---

## Section 9: The worked one-hour interview (full transcript)

---

**[00:00 — The prompt]**

**INTERVIEWER:** Design a search ranking system for an e-commerce marketplace.

**YOU:** Before designing, a few scoping questions. First, what's the corpus size and QPS — that sets the index architecture. Second, what's my success label — clicks, add-to-cart, or purchase — because that decides what I optimize. Third, latency budget. Fourth, do I have hard constraints like in-stock, price, or ACLs, since a perfectly relevant out-of-stock item is useless. And fifth, do I personalize.

**INTERVIEWER:** 100M products, ~50K QPS, success is conversion, latency p99 300ms, yes there are in-stock and category filters, light personalization is fine.

**YOU:** Good, let me pin numbers.
```
100M products, ~50K QPS (~150K peak), p99 < 300ms
funnel: 100M --BM25+dense ANN--> ~1000 --LambdaMART--> ~100 --cross-encoder--> top 10
```
The 300ms over 100M docs forces a staged funnel, and the staging is driven by one fact: the cross-encoder, my most accurate ranker, costs one forward pass per candidate, so I can only run it on ~100, not 1000. That's why retrieval, mid-ranking, and reranking are separate stages with different cost budgets.

---

**[00:06 — Framing & labels]**

**YOU:** I'll frame it as query understanding, then hybrid retrieval, then learning-to-rank, then a cross-encoder rerank. The prediction target is graded relevance of a query-doc-context triple. My non-ML baseline is BM25 plus filters and popularity sort, which is genuinely strong for head queries and is my fallback. The interesting part is the label: I have two sources that each lie. Human relevance judgments are my offline gold but they're topical and ignore price and availability; clicks are abundant but badly biased.

**INTERVIEWER:** Why not just train on clicks? You have tons of them.

**YOU:** Because search clicks are biased three ways, more than recsys. Position bias, top results get clicked for being on top. Trust bias, users click position one because they trust the engine. And caption bias, an attractive snippet gets the click even if the page is bad. So I'd fit a position-based click model that factors P(click) into P(examine given position) times P(relevant), estimate the examination propensity, and train learning-to-rank with inverse-propensity weighting so a click at position ten counts more than a click at position one. And I keep human graded judgments as the stable offline metric, because clicks and human labels lie in different directions.

---

**[00:14 — Query understanding]**

**INTERVIEWER:** You mentioned query understanding. What's in it?

**YOU:** For e-commerce it's load-bearing, not decoration. Spelling and rewriting, intent classification, and crucially entity and attribute extraction: "red nike running shoes size 10" parses into brand nike, color red, category running shoes, size 10. Those become hard filters, not just ranking features. This is exactly why "just embed the query" fails on e-commerce — embeddings blur the size-10 constraint, and the user does not want size 9.

---

**[00:20 — The ranker, to the floor]**

**INTERVIEWER:** Let's go deep on ranking. How do you actually train it?

**YOU:** I climb a quick ladder — BM25, then dense bi-encoder retrieval which adds semantic recall for synonyms and tail queries but misses exact constraints, then my production answer: hybrid retrieval fused, then a LambdaMART ranker. Let me go to the floor on the ranker, because the interesting bit is that I want to optimize NDCG, and NDCG is non-differentiable — it has a sort in it, so I can't backprop it.

The LambdaRank trick is to skip the loss and define the gradient directly. I start from the RankNet pairwise gradient — a force pushing the better doc up and the worse doc down for a pair — and I scale that force by the change in NDCG I'd get from swapping those two documents. So a swap that lifts a relevant doc from rank 100 to rank 1 gets a big gradient; a swap deep in the tail gets a tiny one. I never differentiate NDCG; I weight pairwise forces by how much each swap moves NDCG. LambdaMART is those lambda gradients inside gradient-boosted trees, and it's still a top production ranker. LambdaLoss later gave it a proper probabilistic loss.

**INTERVIEWER:** And the retrieval — why two indices, and why is the cross-encoder last?

**YOU:** Hybrid because BM25 nails exact SKUs and constraints while dense retrieval nails synonyms and tail intent; I fuse them with reciprocal rank fusion or feed both scores as features into the ranker. The bi-encoder factorizes — doc vectors precomputed and ANN-indexed, one query encode at serve — so it scales to 100M. The cross-encoder jointly encodes query and doc so it sees full token interaction and is most accurate, but that's one forward pass per candidate, so it only runs on the final ~100. Same factorization logic as a two-tower recsys, just for text.

---

**[00:32 — Evaluation]**

**INTERVIEWER:** How do you evaluate, and here's a scenario: offline NDCG goes up but online conversion drops. Why?

**YOU:** Offline I use NDCG and MRR on human judgments, sliced by head versus tail intent, plus a bank of bad SERPs. But the search-specific move online is interleaving before a full A/B: Team-Draft Interleaving mixes the two rankers into one results page and attributes clicks, which controls for query and user variance and is ten to a hundred times more sensitive than A/B, so I can detect a ranking change with far less traffic.

On your scenario — NDCG up, conversion down — the likeliest cause is human-label versus user-intent mismatch: I improved topical relevance but the raters ignored price and availability, which drive conversion. Other causes: I overfit position-biased clicks and just learned the old ranking, the tail regressed while the head improved, or the right doc ranked higher but its snippet doesn't convey relevance so nobody clicks. I'd slice head versus tail and check inventory freshness first.

**INTERVIEWER:** Design the A/B.

**YOU:** Interleaving as the cheap pre-filter, then a user-randomized A/B with conversion as primary, guardrails on latency, zero-result rate, reformulation rate, and revenue, ramp 1-5-20-50, sliced head versus tail because tail volume is large in aggregate, run two weeks for query-mix seasonality, auto-rollback on guardrail breach.

---

**[00:42 — Serving, indexing, monitoring]**

**INTERVIEWER:** What runs in production?

**YOU:** Three paths. Serving: query understanding, BM25 and dense ANN in parallel, RRF merge, LambdaMART, cross-encoder on top 100, then filters — separate services, ANN sharded and quantized since 100M times 768-dim is about 300GB. Indexing path: clean and enrich docs, build the inverted and vector indices, attach metadata and ACLs, incremental refresh for inventory and freshness — and I log served features and positions. Feedback path: clicks, dwell, purchases, and shown positions feed the position-based click model that produces IPW-debiased labels for the next training run.

Monitoring: zero-result and reformulation rates as the canaries of broken retrieval, latency, NDCG on a judged holdout, query drift for new trending terms, and inventory staleness. Index refreshes continuously for freshness, the ranker retrains weekly plus on drift. Fallback degrades to BM25 plus popularity, then cached results — never a zero-result page, which is the worst outcome in search.

---

**[00:54 — The close]**

**INTERVIEWER:** Anything to add?

**YOU:** To restate: I optimize graded relevance for conversion via hybrid retrieval and a LambdaMART ranker, with a cross-encoder rerank where I can afford it, protected by latency, zero-result, and safety guardrails. The two things that make search its own case are that NDCG is non-differentiable so I optimize it with LambdaMART's swap-weighted gradients, and that my labels need click-model debiasing because search clicks are biased beyond just position. Concretely, this retrieve-then-rerank funnel with a cross-encoder is the exact pattern I've shipped for product discovery, so the bi-encoder/cross-encoder split and the offline-online label gaps are familiar production ground for me.

**INTERVIEWER:** Strong answer.

---

> **Why this transcript works (study these moves):**
> 1. **Asked the corpus/label/constraint questions** — recognized in-stock/ACL filters reshape the design.
> 2. **Hybrid, not "just embeddings"** — named lexical-vs-semantic explicitly.
> 3. **Query understanding as load-bearing** — attribute extraction → hard filters.
> 4. **LambdaMART to the floor** — the non-differentiable-NDCG trick, cleanly.
> 5. **Click models + IPW** — search's deeper label-bias story.
> 6. **Bi-encoder vs cross-encoder factorization** — the latency argument for staging.
> 7. **Interleaving** — the "I've actually evaluated search" signal.
> 8. **Zero-result rate as the worst outcome** — operational maturity.

---

## Section 10: Junior vs Senior — the highest-leverage contrast

| Decision | Junior answer | Senior answer |
|---|---|---|
| Retrieval | "Embed queries and docs, do ANN." | "Hybrid BM25 + dense — lexical nails SKUs/constraints, dense nails synonyms/tail; fuse via RRF." |
| Query | "Pass the query to the encoder." | "Query understanding extracts entities/attributes that become *hard filters*, not just features." |
| Ranking metric | "Optimize NDCG." | "NDCG is non-differentiable; LambdaMART scales pairwise gradients by ΔNDCG to optimize it." |
| LTR paradigm | "Train a ranker on relevance." | "Listwise (LambdaMART) — pointwise/pairwise ignore that swaps near the top matter more." |
| Labels | "Train on clicks." | "Clicks are position/trust/caption-biased; fit a PBM click model, train with IPW; keep human judgments as offline gold." |
| Rerank | "Use a cross-encoder." | "Cross-encoder is 1 forward pass/candidate → only on top ~100; bi-encoder factorizes for the 100M retrieval." |
| Online eval | "Run an A/B test." | "Interleaving first (10-100x more sensitive for ranking), then A/B on conversion." |
| Offline-up/online-down | "Maybe overfitting." | "Human-label vs user-intent mismatch, click bias, tail regression, caption effect, stale inventory." |
| Failure to avoid | "Search returns something." | "Zero-result and reformulation rates are the worst outcomes; fall back to BM25, never empty." |

---

## Section 11: One-page cheat sheet (whiteboard recall)

```text
SCAFFOLD: Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor

NUMBERS:  corpus, QPS, p99 ~300ms. dense index = N*dim*4B (100M*768*4 ~ 300GB -> shard/quantize).
          funnel 100M --BM25+dense--> 1000 --LambdaMART--> 100 --cross-encoder--> 10.

STAGES:   query understanding (rewrite/intent/ATTRIBUTE extraction -> hard filters)
          -> hybrid retrieve (BM25 + dense, fuse via RRF)
          -> LTR rank -> cross-encoder rerank -> filters (in-stock/ACL/safety).

LTR (the core):
   pointwise (ignores relativity) < pairwise RankNet (all inversions equal)
   < LISTWISE LambdaMART: optimize NDCG. NDCG non-differentiable (it sorts) ->
   lambda_ij = RankNet gradient * |ΔNDCG from swapping i,j|. Never differentiate NDCG.

RETRIEVE vs RERANK:
   bi-encoder: encode q & d SEPARATELY -> precompute doc vecs -> ANN -> scales to 100M.
   cross-encoder: encode (q,d) TOGETHER -> most accurate -> 1 pass/candidate -> top ~100 only.
   (ColBERT late-interaction = middle ground.)

LABELS:   human judgments (graded, offline GOLD) vs clicks (biased: position+trust+caption).
   fix: position-based click model P(click)=P(examine|pos)*P(rel); train LTR with IPW.

EVAL:     offline NDCG@K/MRR on human labels, slice head/tail, bank of bad SERPs.
          INTERLEAVING (Team-Draft) = sensitive online ranking test, then A/B on conversion.
          offline-up/online-down: human-label vs intent mismatch, click bias, tail regression,
          caption effect, stale inventory.

3 PATHS:  serving (BM25||dense -> rank -> rerank) | indexing (inverted + vector, refresh)
          | feedback (click model -> IPW labels -> retrain)

MONITOR:  zero-result + reformulation rate (canaries), latency, NDCG holdout, query drift,
          inventory staleness. fallback: BM25+popularity -> cache. NEVER zero results.
```

---

## Section 12: Follow-up questions the interviewer may ask

- **What changes at huge scale?** Shard inverted + vector indices, run BM25/dense in parallel, cache hot queries, cap and batch the cross-encoder (or distill it), async logging, p95/p99 SLOs.
- **How do you handle cold start?** New items: content embeddings + attributes for retrievability + exploration; new/tail queries: dense retrieval (semantic match where BM25 has no co-occurrence).
- **How do you pick top-K at each stage?** Retrieval K by recall vs latency (can the ranker score them in time?); rerank K by what the cross-encoder can afford; tune on validation, confirm via interleaving.
- **Offline NDCG up, online down — why?** Human-label vs user-intent mismatch, click bias, tail regression, caption/snippet effect, stale inventory. Slice head/tail, check freshness.
- **How do you debias clicks?** Position-based (or cascade/DBN) click model to separate examination from relevance; inverse-propensity-weighted LTR; result randomization to estimate propensities.
- **BM25 vs dense vs hybrid — when?** BM25 for exact constraints/head; dense for synonyms/tail; hybrid is the default because each covers the other's blind spot.
- **Why a cross-encoder only at the end?** O(candidates) cost — one joint forward pass per doc; affordable only on the final ~100.
- **How do you evaluate ranking cheaply online?** Interleaving — far more sensitive than A/B for ranking changes.

---

## Section 13: Common mistakes (anti-patterns to avoid)

- "Just embed everything" — ignoring that BM25/lexical wins exact constraints and that attribute extraction drives hard filters.
- Naming a cross-encoder for retrieval (can't afford O(corpus) forward passes) — confusing the bi-encoder/cross-encoder roles.
- Listing NDCG without explaining that it's non-differentiable and how LambdaMART optimizes it.
- Training on raw clicks without a click model / IPW debiasing.
- Confusing human-judgment relevance with user satisfaction (the offline-online gap).
- Forgetting interleaving as the sensitive online ranking evaluator.
- Ignoring zero-result / reformulation rate (the real product failure signals).
- Treating the index as static — missing freshness/inventory refresh as a separate cadence.
- Treating deployment as "an endpoint" instead of serving/indexing/feedback paths.

---

## Section 14: Transfer — what mastering search unlocks

| Problem | What changes vs search | What stays identical |
|---|---|---|
| **Recommendation (case 01)** | no query; user is the "query" | retrieve→rank funnel, bi-encoder, position bias |
| **Production RAG (case 05)** | retrieval feeds an LLM, not a SERP; chunking | hybrid retrieval, rerank, the bi/cross-encoder split |
| **Enterprise copilot (case 04)** | retrieval + generation + grounding | query understanding, hybrid retrieval, ACLs |
| **Ads (case 11)** | sponsored results + auction + calibration | LTR, position bias, NDCG |
| **Autocomplete (case 18)** | prefix-time ranking, tail latency | query understanding, ranking features |
| **QA / semantic search** | passage-level relevance | dense + cross-encoder rerank |

The leverage: **the entire retrieval half of every RAG case is this document.** Master hybrid retrieval + the bi/cross-encoder split here and cases 04, 05, and 15 are mostly "add an LLM on top."

---

## Sources
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research, Advances in TF-Ranking: https://research.google/blog/advances-in-tf-ranking/
- From RankNet to LambdaRank to LambdaMART: An Overview (Burges, MSR-TR-2010-82): https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- The LambdaLoss Framework for Ranking Metric Optimization (Wang et al., CIKM 2018): https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/
- Unbiased Learning-to-Rank with Biased Feedback (Joachims et al., WSDM 2017, position-based click model + IPW): https://arxiv.org/abs/1608.04468
- Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020, bi-encoder retrieval): https://arxiv.org/abs/2004.04906
- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction (Khattab & Zaharia, 2020): https://arxiv.org/abs/2004.12832
- Reciprocal Rank Fusion (Cormack et al., SIGIR 2009): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
