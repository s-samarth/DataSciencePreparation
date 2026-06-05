# 05. Production RAG / Enterprise Search System

**Company tags:** Every AI startup, Google, Microsoft, Amazon, Glean, Anthropic ecosystem
**Interview frequency:** Very high for LLM / applied-AI roles
**Why it matters:** RAG is the default LLM system-design case because it forces you to be precise about *retrieval quality*. Most candidates can draw the pipeline. Very few can tell the interviewer **which box is failing and how they would measure it**. That gap is the whole interview.

---

## 0. How to use this doc

This document is built two ways, and you should read it twice.

1. **As a thinking guide.** The section headers are the order you should attack the problem on a whiteboard: clarify, frame, data/chunking, baseline, retrieval, generation, eval, deploy, monitor. Internalize the *order* and the *triggers* for moving up each ladder rung. That is what transfers to every other case.
2. **As a worked transcript.** Section 11 is a full one-hour interview, timestamped, with an interviewer who pushes. Read it after you understand the pieces, and use it for active recall: cover the `YOU:` lines and try to answer before reading.

The one idea to carry out of this doc: **a RAG system has two independent failure surfaces — retrieval and generation — and a senior answer keeps them separate at every step (data, model, metric, monitoring).** If you blur them, you cannot debug the system and you will say "accuracy" when you mean four different things.

The reusable scaffold (identical across all cases in this set):

```
Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor
```

---

## 1. The reusable scaffold, stated once

Every case in this folder uses the same spine. Memorize it once:

| Phase | The question you are answering |
|---|---|
| Clarify | What exactly are we building, for whom, under what constraints? |
| Frame | What is the learnable target, and what is the non-ML baseline? |
| Data / Labels | Where does signal come from, and how is it biased? |
| Baseline | What is the simplest thing that ships, and what breaks it? |
| Model | One architecture, explained to the floor: losses, serving, cost. |
| Eval | The offline to online gap, and one fully specified A/B test. |
| Deploy | Three paths: serving, data, feedback. Plus rollout discipline. |
| Monitor | What pages someone at 3am, and what is the fallback? |

For RAG specifically, the **Model** phase splits into two sub-systems — the retriever and the generator — and you must talk about them separately or you will sound junior.

---

## 2. Clarify requirements (scripted)

Do not start drawing. Spend the first three minutes asking questions that *change the design*. Below is the script, and for each question, why the answer moves the architecture.

| Question | Why it changes the design |
|---|---|
| "Internal employee search or customer-facing support deflection?" | Internal means heterogeneous corpora (wikis, code, tickets, Slack) and hard ACL requirements. Customer-facing means a narrower curated KB but stricter hallucination tolerance and public latency SLOs. |
| "Is the answer a generated paragraph with citations, or a ranked list of documents?" | Generation adds a whole grounding/faithfulness failure surface and LLM cost. Pure ranked-list search is the older 'enterprise search' problem with no generation eval. I will assume **generated, cited answers**. |
| "How fresh must answers be? Seconds, hours, or daily?" | Freshness sets the ingestion architecture. Daily means batch re-embed. Minutes means a streaming index + incremental upsert + cache invalidation, which is much more infra. |
| "Are permissions per-document (ACLs)?" | If yes, retrieval must be permission-aware: I cannot let user A see a chunk from a doc only user B can read. This forces ACL filtering *inside* the retrieval call, not after. (Deep treatment lives in the copilot case; here I keep it as a hard filter.) |
| "What is the latency SLO, and is the answer streamed?" | Streaming hides generation latency behind time-to-first-token. If the product streams, my budget is dominated by retrieval + rerank + TTFT, not full-answer time. |
| "What does success mean to the business?" | Deflection rate (tickets avoided) for support; task-completion / time-saved for internal. This determines the north-star metric and the A/B readout. |

**The numbers I will commit to and carry through the whole answer** (state these out loud so the interviewer can challenge them):

- **Corpus:** 2M source documents (KB articles, resolved tickets, policy docs).
- **Chunks:** ~20M chunks after splitting (avg ~10 chunks/doc).
- **Embedding:** 1024-dim, fp16.
- **Traffic:** 5M end users, ~100 QPS peak on the answer endpoint.
- **Latency SLO:** p95 < 3s end-to-end, **streamed**, so TTFT < 1s is the real constraint.
- **Cost target:** < $0.05 per answered query, all-in.
- **Freshness:** new/updated docs searchable within ~5 minutes.

### Latency budget, derived out loud

I have 3s p95 to the *last* token, but because we stream, the constraint users feel is time-to-first-token. I split the pre-generation budget:

```
Query rewrite (small LLM)      ~150 ms
Embed the query                 ~15 ms
Hybrid retrieval (BM25 + ANN)   ~50 ms   (parallel, take the max)
Cross-encoder rerank (100->10) ~250 ms
-----------------------------------------
Pre-generation total           ~465 ms
LLM time-to-first-token        ~500 ms
-----------------------------------------
TTFT budget                    ~965 ms   -> under 1s, OK
Full answer streams over the remaining ~2s.
```

If any stage blows its slice, I know exactly where to cut: drop rerank depth from 100 to 50, or skip query rewrite for short queries.

### Storage back-of-envelope

20M chunks × 1024 dims × 2 bytes (fp16) = **~40 GB** of raw vectors. An HNSW graph adds ~1.5x overhead → ~60 GB, which fits in RAM on a couple of nodes. If the corpus grows 10x, I move to **product quantization (int8/PQ)** → ~10 GB and trade a few points of recall for memory, recovered by reranking. Stating this shows I know the index is a memory-bound system, not a magic box.

---

## 3. Frame as an ML problem

- **Framing:** retrieval-augmented generation — retrieve a small set of authorized, relevant chunks, then condition a generator on them to produce a grounded, cited answer.
- **The target is two targets.** Retrieval target: did the top-K context contain the information needed to answer? Generation target: is the answer supported *only* by that context, and does it actually address the question?
- **Why split it:** if I optimize "answer quality" as one blob I cannot tell whether a wrong answer was caused by missing context (retriever's fault) or fabrication (generator's fault). The fixes are completely different. This single distinction is the senior signal in this case.
- **Non-ML baseline:** BM25 keyword search returning document snippets, no generation. Ships in a day, and you would be surprised how far it gets on exact-term queries.

---

## 4. Data and chunking — confront the real problem head-on

Most candidates wave at "chunk the docs." The domain's actual hard problem is that **chunking decides your retrieval ceiling before any model runs.** Bad chunks cap recall no matter how good your embeddings are.

### The chunking failure modes

- **Too small (e.g. 128 tokens):** a single chunk lacks enough context to be self-contained. "It expires after 30 days" is useless without knowing what "it" is. Embeddings of tiny chunks are semantically mushy.
- **Too large (e.g. 2000 tokens):** the embedding averages over many topics, so it matches everything weakly and nothing strongly. You also waste context-window budget and money stuffing irrelevant text into the prompt.
- **Naive fixed-size splitting:** cuts mid-sentence and mid-table, destroying meaning at boundaries.

### What I actually do

- **Structure-aware splitting:** split on document structure first (headings, sections, list items, table rows), then pack to a target size. A KB article's H2 sections are natural semantic units.
- **Target ~300-500 tokens with ~50-token overlap.** Overlap prevents an answer that straddles a boundary from being lost. These are starting numbers I would tune against a retrieval eval set, not gospel.
- **Attach metadata to every chunk:** source doc ID, title, section path, last-updated timestamp, source authority (official KB > old ticket), and the **ACL** (who may see it). Metadata is half the system — it powers filters, freshness boosts, and citations.
- **Store the parent.** I embed the small chunk for *matching* but, at generation time, I can expand to the parent section for *context* (small-to-big / parent-document retrieval). This decouples "what retrieves well" from "what reads well."

### Labels — the honest part

There is rarely a clean labeled set of (query, correct-chunk) pairs. Where signal comes from, in increasing cost and quality:

1. **Implicit:** which cited chunk did the user click? Did the answer resolve the ticket (no follow-up, thumbs-up, ticket closed)? Cheap, abundant, **biased** — position bias on citations, and "no follow-up" can mean "satisfied" or "gave up."
2. **Synthetic eval set:** prompt a strong LLM to generate questions *from* a chunk; that chunk is then the gold context for that question. This bootstraps a retrieval eval set on day one with zero human labeling. Validate a sample by hand so you trust it.
3. **Human-labeled golden set:** a few hundred real queries with expert-annotated correct answers and source chunks. Expensive, slow, and the **ground truth you anchor everything else to.** This is what you regression-test against before every launch.

Bias to name out loud: **survivorship in the corpus.** If the KB never documented a problem, no retrieval fix will answer it — that is a content gap masquerading as a model failure, and you must monitor "retrieved nothing relevant" rate to catch it.

---

## 5. Baseline -> why it breaks -> next rung

State the ladder with explicit triggers. Never jump to the top.

| Rung | What it is | Why it breaks -> trigger to climb |
|---|---|---|
| 0 | BM25 keyword search, return snippets, no LLM. | Misses synonyms and paraphrases ("can't log in" vs "authentication failure"). Trigger: recall is bad on real queries that don't share vocabulary with docs. |
| 1 | Dense retrieval (embed query + chunks, ANN top-K) + LLM generation. | Misses **exact entities**: error codes, SKUs, version numbers, names. Embeddings smooth over the very tokens that matter. Trigger: failures on precise-identifier queries. |
| 2 | **Hybrid (BM25 + dense) fused, then cross-encoder rerank, then generate with citations.** | This is the production default. Trigger to leave it: multi-hop questions that need reasoning across documents. |
| 3 | Corrective/Self-RAG, query decomposition, graph RAG. | Higher accuracy on multi-hop, but much harder to eval and operate. Trigger: a measured population of multi-hop queries failing at rung 2, *and* you have the eval maturity to validate the added complexity. |

The interviewer wants to hear you **earn** rung 2 by explaining exactly what breaks at rungs 0 and 1. Naming rung 3 first, unprompted, is a junior tell.

---

## 6. The architecture, explained to the floor

I will design rung 2 and explain every box: what it does, why it exists, the loss/mechanics, and the cost.

```
                       Query
                         |
                  Query rewrite (small LLM):
                  spell-fix, expand acronyms,
                  resolve "it"/"that" from history
                         |
        +----------------+-----------------+
        |                                  |
   BM25 / sparse                    Dense / ANN (HNSW)
   (exact terms)                    (semantic match)
        |   top-100                       |   top-100
        +----------------+-----------------+
                         |
              Fusion (Reciprocal Rank Fusion)
                         |  ~150 candidates
                  ACL filter (drop chunks
                  the user may not see)
                         |
              Cross-encoder reranker
              (score each [query, chunk] pair)
                         |  top-8
              Context assembly (expand to
              parent sections, dedupe, order)
                         |
                LLM generation with a
              "cite or abstain" instruction
                         |
              Grounding check (claims map
              to cited spans?) -> citations
                         |
                   Cited answer
```

### Why two retrievers, and how I fuse them

- **BM25 (sparse)** scores on exact lexical overlap. It nails error codes, product names, version numbers — the tokens dense models blur.
- **Dense (bi-encoder)** embeds query and chunk *independently* into the same vector space; an ANN index (HNSW) finds nearest neighbors in ~50ms over 20M vectors. It catches paraphrase and synonymy.
- They fail on opposite query types, so I run both and fuse. **Reciprocal Rank Fusion** is the workhorse: `score(d) = sum over retrievers of 1/(k + rank_r(d))` with `k≈60`. RRF needs no score calibration between the two systems (it uses ranks, not raw scores), which is why it is robust and the standard default.

### The reranker — the box candidates skip, and the one that matters most

The bi-encoder is fast because it embeds query and chunk *separately* — but that means the model never sees them *together*, so it cannot reason about fine-grained relevance. The **cross-encoder reranker** does: it takes `[query, chunk]` as a single input, runs full attention across both, and outputs one relevance score. Far more accurate, far more expensive — O(candidates) forward passes per query.

This is the classic **retrieve-then-rerank factorization**, the same latency argument as recommendation funnels:

- Cheap bi-encoder + ANN narrows 20M → 150 candidates (can't afford the cross-encoder here).
- Expensive cross-encoder reranks 150 → top-8 (can't afford to skip it here — this is where precision comes from).

Reranking 100-150 pairs costs ~250ms on a GPU, which is why it sits in my latency budget. If I need to cut latency, this depth is the first knob. Modern rerankers: Cohere Rerank, BGE-reranker, or a fine-tuned cross-encoder on your own click data.

### The generator — grounding is a *design choice*, not a hope

The prompt instructs: answer **only** from the provided context, cite the chunk ID for each claim, and **abstain** ("I don't have enough information") if the context is insufficient. Abstention is a feature: a senior system would rather say "I don't know" than fabricate, because in support a confident wrong answer is worse than no answer.

- **Citation enforcement:** parse the answer for claims and verify each maps to a retrieved chunk; strip or flag unsupported claims.
- **Grounding / faithfulness check:** an entailment step (NLI model or LLM-judge) asks "is this sentence supported by the cited span?" Unsupported sentences are the definition of hallucination here, and this check is how you *measure* it, not just hope it away.
- **Model tiering for cost:** route easy/short queries to a cheap small model; escalate only ambiguous or high-stakes queries to a frontier model. With ~100 QPS, generation dominates cost, so tiering is the main lever to hit < $0.05/query.

### Canonical references (verified)

- RAG, original formulation — Lewis et al., 2020: https://arxiv.org/abs/2005.11401
- Dense Passage Retrieval (bi-encoder retrieval) — Karpukhin et al., 2020: https://arxiv.org/abs/2004.04906
- Reciprocal Rank Fusion — Cormack et al., 2009: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- RAGAS (reference-free RAG eval) — Es et al., 2023: https://arxiv.org/abs/2309.15217
- Self-RAG (retrieve/critique on demand) — Asai et al., 2023: https://arxiv.org/abs/2310.11511
- Corrective RAG — Yan et al., 2024: https://arxiv.org/abs/2401.15884
- Google Research, role of sufficient context: https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/

---

## 7. Evaluation — localize failures with the RAGAS quadrant

This is the heart of a senior RAG answer. The single move: **decompose quality into a retrieval axis and a generation axis, each with its own metric, so any failure points at one box.**

|  | Needs ground-truth answer? | Measures | Failure it localizes |
|---|---|---|---|
| **Context recall** | Yes | Of the facts needed to answer, what fraction appear in retrieved context? | **Retriever miss** — the answer wasn't there to find. |
| **Context precision** | Partial | Are the relevant chunks ranked at the top, vs buried under noise? | **Reranker / fusion** — right chunk retrieved but not surfaced. |
| **Faithfulness** | No | Is every claim in the answer supported by the retrieved context? | **Generator hallucination** — invented beyond the context. |
| **Answer relevance** | No | Does the answer actually address the question (vs technically-correct-but-off-topic)? | **Generator drift** — ignored the user's real intent. |

The top row is the retriever's report card; the bottom row is the generator's. Two of these (faithfulness, answer relevance) need **no labeled answer**, so I can run them continuously in production on live traffic. That is how you monitor a system that has no ground truth at serving time.

### The offline-to-online gap, including the classic trap

**"Offline RAGAS scores went up, but online deflection dropped."** Enumerate the causes — this is the question that separates levels:

1. **Eval-set staleness.** Your golden set was built on last quarter's corpus; the live corpus drifted, so offline gains don't reflect real queries.
2. **Synthetic-question bias.** If your eval questions were LLM-generated *from chunks*, they are phrased like the chunks — artificially easy for retrieval. Real users phrase things differently. Offline recall is inflated.
3. **Judge–user mismatch.** Your LLM-judge of "answer relevance" rewards thorough, hedged answers; real users abandon long answers and want one line. You optimized the judge, not the user.
4. **Latency regression.** The better reranker added 400ms; deflection drops because users bounce before the answer streams. The quality metric never sees this.
5. **Distribution shift in intent.** Offline set is over-weighted toward answerable questions; live traffic has more "not in the KB" queries where the right behavior is abstain, which your offline metric penalizes.

### One fully specified A/B test

- **Hypothesis:** adding the cross-encoder reranker increases ticket deflection without raising hallucination or breaching latency.
- **Unit of randomization:** user (sticky), so a given user gets a consistent experience and we don't split a single conversation.
- **Arms:** control = hybrid retrieval, no rerank; treatment = hybrid + cross-encoder rerank.
- **Primary metric:** ticket deflection rate (resolved without human escalation within 24h).
- **Guardrails (any breach auto-stops the ramp):** faithfulness rate (live, no-label), p95 latency < 3s, cost/query < $0.05, escalation-to-human rate.
- **Ramp:** 1% → 5% → 25% → 50%, with a guardrail check at each step.
- **Minimum runtime:** size for the deflection delta you care about; run at least one full week to cover weekday/weekend mix and avoid novelty effects. Use **CUPED** with pre-period deflection as covariate to cut variance and reach significance faster.
- **Rollback trigger:** any guardrail breach, or deflection not trending positive by 50% ramp.

### Error analysis ritual

Keep a standing bank of bad cases. After every change, manually read 20-30 failures and tag each as retrieval-miss, rerank-miss, hallucination, or relevance-drift. The *distribution* of failure types tells you which box to fix next — far more actionable than a single aggregate score.

---

## 8. Deployment — three paths

Name them separately; this is the fastest way to sound like an operator.

- **Serving path (synchronous, the request):** API gateway → query rewrite → retrieval (BM25 + ANN in parallel) → fusion + ACL filter → rerank → generate (streamed) → grounding check. Each is a separable service so I can scale, cache, and fail-over independently. Cache embeddings for repeat queries and cache full answers for FAQ-shaped traffic.
- **Data path (the ingestion pipeline):** source connectors → change-data-capture → chunk → embed → upsert into the vector index + sparse index, with ACLs and timestamps. Must reconcile deletes and edits (a deleted doc's chunks must leave the index within the 5-minute freshness SLO, or you cite stale/forbidden content).
- **Feedback path (the flywheel):** log queries, retrieved+reranked chunks, the answer, citations, and user signal (thumbs, click, escalation). This feeds reranker fine-tuning, eval-set refresh, and content-gap detection. Without this path the system never improves.

### Rollout discipline

Shadow (run the new retriever in parallel, log, serve nothing) → canary (1%) → A/B → gradual ramp, with guardrails gating each step. Shadow is especially valuable for retrieval changes because you can compare retrieved-chunk sets offline before any user sees them.

### Monitoring and fallback

- **What pages someone:** faithfulness rate drop (hallucination spike), "retrieved nothing relevant" rate spike (content gap or index broken), p95 latency, ingestion lag (freshness SLO breach), cost/query spike (tiering router misrouting to the frontier model).
- **Fallback ladder:** if generation fails or grounding check rejects, fall back to returning the top reranked snippets as a plain ranked list (rung-0 behavior). If the vector index is down, serve BM25 only. If everything is degraded, route to a human with the retrieved context attached. Degrade, never go dark.
- **Incident response:** freeze the current model/index version, diff against the last-good (retriever config, reranker weights, prompt, corpus snapshot), inspect traces for the failing query class, roll back the offending layer. Because the layers are separable, you can roll back *just* the reranker without touching ingestion.

---

## 9. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Pipeline | "Chunk, embed, retrieve, send to LLM." | Splits retrieval vs generation as two failure surfaces with separate metrics. |
| Chunking | "Split into chunks." | Structure-aware, sized + overlap tuned against a retrieval eval set; embed-small/read-big. |
| Retrieval | "Use embeddings / a vector DB." | Hybrid BM25+dense because they fail on opposite queries; RRF to fuse; explains why dense misses exact entities. |
| Reranking | Skips it. | Cross-encoder retrieve-then-rerank as a latency/precision factorization, with the cost in the budget. |
| Grounding | "The LLM uses the context." | Cite-or-abstain prompt + entailment check; treats abstention as a feature. |
| Eval | "Measure accuracy." | RAGAS quadrant localizing failures; two metrics run label-free in prod. |
| Offline/online | "Offline metrics will predict online." | Enumerates why RAGAS-up/deflection-down happens; specifies an A/B with guardrails and CUPED. |
| Deploy | "Deploy the endpoint." | Three paths; shadow-first for retrieval; degrade-don't-go-dark fallback ladder. |

---

## 10. One-page whiteboard cheat sheet

```
NUMBERS:  2M docs -> 20M chunks | 1024-dim fp16 (~40GB+HNSW)
          100 QPS | p95<3s STREAMED (TTFT<1s) | <$0.05/query | 5-min freshness

BUDGET:   rewrite 150 | embed 15 | retrieve 50 | rerank 250 | TTFT 500 = ~965ms

LADDER:   BM25 -> +dense (misses entities) -> HYBRID+RERANK (default) -> Self/Corrective-RAG
                                              ^ earn this, don't start here

PIPELINE: query -> rewrite -> [BM25 || ANN] -> RRF -> ACL filter
          -> cross-encoder rerank(150->8) -> assemble(small->big)
          -> generate(cite-or-abstain) -> grounding check -> cited answer

TWO SURFACES:
  RETRIEVAL  | context recall (was it there?) | context precision (ranked top?)
  GENERATION | faithfulness (supported?)       | answer relevance (on-topic?)
             ^ faithfulness + relevance need NO labels -> run live

OFFLINE-UP/ONLINE-DOWN: stale eval set | synthetic-Q bias | judge!=user
                        | latency regression | abstain-penalized intent shift

A/B: unit=user | primary=deflection | guard=faithfulness,p95,cost,escalation
     ramp 1->5->25->50 | CUPED | rollback on guardrail breach

DEPLOY: serving / data(ingestion+CDC) / feedback paths
        shadow->canary->A/B | fallback: snippets -> BM25-only -> human
```

---

## 11. Full interview transcript (one hour)

**[00:00] INTERVIEWER:** Design a production RAG system for enterprise customer support.

**[00:30] YOU:** Before I draw anything, a few clarifying questions, because the answers change the architecture. First: is this customer-facing deflection, or internal agent assist? Second: do we generate a cited answer, or return a ranked list of docs? Third: how fresh must answers be? And are there per-document permissions?

**[01:10] INTERVIEWER:** Customer-facing deflection. Generated, cited answers. Freshness within a few minutes. Assume the KB is all public to customers, so no per-user ACLs for now.

**[01:30] YOU:** Good, that simplifies retrieval — I'll note that if ACLs existed, I'd filter inside the retrieval call, but I'll set that aside. Let me commit to numbers so you can challenge them: 2M source docs, about 20M chunks, 100 QPS peak, 5M users. I'll target p95 under 3 seconds, but since we stream, the real constraint is time-to-first-token under 1 second. Cost under five cents a query. Freshness about 5 minutes. Sound right?

**[02:15] INTERVIEWER:** Those are fine. Where does your time go in that 3 seconds?

**[02:25] YOU:** Let me split it. Query rewrite on a small model, ~150ms. Embedding the query, ~15. BM25 and the ANN search run in parallel, so ~50 for the slower one. Cross-encoder rerank of ~150 candidates down to 8, ~250. That's about 465ms before generation. Add ~500ms to first token, so TTFT lands under a second. The rest of the answer streams over the remaining two seconds. The reranker depth is my first knob if I'm over budget.

**[03:10] INTERVIEWER:** You jumped to a reranker. Why not just embed everything and retrieve top-k?

**[03:20] YOU:** Because pure dense retrieval breaks on a specific, common query type: exact entities. If a customer pastes error code `0x80070005` or asks about "version 14.2," the embedding blurs those exact tokens — they get averaged into the surrounding semantics and the right chunk doesn't surface. BM25 nails exact matches. Dense nails paraphrase, like "can't sign in" matching "authentication failure." They fail on opposite queries, so I run both and fuse with Reciprocal Rank Fusion. Then the reranker fixes a different problem — want me to take that next?

**[04:05] INTERVIEWER:** Go ahead.

**[04:10] YOU:** The bi-encoder is fast because it embeds the query and the chunk separately, so a nearest-neighbor search over 20M vectors is ~50ms. But "separately" means the model never sees the query and chunk together, so it can't judge fine-grained relevance. A cross-encoder does — it takes the query and chunk as one input and runs full attention across both, producing a much more accurate score. It's too expensive to run on 20M chunks, so I use the classic funnel: cheap bi-encoder narrows 20M to ~150, expensive cross-encoder reranks 150 to 8. Same latency-versus-precision factorization as a recommendation funnel.

**[05:20] INTERVIEWER:** Let's back up. How do you chunk 2M documents? This feels like a detail but I want to hear it.

**[05:30] YOU:** It's not a detail — chunking sets the retrieval ceiling before any model runs. Three failure modes. Too small, say 128 tokens, and a chunk isn't self-contained: "it expires after 30 days" is useless without knowing what "it" is. Too large, say 2000 tokens, and the embedding averages over many topics, matching everything weakly. And naive fixed-size splitting cuts mid-sentence or mid-table. So I split on document structure first — headings, sections, table rows — then pack to about 300-500 tokens with ~50 tokens of overlap so an answer straddling a boundary isn't lost. I attach metadata to every chunk: source, section path, last-updated, authority. And I embed the small chunk for matching but expand to the parent section at generation time, so what retrieves well and what reads well are decoupled.

**[07:00] INTERVIEWER:** Where do your labels come from? How do you even know retrieval is good?

**[07:15] YOU:** Three sources, increasing cost. Implicit signals: which cited chunk did the user click, did the ticket resolve without a follow-up. Cheap and abundant but biased — "no follow-up" might mean satisfied or might mean they gave up. Second, synthetic: I prompt a strong LLM to generate questions from a chunk, and that chunk becomes the gold context for the question. That bootstraps a retrieval eval set on day one with no human labeling — but I'll flag a bias there in a second. Third, a few hundred human-labeled real queries with expert answers and source chunks: expensive, but that's my regression set before every launch.

**[08:20] INTERVIEWER:** You said you'd flag a bias in the synthetic set.

**[08:30] YOU:** Right. Questions generated *from* a chunk tend to be phrased like the chunk — they share vocabulary, so retrieval finds them too easily. Offline recall looks inflated relative to real users who phrase things their own way. So I validate a sample by hand and I never trust synthetic recall as an absolute number, only as a relative signal between model versions.

**[09:10] INTERVIEWER:** Suppose your offline retrieval metrics improve, you ship, and deflection *drops*. What happened?

**[09:25] YOU:** Classic trap. Several possibilities, and I'd check them in order. One: the synthetic-question bias I just mentioned — offline gains were on artificially easy questions. Two: eval-set staleness — the corpus drifted since I built the set. Three: judge-versus-user mismatch — if my "answer relevance" judge rewards thorough hedged answers but real users want one line and bounce, I optimized the judge, not the user. Four: a latency regression — the better reranker added 400ms and users leave before the answer streams; my quality metric never sees that. Five: intent shift — live traffic has more questions that aren't in the KB at all, where the right behavior is to abstain, and my offline metric might be penalizing abstention.

**[10:50] INTERVIEWER:** That's a good list. How do you measure quality when at serving time you have no ground-truth answer?

**[11:00] YOU:** I decompose quality into a retrieval axis and a generation axis. On retrieval: context recall — were the needed facts in the retrieved context — and context precision — were the relevant chunks ranked at the top versus buried. On generation: faithfulness — is every claim supported by the retrieved context — and answer relevance — does it actually address the question. The key point for your question: faithfulness and answer relevance need *no* labeled answer. Faithfulness is an entailment check between each answer sentence and its cited span. So I run those two continuously on live traffic. Recall and precision need my golden set, so those run offline pre-launch.

**[12:20] INTERVIEWER:** And the value of splitting it that way?

**[12:30] YOU:** Localization. If faithfulness is low, the generator is fabricating — I fix the prompt or add a stricter grounding gate. If context recall is low, the retriever never found the answer — I fix chunking or retrieval, and the generator is blameless. If I only had one "accuracy" number I couldn't tell those apart, and they have completely different fixes. That's also how I run error analysis: I read 20-30 failures after each change and tag them retrieval-miss, rerank-miss, hallucination, or relevance-drift. The distribution tells me which box to fix next.

**[13:40] INTERVIEWER:** Design the A/B test for adding the reranker.

**[13:50] YOU:** Randomize by user, sticky, so a conversation stays consistent. Control is hybrid retrieval with no rerank; treatment adds the cross-encoder. Primary metric is ticket deflection — resolved without human escalation in 24 hours. Guardrails that auto-stop the ramp: live faithfulness rate, p95 latency under 3s, cost under five cents, and escalation rate. Ramp 1, 5, 25, 50 percent with a guardrail check at each step. Run at least a full week for weekday-weekend mix and to avoid novelty effects, and I'd use CUPED with each user's pre-period deflection as a covariate to cut variance and reach significance faster. Rollback on any guardrail breach or if deflection isn't trending positive by the 50% step.

**[15:20] INTERVIEWER:** Freshness — docs change. How do you keep the index current within 5 minutes?

**[15:30] YOU:** A separate ingestion path with change-data-capture on the sources. On an edit: re-chunk, re-embed, upsert into both the vector and sparse indexes, with updated timestamps. The subtle part is *deletes and edits* — if a doc is deleted, its chunks must leave the index within the freshness window, or I'll cite content that no longer exists. So ingestion reconciles deletes, not just adds. I'd also invalidate any cached answers that cited a changed chunk.

**[16:40] INTERVIEWER:** What's your fallback when generation fails or hallucinates?

**[16:50] YOU:** A degradation ladder, never go dark. If the grounding check rejects the answer or generation errors, I fall back to returning the top reranked snippets as a plain ranked list — that's the rung-0 product and it's still useful. If the vector index is down, I serve BM25-only. If everything's degraded, I route to a human with the retrieved context attached so the agent has a head start. And the system is built so I can roll back just the reranker, or just the prompt, without touching ingestion, because the layers are separate services.

**[18:00] INTERVIEWER:** When would you reach for something fancier — Self-RAG, graph RAG?

**[18:10] YOU:** Only when I've *measured* a population of queries that rung 2 fails on, and that population is multi-hop — questions needing reasoning across multiple documents, like "does policy X apply given exception Y." Corrective or Self-RAG add a retrieve-critique-maybe-retrieve-again loop that helps there. But they're much harder to evaluate and operate — more moving parts, harder traces to debug. So I'd only climb that rung if the multi-hop failure rate justified it *and* my eval maturity could validate the gain. Leading with graph RAG unprompted would be a red flag that I'm chasing fashion over the constraint.

**[19:30] INTERVIEWER:** Last one. What dominates your cost at 100 QPS, and what do you do about it?

**[19:40] YOU:** Generation dominates — the LLM tokens, not the retrieval. The main lever is model tiering: route short, easy, high-confidence queries to a cheap small model, and escalate only ambiguous or high-stakes queries to a frontier model. I'd also cache full answers for FAQ-shaped traffic, which is a big fraction in support, and cache query embeddings for repeats. Those three together are how I hold under five cents a query. And I'd monitor cost-per-query as a guardrail, because a misrouting bug in the tiering router would silently send everything to the expensive model.

**[21:00] YOU:** To connect this to what I've actually built — on the Seller Copilot work I did, the thing that bit us in production was exactly this retrieval-versus-generation split. We kept seeing "wrong answers" and treating them as a model problem, when half of them were retrieval misses — the right policy doc was never in context. Once we instrumented context recall separately from faithfulness, we could finally tell the two apart and fix the right layer. That experience is why I lead with that decomposition instead of a single accuracy number.

**[21:40] INTERVIEWER:** That's exactly what I wanted to hear. Good.

### Why this transcript works

- **Clarifies before drawing**, and ties each question to a design consequence.
- **Commits to numbers early** and derives the latency budget out loud, so every later claim is grounded.
- **Earns each ladder rung** — never names the fancy thing first, and explicitly calls leading with graph RAG a red flag.
- **Keeps retrieval and generation separate at every step** — data, model, metric, monitoring, incident response — which is the entire senior signal of this case.
- **Handles the offline/online trap** with an enumerated, ordered checklist rather than a shrug.
- **Specifies a real A/B test** with unit, guardrails, ramp, variance reduction, and rollback.
- **Closes on real production experience**, turning the abstract decomposition into a scar earned on a shipped system.

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x the corpus (200M chunks)?** Move from fp16 flat vectors to product-quantized (int8/PQ) vectors to stay RAM-resident, accept a small recall hit, and recover it with the reranker. Shard the index; consider a two-level ANN. Revisit ingestion throughput.
- **How do you handle multi-hop questions?** Detect them (a question needing facts from multiple docs), decompose into sub-queries, retrieve per sub-query, then synthesize — or adopt Corrective/Self-RAG. Only after measuring that they're a real failure population.
- **How do you pick top-K and rerank depth?** Tune against the retrieval eval set: raise candidate count until context recall plateaus, then set rerank output (top-8) by how much context the generator needs without diluting the prompt. Confirm online.
- **Offline metrics up, online down — what do you check?** Synthetic-question bias, stale eval set, judge-vs-user mismatch, latency regression, abstain-penalizing intent shift. (See section 7.)
- **How do you debug a bad launch?** Diff the layers (retriever config, reranker, prompt, corpus snapshot) against last-good; pull traces for the failing query class; check whether it's a retrieval-miss or a hallucination via the RAGAS split; roll back just the offending layer.
- **How do you prevent the feedback loop from degrading?** Click signal on citations has position bias; downweight it, reserve some exploration in ranking, and refresh the eval set from real (not synthetic) traffic so you don't optimize toward your own outputs.
- **RAG or fine-tune?** RAG when knowledge changes, citations are required, or permissions matter. Fine-tune when you need behavior/format/style/policy internalized. For enterprise knowledge, RAG is the base and fine-tuning is optional (often used to improve the *reranker* or to teach the *format*, not to memorize facts).

---

## 13. Common mistakes

- Drawing the pipeline but never separating **retrieval failure** from **generation failure** — then saying "accuracy" to mean four different things.
- Treating **chunking** as a throwaway step, when it sets the retrieval ceiling.
- Going **dense-only** and getting wrecked by exact-entity queries (error codes, versions, names).
- **Skipping the reranker**, or adding it without putting its latency cost in the budget.
- Hoping the LLM "just uses the context" instead of designing **cite-or-abstain + a grounding check**.
- Listing offline metrics with **no A/B test**, no guardrails, and no explanation of the offline/online gap.
- Forgetting **ingestion deletes/edits** and freshness, so the system cites stale or removed content.
- Reaching for **graph/Self-RAG first** instead of earning it from a measured failure population.
- Treating deployment as "an endpoint" rather than three paths with a **degrade-don't-go-dark** fallback ladder.

---

## 14. Transfer — what this case unlocks

- **04 Enterprise AI Copilot:** shares the retrieval substrate but adds the agent control plane and deep permission-aware retrieval. This case owns retrieval *quality*; that case owns *actions and ACLs*.
- **06 LLM Evaluation & Monitoring Platform:** the RAGAS quadrant and label-free live metrics generalize into a full eval platform; faithfulness/relevance judging is the seed.
- **07 AI Agent for Ticket Resolution:** RAG is the "look it up" tool inside the agent; the grounding and abstention discipline carries directly.
- **13 LLM Safety Gateway:** the grounding/entailment check is a sibling of safety filtering; both gate generated output before it reaches the user.
- **14 Document Intelligence Pipeline:** the ingestion/chunking/metadata path is the same data plane, pushed toward extraction.
- **03 Search Ranking:** the retrieve-then-rerank funnel, hybrid fusion, and click-bias debiasing are the same machinery without the generation head.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research, RAG sufficient context: https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/
- OpenAI text generation guide: https://platform.openai.com/docs/guides/text-generation

Added (verified canonical):
- RAG (Lewis et al., 2020): https://arxiv.org/abs/2005.11401
- Dense Passage Retrieval (Karpukhin et al., 2020): https://arxiv.org/abs/2004.04906
- Reciprocal Rank Fusion (Cormack et al., 2009): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- RAGAS (Es et al., 2023): https://arxiv.org/abs/2309.15217
- Self-RAG (Asai et al., 2023): https://arxiv.org/abs/2310.11511
- Corrective RAG (Yan et al., 2024): https://arxiv.org/abs/2401.15884
