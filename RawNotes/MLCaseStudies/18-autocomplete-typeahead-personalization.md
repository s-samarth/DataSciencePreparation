# 18. Autocomplete / Typeahead with Personalization

**Company tags:** Google, Amazon, LinkedIn, Airbnb, Booking, any large search box
**Interview frequency:** Medium
**Why it matters:** Typeahead is the one ranking problem where the latency budget is measured per keystroke, the system *authors* the text the user ends up sending, and the product actively reshapes its own training data. Get those three things and you sound like someone who has shipped a search box; miss them and you sound like you re-skinned a document ranker.

---

## 0. How to use this doc

Two passes.

**Pass 1 (intuition).** Read sections 1-6 once without trying to memorize. The single idea that makes everything else fall out: *you are not ranking documents for a query, you are completing a query before it exists.* The unit you retrieve is a query string, the candidate set is "what could this prefix become," and you must produce it on every keystroke inside a budget so small that a normal model server is already too slow. Hold that and the data structures, the caching, and the feedback-loop danger all follow.

**Pass 2 (active recall).** Cover the page. On a whiteboard, derive the per-keystroke latency budget out loud, draw the trie-plus-reranker split, and explain why popularity-by-frequency is a feedback loop that eats itself. Then read the section 9 transcript and run it as a simulation: pause at every INTERVIEWER line and answer before reading YOU. If you cannot reconstruct the latency math and the exposure-bias argument from memory, you have not learned it yet.

The reusable scaffold (same one across every case in this set):

> **Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor**

This file leans hard on Clarify (the latency number is the whole design), Model (prefix retrieval is a distinct beast), and Monitor (the feedback loop is unusually vicious here).

---

## 1. Clarify: the questions that change the design

Do not start drawing. Spend the first three minutes scripting these, because the answers swing the architecture more than any model choice.

| Question | Why it changes the design |
|---|---|
| "Do we re-rank on every keystroke, or only after a debounce pause?" | This sets the entire latency budget. Per-keystroke means ~10-20 ms of server time after network; debounced means you can afford a real model. The whole architecture pivots here. |
| "Are we completing *search queries* or *entities* (products, people, places)?" | Query completion expands a prefix into full query strings from a log. Entity typeahead retrieves structured records with their own attributes, availability, and ACLs. Different candidate sources, different ranking signals. |
| "How personalized do we get, and what's the creepiness ceiling?" | Showing a stranger's name or someone's past private query as a suggestion is a trust incident. Personalization here authors text in the user's mouth, so the bar is higher than in passive ranking. |
| "What's the safety/liability surface? Can a suggestion defame, harass, or surface illegal content?" | Autocomplete suggestions are *attributable to the platform*. A toxic organic result is the web's fault; a toxic suggestion is your product putting words in someone's mouth. This forces a hard policy filter, not a soft demotion. |
| "Mobile-heavy or desktop? What's the p99 network RTT to our edge?" | On mobile, network RTT can be 100-150 ms and eat the whole human-perceptible budget. That pushes work to the edge/CDN and on-device prefix caching. |
| "What does 'good' mean: acceptance rate, keystrokes saved, or downstream search success?" | Optimizing raw acceptance rewards showing the obvious popular completion. Optimizing downstream success (the user found what they wanted) is the real goal and changes the label. |
| "How fresh must suggestions be? Do we need to surface a term that started trending an hour ago?" | Freshness sets how often the candidate index rebuilds and whether you need a streaming fast-path for breakout terms (news, product drops, outages). |
| "Languages, scripts, transliteration, typo tolerance?" | A Latin-only ASCII trie is a different problem from CJK input-method completion or Arabic/Hindi transliteration. Pin this early so you do not design the wrong index. |

The two answers that dominate everything else: **per-keystroke vs debounced** (the budget) and **query vs entity** (the candidate source). Get those on the board first.

---

## 2. Numbers up front (carry these through the whole answer)

State these out loud and keep referring back. Made-up but realistic marketplace scale.

- **Traffic:** 50M searches/day. Average query ~4 keystrokes shown suggestions, and we re-rank on roughly every keystroke after the 2nd character -> call it **~3 typeahead requests per search**, so **~150M typeahead requests/day ≈ 1,700 req/s average, ~5,000 req/s peak.**
- **Latency target (the headline number):** total user-perceptible budget for a suggestion to feel instant is **~100 ms** from keystroke to painted dropdown. Subtract network: mobile RTT p99 ~120 ms blows the budget on its own, so we serve from edge POPs; assume ~30 ms RTT to a regional edge. Subtract client render ~10 ms. That leaves a **server budget of ~50 ms p99, and we target ~20 ms p99 to leave headroom.** Within that 20 ms: prefix lookup ~2-5 ms, candidate fetch ~5 ms, rerank ~5-10 ms, post-filter/format ~2 ms. **There is no room for a 50 ms model call.** That single fact dictates the whole serving design.
- **Catalog of completions:** ~500M distinct historical queries, long-tail. After pruning to queries seen >= N times in the trailing window, keep ~50M completion candidates in the served index.
- **Per-prefix fan-out:** a 2-3 character prefix can match tens of thousands of completions; we must retrieve a small top-set (a few hundred) cheaply, then rerank ~50-200 of them.
- **Storage back-of-envelope:** 50M completions x (avg 30 bytes string + ~40 bytes of packed features/scores) ≈ **~3.5 GB** for the core completion index. A trie/FST representation of the strings compresses shared prefixes well, often **under 1 GB** for the string structure; feature payloads dominate. This fits in RAM on every serving node, which is the point: **you replicate the whole index per node and never do a network hop for candidates.**
- **Personalization store:** per-user recent queries + clicked entities, ~a few KB/user, 100M MAU -> a few hundred GB in a fast KV store, fetched in the same call as the request (p99 < 5 ms).
- **Freshness:** core index rebuild every few hours; a streaming side-channel injects breakout terms within minutes.

If you derive the 20 ms server budget out loud from the 100 ms human budget minus network minus render, you have already won most of the interview. Most candidates never subtract network and silently assume they have 200 ms.

---

## 3. The conceptual spine: completing a query, not ranking a document

Internalize four ideas. Everything in the architecture is one of these wearing a costume.

**(1) The retrieved unit is a query string.** In document search (file 03) the query is fixed and you rank documents. In typeahead the "query" is a fragment and the *candidates are themselves queries* drawn from a log of things people have searched. Candidate generation = "expand this prefix into plausible full queries." That is a prefix-match over a precomputed dictionary, not a BM25/ANN search over a corpus.

**(2) The budget is per keystroke, so retrieval must be a data-structure problem, not a model problem.** You cannot afford a learned candidate generator on the hot path. You precompute. The classic structure is a **trie** (prefix tree) or, better, an **FST / finite-state transducer** (as used by Lucene's completion suggester) that stores prefix -> top completions compactly and returns the top-k for a prefix in microseconds. The ML lives *after* retrieval, on a tiny candidate set, and even then it must be cheap (a small GBDT or a distilled linear/factorization model, or fully precomputed scores).

**(3) Popularity-by-frequency is a feedback loop that eats itself.** Whatever you suggest gets clicked more, which makes it more frequent, which makes you suggest it more. Organic ranking has this too, but typeahead is worse because the suggestion *creates* the query text. Left alone, the head ossifies and the tail and anything new starve. So the data section is dominated by **exposure debiasing and a deliberate exploration channel**, not by feature engineering.

**(4) A suggestion is an authored statement, so safety is a hard gate, not a soft demotion.** When you complete "is [person] a ..." you are publishing a sentence. The mitigation is a **policy blocklist + classifier that removes candidates entirely** (defamation, harassment, illegal, adult, hate, named private individuals), applied as a filter before ranking ever runs. You cannot "rank it lower"; a single bad suggestion screenshot is the incident.

Hold these four and the rest of the doc is mechanics.

---

## 4. Data and labels: the bias problem is the whole game

**Where candidates come from.** The completion dictionary is built offline from the query log: aggregate raw queries, normalize (lowercase, trim, collapse whitespace, optional spell-correct/canonicalize), count frequency in a trailing window with time decay, drop queries below a frequency floor (privacy + quality: a query only one person ever typed must not become a public suggestion — that is both noise and a privacy leak), and drop anything the safety filter rejects. Then build the prefix index keyed by every prefix of every surviving completion, storing the top completions per prefix.

**The labels.** A typeahead "impression" is a (prefix, shown-list, position, chosen-completion, downstream-outcome) tuple. The naive positive is "user clicked the suggestion." But:

- **Position/exposure bias:** users click the top suggestion because it is on top, not because it is best. You must model or randomize position, or you train a model that just learns "predict what is already on top."
- **The acceptance trap:** optimizing click-on-suggestion rewards the obvious. The completion the user would have typed anyway gets the credit. The *valuable* suggestion is the one that saved keystrokes AND led to a successful search (results clicked, purchase, no immediate reformulation). So the real label is **accepted AND the resulting search succeeded.** Tie the suggestion to the session outcome, not just the click.
- **Survivorship:** you can only ever collect labels on completions you chose to show. Anything never shown has no data. This is the structural argument for an **exploration slot** (below).
- **Delayed and noisy outcomes:** "successful search" resolves seconds to minutes later; bot/scraper traffic pollutes frequency counts and must be filtered before they inflate the dictionary.

**Cold start (three flavors).**
- *New query/term cold start* (the important one): a term that started trending an hour ago is not in the trailing-window dictionary yet. Solve with a **streaming fast-path** that detects breakout prefixes (sudden velocity) and injects them with a freshness boost, decayed over time.
- *New user:* no personal history -> fall back to popularity + geo/locale priors, blend in personalization as signal accrues within the session.
- *New surface/locale:* seed from a related locale or global popularity, open an exploration budget to learn fast.

**The exploration channel.** Reserve a small fraction of suggestion slots (or a small traffic slice) to show candidates the popularity model would not have surfaced — fresh terms, tail completions, personalized long shots — and log their outcomes. This is the only way to get unbiased labels on unshown candidates and the only structural defense against the feedback loop. Treat it as a first-class part of the data pipeline, not a nice-to-have.

---

## 5. Baseline -> why it breaks -> the next rung

Climb this ladder out loud. Each rung is justified by a concrete trigger from the previous rung breaking.

**Rung 0 — Static trie by global frequency.** Build a trie over the top queries, each prefix maps to its highest-frequency completions, return top-k. Microsecond lookups, trivially cacheable, fits in RAM.
*Why it breaks:* no personalization, no freshness (trending terms missing for hours), no availability/inventory awareness, and it hard-codes the feedback loop (head ossifies). Also no safety filtering yet.

**Rung 1 — Trie + freshness/recency blending + safety filter.** Add a time-decayed frequency score, a streaming injector for breakout terms, and a hard policy filter that strips unsafe completions before serving.
*Why it breaks:* the ranking is still hand-weighted (how much do freshness vs frequency vs length matter?). Hand weights do not generalize across prefixes, surfaces, or locales, and they cannot use user context.

**Rung 2 — Prefix retrieval + lightweight ML reranker (the production default).** Trie/FST retrieves a few hundred candidates in microseconds; a cheap model (GBDT or distilled linear/FM) reranks the top ~50-200 using prefix-match features, popularity/freshness, session context, and light personalization. This is the rung I would build and defend.
*Why it breaks (if pushed):* a flat reranker ignores *sequence* — the last two queries this session strongly predict the next, and a per-candidate scorer cannot model "given what you just searched, this completion." Also at very high QPS the model call competes with the latency budget.

**Rung 3 — Session-aware / sequence model, mostly precomputed.** A sequence model (e.g., a small transformer or RNN over session history) captures intent drift. But you almost never run it synchronously on every keystroke. Instead **precompute**: run it to produce per-user or per-session completion scores asynchronously and cache them, or distill it into the rung-2 reranker's features. The synchronous path stays a fast lookup + cheap rerank.
*When to actually go here:* only when rung 2's session blindness is a measured loss in A/B (e.g., poor multi-step query journeys) and you have the infra to precompute/distill without blowing the budget.

State explicitly: **rung 2 is the answer.** Rung 3 is an extension you reach for with evidence, and even then you push its cost off the hot path.

---

## 6. One architecture, explained to the floor

Three paths, drawn separately (serving, data, feedback). This is the "I have operated this" tell.

### 6a. Serving path (the hot, latency-critical path)

```
Keystroke -> Edge POP
  -> normalize prefix (lowercase, trim, transliterate)
  -> PREFIX RETRIEVAL: FST/trie lookup -> top ~200 candidate completions   [~2-5 ms]
  -> fetch context in parallel: user recent queries, session, geo, inventory flags  [~5 ms]
  -> SAFETY FILTER: drop policy-violating candidates (hard gate)            [~1 ms]
  -> RERANK: cheap GBDT/linear over ~50-200 candidates                      [~5-10 ms]
  -> dedupe / diversity / format -> top 5-10                                [~2 ms]
  -> return  (server p99 ~20 ms)
```

Key serving decisions, and why:
- **The whole completion index is replicated in RAM on every serving node.** No network hop for candidates — a network hop alone would blow the budget. This is why we prune to ~50M completions: it has to fit.
- **Retrieval is an FST/completion-suggester, not BM25/ANN.** Prefix top-k is a path traversal, returning weighted tops in microseconds. (Lucene's completion suggester is exactly this; it is a different index from the normal inverted index.)
- **The model is small and the candidate set is tiny.** A GBDT scoring 200 candidates with ~30 features is single-digit milliseconds. A neural reranker only appears here if distilled to that cost or if its scores were precomputed.
- **Caching is layered:** edge cache keyed by (prefix, coarse-context bucket) absorbs the popular head; personalization is blended on top for logged-in users. Short-prefix requests (1-2 chars) are almost fully cacheable.
- **Features:** prefix-match quality (exact/prefix/fuzzy, how much is typed vs completed = keystrokes saved), completion popularity + freshness/velocity, session signals (last queries, dwell), light personalization (has the user searched this before / clicked this category), inventory/availability for marketplace (do not suggest a query that returns zero results), locale/device.

### 6b. Data path (builds the index offline)

```
Raw query logs -> bot/scraper filter -> normalize/canonicalize
  -> aggregate frequency w/ time decay (trailing window)
  -> frequency-floor prune (privacy + quality: drop ultra-rare)
  -> safety classifier + blocklist prune
  -> compute per-completion features/scores
  -> BUILD FST/trie + feature payloads -> ship index to all serving nodes (every few hours)
```

### 6c. Feedback path (closes the loop, and must be debiased)

```
Served impressions (prefix, shown list, positions, choice, session outcome)
  -> join to downstream success (results clicked? reformulated? purchased?)
  -> position/exposure-bias correction (IPS weighting or logged exploration data)
  -> training set for reranker
  -> EXPLORATION slot logs feed unbiased tail/fresh-candidate labels
  -> retrain reranker -> offline eval -> canary -> ramp
```

The feedback path is where you earn senior points: say out loud that **without exposure-bias correction and an exploration channel, this loop trains a model to reproduce yesterday's suggestions**, and the head ossifies while new and tail intent starves.

**Losses / objective.** The reranker is trained as a learning-to-rank model (pairwise/listwise, e.g., LambdaMART-style on a GBDT, or a logistic per-candidate scorer) where the target is **accepted-and-successful**, with position-bias correction via inverse-propensity weighting on logged positions (or clean labels from the exploration slot). The objective is downstream search success per keystroke spent, not raw clicks.

---

## 7. Evaluation: offline metrics, the offline-up/online-down trap, and a real A/B

**Offline metrics.**
- **MRR / NDCG@k** of the accepted completion among shown suggestions.
- **Success-weighted acceptance:** fraction of impressions where the chosen completion led to a successful search (the label that matters), not raw click.
- **Keystrokes saved:** how many characters the completion saved vs typing the full query — the literal product value.
- **Prefix coverage:** fraction of real prefixes for which we return a useful suggestion at all (tail and fresh coverage especially).
- **Calibration** of the acceptance/success score, sliced by prefix length, head/tail, locale, logged-in/out.
- Always slice by prefix length (1-2 char behaves nothing like 5+), head vs tail, new vs returning user, and locale.

**The offline-up / online-down trap (enumerate the causes — interviewers love this).**
1. **Exposure/position bias in the offline set:** the model learned to predict what was already shown on top; offline MRR looks great because it is graded against biased logs, but it adds no real value online.
2. **You optimized clicks, not success:** offline acceptance rose, but the extra clicks were on obvious completions the user would have typed anyway — zero downstream lift, maybe even more reformulations.
3. **Latency regression:** the better model is slower; the extra 15 ms pushed past the budget, suggestions arrived after the user finished typing, and acceptance fell despite "better" ranking.
4. **Freshness blind spot:** offline eval ran on a static snapshot and never tested breakout terms; online, trending queries are missing and users notice.
5. **Novelty/feedback effects:** offline metrics computed on old logs do not capture that surfacing new completions changes user behavior (good) or that personalization felt creepy (bad) — both only show online.
6. **Train-serve skew:** a feature (e.g., session-recency) computed differently offline vs online silently degrades the live model.

**A fully-specified A/B test.**
- **Hypothesis:** the rung-2 personalized reranker increases success-weighted acceptance and keystrokes saved without regressing latency or safety vs the rung-1 freshness-blended trie.
- **Unit:** user (sticky assignment so personalization is coherent within a user), 1% -> 5% -> 20% -> 50% ramp.
- **Primary metric:** success-weighted suggestion acceptance (accepted AND search succeeded), per search session.
- **Secondary:** keystrokes saved, downstream search success rate, query reformulation rate (lower is better), zero-result query rate.
- **Guardrails (any breach auto-pauses ramp):** server p99 latency <= 20 ms, unsafe-suggestion rate = 0 (hard), personalization-complaint rate flat, search abandonment flat-or-down, head-diversity not collapsing.
- **Minimum runtime:** at least one full week to cover weekday/weekend and let delayed success outcomes resolve; size for the MDE on acceptance given ~150M daily requests (powered quickly, but hold for the weekly cycle and novelty decay).
- **Rollback trigger:** any guardrail breach, or primary metric negative with significance, flips the index/model pointer back to control (seconds).
- **CUPED** on pre-period per-user acceptance to cut variance and read the result faster.

---

## 8. Deployment, monitoring, incident response

**Three deployment surfaces, each with its own discipline.**
- *Serving (model + index):* the served artifact is the FST/trie index + reranker model + feature config, versioned together and shipped as an immutable bundle. A new index is a **pointer flip**; rollback is flipping back (sub-second). Never hot-mutate the live index in place.
- *Data (the offline build):* validate every rebuilt index before promotion — size/cardinality sanity, safety-filter coverage (no known-bad term leaked), coverage diff vs previous build, no sudden head collapse. A bad index build is the most likely silent outage.
- *Feedback (training):* scheduled retrains for the stable head; triggered retrains when drift or success-rate drop crosses thresholds.

**Rollout discipline.** Shadow (mirror traffic to the new index/model, compare suggestions and latency, serve nothing) -> canary (1% sticky) -> ramp with the guardrails above -> auto-rollback on breach.

**Monitoring (what actually pages someone).**
- Latency p50/p99 per stage (retrieval, context fetch, rerank) — a regression here is a user-visible outage even if "accuracy" is fine.
- Unsafe-suggestion rate and policy-filter coverage (hard guardrail; alert on any leak).
- Suggestion acceptance + downstream success, sliced by head/tail and prefix length.
- Freshness lag: time from a term trending to it appearing as a suggestion.
- Coverage/diversity: is the head ossifying? is tail coverage dropping? (feedback-loop early warning).
- Zero-result suggestion rate (suggesting queries that return nothing is a marketplace failure).
- Index build health: cardinality, size, build success, safety-filter pass.

**Fallback ladder.** If the reranker is unhealthy or slow, **degrade to the trie's static top-k** (still useful, still fast). If the index build is bad, **serve the last-known-good index**. If personalization store is down, **serve global popularity**. The product degrades gracefully to a worse-but-fine experience rather than failing the keystroke.

**Incident response.** Bad suggestion in the wild: hot-patch the blocklist (takes effect on next read, seconds), flip index/model pointer to last-known-good, then root-cause in the offline build. Latency incident: shed the reranker (fall back to trie tops), then debug. Always compare against the previous version's logged suggestions for the same prefixes.

---

## 9. One-hour interview, transcribed

*(INTERVIEWER / YOU. Read once normally; on pass 2, answer each prompt before reading YOU.)*

**INTERVIEWER:** Design autocomplete for the search box in a large marketplace.

**YOU:** Before I design, two clarifying questions decide the whole architecture. First, do we re-rank on every keystroke or only after the user pauses? That sets the latency budget. Second, are we completing free-text search *queries* or structured *entities* like products and stores? Different candidate sources.

**INTERVIEWER:** Every keystroke, and free-text query completion.

**YOU:** Then let me pin the budget, because it dominates. To feel instant, suggestions must paint within about 100 ms of the keystroke. On mobile the network round trip can eat 100+ ms by itself, so we serve from regional edge POPs — call it 30 ms RTT — and client render is ~10 ms. That leaves roughly a 50 ms server budget, and I want headroom, so I design to ~20 ms p99. Inside 20 ms I cannot afford a normal model call. So the architecture splits into a microsecond prefix-retrieval step and a tiny, cheap rerank on a small candidate set. Can I write the numbers down? 50M searches/day, ~3 typeahead calls each, ~150M calls/day, ~5k QPS peak.

**INTERVIEWER:** Go ahead. How do you retrieve candidates that fast?

**YOU:** Not with BM25 or vector search — those are for ranking documents against a complete query. Here the candidates *are queries*: I precompute a dictionary of popular historical queries from the log, and store them in a trie or, better, an FST — a finite-state transducer like Lucene's completion suggester — keyed so that any prefix returns its top completions in microseconds. The whole index, pruned to ~50M completions, is a few GB and I replicate it in RAM on every serving node. No network hop for candidates; a hop would blow the budget.

**INTERVIEWER:** Why prune to 50M? You have 500M distinct queries.

**YOU:** Two reasons. RAM: the index has to fit per node. And privacy plus quality: a query only one person ever typed should never become a public suggestion — it is noise and a potential leak of someone's private search. So I apply a frequency floor with time decay, and a safety filter, before anything enters the served index.

**INTERVIEWER:** Say more about that safety filter.

**YOU:** A suggestion is an authored statement — the platform is putting words in the user's mouth. Completing "is [person] a ..." publishes a sentence we are liable for. So safety is a *hard gate*, not a soft demotion: a blocklist plus a policy classifier strips defamation, harassment, illegal, adult, hate, and named private individuals out of the candidate set before ranking ever runs. You cannot rank a bad suggestion lower; one screenshot is the incident.

**INTERVIEWER:** Okay, you have a few hundred candidates from the FST. How do you rank them?

**YOU:** A cheap reranker — a GBDT or distilled linear model — over the top ~50-200, with features: how much the completion saves vs what is typed (keystrokes saved), popularity with freshness decay, session context like the last couple of queries, light personalization such as whether the user has searched this category, and inventory availability so I never suggest a query that returns zero results. Single-digit milliseconds for 200 candidates. I'd start here — call it the production default — rather than a session transformer, which I cannot afford synchronously.

**INTERVIEWER:** When would you use the session transformer then?

**YOU:** Only with evidence, and even then I precompute it off the hot path. If A/B shows we are losing multi-step query journeys because the flat reranker ignores sequence, I run a sequence model asynchronously to produce cached per-session completion scores, or I distill it into the reranker's features. The synchronous path stays a lookup plus cheap rerank. I never put a 50 ms model on the keystroke.

**INTERVIEWER:** What's your label? What are you training on?

**YOU:** Not raw clicks. Clicking the top suggestion mostly reflects that it was on top — position bias. And acceptance alone rewards completing the obvious query the user would have typed anyway. The label that matters is **accepted AND the resulting search succeeded** — results clicked, no immediate reformulation, ideally a purchase. I tie the suggestion to the session outcome.

**INTERVIEWER:** You mentioned position bias. How do you handle it in training?

**YOU:** Inverse-propensity weighting on the logged positions, or cleaner: a small **exploration slot**. I reserve a fraction of slots to show candidates the popularity model would not have surfaced — fresh terms, tail completions — and log their outcomes. That is the only way to get unbiased labels on completions I never normally show, and it is also my structural defense against the feedback loop.

**INTERVIEWER:** What feedback loop?

**YOU:** Whatever I suggest gets clicked, which raises its frequency, which makes me suggest it more. Organic ranking has this, but typeahead is worse because the suggestion literally creates the query text. Unchecked, the head ossifies and new or tail intent starves. The exploration channel plus freshness boosting plus monitoring head-diversity is how I keep the loop from eating itself.

**INTERVIEWER:** Suppose offline NDCG goes up but online acceptance drops. Debug it.

**YOU:** A few usual suspects. One: the offline set is exposure-biased, so I trained a model that predicts what was already on top — looks great offline, no value online. Two: I optimized clicks not success, so the extra clicks were on obvious completions with no downstream lift. Three, and I'd check this first for typeahead: latency. A better-but-slower model that adds 15 ms misses the budget; suggestions arrive after the user finished typing and acceptance falls even though ranking "improved." Four: freshness — offline ran on a static snapshot and never tested trending terms. Five: train-serve skew on a session-recency feature.

**INTERVIEWER:** Design the A/B to prove the new reranker is better.

**YOU:** Unit is the user with sticky assignment, so personalization is coherent. Ramp 1 to 5 to 20 to 50%. Primary metric: success-weighted acceptance per session. Secondary: keystrokes saved, downstream success, reformulation rate down, zero-result rate. Guardrails that auto-pause the ramp: server p99 <= 20 ms, unsafe-suggestion rate strictly zero, personalization complaints flat, abandonment flat-or-down, head-diversity not collapsing. Run a full week for weekday/weekend and to let delayed success resolve, and I'd use CUPED on pre-period acceptance to cut variance. Rollback is a pointer flip to the old index and model — sub-second.

**INTERVIEWER:** How do you deploy and roll back safely?

**YOU:** The served artifact is the FST index plus reranker plus feature config, versioned as one immutable bundle. Deploy is a pointer flip; rollback is flipping back. Before any index build is promoted I validate it — cardinality, size, safety-filter coverage, coverage diff vs previous, no head collapse — because a bad index build is the most likely silent outage. Shadow, then 1% canary, then guarded ramp. Fallback ladder if anything is unhealthy: degrade to the trie's static top-k, serve last-known-good index, drop to global popularity if the personalization store is down. The keystroke never fails; it degrades to worse-but-fine.

**INTERVIEWER:** Last one — a defamatory suggestion is trending right now. What do you do?

**YOU:** Hot-patch the blocklist; it takes effect on the next read, within seconds. If it came from a bad index build, flip to last-known-good in parallel. Then root-cause in the offline pipeline — why did the safety classifier miss it, was it a fresh-term injection that bypassed the filter — and add it to the regression set so it cannot recur. This connects to my production background: I have built guarded rollout and fast-rollback for ML services where a bad artifact is a pointer-flip away from being reverted, and where the safety filter is a hard gate sitting in front of the model rather than something the model is trusted to learn.

**Why this transcript works:**
- Opens by making the latency budget the first-class constraint and *derives* it (100 ms minus network minus render) instead of assuming it.
- Names the distinct retrieval structure (FST/trie, not BM25/ANN) and justifies in-RAM replication from the budget.
- Treats safety as a hard pre-rank gate and explains *why* (authored statement / liability), not as an afterthought.
- Picks rung 2 as the answer and defers the neural model with an explicit trigger and a precompute strategy — senior restraint.
- Gets the label right (accepted-and-successful, not clicks) and connects it to position bias and the exploration slot.
- Names the feedback loop unprompted and gives a structural defense.
- The offline-up/online-down answer leads with latency, which is the typeahead-specific cause most candidates miss.
- Closes by connecting to the candidate's real production experience with guarded rollout, fast rollback, and safety-as-gate.

---

## 10. Junior vs senior answer

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Opening | "I'll build a trie and rank with a model." | "First: per-keystroke or debounced? That sets the budget. Then I derive the ~20 ms server budget from the 100 ms human budget." |
| Retrieval | "Use search / embeddings to find matches." | FST/completion-suggester over a query dictionary, replicated in RAM; candidates are queries, not documents. |
| Latency | Assumes ~200 ms is fine. | Subtracts network and render out loud; designs to 20 ms p99; refuses a 50 ms model on the hot path. |
| Model | "A neural ranker / transformer." | Cheap GBDT on a tiny candidate set as default; sequence model only with evidence and precomputed off-path. |
| Label | "Did the user click the suggestion." | Accepted AND search succeeded; corrects for position/exposure bias. |
| Feedback loop | Not mentioned. | Names the self-reinforcing loop and defends with exploration slot + freshness + diversity monitoring. |
| Safety | "Filter bad words." | Hard pre-rank gate (blocklist + classifier) because a suggestion is an authored, attributable statement. |
| Eval | Lists MRR/NDCG. | Success-weighted acceptance + keystrokes saved; enumerates offline-up/online-down causes leading with latency; fully specs the A/B. |
| Deploy | "Serve the model behind an endpoint." | Immutable index+model bundle, pointer-flip rollback, index-build validation, graceful degrade ladder. |

---

## 11. One-page whiteboard cheat sheet

```
TYPEAHEAD = complete a QUERY before it exists (not rank documents)

CLARIFY (decides everything):
  per-keystroke vs debounced?   query vs entity completion?
  creepiness ceiling?  safety/liability surface?  mobile RTT?  freshness need?

NUMBERS:
  100ms human budget  - 30ms net (edge!) - 10ms render = ~50ms server -> design 20ms p99
  retrieval 2-5ms | context 5ms | rerank 5-10ms | filter/format 2ms
  50M searches/day -> ~150M typeahead req/day -> ~5k QPS peak
  50M pruned completions, index few GB, REPLICATED IN RAM per node

SPINE:
  1. retrieved unit = a query string (candidates ARE queries from the log)
  2. per-keystroke budget -> retrieval is a DATA STRUCTURE (FST/trie), not a model
  3. popularity-by-frequency = feedback loop that eats itself -> exploration slot
  4. a suggestion is AUTHORED -> safety is a HARD GATE, not a demotion

LADDER:
  0 static trie by freq -> no perso/fresh/safety, loop ossifies
  1 trie + freshness + SAFETY FILTER -> hand weights don't generalize
  2 prefix retrieval + cheap GBDT rerank   <-- THE ANSWER
  3 session/sequence model -> PRECOMPUTE off hot path, only with A/B evidence

SERVING:  keystroke->edge->normalize->FST top200->ctx(parallel)->SAFETY GATE
          ->cheap rerank->dedupe/diversity->top5-10   (p99 ~20ms, index in RAM)
DATA:     logs->bot filter->normalize->decay freq->FREQ FLOOR->safety prune->build FST
FEEDBACK: impressions+outcome->position-bias correct (IPS/exploration)->retrain

LABEL: accepted AND search succeeded (NOT raw clicks)

EVAL: success-weighted acceptance, keystrokes saved, NDCG, prefix coverage, calibration
  offline-up/online-down: (1) exposure bias (2) clicks!=success (3) LATENCY regress
    (4) freshness blind (5) novelty (6) train-serve skew
  A/B: user-sticky, ramp 1-5-20-50, guardrail p99<=20ms + unsafe=0, CUPED, 1wk

DEPLOY: immutable index+model bundle, pointer-flip rollback, validate index build,
  degrade ladder: rerank down->trie tops | bad build->last-good | no perso->global pop
INCIDENT: bad suggestion -> hot-patch blocklist (seconds) + flip to last-good
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 100M+ users / 10x QPS?** Push more to edge POPs and on-device prefix caching; the in-RAM index already scales horizontally (replicate per node, it is read-only); shard personalization KV; make logging fully async; tighten the head cache. The index size, not QPS, is usually the binding constraint.
- **How do you handle typos / fuzzy prefixes?** Add bounded edit-distance traversal in the FST (Levenshtein automaton) or a spell-correction normalization step before lookup; cap fuzziness tightly to protect latency.
- **CJK / transliteration / IMEs?** Index on both the typed script and romanization; for CJK, complete at the input-method level and on segmented tokens, not raw bytes. Pin language early — it is a different index.
- **How do you pick how many suggestions to show / top-K?** Business choice under guardrails: more suggestions help discovery but hurt scannability and increase the chance of a bad one; tune K with online experiments, often 5-10.
- **Offline up, online down — what first?** For typeahead, check latency first (slower model misses the budget), then exposure bias, then clicks-vs-success.
- **How do you prevent the feedback loop / popularity bias?** Exploration slot for unbiased tail/fresh labels, freshness boosting, downweight raw popularity, monitor head-diversity and tail coverage, use counterfactual/IPS-corrected training.
- **Personalization without creepiness?** Use aggregate behavioral signals, never surface another user's private query or a named private individual, keep personal completions to the user's own history, and treat complaint rate as a hard guardrail.
- **Zero-result suggestions in a marketplace?** Join inventory/availability at index-build and (coarsely) at serve time; never suggest a query that currently returns nothing.

---

## 13. Common mistakes

- Assuming a generous latency budget — not subtracting network and render, then proposing a model that cannot fit in the real ~20 ms server window.
- Reusing a document-retrieval mental model (BM25/ANN over a corpus) when the candidates are *queries* served from a prefix structure.
- Putting a heavy neural reranker on the synchronous keystroke path instead of precomputing/distilling it off the hot path.
- Optimizing raw click/acceptance instead of accepted-and-successful, rewarding obvious completions with no downstream value.
- Ignoring the self-reinforcing feedback loop and shipping a model that just reproduces yesterday's suggestions.
- Treating safety as a soft demotion instead of a hard pre-rank gate — forgetting that a suggestion is an authored, attributable statement.
- No freshness/streaming path, so trending terms are missing for hours.
- Treating deployment as "an endpoint" instead of an immutable index+model bundle with index-build validation, pointer-flip rollback, and a graceful degrade ladder.

---

## 14. Transfer: what this case unlocks

- **File 03 (search ranking):** typeahead shares the LTR reranker, exposure-bias correction, and offline/online-gap reasoning — but here the candidate is a query and the budget is per keystroke. Contrast them explicitly in an interview.
- **File 02 (feed ranking):** same position-bias and feedback-loop machinery; feed has a looser budget and richer per-item content.
- **File 16 (PYMK / entity typeahead):** entity-completion typeahead reuses the prefix retrieval but adds candidate-generation-as-graph and creepiness governance.
- **File 11 (ads CTR / experimentation):** the A/B discipline, guardrails, CUPED, and sticky assignment transfer directly.
- **File 20 (drift / retraining):** the feedback loop and freshness lag here are a concrete instance of the drift/retraining-trigger machinery.
- **The reusable muscle:** deriving a latency budget out loud, splitting retrieval (data structure) from ranking (model), and refusing to put expensive compute on a hot path — applies to any low-latency ML serving problem.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research: TF-Ranking: https://research.google/blog/advances-in-tf-ranking/

Added (canonical, for the techniques cited above):
- Lucene/Elasticsearch completion suggester (FST-based prefix completion): https://www.elastic.co/guide/en/elasticsearch/reference/current/search-suggesters.html#completion-suggester
- FST in Lucene (Michael McCandless, "Using Finite State Transducers in Lucene"): https://blog.mikemccandless.com/2010/12/using-finite-state-transducers-in.html
- Cai & de Rijke, "A Survey of Query Auto Completion in Information Retrieval" (Foundations and Trends in IR, 2016): https://www.nowpublishers.com/article/Details/INR-055
- Bar-Yossef & Kraus, "Context-Sensitive Query Auto-Completion" (WWW 2011): https://dl.acm.org/doi/10.1145/1963405.1963424
- Joachims et al., "Unbiased Learning-to-Rank with Biased Feedback" (WSDM 2017, position-bias / IPS): https://arxiv.org/abs/1608.04468
- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview" (2010): https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
