# 01. Recommendation System — ML System Design Case Study

**Company tags:** Meta, Amazon, Google, Pinterest, Netflix, Spotify, TikTok, YouTube
**Interview frequency:** Very high
**Why it matters:** This is the canonical ML system design case. One problem forces you through candidate generation, ranking, feedback loops, cold start, multi-task objectives, position bias, A/B testing, and serving infra. If you can drive this one to the floor, ad ranking, feed ranking, search ranking, and "people you may know" are the *same skeleton* with different nouns.

---

## How to use this document

This is not a checklist to memorize. It is built two ways at once:

1. **A thinking guide** — the reusable mental scaffold for *any* ML system design question, with the recsys-specific content poured in.
2. **A worked one-hour interview transcript** — `INTERVIEWER:` / `YOU:` dialogue showing how the conversation actually unfolds under pressure, where the interviewer interrupts, and how a strong candidate answers in tradeoff language.

Read the thinking guide first (Sections 0–8). Then read the transcript (Section 9) and notice how every line maps back to the scaffold. The goal is that you internalize *the route, not the destination* — so when an interviewer takes you somewhere unexpected, you still know which scaffold-step you're standing on.

> **The single most important habit:** every design decision must trace back to a number you stated up front (scale, latency, QPS) or a constraint the interviewer gave you. "I'd use a two-tower model" is junior. "At 100M items I can't score the full catalog in 100ms, so I split into cheap retrieval and expensive ranking — that forces a two-tower retriever" is senior. The *reason* is the content. The model name is not.

---

## Section 0: The reusable scaffold (learn this ONCE)

Every ML system design answer is this skeleton. You are not learning 15 case studies — you are learning **one structure** and 15 sets of fill-ins.

```text
1. Clarify        -> turn an ambiguous product goal into a scoped problem + numbers
2. Frame as ML    -> what exactly are we predicting? what is the label? non-ML baseline?
3. Data & labels  -> where do labels come from, and WHY are they biased?
4. Baseline       -> simplest shippable thing, then name what breaks
5. Model          -> climb the ladder; explain ONE architecture to the floor
6. Evaluation     -> offline metrics, online A/B, and the gap between them
7. Deploy         -> serving path / data path / feedback path (three separate things)
8. Monitor        -> drift, delayed labels, retraining, fallback, incident response
```

Memorize this column. In the interview, you can literally say "I'll go: requirements, ML framing, data, baseline, model, eval, deployment, monitoring — and I'll start simple and add complexity where it's earned." Stating the map up front is itself a senior signal: it shows you've operated systems, not just trained models.

**The three-path mantra (say this out loud during deployment):** a production ML system is not "a model behind an endpoint." It is three independent paths:
- **Serving path** — request comes in, features fetched, candidates retrieved, ranked, returned. Tight latency budget.
- **Data path** — logs → feature store → training data. Offline, batch, can be slow.
- **Feedback path** — what the user did → labels → next training set. *This is where bias enters and where most candidates sound shallow.*

---

## Section 1: Clarify requirements (and pin down NUMBERS)

The interviewer gives you a deliberately vague prompt. Your first job is to scope it and extract numbers. **Do not start drawing models.** Spend the first 5 minutes here.

### Interview prompt
> "Design a recommendation system for a large consumer platform." (movies / products / music / short video — ask which.)

### The clarifying questions that actually change the design

Ask these, and notice *why each one branches the design*:

| Question | Why it changes the design |
|---|---|
| **Which surface?** Homepage feed, "related items" on a product page, search results, or notifications? | A "related items" surface is item→item (you have a strong query item, personalization matters less). A cold homepage feed is user→items (personalization and freshness dominate). Completely different retrieval. |
| **What is the business objective?** Engagement, revenue, retention, or discovery? | Optimizing pure CTR creates clickbait and rich-get-richer loops. Optimizing watch-time creates different failures. The *objective* defines your label and your guardrails. |
| **How many users and items?** | This sets your whole architecture. 1M items vs 1B items is the difference between "score everything" and "mandatory multi-stage funnel." |
| **Latency budget?** | A user-facing feed needs p99 < ~200ms end-to-end. That budget gets *split* across retrieval + ranking + reranking. It's the constraint that forces the funnel. |
| **How fresh must recommendations be?** | News/short-video need second-to-minute freshness (item embeddings must update constantly). Movies can tolerate hourly/daily. This decides batch vs streaming feature pipelines. |
| **Is there explicit feedback** (ratings, thumbs) or only implicit (clicks, dwell)? | Implicit-only means you have a positive-unlabeled problem and heavy bias. Explicit feedback is rarer but cleaner. |

> **Junior move:** start listing models.
> **Senior move:** "Before I design anything — which surface, what objective, roughly how many items, and what's my latency budget? Those four answers change the architecture completely."

### Pin the numbers (carry these through the ENTIRE answer)

State assumptions explicitly and *use them later*. This numerical spine is what separates a real answer from a rubric. Example assumption set for a video platform:

```text
Users (DAU):        100M
Items (catalog):    100M videos
Requests:           each active user loads a feed ~10x/day
                    100M * 10 / 86400s  ~= 11,500 QPS average
                    peak ~3x average     ~= 35,000 QPS
Latency SLO:        p99 < 200ms end-to-end for the feed
Items returned:     ~10-20 per page
```

**The latency budget split** (this is gold — derive it out loud):
```text
200ms total budget, roughly:
  feature fetch        ~20ms
  candidate retrieval  ~30ms   (must be cheap -> ANN over embeddings)
  ranking              ~80ms   (richer model, but only on ~hundreds of items)
  reranking + business ~30ms
  network/overhead     ~40ms
```
Now every later decision has a reason: "I can't run a heavy cross-attention model over 100M items in 30ms, so retrieval must be a precomputed-embedding ANN lookup. The expensive model only runs on the few hundred survivors."

### Storage back-of-envelope (do this reflexively)
```text
100M item embeddings * 128 dims * 4 bytes (float32) = ~51 GB
-> too big for comfortable single-box RAM at scale + replicas
-> shard the ANN index, or quantize embeddings to int8 (~13 GB)
```
Saying "100M × 128 × 4B ≈ 51GB, so I'd shard or quantize the index" in 10 seconds is a strong signal. Practice this until it's reflexive.

---

## Section 2: Frame it as an ML problem

The product goal ("recommend good stuff") is not learnable. Your job is to convert it into a precise prediction target with a measurable label.

- **Framing:** multi-stage **retrieval and ranking**. (Say this phrase — it signals you know the funnel exists.)
- **Prediction target:** the probability (or expected value) that a given **user** will have a positive interaction with a given **item** in a given **context**.
- **What is a positive label?** This is a *design choice*, not a given. Click? Long watch (>30s or >70% completion)? Add-to-cart? Purchase? Like? A click is cheap but noisy (clickbait). A purchase is sparse but high-signal. **Strong answer: define the positive explicitly and acknowledge the tradeoff.** e.g. "I'll use long-watch as the primary positive because raw clicks reward clickbait and hurt long-term retention."
- **Non-ML baseline (always name one):** most-popular-by-category, editorially-curated lists, trending-now, or "more from creators you follow." This is your launch-day fallback *and* your cold-start safety net. Never skip it — interviewers notice.

> **Why this framing matters (say it):** separating the product goal from the learnable target gives you measurable labels, lets you stage complexity, and gives you a fallback when the model fails. It also makes the objective explicit so you can argue about it: "we're not predicting clicks, we're predicting long-term satisfaction, and clicks are a biased proxy for that."

---

## Section 3: Data, features, and the label-bias problem

This section is the **intellectual core** of recsys. Most candidates list feature nouns and move on. The signal is in confronting *where labels come from and why they lie to you.*

### Data sources
- **Interaction logs (the gold):** impressions, clicks, watches, dwell time, purchases, skips, hides, follows.
- **Item metadata:** category, creator, language, age, text/title, thumbnail, price, availability.
- **User profile:** demographics (sparingly — fairness/privacy), declared interests, history.
- **Context:** time of day, device, location, session state, what they *just* watched.

### Features — as engineered objects, not nouns

The point of the funnel is that **retrieval and ranking use different features for latency reasons.** Make this concrete:

**Retrieval (two-tower) features — must be cheap & precomputable:**
- *User tower input:* sequence of last-N interacted item IDs (pooled or sequence-encoded), declared interests, coarse demographics, current context.
- *Item tower input:* item ID embedding, category, creator, content embedding (text/image), age bucket.
- Critically: the towers **never see user×item cross features** — they're computed independently so item embeddings can be precomputed and indexed. That independence is the whole trick.

**Ranking features — can be rich, including cross features:**
- User×item cross features the retriever *can't* afford: "how many videos from this creator did this user finish in the last 7 days?", "user's affinity to this category × item's category", "time since user last saw this item."
- Real-time counts: item's CTR in the last hour, item velocity/trending score.
- Sequence features: full attention over the user's recent session.
- These cross features are *why* a separate ranking stage exists. Say this explicitly: **"retrieval can only use representations that factorize into user-side and item-side; ranking can use features that depend on the pair, which is strictly more expressive but only affordable on a few hundred candidates."**

### Online vs offline features
- **Offline (batch):** long-window aggregates, embeddings, labels, historical CTR. Computed in the data path, written to the feature store.
- **Online (real-time):** current session events, recent counts, inventory/availability, what they clicked 5 seconds ago. Fetched live in the serving path.
- **The train-serve skew trap:** if a feature is computed one way in training (batch, full history) and another way at serving (streaming, partial), your offline metrics lie. Name this risk and the fix: **compute features once in a shared feature store, log the exact features served, and train on logged features (not recomputed ones).**

### The label-bias problem (THE part to nail)

Your labels are generated by *your own current system*. This poisons everything:

1. **Position bias:** items at the top get more clicks *because they're at the top*, not because they're better. If you train naively, you learn "things we already rank high are good" — a self-fulfilling loop.
2. **Exposure / selection bias:** you only observe feedback on items you *chose to show*. You have no labels for the 99.9999% of the catalog the user never saw. This is a **positive-unlabeled** problem, not a clean binary classification.
3. **Popularity bias:** popular items appear more, get more positive labels, get recommended more → rich-get-richer. Niche/long-tail items starve.
4. **Feedback delay:** a purchase or a "did they churn" label arrives hours or days later. Training on immediate clicks optimizes the wrong thing.

**What you DO about it (this is the senior content):**
- **Position bias correction:** the *shallow tower / position-as-feature* trick (from YouTube's "Recommending What Video to Watch Next" paper). At training time, feed the item's display position as an input feature so the model attributes part of the click to position. At serving time, set that feature to a fixed value (e.g., position 1 or 0) for all items, so the model predicts click probability *as if position-neutral*. The bias is absorbed into the position feature instead of contaminating the content features.
- **Exposure bias / negatives:** you must *manufacture* negatives because you only logged positives. See negative sampling below.
- **Popularity bias:** down-weight frequent items in the loss (the **sampled-softmax with log-Q correction** — subtract `log(sampling_probability)` from the logit so popular items don't dominate). Reserve **exploration traffic** to gather unbiased labels.
- **Counterfactual / randomized logging:** reserve a small slice of traffic that shows randomized or inverse-propensity-weighted results, giving you (expensive but) unbiased data to evaluate against.

> **Junior says:** "I'll train on clicks."
> **Senior says:** "Clicks are biased by position, exposure, and popularity. I'll correct position bias with a position feature zeroed at serve time, manufacture negatives via in-batch + hard negative sampling, apply a log-Q popularity correction, and reserve exploration traffic for unbiased evaluation."

### Cold start (make it a strategy, not a list)
- **New item:** no interaction history → lean on the **content embedding** (text/image/audio of the item itself) so the item tower produces a reasonable vector from content alone. Then use **exploration** (show it to a controlled slice) to gather initial signal.
- **New user:** no history → use onboarding signals (declared interests), coarse demographics, popularity priors, and rapid online adaptation from the first few in-session clicks.
- **Frame it as explore/exploit:** new items/users are an **exploration** problem. Mention **epsilon-greedy** (show random items ε% of the time), or **Thompson sampling / contextual bandits** for a principled exploration budget. Saying "cold start is fundamentally an exploration problem, so I'd allocate an exploration budget via a bandit" is a strong, concrete answer.

---

## Section 4: Baseline first — then name exactly what breaks

The ladder is the content. Don't jump to two-tower. *Climb*, and name the **trigger** for each rung — the specific thing that breaks and forces the next step.

```text
RUNG 0: Popularity / rules
   -> "Top items by category, trending now."
   BREAKS: zero personalization; rich-get-richer; useless for niche intent.
   TRIGGER to climb: we need per-user personalization.

RUNG 1: Collaborative filtering / matrix factorization
   -> Factor the user-item interaction matrix into user & item latent vectors.
   FIXES: personalization from co-occurrence ("users like you watched X").
   BREAKS: severe cold start (new user/item has no row/column);
           can't use content or context features; sparse at scale.
   TRIGGER: we need to handle new items/users and use features + context.

RUNG 2: Two-tower retrieval  +  ranking model   <-- DEFAULT PRODUCTION ANSWER
   -> Tower model maps user & item into a shared embedding space (handles
      cold start via content features); ANN retrieves top candidates;
      a richer ranker scores them with cross features.
   FIXES: cold start, content/context features, scales via ANN, two-stage latency.
   BREAKS / costs: embedding freshness, negative sampling care, train-serve skew,
      multi-objective tradeoffs.
   TRIGGER (only if pushed): session dynamics / sequence matter a lot.

RUNG 3: Sequence-aware transformer reranker (advanced extension)
   -> Self-attention over the user's recent session for final top-K ordering.
   USE WHEN: session intent shifts fast (short video, shopping sessions) and
      you can afford it on a small candidate set, possibly distilled.
   COST: latency and serving cost; needs distillation or strict candidate cap.
```

> **Say this:** "I'd ship Rung 0 day one as both a launch baseline and a permanent fallback. I'd move to Rung 2 as the production design because it's the first rung that handles cold start, content features, and 100M-item scale simultaneously. I'd only reach for Rung 3 if session dynamics turn out to dominate, because it costs latency I have to justify."

This earns the jump to a complex model instead of cargo-culting it.

---

## Section 5: Model architecture — ONE design, to the floor

Depth on one architecture beats naming four. Here is the two-tower retriever + multi-task ranker, explained as if you have to draw it on the whiteboard.

### 5.1 The two-tower retriever (candidate generation)

**The picture:**
```text
   USER TOWER                         ITEM TOWER
   --------------                     --------------
   user features                     item features
   (last-N item IDs,                 (item ID, category,
    interests, context)               creator, content emb)
        |                                  |
     MLP / encoder                     MLP / encoder
        |                                  |
   user vector  u  (d-dim)           item vector  v  (d-dim)
        \                                  /
         \________  score = u . v  _______/
                    (dot product / cosine)
```

**Why two separate towers — the entire reason it works:**
- The score is just `u · v`. Because the user vector and item vector are computed **independently**, you can **precompute every item vector offline** and load them into an **ANN index** (FAISS, ScaNN, HNSW).
- At serving time you compute **only the user vector** (one forward pass), then do an approximate nearest-neighbor search to get the top ~hundreds of items by dot product — in ~tens of milliseconds over 100M items.
- A single combined model that takes (user, item) together as input *cannot* do this — you'd have to run it 100M times per request. **That's the latency argument for why retrieval must factorize.** Say exactly this when asked "why not one model?"

**The loss — sampled softmax with in-batch negatives:**
- Treat retrieval as "given this user, pick the item they interacted with out of the whole catalog" → a giant softmax over 100M items. That's intractable, so you **sample negatives.**
- **In-batch negatives:** within a training batch of (user, positive-item) pairs, every *other* item in the batch is a negative for this user. Free negatives, very efficient.
- **The popularity correction (log-Q / sampled softmax correction):** in-batch negatives are sampled proportional to popularity, so popular items appear as negatives too often and get unfairly penalized. Correct by subtracting `log(P(item))` from each logit. (This is the YouTube/Google "sampling-bias-corrected" two-tower trick — name it.)
- **Hard negatives:** in-batch negatives are usually *easy* (random items the user obviously wouldn't pick). Mixing in **hard negatives** — items similar to the positive but not interacted with — sharpens the decision boundary. e.g. items the user saw but skipped.

**Embedding freshness (a real operational concern):** item embeddings are precomputed, so a new or changed item's vector is stale until the next batch job. For fast-moving catalogs you re-embed hot items more frequently or run a streaming embedding update. Name this as the cost of the precompute trick.

### 5.2 The ranker (precision stage)

Retrieval gives you ~hundreds of candidates optimized for **recall** (don't miss good items). The ranker reorders them for **precision/utility** on a small set, so it can afford to be expensive and use cross features.

**Model choice ladder for the ranker:**
- **GBDT (XGBoost/LightGBM)** on hand-engineered cross features — strong, cheap, interpretable baseline. Great first ranker.
- **DNN / DLRM-style** — embeddings for categoricals + MLP over dense + cross features. Scales to richer features and interactions.
- **DCN / Wide&Deep** — explicitly model feature crosses. Industry-standard for ranking.

**Loss — pointwise vs pairwise vs listwise (know the difference cold):**
- **Pointwise:** predict P(click) per item independently, binary logloss. Simplest, well-calibrated, easy to combine with business value. Most production rankers start here.
- **Pairwise (e.g., BPR, RankNet):** learn that item A > item B for this user. Optimizes ordering directly, ignores absolute calibration.
- **Listwise (e.g., LambdaMART, ListNet):** optimize a whole-list metric like NDCG directly. Best ranking quality, most complex.
- **Say:** "I'd start pointwise logloss because it's calibrated and lets me blend in business value as `P(click) × value`, then move to pairwise/listwise if ordering quality is the bottleneck."

### 5.3 Multi-objective ranking (2024+ table stakes — most candidates miss this)

Real systems don't optimize one thing. You care about clicks AND long watches AND likes AND shares AND *not* showing things people hide. The modern answer:

- **Multi-task learning:** one network with **multiple heads**, each predicting a different objective (P(click), P(long-watch), P(like), P(share), P(hide)). Shared bottom layers learn common representation; task-specific heads specialize.
- **MMoE (Multi-gate Mixture-of-Experts):** the standard architecture (YouTube, used widely) — a set of shared "expert" subnetworks, and a per-task **gate** that learns how much each task draws from each expert. Solves the problem that some objectives conflict (clickbait drives clicks but kills watch-time).
- **Combining the heads into one score:** a weighted combination, e.g.
  `final_score = w1·P(watch) + w2·P(like) + w3·P(share) − w4·P(hide)`
  where weights encode business priorities. The weights are tuned via **online A/B testing**, not learned offline — because the "right" tradeoff is a business decision, not a loss-minimization. Saying this is a strong senior signal.

### 5.4 Reranking / business logic layer (the final ~30ms)

After the ML score, a post-processing layer applies non-ML constraints:
- **Diversity / dedup:** don't show 8 videos from the same creator (MMR — maximal marginal relevance, or rule-based interleaving).
- **Freshness boost:** inject some new/recent items.
- **Business rules:** promoted content, policy filters, already-seen suppression, inventory/availability.
- **Exploration injection:** slot in a few exploration items to gather unbiased labels.

Saying "the final ranked list isn't the raw model output — there's a business/diversity reranking layer on top" shows you've shipped real systems where the model is one input among several.

### 5.5 The full architecture diagram (draw this)

```text
                    REQUEST (user_id, context)
                              |
                     [ Feature Store ]  <- online + offline features
                              |
          user features ------+------ (item features precomputed offline)
                              |
                     [ User Tower ] -> user vector u
                              |
                   [ ANN Index (FAISS/ScaNN) ]   <- 100M precomputed item vecs
                              |
                  top ~500 candidates (RECALL-optimized)
                              |
                     [ Ranker: MMoE multi-task ]   <- rich cross features
                              |
              scored candidates (PRECISION-optimized)
                              |
        [ Reranker: diversity / business rules / exploration ]
                              |
                      Top 10-20 -> USER
                              |
                     (impressions + interactions logged)
                              |
            [ Feedback path -> labels -> next training set ]
```

---

## Section 6: Evaluation — and the offline/online gap

Evaluation is not "list NDCG." The interesting content is the **gap between offline and online**, because that gap is where real systems fail.

### Offline metrics
- **Retrieval:** **Recall@K** — of the items the user actually engaged with, how many were in the top-K retrieved set? Retrieval is graded on recall because its job is "don't lose good items." Also **coverage** (what fraction of the catalog ever gets retrieved — catches popularity collapse).
- **Ranking:** **NDCG@K** (rewards putting relevant items high, with position discount), **MAP**, **MRR**. **Calibration** (do predicted probabilities match observed rates? — matters a lot if you blend `P(click) × value`).
- **Always slice by cohort:** new vs returning users, head vs tail items, by country/device. Aggregate metrics hide failures. Say "I'd slice every metric by cohort and keep a bank of bad examples."

### Online metrics (the ones that actually decide launch)
- **Product:** CTR, long-watch rate, completion, add-to-cart, repeat sessions, session length.
- **Business:** revenue per session, retention, DAU/WAU, subscription renewal.
- **Guardrails (must not regress):** p95/p99 latency, diversity, hide/complaint rate, creator fairness, long-tail exposure.

### The offline→online gap (THE evaluation talking point)

> **The classic interview question: "Offline NDCG improved but online CTR dropped. What happened?"**

Strong answer enumerates causes:
1. **Train-serve skew** — a feature computed differently offline vs online.
2. **Label bias** — offline metric rewards matching *past* (biased) logs; you got better at predicting the old system's behavior, not at making users happy.
3. **Novelty / feedback loop** — offline data can't capture that users get bored, or that the new model collapses diversity.
4. **Proxy mismatch** — offline you optimized P(click), but clicks proxy long-term satisfaction badly; online you measured something closer to truth.
5. **Position bias not corrected** — offline metric computed on biased positions.

**The lesson to state:** offline metrics are necessary for iteration speed but never sufficient. Online A/B is the only real arbiter. Offline is a filter to decide *what's worth A/B testing.*

### A concrete A/B test (fully specified — don't hand-wave)

Interviewers love "design the experiment." Give all of this:

```text
Hypothesis:        The new MMoE ranker increases long-watch rate
                   without hurting diversity or latency.
Unit of randomization: user (not request) — so a user gets a
                   consistent experience and we avoid contamination.
Control:           current production ranker.
Treatment:         new MMoE ranker.
Primary metric:    long-watch rate per session.
Guardrails:        p99 latency, hide rate, creator-diversity index,
                   long-tail exposure share. Any guardrail regression
                   beyond threshold = auto-rollback.
Ramp:              1% -> 5% -> 20% -> 50%, holding at each stage to
                   check guardrails.
Sample size / MDE: size the experiment to detect, say, a 1% relative
                   lift in long-watch at 95% confidence / 80% power.
                   (Mention you'd compute n from baseline variance &
                   MDE; the point is you KNOW you must power it.)
Duration:          run >= 1-2 full weeks to cover weekly seasonality
                   and novelty decay (the "primacy/novelty effect" —
                   a new model gets a temporary bump that fades).
Decision / rollback: ship if primary metric up & no guardrail
                   regression; rollback on any guardrail breach.
```

Mention **novelty effect** and **weekly seasonality** explicitly — they're the two things that make people read short experiments wrong.

---

## Section 7: Deployment & serving (the three paths, concretely)

> Reframe from "an endpoint" to **serving path / data path / feedback path.** This is the single fastest way to sound like someone who has operated ML systems.

### Serving path (online, latency-critical)
```text
request -> feature store lookup -> user tower forward pass
       -> ANN retrieval (top ~500) -> ranker (top ~50)
       -> rerank/business -> return top 10-20
```
- Keep **candidate generation, feature retrieval, scoring, post-processing as separate services** so you can scale and fail them independently.
- ANN index sharded across machines (recall the 51GB estimate → must shard or quantize).
- Cache aggressively: user embeddings, hot candidates, feature aggregates.

### Data path (offline, batch)
```text
raw logs -> ETL -> feature store (offline tables) -> training data
        -> train two-tower + ranker -> validate -> push embeddings to ANN
        -> push model to server
```
- Item embeddings recomputed on a schedule (and hot items more often).
- **Log the exact features served** so you train on what was actually served (kills train-serve skew).

### Feedback path (closes the loop — where bias lives)
```text
impressions + clicks + watches + skips/hides -> labeled events
   -> join with logged features -> next training set
   -> (apply position-bias correction, exploration data) -> retrain
```

### Rollout discipline
```text
shadow mode (score live traffic, don't serve) -> canary (small %)
   -> A/B test (ramp) -> gradual full rollout
```
Shadow mode catches latency/crash issues with zero user risk before you ever serve a prediction.

---

## Section 8: Monitoring, retraining, incident response

- **Monitor:** data drift (feature distributions shift), prediction drift (output distribution shifts), **delayed-label performance** (today's model judged on labels that arrive tomorrow), latency p95/p99, throughput, error rate, and **business KPI movement** — the model can look healthy while the KPI quietly drops.
- **Retraining:** scheduled retrains for stable domains; **triggered** retrains when drift or performance crosses a threshold. Fast-moving catalogs (news, short video) may retrain hourly or continuously; movies daily/weekly.
- **Fallback:** when the model/feature store fails, degrade gracefully to cached results → popularity baseline → editorial rules. Never serve nothing. (This is why Rung 0 stays in production forever.)
- **Incident response:** freeze the current model version, diff against the previous release (features, model, data), inspect traces and a bank of bad examples, roll back if guardrails moved.

---
## Section 9: The worked one-hour interview (full transcript)

This is how the conversation actually unfolds. Notice: the interviewer interrupts, challenges, and redirects. Watch which scaffold-step each exchange maps to, and how every answer uses tradeoff language and traces back to a stated number.

---

**[00:00 — The prompt]**

**INTERVIEWER:** Let's design a recommendation system. Say it's for the home feed of a short-video app.

**YOU:** Great. Before I design anything, let me scope it with a few questions, because the answers change the architecture a lot. First — is this the cold home feed, where I don't have a query item and personalization plus freshness dominate, as opposed to a "related videos" surface? I'll assume the home feed. Second, what's the primary objective — raw engagement, long-term retention, or something else? Third, rough scale — how many daily users and how many videos in the catalog? And fourth, what's my end-to-end latency budget for the feed?

**INTERVIEWER:** Home feed, yes. Optimize for long-term engagement, not just clicks. Assume 100M DAU, 100M videos. Latency budget, say 200ms p99.

**YOU:** Perfect, let me pin those down and carry them through.
```
100M DAU, ~10 feed loads/day -> ~11.5K QPS avg, ~35K QPS peak
100M videos, p99 < 200ms
```
That 200ms is the constraint that forces everything. I can't score 100M videos with a rich model in 200ms, so I already know I need a **multi-stage funnel**: cheap retrieval to go from 100M to a few hundred, then an expensive ranker on those few hundred, then a light reranking layer. Let me also note: "long-term engagement, not clicks" tells me my **label** can't be raw clicks — clicks reward clickbait and hurt retention. I'll come back to that.

**INTERVIEWER:** Good. Go on.

---

**[00:05 — ML framing & the label]**

**YOU:** I'll frame this as multi-stage retrieval and ranking. The core prediction is: for a (user, video, context) triple, the probability of a *positive* interaction. The key design decision is what counts as positive. Given the "long-term engagement" objective, I'd use **long-watch** — say watching past 70% or some absolute threshold — as the primary positive, not a click. Clicks are a biased, short-term proxy. I'd also keep a non-ML baseline: trending-by-category and "from creators you follow." That's my launch-day fallback and my cold-start safety net.

**INTERVIEWER:** Why not just predict clicks? They're abundant and clean.

**YOU:** They're abundant but not clean. Clicks are biased three ways — position (top items get clicked because they're on top), exposure (I only see clicks on what I chose to show), and popularity. And optimizing clicks directly produces clickbait, which raises CTR while lowering the long-term engagement the business actually asked for. So clicks are a useful *auxiliary* signal but a bad *primary* label. I'd predict long-watch as primary and use click as one of several auxiliary heads.

---

**[00:09 — Data & the bias problem]**

**INTERVIEWER:** Tell me about your training data and labels.

**YOU:** Main source is interaction logs — impressions, clicks, watch-time, skips, hides, follows — joined with item metadata and user/context features. The thing I want to flag immediately is that **these labels are generated by my own current system, so they're biased**, and how I handle that bias matters more than the model choice.

Three biases. **Position bias:** top items get more positive labels regardless of quality. I'll correct it with the position-as-feature trick — feed display position as an input at training time, fix it to a constant at serving time, so the model attributes position-driven clicks to the position feature instead of contaminating content features. **Exposure bias:** I only have labels for items I showed — it's a positive-unlabeled problem, so I have to manufacture negatives via sampling. **Popularity bias:** popular items get over-represented as both positives and as in-batch negatives, so I'll apply a log-Q sampled-softmax correction and reserve a small exploration traffic slice for unbiased labels.

**INTERVIEWER:** Okay, that's the part most people skip. Let's get into the model.

---

**[00:15 — Baseline ladder]**

**YOU:** I'll climb a ladder and only add complexity where it's earned. Rung zero is popularity and rules — I ship it day one, and it stays forever as the fallback. It breaks because it's not personalized. Rung one is matrix factorization — personalizes from co-occurrence, but it has brutal cold start and can't use content or context. For a short-video app with constant new content, cold start is fatal, so I can't stop there. Rung two is a **two-tower retriever plus a ranker** — that's my production answer, because it's the first rung that handles cold start via content features, uses context, and scales to 100M items through ANN. Rung three, a sequence transformer reranker, I'd hold in reserve for when session dynamics dominate — which, for short video, they actually might, so I'll flag it as the most likely extension.

**INTERVIEWER:** Let's go with two-tower. Walk me through it. And tell me why one model can't just score everything.

---

**[00:20 — Two-tower, to the floor]**

**YOU:** *(drawing)* Two towers. A user tower takes the user's recent video IDs, interests, and context and produces a user vector u. An item tower takes item ID, category, creator, and a content embedding and produces an item vector v. The score is just `u · v`.

The reason it's two separate towers — and the answer to "why not one model" — is **latency**. Because u and v are computed independently, I can precompute all 100M item vectors offline and load them into an ANN index like ScaNN or FAISS. At request time I compute *only* the user vector — one forward pass — then do approximate nearest-neighbor search to get the top few hundred items in maybe 30ms. A single model that takes (user, item) jointly can't factorize like that — I'd have to run it 100M times per request, which is impossible in my budget. So the latency constraint *forces* the architecture. The cost I pay is that item vectors are precomputed and therefore slightly stale — for a fast catalog I'd re-embed hot items frequently.

**INTERVIEWER:** How do you train it? Where do negatives come from?

**YOU:** I treat it as "pick the watched item out of the whole catalog" — a softmax over 100M items, which I approximate with sampled softmax. The cheapest negatives are **in-batch negatives**: for each (user, positive) pair in a batch, every other item in the batch is a negative. But in-batch negatives are sampled by popularity, so popular items get unfairly punished — I correct that by subtracting log of the item's sampling probability from each logit, the log-Q correction. And in-batch negatives are mostly *easy*, so I mix in **hard negatives** — items the user was shown but skipped — to sharpen the boundary.

**INTERVIEWER:** And cold start for a brand-new video with zero history?

**YOU:** The item tower includes a **content embedding** — from the thumbnail, audio, and text — so even a video with zero interactions gets a sensible vector from content alone. Then I treat surfacing it as an **exploration** problem: allocate a small exploration budget, via epsilon-greedy or a contextual bandit, to show new items to a controlled slice and gather initial signal. Cold start is fundamentally explore/exploit, not a feature-engineering afterthought.

---

**[00:30 — The ranker & multi-objective]**

**INTERVIEWER:** Retrieval gives you candidates. Now rank them.

**YOU:** Retrieval handed me ~500 candidates optimized for recall. The ranker reorders them for precision, and because it only runs on a few hundred items, it can afford rich features — crucially, **user×item cross features** the towers couldn't use: "how many videos from this creator did this user finish this week," "user-category affinity × item category," real-time item velocity. Those cross features are the entire reason a separate ranking stage exists — they depend on the pair, so they can't be precomputed per item.

For the model, I'd start with a DNN ranker, pointwise logloss, because it's calibrated and lets me blend business value as probability times value. But here's the important part: the business asked for long-term engagement, which is *not one objective*. So I'd use **multi-task learning** — one network, multiple heads predicting P(long-watch), P(like), P(share), P(hide). I'd use an **MMoE** architecture so conflicting objectives — clickbait drives clicks but kills watch-time — can draw on different experts via per-task gates. Then I combine the heads into a final score like `w1·P(watch) + w2·P(like) + w3·P(share) − w4·P(hide)`, and critically, **I tune those weights via online A/B testing, not offline loss**, because the right tradeoff is a business decision, not a minimization.

**INTERVIEWER:** Where do the final results come from — straight off the ranker?

**YOU:** No — there's a reranking layer. The raw model order gets diversity constraints so I'm not showing eight clips from one creator — maximal marginal relevance or interleaving — plus freshness boosts, already-seen suppression, policy filters, and a few injected exploration items. The served list is the model output as *one input* among several.

---

**[00:40 — Evaluation & the offline/online gap]**

**INTERVIEWER:** How do you know any of this works before launch? And here's a scenario: your offline NDCG goes up, but in the A/B test CTR drops. What happened?

**YOU:** Offline, I grade retrieval on Recall@K and coverage, and ranking on NDCG@K and calibration, sliced by cohort — new vs returning users, head vs tail items — because aggregates hide failures. But offline is only a *filter* for what's worth A/B testing; the online test is the real arbiter.

On your scenario — NDCG up, CTR down — that's the classic gap, and there are a few likely causes. **Train-serve skew**: a feature computed differently in batch training than in streaming serving. **Proxy mismatch**: offline NDCG was computed against biased historical logs, so I got better at predicting my *old system's* behavior, not at satisfying users. **Position bias** not fully corrected in the offline metric. **Diversity collapse**: the new model is more "accurate" by NDCG but narrows the feed, so users engage less. I'd diff features between train and serve first, check whether my offline labels are just echoing the old ranker, and look at the diversity guardrail. The lesson is that offline metrics are necessary for iteration speed but never sufficient.

**INTERVIEWER:** Design the A/B test you'd run for the new ranker.

**YOU:** Hypothesis: the MMoE ranker raises long-watch rate without hurting diversity or latency. Randomize by **user**, not request, so the experience is consistent and there's no contamination. Control is the current ranker, treatment is MMoE. Primary metric: long-watch rate per session. Guardrails: p99 latency, hide rate, creator-diversity index, long-tail exposure — any breach auto-rolls-back. Ramp 1 to 5 to 20 to 50 percent, checking guardrails at each hold. I'd power it to detect about a 1% relative lift at 95% confidence and 80% power, sizing n from baseline variance. And I'd run it at least one to two full weeks to cover weekly seasonality and let the **novelty effect** decay — a new model gets a temporary bump that fades, and short tests misread that as a win.

---

**[00:48 — Serving, scale, monitoring]**

**INTERVIEWER:** What changes at 100M users on the serving side?

**YOU:** I think in three paths. The **serving path** — feature lookup, user-tower pass, ANN retrieval, ranker, rerank — I keep as separate services so they scale and fail independently, and I shard the ANN index because 100M times 128-dim float32 is about 51GB, too big for one box, so I shard or quantize to int8. I cache user embeddings and hot candidates and use async logging so logging never blocks serving. The **data path** is the offline batch that recomputes embeddings and training data, and it logs the exact features served so I train on what was actually served — that's my main defense against train-serve skew. The **feedback path** joins interactions back into labels with position-bias correction applied, and feeds the next retrain.

**INTERVIEWER:** And keeping it healthy in production?

**YOU:** Monitor data drift, prediction drift, delayed-label performance, latency, and — most importantly — the business KPI, because the model can look healthy while engagement quietly drops. Retrain on a schedule for stability plus triggered retrains on drift; for short video that's probably hourly-ish given how fast content moves. Fallback degrades gracefully: cached results, then popularity baseline, then editorial rules — never serve nothing, which is exactly why Rung 0 lives in prod forever. On an incident I freeze the version, diff against the last release, inspect a bank of bad examples, and roll back if guardrails moved.

---

**[00:55 — The close]**

**INTERVIEWER:** Anything you'd add?

**YOU:** To restate the core tradeoff: I optimize long-watch and the multi-objective engagement score while protecting diversity, freshness, latency p99, and creator fairness as guardrails. I start with a popularity baseline that doubles as a permanent fallback, use a two-tower retriever because the 200ms budget over 100M items forces a factorized cheap-retrieval stage, and an MMoE ranker because "long-term engagement" is inherently multi-objective. The thing I'd watch hardest is the feedback loop — biased labels, position bias, popularity collapse — because that's where these systems quietly fail.

And concretely, I've shipped this shape before: in Seller Copilot we did semantic retrieval plus cross-encoder re-ranking for product discovery, which is exactly this retrieval-then-rank funnel — so I've operated the two-stage pattern, the feature-skew problems, and the rollout discipline in production, not just on a whiteboard.

**INTERVIEWER:** That's a strong answer.

---

> **Why this transcript works (study these moves):**
> 1. **Scoped before designing** — 4 questions, each tied to *why it changes the design*.
> 2. **Pinned numbers and carried them** — every architecture choice traced back to 200ms / 100M.
> 3. **Climbed the ladder** — earned the two-tower instead of cargo-culting it.
> 4. **Explained ONE thing to the floor** — the "why two towers" latency argument.
> 5. **Confronted label bias head-on** — the real intellectual core.
> 6. **Tradeoff language throughout** — "I'd start here because of X, move there if Y."
> 7. **Multi-objective + MMoE** — the 2024+ depth most candidates miss.
> 8. **Offline/online gap** — answered the trap scenario with 4 named causes.
> 9. **Fully specified A/B test** — with novelty effect and seasonality.
> 10. **Closed by connecting to real shipped work** — Seller Copilot, *leading* with it, not burying it.

---
## Section 10: Junior vs Senior — the highest-leverage contrast

The fastest way to level up an answer is to know what a senior says differently on the same decision.

| Decision | Junior answer | Senior answer |
|---|---|---|
| Where to start | "I'd use a two-tower model." | "I'd ship popularity baseline day one, then climb to two-tower because it's the first rung that handles cold start + content + 100M scale." |
| Why a funnel | "First retrieve, then rank." | "I can't score 100M items in a 200ms budget, so retrieval must factorize into precomputable embeddings; ranking gets cross features only affordable on hundreds of items." |
| The label | "Train on clicks." | "Clicks are biased and proxy long-term value poorly; primary label is long-watch, clicks are an auxiliary head." |
| Negatives | (doesn't mention) | "In-batch negatives with log-Q popularity correction, plus hard negatives from shown-but-skipped items." |
| Position bias | (doesn't mention) | "Position-as-feature at train, fixed at serve, so position-driven clicks don't contaminate content features." |
| Objective | "Predict the click probability." | "It's multi-objective — MMoE with heads for watch/like/share/hide, weights tuned via A/B, not offline loss." |
| Eval | "I'd measure NDCG." | "Offline NDCG filters what to test; online A/B is the arbiter; and here's why offline can rise while online falls." |
| Deployment | "Deploy the model to an endpoint." | "Three paths — serving, data, feedback — and the feedback path is where bias enters." |
| Cold start | "Use popularity for new users." | "Cold start is explore/exploit; allocate an exploration budget via bandit; new items lean on content embeddings." |

---

## Section 11: One-page cheat sheet (whiteboard recall)

```text
SCAFFOLD:  Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor

NUMBERS:   state DAU, #items, QPS (=DAU*loads/86400), latency SLO, storage (#items*dim*4B).
           Split the latency budget across funnel stages OUT LOUD.

FUNNEL:    100M items --retrieval(recall)--> ~500 --ranker(precision)--> ~50 --rerank--> 10-20

TWO-TOWER: user tower -> u ; item tower -> v ; score = u.v
           item vecs PRECOMPUTED -> ANN index -> only user vec computed live.
           "Why not one model?" -> latency: can't score 100M/request.
           Loss: sampled softmax, in-batch negatives + log-Q correction + hard negatives.

RANKER:    cross features (user x item) ; pointwise logloss to start.
           MULTI-TASK -> MMoE -> heads {watch, like, share, hide}
           final = w1*watch + w2*like + w3*share - w4*hide ; weights tuned by A/B.

RERANK:    diversity (MMR) + freshness + business rules + exploration.

LABEL BIAS (the core):
   position bias -> position feature, zeroed at serve
   exposure bias -> manufacture negatives (PU problem)
   popularity bias -> log-Q correction + exploration traffic
   feedback delay -> delayed-label monitoring

EVAL:      retrieval=Recall@K, coverage ; ranking=NDCG@K, calibration ; slice by cohort.
           OFFLINE filters, ONLINE A/B decides. Know the offline-up/online-down causes.

A/B:       randomize by USER ; primary metric + guardrails(latency,diversity,hide) ;
           ramp 1->5->20->50 ; power it ; run 1-2 wks (novelty + seasonality) ; auto-rollback.

3 PATHS:   serving (latency) | data (batch, log served features) | feedback (bias enters)

MONITOR:   data/prediction drift, delayed labels, latency, BUSINESS KPI.
           fallback: cached -> popularity -> rules (never serve nothing).

COLD START: explore/exploit. new item=content emb. new user=onboarding+popularity+bandit.
```

---

## Section 12: Follow-up questions the interviewer may ask

- **What changes at 100M users?** Separate candidate generation from ranking into independent services, shard the ANN index (recall the ~51GB estimate), cache hot embeddings/candidates, async logging, define p95/p99 SLOs, and degrade gracefully under load.
- **How do you handle cold start?** Content embeddings for new items, onboarding + popularity priors for new users, and an exploration budget (epsilon-greedy / contextual bandit) — frame it as explore/exploit, not a feature hack.
- **How do you pick top-K at each stage?** By the latency budget at retrieval (how many candidates can the ranker score in time?) and by the business objective + guardrails at the final stage; tune with validation curves then confirm online.
- **Offline metrics improve but online drops — why?** Train-serve skew, biased offline labels echoing the old system, position bias, novelty effect, diversity collapse, proxy mismatch. Diff features, check label provenance, check guardrails.
- **How do you prevent feedback loops / rich-get-richer?** Exploration traffic, popularity down-weighting (log-Q), diversity reranking, monitor long-tail coverage, counterfactual/randomized logging for unbiased eval.
- **Why not one big model end-to-end?** Latency and indexability — retrieval needs representations that factorize so item vectors can be precomputed and ANN-searched; only the final stage can afford pairwise cross features.
- **How do you balance multiple objectives?** MMoE with per-objective heads; combine into a weighted score; tune weights via online experiments because the tradeoff is a business decision.
- **How do you correct position bias specifically?** Position-as-feature (shallow tower) at training, fixed at serving; optionally inverse-propensity weighting on logged data.

---

## Section 13: Common mistakes (anti-patterns to avoid)

- Jumping to a deep model before defining the business objective and the *label*.
- Naming four architectures shallowly instead of explaining one to the floor.
- Listing feature *nouns* without saying how they're engineered or which stage uses them.
- Ignoring the label-bias problem — the single most common "shallow" tell.
- Forgetting negative sampling entirely, or hand-waving it.
- Treating ranking as single-objective click prediction in 2024+ (no multi-task).
- Listing NDCG but never addressing the offline→online gap or designing a real A/B test.
- Treating deployment as "an endpoint" instead of serving/data/feedback paths.
- No non-ML baseline and no fallback path.
- Never stating a single number (QPS, latency split, storage) — the biggest "never operated a system" tell.

---

## Section 14: Transfer — one case unlocks five

This same skeleton, with different nouns, *is* the answer to:

| Problem | What changes | What stays identical |
|---|---|---|
| **Ad ranking** | label = P(click)×bid (eCPM); strict latency; auction layer | two-tower retrieval + ranker, bias correction, A/B |
| **Feed ranking (FB/LinkedIn)** | items = posts; multi-objective (meaningful interaction) | MMoE, funnel, position bias |
| **Search ranking** | strong query signal; relevance labels | retrieval+rank funnel, NDCG, listwise loss |
| **YouTube "watch next"** | item→item; session sequence matters | two-tower, MMoE multi-task (the canonical paper) |
| **"People you may know"** | graph features, social signals | retrieval + ranker, cold start, exploration |
| **E-commerce product rec** | inventory, price, availability constraints | exactly this case |

Master this one to the floor, and you walk into any of the above knowing 80% of the answer. That's the leverage: **the unit of study is "one case to the floor," then transfer — not breadth across shallow cases.**

---

## Sources
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research, Advances in TF-Ranking: https://research.google/blog/advances-in-tf-ranking/
- Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (Google, two-tower + log-Q): https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
- Recommending What Video to Watch Next (YouTube, MMoE + position-bias shallow tower): https://dl.acm.org/doi/10.1145/3240323.3240374
- Modeling Task Relationships in Multi-task Learning with MMoE (Google): https://dl.acm.org/doi/10.1145/3219819.3220007
- Wide & Deep Learning for Recommender Systems (Google): https://arxiv.org/abs/1606.07792
- Deep Neural Networks for YouTube Recommendations: https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/