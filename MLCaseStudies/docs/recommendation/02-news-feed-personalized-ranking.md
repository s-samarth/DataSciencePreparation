# 02. News Feed / Personalized Ranking System — ML System Design Case Study

**Company tags:** Meta, LinkedIn, X, TikTok, YouTube, Reddit
**Interview frequency:** Very high
**Why it matters:** Feed ranking is recsys where the *objective itself is the hard part*. Retrieval and ranking are the same funnel you already know; what makes feed its own case is that "good feed" is irreducibly **multi-objective** (likes vs comments vs shares vs dwell vs *hides* vs *reports*), the engagement signal you'd naively optimize is actively harmful (it rewards outrage and clickbait), and you must run an **integrity / quality-demotion** layer that *suppresses* highly-engaging content on purpose. If you can drive recsys to the floor, the funnel here is free. This document trains the part that isn't: the value model and integrity.

---

## How to use this document

This is not a checklist to memorize. It is built two ways at once:

1. **A thinking guide** — the reusable mental scaffold for *any* ML system design question, with the feed-specific content poured in.
2. **A worked one-hour interview transcript** — `INTERVIEWER:` / `YOU:` dialogue showing how the conversation unfolds under pressure.

Read the thinking guide first (Sections 0–8), then the transcript (Section 9). Internalize *the route, not the destination.*

> **The single most important habit:** every decision traces back to a number you stated up front (QPS, candidate pool, latency) or a constraint the interviewer gave you.

> **The feed-specific habit on top:** never say "optimize engagement" without immediately defining *which* engagement and naming its failure mode. "Optimize clicks" is the junior tell. "Engagement is a biased proxy that rewards clickbait and outrage, so I optimize a multi-objective value model weighted toward *meaningful* interactions and demote borderline content" is the senior answer. The value model and the integrity layer are the case.

---

## Section 0: The reusable scaffold (learn this ONCE)

```text
1. Clarify        -> turn an ambiguous product goal into a scoped problem + numbers
2. Frame as ML    -> what exactly are we predicting? what is the label? non-ML baseline?
3. Data & labels  -> where do labels come from, and WHY are they biased?
4. Baseline       -> simplest shippable thing, then name what breaks
5. Model          -> climb the ladder; explain ONE thing to the floor
6. Evaluation     -> offline metrics, online A/B, and the gap between them
7. Deploy         -> serving path / data path / feedback path (three separate things)
8. Monitor        -> drift, delayed labels, retraining, fallback, incident response
```

**The three-path mantra:** a production ML system is not "a model behind an endpoint." It's a **serving path** (request → candidates → rank → integrity → return; tight latency), a **data path** (logs → feature store → training data; batch), and a **feedback path** (what users did → labels → next training set; *where bias enters*). In feed, the feedback path is where the engagement-optimization trap lives.

---

## Section 1: Clarify requirements (and pin down NUMBERS)

### Interview prompt
> "Design the ranking system for a personalized news feed that shows posts from friends, creators, and recommended accounts."

### The clarifying questions that actually change the design

| Question | Why it changes the design |
|---|---|
| **What is the objective — raw engagement, long-term retention, or "time well spent"?** | This is *the* question. Optimizing raw engagement (clicks, time-spent) is known to reward clickbait and outrage and to hurt long-term retention. The objective defines the label and the entire integrity story. |
| **In-network only, or in-network + recommended out-of-network?** | A pure friends/follows feed is a ranking-only problem (the candidate set is small and given). Adding recommended accounts (TikTok-style) makes it a retrieval problem over a huge corpus, like recsys. |
| **How fresh must it be?** | Feed content decays in minutes-to-hours. This forces near-real-time feature pipelines and frequent re-ranking, unlike a movie catalog. |
| **How many candidates per request?** | In-network since last visit might be thousands of posts; add out-of-network recommendations and you're retrieving from millions. This sets whether you need a retrieval stage at all. |
| **Latency budget?** | A feed assembles on page load; p99 < ~200ms end-to-end, split across candidate gen, ranking, and the integrity/rerank pass. |
| **What are the integrity constraints?** | Misinformation, borderline content, engagement-bait. These aren't afterthoughts; they're a mandatory demotion layer that *reduces* engagement on purpose. Ask whether the platform has a policy line. |

> **Junior move:** "I'll predict click probability and rank by it."
> **Senior move:** "Before models — is the objective raw engagement or long-term value? Because if it's value, my *label* can't be clicks, and I'll need an integrity layer that demotes engaging-but-harmful content. That decision drives everything."

### Pin the numbers (carry these through the ENTIRE answer)

```text
Users (DAU):          500M
Feed sessions:        ~8 / active user / day
                      500M * 8 / 86400s  ~= 46,000 QPS average
                      peak ~3x            ~= 140,000 QPS
Candidate pool / req: ~10K eligible (in-network posts since last visit
                      + out-of-network recommended candidates)
Funnel:               10K --light ranker--> ~500 --heavy MMoE ranker--> ~50
                      --integrity + rerank--> ~top 20-30 shown per scroll
Latency SLO:          p99 < 200ms end-to-end for feed assembly
```

**The latency budget split** (derive out loud):
```text
200ms total, roughly:
  candidate gen (in-network fetch + out-of-network ANN)  ~40ms
  feature fetch                                           ~25ms
  light first-pass ranker (10K -> ~500)                   ~25ms
  heavy MMoE ranker (~500 candidates)                     ~60ms
  integrity classifiers + diversity rerank                ~30ms
  overhead / network                                      ~20ms
```
Now decisions have reasons: "I can't run the MMoE ranker on 10K candidates in budget, so a cheap first-pass ranker prunes to ~500; the expensive multi-task model runs on those."

**Storage back-of-envelope:** out-of-network corpus, say 100M candidate posts active in the recommendation window × 128-dim × 4B ≈ 51GB embedding index → shard or quantize the ANN index (same as recsys).

---

## Section 2: Frame it as an ML problem

- **Framing:** **multi-objective value-model ranking** over candidates from in-network + out-of-network sources. Say "multi-objective value model" explicitly.
- **Prediction target:** not a single label. A **vector of action probabilities** for each (viewer, post, context): P(like), P(comment), P(meaningful comment), P(reshare), P(dwell > t), P(hide), P(report), P("see fewer like this"). These are combined into one **value score**.
- **What is a positive?** This is the design choice that *defines the product*. Raw clicks/time-spent are easy but reward clickbait. **Meaningful interactions** (a comment with text, a reshare, a long dwell, especially *between people who know each other*) are a better proxy for long-term value. State the tradeoff explicitly: you are choosing a label that you *believe* correlates with retention even though it isn't the easiest to optimize.
- **Non-ML baseline:** reverse-chronological feed, or a hand-tuned weighted score (recency × friend-affinity). This is the launch fallback and the thing every "bring back chronological feed" debate is about — worth naming.

> **Why this framing matters (say it):** the product goal "a feed people value" is not learnable. Decomposing it into predicted *actions* and a *value function over those actions* makes it measurable, lets you encode policy in the weights, and — crucially — lets you argue about the objective: "we are not predicting clicks, we're predicting meaningful interaction, and clicks are a biased proxy we deliberately down-weight."

---

## Section 3: Data, labels, and the engagement-trap (the intellectual core)

Recsys's core was label *bias from your own logs*. Feed inherits that and adds a deeper problem: **the easy label is actively harmful.** This is the section that wins the case.

### Data sources
- **The viewer graph:** friends, follows, groups, prior interactions with this author. Social-tie strength is a top feature.
- **Engagement logs:** impressions, dwell, likes, comments (with text length), reshares, hides, reports, "see fewer," follows/unfollows.
- **Post content:** text, image/video embeddings, link, author, age (freshness), language.
- **Explicit surveys:** periodic "Is this post worth your time?" / "Do you want to see this?" prompts — a *less-biased* label source. Naming surveys is a strong feed-specific signal.
- **Integrity signals:** classifier scores for misinformation, clickbait, engagement-bait, borderline/near-policy-line content.

### Features — engineered, split by stage (latency)
- **Candidate gen (cheap):** social-graph edges (friends' recent posts), follow-based retrieval, plus a two-tower ANN for out-of-network recommendations.
- **Ranking (rich cross features):** viewer×author affinity ("how many of this author's posts did the viewer comment on in 30 days"), viewer×topic affinity, post freshness, author quality, real-time post velocity, and the **display-position feature** for bias correction.

### The two label problems you MUST confront

**Problem 1 — position & exposure bias (inherited).** Top-of-feed posts get more engagement because they're on top. Fix: position-as-feature at training, fixed at serve; plus exploration traffic for unbiased labels.

**Problem 2 — the engagement trap (the feed-specific one).** Optimizing the *easy* signal (clicks, time-spent, reactions) makes the product worse in ways the metric can't see: it rewards outrage, clickbait, and engagement-bait, and it erodes long-term retention and trust. This isn't hypothetical — it's the single most important real-world lesson in feed ranking.

- **The canonical event: Meta's 2018 "Meaningful Social Interactions" (MSI) relabeling.** They moved feed ranking away from passive consumption (raw time-spent / video views) toward *meaningful interactions* — comments, reshares, and especially exchanges between friends — and **publicly accepted a measured decrease in time-spent** as the price of a healthier product. The lesson to state: *they changed the label and shipped a deliberate short-term metric regression because the long-term objective differed from the easy one.* That's the most senior thing you can say in this interview.
- **The borderline-content phenomenon (also Meta, 2018):** engagement *rises* as content approaches the policy line (more sensational = more engaging). So a pure value model will *naturally* surface near-violating content. The fix is an explicit **demotion** layer that suppresses borderline content even though it's highly engaging. Name this; it's the reason an integrity layer exists at all.
- **Survey labels as the un-biased anchor:** because behavioral labels are biased and gameable, feed teams collect explicit "worth your time?" surveys and train a model to *predict the survey answer* from behavior, then use that as a value signal. This breaks the loop where the system optimizes its own biased clicks.

> **Junior says:** "I'll train on clicks and reactions."
> **Senior says:** "Behavioral labels are biased (position) and the *easy* ones are harmful (clickbait/outrage). I'd weight toward meaningful interactions, anchor with explicit 'worth your time' surveys, and demote borderline content — like Meta's 2018 MSI shift, where they accepted a time-spent drop for long-term value."

### Cold start
- **New user:** onboarding interests + popularity priors + rapid in-session adaptation; lean on in-network (their first follows) before out-of-network recs are confident.
- **New post/creator:** content embedding gives a vector with zero engagement history; exploration budget surfaces it to a controlled slice. Frame as explore/exploit.

---

## Section 4: Baseline first — then name exactly what breaks

```text
RUNG 0: Reverse-chronological / heuristic score
   -> "Show newest first, or recency x friend-affinity."
   BREAKS: drowns in volume; can't balance objectives; no quality control.
   TRIGGER: need per-user relevance and multi-signal balancing.

RUNG 1: GBDT / logistic ranker on engineered features, single objective
   -> predict P(engagement), rank by it.
   FIXES: personalized relevance, interpretable.
   BREAKS: single-objective -> optimizes clickbait; no sequence/content understanding;
           ignores hides/reports.
   TRIGGER: we need MULTIPLE objectives and to subtract negative actions.

RUNG 2: Multi-task DNN value model (MMoE) + integrity layer   <-- PRODUCTION DEFAULT
   -> heads for {like, comment, meaningful-comment, share, dwell, hide, report};
      combine into a value score; integrity classifiers demote borderline content.
   FIXES: multi-objective tradeoff, subtract harmful actions, quality control.
   BREAKS / costs: objective-weight tuning is a business decision; expensive;
      can still over-optimize an addictive proxy if weights are wrong.
   TRIGGER (if pushed): session/sequence dynamics, or huge out-of-network corpus.

RUNG 3: Sequence/transformer reranker + exploration bandit (advanced)
   -> attention over the session for final ordering; bandit allocates exploration
      to fresh/uncertain content.
   USE WHEN: session intent shifts fast (short-video feeds) and you can afford it.
   COST: latency, serving cost; needs distillation or a strict candidate cap.
```

> **Say this:** "I'd ship Rung 0 as launch baseline and permanent fallback. Production is Rung 2 — a multi-task MMoE value model with an integrity demotion layer — because feed is inherently multi-objective and the easy single objective is harmful. Rung 3 I'd reach for if session dynamics dominate, e.g. a short-video feed."

---

## Section 5: The value model + integrity — ONE design, to the floor

Depth on one thing beats naming four. The thing to explain to the floor in feed is **the multi-task value model and how its outputs combine into a ranking score, and how the integrity layer modifies that score.**

### 5.1 The multi-task ranker (MMoE)

```text
   viewer x post x context features  (incl. POSITION for bias correction)
                          |
        [ Shared expert subnetworks: E1 E2 ... Ek ]   <- MMoE
                          |
   per-task gates pick a weighted mix of experts for each task
                          |
   heads:  P(like) P(comment) P(meaningful-comment) P(share)
           P(dwell>t) P(hide) P(report) P(see-fewer)
```
- **Why MMoE (Multi-gate Mixture-of-Experts), not a single shared-bottom net:** the objectives *conflict* — clickbait drives clicks but kills meaningful comments; a single shared representation forces a compromise. MMoE gives each task a learned **gate** over shared experts, so conflicting tasks can draw on different experts. This is the standard production architecture (YouTube's "Recommending What Video to Watch Next" uses MMoE for exactly this reason). Name it.
- **Loss:** each head is a binary classifier with logloss (calibration helps when you combine probabilities). Trained jointly with summed per-task losses.
- **Position bias:** position is a training feature, fixed at serve.

### 5.2 Combining heads into one score (the value model)

```text
value = w_like * P(like)
      + w_comment * P(comment)
      + w_mc * P(meaningful-comment)     <- MSI weighting: text comments,
      + w_share * P(reshare)                friend-to-friend, weighted UP
      + w_dwell * P(dwell > t)
      - w_hide * P(hide)                  <- negative actions SUBTRACTED
      - w_report * P(report)
      - w_seefewer * P(see-fewer)
```
- **The weights encode product policy, and they are tuned via online A/B, not learned offline.** Say this — it's the most senior point in the section. The "right" tradeoff between comments and time-spent is a business/values decision, not a loss-minimization. Meta literally re-tuned these weights in the 2018 MSI change to up-weight meaningful comments and reshares.
- **Negative actions are first-class:** hides and reports are *subtracted*, so the model learns to avoid annoying content, not just chase clicks.

### 5.3 The integrity / quality layer (feed-specific, recsys doesn't have it)

After the value score, an integrity pass modifies ranking:
```text
final = value * quality_demotion_multiplier
        - hard_filter(policy_violating)        <- removed, not ranked
```
- **Hard removals:** content that violates policy (CSAM, incitement) is removed upstream, not ranked.
- **Soft demotions:** borderline/near-policy-line content, misinformation flagged by fact-checkers, clickbait, and engagement-bait get a **demotion multiplier** that pushes them down even though they're highly engaging. This is the explicit fix for the "engagement rises near the policy line" problem.
- **Diversity / dedup:** don't show 5 posts from one author or topic (MMR-style).

Saying "the ranked output isn't the raw value score — there's an integrity demotion layer that *suppresses engaging-but-harmful content on purpose*" is the line that separates someone who's thought about feed from someone who's only thought about recsys.

### 5.4 The full architecture diagram (draw this)

```text
                REQUEST (viewer_id, context)
                          |
   [ Candidate gen ]  in-network (graph fetch) + out-of-network (two-tower ANN)
                10K candidates
                          |
                 [ Feature store ]
                          |
            [ Light first-pass ranker ]  10K -> ~500
                          |
            [ MMoE multi-task value model ]  heads -> value score (~500)
                          |
            [ Integrity layer: demote borderline / misinfo / clickbait;
              remove policy violations ]
                          |
            [ Diversity / dedup rerank + exploration injection ]
                          |
                  Top 20-30 -> viewer
                          |
        (impressions, dwell, likes, comments, hides, reports, surveys logged)
                          |
   [ Feedback path -> labels (+ position-bias correction, survey labels) -> retrain ]
```

---

## Section 6: Evaluation — offline metrics and the engagement-trap gap

### Offline metrics
- **Per-head:** AUC / logloss / calibration for each action head (like, comment, hide...).
- **Ranking:** NDCG@K against a value-weighted relevance label.
- **Always slice by cohort:** new vs core users, content type, author size, locale. And keep a **bank of bad feeds** (full ranked lists that look wrong) — for feed, inspecting whole lists beats aggregate metrics.

### Online metrics (what decides launch)
- **Primary (value):** meaningful-interaction rate per session, and/or the predicted "worth your time" survey score. *Not* raw time-spent.
- **Health/guardrails:** hide+report rate, borderline-content exposure, content diversity, integrity prevalence, p95/p99 latency, and a **time-spent floor** (you accept some time-spent loss, but not a collapse).
- **Long-term:** retention (DAU/MAU), measured via long-term holdouts (below).

### The offline→online gap (THE talking point)

> **Classic question: "Offline NDCG and predicted-engagement went up, but long-term retention / survey scores dropped. What happened?"**

1. **Proxy mismatch / engagement trap** — you optimized a biased short-term proxy (clicks/dwell) that diverges from long-term value. *This is the feed-specific #1 cause and the answer they want.*
2. **Position bias not corrected** — offline metric computed on biased positions.
3. **Diversity collapse / filter bubble** — the "more accurate" model narrows the feed; engagement per item rises but the experience degrades.
4. **Train-serve skew** — features computed differently offline vs online.
5. **Novelty effect** — a new ranking gets a temporary engagement bump that fades.

**The lesson (state it):** in feed, the *metric you optimize is itself a hypothesis about long-term value*. Offline metrics and even short-term online engagement can rise while the product gets worse. This is exactly why Meta moved to MSI and to long-term holdouts.

### A concrete A/B test (fully specified)

```text
Hypothesis:        The new MMoE value model + MSI weighting raises meaningful
                   interactions and 'worth your time' survey scores without
                   collapsing time-spent or raising hide/report rates.
Unit of randomization: user (consistent experience; avoids network contamination
                   as much as possible -- see note).
Control:           current production ranker.
Treatment:         new MMoE value model + integrity layer.
Primary metric:    meaningful interactions per session (or survey value score).
Guardrails:        hide+report rate, borderline-content exposure, diversity index,
                   time-spent floor (accept small loss, block a collapse),
                   p99 latency. Any breach = auto-rollback.
Ramp:              1% -> 5% -> 20% -> 50%, holding to check guardrails.
Sample size / MDE: power for ~1% relative lift in the primary metric; use CUPED
                   for variance reduction on engagement metrics.
Duration:          >= 2 weeks for weekly seasonality + novelty decay.
LONG-TERM HOLDOUT: keep a small % of users on the OLD model for weeks-to-months,
                   because retention and trust move SLOWLY and a 2-week A/B
                   cannot see them. This is essential for feed.
Decision/rollback: ship if primary + survey up and guardrails hold.
```

Mention **long-term holdouts** and **network interference** explicitly — they're the feed-specific A/B subtleties (social feedback means one user's treatment leaks to friends; long-term value needs months, not weeks).

---

## Section 7: Deployment & serving (the three paths)

### Serving path (latency-critical)
```text
request -> candidate gen (graph fetch + out-of-network ANN) -> feature store
       -> light ranker (10K->500) -> MMoE value model (500) -> integrity demotion
       -> diversity rerank -> top 20-30
```
- Candidate gen, ranking, integrity as **separate services** that scale/fail independently.
- Shard/quantize the out-of-network ANN index (the 51GB estimate).
- Cache viewer embeddings, social-graph edges, hot post features; async logging.

### Data path (batch + streaming)
```text
raw logs (engagement + surveys) -> ETL -> feature store -> training data
   -> train MMoE + integrity classifiers -> validate -> push
   -> LOG SERVED FEATURES + POSITION (kills skew, enables bias correction)
```
- Posts are fresh-decaying, so feature pipelines are near-real-time (post velocity, author counts).

### Feedback path (where the engagement trap lives)
```text
likes/comments/shares/hides/reports + SURVEY responses -> join with logged features
   -> position-bias correction -> next training set -> retrain
```
- The survey-label join is what keeps the loop from optimizing its own biased clicks.

### Rollout discipline
```text
shadow (score live, don't serve) -> canary -> A/B (ramp) -> long-term holdout
```

---

## Section 8: Monitoring, retraining, incident response

- **Monitor:** data/prediction drift, per-head calibration, **integrity prevalence** (how much borderline/misinfo is reaching feeds), diversity, hide/report rate, latency, and the slow-moving **retention / survey value** via holdouts. The model can look healthy on engagement while trust quietly erodes — watch surveys and reports, not just clicks.
- **Retraining:** frequent (content decays fast) — often daily or faster for the ranker; integrity classifiers retrained as adversaries adapt.
- **Fallback:** degrade to cached feed → friend-affinity heuristic → reverse-chronological. Never serve nothing; this is why Rung 0 stays in prod.
- **Incident response:** freeze the version, diff against last release, inspect a bank of bad feeds, check whether a weight change over-amplified a harmful proxy, roll back on guardrail breach.

---

## Section 9: The worked one-hour interview (full transcript)

---

**[00:00 — The prompt]**

**INTERVIEWER:** Design the ranking system for a personalized news feed — posts from friends, creators, and recommended accounts.

**YOU:** Before I design anything, a few scoping questions, because the answers change the architecture a lot. First and most important: what's the objective — raw engagement, or long-term value / "time well spent"? That changes my *label*, not just my tuning. Second, is this in-network only, or in-network plus recommended out-of-network accounts — that decides whether I need a retrieval stage over a huge corpus. Third, how fresh must it be. Fourth, candidate pool size and latency budget. And fifth, what are the integrity constraints — is there a policy line I have to enforce?

**INTERVIEWER:** Optimize long-term value, not just engagement. In-network plus recommended. Fresh — content is hours old. Assume 500M DAU, and yes, there's an integrity policy. Latency, say 200ms.

**YOU:** Good, let me pin numbers.
```
500M DAU, ~8 sessions/day -> ~46K QPS avg, ~140K peak
candidate pool ~10K/request, p99 < 200ms
funnel: 10K --light ranker--> 500 --MMoE--> 50 --integrity+rerank--> top 20-30
```
Two things follow immediately. The 200ms over 10K candidates forces a funnel — a cheap first-pass ranker prunes to ~500, then an expensive multi-task model runs on those. And "long-term value, not engagement" tells me my label can't be clicks or time-spent, because those reward clickbait and outrage. I'll come back to that — it's the core of this design.

**INTERVIEWER:** Go on.

---

**[00:06 — Framing & the label]**

**YOU:** I'll frame it as a multi-objective value model. For each viewer-post-context, I predict a *vector* of action probabilities — like, comment, meaningful comment, reshare, long dwell, and the negatives, hide and report — and combine them into a value score. My non-ML baseline is reverse-chronological or recency-times-affinity, which is also my fallback. The key decision is the positive label: I'd weight toward *meaningful* interactions — a text comment, a reshare, especially between people who know each other — not raw clicks.

**INTERVIEWER:** Why not just optimize engagement? It's abundant and it's what the business cares about.

**YOU:** Because the easy engagement signal is actively harmful here, and this is the most important lesson in feed ranking. Optimizing clicks and time-spent rewards clickbait and outrage and erodes long-term retention and trust — the metric goes up while the product gets worse. The canonical example is Meta's 2018 Meaningful Social Interactions change: they moved ranking toward comments and reshares between friends and *publicly accepted a drop in time-spent* because long-term value differed from the easy proxy. So I optimize a value model weighted toward meaningful interactions, and I anchor it with explicit "is this worth your time" surveys, because surveys are a less-biased label than behavior.

---

**[00:13 — Data & the engagement trap]**

**INTERVIEWER:** Tell me about your data and labels.

**YOU:** Sources: the viewer graph, engagement logs with comment text length and negatives like hide and report, post content embeddings, freshness, and explicit surveys. Two label problems. First, position bias — top posts get engaged with because they're on top — fixed with the position-as-feature trick, feeding position at training and fixing it at serving. Second, and feed-specific, the engagement trap: behavioral labels are biased, and the *easy* ones are harmful. There's also the borderline-content phenomenon — engagement actually rises as content approaches the policy line, because sensational content is engaging. So a pure value model naturally surfaces near-violating content, which is exactly why I need an explicit integrity demotion layer, not just a ranker.

**INTERVIEWER:** That's the part most people miss. Let's get into the model.

---

**[00:20 — The value model, to the floor]**

**YOU:** I'll climb a quick ladder: reverse-chron, then a single-objective GBDT ranker — which I reject because single-objective optimizes clickbait and ignores hides — then my production answer, a multi-task MMoE value model plus an integrity layer.

*(drawing)* The ranker is multi-task: shared expert subnetworks, and a per-task gate for each head — like, comment, meaningful comment, share, dwell, hide, report. I use MMoE specifically because the objectives *conflict*: clickbait drives clicks but kills meaningful comments, and a single shared-bottom network would force a bad compromise. The gates let conflicting tasks draw on different experts. Each head is a logloss classifier, trained jointly. Then I combine them into a value score: a weighted sum of the positive heads *minus* the negative heads — hide and report are subtracted, so the model avoids annoying content, not just chases clicks.

**INTERVIEWER:** How do you set those weights?

**YOU:** Online, by A/B testing — not learned offline. The weights encode product policy: the tradeoff between, say, comments and time-spent is a values decision, not a loss-minimization. That's literally what Meta re-tuned in the MSI change. Learning them offline would just re-learn the biased status quo.

---

**[00:30 — Integrity layer]**

**INTERVIEWER:** Where does safety come in?

**YOU:** A dedicated integrity layer after the value score. Policy-violating content is removed upstream, not ranked. Then borderline content — near the policy line, misinformation flagged by fact-checkers, clickbait, engagement-bait — gets a *demotion multiplier* that pushes it down even though it's highly engaging. That's the explicit fix for the borderline phenomenon: I'm deliberately suppressing engaging-but-harmful content. On top, diversity and dedup so one author or topic doesn't dominate, and a little exploration injection for fresh content. The served list is the value score *modified by integrity*, not the raw model output.

---

**[00:38 — Evaluation & the gap]**

**INTERVIEWER:** How do you evaluate? And a scenario: offline NDCG and predicted engagement go up, but retention and survey scores drop. Why?

**YOU:** Offline I check per-head calibration and value-weighted NDCG, sliced by cohort, plus a bank of full bad feeds. But offline only filters what's worth testing. On your scenario — that's the engagement trap, the feed-specific number-one cause: I optimized a biased short-term proxy that diverges from long-term value. Other causes: position bias uncorrected, diversity collapse into a filter bubble, train-serve skew, or a novelty bump. I'd check whether my offline label is just echoing biased clicks, look at the diversity guardrail, and lean on survey scores over engagement.

**INTERVIEWER:** Design the A/B test.

**YOU:** Hypothesis: the new value model raises meaningful interactions and survey scores without collapsing time-spent or raising hide/report. Randomize by user, primary metric meaningful interactions per session, guardrails hide-plus-report rate, borderline exposure, diversity, a time-spent *floor* — I accept a small loss but block a collapse — and latency. Ramp 1-5-20-50 with CUPED for variance reduction, run at least two weeks for seasonality and novelty. And critically for feed, a long-term holdout: keep a small slice on the old model for months, because retention and trust move too slowly for a two-week test to see. I'd also flag network interference — one user's treatment leaks to their friends through social actions.

---

**[00:48 — Serving, scale, monitoring]**

**INTERVIEWER:** What changes at 500M users on serving, and what do you monitor?

**YOU:** Three paths. Serving: candidate gen as graph fetch plus an out-of-network two-tower ANN — which I shard or quantize since 100M post embeddings is about 51GB — then light ranker, MMoE, integrity, rerank, as separate services. Cache viewer embeddings and graph edges, async logging. Data path: batch plus near-real-time for post velocity, and I log served features and position to kill skew and enable bias correction. Feedback path: engagement plus survey responses joined back with position-bias correction — the survey join is what stops the loop from optimizing its own biased clicks.

Monitoring: drift, per-head calibration, integrity prevalence, diversity, hide/report, latency, and the slow-moving retention and survey value via holdouts. The trap is that engagement can look healthy while trust erodes, so I watch surveys and reports, not just clicks. Retrain daily-ish since content decays fast; integrity classifiers retrained as adversaries adapt. Fallback degrades to cached feed, then affinity heuristic, then reverse-chron — never nothing.

---

**[00:55 — The close]**

**INTERVIEWER:** Anything to add?

**YOU:** To restate the core tradeoff: I optimize a multi-objective value model weighted toward meaningful interactions, protected by integrity demotion and guardrails on hide/report, diversity, and a time-spent floor. The thing that makes feed harder than recommendations is that the *objective itself* is the hard part — the easy label is harmful — which is why I lean on meaningful-interaction weighting, survey anchors, and a demotion layer, and why I'd use long-term holdouts to catch the slow damage. Concretely, I've shipped multi-objective ranking with this offline-online gap before, so the value-model tuning and the feedback-loop pitfalls are familiar; the integrity layer is the feed-specific addition.

**INTERVIEWER:** Strong answer.

---

> **Why this transcript works (study these moves):**
> 1. **Led with the objective question** — recognized the label, not the model, is the hard part.
> 2. **Named the engagement trap and the MSI precedent** — the real-world depth.
> 3. **Explained MMoE *and why* (conflicting objectives)** — not just the acronym.
> 4. **Subtracted negative actions** — hides/reports as first-class.
> 5. **Weights tuned online, not learned** — the senior point about policy in the objective.
> 6. **Integrity demotion as a deliberate engagement *reducer*** — the borderline-content fix.
> 7. **Survey labels as the un-biased anchor.**
> 8. **Long-term holdouts + network interference** — feed-specific A/B subtleties.

---

## Section 10: Junior vs Senior — the highest-leverage contrast

| Decision | Junior answer | Senior answer |
|---|---|---|
| Objective | "Maximize engagement." | "Multi-objective value model weighted toward *meaningful* interactions — the easy engagement signal is harmful." |
| The label | "Train on clicks/reactions." | "Behavioral labels are biased and the easy ones reward clickbait; anchor with 'worth your time' surveys (cf. Meta MSI 2018)." |
| Model | "A deep ranker." | "MMoE multi-task — objectives *conflict*, so per-task gates over shared experts; subtract hide/report." |
| Objective weights | "Learn them from data." | "Tune online via A/B — the tradeoff is a product/values decision, not a loss-minimization." |
| Safety | "Filter bad content." | "Hard-remove violations *and* demote borderline content — engagement rises near the policy line, so I suppress it on purpose." |
| Eval | "NDCG and engagement." | "Survey value + meaningful interactions; engagement can rise while trust erodes (the trap)." |
| A/B | "Split users, run a week." | "User-split + CUPED + 2 weeks + a months-long *holdout* for slow retention; mind network interference." |
| Deployment | "Deploy the ranker." | "Serving/data/feedback paths; the survey-label join in the feedback path breaks the bias loop." |

---

## Section 11: One-page cheat sheet (whiteboard recall)

```text
SCAFFOLD: Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor

NUMBERS:  DAU, sessions/day -> QPS ; candidate pool ~10K ; latency p99 ~200ms.
          funnel 10K --light ranker--> 500 --MMoE--> 50 --integrity+rerank--> 20-30.

OBJECTIVE (the core): NOT raw engagement. Multi-objective VALUE MODEL.
          easy label (clicks/time) is HARMFUL -> rewards clickbait/outrage.
          cf. Meta MSI 2018: re-weighted to meaningful interactions, accepted time-spent drop.

VALUE:    value = w_like*P(like)+w_mc*P(meaningful-comment)+w_share*P(share)+w_dwell*P(dwell)
                  - w_hide*P(hide) - w_report*P(report)
          WEIGHTS TUNED BY A/B (product policy), not learned offline.

MODEL:    MMoE multi-task (shared experts + per-task gates) BECAUSE objectives conflict.
          per-head logloss; position-as-feature, fixed at serve.

INTEGRITY (feed-only): remove policy violations; DEMOTE borderline/misinfo/clickbait
          (engagement rises near the policy line -> suppress on purpose); diversity rerank.

LABELS:   position bias -> position feature zeroed at serve.
          engagement trap -> meaningful-interaction weighting + SURVEY anchor labels.

EVAL:     per-head calibration + value NDCG, slice by cohort, bank of bad FEEDS.
          online: meaningful interactions + survey value (NOT raw time).
          offline-up/online-down #1 = engagement trap (biased proxy diverges from value).

A/B:      randomize by user, CUPED, 2 wks + LONG-TERM HOLDOUT (months) for retention.
          guardrails: hide/report, borderline exposure, diversity, time-spent FLOOR, latency.
          watch NETWORK INTERFERENCE.

3 PATHS:  serving (candgen->rank->integrity) | data (log served features+position)
          | feedback (engagement + SURVEYS join -> breaks bias loop)

MONITOR:  drift, calibration, INTEGRITY PREVALENCE, diversity, hide/report, slow retention.
          fallback: cached -> affinity heuristic -> reverse-chron (never nothing).
```

---

## Section 12: Follow-up questions the interviewer may ask

- **What changes at 500M users?** Separate candidate gen / ranking / integrity services, shard the out-of-network ANN index (recall ~51GB), cache viewer embeddings + graph edges, async logging, p95/p99 SLOs, graceful degradation.
- **How do you handle cold start?** New user: onboarding interests + popularity + in-network first; new post/creator: content embeddings + exploration budget (explore/exploit).
- **How do you pick the objective weights / top-K?** Weights tuned online via A/B because they encode policy; top-K by latency at the funnel and by guardrails at the final stage.
- **Offline up, online (retention/surveys) down — why?** The engagement trap (biased proxy), position bias, diversity collapse, train-serve skew, novelty. Check label provenance and the diversity guardrail.
- **How do you prevent feedback loops / filter bubbles?** Exploration traffic, diversity reranking, survey-anchored labels, monitor diversity + long-tail exposure, long-term holdouts.
- **How do you balance engagement vs integrity?** Integrity is a demotion layer with its own guardrails; you accept measured engagement loss (time-spent floor, not maximization) for trust and retention.
- **How do you correct position bias?** Position-as-feature at train, fixed at serve; exploration / IPW on logged data.

---

## Section 13: Common mistakes (anti-patterns to avoid)

- Saying "optimize engagement" without naming that the easy engagement signal is harmful (clickbait/outrage). The single biggest feed tell.
- Treating feed as single-objective click prediction — no multi-task, no subtraction of hide/report.
- Naming MMoE without explaining *why* (conflicting objectives need per-task gates).
- Forgetting the integrity / borderline-demotion layer entirely.
- Learning the objective weights offline instead of tuning them via A/B (re-learns the biased status quo).
- Evaluating on raw time-spent / engagement instead of survey value + meaningful interactions.
- Running only a 1-2 week A/B and missing slow retention damage (no long-term holdout).
- Listing feature nouns; ignoring position bias and the survey-label anchor.
- Treating deployment as "an endpoint" instead of serving/data/feedback paths.

---

## Section 14: Transfer — what mastering feed unlocks

| Problem | What changes vs feed | What stays identical |
|---|---|---|
| **Recommendation (case 01)** | usually single richer objective; no integrity-demotion layer | retrieve→rank funnel, position bias, A/B, MMoE |
| **Ads ranking (case 11)** | money + auction + calibration + pacing | MMoE multi-task, position bias, A/B |
| **Search ranking (case 03)** | strong query signal; relevance labels; listwise | funnel, NDCG, offline-online gap |
| **Short-video feed (TikTok)** | session dynamics dominate -> sequence reranker (Rung 3) | value model, integrity, exploration |
| **Content moderation (case 10)** | the integrity classifiers *are* the product | borderline demotion, precision/recall at threshold |
| **Notification ranking (case 19)** | volume/fatigue control, explore/exploit | multi-objective value, long-term holdouts |

The leverage: **the funnel is shared with recsys; the genuinely new ideas are the multi-objective value model and the integrity-demotion layer.** Learn those here and ads, short-video, and moderation all reuse them.

---

## Sources
- [IGotAnOffer ML System Design Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [Exponent ML System Design Interview Guide](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)
- [Hello Interview ML/System Design Learning](https://www.hellointerview.com/learn)
- [Designing Machine Learning Systems, Chip Huyen](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [Google Research, Advances in TF-Ranking](https://research.google/blog/advances-in-tf-ranking/)
- [Modeling Task Relationships in Multi-task Learning with MMoE (Google)](https://dl.acm.org/doi/10.1145/3219819.3220007)
- [Recommending What Video to Watch Next (YouTube, MMoE multi-task + position bias)](https://dl.acm.org/doi/10.1145/3240323.3240374)
- [Meta, "Bringing People Closer Together" — Meaningful Social Interactions News Feed change (2018)](https://about.fb.com/news/2018/04/inside-feed-meaningful-interactions/)
- [Meta Transparency Center, Types of content we demote (borderline / integrity demotion)](https://transparency.meta.com/features/approach-to-ranking/types-of-content-we-demote/)
- [Meta Transparency Center, Our Approach to Feed Ranking](https://transparency.meta.com/features/ranking-and-content/)
