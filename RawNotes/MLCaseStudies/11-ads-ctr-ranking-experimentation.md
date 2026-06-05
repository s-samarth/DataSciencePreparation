# 11. Ads CTR / Ranking / Experimentation System — ML System Design Case Study

**Company tags:** Google, Meta, Amazon Ads, LinkedIn, Uber, Pinterest, TikTok
**Interview frequency:** High
**Why it matters:** Ads is recsys with money attached, and the money changes the math. The same retrieval-then-rank funnel you learned for recommendations is here, but two things make ads its own case: the prediction must be **calibrated** (a real dollar bid is multiplied by your predicted probability, so a 20% over-prediction is a 20% revenue distortion, not just a worse ordering), and the ranking sits inside an **auction** with **budget pacing** and **delayed conversions**. If you can drive recsys to the floor, you already have 70% of this. The remaining 30% — calibration, the auction, pacing, delayed labels — is what this document trains.

---

## How to use this document

This is not a checklist to memorize. It is built two ways at once:

1. **A thinking guide** — the reusable mental scaffold for *any* ML system design question, with the ads-specific content poured in.
2. **A worked one-hour interview transcript** — `INTERVIEWER:` / `YOU:` dialogue showing how the conversation unfolds under pressure, where the interviewer interrupts, and how a strong candidate answers in tradeoff language.

Read the thinking guide first (Sections 0–8), then the transcript (Section 9), and notice how every line maps back to the scaffold. Internalize *the route, not the destination.*

> **The single most important habit:** every decision traces back to a number you stated up front (QPS, latency, budget) or a constraint the interviewer gave you. "I'd use a DNN for CTR" is junior. "At 80ms over ~10K eligible ads I can score with a DNN, but the output must be *calibrated* because eCPM = pCTR × bid feeds a second-price auction — miscalibration leaks revenue and breaks pricing" is senior. The reason is the content.

> **The ads-specific habit on top of that:** in recsys you only need the *ordering* to be right. In ads you need the *probability* to be right, because it gets multiplied by money. Say the word "calibration" early and mean it.

---

## Section 0: The reusable scaffold (learn this ONCE)

Every ML system design answer is this skeleton. You are not learning 15 case studies — you are learning **one structure** and 15 sets of fill-ins.

```text
1. Clarify        -> turn an ambiguous product goal into a scoped problem + numbers
2. Frame as ML    -> what exactly are we predicting? what is the label? non-ML baseline?
3. Data & labels  -> where do labels come from, and WHY are they biased/delayed?
4. Baseline       -> simplest shippable thing, then name what breaks
5. Model          -> climb the ladder; explain ONE thing to the floor
6. Evaluation     -> offline metrics, online A/B, and the gap between them
7. Deploy         -> serving path / data path / feedback path (three separate things)
8. Monitor        -> drift, delayed labels, calibration decay, retraining, fallback
```

**The three-path mantra (say this during deployment):** a production ML system is not "a model behind an endpoint." It is three independent paths — **serving path** (request → features → retrieve → score → auction → return; tight latency), **data path** (logs → feature store → training data; offline, batch), and **feedback path** (clicks/conversions → labels → next training set; *where bias and delay enter*). In ads the feedback path is unusually hard because conversions arrive late.

---

## Section 1: Clarify requirements (and pin down NUMBERS)

The prompt is vague on purpose. Scope it and extract numbers before drawing anything.

### Interview prompt
> "Design an ad click prediction and ranking system for a marketplace or feed."

### The clarifying questions that actually change the design

| Question | Why it changes the design |
|---|---|
| **What are we billing on — CPC, CPM, or CPA?** | This sets the prediction target. CPC bills per click → you need **pCTR**. CPA bills per conversion → you need **pCTR × pCVR** and you inherit the **delayed-conversion** problem. CPM bills per impression → prediction matters less, targeting/pacing matter more. |
| **What's the ad surface?** Search ads, feed ads, or a banner slot? | Search ads have strong query intent (keyword match dominates). Feed ads compete with organic content and need a blended ranking. Different retrieval, different auction integration. |
| **One ad slot or many?** | A single slot is a top-1 auction. Multiple slots (a feed with ads every N posts) needs **slot allocation** and **position-bias** handling across slots. |
| **Whose objective — advertiser, user, or platform?** | This is a three-sided market. Pure revenue maximization burns user experience (ad fatigue) and advertiser ROI (clicks that don't convert). The objective is a **constrained** one: maximize platform revenue *subject to* user-experience and advertiser-ROI guardrails. |
| **How many eligible ads per request, and what's the latency budget?** | This forces the funnel. Millions of active ads, ~10K survive targeting, and an ~80ms budget means you can't run a heavy model on all of them. |
| **Is there a budget/pacing constraint?** | Advertisers set daily budgets. You cannot spend a day's budget in the first hour. Pacing is a control problem layered on top of ranking, and most candidates forget it entirely. |

> **Junior move:** "I'll predict CTR with a deep model."
> **Senior move:** "Are we billing CPC or CPA? That decides whether I predict pCTR alone or pCTR × pCVR with delayed conversions. And what's my latency budget over how many eligible ads — that decides the funnel."

### Pin the numbers (carry these through the ENTIRE answer)

```text
Users (DAU):          200M
Ad opportunities:     ~30 monetizable slots / active user / day
                      200M * 30 / 86400s  ~= 70,000 QPS average
                      peak ~3x average     ~= 210,000 QPS
Active ads:           ~10M (campaign x creative) eligible in the system
Per-request funnel:   10M ads --targeting--> ~10K --rank/score--> ~hundreds
                      --auction--> top 1-3 slots
Latency SLO:          p99 < 80ms for the ad decision (it sits INSIDE the
                      page-render budget, so it's tighter than organic feed)
```

**The latency budget split** (derive it out loud):
```text
80ms total budget, roughly:
  targeting / retrieval (10M -> ~10K)   ~15ms  (inverted index + match rules)
  feature fetch                          ~15ms  (feature store, cached aggregates)
  pCTR / pCVR scoring (~10K candidates)  ~30ms  (the model; richest stage)
  auction + pacing + allocation          ~10ms  (cheap arithmetic, but on the hot path)
  overhead / network                     ~10ms
```
Now every decision has a reason: "I can't run a heavy cross-feature model on 10M ads in 30ms, so targeting prunes to ~10K cheaply; the scored survivors go to the auction."

**Storage / scale back-of-envelope (do this reflexively):**
```text
10M ads * (ad embedding 64d * 4B + dense feats) ~ a few GB -> fits in RAM, shard for QPS
Logged events: 70K QPS * 86400 ~= 6B ad events/day. At ~1KB/event ~ 6TB/day raw logs
   -> you do NOT train on all of it raw; you down-sample negatives (see Section 3)
```

---

## Section 2: Frame it as an ML problem

The product goal ("show good ads, make money") is not learnable. Convert it into a precise target.

- **Framing:** **calibrated probability prediction feeding a utility auction.** Say all of that. "Calibrated" and "auction" are the two words that signal you know this isn't plain recsys.
- **Prediction target(s):**
  - **pCTR** = P(click | impression, user, ad, context).
  - **pCVR** = P(conversion | click, ...) if billing CPA.
  - The **expected value** of showing an ad is `eCPM = pCTR × pCVR × value` (for CPA) or `eCPM = pCTR × bid` (for CPC).
- **What is a positive label?**
  - Click for pCTR: cheap, abundant, but biased by position and noisy (fat-finger, bot clicks).
  - Conversion for pCVR: sparse, *delayed* (arrives hours to days later), defined by an **attribution window** (e.g., 7-day click / 1-day view). The window definition is a design choice with real consequences.
- **Non-ML baseline (always name one):** targeting rules + bid-sorted ads filtered by category, frequency cap, and policy. This is your launch-day fallback and your safety net when the model misbehaves.

> **Why this framing matters (say it):** separating "make money" from "predict a calibrated click/conversion probability" gives you a measurable label, lets you stage complexity, and — critically — makes the auction a clean layer on top: the model outputs a probability, the auction turns probabilities + bids into an allocation and a price. Keeping those two concerns separate is the whole architecture.

---

## Section 3: Data, labels, and the bias/delay problem (the intellectual core)

Recsys's core was label *bias*. Ads inherits that and adds label *delay* and the need for *calibration*. This section is where the case is won.

### Data sources
- **Auction & impression logs (the gold):** every request, the eligible set, what was shown, at what position, the winning/losing bids, the price paid.
- **Engagement logs:** clicks, dwell, hides, "why am I seeing this," conversions (via pixel / SDK / server-side postback).
- **Ad metadata:** creative (text/image/video), advertiser, category, landing-page quality, historical CTR.
- **User/context:** intent signals, query (for search ads), session, device, recent behavior, frequency of seeing this ad.

### Features — as engineered objects, split by stage (latency reason)

**Targeting / retrieval features (cheap, precomputable):** ad keyword/category match, geo/audience targeting predicates, coarse user-segment membership. These run as an inverted-index lookup, not a model.

**Scoring features (rich, on ~10K candidates):**
- **User × ad cross features:** "user's historical CTR on this advertiser's category," "times this user saw this exact creative this week" (frequency), "query × ad-keyword semantic match."
- **Ad-side:** creative embedding, advertiser quality score, landing-page quality, ad age.
- **Real-time:** ad's CTR in the last hour, campaign pacing state, inventory.
- **The position feature** (for bias correction — see below).

### The two label problems you MUST confront

**Problem 1 — position & exposure bias (inherited from recsys).** Top ad slots get more clicks because they're on top. Train naively and you learn "ads we already rank high are good." Fix with the **position-as-feature** trick: feed display position as an input at training time, set it to a fixed value at serving time so the model predicts click probability position-neutrally. Plus exploration traffic for unbiased labels.

**Problem 2 — delayed conversions (the ads-specific one).** This is the part most candidates have never seen. A click happens now; the conversion (purchase, signup) may land **days later** inside the attribution window. So at training time, an example labeled "no conversion" might just be a conversion that *hasn't arrived yet* — a censored positive, not a true negative.

- **Naive failure:** wait for the full attribution window before training → your model is always days stale, which is fatal for fast-moving campaigns.
- **The fix (Chapelle's delayed-feedback model, Criteo 2014):** model the **delay distribution** jointly with the conversion probability. Treat unconverted clicks as *censored* — the model estimates P(will eventually convert) and P(delay), so a recent unconverted click is appropriately discounted rather than hard-labeled negative. Modern variants: **fake-negative weighting / importance weighting** — ingest clicks immediately as negatives, then when a conversion arrives, insert a positive and correct the earlier negative's weight. Name the paper; it's a strong signal.
- **Attribution-window definition** is itself a lever: a 1-day window undercounts slow conversions; a 30-day window is fresher-blind. State that you'd pick the window with the business and treat it as a fixed assumption.

### The calibration problem (the reason ads ≠ recsys)

In recsys, if the *ordering* is right, you ship. In ads, the **absolute probability** is multiplied by a dollar bid and fed to an auction that sets prices. If your pCTR is systematically 20% too high, you overcharge advertisers (CPC pricing uses pCTR), distort the auction, and torch advertiser ROI and trust. So:

- **Train for ranking AND calibrate the output.** Standard loop: train pCTR (logloss is already a proper scoring rule, so a well-trained logloss model is roughly calibrated), then apply a **calibration layer** — **Platt scaling** (a logistic fit on a held-out set) or **isotonic regression** (non-parametric, monotonic) — to map raw scores to true rates.
- **Calibrate per segment.** Aggregate calibration can be perfect while *every* segment is miscalibrated in opposite directions. Calibrate (or at least monitor) by ad vertical, device, position, and new-vs-old ad.
- **Calibration drifts.** Click rates shift with seasonality, new inventory, and competitive dynamics, so calibration is re-fit frequently online, not once. Monitoring **expected calibration error (ECE)** in production is mandatory.

> **Junior says:** "I'll train a CTR model on clicks."
> **Senior says:** "Clicks are position-biased — I'll use the position-as-feature trick. Conversions are delayed — I'll use a delayed-feedback model so recent non-conversions are treated as censored, not negative. And because eCPM = pCTR × bid feeds a second-price auction, the output must be *calibrated*, so I add an isotonic/Platt layer and monitor ECE by segment."

### Cold start (ads-flavored)
New ad with no history → lean on **creative embedding** + advertiser-level priors (the advertiser's other ads' CTR) + an **exploration budget** so it gets enough impressions to estimate its true CTR. Frame as explore/exploit: a contextual bandit allocates exploration so new ads aren't starved by incumbents (the rich-get-richer trap that kills advertiser onboarding).

---

## Section 4: Baseline first — then name exactly what breaks

Climb the ladder; name the **trigger** for each rung.

```text
RUNG 0: Targeting rules + bid sort
   -> "Match by keyword/audience, sort by bid, apply policy + frequency cap."
   BREAKS: ignores relevance/quality; high bid != high value; bad user experience.
   TRIGGER: we need to predict click probability so we can rank by expected value.

RUNG 1: Logistic regression on crossed features (+ FTRL online learning)
   -> Wide sparse model; well-calibrated; trains online cheaply at huge scale.
      (Google's "Ad Click Prediction: a View from the Trenches" lives here.)
   FIXES: calibrated pCTR, billions of sparse features, online updates.
   BREAKS: misses nonlinear feature interactions unless you hand-cross everything.
   TRIGGER: feature-cross engineering becomes the bottleneck.

RUNG 2: GBDT feature transforms + LR, or Wide & Deep / DCN   <-- PRODUCTION DEFAULT
   -> GBDT+LR (Facebook 2014): trees learn crosses, LR stays calibrated & online.
      Or Wide&Deep / DCN-v2: deep part learns interactions, wide part memorizes.
   FIXES: nonlinear crosses + calibration + scale together.
   BREAKS / costs: feature freshness, calibration drift, training-serving skew.
   TRIGGER (only if pushed): need multi-objective (CTR + CVR + quality) or
      sequence/session modeling.

RUNG 3: Multi-task deep ranker + auction-aware utility (advanced)
   -> Shared-bottom / MMoE heads for pCTR, pCVR, quality; outputs feed eCPM.
   USE WHEN: CPA billing (need pCVR) and multiple objectives must be balanced.
   COST: harder to calibrate, harder to debug (ML x auction interaction).
```

> **Say this:** "I'd ship Rung 0 day one as launch baseline and permanent fallback. I'd run Rung 1 or 2 as production because logistic-regression-with-crosses (or GBDT+LR) is calibrated, trains online at billions of features, and is debuggable — calibration matters more here than squeezing the last AUC point. I'd only reach Rung 3 when CPA billing forces a pCVR head."

This is a key ads insight: the *fanciest* model is often **not** the right answer, because calibration and online-update stability beat raw AUC when the output feeds money.

---

## Section 5: Model + auction — ONE design, to the floor

Depth on one thing beats naming four. The thing to explain to the floor in ads is **how the model output flows into the auction and pricing**, because that's the part recsys doesn't have.

### 5.1 The scoring model (calibrated pCTR / pCVR)

```text
   features (user x ad cross, ad, context, POSITION)
                     |
        [ Wide part: sparse crossed features (memorization) ]
        [ Deep part: embeddings + MLP (generalization)      ]   <- Wide & Deep / DCN-v2
                     |
            raw score  ->  [ Calibration layer: isotonic / Platt ]
                     |
            calibrated pCTR  (and pCVR head if CPA)
```
- **Loss:** pointwise **logloss** (binary cross-entropy). Logloss is a *proper scoring rule*, so minimizing it pushes toward calibrated probabilities — exactly what the auction needs. (Contrast recsys, where pairwise/listwise ranking losses are fine because only order matters; here calibration wins.)
- **Negatives:** impressions without clicks are negatives, but there are billions, so **down-sample negatives** for training and **correct the bias** the sampling introduces (re-scale the intercept / use the known sampling rate) so calibration is preserved. State this explicitly; sampling without correction silently breaks calibration.
- **Position handling:** position is a training feature, fixed at serve.

### 5.2 The auction — the part that makes this "ads" (explain this to the floor)

The model produces a probability. The **auction** turns {probability, bid} pairs into an allocation and a price.

```text
For each eligible ad i:
   eCPM_i = pCTR_i * bid_i              (CPC billing)
          = pCTR_i * pCVR_i * value_i   (CPA billing)
   (often x quality_i to protect user experience: eCPM_i = pCTR_i * bid_i * quality_i)

Rank ads by eCPM. Winner = argmax eCPM.

Pricing (Generalized Second Price / VCG idea):
   the winner pays the MINIMUM bid it would have needed to keep its slot,
   i.e. price_1 = eCPM_2 / pCTR_1   (so the winner pays based on the
   runner-up's expected value, not its own bid).
```

**Why second-price / GSP matters (say it):** charging the winner its own bid incentivizes advertisers to shade bids downward and game the system. A second-price rule makes **truthful bidding** closer to optimal, which stabilizes the marketplace. This is mechanism design, and it's why "just sort by bid" (Rung 0) is wrong: value, not bid, should win, and price should be set by competition.

**Why calibration is now load-bearing:** the price `eCPM_2 / pCTR_1` literally divides by your predicted pCTR. If pCTR_1 is inflated, the advertiser is *undercharged* (revenue leak) or the allocation is wrong. Miscalibration is not a ranking nuisance here; it's a direct dollar error. This single sentence is the strongest ads-specific signal you can give.

### 5.3 Budget pacing (the constraint layer most candidates forget)

Each campaign has a **daily budget**. Without pacing, a high-eCPM campaign wins every auction at midnight and exhausts its budget in an hour, leaving the rest of the day un-monetized for that advertiser and starving the auction of competition.

- **Pacing as control theory:** treat it as a **PID / feedback controller** per campaign. Target a smooth spend curve (often proportional to traffic over the day). If a campaign is spending too fast, throttle its **participation probability** or apply a **bid multiplier (pacing factor)** that scales its effective bid down; if too slow, scale up.
- **Why a multiplier and not a hard stop:** hard on/off creates oscillation and misses cheap impressions late in the day. A smooth multiplier keeps the campaign competitive while hitting the budget.
- Say: "Pacing is a per-campaign closed-loop controller adjusting a bid multiplier so spend tracks a target curve — it's a constraint on top of the auction, not part of the model."

### 5.4 The full architecture diagram (draw this)

```text
                 AD REQUEST (user_id, context, query?)
                              |
                  [ Targeting / Retrieval: inverted index ]
                       10M ads -> ~10K eligible
                              |
                     [ Feature Store ]  (online + offline)
                              |
              [ Scoring model: Wide&Deep/DCN -> calibration layer ]
                       pCTR (x pCVR) per candidate
                              |
              [ Pacing controller ]  -> bid multipliers per campaign
                              |
              [ AUCTION: eCPM = pCTR x bid x quality; rank; GSP pricing ]
                              |
              [ Allocation + policy + frequency cap + slot assignment ]
                              |
                    Top 1-3 ads -> shown to user
                              |
              (impressions, position, prices, clicks logged)
                              |
        [ Feedback path: clicks now, conversions LATER (delayed-feedback model) ]
                              |
                     next training set + recalibration
```

---

## Section 6: Evaluation — offline metrics, calibration, and the online gap

### Offline metrics
- **Ranking quality:** **AUC** (does the model order clicked > non-clicked?). AUC is rank-only and **ignores calibration**, so it is necessary but not sufficient in ads.
- **Calibration:** **logloss** (rewards both discrimination and calibration), **Expected Calibration Error (ECE)**, and the simple **predicted-CTR / actual-CTR ratio** (should be ≈ 1.0) sliced by segment. Report these *alongside* AUC always.
- **Normalized entropy (NE)** — from the Facebook paper — logloss normalized by the entropy of the background CTR; lets you compare across datasets with different base rates.
- **Always slice:** by vertical, device, position, new-vs-old ad, advertiser size. Aggregate calibration hides per-segment disasters.

### Online metrics (what decides launch)
- **Platform:** revenue, RPM (revenue per mille), auction yield.
- **Advertiser:** ROAS / ROI, conversion rate, advertiser retention, fill rate for small advertisers (marketplace health).
- **User guardrails (must not regress):** ad hide rate, "see fewer ads like this," ad load, session length, p99 latency, organic-content health.

### The offline→online gap (THE evaluation talking point)

> **Classic question: "Offline AUC improved but online revenue dropped. Why?"**

1. **AUC improved but calibration got worse** — better ordering, but pCTR now biased high, so the auction misprices and the bid landscape shifts. *This is the #1 ads-specific cause and the answer interviewers want.*
2. **Train-serve skew** — a feature computed differently offline vs online.
3. **Delayed-conversion mislabeling** — offline you scored CVR on a window where late conversions weren't yet counted.
4. **Auction feedback loop** — a new model changes which ads win, which changes the logged data, which the offline metric (trained on old logs) can't see.
5. **Novelty / advertiser reaction** — advertisers re-tune bids in response, so the first-week numbers aren't the steady state.

**The lesson:** offline AUC filters what to test; online A/B with revenue + advertiser-ROI + user guardrails decides. And in ads you *always* check calibration offline before trusting an AUC win.

### A concrete A/B test (fully specified)

```text
Hypothesis:        The new DCN-v2 pCTR model raises revenue per session
                   without hurting advertiser ROI or ad hide rate, AND
                   stays calibrated (predicted/actual CTR within +-2%).
Unit of randomization: user (consistent experience, avoids contamination).
                   NOTE: ads have a market-interference subtlety -- user-split
                   leaks because advertisers/budgets are shared across arms.
                   For budget-sensitive changes, use BUDGET-SPLIT / geo-split
                   experiments so each arm has its own budget pool.
Control:           current production model + auction.
Treatment:         new model + same auction.
Primary metric:    revenue per session (or RPM).
Guardrails:        advertiser ROAS, ad hide rate, p99 latency, calibration
                   (predicted/actual CTR ratio), small-advertiser fill rate.
                   Any breach beyond threshold = auto-rollback.
Ramp:              1% -> 5% -> 20% -> 50%, holding to check guardrails + calibration.
Sample size / MDE: power to detect ~0.5-1% relative revenue lift at 95%/80%;
                   revenue is high-variance and heavy-tailed, so size generously
                   and consider variance-reduction (CUPED).
Duration:          >= 1-2 full weeks for weekly seasonality + let advertisers
                   re-equilibrate their bids (the market reaction effect).
Decision/rollback: ship if revenue up, no guardrail/calibration regression.
```

Mention **market interference / budget-split** explicitly — it's the ads-specific A/B trap (you can't cleanly user-split when arms share an advertiser's finite budget) and it separates people who've run ads experiments from those who haven't.

---

## Section 7: Deployment & serving (the three paths, concretely)

### Serving path (online, latency-critical)
```text
request -> targeting (inverted index, 10M->10K) -> feature store lookup
       -> scoring model (calibrated pCTR/pCVR) -> pacing multipliers
       -> auction (eCPM rank + GSP price) -> policy/freq-cap -> top 1-3
```
- Targeting, scoring, auction as **separate services** so they scale and fail independently.
- Cache user features, ad embeddings, hourly ad-CTR aggregates.
- The auction itself is cheap arithmetic but on the hot path — keep it in-process after scoring.

### Data path (offline, batch + streaming)
```text
raw auction/impression/click/conversion logs -> ETL -> feature store
   -> down-sample negatives (with correction) -> train -> validate -> push model
   -> fit calibration layer on held-out recent data
```
- **Log the exact features and the served position** so you train on what was served (kills skew and enables position-bias correction).
- **Conversions stream in late** — the data path must support **label backfill / late-binding joins** within the attribution window.

### Feedback path (closes the loop — where bias AND delay live)
```text
clicks (immediate) + conversions (delayed, days) -> attribution join
   -> delayed-feedback correction -> next training set
   -> frequent recalibration (calibration drifts fast)
```

### Rollout discipline
```text
shadow (score live traffic, don't serve) -> canary (small %)
   -> A/B (budget-split for budget-sensitive changes) -> ramp
```
Shadow mode catches calibration regressions and latency before a single advertiser is charged based on the new model.

---

## Section 8: Monitoring, retraining, incident response

- **Monitor:** data drift, prediction drift, **calibration (predicted/actual CTR by segment) — the ads-specific must-watch**, delayed-conversion completion curves (are conversions arriving as expected?), pacing health (are campaigns hitting budgets smoothly?), latency p95/p99, and **revenue + advertiser-ROI KPIs** (the model can look healthy while revenue quietly drops).
- **Retraining:** ads domains move fast (new campaigns, seasonality), so retrain frequently — often **continuous / online learning (FTRL)** for the linear layer plus periodic full deep-model retrains. Recalibrate even more often than you retrain.
- **Fallback:** on model/feature failure, degrade to last-good model → calibrated historical CTR priors → targeting + bid sort (Rung 0). Never serve nothing; serving nothing means zero revenue *and* empty slots.
- **Incident response:** freeze the model version, **diff calibration first** (the most common ads incident is a calibration break, not a crash), inspect a bank of bad auctions, check pacing controllers for oscillation, roll back on guardrail breach.

---

## Section 9: The worked one-hour interview (full transcript)

---

**[00:00 — The prompt]**

**INTERVIEWER:** Design an ad click prediction and ranking system for a feed.

**YOU:** Before I design anything, a few scoping questions, because the answers change the architecture a lot. First, what are we billing on — CPC, CPM, or CPA? That decides whether I predict click probability alone or also conversion probability with delayed labels. Second, is this a single ad slot or ads interleaved through a feed — that changes the auction and position handling. Third, whose objective am I optimizing — because this is a three-sided market between users, advertisers, and the platform. And fourth, how many eligible ads per request and what's my latency budget — that forces the funnel.

**INTERVIEWER:** CPC billing, ads interleaved in a feed, optimize platform revenue but don't wreck user experience or advertiser ROI. Assume 200M DAU, millions of active ads, and an 80ms budget for the ad decision.

**YOU:** Good, let me pin those.
```
200M DAU, ~30 ad slots/day -> ~70K QPS avg, ~210K peak
~10M eligible ads, p99 < 80ms
funnel: 10M --targeting--> ~10K --score--> auction --> top 1-3
```
That 80ms over 10K candidates forces a funnel: cheap targeting prunes 10M to ~10K, then a richer model scores the survivors. And one thing I want to flag immediately because it shapes the whole design: this isn't plain recommendation. The model output gets multiplied by a real bid and fed into an auction that sets prices, so the prediction has to be **calibrated**, not just well-ordered. I'll keep coming back to that.

**INTERVIEWER:** Go on.

---

**[00:06 — Framing & the label]**

**YOU:** I'll frame it as calibrated pCTR prediction feeding a utility auction. The model predicts P(click | user, ad, context). Because we're CPC, I don't need a conversion head yet, but I'd note that if we moved to CPA I'd add a pCVR head and inherit a delayed-conversion problem. My non-ML baseline is targeting rules plus bid sort with policy and frequency caps — that's my launch fallback. The positive label is a click, but clicks are biased and noisy, so how I treat them matters more than the model.

**INTERVIEWER:** Why calibration? If you're just ranking ads, doesn't ordering suffice like in recsys?

**YOU:** In recsys, yes — if the order is right you ship. In ads, no, because of the auction. I rank by eCPM, which is pCTR times bid. And under second-price pricing the winner pays roughly the runner-up's eCPM divided by the winner's pCTR. That price literally divides by my predicted pCTR. So if my pCTR is 20% too high, I undercharge the advertiser by 20% and the allocation is wrong — that's a direct revenue and trust error, not just a worse ranking. So I train with logloss, which is a proper scoring rule, and add an isotonic or Platt calibration layer, and I monitor calibration per segment in production.

---

**[00:12 — Data & the bias/delay problem]**

**INTERVIEWER:** Tell me about your labels and training data.

**YOU:** Main source is auction and impression logs joined with clicks. Two label problems I want to confront head-on. First, position bias — top slots get clicked because they're on top, so I use the position-as-feature trick: feed display position at training time, fix it to a constant at serving. Second — and this is the one that'd bite us if we went CPA — delayed conversions: a click is immediate but a conversion arrives days later inside the attribution window, so a click labeled "no conversion" today might just be a conversion that hasn't happened yet. The fix is a delayed-feedback model, like Chapelle's Criteo work, that models the conversion-delay distribution and treats recent non-conversions as censored rather than negative. For CPC today it's just clicks, but I'd still down-sample the billions of negative impressions and correct the sampling bias so calibration survives.

**INTERVIEWER:** That's the part most people skip. Keep going to the model.

---

**[00:18 — Baseline ladder]**

**YOU:** I'll climb a ladder. Rung zero is targeting plus bid sort — ships day one, stays as fallback, but it ignores relevance and lets a high bid beat a high-value ad. Rung one is logistic regression on crossed features with online FTRL updates — that's Google's "view from the trenches" — it's calibrated, handles billions of sparse features, and updates online. Rung two, my production default, is GBDT-plus-LR like the Facebook paper, or Wide-and-Deep / DCN-v2, which captures nonlinear feature crosses while staying calibratable. Rung three is a multi-task deep ranker with pCTR and pCVR heads, which I'd reach for under CPA. Notice the fanciest model isn't automatically best here — calibration stability and cheap online updates often beat a small AUC gain, because the output feeds money.

**INTERVIEWER:** Let's go with DCN-v2. Walk me through how its output becomes an ad shown to the user.

---

**[00:24 — The auction, to the floor]**

**YOU:** *(drawing)* DCN-v2 gives me a raw score per candidate; I pass it through a calibration layer to get a true pCTR. Then the auction. For each eligible ad I compute eCPM equals pCTR times bid, and I multiply by a quality factor so a spammy high-bid ad doesn't win and trash the user experience. I rank by eCPM, the top eCPM wins the slot. Pricing is generalized second price: the winner pays the minimum it would've needed to keep its slot, roughly the runner-up's eCPM divided by the winner's pCTR. 

The reason for second price rather than charging the bid is mechanism design: if I charge advertisers their own bid, they shade bids down and game me; second price makes near-truthful bidding optimal, which stabilizes the marketplace. And this is exactly why calibration is load-bearing — the price divides by pCTR, so a miscalibrated model misprices every auction.

**INTERVIEWER:** Advertisers have daily budgets. How do you stop one campaign from blowing its whole budget at midnight?

**YOU:** Budget pacing, which I model as a closed-loop controller per campaign — a PID controller targeting a smooth spend curve over the day. If a campaign spends too fast, I scale down its bid with a pacing multiplier so it's less competitive; too slow, I scale up. I use a smooth multiplier rather than a hard on/off because hard stops oscillate and miss cheap late-day impressions. Pacing is a constraint layer on top of the auction, not part of the model — keeping those separate keeps the system debuggable.

---

**[00:34 — Evaluation & the offline/online gap]**

**INTERVIEWER:** How do you evaluate before launch? And here's a scenario: offline AUC goes up, but online revenue drops. What happened?

**YOU:** Offline I report AUC for ranking, but always alongside calibration — logloss, ECE, and predicted-over-actual CTR ratio, sliced by vertical, device, and position, because aggregate calibration hides per-segment disasters. AUC alone is a trap in ads. On your scenario, the most likely ads-specific cause is exactly that: AUC improved but calibration got worse, so pCTR is now biased high, the auction misprices, and revenue drops even though ordering improved. Other causes: train-serve skew, an auction feedback loop where the new model changes which ads win and the offline metric trained on old logs can't see it, or advertisers re-tuning bids in response. I'd diff calibration first, then features, then look at the bid landscape.

**INTERVIEWER:** Design the A/B test.

**YOU:** Hypothesis: the new model raises revenue per session without hurting advertiser ROAS or ad hide rate and stays calibrated within plus or minus 2%. Here's the ads-specific subtlety: I can't cleanly randomize by user, because arms share each advertiser's finite budget, so a user-split leaks through the shared budget pool. For budget-sensitive changes I'd use a budget-split or geo-split so each arm has its own budget. Primary metric revenue per session; guardrails are advertiser ROAS, ad hide rate, p99 latency, calibration ratio, and small-advertiser fill rate. Ramp 1 to 5 to 20 to 50 percent checking guardrails and calibration at each hold. Revenue is heavy-tailed so I'd size generously and use CUPED for variance reduction, and run one to two weeks so advertisers re-equilibrate their bids and weekly seasonality washes out.

---

**[00:44 — Serving, scale, monitoring]**

**INTERVIEWER:** What runs in production and what do you watch?

**YOU:** Three paths. Serving path: targeting via inverted index prunes 10M to 10K, feature store lookup, calibrated scoring, pacing multipliers, then the auction and policy filters — separate services so they scale and fail independently, with the auction in-process after scoring since it's cheap arithmetic on the hot path. Data path: logs including the served position, down-sample negatives with correction, train, and refit calibration on recent held-out data; conversions stream in late so I support label backfill. Feedback path: clicks immediately, conversions delayed, joined via attribution with delayed-feedback correction, feeding frequent recalibration.

For monitoring, the ads-specific must-watch is calibration by segment — predicted over actual CTR — plus pacing health, conversion completion curves, latency, and the revenue and advertiser-ROI KPIs. I retrain frequently, often continuous FTRL for the linear layer plus periodic deep retrains, and recalibrate even more often because calibration drifts faster than the model goes stale. Fallback degrades to last-good model, then historical CTR priors, then targeting plus bid sort — never serve an empty slot. And the most common incident isn't a crash, it's a calibration break, so on an incident I diff calibration first.

---

**[00:55 — The close]**

**INTERVIEWER:** Anything to add?

**YOU:** To restate the core tradeoff: I maximize platform revenue through eCPM ranking while protecting advertiser ROI, user experience, latency, and — uniquely for ads — calibration, as guardrails. The two things that make this harder than recsys are that the prediction feeds money so it must be calibrated, and that conversions are delayed so labels lie about recent data. I keep the model, the auction, and the pacing controller as separate layers so each is debuggable.

And concretely, I've operated this shape: the retrieve-then-rank funnel and the offline/online calibration-and-skew problems are the same ones I dealt with shipping ranking systems in production, so the auction and pacing layers are the main new pieces, not the ML plumbing.

**INTERVIEWER:** Strong answer.

---

> **Why this transcript works (study these moves):**
> 1. **Scoped before designing** — CPC-vs-CPA question alone reshapes the whole label strategy.
> 2. **Led with calibration** — named the ads-specific core in the first two minutes.
> 3. **Explained the auction to the floor** — eCPM, GSP pricing, *why* second price, *why* calibration divides into price.
> 4. **Confronted both label problems** — position bias AND delayed conversions, with named fixes/papers.
> 5. **Didn't cargo-cult the fanciest model** — argued calibration/online-update beats raw AUC.
> 6. **Knew the A/B trap** — budget-split vs user-split market interference.
> 7. **Answered the offline-up/online-down trap with the calibration cause first.**
> 8. **Pacing as a controller** — showed the constraint layer most candidates forget.

---

## Section 10: Junior vs Senior — the highest-leverage contrast

| Decision | Junior answer | Senior answer |
|---|---|---|
| The objective | "Maximize CTR." | "Maximize platform revenue via eCPM, subject to advertiser-ROI and user-experience guardrails — it's a three-sided market." |
| Why calibration | (doesn't mention) | "eCPM = pCTR × bid feeds a second-price auction whose price divides by pCTR; miscalibration is a direct dollar error, so I calibrate and monitor ECE by segment." |
| Model choice | "A deep CTR model for best AUC." | "LR/GBDT+LR or DCN-v2 — calibration and online-update stability beat a small AUC gain when the output feeds money." |
| The label | "Train on clicks." | "Clicks are position-biased (position-as-feature fix); conversions are delayed (delayed-feedback model, censored not negative)." |
| Negatives | (ignores the scale) | "Down-sample billions of negatives and correct the sampling rate so calibration survives." |
| Ranking → shown ad | "Sort by bid" or "sort by score." | "Rank by eCPM, price by GSP (runner-up eCPM / winner pCTR) so truthful bidding is near-optimal." |
| Budgets | (doesn't mention) | "Per-campaign PID pacing controller adjusts a bid multiplier so spend tracks a target curve." |
| Eval | "Measure AUC." | "AUC alongside calibration (logloss, ECE, pred/actual ratio) sliced by segment; AUC alone is a trap." |
| A/B | "Split users 50/50." | "Budget-split/geo-split for budget-sensitive changes — user-split leaks through shared advertiser budgets." |
| Top incident | "Server crashed." | "Calibration drift — diff calibration first; it's the most common ads incident." |

---

## Section 11: One-page cheat sheet (whiteboard recall)

```text
SCAFFOLD:  Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor

NUMBERS:   DAU, ad slots/day -> QPS ; #active ads ; latency SLO (ads ~80ms, tight) ;
           funnel 10M --target--> 10K --score--> auction --> 1-3 slots.

BILLING:   CPC -> pCTR.  CPA -> pCTR x pCVR + DELAYED conversions.  CPM -> pacing/targeting.

CALIBRATION (the ads core, say it early):
   eCPM = pCTR x bid (x quality).  GSP price_1 = eCPM_2 / pCTR_1  <- divides by pCTR.
   miscalibration = direct revenue/pricing error, NOT just bad ranking.
   train logloss (proper scoring rule) + isotonic/Platt + monitor ECE BY SEGMENT.

AUCTION:   rank by eCPM ; second-price/GSP so truthful bidding is ~optimal ;
           quality factor protects user experience.

PACING:    per-campaign PID controller -> bid multiplier so spend tracks a target curve.

LABELS:    position bias -> position-as-feature, fixed at serve.
           delayed conversions -> Chapelle delayed-feedback (censored, not negative).
           billions of negatives -> down-sample + correct sampling rate (keep calibration).

MODEL:     LR+FTRL (trenches) -> GBDT+LR (FB) / Wide&Deep / DCN-v2.
           fanciest != best: calibration + online stability > small AUC gain.

EVAL:      AUC (rank) ALONGSIDE logloss/ECE/(pred-CTR/actual-CTR) sliced by segment.
           offline-up/online-down #1 cause = calibration broke while AUC rose.

A/B:       budget-split/geo-split (NOT user-split) for budget-sensitive changes ;
           guardrails: ROAS, hide rate, latency, calibration, small-advertiser fill ;
           CUPED for heavy-tailed revenue ; 1-2 wks for bid re-equilibration.

3 PATHS:   serving (target->score->auction) | data (log served position, late conv backfill)
           | feedback (clicks now, conversions later -> delayed-feedback correction)

MONITOR:   calibration by segment (must-watch), pacing health, conv-completion curves,
           latency, revenue+ROI. Most common incident = calibration drift.
```

---

## Section 12: Follow-up questions the interviewer may ask

- **What changes at 100M+ users / 200K QPS?** Separate targeting/scoring/auction services, shard indexes, cache features + ad embeddings + hourly CTR aggregates, async logging, p95/p99 SLOs, and graceful degradation to historical-CTR priors under load.
- **How do you handle cold-start ads?** Creative embedding + advertiser-level CTR priors + an exploration budget (contextual bandit) so new ads aren't starved by incumbents (rich-get-richer kills advertiser onboarding).
- **CPC vs CPA — what changes?** CPA adds a pCVR head and the delayed-feedback problem; eCPM becomes pCTR × pCVR × value; calibration matters on *both* heads.
- **Offline AUC up, online revenue down — why?** Calibration regressed while ranking improved (the #1 cause), train-serve skew, auction feedback loop, or advertiser bid re-tuning. Diff calibration first.
- **How do you correct position bias?** Position-as-feature at train, fixed at serve; plus exploration / inverse-propensity weighting on logged data.
- **How do you keep the auction truthful?** Second-price / GSP pricing so bid shading isn't rewarded; reserve prices to protect revenue floors; a quality factor in eCPM to protect users.
- **How do you pace budgets?** Per-campaign closed-loop (PID) controller adjusting a bid multiplier to track a target spend curve; smooth, not hard on/off.
- **Why not just sort by bid?** A high bid is not high value; you'd show irrelevant ads, tank user experience and advertiser ROI, and invite gaming. Rank by *expected value* (eCPM) and price by competition.

---

## Section 13: Common mistakes (anti-patterns to avoid)

- Treating ads as plain recsys and never saying the word **calibration** — the single biggest tell.
- Reporting AUC without calibration; optimizing AUC and shipping a miscalibrated model that leaks revenue.
- Forgetting the **auction and pricing** entirely — designing a ranker but never explaining how a probability + bid becomes a shown ad at a price.
- Ignoring **budget pacing** — a real production constraint that most candidates never mention.
- Hard-labeling recent non-conversions as negatives (ignoring **delayed feedback**).
- Down-sampling negatives without correcting the sampling rate, silently breaking calibration.
- User-splitting an A/B test that's budget-sensitive (market interference).
- Jumping to the fanciest deep model when calibration + online-update stability matter more.
- Treating deployment as "an endpoint" instead of serving/data/feedback paths, and missing that conversions arrive *late*.

---

## Section 14: Transfer — what mastering ads unlocks

This case shares the funnel skeleton with recsys and adds the auction/calibration/pacing layer. Mastering it transfers directly:

| Problem | What changes vs ads | What stays identical |
|---|---|---|
| **Recommendation (case 01)** | no auction, ordering-only (no calibration need), engagement label | retrieve→rank funnel, position bias, A/B, feedback loop |
| **Feed ranking (case 02)** | organic multi-objective (MMoE), no money/auction | funnel, position bias, calibration ideas for blending |
| **Search ranking (case 03)** | strong query signal, relevance labels, listwise loss | retrieve→rank, NDCG/calibration, offline-online gap |
| **Sponsored search** | ads + a query — *this case plus search intent* | the entire auction + pacing + calibration stack |
| **Marketplace bidding / RTB** | you're the bidder, not the seller; bid optimization | pCTR/pCVR calibration, delayed feedback, value estimation |
| **Notification optimization (case 19)** | "bid" replaced by long-term value; explore/exploit | calibration of pAction, pacing/volume control, A/B traps |

The leverage: **the auction is the one genuinely new mechanism.** Learn it once here and sponsored search, RTB, and bidding all collapse into "the model output feeds a market."

---

## Sources
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research, Advances in TF-Ranking: https://research.google/blog/advances-in-tf-ranking/
- Ad Click Prediction: a View from the Trenches (Google, FTRL + online learning): https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
- Practical Lessons from Predicting Clicks on Ads at Facebook (GBDT + LR, calibration, normalized entropy): https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
- DCN V2: Improved Deep & Cross Network for Web-scale Ranking (Google): https://arxiv.org/abs/2008.13535
- Wide & Deep Learning for Recommender Systems (Google): https://arxiv.org/abs/1606.07792
- Modeling Delayed Feedback in Display Advertising (Chapelle, Criteo, KDD 2014): https://dl.acm.org/doi/10.1145/2623330.2623634
