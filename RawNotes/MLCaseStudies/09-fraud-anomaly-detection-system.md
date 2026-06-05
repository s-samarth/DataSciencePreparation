# 09. Fraud / Anomaly Detection System

**Company tags:** Stripe, PayPal, Amazon, JP Morgan, Visa, Adyen, Microsoft
**Interview frequency:** High
**Why it matters:** Fraud is the case where the textbook assumptions all break at once: the classes are extremely imbalanced, the labels are *delayed and censored by your own decisions*, and the data-generating process is an **adversary who adapts to your model**. Candidates who treat it as "train a classifier on labeled fraud" miss the entire problem.

---

## 0. How to use this doc

Built two ways; read it twice.

1. **As a thinking guide.** The headers are the whiteboard order. Internalize the *triggers* for each rung.
2. **As a worked transcript.** Section 11 is a full timestamped hour. Cover the `YOU:` lines and answer from memory.

The one idea to carry out: **fraud is an adversarial game where your labels are both delayed (chargebacks arrive months later) and censored (you only learn the truth about transactions you *approved* — blocked ones never get a label). A senior design treats the label-generation process as a first-class problem and the model as something that must be retrained fast against a moving opponent.** Say that and you separate yourself from everyone who just says "use XGBoost on the fraud labels."

Scaffold (identical across all cases):

```
Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor
```

The **Data/Labels** phase is where this case is won. Spend disproportionate time there.

---

## 1. The reusable scaffold, stated once

| Phase | The question |
|---|---|
| Clarify | What are we protecting, what's the decision, what's the cost asymmetry? |
| Frame | What's the learnable target, given labels are delayed and censored? |
| Data / Labels | Where do labels come from, and how does our own blocking bias them? |
| Baseline | Simplest shippable thing (rules), and what breaks it? |
| Model | Rules + GBDT + anomaly + graph, explained to the floor. |
| Eval | Imbalanced metrics, cost-sensitive threshold; the offline/online gap. |
| Deploy | Three paths; inline scoring; the review-queue feedback loop. |
| Monitor | Adversarial drift; what pages someone; the fallback. |

---

## 2. Clarify requirements (scripted)

| Question | Why it changes the design |
|---|---|
| "What are we protecting — card payments, account takeover, fake accounts, invoice/seller fraud?" | Card fraud is a per-transaction inline decision. Account takeover is behavioral/sequence. Fake-account/abuse is graph-heavy (overlaps the spam-bot case). I'll assume **real-time payment fraud** as the spine. |
| "Is the decision inline (block/allow before authorizing) or after-the-fact review?" | Inline means a hard latency SLO in the payment path (sub-100ms) and an automated action. After-the-fact means a review queue with looser latency. Most systems do both: inline auto-block for high risk, queue for medium. |
| "What's the cost of a missed fraud vs a false decline?" | The whole threshold is set by this. A missed fraud loses the transaction amount + chargeback fee; a false decline loses a sale, adds friction, and can churn a good customer. These are *unequal and amount-dependent*, so the threshold must be cost-sensitive, not a fixed 0.5. |
| "How fast do confirmed labels arrive?" | Chargebacks land 30-90 days later; some fraud is never reported. This delay defines how I train and how I know today's model is working before labels mature. |
| "Is there a human investigation team, and what capacity?" | The review queue is a scarce resource; my medium-risk band must fit reviewer capacity, and their verdicts are my cleanest labels. |
| "Are there fairness/compliance constraints (adverse-action, disparate impact)?" | Financial decisions are regulated. I need explainability (reason codes) per decision and cohort fairness monitoring, not just AUC. |

**Numbers I'll commit to and carry through:**

- **Volume:** ~5,000 transactions/sec peak (~400M/day).
- **Base fraud rate:** ~0.2% of transactions (1 in 500) — extreme imbalance.
- **Latency SLO:** inline decision p99 < 100ms (it sits in the payment authorization path; if we're slow, the payment fails).
- **Label delay:** chargebacks mature over 30-90 days.
- **Cost asymmetry:** a missed fraud ≈ full transaction amount + ~$15 chargeback fee; a false decline ≈ lost margin + churn risk. Amount-dependent, so a $5 and a $5,000 transaction get different thresholds.
- **Review capacity:** humans can review ~the top N flagged/day; the medium-risk band must fit it.

### Latency budget, derived out loud

```
Feature fetch (velocity counters, entity history from feature store/Redis)  ~40 ms
Rules engine (deterministic checks: blocklists, hard limits)                ~5 ms
ML score (GBDT, milliseconds; +anomaly layer)                               ~10 ms
Decision logic + action + audit log                                         ~10 ms
-------------------------------------------------------------------------------
Total p99 well under 100 ms.
```

The dominant cost is **feature retrieval**, not model inference — GBDT scoring is microseconds. That tells me the engineering effort goes into a fast, fresh feature store for streaming aggregates, not into a fancier model.

### Scale / storage note

400M txns/day, each scored with ~hundreds of features and an audit record (~2KB) = **~800 GB/day** of decision logs, retained for the chargeback window + compliance (often years, tiered to cold storage). Velocity counters (per card/device/IP over 1m/1h/24h windows) live hot in a streaming store.

---

## 3. Frame as an ML problem

- **Framing:** a cost-sensitive, real-time risk score on each event, combined with rules and an anomaly layer, mapped to actions (allow / challenge / block / review) by thresholds tuned on *business cost*, not probability.
- **The target:** "confirmed fraud" — but confirmation comes from chargebacks, investigator verdicts, or admitted abuse, all of which are **delayed and incomplete**. The honest framing: my label is a *noisy, censored proxy* that matures over time.
- **Why this framing wins:** it forces the two questions that define seniority here — (1) how do I act *now* when labels won't exist for 60 days, and (2) how do I avoid training on a label set my own blocking decisions corrupted?
- **Non-ML baseline:** a rules engine — amount thresholds, velocity limits, blocked geographies, known-bad device/card lists. Transparent, compliance-friendly, ships immediately, and you *never delete it* — it stays as the fast, auditable first line and the fallback.

---

## 4. Data and labels — the censored-label problem, head-on

This is the section that wins the interview. Three intertwined label problems:

### Problem 1: extreme imbalance

Fraud is ~0.2%. A model that predicts "never fraud" is 99.8% accurate and useless. Consequences:
- **Accuracy and ROC-AUC are misleading** (ROC-AUC looks great because true negatives dominate). Use **PR-AUC** and **recall at a fixed low false-positive rate** — those reflect the operating regime.
- Handle imbalance with **class weighting / focal loss** or **downsampling negatives with a correction** (reweight so calibrated probabilities survive — same trick as the ads case), never by blindly oversampling in a way that distorts calibration.

### Problem 2: labels are CENSORED by your own decisions (the deep one)

You only observe the fraud outcome of transactions you **approved**. Every transaction you **blocked** never completes, so it never gets a chargeback — you never learn if it was actually fraud. Train naively on "approved + later-charged-back = fraud" and your training set is missing exactly the cases your current system already catches. The model learns "fraud looks like the fraud that *slips past my current rules*," and re-deploying it creates a blind spot that compounds.

Mitigations to name:
- **Reject inference / counterfactual labeling:** explicitly model the selection. At minimum, treat blocked cases as a distinct, missing-label population, not as negatives.
- **A small random holdout of approvals:** let a tiny, bounded fraction of *would-be-blocked* (or all) transactions through unscored/unblocked to collect unbiased ground truth on the full distribution. This is costly (you eat some fraud) but it's the only way to measure true recall and de-bias training. The explore/exploit framing from recsys applies: you pay for exploration to keep the label set honest.
- **Friction instead of hard block** where possible (step-up auth, 3-D Secure) so a flagged-but-legitimate user can still complete and generate a "good" label.

### Problem 3: labels are DELAYED

Chargebacks mature over 30-90 days. So:
- **You cannot wait** for mature labels to know if today's model works — you need leading indicators (model score distribution shift, rules-hit rate, customer-complaint rate, early chargeback signals).
- **Training-set maturity:** a transaction labeled "good" today may flip to "fraud" in 45 days. Use a label-maturity cutoff and reweight recent data carefully (the delayed-feedback modeling from the ads case transfers directly).

### Features (the real engineering)

- **Velocity / aggregates:** count and sum of transactions per card / device / IP / merchant over 1m, 1h, 24h windows — the single most predictive feature family. These live in a streaming feature store.
- **Deviation from the entity's own history:** amount, geo, time-of-day vs the card's baseline (this is the "anomaly" signal even within supervised features).
- **Entity graph features:** shared device / IP / shipping address across accounts surfaces fraud rings (a bridge to the graph case). One device tied to 50 cards is a screaming signal.
- **Train-serve skew is lethal here:** a velocity counter computed differently offline vs online silently destroys the model. Compute features once in a shared library used by both paths.

---

## 5. Baseline -> why it breaks -> next rung

| Rung | What it is | Why it breaks -> trigger to climb |
|---|---|---|
| 0 | Rules engine: thresholds, velocity limits, blocklists. | Brittle; adversaries probe and evade; can't weigh many weak signals. Trigger: fraud slips through that no single rule catches but many weak signals would. |
| 1 | **Calibrated GBDT** on tabular features + rules, cost-sensitive threshold, review queue for the medium band. | Misses *novel* fraud patterns absent from labels; labels lag the adversary. Trigger: a new fraud MO appears that the supervised model (trained on old fraud) can't see. |
| 2 | **GBDT + unsupervised anomaly layer + graph features**, rules retained, cost-sensitive multi-band thresholds, fast retraining. | Production default. Trigger to extend: organized rings, sequence/account-takeover patterns, or scale that needs deep models. |
| 3 | Graph neural nets for rings, sequence models for account takeover, real-time graph features. | Heavy infra and harder explainability. Trigger: measured ring/sequence fraud that rung 2's tabular+graph features can't capture. |

GBDT is the workhorse (rung 1-2): tabular data, fast inference, naturally calibrated-ish, and SHAP gives per-decision reason codes for compliance. Earn the anomaly layer by explaining why a supervised model trained on *past* fraud is structurally blind to *new* fraud.

---

## 6. The architecture, explained to the floor

```
   Transaction event
        |
   Feature service (streaming velocity counters + entity history + graph features)
        |
   +----+------------------------------+
   |                                   |
   Rules engine (deterministic,        Supervised GBDT (calibrated risk score)
   auditable, instant block on              |
   known-bad)                          Unsupervised anomaly score
   |                                    (isolation forest / autoencoder:
   |                                     "weird vs this entity's baseline")
   +----+------------------------------+
        |
   Score fusion + COST-SENSITIVE thresholds (per amount, per cohort)
        |
   +----+--------+-------------+--------------+
   | allow       | challenge   | review queue | hard block
   |             | (step-up    | (medium risk,| (high risk)
   |             |  auth/3DS)  |  humans)     |
        |
   Action + reason codes (SHAP) + immutable audit log
        |
   Outcomes (chargebacks 30-90d, investigator verdicts, step-up success) --> labels
```

### Why each model exists

- **Rules:** the fast, transparent floor. Instant block on known-bad entities, hard regulatory limits. Adversaries evade them, but they're cheap, explainable, and catch the obvious. Never removed.
- **Supervised GBDT (the core):** weighs hundreds of weak signals into a calibrated probability of fraud. Gradient-boosted trees because the data is tabular, inference is microseconds, it handles mixed feature types and missingness well, and SHAP yields per-transaction reason codes ("flagged: 8 txns in 2 min on a new device, amount 12x baseline") that satisfy adverse-action explainability. **Calibration matters** because the threshold is cost-based: I need P(fraud) to *mean* 0.2, so I can compare expected-loss = P(fraud) × amount against the cost of declining. Calibrate with Platt/isotonic; monitor ECE.
- **Unsupervised anomaly layer:** the supervised model can only recognize fraud *like the labeled past*. A brand-new MO has no labels yet, so it's invisible to GBDT — but it often looks *anomalous* relative to an entity's baseline. Isolation forest or an autoencoder flags "this is weird" without needing fraud labels, catching novel patterns early and feeding investigators (who then create the first labels for the new pattern). This is the answer to the adversary's adaptation.
- **Graph features / GNN (rung 2-3):** fraud is often organized — shared devices, shipping addresses, funding sources. Graph features (entity connectivity, ring detection) catch coordinated fraud that per-transaction features miss.

### The cost-sensitive decision — not a 0.5 threshold

The output isn't "fraud/not-fraud at 0.5." It's an **expected-loss** decision:

```
expected fraud loss  = P(fraud) x (amount + chargeback fee)
expected decline cost = P(good) x (margin + churn risk)
act to minimize expected cost; the threshold therefore MOVES with amount.
```

A $5,000 transaction gets blocked at a lower P(fraud) than a $5 one. And instead of binary allow/block, use **bands**: allow / challenge (step-up auth — friction, not a block) / human review / hard block. Step-up is the senior touch: it lets a flagged-but-legit user prove themselves and generate a clean label, instead of losing the sale.

### Canonical references (verified)

- Isolation Forest — Liu et al., 2008: https://ieeexplore.ieee.org/document/4781136
- XGBoost — Chen & Guestrin, 2016: https://arxiv.org/abs/1603.02754
- SHAP (per-prediction explanations) — Lundberg & Lee, 2017: https://arxiv.org/abs/1705.07874
- Delayed feedback modeling (chargeback/conversion delay) — Chapelle, 2014: https://dl.acm.org/doi/10.1145/2623330.2623634
- Google Rules of Machine Learning: https://developers.google.com/machine-learning/guides/rules-of-ml

---

## 7. Evaluation — imbalanced, cost-weighted, and adversarially honest

- **Offline metrics:** **PR-AUC** and **recall at a fixed low FPR** (e.g., recall@0.5% FPR) — these match the operating regime; never lead with accuracy or ROC-AUC. Plus **calibration** (reliability diagram, ECE) because the threshold is cost-based. Slice by amount band, geography, customer cohort, and fraud MO.
- **The real metric is money:** dollars of fraud prevented minus dollars of good-customer friction/churn, at the chosen operating point. Translate model metrics into expected-loss curves the business can read.
- **Online metrics:** blocked-fraud amount, false-decline rate, step-up success rate, review-queue precision (fraction of flagged that were truly fraud), time-to-alert, and customer-complaint rate (a fast leading indicator of false positives).
- **Fairness/compliance:** false-positive rate parity across protected cohorts; explainability coverage (every adverse action has reason codes).

### The offline-to-online gap, including the classic trap

**"Offline PR-AUC was great, online we're missing fraud / declining good users."** Causes, ordered:

1. **Label censoring (the signature cause).** Offline you scored well *on the biased label set your own blocking created*. In production you face the full distribution, including fraud your old rules used to catch — which isn't in training — so recall is worse than offline claimed.
2. **Adversarial drift.** Between training and serving, fraudsters changed tactics. The model is always fighting the last war; offline metrics on historical fraud overstate live performance by construction.
3. **Label immaturity.** Offline "recall" was computed before recent chargebacks matured, so some "false positives" were actually fraud you correctly caught (and vice versa). Your offline labels were wrong.
4. **Train-serve skew.** A velocity counter computed differently online silently shifts every score.
5. **Threshold/cost mismatch.** You optimized PR-AUC offline but set the threshold without the amount-dependent cost model, so it's miscalibrated to the business.

### One fully specified A/B test

- **Hypothesis:** the GBDT+anomaly model (vs rules-only) reduces fraud loss without raising false declines beyond budget.
- **Unit:** randomize by **customer/account** (sticky), so a user gets consistent treatment and we can attribute churn.
- **Arms:** control = rules-only; treatment = rules + model with cost-sensitive bands.
- **Primary:** net dollars saved = fraud-loss reduction − incremental good-customer friction cost.
- **Guardrails (auto-stop):** false-decline rate, customer-complaint rate, cohort FP parity, p99 latency, review-queue overflow.
- **The hard part — delayed labels:** the primary metric needs ~60 days to mature. So I read **leading indicators** during the test (model-flagged rate, step-up success, complaint rate, early chargeback signal) and only *confirm* the win after labels mature. I'd also keep a **random holdout** (a small fully-unblocked control) running permanently to measure true recall against unbiased labels.
- **Ramp:** 1 → 5 → 25 → 50%, guardrail check each step.
- **Rollback:** any guardrail breach, or leading indicators trending bad.

### Error analysis ritual

Review false negatives (fraud that got through — each is a new pattern to learn) and false positives (good users blocked — each is friction/churn). Maintain a bank of both; after every retrain, re-check. Investigators' verdicts on the review queue are your cleanest, freshest labels — close that loop tightly.

---

## 8. Deployment — three paths

- **Serving path (inline, <100ms):** event → feature service → rules + GBDT + anomaly → cost-sensitive decision → action + reason codes → audit log. Rules run first (instant block on known-bad, cheap). Feature retrieval is the latency bottleneck, so the feature store is the engineering focus.
- **Data path:** streaming aggregation of velocity counters and entity/graph features, written by the same shared feature library the model trained on (kills train-serve skew). Plus the immutable decision/audit log for compliance and label joining.
- **Feedback path (the flywheel + its bias):** chargebacks, investigator verdicts, and step-up outcomes flow back as labels — but this path is exactly where censoring bias enters, so it includes the **random holdout** of unblocked transactions that keeps the label set honest. Investigator verdicts feed fast retraining of the supervised model and seed labels for newly-detected anomaly patterns.

### Rollout discipline

Shadow-score new models against production on live traffic (compare scores, no action) → canary by account % → ramp, with the delayed-label caveat: gate the *ramp* on leading indicators and guardrails, confirm the *win* only after labels mature. Never fully promote a fraud model on offline metrics alone.

### Monitoring and fallback

- **What pages someone:** fraud-loss spike (the adversary found a hole), false-decline/complaint spike (model too aggressive), score-distribution drift (adversarial shift), feature-pipeline staleness (velocity counters frozen → every score wrong), review-queue overflow, cohort FP-parity breach, calibration drift.
- **Fallback ladder:** if the model is misbehaving, **fall back to the rules engine** (the always-present floor) — degrade to conservative, auditable blocking, never to "allow everything." If features are stale, widen rules and raise friction (more step-up) rather than flying blind. Critical-amount transactions get the most conservative treatment under degradation.
- **Incident response:** because fraud is adversarial, a sudden loss spike is often an *attack*, not a bug — freeze the model, tighten rules immediately as a stopgap, pull the audit trail for the attacked segment, identify the new MO, label it, and fast-retrain or add a targeted rule. The rules engine is your emergency lever because it deploys in minutes; model retraining takes longer.

---

## 9. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Core insight | "Train a classifier on fraud labels." | Adversarial game with delayed + censored labels; label-generation is the hard part. |
| Imbalance | "Use accuracy / ROC-AUC." | PR-AUC, recall@fixed-FPR, calibration; accuracy is meaningless at 0.2%. |
| Labels | "We have fraud labels." | Labels are censored by our own blocking and delayed by chargebacks; needs reject inference + random holdout. |
| Threshold | "Threshold at 0.5." | Cost-sensitive, amount-dependent expected-loss decision with action bands. |
| Novel fraud | Only supervised. | Adds unsupervised anomaly layer because supervised is blind to unlabeled new MOs. |
| Friction | "Block or allow." | Step-up auth as a middle band — recovers good users and generates clean labels. |
| Features | "Use transaction features." | Streaming velocity counters + entity-baseline deviation + graph/ring features; shared lib to kill skew. |
| Explainability | Ignores it. | SHAP reason codes for adverse-action compliance and investigator triage. |
| Offline/online | "Offline PR-AUC predicts prod." | Censoring, adversarial drift, label immaturity break it; confirm wins only after labels mature. |
| Incident | "Roll back the model." | Treats loss spikes as attacks; rules engine as the minutes-fast emergency lever. |

---

## 10. One-page whiteboard cheat sheet

```
NUMBERS: 5K txn/s (~400M/day) | fraud ~0.2% (1 in 500) | p99<100ms inline
         chargebacks mature 30-90d | cost asymmetry amount-dependent

BIG IDEA: adversarial game + DELAYED + CENSORED labels.
  censored: you only learn truth about txns you APPROVED.
  -> train naively => blind to fraud your rules already catch.
  fix: reject inference + small RANDOM HOLDOUT of unblocked txns.

LADDER: rules -> +calibrated GBDT (cost threshold) -> +ANOMALY layer +graph
        -> GNN rings / sequence ATO
  anomaly layer = answer to NOVEL fraud (supervised is blind to unlabeled MOs)

METRICS: PR-AUC, recall@fixed-FPR, CALIBRATION (not accuracy/ROC-AUC)
         real metric = $ fraud prevented - $ good-user friction

DECISION (not 0.5):
  E[fraud loss]=P(fraud)x(amount+fee) vs E[decline cost]=P(good)x(margin+churn)
  threshold MOVES with amount | bands: allow / CHALLENGE(step-up) / review / block

FEATURES: velocity counters (1m/1h/24h per card/device/IP) = top family
          entity-baseline deviation | graph (shared device/addr = ring)
          shared feature lib offline==online (skew is lethal)

OFFLINE-GREAT/ONLINE-BAD: LABEL CENSORING | adversarial drift
                          | label immaturity | train-serve skew | cost mismatch

A/B: unit=account | primary=net $ saved (matures 60d!)
     read LEADING indicators during test | permanent random holdout for true recall
     guard=false-decline, complaints, cohort FP parity, latency

DEPLOY: serving(inline)/data(streaming features)/feedback(+holdout to debias)
        FALLBACK = rules engine (minutes-fast emergency lever); never "allow all"
        loss spike = likely ATTACK -> tighten rules now, label, fast-retrain
        SHAP reason codes for compliance
```

---

## 11. Full interview transcript (one hour)

**[00:00] INTERVIEWER:** Design a real-time fraud detection system.

**[00:30] YOU:** A few clarifying questions first. What are we protecting — card payments, account takeover, fake accounts? Is the decision inline, blocking before we authorize, or after-the-fact review? What's the cost of a missed fraud versus a false decline? And how fast do confirmed labels arrive?

**[01:00] INTERVIEWER:** Real-time card payments, inline. Missed fraud costs us the transaction amount plus a chargeback fee; a false decline costs a sale and annoys a good customer. Chargebacks come back over one to three months.

**[01:20] YOU:** Then let me name the two things that make fraud hard and different from a normal classifier. First, it's adversarial — fraudsters adapt to whatever I deploy, so the distribution is non-stationary by design. Second, and this is the subtle one, my labels are both delayed and *censored*. Delayed because chargebacks take up to 90 days. Censored because I only ever learn the true outcome of transactions I *approved* — anything I block never completes, so it never gets a chargeback, so I never find out if it was really fraud. If I train naively, my training set is missing exactly the fraud my current system already catches, and the new model goes blind to it. So the label-generation process is itself a first-class design problem.

**[02:40] INTERVIEWER:** That's a real issue. How do you deal with the censoring?

**[02:50] YOU:** A few ways. Conceptually, reject inference — I treat blocked transactions as a distinct missing-label population, not as negatives. Practically, the honest fix is a small, bounded **random holdout**: let a tiny fraction of transactions through unblocked even when the model would block them, to collect unbiased ground truth on the full distribution. It costs me some fraud losses, but it's the only way to measure true recall and de-bias training — it's the explore/exploit trade from recommendations applied to fraud. And wherever I can, I prefer **friction over a hard block** — step-up authentication like 3-D Secure — so a flagged-but-legitimate customer can still complete and generate a clean "good" label instead of just vanishing.

**[04:10] INTERVIEWER:** Okay. The labels are also delayed. How do you train and know it's working?

**[04:20] YOU:** Two consequences. For training, a transaction labeled "good" today might flip to fraud in 45 days, so I use a label-maturity cutoff and reweight recent data carefully — the same delayed-feedback modeling used for ad conversions. For knowing it works *now*, I can't wait 60 days, so I watch leading indicators: model-score distribution shift, rules-hit rate, step-up success rate, customer-complaint rate, and early chargeback signal. I only *confirm* a win once labels mature, but I *steer* on the leading indicators.

**[05:30] INTERVIEWER:** Let's talk metrics. How do you evaluate offline?

**[05:40] YOU:** Not with accuracy or ROC-AUC — at a 0.2% base rate, "never fraud" is 99.8% accurate and ROC-AUC is flattered by the huge true-negative mass. I use PR-AUC and recall at a fixed low false-positive rate, because that's the regime I actually operate in. And calibration — reliability diagram and ECE — because my threshold is cost-based and I need the predicted probability to actually mean what it says. But the real metric is money: dollars of fraud prevented minus dollars of good-customer friction and churn, at the chosen operating point.

**[06:50] INTERVIEWER:** Say more about the threshold being cost-based.

**[07:00] YOU:** It's not a 0.5 cutoff. For each transaction I compare expected fraud loss — probability of fraud times amount plus fee — against expected decline cost — probability it's good times lost margin plus churn risk — and act to minimize expected cost. Because the loss scales with amount, the threshold *moves*: a $5,000 transaction gets blocked at a much lower fraud probability than a $5 one. And I don't just allow or block — I use bands. Allow, challenge with step-up auth, send to human review, or hard block. The step-up band is the important one: it turns a lost sale into a recoverable customer and a fresh label.

**[08:20] INTERVIEWER:** Walk me up the model ladder.

**[08:30] YOU:** Rung 0 is a rules engine — amount limits, velocity caps, blocklists. Transparent, compliance-friendly, ships day one, and I never delete it; it's the floor and the fallback. It breaks because it's brittle and adversaries probe and evade it, and it can't combine many weak signals. Rung 1 is a calibrated gradient-boosted tree on tabular features plus the rules, with the cost-sensitive threshold and a review queue for the medium band. GBDT because the data is tabular, inference is microseconds, it handles mixed types and missingness, and SHAP gives me per-decision reason codes for adverse-action compliance. It breaks on *novel* fraud — a supervised model only recognizes fraud like the labeled past, so a brand-new MO with no labels is invisible. That triggers rung 2: add an unsupervised anomaly layer — isolation forest or an autoencoder — that flags "this is weird relative to this entity's baseline" without needing fraud labels, catching new patterns early and feeding investigators who then create the first labels for them. Plus graph features for rings. Rung 3, if needed, is GNNs for organized rings and sequence models for account takeover.

**[10:30] INTERVIEWER:** Why does the anomaly layer help if it has high false positives?

**[10:40] YOU:** Because I don't auto-block on it — I route its hits to human review, where false positives cost reviewer time, not customer churn. Its job is recall on the unknown: surface the novel pattern fast so investigators can confirm it and generate labels, which then teach the supervised model. It's the bridge that lets the system adapt to the adversary instead of always fighting the last war. The supervised model gives precision on known fraud; the anomaly layer gives early recall on new fraud.

**[11:50] INTERVIEWER:** What features matter most?

**[12:00] YOU:** Velocity aggregates, by far — counts and sums per card, device, IP, and merchant over 1-minute, 1-hour, and 24-hour windows. Fraud is bursty, so "8 transactions in 2 minutes on a new device" is gold. Then deviation from the entity's own baseline — amount, geography, time-of-day versus this card's history. Then graph features — shared device, IP, or shipping address across accounts surfaces rings; one device tied to 50 cards is a screaming signal. And one engineering point I'd stress: train-serve skew is lethal here. If a velocity counter is computed even slightly differently offline versus online, every score shifts. So I compute features once in a shared library used by both training and serving.

**[13:30] INTERVIEWER:** Your offline PR-AUC is great, you launch, and you're missing fraud and declining good users. Why?

**[13:40] YOU:** The signature cause is label censoring — offline I scored well on the biased label set my own blocking created, but in production I face the full distribution, including fraud my old rules used to catch, which isn't in my training data, so real recall is worse than offline claimed. Second, adversarial drift — tactics changed between training and serving, so historical metrics overstate live performance by construction. Third, label immaturity — my offline "false positives" might actually be fraud whose chargebacks hadn't landed yet, so my offline labels were just wrong. Fourth, train-serve skew in the velocity features. Fifth, a threshold-cost mismatch — I optimized PR-AUC but set the threshold without the amount-dependent cost model.

**[15:00] INTERVIEWER:** Design the A/B test, given those delayed labels.

**[15:10] YOU:** Randomize by account, sticky, so treatment is consistent and I can attribute churn. Control is rules-only, treatment is rules plus the model with cost-sensitive bands. Primary metric is net dollars saved — fraud-loss reduction minus incremental good-customer friction. The catch is that the primary metric needs ~60 days to mature, so during the test I steer on leading indicators — flagged rate, step-up success, complaint rate, early chargebacks — and only confirm the win after labels mature. Guardrails that auto-stop: false-decline rate, complaint rate, cohort false-positive parity, p99 latency, review-queue overflow. Ramp 1, 5, 25, 50. And I'd keep a small permanent random holdout running so I always have unbiased ground truth for true recall.

**[16:40] INTERVIEWER:** Fraud loss suddenly spikes in production. What do you do?

**[16:50] YOU:** I treat a sudden spike as an *attack*, not a bug — someone found a hole. The rules engine is my emergency lever because it deploys in minutes, while retraining takes longer. So: freeze the model, immediately tighten rules around the attacked segment as a stopgap, pull the audit trail to identify the new MO, label those cases, and then fast-retrain or add a targeted rule. The audit log and SHAP reason codes make the attack pattern legible. The key reflex is that I have a fast, auditable fallback that doesn't require a model deploy.

**[18:00] INTERVIEWER:** And compliance — these are financial decisions.

**[18:10] YOU:** Every adverse action needs reason codes, which is why I lean on GBDT plus SHAP — I can tell a declined customer and a regulator the top factors: velocity, amount deviation, device risk. I monitor false-positive-rate parity across protected cohorts as a guardrail, not just aggregate AUC, and I keep the rules engine auditable. Explainability coverage — every block has reasons — is a first-class metric.

**[19:00] YOU:** This connects directly to an anomaly-detection system I built with SHAP explainability for revenue-leakage prevention. The lesson that shaped my thinking was exactly the censoring problem — we initially measured "recall" against labels our own controls had already shaped, and it took a holdout to see our true blind spots. And SHAP wasn't just for compliance; it was how investigators triaged the queue fast. So I lead with the label-generation problem and per-decision explanations rather than with the model choice.

**[19:50] INTERVIEWER:** That's the senior answer. Good.

### Why this transcript works

- **Names the adversarial + delayed + censored label structure up front** — the defining insight.
- **Proposes the random holdout** as the honest fix for censoring, framed as explore/exploit.
- **Uses cost-sensitive, amount-dependent thresholds and action bands**, not a 0.5 cutoff.
- **Justifies the anomaly layer** as the structural answer to novel fraud the supervised model can't see.
- **Treats train-serve skew and feature freshness** as lethal, with a shared feature library.
- **Designs the A/B around delayed labels** (leading indicators now, confirm later, permanent holdout).
- **Treats loss spikes as attacks** with the rules engine as the minutes-fast emergency lever.
- **Closes on real anomaly/SHAP/revenue-leakage experience**, anchored by the censoring scar.

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x volume?** Feature store becomes the bottleneck (it already is) — shard the velocity counters, keep them hot, and ensure the streaming aggregation keeps up; model inference stays cheap. Tighten review-queue automation since human capacity doesn't scale 10x.
- **How do you handle cold start (new user/merchant)?** No history, so lean on population priors, conservative rules, step-up friction, and graph signals (is this new entity connected to known-bad ones?) until a behavioral baseline forms.
- **How do you set the threshold?** From the expected-loss curve per amount band, constrained by review capacity and a false-decline budget; validate online and re-tune as the cost model and fraud mix shift.
- **Offline great, online bad — what do you check?** Label censoring, adversarial drift, label immaturity, train-serve skew, threshold-cost mismatch. (Section 7.)
- **How do you debug a bad launch?** Slice by amount/cohort/MO, compare score distributions vs control, check feature freshness, inspect false negatives for a new pattern, and lean on the rules fallback while investigating.
- **How do you prevent feedback loops / keep labels honest?** The random holdout, step-up instead of block, reject inference, and never training solely on the biased approved-only set.
- **Supervised vs unsupervised — when each?** Supervised for precision on known fraud with mature labels; unsupervised for recall on novel patterns with no labels. Run both; route anomaly hits to review, not auto-block.

---

## 13. Common mistakes

- Treating fraud as **ordinary classification** and ignoring the adversary.
- Ignoring **label censoring** — training only on approved-then-charged-back data and going blind to caught fraud.
- Ignoring **label delay** — evaluating on immature labels or waiting 60 days to react.
- Leading with **accuracy / ROC-AUC** instead of PR-AUC, recall@fixed-FPR, and calibration.
- A fixed **0.5 threshold** instead of cost-sensitive, amount-dependent decisions with action bands.
- Only **supervised** modeling, so novel fraud is invisible.
- **Train-serve skew** in velocity features silently corrupting scores.
- No **explainability** for regulated adverse actions.
- No fast **rules fallback**, or a fallback that "allows everything" under degradation.

---

## 14. Transfer — what this case unlocks

- **17 Spam / Bot Detection:** shares adversarial dynamics and graph/ring detection; that case owns coordinated inauthentic behavior and signup-time abuse, this one owns payments + censored labels + cost-sensitive thresholds.
- **11 Ads CTR / Experimentation:** delayed-feedback modeling, downsampling-with-correction, and calibration are the same machinery.
- **07 AI Agent Ticket Resolution:** refund-abuse detection and per-account anomaly limits borrow directly from here.
- **10 Content Moderation:** cost-asymmetric thresholds, human-review queues, and adversarial drift are shared.
- **20 ML Monitoring & Drift:** adversarial concept drift and fast-retraining triggers are a drift-detection special case.
- **01 Recommendations:** the explore/exploit logic behind the random holdout is the same idea pointed at label honesty.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Rules of Machine Learning: https://developers.google.com/machine-learning/guides/rules-of-ml

Added (verified canonical):
- Isolation Forest (Liu et al., 2008): https://ieeexplore.ieee.org/document/4781136
- XGBoost (Chen & Guestrin, 2016): https://arxiv.org/abs/1603.02754
- SHAP (Lundberg & Lee, 2017): https://arxiv.org/abs/1705.07874
- Delayed feedback modeling (Chapelle, 2014): https://dl.acm.org/doi/10.1145/2623330.2623634
