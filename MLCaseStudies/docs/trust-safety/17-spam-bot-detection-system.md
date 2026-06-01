# 17. Spam / Bot Detection System

**Company tags:** Meta, Google, LinkedIn, X/Twitter, marketplaces, payments, any platform with open signup
**Interview frequency:** Medium-high
**Why it matters:** This is an adversarial case like fraud (file 09) and the safety gateway (file 13), but it owns a different axis: **the adversary operates at scale through coordinated networks of accounts.** A single fake account, examined alone, looks fine — the signal is in the *correlation across thousands of accounts* acting in lockstep. And you often must judge an account at signup, with **near-zero behavioral history**, precisely when you most want to stop it. If you frame this as "train a classifier on account features," you'll catch the dumb bots and miss the professional adversary entirely.

---

## 0. How to use this doc

Two passes.

**Pass 1 (intuition).** Read the prose and transcript. Lock three ideas: (1) **individual-level detection is necessary but loses to sophisticated abuse; coordination is the signal** — look across accounts (shared device/IP/fingerprint, synchronized timing, near-duplicate content, dense bipartite subgraphs) to find the *ring*, not the member; (2) **the signup-time adversary has no history**, so you lean on registration-time signals (device/IP reputation, fingerprint, velocity) and let risk *graduate over the account lifecycle*; (3) the goal isn't a binary block — it's to **raise the attacker's cost** and apply **graduated friction** matched to confidence, without tipping off the adversary about where your boundary is.

**Pass 2 (active recall).** Cover the page. Can you (a) explain why a per-account classifier misses coordinated abuse and what catches it instead, (b) explain why you score at signup with no history and what features survive there, (c) explain why labels are noisy/delayed/*adversarially polluted* and how that shapes training, and (d) lay out the graduated-friction action ladder and why you'd shadow-restrict rather than hard-block? Those four are the case.

**The scaffold (shared across this set):**

> **Clarify → Frame → Data/Labels → Baseline → Model → Eval → Deploy → Monitor**

Bends here: "Model" is a *layered* system (rules → per-account ML → coordination/graph jobs → human review) operating at multiple latencies; "Data/Labels" confronts noisy, delayed, adversarial labels and extreme imbalance; "Eval" carries recall-at-fixed-FPR (false-positive = blocking a real user, expensive) and prevalence; "Monitor" treats the adversary's adaptation as the primary drift source.

**The senior tell, stated once:** say early that "the dumb bots are an individual-classification problem, but the money is in *coordinated inauthentic behavior*, which is invisible per-account and only shows up at the cluster level — so my design is layered: cheap per-event scoring plus batch graph/coordination detection plus human review, with graduated friction, not a single classifier." That reframes it from "spam classifier" to an adversarial defense-in-depth system.

---

## 1. Clarify (scripted, with *why each answer changes the design*)

| Question | Why it changes the design |
|---|---|
| **What are we protecting — signups, content/messages, engagement (fake likes/follows), or marketplace listings?** | Each has a different attack and signal. Fake-engagement abuse is heavily *coordinated* (one operator, thousands of puppets) → graph detection. Spam messages → content + velocity. Fake signups → registration-time signals. I'll design the layered system but call out where each layer matters. |
| **What's the adversary's sophistication and motive?** | Script kiddies (cheap, obvious, high volume) vs professional spam/fraud operations (residential proxies, real device farms, aged accounts, human-in-the-loop CAPTCHA solving). The latter forces coordination/graph detection and expensive-to-forge signals. Motive (scam, propaganda, scraping, growth-hacking) shapes which behavior to watch. |
| **Cost of a false positive vs a false negative?** | Blocking a *real* user (false takedown) is a trust/PR/appeals cost; missing a bot is an abuse cost. Asymmetric and action-dependent — drives graduated friction, not a single threshold. |
| **Can I act at signup, or only after behavior?** | If I can gate signup, I stop bots before they act, but with near-zero history. If only post-hoc, detection is richer but damage is already done. Realistic answer: both — score at signup *and* re-score as behavior accrues. |
| **Latency budget?** | Signup/login scoring is inline (p99 ~100–200ms). Per-action scoring is near-real-time. Coordination/graph detection is batch/streaming (minutes–hours). Different layers, different latencies. |
| **Is there an enforcement/appeals + human-review team?** | Determines how much I can route to review vs auto-act, and the appeals path that bounds false-takedown harm. |

State assumptions and move: protecting signups + content + engagement, mixed adversary up to professional, asymmetric cost favoring caution on real users, act at both signup and post-behavior, layered latencies, a review+appeals team exists.

---

## 2. Numbers up front (carry them through)

- **Scale:** ~1M signup attempts/day, billions of actions/day (messages, follows, likes). Inauthentic prevalence varies wildly by surface — assume ~5–15% of signup attempts are abusive, higher during attack waves.
- **Latency budget (derived out loud):** signup/login scoring is **inline, p99 < ~200ms** — so a fast model (GBDT/logistic over precomputed reputation features), not a heavy graph job, decides at the gate. Per-action scoring near-real-time (tens of ms, often async). **Coordination/graph detection runs in batch or streaming windows (minutes to hours)** because it needs to see many accounts together — it cannot and need not be inline.
- **The core metric (the headline):** **recall at a fixed, low false-positive rate.** Because a false positive = blocking/suspending a *real* user, the FPR ceiling is tight (e.g., < 0.1% on hard actions). You report "what fraction of abuse do we catch while keeping real-user false-suspensions under the ceiling," exactly like fraud's recall-at-fixed-FPR (file 09) — accuracy and ROC-AUC are useless under this imbalance.
- **Prevalence north star:** % of active accounts/content that are inauthentic, **measured on a randomly-sampled, human-audited set** (not on flagged accounts — that only measures precision of what you caught). Same discipline as moderation prevalence (file 10). This is the number leadership actually cares about.
- **Time-to-detection:** how fast after an account turns abusive do we act — bounded by how much damage we tolerate, traded against tipping off the adversary (§6.4).
- **Cost framing:** the goal is to **raise the attacker's cost** above their expected gain. You don't need 100% detection; you need to make abuse uneconomic. Say this — it reframes the objective from "perfect classifier" to "shift the cost curve."

---

## 3. The conceptual spine: coordination is the signal, and the cost asymmetry

### 3.1 Individual detection loses; coordination wins
A per-account supervised model catches obvious bots (brand-new account, datacenter IP, posting identical spam 100×/min). But a professional adversary makes each account look *individually plausible* — residential proxy, real-ish device, human-paced actions, unique-enough content. Examined alone, each puppet passes.

The signal they cannot easily hide is **coordination**: to be economical, the operator reuses infrastructure and behavior across thousands of accounts. So you detect the *ring*, not the member:
- **Shared infrastructure:** same device fingerprint, IP/ASN, payment instrument, phone-number block, email pattern across many accounts.
- **Synchronized behavior:** accounts acting at the same times, in the same order, with the same cadence (temporal correlation).
- **Near-duplicate content:** the same message/profile/image (perceptual/text hashing) across accounts.
- **Dense bipartite subgraphs:** 10,000 accounts all liking the same 50 posts, or following the same target — a structure that essentially never occurs organically.

Methods: connected-components / community detection over the "shared-attribute" graph, **Sybil detection** (SybilRank / SybilGuard-style trust propagation from known-good seeds), loopy belief propagation over the account graph, and clustering on behavioral embeddings. **This is what differentiates spam/bot from individual-transaction fraud (file 09):** there the unit is one transaction; here the most valuable unit is a *cluster*. State this.

### 3.2 The cost asymmetry → graduated friction, not a binary block
Blocking a real user is expensive (trust, PR, appeals, lost growth); missing a bot is also costly but usually cheaper per instance. And confidence varies continuously. So the action is not block/allow — it's a **graduated-friction ladder** matched to risk (this case's analogue of fraud's step-up auth, the support agent's reversibility tiers, and moderation's graduated enforcement):

```
low risk           -> allow
slightly elevated  -> invisible friction (extra fingerprinting, soft rate-limit)
elevated           -> challenge: CAPTCHA, email/phone verification
high               -> hard rate-limit / feature restriction / shadow-restrict
very high + corrob.-> suspend (+ appeals path)
```

The friction is chosen to be cheap for a real user (a CAPTCHA they pass) but expensive for an attacker at scale (CAPTCHA × 10,000 accounts costs real money/time). That asymmetry is the lever.

### 3.3 Raise the attacker's cost; lean on expensive-to-forge signals
A senior reframes the objective: you will never "solve" an adaptive adversary; you make abuse **uneconomic**. Weight signals by how expensive they are for the attacker to change:
- **Cheap to change** (username, profile text, user-agent string): weak, evadable signals.
- **Expensive to change** (verified phone, payment instrument, device hardware fingerprint, aged-account reputation, IP cost): strong, sticky signals.
Designing detection around expensive-to-forge signals is what makes evasion costly. This is the spam/bot version of fraud's "money is the hard-to-fake signal."

---

## 4. The data/label problem for *this* domain: noisy, delayed, *adversarially polluted* labels + extreme imbalance

Every case has a signature data problem. Spam/bot's labels are uniquely bad:

1. **No clean ground truth; multiple weak sources.** Confirmed labels come from investigators (accurate but slow and scarce), user reports (fast, abundant, but **noisy and gameable** — adversaries mass-report real users to get them banned), retrospective takedowns, and high-confidence rules. You must fuse these with confidence weights, not treat them as equal.

2. **Adversarial label pollution (unique twist).** The adversary actively corrupts your labels — false-reporting good users, and making bots *look* good to poison your negatives. So you cannot blindly trust report-based labels; weight by reporter reputation and require corroboration for high-stakes actions.

3. **Delayed labels.** Like fraud (file 09), confirmation lags — an account may not be confirmed abusive for days/weeks. Train with delayed-label awareness; don't treat "not yet caught" as "legitimate."

4. **Extreme imbalance + censored/unlabeled negatives → PU learning.** Abuse is a small fraction, and most accounts are *unlabeled* (not confirmed either way), not confirmed-negative. This is a **positive-unlabeled (PU) learning** setup: you have some confirmed positives, a vast unlabeled pool (mostly but not entirely legit), and few confirmed negatives. Treating unlabeled as negative biases the model. Use PU methods / semi-supervised learning, and reserve a **randomly-sampled human-audited holdout** for honest prevalence + FPR estimation (you cannot measure the false-positive rate on real users any other way — same logic as fraud's random holdout and moderation's sampled prevalence).

5. **The signup cold-start (the distinctive one).** At account creation there is *no behavioral history*, yet that's when you most want to act. So registration-time features are reputation- and infrastructure-based (device fingerprint, IP/ASN reputation, email/phone validity, signup velocity from shared infra, form-fill behavioral biometrics). Then the account's risk score **graduates over its lifecycle** as behavior accrues — score at signup, re-score continuously.

---

## 5. The baseline → why-it-breaks → next-rung ladder

**Rung 0 — Rules, rate limits, CAPTCHA, blocklists.** Velocity caps, IP/domain blocklists, keyword filters, CAPTCHA at signup.
- *Works:* immediate protection against high-volume dumb abuse; deterministic; a **fast emergency lever** during an attack (minutes to deploy a rule).
- *Breaks:* attackers adapt within hours (rotate IPs, solve CAPTCHAs via human farms), rules over-block edge cases, and combinatorial behavior escapes simple thresholds. **Trigger:** abuse continues despite rules, or rules start false-positiving real users.

**Rung 1 — Supervised per-account/per-event risk model.** GBDT/NN over behavioral + reputation + content features; outputs a calibrated risk score → graduated friction.
- *Adds:* learns complex feature combinations, calibrated scores for the friction ladder, handles the signup-time model with reputation features.
- *Breaks:* label delay and adversarial drift degrade it; and crucially it scores accounts **independently**, so it misses coordinated rings where each member looks fine. **Trigger:** a coordinated campaign where individual accounts pass the model but the aggregate is obviously fake.

**Rung 2 — Coordination / graph detection (the distinctive layer).** Batch/streaming jobs over the account graph: shared-attribute connected components, community detection, Sybil trust-propagation, synchronized-behavior and near-duplicate-content clustering.
- *Adds:* catches the professional adversary by finding the *ring*; one detected cluster takes down thousands at once; robust to per-account plausibility.
- *Breaks alone:* expensive, batch (not inline), and can over-cluster (a legit community looks dense too) → needs human review and corroboration before mass action. **Trigger:** need to combine fast inline decisions with slow coordination evidence.

**Rung 3 — Layered risk system (recommended production design).** Rules (fast lever) + per-event ML (inline friction) + coordination/graph jobs (batch, catches rings) + human review (high-stakes, ambiguous) + a feedback loop, with **threshold/action governance** and an appeals path. Detailed in §6. This is where you stop.

Meta-rule out loud: "I start with rules and a per-account model because they handle volume and give me a fast lever, but the load-bearing detection for sophisticated abuse is the coordination layer — and I keep humans in the loop for high-stakes actions because both the model and the graph clustering have costly false positives."

---

## 6. The architecture explained to the floor (layered, multi-latency)

```text
              ┌──────────────────── feedback: investigator labels + audited holdout + appeals ───────────────────┐
 signup/      v                                                                                                   |
 action --> [RULES + rate limits](ms, fast lever) --> [per-event ML SCORE](inline, ~50-200ms) --> graduated friction
                                                          |                                                       |
            [COORDINATION / GRAPH JOBS](batch/stream, min-hrs): shared-infra CC, Sybil rank, sync/dup clustering  |
                                                          |                                                       |
                                                   [cluster evidence] --> [HUMAN REVIEW for high-stakes] --> enforce
                                                                                                                  |
                          (all decisions + scores + cluster IDs logged) ---------------------------------------- ┘
```

### 6.1 The fast inline layer (rules + per-event ML)
Rules are the minutes-fast emergency brake (deploy a velocity cap or IP block during an attack). The per-event model scores at signup (reputation/infra features, since no history) and at action-time (behavioral features as they accrue), outputting a **calibrated** risk score so the friction ladder (§3.2) is meaningful. Calibration matters: friction tiers are thresholds on probability, and you tune them by cost/cohort.

### 6.2 The coordination layer (batch/streaming, the distinctive part)
Builds the account graph (nodes = accounts; edges/attributes = shared device/IP/payment/phone, co-engagement, content similarity, temporal sync) and runs:
- **Connected components / community detection** on shared-attribute graphs to surface clusters.
- **Sybil detection** (trust propagation from verified-good seeds; Sybils are poorly connected to the honest region).
- **Behavioral synchrony & near-duplicate** clustering (same actions same time; perceptual/text hashes).
Output: cluster IDs + a cluster-level risk score that *feeds back* into per-account scores and into review queues. A single confirmed bad cluster enables mass takedown — high leverage. Because it's batch, it doesn't block signup; it cleans up and informs the inline layer.

### 6.3 Human review + governance
High-stakes actions (hard suspension, mass cluster takedown) and ambiguous cases route to investigators. Their decisions are the high-quality labels (§4) and the corroboration that prevents mass false-takedowns. **Threshold/action governance**: documented mapping from risk → friction tier, tuned by cohort and cost, with an **appeals path** that bounds false-positive harm and itself generates labels (a successful appeal = a false positive to learn from).

### 6.4 Don't tip off the adversary (the adversarial-ops nuance)
Two senior moves: (1) **shadow-restrict** (let the abusive account *think* it's working — its spam isn't actually delivered) instead of hard-blocking, so the operator doesn't immediately learn the boundary and iterate; (2) **batch takedowns** — sometimes let a detected cluster run briefly and take it down all at once, denying the adversary the per-account feedback that tells them which signal got them caught. This is the time-to-detection ↔ don't-reveal-detection-logic tradeoff: acting instantly minimizes damage but trains the attacker. State the tradeoff explicitly.

### 6.5 The three paths, named
- **Serving path:** rules → inline ML → friction; coordination jobs async.
- **Data path:** reputation stores (device/IP/phone), behavioral features, the account graph, content hashes.
- **Feedback path:** investigator labels + audited holdout + appeals + confirmed takedowns → retrain (delayed-label-aware, PU) + recalibrate thresholds + monitor adversary drift.

### 6.6 Costs
Inline scoring is cheap; the coordination/graph jobs are the expensive compute (trillion-edge-ish graphs, like file 16) — run incrementally, shard by community, prioritize high-risk regions. Reputation lookups must be fast KV (feature store).

---

## 7. Evaluation

### 7.1 Metrics
- **Offline (imbalanced, like fraud):** **PR-AUC and recall at a fixed low FPR** (the operating metric), time-to-detection, sliced by surface, account age, geography/cohort. **Never accuracy or ROC-AUC** under this imbalance.
- **Prevalence** of inauthentic accounts/content on a **randomly-sampled human-audited set** (the north star; flagged-precision alone is misleading).
- **Cluster-level metrics:** precision/recall of detected coordinated clusters (a different unit than per-account).
- **Online:** report rate, takedown accuracy (audited), spam/abuse prevalence trend, user-trust signals.
- **Guardrails (first-class):** **false-takedown rate / appeal-overturn rate** (the key harm metric — a high appeal-success rate means you're banning real users), cohort fairness (is one demographic over-flagged?), reviewer load, latency.

### 7.2 The offline↔online gap (the trap, adversarial form)
*"Recall-at-FPR was great offline; why is online detection dropping / false-takedowns rising?"* Causes, in order:
1. **Adversarial drift — the adversary adapted** the day after you shipped. The #1 cause and unique in its *intentionality*: unlike natural drift, someone is actively probing your boundary. Offline test set is from the old attack distribution.
2. **Label pollution / leakage** — your offline labels were contaminated by adversarial false-reports or by features that encode the *current* enforcement (e.g., "was rate-limited" leaks the label).
3. **PU/imbalance mis-handling** — you treated unlabeled as negative, so offline precision was inflated relative to reality.
4. **Coordination invisible offline per-account** — a per-account offline eval looks fine while a coordinated campaign you can only see in aggregate sails through.
5. **Cohort skew** — offline aggregate good, but a new-user or specific-geography cohort is over-flagged (false-takedowns concentrated there).
6. **Feedback loop** — you only get labels on accounts you *acted on*, so the model becomes blind to abuse it never challenges.

Cure: continuous adversarial monitoring, audited-holdout prevalence/FPR, exploration (don't act on a small random slice to keep unbiased labels), and the coordination layer to catch what per-account eval misses.

### 7.3 A fully-specified A/B test (with the adversarial wrinkle)
- **Hypothesis:** model/threshold v5 raises recall-at-fixed-FPR and lowers prevalence without raising appeal-overturn rate vs v4.
- **The wrinkle:** you can't cleanly A/B by showing different enforcement to *accounts*, because (a) the adversary controls many accounts and will notice/contaminate arms, and (b) abuse is coordinated (treating one puppet affects the campaign — interference, like file 16). Mitigations: randomize at a higher unit (geography/segment) or run **shadow mode** (score with the new model, log would-be actions, don't enforce) and compare against the audited holdout — often the cleanest honest read.
- **Unit:** segment or shadow. **Primary:** recall-at-FPR / prevalence. **Guardrails:** appeal-overturn (false-takedown) rate — hard, cohort fairness, reviewer load, latency.
- **Runtime/ramp:** shadow → canary → ramp; watch for adversary reaction; ≥ enough time for delayed labels to mature.
- **Rollback trigger:** appeal-overturn spike (banning real users), cohort-fairness regression, prevalence rise, latency breach.

---

## 8. Deployment, monitoring, fallback, incident response

- **Rollout:** **shadow mode is the workhorse here** — score new models on live traffic, log would-be actions, compare to the audited holdout before enforcing (you really don't want to learn you're banning real users in production). Then canary/ramp. Rules deploy fast as the emergency lever.
- **Monitoring:** prevalence trend (audited), recall-at-FPR, **appeal-overturn rate** (the false-takedown alarm — a spike means a model/threshold change started hitting real users), report-rate spikes (attack wave *or* mass false-reporting campaign — investigate which), cluster-detection volume, score-distribution drift (adversary adapting), cohort fairness, reviewer queue depth, latency. Slice by surface, account age, geography.
- **Fallback:** if the ML/graph layer degrades, fall back to rules + rate limits + CAPTCHA + manual review — safe, higher-friction, never "abuse protection off." For real users, prefer *softer* friction on uncertainty (challenge, not ban) so degradation doesn't mass-ban legit users.
- **Incident response:** an attack wave → deploy a rule as the minutes-fast brake, spin up coordination jobs on the affected region, batch-takedown the cluster, then train a proper model update; a *false-takedown* incident (banning real users) → freeze the offending model/threshold, identify the cohort/feature from logs, roll back, auto-reinstate via the appeals pipeline, add to the audited set. Every decision logs score + cluster ID + reason, so forensics is tractable.
- **Adversarial ops:** prefer shadow-restrict and batch takedowns (§6.4) to avoid teaching the adversary; rotate detection logic; keep some signals secret.

---

## 9. Full one-hour interview transcript

**[0:00] INTERVIEWER:** Design a system that detects spam accounts, bot behavior, or abusive messages.

**[0:30] YOU:** Let me scope it, because the attack and the signal change a lot. What am I protecting — signups, content/messages, or engagement like fake likes and follows? How sophisticated is the adversary — script kiddies, or professional operations with residential proxies and device farms? What's the cost of a false positive versus a false negative? And can I act at signup, or only after I've seen behavior?

**[1:15] INTERVIEWER:** All three surfaces, adversary up to professional. False-positiving a real user is very costly. You can act at both signup and afterward.

**[1:30] YOU:** Then let me state the framing I think is the crux and what makes this different from individual-transaction fraud. The dumb bots — brand-new account, datacenter IP, identical spam a hundred times a minute — are a per-account classification problem, and I'll handle them with rules and a supervised model. But the money is in **coordinated inauthentic behavior**: a professional adversary makes each account look individually plausible, so examined alone every puppet passes. The signal they can't hide is *coordination* — to be economical they reuse infrastructure and behavior across thousands of accounts. So I detect the *ring*, not the member: shared device fingerprints, IPs, payment instruments, phone blocks; synchronized timing; near-duplicate content; dense bipartite subgraphs where ten thousand accounts all like the same fifty posts. That's a graph and clustering problem, batch, not a per-account model. So my design is layered, not a single classifier.

**[3:30] INTERVIEWER:** Let's start with signup. You said you can act there but there's no history.

**[3:40] YOU:** Right, that's the cold-start adversary, and it's distinctive — I most want to stop a bot before it acts, but at creation it has zero behavioral history. So signup-time features are reputation- and infrastructure-based: device fingerprint, IP and ASN reputation, email and phone validity, signup velocity from shared infrastructure, and form-fill behavioral biometrics like typing and mouse dynamics. I score inline, under about 200ms, with a fast model. And critically, the account's risk score **graduates over its lifecycle** — I score at signup and re-score continuously as behavior accrues. A signup-time "maybe" plus later behavioral evidence becomes a confident decision.

**[5:30] INTERVIEWER:** What do you do with the score — block or allow?

**[5:40] YOU:** Neither, mostly. The cost is asymmetric — blocking a real user is a trust and appeals disaster — and confidence is continuous, so I use a **graduated-friction ladder**. Low risk, allow. Slightly elevated, invisible friction like extra fingerprinting or a soft rate-limit. Elevated, a challenge — CAPTCHA or phone verification. High, hard rate-limit or feature restriction. Very high *and corroborated*, suspend with an appeals path. The friction is designed to be cheap for a real user — a CAPTCHA they pass once — but expensive for an attacker at ten thousand accounts. That asymmetry is the whole lever. It's the same idea as step-up auth in fraud or graduated enforcement in moderation.

**[7:30] INTERVIEWER:** Tell me more about the coordination detection.

**[7:40] YOU:** Batch and streaming jobs over an account graph — nodes are accounts, edges and shared attributes are things like the same device, IP, payment instrument, co-engagement on the same targets, content similarity, and temporal synchrony. Then I run connected-components and community detection on the shared-attribute graph to surface clusters, Sybil trust-propagation from verified-good seeds — Sybils are poorly connected to the honest region — and synchrony plus near-duplicate clustering with perceptual and text hashes. A single confirmed bad cluster lets me take down thousands of accounts at once, which is huge leverage. It feeds a cluster-level risk score back into per-account scores and into the review queue. It's batch because it needs to see many accounts together, so it doesn't gate signup — it cleans up and informs the inline layer.

**[9:30] INTERVIEWER:** Coordinated clusters — couldn't a legitimate community look dense too?

**[9:40] YOU:** Exactly the false-positive risk, which is why mass cluster takedowns go through human review and require corroboration, not auto-action. A real fan community is dense but has organic diversity — varied devices, varied content, varied timing — whereas a bot ring shares infrastructure and acts in lockstep. But the boundary is genuinely fuzzy, so high-stakes actions get an investigator and an appeals path. The graph layer proposes; humans dispose for the irreversible stuff.

**[11:00] INTERVIEWER:** How do you get labels?

**[11:10] YOU:** This is the ugliest part. There's no clean ground truth — I fuse investigator confirmations, which are accurate but slow and scarce; user reports, which are fast and abundant but noisy and *gameable*; retrospective takedowns; and high-confidence rules. And there's a twist unique to this domain: **adversarial label pollution.** The adversary mass-reports real users to get them banned and makes bots look good to poison my negatives. So I weight reports by reporter reputation and require corroboration for high-stakes actions — I never blindly trust report labels. Labels are also delayed, like fraud, so I don't treat "not yet caught" as legitimate. And it's a **positive-unlabeled** problem — most accounts are unlabeled, not confirmed-negative — so I use PU and semi-supervised methods rather than treating unlabeled as negative, which would inflate my precision. To measure honestly, I keep a randomly-sampled, human-audited holdout, the only way to estimate true prevalence and my false-positive rate on real users.

**[13:30] INTERVIEWER:** What's your primary metric?

**[13:40] YOU:** Recall at a fixed, low false-positive rate, because a false positive is banning a real user and that ceiling is tight — accuracy and ROC-AUC are meaningless at this imbalance, same as fraud. Plus prevalence on the audited sample as the north star, time-to-detection, and cluster-level precision and recall as a separate unit. And as a hard guardrail, the **appeal-overturn rate** — if appeals are frequently succeeding, I'm banning real users, full stop.

**[15:00] INTERVIEWER:** Offline recall-at-FPR is great, but online detection drops and false-takedowns rise. Why?

**[15:10] YOU:** Top cause and unique in its intentionality: **adversarial drift** — someone actively adapted to my boundary the day after I shipped, so my offline test set is from the old attack distribution. Second, label pollution or leakage — my offline labels were contaminated by adversarial false-reports, or a feature like "was rate-limited" leaked the current enforcement. Third, I mishandled PU and treated unlabeled as negative, inflating offline precision. Fourth, coordination is invisible to a per-account offline eval — the per-account numbers look fine while a coordinated campaign I can only see in aggregate sails through. Fifth, cohort skew — aggregate is fine but new users or one geography are over-flagged, which is where the false-takedowns concentrate. Sixth, the feedback loop — I only get labels on accounts I acted on, so I go blind to abuse I never challenge. The fixes are continuous adversarial monitoring, the audited holdout, a small no-action exploration slice to keep unbiased labels, and the coordination layer.

**[17:30] INTERVIEWER:** How would you A/B a new model?

**[17:40] YOU:** Carefully, because normal account-level A/B breaks here. The adversary controls many accounts and will notice and contaminate the arms, and abuse is coordinated so treating one puppet affects the whole campaign — interference, like the graph case. So I lean on **shadow mode**: score with the new model, log would-be actions, don't enforce, and compare against the audited holdout. That's often the cleanest honest read, and it means I learn whether I'd ban real users *before* I actually do. If I need a live test I randomize at a higher unit like geography. Primary metric recall-at-FPR and prevalence, hard guardrail on appeal-overturn rate and cohort fairness, ramp slowly watching for adversary reaction, roll back on an overturn spike.

**[19:30] INTERVIEWER:** When you catch a bot, do you block it immediately?

**[19:40] YOU:** Often no, and this is an adversarial-ops nuance. If I hard-block instantly, the operator immediately learns where my boundary is and iterates. So I prefer to **shadow-restrict** — let the account think it's working while its spam isn't actually delivered — and to **batch takedowns**, letting a detected cluster run briefly and removing it all at once, which denies the attacker the per-account feedback about which signal caught them. There's a real tradeoff: acting instantly minimizes damage but trains the adversary; waiting preserves my detection logic but allows more abuse. I'd tune that by surface and severity — instant for high-harm, batched-and-shadowed for low-harm scraping.

**[21:30] INTERVIEWER:** An attack wave hits right now. What do you do?

**[21:40] YOU:** Rules are my minutes-fast brake — deploy a velocity cap or block the abusing IP range or device pattern immediately, accepting some over-blocking temporarily. Spin up the coordination jobs focused on the affected region to find the cluster, batch-takedown with review for the high-stakes accounts, then train a proper model and feature update for durability. Throughout, watch the appeal-overturn rate so my emergency rule isn't catching real users. Every decision is logged with score, cluster ID, and reason, so I can do forensics and feed the labels back.

**[23:30] INTERVIEWER:** Wrap up.

**[23:40] YOU:** To close: the dumb bots are individual classification, but the real adversary is coordinated, so my system is layered — rules as a fast lever, a calibrated per-event model driving graduated friction, a batch coordination and graph layer to catch the rings, and humans for high-stakes actions — with noisy, delayed, adversarially-polluted labels handled by PU learning and an audited holdout, recall-at-fixed-FPR as the metric, and shadow-restrict plus batch takedowns so I don't tip off the adversary. The throughline is raising the attacker's cost on expensive-to-forge signals, not chasing a perfect classifier. This connects to my anomaly-detection and drift work — coordinated-behavior clustering and adversarial concept drift — rather than to consumer content moderation.

### Why this transcript works
- **Distinguishes individual vs coordinated detection** in the first two minutes — the senior reframe that separates this from fraud.
- **Owns the signup cold-start adversary** and lifecycle risk graduation.
- **Graduated friction** matched to cost, not a binary block.
- **Confronts adversarially-polluted, delayed, PU labels** honestly with the right methods and the audited holdout.
- **Uses recall-at-fixed-FPR and audited prevalence** (not accuracy) and the appeal-overturn guardrail.
- **Knows the adversarial-ops tricks** — shadow-restrict, batch takedowns, the tip-off tradeoff — and shadow-mode A/B under interference.
- **Closes by connecting to anomaly/drift experience**, deliberately *not* claiming content-moderation expertise (per the file's own note).

---

## 10. Junior vs senior contrast

| Dimension | Junior | Senior |
|---|---|---|
| Core | "Train a spam classifier on account features." | "Individual classification catches dumb bots; **coordination/graph detection** catches the professional adversary." |
| Signup | "Use account history." | Cold-start adversary → reputation/infra + behavioral-biometric signals; risk **graduates over lifecycle**. |
| Action | "Block bad accounts." | **Graduated friction** ladder matched to cost/confidence; CAPTCHA→verify→restrict→suspend+appeals. |
| Labels | "Use reports / labeled data." | Noisy, delayed, **adversarially polluted**, **PU** → weight by source, corroborate, semi-supervised, audited holdout. |
| Metric | "Accuracy / ROC-AUC." | **Recall at fixed low FPR**, audited prevalence, cluster precision/recall, **appeal-overturn** guardrail. |
| Objective | "Catch all bots." | **Raise attacker cost** on expensive-to-forge signals; make abuse uneconomic. |
| Adversary | "Retrain when accuracy drops." | Adversarial drift is intentional; shadow-restrict + batch takedowns to avoid tipping off; rotate logic. |
| A/B | "Randomize accounts." | Interference + adversary contamination → **shadow mode** / higher-unit randomization. |

---

## 11. One-page whiteboard cheat sheet

```text
SPINE: dumb bots = individual classification; PRO adversary = COORDINATED -> detect the RING not the member
       objective = RAISE ATTACKER COST on expensive-to-forge signals (not a perfect classifier)

NUMBERS: ~1M signups/day, billions of actions; prevalence ~5-15% (spiky)
         signup scoring inline p99<200ms; coordination = BATCH/stream (min-hrs)
         METRIC: recall @ FIXED LOW FPR (FP = ban real user). prevalence on AUDITED random sample. NOT accuracy.

COORDINATION SIGNALS: shared device/IP/payment/phone · synchronized timing · near-dup content · dense bipartite subgraphs
  methods: connected components / community detection · SYBIL trust-prop from good seeds · behavioral clustering

SIGNUP COLD-START: no history -> device fingerprint, IP/ASN rep, email/phone validity, velocity, behavioral biometrics
                   risk GRADUATES over lifecycle (score @signup, re-score as behavior accrues)

GRADUATED FRICTION: allow -> invisible friction -> CAPTCHA/verify -> rate-limit/restrict/SHADOW -> suspend(+appeals)
  cheap for real user, expensive at 10K accounts.

LABELS: investigators(slow,good) + reports(noisy,GAMEABLE) + takedowns + rules; ADVERSARIAL POLLUTION (mass false-report)
        delayed; PU learning (unlabeled != negative); AUDITED HOLDOUT for true prevalence/FPR

LADDER: 0 rules/ratelimit/CAPTCHA (fast lever) -> 1 per-event ML (misses coordination) 
        -> 2 COORDINATION/GRAPH (catches rings; batch; needs review) -> 3 LAYERED system [default]

ADVERSARIAL OPS: shadow-restrict + BATCH takedowns -> don't tip off boundary. tradeoff: time-to-detect vs reveal-logic
A/B: interference + adversary contamination -> SHADOW MODE / higher-unit randomization; guardrail = APPEAL-OVERTURN rate

OFFLINE!=ONLINE: adversarial DRIFT -> label pollution/leakage -> PU mishandled -> coordination invisible per-acct -> cohort skew -> feedback loop
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 100M+ users?** Coordination/graph jobs become the cost center (shard by community, incremental, prioritize high-risk regions, like file 16); inline scoring stays cheap via reputation feature stores; async logging; tight signup-path SLOs.
- **How would you handle cold start (new account)?** Reputation/infra + behavioral-biometric signals at signup; graduate risk as behavior accrues; soft friction on uncertainty rather than ban.
- **How do you pick thresholds?** Per friction tier by cost/cohort on the recall-vs-FPR curve; conservative (favor friction over ban) where false-takedown cost is high; require calibration and corroboration for hard actions.
- **Offline up, online down?** §7.2 list: adversarial drift, label pollution/leakage, PU mishandling, coordination invisible per-account, cohort skew, feedback loop.
- **How would you debug a false-takedown spike?** Freeze offending model/threshold, localize cohort/feature from logs, roll back, auto-reinstate via appeals, add to audited set.
- **How do you prevent feedback loops / keep unbiased labels?** Small no-action exploration slice, audited random holdout, weight labels by source reputation.
- **How do you avoid tipping off the adversary?** Shadow-restrict, batch takedowns, rotate detection logic, keep some signals secret; trade time-to-detection against revealing your boundary.
- **Fairness?** Slice FPR by cohort/geography; ensure friction doesn't disproportionately hit legitimate new or minority users; appeals as a safety valve.

---

## 13. Common mistakes

- Building only a per-account classifier and missing **coordinated** abuse (each puppet looks fine; the ring is the signal).
- Ignoring the signup cold-start adversary (no history) and the reputation/biometric signals + lifecycle risk graduation that address it.
- Binary block/allow instead of **graduated friction** matched to asymmetric cost.
- Trusting user reports as clean labels — ignoring that they're noisy and **adversarially gameable** (mass false-reporting).
- Treating unlabeled accounts as negatives instead of a **PU** problem; no audited holdout → no honest FPR/prevalence.
- Using accuracy/ROC-AUC under extreme imbalance instead of recall-at-fixed-FPR + prevalence + appeal-overturn guardrail.
- Acting on every detection instantly, training the adversary, instead of shadow-restrict / batch takedowns.
- Running an account-level A/B that the adversary contaminates and that violates the coordination/interference assumption.

---

## 14. Transfer: what this case unlocks

- **File 09 (fraud):** shared DNA — extreme imbalance, recall-at-fixed-FPR, delayed labels, random/audited holdout, cost-sensitive thresholds, adaptive adversary. Difference: fraud scores *individual transactions/money*; this scores *accounts and coordinated networks*.
- **File 16 (PYMK / graph):** the coordination layer *is* graph ML (Sybil detection, community detection); spam rings attack PYMK to harvest connections — sibling systems sharing graph infra.
- **File 13 (safety gateway):** both are adversarial defense-in-depth with a "raise the attacker's cost" mindset and fast emergency levers; file 13 is per-interaction/generative, this is per-account/behavioral.
- **File 10 (moderation):** sampled-prevalence north star, graduated enforcement, contested labels, severity-tiered action transfer directly.
- **File 20 (drift/monitoring):** adversarial concept drift is the extreme case of drift detection and triggered retraining.
- **General skill:** "find the signal the adversary can't afford to hide (coordination, expensive-to-forge attributes), act with graduated friction, and measure honestly under polluted labels" transfers to all abuse/integrity problems.

---

## 15. Sources

Original guides (kept):
- [IGotAnOffer ML System Design Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [Exponent ML System Design Interview Guide](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)
- [Hello Interview ML/System Design Learning](https://www.hellointerview.com/learn)
- [Designing Machine Learning Systems, Chip Huyen](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)

Added canonical references (verify titles; well-established works):
- [Cao et al., "Aiding the Detection of Fake Accounts in Large Scale Social Online Services (SybilRank)," NSDI 2012](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/cao)
- [Yu et al., "SybilGuard: Defending Against Sybil Attacks via Social Networks," SIGCOMM 2006](https://www.comp.nus.edu.sg/~yuhf/sybilguard-sigcomm06.pdf)
- [Beutel et al., "CopyCatch: Stopping Group Attacks by Spotting Lockstep Behavior in Social Networks," WWW 2013](https://dl.acm.org/doi/10.1145/2488388.2488400)
- [Jiang et al., "Inferring Strange Behavior from Connectivity Pattern in Social Networks (CatchSync)," PAKDD 2014](https://www.cs.cmu.edu/~neilshah/pubs/catchsync-pakdd2014.pdf)
- [Elkan & Noto, "Learning Classifiers from Only Positive and Unlabeled Data (PU learning)," KDD 2008](https://cseweb.ucsd.edu/~elkan/posonly.pdf)
- [Liu et al., "Isolation Forest" (unsupervised anomaly detection), ICDM 2008](https://ieeexplore.ieee.org/document/4781136)
