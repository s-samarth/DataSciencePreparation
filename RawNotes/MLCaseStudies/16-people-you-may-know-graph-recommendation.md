# 16. People You May Know / Graph Recommendation System

**Company tags:** Meta, LinkedIn, Google, Twitter/X, any social/professional network
**Interview frequency:** Medium-high
**Why it matters:** This case has three traps that other recommenders don't. First, you literally **cannot score all candidate pairs** — a billion users is 10^18 pairs — so candidate generation is a *graph-structure* problem, not an afterthought. Second, the binding constraint is not accuracy, it is **"creepiness"**: a suggestion that is correct but reveals a relationship the user wanted private (the infamous PYMK incidents) is a catastrophic product, PR, and legal failure. Third, the system **reshapes the very graph it learns from** — recommend the popular and they get more popular. If you frame it as "rank candidates by mutual-friend count," you've missed all three.

---

## 0. How to use this doc

Two passes.

**Pass 1 (intuition).** Read the prose and transcript. Lock three ideas: (1) the graph *gives you the candidate set for free* — the people you're likely to connect with are overwhelmingly your friends-of-friends, which collapses an impossible 10^18 problem into a tractable few-hundred-thousand per user; (2) **the objective is not max accuracy, it's max *acceptable* accuracy** — the source of a signal can leak a private relationship even when the prediction is "right," so signal-source governance and explainability are part of the model, not a filter bolted on after; (3) PYMK is a feedback loop on the graph itself — design against degree bias or the network homogenizes.

**Pass 2 (active recall).** Cover the page. Can you (a) explain why you generate candidates from 2-hop neighborhoods and what that costs, (b) name two link-prediction features beyond raw mutual-friend count and why degree-weighting matters, (c) explain the "accurate but creepy" failure with a concrete mechanism and the design response, and (d) explain why a *random* train/test split is wrong here and what you do instead? Those four are the case.

**The scaffold (shared across this set):**

> **Clarify → Frame → Data/Labels → Baseline → Model → Eval → Deploy → Monitor**

Bends here: "Model" is a two-stage **graph candidate-generation → ranking** funnel where stage one is the hard part; "Data/Labels" must confront temporal leakage and the endogenous, logged-bandit label; "Eval" and "Monitor" carry the privacy/abuse and degree-bias guardrails as first-class, not extras.

**The senior tell, stated once:** say early that "this is link prediction, but the candidate-generation step is the whole game at this scale, and the optimization target is *acceptable* accuracy under privacy constraints — a correct-but-creepy suggestion is a failure." That reframes it from "rank by mutual friends" to a graph-systems + responsible-ML problem.

---

## 1. Clarify (scripted, with *why each answer changes the design*)

| Question | Why it changes the design |
|---|---|
| **Social (symmetric friendship) or professional (asymmetric, intent-driven)?** | Facebook friendship is mutual and personal; LinkedIn connection is professional and recruiter-driven. Professional networks tolerate weaker-tie suggestions ("you both work in fintech"); social networks punish them as creepy. Sets the privacy bar and the feature set. |
| **What signal sources are we allowed to use?** | This is the most important question and most candidates skip it. Mutual friends are safe. **Contact-book imports, co-location, shadow profiles of non-users** are powerful *and* the source of every creepiness scandal. The allowed sources define both the model's power and its risk. |
| **Scale?** | 100M vs 1B+ users changes everything: at 1B you cannot score all pairs (10^18), so candidate generation must be graph-structural and largely precomputed offline. Assume 1B users, avg degree ~few hundred. |
| **What is a "good" suggestion — accepted, or accepted *and* leads to real interaction?** | Optimizing raw acceptance invites spammy, low-value, or coerced connections. The real target is accepted + *reciprocal meaningful interaction*. Defines the label. |
| **Latency / surface?** | PYMK is usually a module that loads in ~100–300ms, not an ultra-low-latency inline decision. Candidates can be precomputed in daily batch; only ranking is online. Relaxes the latency budget. |
| **Privacy/regulatory regime?** | GDPR/data-protection constraints on inferred edges and non-user data force hard rules on what can ever surface and require explainability ("why am I seeing this"). |

State assumptions and move: professional-ish network, 1B users, mutual-friends + profile + activity allowed, contact-import treated as sensitive/consented-only, target = accepted + reciprocal interaction, batch candidate gen + online ranking, GDPR-grade privacy.

---

## 2. Numbers up front (carry them through)

- **Scale:** ~1B users, average degree ~300–500 connections → on the order of **5×10^11 edges**. All-pairs scoring is **~10^18** — impossible; this single number justifies graph-structured candidate generation. Say it out loud.
- **Candidate-generation math:** a user's 2-hop neighborhood (friends-of-friends) with avg degree 400 is up to ~400×400 = 160K paths, deduping to perhaps tens of thousands of *distinct* FoF candidates. That is the candidate set per user — **five orders of magnitude smaller than all-pairs** and where the vast majority of real future edges live (triadic closure). Precompute it in daily batch.
- **Serving:** rank a few hundred to a few thousand candidates per user online, return top ~10–20 to display. Ranking latency budget < ~100ms (PYMK module, not inline-critical). Candidate refresh: daily batch for most, with a fast path for high-velocity new users (just signed up, imported contacts).
- **Storage back-of-envelope:** the edge list ~5×10^11 edges × ~16 bytes (two 8-byte IDs) ≈ 8TB raw, sharded across a distributed graph store; per-user precomputed candidate lists (top few hundred) ≈ 1B × few hundred × small record ≈ low tens of TB. Tractable with sharding; the compute (batch FoF traversal over a trillion-edge graph) is the expensive part, not storage.
- **Quality + guardrail targets:** maximize **connection acceptance that leads to reciprocal interaction**; **hard guardrails:** report/block rate, privacy complaints, sensitive-attribute or sensitive-relationship leakage (zero tolerance), spam/fake-connection rate. And a *health* guardrail: **degree-distribution / diversity** so the feedback loop doesn't concentrate edges on a popular few.

---

## 3. The conceptual spine: candidate generation is the game, and the objective is *acceptable* accuracy

### 3.1 You cannot score all pairs — the graph hands you the candidates
With 10^18 pairs, the only viable design is two-stage, and unlike item recsys where stage one is "an ANN over embeddings," here stage one is **graph traversal**: the people you are most likely to connect with are your **friends-of-friends** (triadic closure — if A–B and B–C are edges, A–C is far likelier than a random pair). So candidate generation = enumerate the 2-hop neighborhood (plus a few other generators: same employer/school cohort, imported-contact matches *if consented*, geographic/affinity clusters), precomputed in batch. This collapses the impossible problem into a few-hundred-thousand-per-user one. **This is the senior framing: at this scale, candidate generation *is* the system; ranking is the easy second stage.**

### 3.2 The objective is *acceptable* accuracy, not maximum accuracy (the creepiness axis)
This is the distinctive responsible-ML insight and the thing the interviewer is probing. A PYMK model can be *too* good in a way that backfires:
- The *source* of a correct prediction can reveal a relationship the user wanted private. Classic real-world mechanisms: contact-book imports stitch together people who never connected online (a patient and their therapist, two people from a support group, an affair, a sex worker and a client); shadow profiles built from non-users' data; co-location from device signals. The prediction is "accurate" — they *do* know each other — but surfacing it **exposes a private relationship**, which is a product failure, a PR disaster, and in many regimes illegal.
- Therefore: **governance of signal sources is part of the model design, not a downstream filter.** Some signals (mutual friends, shared explicit affiliations) are safe to *use and to explain*; others (contact imports, co-location, sensitive-attribute correlations) are restricted or excluded outright. And every suggestion should be **explainable** ("you both know X," "you both worked at Y") — if you cannot give a non-creepy reason, don't show it. Explainability is a privacy mechanism here, not just UX.

Say it directly: "I optimize acceptance *subject to* a hard privacy/creepiness constraint, and I treat which signals I'm allowed to use — and to explain — as a first-class design decision."

### 3.3 PYMK reshapes its own graph (the feedback loop)
PYMK doesn't just predict the graph; it *creates* it. Recommend high-degree (popular) nodes and they gain edges, get recommended more — rich-get-richer / preferential attachment. Left unchecked this homogenizes the network, buries new and low-degree users, and biases your training labels (you only see acceptances for pairs you showed). Design responses: degree-normalize features and exposure, reserve exploration for low-degree/new users, monitor the degree distribution as a health metric, and use logged-bandit/exploration data to debias the labels.

---

## 4. The data/label problem for *this* domain: temporal leakage + endogenous labels

Every case has a signature data problem. Graph link-prediction has two:

1. **Temporal leakage — a *random* train/test split is flat wrong.** The label is "a future edge forms." If you split edges randomly, you train on edges that formed *after* the ones you're predicting, leaking the future into the past — and graph features (mutual-friend counts) computed on the full graph already encode the edges you're trying to predict. You will get spectacular offline numbers and a broken model. **The correct setup: snapshot the graph at time *t*, compute features from the *t*-graph only, and label by edges that form in (*t*, *t+Δ*].** Strict temporal split, point-in-time-correct features. This is the graph version of the train-serve-skew/leakage discipline from file 12, and stating it is a strong senior signal.

2. **Endogenous, logged-bandit labels.** You only observe acceptance for pairs you *recommended* (or that users found organically). The training data is selection-biased toward what the current system shows — you have almost no labels for pairs the system never surfaces, and "not accepted" is censored (they may never have seen it). This is the same logged-bandit / counterfactual problem as ranking (files 02/03): reserve **exploration traffic** to get unbiased data on under-shown candidates, and treat negatives carefully (a non-shown pair is not a true negative).

**Negatives / class imbalance:** real future edges are vanishingly rare among all candidate pairs, so sampling negatives matters — hard negatives (FoF that did *not* connect) teach more than random pairs.

**Cold start** (kept and sharpened): a brand-new user has no edges, so FoF generation produces nothing. Bridge with imported contacts (consented), onboarding (school/employer), profile-content similarity, and popularity priors *within* a safe cohort — then switch to graph signals as their first edges form.

---

## 5. The baseline → why-it-breaks → next-rung ladder

**Rung 0 — Mutual-connection heuristics.** Rank candidates by raw count of mutual friends (+ same school/employer rules).
- *Works:* shockingly strong baseline, transparent, trivially explainable ("you have 8 mutual connections"), no creepiness if sourced from explicit edges.
- *Breaks:* raw count over-weights high-degree hubs (a popular node is "mutual" with everyone), no personalization, can't use richer structure or profile signals. **Trigger:** the suggestions are dominated by celebrities/recruiters; precision plateaus.

**Rung 1 — Graph-feature GBDT ranker.** Engineer link-prediction features — common neighbors, **Adamic/Adar** and **resource-allocation** (degree-weighted common neighbors, so a shared *rare* connection counts more than a shared hub), Jaccard, preferential-attachment, graph distance, profile/affiliation similarity, interaction recency — and rank with a gradient-boosted tree.
- *Adds:* degree-aware structural signals, interpretable feature importances (good for the privacy/explainability story), strong practical accuracy.
- *Breaks:* hand-engineered features miss deep structural similarity; scales poorly to capturing multi-hop patterns; cold nodes still weak. **Trigger:** wanting structural similarity that handcrafted features can't express, or candidate generation beyond 2-hop.

**Rung 2 — Graph embeddings / GNN for candidate generation + GBDT/NN ranker (recommended production design).** Learn node embeddings (node2vec, or inductive **GraphSAGE / GNN** so new nodes get embeddings from their features+neighbors without full retraining); generate candidates by ANN over embeddings (captures structural/affinity similarity beyond strict 2-hop) *union* the FoF set; rank with a model over graph + embedding + profile + interaction features.
- *Adds:* captures latent structural similarity, scales candidate gen, handles inductive cold nodes.
- *Breaks/risks:* harder to explain (privacy story weakens — mitigate by always attaching a human-readable reason from explicit signals), can **encode and amplify bias** (homophily → filter bubbles), and embeddings can *learn* the sensitive correlations you wanted to exclude. **Trigger:** you've maxed structure and now need explicit privacy/abuse enforcement at scale.

**Rung 3 — Two-stage graph retrieval + dedicated safety/privacy ranker (the full design).** The above, plus a mandatory **privacy + abuse layer that runs before display**: block/mute lists, consent checks on signal sources, sensitive-relationship exclusion, spam/fake-account scoring, diversity/degree caps, and an explanation requirement. Mention as the production-complete version.

Meta-rule out loud: "Mutual-friend count is a strong rung-0 baseline; I climb to degree-weighted graph features and then GNN candidate generation for structural reach — but I keep an explicit, explainable privacy layer because at rung 2 the embeddings can silently learn exactly the sensitive correlations I'm trying to avoid."

---

## 6. The architecture explained to the floor

```text
[GRAPH STORE: ~5e11 edges, sharded]
        |
 (BATCH, daily) candidate gen:  2-hop FoF  ∪  embedding-ANN  ∪  cohort(school/employer)  ∪  consented contacts
        |  -> per-user candidate list (top few hundred), precomputed
        v
 (ONLINE) rank candidates: GBDT/NN over {degree-weighted graph feats, embeddings, profile sim, interaction}
        |
        v
 [PRIVACY + ABUSE LAYER  <-- runs BEFORE display, non-bypassable]
   consent/source check · block-mute lists · sensitive-relationship exclusion · spam/fake scoring
   · diversity & degree caps · EXPLANATION required ("you both know X")
        |
        v
   top ~10-20 shown  -> (feedback: accept / ignore / report) -> labels + exploration log
```

### 6.1 Candidate generation (the hard, distinctive stage)
Precomputed in **daily batch** over the distributed graph (FoF traversal is a join you cannot do online at 10^18 scale). Union of generators: 2-hop FoF (the workhorse), embedding-ANN (structural similarity beyond 2-hop), explicit cohorts (same employer/school), and consented contact-import matches. Cap and dedupe to a few hundred per user. A fast online path handles brand-new users (just imported contacts / just signed up) so they aren't stuck waiting for the nightly batch.

### 6.2 Ranking
Online over the precomputed candidate set. Features: **degree-weighted** structural signals (Adamic/Adar, resource allocation — the degree-weighting is what stops hubs from dominating), embedding similarity, profile/affiliation overlap, interaction recency, and a calibrated acceptance-probability head (calibration matters because downstream caps/diversity decisions use the probability). Blend predicted acceptance with predicted *reciprocal interaction* (the real objective), not raw acceptance.

### 6.3 The privacy + abuse layer (non-negotiable, runs before display)
This is what separates a real answer from a naive one. Before any suggestion is shown:
- **Consent/source check:** is every signal that produced this candidate one the user consented to? (e.g., contact-import-derived candidates only if the relevant party consented.)
- **Block/mute exclusion:** never suggest blocked/reported/muted users.
- **Sensitive-relationship exclusion:** suppress candidates whose only linkage runs through sensitive inferred signals (co-location, sensitive-attribute correlation) — the anti-creepiness rule.
- **Spam/fake-account scoring:** abuse models (cross-link file 17) keep fake/scraper accounts out of recommendations and stop spam networks from amplifying.
- **Diversity / degree caps:** limit over-suggesting hubs; reserve slots for new/low-degree users (feedback-loop control).
- **Explanation requirement:** each shown suggestion carries a human-readable, non-creepy reason. No acceptable reason → don't show.

### 6.4 The three paths, named
- **Serving path:** batch candidate gen → online ranking → privacy/abuse layer → display.
- **Data path:** the graph store, node embeddings, profile/affiliation data, consent/block records.
- **Feedback path:** accept/ignore/report signals + exploration log → labels (temporally correct), bias debiasing, retraining, and degree-distribution monitoring.

### 6.5 Costs
The expensive resource is the **batch graph compute** (trillion-edge FoF traversal + embedding training), not online serving. Mitigate with incremental candidate updates (only recompute users whose neighborhood changed), graph sharding by community, and caching. Online ranking over a few hundred candidates is cheap.

---

## 7. Evaluation

### 7.1 Metrics
- **Offline (temporal holdout — see §4.1):** Precision@K / Recall@K / NDCG@K against edges that actually formed in the held-out future window, **calibration** of acceptance probability, sliced by degree band (a model that's great for hubs and useless for new users is a failure), by tenure, and by cohort.
- **Online:** connection **acceptance rate**, and crucially **reciprocal-interaction rate** (accepted *and* led to real engagement) — the true objective; long-term network activity / retention.
- **Guardrails (first-class, can sink a launch even with great accuracy):** report/block rate on suggestions, privacy complaints, sensitive-relationship-leak incidents (zero tolerance), spam-connection rate, and **degree-distribution health / diversity** (feedback-loop control).

### 7.2 The offline↔online gap (the trap)
*"Precision@K jumped offline; online acceptance fell / reports rose — why?"* Causes, in order:
1. **Temporal leakage in the offline setup (§4.1)** — features or split peeked at the future, inflating offline. The #1 cause and unique to graphs.
2. **Endogenous-label / selection bias** — offline measured on pairs the *old* system showed; the new model surfaces different pairs you have no labels for.
3. **Optimized acceptance, not reciprocal interaction** — you boosted accepts with low-value/coerced connections that users later regret (and report).
4. **Network interference / SUTVA violation** — connections are *contagious*: helping user A connect changes B's graph and future suggestions, so a naive user-randomized A/B leaks treatment across the graph and mis-measures effect (see §7.3).
5. **Creepiness not in the offline metric** — Precision@K rewards being *right*, and the very-right suggestions are sometimes the creepy ones; the report-rate guardrail, not accuracy, catches this.
6. **Degree-bias feedback** — short-term accept lift comes from spamming hubs, which degrades long-term network health.

### 7.3 A fully-specified A/B test (with the graph-interference wrinkle)
- **Hypothesis:** ranker v4 raises reciprocal-interaction rate without raising report rate vs v3.
- **The interference problem (the senior point):** users are connected, so a standard user-level randomization violates SUTVA — treating A changes B's experience through the shared graph. **Use cluster/graph-cluster randomization** (assign whole communities/components to treatment or control) or ego-network randomization to contain spillover. State this explicitly; it's the graph analogue of the network-interference caveat in feed ranking (file 02).
- **Unit:** graph cluster. **Primary metric:** reciprocal-interaction rate. **Guardrails:** report/block rate (hard), privacy complaints, spam-connection rate, degree-distribution shift, latency.
- **Runtime/ramp:** shadow → canary cluster → ramp; ≥ several weeks because connection→interaction is a slow, delayed signal.
- **Rollback trigger:** report-rate spike, any privacy-leak incident, reciprocal-interaction drop, degree concentration.

---

## 8. Deployment, monitoring, fallback, incident response

- **Rollout:** shadow new candidate-gen/ranker (log suggestions without showing) to check distribution and privacy-filter pass rates; canary on graph clusters; ramp. Embedding-model refreshes and graph-snapshot updates roll out the same disciplined way.
- **Monitoring:** acceptance + reciprocal-interaction rate, **report/block rate per suggestion** (the creepiness alarm — a spike means a signal source or model change started surfacing bad suggestions), privacy-complaint stream, spam-connection rate, **degree distribution / diversity** (feedback-loop health), candidate-coverage for new/low-degree users, latency. Slice by degree band and tenure.
- **Fallback:** if the GNN/ranker is degraded, fall back to the rung-0/1 mutual-friends heuristic — safe, explainable, never creepy. Graceful degradation = less personalized, never higher-risk.
- **Incident response (privacy-first):** a privacy/creepiness incident is treated like a security incident — freeze, identify which signal source / model version produced the bad suggestions from traces, **kill the offending signal source immediately** (the minutes-fast lever, like a regex blocklist in the safety gateway), roll back, notify per policy, and add the case to the sensitive-exclusion rules. Because every suggestion logs its generating signal + explanation, forensics is tractable.
- **Abuse:** continuous monitoring for spam/scraper networks using PYMK to harvest connections; rate-limit, integrate file-17 bot/spam scores into the privacy/abuse layer.

---

## 9. Full one-hour interview transcript

**[0:00] INTERVIEWER:** Design a People You May Know system for a large social or professional network.

**[0:30] YOU:** Let me scope it first, because three things swing the design. Is it a social network with symmetric friendship or a professional one with asymmetric, intent-driven connections — because the privacy bar is very different. What signal sources am I allowed to use — mutual friends are safe, but contact-book imports, co-location, and shadow profiles are powerful *and* the source of every PYMK scandal. And what's the scale, because at a billion users I literally cannot score all pairs.

**[1:15] INTERVIEWER:** Professional-ish, a billion users. Mutual friends, profile, and activity are fine; treat contact imports as sensitive and consent-gated.

**[1:30] YOU:** Then let me put the number that drives the architecture on the board: a billion users is 10^18 pairs. I cannot score that. So candidate generation is the entire game, and the graph hands me the candidates for free — the people you're most likely to connect with are your friends-of-friends, by triadic closure. A 2-hop neighborhood at average degree ~400 is on the order of tens of thousands of distinct candidates per user after dedup — five orders of magnitude smaller than all-pairs, and that's where almost all real future edges live. I precompute that in daily batch and rank online. The senior framing is: at this scale candidate generation *is* the system, ranking is the easy second stage.

**[3:15] INTERVIEWER:** Okay, and how do you rank?

**[3:25] YOU:** I climb a ladder. Rung zero is raw mutual-friend count — a surprisingly strong, perfectly explainable baseline, but it over-weights hubs: a celebrity is "mutual" with everyone. So rung one is a GBDT over real link-prediction features, and the key is **degree-weighting** — Adamic/Adar and resource-allocation count a shared *rare* connection far more than a shared hub, which fixes the celebrity problem. Plus profile and affiliation overlap and interaction recency. Rung two, my production default, adds graph embeddings — GraphSAGE so new nodes get inductive embeddings from their features and neighbors — and generates extra candidates by ANN over those embeddings, capturing structural similarity beyond strict 2-hop, then ranks with a model over graph plus embedding plus profile features.

**[5:30] INTERVIEWER:** What's the biggest risk in this system?

**[5:40] YOU:** That the model is *too accurate* in a way that's creepy. This is the part I'd emphasize: my objective is not maximum accuracy, it's maximum *acceptable* accuracy. The source of a correct prediction can reveal a relationship the user wanted private — the classic incidents are contact-import or co-location signals stitching together a patient and their therapist, or people from a support group, or an affair. The prediction is "right," they do know each other, but surfacing it exposes a private relationship, and that's a PR disaster and often illegal. So I treat *which signals I'm allowed to use and to explain* as a first-class design decision, not a downstream filter. Mutual friends and explicit affiliations are safe to use and to explain; contact imports are consent-gated; co-location and sensitive-attribute correlations are excluded. And every suggestion must carry a non-creepy, human-readable reason — "you both know X." If I can't give an acceptable reason, I don't show it. Explainability is a privacy mechanism here, not just UX.

**[8:00] INTERVIEWER:** But your GNN embeddings might learn those sensitive correlations anyway.

**[8:10] YOU:** Exactly the danger, and it's why I keep an explicit privacy layer at rung three even with embeddings. Embeddings can silently reconstruct "people who are near each other a lot" or sensitive-attribute homophily. So before display, a non-bypassable privacy and abuse layer runs: consent and source checks, block and mute exclusion, sensitive-relationship suppression, spam and fake-account scoring, diversity and degree caps, and the explanation requirement. The embedding gives me reach; the privacy layer keeps the reach acceptable. And I'd audit the embeddings for whether they encode protected correlations.

**[10:00] INTERVIEWER:** Let's talk labels. How do you build training data?

**[10:10] YOU:** Two things, and the first is a classic trap. The label is "a future edge forms," so a *random* train/test split is flat wrong — it trains on edges that formed after the ones I'm predicting, and graph features computed on the full graph already encode the very edges I'm trying to predict. I'd get gorgeous offline numbers and a broken model. The correct setup is a strict **temporal split**: snapshot the graph at time *t*, compute features from the *t*-graph only, and label by edges that form in the next window. Point-in-time-correct features. Second, the labels are **endogenous** — I only observe acceptance for pairs the current system showed, so the data is selection-biased and "not accepted" is censored, since they may never have seen it. I reserve exploration traffic to get unbiased data on under-shown candidates, and I sample hard negatives — friends-of-friends who did *not* connect — rather than random pairs.

**[12:30] INTERVIEWER:** Suppose offline Precision@K jumps but online acceptance falls and reports rise. What happened?

**[12:40] YOU:** In order: first, temporal leakage in my offline setup inflated Precision@K — the most common and graph-specific cause. Second, endogenous-label bias — I measured offline on pairs the *old* system showed, and the new model surfaces different pairs I have no labels for. Third, I optimized acceptance rather than reciprocal interaction, so I boosted low-value or coerced connections users later regret and report. Fourth, creepiness isn't *in* Precision@K — being very right is sometimes being creepy, and only the report-rate guardrail catches that. Fifth, I might be getting short-term accepts by spamming hubs, which degrades long-term network health. The fix is to optimize reciprocal interaction, keep report-rate and degree-health as hard guardrails, and trust the online cluster test over offline.

**[15:00] INTERVIEWER:** You said cluster test — why not a normal A/B?

**[15:10] YOU:** Because connections are contagious, which violates SUTVA. If I treat user A and help them connect, I've changed B's graph and B's future suggestions — treatment leaks across the social graph, so a user-randomized A/B mis-measures the effect, usually understating it. So I use **graph-cluster randomization**: assign whole communities or components to treatment or control to contain spillover, or ego-network randomization. Unit is the cluster, primary metric is reciprocal-interaction rate, guardrails are report/block rate, privacy complaints, spam rate, and degree-distribution shift, and I run it for weeks because connection-to-interaction is a slow signal. Roll back on a report spike or any privacy incident.

**[17:30] INTERVIEWER:** How do you keep the system from making the rich richer?

**[17:40] YOU:** That's the feedback loop — PYMK creates the graph it learns from, so recommending hubs gives them more edges and more exposure, preferential attachment. I counter it three ways: degree-normalize both features and exposure so hubs don't dominate slots, reserve exploration capacity for new and low-degree users so they're discoverable, and monitor the degree distribution as a health metric — if it's concentrating, that's a regression even if acceptance looks fine. The exploration data also debiases my labels.

**[19:30] INTERVIEWER:** New user just signed up. What do you show them?

**[19:40] YOU:** They have no edges, so friend-of-friend generates nothing — cold start. I bridge with consented imported contacts, onboarding signals like school and employer, profile-content similarity, and popularity priors within a safe cohort, on a fast online path so they're not waiting for the nightly batch. As their first few edges form, I switch to graph signals. I'm careful that the contact-import path is consented and runs through the same privacy layer.

**[21:30] INTERVIEWER:** A privacy incident hits — bad suggestions are leaking a sensitive relationship. Walk me through it.

**[21:40] YOU:** I treat it like a security incident. Freeze, then use the traces — every suggestion logs its generating signal and its explanation — to identify which signal source or model version is producing the bad suggestions. Kill the offending signal source immediately; that's my minutes-fast lever, the same idea as a regex blocklist in the safety gateway. Roll back the model, notify per policy and regulation, add the pattern to the sensitive-exclusion rules so it can't recur, and post-mortem. Fallback while fixing is the rung-zero mutual-friends heuristic — safe and explainable.

**[23:30] INTERVIEWER:** Wrap up.

**[23:40] YOU:** To close: this is link prediction, but at a billion users candidate generation *is* the system — friends-of-friends plus embedding ANN, precomputed in batch — and the objective is *acceptable* accuracy, not maximum accuracy, because a correct-but-creepy suggestion is a failure. So signal-source governance and explainability are first-class, a non-bypassable privacy and abuse layer runs before display, I use strict temporal splits to avoid future leakage, optimize reciprocal interaction over raw acceptance, randomize A/Bs at the graph-cluster level to handle interference, and watch the degree distribution to keep my own feedback loop from homogenizing the network. I'd lean on ranking and evaluation language here and keep any enterprise-relationship-graph story light.

### Why this transcript works
- **Leads with the 10^18 number** to justify graph-structured candidate generation as the core, not an afterthought.
- **Owns the "acceptable accuracy" / creepiness axis** with concrete mechanisms and a design response (signal governance + explainability as privacy).
- **Catches that embeddings can re-learn sensitive correlations** — why the explicit privacy layer survives the model upgrade.
- **Nails the temporal-split leakage trap** and the endogenous-label/exploration problem.
- **Uses graph-cluster randomization** for the network-interference A/B — a sharp senior detail.
- **Designs against the degree-bias feedback loop** as a health metric.
- **Closes honestly** in ranking/eval language without overclaiming domain experience.

---

## 10. Junior vs senior contrast

| Dimension | Junior | Senior |
|---|---|---|
| Core | "Rank candidates by mutual friends." | "10^18 pairs → candidate generation (FoF + embedding ANN) *is* the system; ranking is stage two." |
| Objective | "Maximize acceptance/accuracy." | "Maximize *acceptable* accuracy — correct-but-creepy is a failure." |
| Features | "Count mutual friends." | Degree-weighted (Adamic/Adar, resource allocation); embeddings; calibrated acceptance. |
| Privacy | "Add a filter at the end." | Signal-source governance + explainability *as* the privacy mechanism; non-bypassable layer before display. |
| Embeddings risk | "GNN = better." | GNN can silently re-learn sensitive correlations → keep explicit exclusion + audit. |
| Labels | "Predict edges, split data." | Strict **temporal split** (no future leakage); endogenous/logged-bandit labels → exploration + hard negatives. |
| A/B | "Randomize users." | **Graph-cluster randomization** for network interference (SUTVA). |
| Feedback loop | unaware | Degree-bias / rich-get-richer → degree-normalize, explore for new users, monitor degree distribution. |

---

## 11. One-page whiteboard cheat sheet

```text
SPINE: 1B users = 10^18 pairs -> CAN'T score all -> CANDIDATE GEN (graph) IS the system. Rank = stage 2.
       objective = ACCEPTABLE accuracy (correct-but-creepy = FAILURE)

NUMBERS: 1B users, deg ~400 -> ~5e11 edges; 2-hop FoF ~tens of K candidates/user (precompute BATCH)
         rank few hundred online <100ms; show ~10-20

CANDIDATE GEN: 2-hop FoF (triadic closure)  ∪  embedding-ANN  ∪  cohort(school/employer)  ∪  consented contacts

RANK FEATURES: DEGREE-WEIGHTED (Adamic/Adar, resource alloc) > raw mutual count; embeddings; profile; recency
               objective = reciprocal INTERACTION, not raw acceptance; calibrated p(accept)

CREEPINESS (the axis): signal SOURCE can leak private relationship even when prediction is right
  (contact imports, co-location, shadow profiles) -> SCANDALS
  FIX: govern signal sources (use+explain only safe ones); EXPLANATION required ("you both know X"); 
       no acceptable reason -> don't show. Explainability = privacy mechanism.

PRIVACY+ABUSE LAYER (before display, non-bypassable): consent/source · block/mute · sensitive-rel exclusion
       · spam/fake scoring · diversity+degree caps · explanation

LABELS: TEMPORAL split (snapshot @t, features from t-graph, label edges in (t,t+Δ]) -- random split LEAKS future
        endogenous/logged-bandit -> exploration traffic + hard negatives (FoF that didn't connect)

A/B: connections contagious -> SUTVA violated -> GRAPH-CLUSTER randomization; metric=reciprocal interaction; weeks
GUARDRAILS: report/block rate, privacy complaints (0), spam rate, DEGREE-DISTRIBUTION health (feedback loop)

OFFLINE!=ONLINE: temporal leakage -> endogenous bias -> optimized accept not interaction -> creepiness not in P@K -> degree-bias
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 1B+ users?** Candidate gen must be batch + incremental (only recompute changed neighborhoods), graph sharded by community, embeddings trained distributed; online stays cheap (rank a few hundred). Storage ~tens of TB; compute is the cost.
- **How would you handle cold start?** Consented contacts + onboarding (school/employer) + content similarity + cohort popularity on a fast online path; switch to graph signals as first edges form — all through the privacy layer.
- **How do you pick K / thresholds?** By acceptance-vs-report tradeoff and slot diversity; cap hub exposure; require an explanation per slot; tune online by cluster A/B.
- **Offline up, online down?** §7.2 list: temporal leakage, endogenous-label bias, optimizing acceptance not interaction, creepiness invisible to Precision@K, degree-bias.
- **How would you debug a bad/creepy launch?** Traces log generating signal + explanation per suggestion; identify offending source/version, kill the signal source (fast lever), roll back to mutual-friends fallback, add to sensitive-exclusion rules.
- **How do you prevent feedback loops?** Degree-normalize features/exposure, reserve exploration for new/low-degree users, monitor degree distribution; use exploration logs to debias labels.
- **Fairness?** Slice accuracy by degree band, tenure, and protected cohorts; audit embeddings for sensitive-attribute encoding; ensure new/minority users get discoverable.

---

## 13. Common mistakes

- Treating candidate generation as trivial — not realizing all-pairs (10^18) is impossible and the graph (FoF) *is* the candidate mechanism.
- Ranking by raw mutual-friend count without degree-weighting, so hubs/celebrities dominate.
- Treating privacy as a final filter instead of governing *which signals you use and explain* — and ignoring that embeddings can re-learn sensitive correlations.
- Optimizing raw acceptance instead of reciprocal meaningful interaction, inviting spam/coerced connections.
- Using a random train/test split → leaking future edges → great offline, broken online.
- Ignoring endogenous/logged-bandit labels and treating non-shown pairs as true negatives.
- Running a user-randomized A/B on a contagious graph (SUTVA violation) instead of cluster randomization.
- Ignoring the degree-bias feedback loop that homogenizes the network and buries new users.

---

## 14. Transfer: what this case unlocks

- **Files 02 / 03 (feed & search ranking):** the two-stage retrieve→rank funnel, logged-bandit labels, exploration, and network-interference A/B caveats are shared; PYMK is the *graph* instance.
- **File 17 (spam/bot detection):** the abuse/spam-scoring layer here *is* file 17's output; coordinated fake-account networks attack PYMK to harvest connections — sibling problems.
- **File 12 (deployment platform):** the temporal-split / point-in-time-correct feature discipline is the same leakage-prevention as the feature store.
- **File 11 (experimentation):** cluster randomization, delayed signals, and guardrail-aware ramps are experiment-design transfers.
- **General skill:** "when accuracy and acceptability diverge, optimize acceptability and make the constraint a first-class part of the model" transfers to every responsible-ML system (moderation file 10, safety file 13).

---

## 15. Sources

Original guides (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research: TF-Ranking: https://research.google/blog/advances-in-tf-ranking/
- Stanford CS224W graph ML course: https://web.stanford.edu/class/cs224w/

Added canonical references (verify titles; well-established works):
- Hamilton et al., "Inductive Representation Learning on Large Graphs (GraphSAGE)," NeurIPS 2017: https://arxiv.org/abs/1706.02216
- Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks," KDD 2016: https://arxiv.org/abs/1607.00653
- Liben-Nowell & Kleinberg, "The Link Prediction Problem for Social Networks," 2003: https://www.cs.cornell.edu/home/kleinber/link-pred.pdf
- Adamic & Adar, "Friends and Neighbors on the Web," 2003 (Adamic/Adar index): https://www.sciencedirect.com/science/article/abs/pii/S0378873303000091
- Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage)," KDD 2018: https://arxiv.org/abs/1806.01973
- Ugander et al. / Eckles et al., "Design and Analysis of Experiments in Networks" (graph-cluster randomization, interference): https://arxiv.org/abs/1404.7530
