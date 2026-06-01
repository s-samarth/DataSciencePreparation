# 12. ML Model Deployment Platform

**Company tags:** Meta, Amazon, Google, Microsoft, Stripe, Uber, any company with >20 ML teams
**Interview frequency:** Medium-high (and rising as "MLOps platform" roles grow)
**Why it matters:** This is the case that separates people who *train* models from people who *operate* them. The hard question is not "how do you serve a model" (file 08 covers that). It is: how do you let fifty teams ship hundreds of learned artifacts continuously, every day, without any one of them silently corrupting production, when the thing being shipped is a statistical object whose behavior you cannot fully verify before it touches real traffic?

---

## 0. How to use this doc

This is a teaching document, not a checklist. Read it in two passes.

**Pass 1 (intuition).** Read the prose and the transcript. Internalize the single load-bearing idea: a model is not code. Code is deterministic and you can unit-test correctness. A model is a fit to a data distribution, and "correct" is undefined except relative to live data you have not seen yet. Every piece of the platform exists to manage that one uncomfortable fact. If you understand *that*, you can re-derive every gate, every rollout stage, every monitor from first principles.

**Pass 2 (active recall).** Cover the page. From the prompt alone, can you (a) derive the deploy throughput and rollback-latency numbers, (b) draw the registry → shadow → canary → ramp ladder and say what breaks if you skip a rung, (c) explain train-serve skew and the *one* infra fix that kills 80% of it, and (d) name the rollback trigger and how fast it fires? If you can whiteboard those four, you can hold this case for an hour.

**The scaffold (every case in this set uses it):**

> **Clarify → Frame → Data/Labels → Baseline → Model → Eval → Deploy → Monitor**

For a platform case the scaffold bends: "Model" becomes "platform architecture," "Data/Labels" becomes "what the platform stores about every model version," and "Eval" splits into *eval of the platform* (does it catch bad releases?) and *eval the platform runs for tenants* (does each model version pass its gates?). Keep the bones; bend the joints.

**A senior tell, stated once:** platform cases are *interface* cases. The interviewer is listening for "here is the contract the model owner signs, and here is what the platform guarantees in return." If you only describe boxes and arrows you sound like an infra tourist. If you describe the contract you sound like someone who has been paged at 3am because a teammate shipped a model trained on a leaked feature.

---

## 1. Clarify (the scripted opening, with *why each answer changes the design*)

Do not start drawing. Spend the first three minutes pinning down the problem, because "deployment platform" can mean five different systems. Ask these, and say out loud why each matters:

| Question | Why it changes the design |
|---|---|
| **Batch, online, or both?** | Batch scoring (nightly fraud scores over 2B rows) and online serving (sub-100ms ranking) have totally different rollout, rollback, and skew problems. Both is the realistic answer and the harder one. I will design for both and call out where they diverge. |
| **How many teams and models?** | One team needs a deploy script. Fifty teams need *multi-tenancy, isolation, and self-service* — the platform's whole reason to exist is that you cannot review every release by hand. I will assume ~50 teams, ~200 models in production. |
| **What is the blast radius of a bad model?** | A typo recommender that shows worse ads loses money slowly. A credit-decisioning model that goes haywire creates legal liability in minutes. The gate strictness must be *risk-tiered*, not uniform. |
| **Who owns a model in production — the team or the platform?** | This is the cultural question that decides whether the platform is a paved road or a bottleneck. Answer: the team owns correctness and the on-call; the platform owns reproducibility, rollout mechanics, and the kill switch. |
| **Do we control the training side or only deployment?** | If the platform also owns training/feature pipelines, we can kill train-serve skew at the source with a shared feature store. If we only own deploy, skew becomes a contract we must *validate* rather than *prevent*. I will assume we own enough to offer a feature store. |
| **Regulatory constraints?** | Finance/health need full lineage, audit, and reproducibility of any decision ever made. That turns the registry from a convenience into a legal requirement and forces immutable artifacts. |

If the interviewer is vague, *state your assumptions and move*. Senior candidates de-risk ambiguity by choosing; they do not stall waiting for permission.

---

## 2. Numbers up front (carry these through the whole answer)

Pin the scale before architecture so every later decision has something to push against.

- **Tenancy:** ~50 ML teams, ~200 models in production, mix of batch + online.
- **Deploy velocity (the real target):** ~50 model-version deploys per *day* across the org at steady state. This is the number that kills manual review. 50 deploys/day × even 30 min of human gate review = 25 engineer-hours/day of pure gatekeeping. The platform exists to make the common case need *zero* human gate.
- **Online serving:** aggregate ~500K predictions/sec across all online models; per-model SLOs vary, typical user-facing p99 budget 50–100ms end to end.
- **Batch serving:** nightly jobs scoring 10^8–10^9 rows; SLA is "done before the 6am business process reads it," not latency per row.
- **Feature store online reads:** if a ranking model pulls 200 features per request at 500K req/s, that is 10^8 feature lookups/sec — the feature store, not the model, is often the latency and cost bottleneck. p99 read < 10ms is the budget.
- **Rollback discipline (the headline safety number):** **time-to-detect a bad release < 10 min, time-to-roll-back < 1 min, both automatable.** A platform whose rollback is a human running a script during an incident is not a safe platform. I will design so rollback is "repoint a pointer," not "redeploy."
- **Storage back-of-envelope for the registry:** every model version is an immutable bundle = weights (100MB–10GB) + a *reference* to the exact training data snapshot (not a copy) + feature schema + training code commit + container image digest + eval report. Say 1GB average × 200 models × keep last ~20 versions each = ~4TB of artifacts, trivial; the expensive part is the data snapshots, which is why we store *pointers + content hashes*, not copies.

**Latency budget derived out loud (online path):** user-facing p99 = 80ms. Network + gateway ~10ms. Post-processing/business logic ~10ms. That leaves ~60ms for {feature fetch + model forward}. Feature fetch of 200 features at p99 10ms, model forward must fit in ~50ms. That budget is what tells you whether you can afford a cross-encoder or must stay with a GBDT — *the platform must surface this budget to the tenant before they ship*, not after they page on-call.

---

## 3. Why a model is not code (the conceptual spine — do not skip this in the interview)

State this explicitly; it is the senior signal that reframes the entire problem.

Normal software CI/CD rests on one assumption: **you can write a test that asserts correct behavior before you ship.** `assert add(2,2) == 4`. If the test passes, the function is correct, forever, for that input.

A model breaks every word of that sentence:
- **Correctness is undefined in isolation.** A model's output is only "right" relative to a data distribution. `model.predict(x) == ?` has no ground-truth answer you can hard-code.
- **Behavior is data-dependent and shifts under your feet.** The same model is good in January and bad in March because the world moved (drift — file 20 owns this). Your test suite from January still passes.
- **The failure mode is silent.** Broken code throws. A broken model returns a confident, well-typed, plausible number that is wrong. No exception, no stack trace. It just quietly loses money or harms users until a *metric* (not an error) reveals it.
- **The artifact is huge and opaque.** You cannot read a diff of 10GB of weights and reason about it. Code review does not transfer.

**Consequence for the platform:** since you cannot *prove* a model good before traffic, the platform's job is to **expose every new model to reality gradually and cheaply, measure it against the current production model, and make undo instant.** That is the whole philosophy. Offline eval is necessary but never sufficient. The rollout ladder is not bureaucracy — it is the only way to "test" a thing whose correctness is only knowable in production.

Everything below is a corollary of this paragraph.

---

## 4. The data/label problem for *this* domain: train-serve skew and reproducibility

Every case in this set has a signature data problem. Fraud has censored labels. Moderation has contested labels. **A deployment platform's signature data problem is train-serve skew**: the model sees different inputs in production than it saw in training, so it silently underperforms even though nothing "broke."

Skew has three concrete sources, and a senior answer names the *mechanism* for each:

1. **Feature computation skew.** Training computes `avg_purchase_last_30d` with a batch Spark job; serving computes it with a streaming Python service. Subtle differences (timezone, null handling, rounding, a different window boundary) mean the *same user* gets different feature values offline vs online. The model trained on the offline version degrades on the online version. **This is the #1 cause of "offline up, online down."**
2. **Point-in-time leakage.** During training you accidentally join a feature that was computed *after* the label event (e.g., "total lifetime value" includes the very transaction you are predicting). Offline metrics look spectacular; the feature does not exist at serving time, so production is garbage. This is the most embarrassing failure mode and it is *invisible* to standard offline eval because the leak is in both train and validation splits.
3. **Distribution skew over time.** Even with identical computation, the live distribution drifts from the training snapshot. (Owned by file 20; here we just detect it.)

**The one infra fix that kills ~80% of skew: a shared feature store with online/offline parity.** The platform offers a feature store where:
- A feature is *defined once* (one transformation), and the store guarantees the **same code path** materializes it for both offline training tables and online low-latency reads. Same code = no computation skew (source 1 gone).
- The store enforces **point-in-time-correct joins**: when you build a training set, it gives you the feature value *as of the label timestamp*, never later. This structurally prevents leakage (source 2 gone). This is the single most valuable thing the platform offers and the detail that proves you have built one.
- Online reads are a low-latency KV lookup (Redis/DynamoDB/Feast-style) at p99 < 10ms.

Canonical references: Uber Michelangelo (the original ML platform paper, Hermann & Del Balso 2017), Feast feature store, Google TFX. Cite these by name; do not invent URLs.

**What the platform stores about every model version (the reproducibility contract):** an immutable bundle keyed by content hash —
- model weights (artifact),
- **pointer + hash of the exact training data snapshot** (so the run is reproducible and auditable; pointer not copy),
- feature schema + feature store feature IDs + versions,
- training code git commit + container image digest,
- hyperparameters + random seeds,
- the eval report (metrics, slices, fairness),
- owner + on-call + risk tier.

If you cannot reproduce a model bit-for-bit from its registry entry, you cannot debug it, audit it, or trust your rollback. Immutability is non-negotiable: you never overwrite a version, you only add new ones, because "what exactly was serving when the incident happened" must always be answerable.

---

## 5. The baseline → why-it-breaks → next-rung ladder

Build the platform the way you would build any system: start embarrassingly simple, then add a rung *only* when you can name the specific thing that broke. This is the structure the interviewer most wants to see.

**Rung 0 — Manual deploy scripts.** Each team SSHes a pickle file onto a box and restarts a service.
- *Works when:* one team, one model, low stakes.
- *Breaks when:* you cannot reproduce what is running (no registry), there is no rollback except "find the old pickle," and a bad model is live for hours before anyone notices. Skew is invisible. **Trigger to climb:** the second team, or the first incident where nobody could say which model version caused it.

**Rung 1 — Model registry + reproducible artifacts.** Every model is registered as the immutable bundle from §4. Deploys reference a version ID.
- *Adds:* reproducibility, audit, "what is running right now" is answerable, rollback = repoint to previous version ID.
- *Breaks when:* you can roll back, but you still find out a model is bad from a *product* dashboard hours later, and you still have train-serve skew because features are computed two different ways. **Trigger:** an "offline-up/online-down" incident traced to feature skew.

**Rung 2 — Feature store + offline eval gate.** Shared feature store kills computation skew and leakage; a CI gate runs offline eval on a held-out set and blocks deploy if the primary metric regresses.
- *Adds:* skew prevention, automated quality floor, leakage prevention via point-in-time joins.
- *Breaks when:* offline eval passes but the model is still bad online (novelty effects, feedback loops, a slice that is not in your offline set, a latency regression that offline eval never measures). Offline green is necessary, not sufficient (§3). **Trigger:** a model that passed all offline gates and still tanked a live metric.

**Rung 3 — Shadow + canary + ramp + auto-rollback (the recommended production design).** Expose the model to reality gradually and undo instantly. This is the heart of the answer; full mechanics in §6.
- *Adds:* the only real test of a model — production traffic, measured against the incumbent, at controlled blast radius, with automatic rollback.
- *Breaks when:* nothing, for most orgs. This is where you stop.

**Rung 4 — Full policy-gated ML CI/CD with risk tiers.** Mention as an *extension*, not the start. High-risk models (credit, health, safety) get mandatory human sign-off, fairness gates, and explainability checks; low-risk models get the fully automated paved road. The point is *graduated* gates so the platform speeds up the 90% safe case instead of taxing it.

State the meta-rule out loud: "I start at the registry and feature store because those buy the most safety per unit of effort, and I climb to shadow/canary because offline eval can never certify a model. I would only add mandatory human gates for the high-blast-radius tier."

---

## 6. The architecture explained to the floor: shadow → canary → ramp → rollback

This is the one mechanism to explain in real depth. Walk the interviewer through what *actually happens to a request* at each stage.

### 6.1 Shadow mode (mirror traffic, zero user impact)
The new candidate model is deployed alongside the production model. **Live requests are mirrored** to the candidate: it scores the real input but **its output is logged and discarded, never returned to the user.** No side effects, no user impact, real-world inputs.

What you learn that offline eval cannot tell you:
- **Operational reality:** does it actually meet the latency budget under real load and real feature distributions? p99 forward-pass time, memory, throughput. Offline eval on a laptop never measures this.
- **Prediction-distribution diff:** compare the candidate's output distribution to production's on identical inputs. A big shift is a red flag *before* any user sees it.
- **Skew detection:** because the candidate scores *live* features, you can diff "feature values it saw in shadow" vs "feature values it saw in training." If they differ, you have caught skew with zero blast radius.

Shadow has one blind spot: it cannot measure *outcomes* (clicks, conversions, fraud caught) because its predictions never affect anything. For that you need real traffic — the next rung.

### 6.2 Canary (small slice of real traffic, real outcomes)
Route a small fraction — start ~1%, by user/entity hash so it is sticky and consistent — to the candidate. Its outputs are *returned* now, so you finally measure real outcomes against guardrails: business KPI, latency, error rate. Blast radius is bounded to 1%. Automated statistical comparison vs the control slice; if guardrails breach, auto-rollback (below) fires before the ramp.

### 6.3 A/B + ramp
Once the canary is clean, widen to a proper A/B (file 11 owns experiment design; here the platform just runs it: hypothesis, control = incumbent model, treatment = candidate, primary metric + guardrails, minimum runtime for power, sticky assignment). On a clean read, ramp 1% → 5% → 25% → 50% → 100% with a hold at each step and automated guardrail checks between steps.

### 6.4 Rollback (the part people forget — make it instant and automatic)
**Rollback must be a pointer flip, not a redeploy.** Production traffic routes through a router that maps `model_name → active_version_id`. Rollback = set the pointer back to the previous version, which is still warm and loaded. Sub-second. Because the previous artifact is immutable in the registry and ideally still resident, there is nothing to rebuild.

**Automated rollback triggers** (this is the 15-point-bar detail):
- guardrail metric breaches a pre-registered threshold (error rate, latency p99, business KPI drop beyond bound),
- prediction distribution diverges beyond a set KL/PSI threshold from the incumbent,
- a hard error spike or NaN/null-prediction rate above floor.

The candidate is *not* deleted on rollback; it is parked so you can debug from the immutable bundle (§4). This closes the loop with the rollback numbers from §2: detect < 10 min (monitors on a tight window), roll back < 1 min (pointer flip).

### 6.5 The three deployment paths, named separately (senior framing)
Always decompose the running system into three independent paths so you sound like an operator:
- **Serving path:** router → feature fetch → model forward → post-process. Versioned, canaried, instantly rollback-able.
- **Data path:** feature store materialization (batch + streaming) with online/offline parity; this is where skew lives or dies.
- **Feedback path:** logging predictions + delayed labels back to the warehouse to (a) compute online metrics, (b) build the next training set, (c) detect drift. Without this path you are flying blind and cannot retrain (file 20).

### 6.6 Costs
The expensive resources are: GPU/CPU for online forward passes (mitigate with autoscaling on the canary/ramp, scale-to-zero for idle models, shared model servers for small models), feature store reads (cache hot features, batch lookups), and shadow mode (you are now paying to run two models — bound shadow duration and sample traffic if cost-sensitive).

---

## 7. Evaluation: of the tenant's model *and* of the platform itself

Two distinct eval questions; senior candidates separate them.

### 7.1 Eval the platform runs *for each model version* (the gates)
- **Offline gate:** primary metric must not regress beyond a bound vs the incumbent, on a frozen held-out set, **always sliced by important cohorts** (per-region, per-segment, high-value users). An aggregate win that hides a segment loss must block.
- **Leakage check:** point-in-time correctness enforced by the feature store; flag any feature whose importance is suspiciously high (a classic leakage smell).
- **Operational gate (from shadow):** p99 latency, memory, throughput under live load within budget.
- **Fairness/safety gate (risk-tiered):** cohort parity for high-stakes models.

### 7.2 The offline↔online gap — the trap, with enumerated causes
The interviewer will ask: *offline metrics improved but online dropped — why?* For a deployment platform this is THE diagnostic skill. Enumerated causes, in the order you should check them:
1. **Train-serve feature skew** (§4 source 1) — most common; diff shadow features vs training features.
2. **Point-in-time leakage** (§4 source 2) — offline was inflated by a feature unavailable at serving; check feature availability at request time.
3. **Latency-induced quality loss** — the model is "better" but slower, so it times out and falls back, or hurts a latency-sensitive KPI. Offline eval never measured latency.
4. **Distribution shift** between the frozen offline set and live traffic (novelty, seasonality).
5. **Feedback loops / interference** — the new model changes what data you collect (a recommender that shows different items changes future clicks), so offline replay on logged data is biased.
6. **Wrong offline proxy** — offline optimized AUC, the business cares about revenue at a threshold; they decorrelate.

The cure is structural: shadow catches 1–3 before launch; A/B catches 4–6 because it measures reality. *This is exactly why the rollout ladder exists* — say that.

### 7.3 A fully-specified A/B test (the platform runs these as a service)
- **Hypothesis:** candidate model v37 increases conversion vs incumbent v36 with no latency or error regression.
- **Unit + assignment:** user-hash, sticky, 50/50 after ramp.
- **Primary metric:** conversion rate. **Guardrails:** p99 latency, error rate, revenue-per-session, a fairness slice.
- **Power/runtime:** size for the minimum detectable effect; run ≥ the business cycle (≥1 week to cover weekday/weekend), do not peek-and-stop.
- **Rollout:** 1% canary → ramp with holds → 50/50.
- **Rollback trigger:** any guardrail breaches its pre-registered bound, or primary metric drops with significance.
- (CUPED for variance reduction if pre-period data exists — cuts required runtime.)

---

## 8. Deployment, monitoring, fallback, incident response

- **Rollout discipline:** §6 ladder. Never skip shadow for a model whose latency you have not measured under load. Never skip canary for anything user-facing.
- **Monitoring (three layers):** (1) *operational* — latency, throughput, error rate, saturation, per model version; (2) *statistical* — input feature drift, prediction drift (PSI/KL vs a reference window), null/NaN rates, feature freshness/staleness; (3) *outcome* — delayed-label performance, business KPI, sliced by cohort. Alerts wired to auto-rollback for the fast signals (§6.4).
- **Fallback:** every online model declares a fallback — previous version, a simple rule, a cached default, or "abstain and route to a safe path." A model server returning 500 must degrade gracefully, never hard-fail the product.
- **Incident response runbook:** freeze (pin current version, halt any in-flight ramp) → identify (which version, which slice, from the immutable registry bundle + traces) → roll back (pointer flip) → root-cause offline against the parked artifact → add a regression test/gate so it cannot recur. The registry's immutability is what makes "what was running" answerable during the incident.
- **Multi-tenancy/isolation:** noisy-neighbor protection (one team's batch job must not starve another's online serving), per-tenant quotas, and the risk-tiered gates from rung 4 so the platform is a paved road for the safe 90% and a guardrail for the dangerous 10%.

---

## 9. Full one-hour interview transcript

*(INTERVIEWER / YOU. This is what "good" sounds like end to end.)*

**[0:00] INTERVIEWER:** Design a platform that lets ML teams train, register, deploy, monitor, and roll back models.

**[0:30] YOU:** Before I draw anything, let me scope it, because "deployment platform" can mean five systems. Four questions. First — batch, online, or both? Second — how many teams and models, because one team needs a script and fifty teams need multi-tenancy and self-service. Third — what is the blast radius of a worst-case bad model, because that sets how strict the gates are. Fourth — does the platform own the training/feature side too, or only deployment? That decides whether I can prevent train-serve skew at the source or only validate against it.

**[1:30] INTERVIEWER:** Both batch and online. ~50 teams, ~200 production models. Blast radius varies — some models touch credit decisions, most touch ranking. Assume you own enough to offer a feature store.

**[2:00] YOU:** Good, that is the realistic and harder version. Let me put numbers down. ~50 deploys a day across the org at steady state — that single number is why we cannot human-review every release; 50 deploys times 30 minutes of review is 25 engineer-hours a day of pure gatekeeping. So the design goal is: the common-case deploy needs *zero* humans, and only the high-blast-radius tier gets mandatory sign-off. Online serving aggregate ~500K predictions a second; user-facing p99 budget call it 80ms. And the safety headline I will design toward: detect a bad release in under 10 minutes, roll it back in under one minute, both automated.

**[3:30] INTERVIEWER:** Why is rollback such a focus for you?

**[3:45] YOU:** Because of the one idea this whole case turns on: a model is not code. With code I write a test that asserts correctness before I ship — `assert add(2,2)==4`. A model has no such test. Its "correctness" is only defined relative to live data I have not seen, it shifts under me as the world drifts, and when it fails it does not throw — it returns a confident, well-typed, wrong number and quietly loses money. So I can never *prove* a model good before traffic. The only real test is production. That means the platform's job is to expose each new model to reality gradually, cheaply, measured against the incumbent, and make undo instant. Rollback is not an afterthought; it is the thing that makes shipping safe at all.

**[5:30] INTERVIEWER:** Okay. Walk me up from the simplest thing.

**[5:45] YOU:** Rung zero is manual scripts — someone SCPs a pickle and restarts a box. Fine for one team. It breaks the moment you have a second team or your first incident, because you cannot say what is running, you cannot reproduce it, and rollback is "go find the old pickle." So rung one is a **model registry**: every model is an immutable bundle keyed by content hash — weights, a *pointer and hash* to the exact training-data snapshot, the feature schema, the training code commit, the container digest, seeds, the eval report, and the owner and on-call. Immutable, because during an incident "what exactly was serving" must always be answerable, and for the credit models it is a legal requirement. Now rollback is "repoint to the previous version ID."

**[7:30] INTERVIEWER:** What does the registry not solve?

**[7:40] YOU:** Two things. One, I can roll back but I still learn a model is bad from a product dashboard hours later. Two, and worse, train-serve skew. So rung two is a **feature store**, and it is the highest-leverage thing on the platform. A feature is defined once, and the store guarantees the *same code path* materializes it for both offline training tables and online sub-10ms reads — that kills feature-computation skew, which is the number-one cause of "offline up, online down." And it enforces point-in-time-correct joins: when you build a training set it gives the feature value as of the label timestamp, never later, which structurally prevents the most embarrassing bug in ML — leakage, where you accidentally train on a feature computed after the event, get spectacular offline numbers, and ship garbage because that feature does not exist at serving time. The feature store plus an offline-eval CI gate is rung two.

**[9:30] INTERVIEWER:** And offline eval is enough?

**[9:40] YOU:** No — necessary, never sufficient, by the "a model is not code" argument. Offline green can still mean online red: skew, leakage, a latency regression offline never measured, novelty, feedback loops, or the offline metric being the wrong proxy. So rung three is the real production design: **shadow, canary, ramp, auto-rollback.** Let me walk a request through it.

**[10:30] YOU:** In **shadow**, the candidate is deployed beside production and live requests are *mirrored* to it. It scores the real input, but its output is logged and discarded — never returned, no side effects. That buys me three things offline eval cannot: real operational latency under real load, a prediction-distribution diff against the incumbent on identical inputs, and skew detection — because it is scoring live features I can diff "features seen in shadow" against "features seen in training" at zero blast radius. Shadow's blind spot is outcomes: its predictions affect nothing, so it cannot measure clicks or conversions. For that I go to **canary** — route ~1% of traffic, sticky by user hash, and now outputs are returned, so I measure real outcomes against guardrails on a bounded blast radius. Clean canary, then a proper **A/B and ramp** — 1, 5, 25, 50, 100% with a hold and automated guardrail check at each step.

**[13:00] INTERVIEWER:** Tell me exactly how rollback works.

**[13:10] YOU:** It is a pointer flip, not a redeploy. Traffic goes through a router mapping model-name to active-version-id. The previous version is immutable in the registry and kept warm. Rollback sets the pointer back — sub-second. And it is *automatic*: triggers are a guardrail breach past a pre-registered threshold, a prediction-distribution divergence past a KL or PSI bound, or an error/NaN spike. The bad candidate is not deleted, it is parked, so I can root-cause from its immutable bundle. That is how I hit detect-under-10, rollback-under-1.

**[15:00] INTERVIEWER:** Suppose a team's model passed every offline gate, the canary looked fine, you ramped to 100%, and a week later conversion is down. Walk me through it.

**[15:20] YOU:** I work the offline-online gap checklist in priority order. First, train-serve skew — diff the live feature distribution against training; the usual culprit is a feature computed slightly differently in the streaming path. Second, leakage that inflated the offline number — check whether a high-importance feature was actually available at request time. Third, latency — is the "better" model slower, timing out, and hitting the fallback on a latency-sensitive surface? Offline eval never measured that. Fourth, distribution shift since the frozen offline set — seasonality or novelty. Fifth, a feedback loop — the new model changed what items we showed, which changed the clicks we logged, so any offline replay is now biased. Sixth, wrong proxy — we optimized AUC offline, the business is revenue at a threshold, and they decorrelated. The fact that canary looked fine but the week-long read did not points me at four through six — a slow effect that 1% for a day could not surface, which is exactly why the ramp has holds and the A/B runs a full business cycle.

**[18:00] INTERVIEWER:** How do you keep this from becoming a bureaucratic bottleneck that teams hate?

**[18:15] YOU:** Risk-tiered gates. The 90% of models that are low-blast-radius get the fully automated paved road — register, offline gate, shadow, canary, auto-ramp, no humans. The high-blast-radius tier — credit, anything regulated — additionally gets mandatory human sign-off, a fairness gate, and an explainability check. The platform owns reproducibility, rollout mechanics, and the kill switch; the *team* owns model correctness and the on-call. If the platform tried to own correctness it would become the bottleneck and teams would route around it, which is the death of any platform.

**[20:00] INTERVIEWER:** What about the feature store at 500K requests a second?**

**[20:15] YOU:** A ranking model pulling 200 features at 500K req/s is 10^8 feature lookups a second — the store, not the model, is the bottleneck. Online reads are a low-latency KV layer, p99 under 10ms, hot features cached, lookups batched per request. Materialization runs both batch (for stable aggregates) and streaming (for fresh ones), and crucially through the *same* transformation code as the offline path so parity holds. Noisy-neighbor isolation matters too: one team's nightly batch backfill must not starve another team's online reads, so quotas and separate pools.

**[22:00] INTERVIEWER:** Last thing — how do you support retraining?

**[22:10] YOU:** Through the feedback path, which I would call out as one of three independent paths in the running system — serving, data, and feedback. The feedback path logs every prediction and joins delayed labels back in the warehouse. That serves three jobs: compute true online metrics, build the next training set with point-in-time correctness, and feed drift detection. Scheduled retrains for stable domains, drift- or performance-triggered retrains otherwise — and every retrain is just a new immutable registry version that re-enters the same shadow-canary-ramp ladder. I would keep the retraining *policy* itself as its own concern rather than bolt it on here.

**[24:00] INTERVIEWER:** Great, let us stop.

**[24:10] YOU:** Quick close: the platform exists because a model is not code, so we cannot certify it before traffic — we make releases reproducible via an immutable registry, prevent skew and leakage via a parity-guaranteed feature store, expose each version to reality through shadow then canary then ramp, and make undo a sub-second automatic pointer flip. This is the same regression-detection and production-reliability discipline I have run for my own models — here I am generalizing it into a paved road so fifty teams get it for free.

### Why this transcript works
- **Leads with the one idea** ("a model is not code") and derives every gate from it, instead of listing boxes.
- **Climbs the ladder with explicit triggers** — each rung is justified by the specific thing the previous rung could not do.
- **Goes deep on one mechanism** (shadow/canary/rollback) at the request level, the way an operator talks.
- **Nails the signature trap** (offline-up/online-down) with an ordered, mechanistic checklist, not hand-waving.
- **Handles the human/political question** (bottleneck) with risk-tiered gates — a senior, organizational answer.
- **Separates the three paths** and treats rollback and the feature store as first-class, which is where juniors are thin.
- **Closes by connecting to real experience** without overclaiming.

---

## 10. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Core framing | "Build a registry and a serving endpoint." | "A model is not code; the platform exists to certify-by-production because nothing else can." |
| Skew | Not mentioned. | Names the three sources and fixes the top two with a parity-guaranteeing feature store + point-in-time joins. |
| Eval | "Run offline metrics, block on regression." | Offline is necessary-not-sufficient; shadow for ops/skew, canary/A-B for outcomes; ordered offline-online-gap checklist. |
| Rollout | "Deploy with a canary." | Shadow (mirror, discard) → canary (1%, sticky) → ramp with holds, and says what each rung uniquely buys. |
| Rollback | "We can redeploy the old version." | Pointer flip, sub-second, automatic triggers, parked-not-deleted artifact; detect<10min / rollback<1min. |
| Reproducibility | "Save the model file." | Immutable bundle: weights + data-snapshot pointer/hash + code commit + image digest + seeds + eval report. |
| Org reality | Treats gates as uniform. | Risk-tiered gates; platform owns mechanics, team owns correctness; avoids becoming the bottleneck. |
| Cost/scale | Ignores feature store. | Feature store is the 10^8 lookups/sec bottleneck; caching, batching, isolation, scale-to-zero. |

---

## 11. One-page whiteboard cheat sheet

```text
SPINE: a model is NOT code -> cannot certify before traffic -> certify BY production, undo instantly

NUMBERS: 50 teams / 200 models / ~50 deploys/day (=> no human in common path)
         500K pred/s online, p99 ~80ms; feature store 1e8 lookups/s, p99<10ms
         SAFETY: detect bad release <10min, rollback <1min (pointer flip)

LADDER (climb only on a named break):
  0 manual script      -> can't reproduce, no rollback
  1 immutable registry -> still skew, still slow to detect
  2 feature store +     -> parity kills compute-skew; point-in-time kills leakage
    offline gate           ...but offline green != online good
  3 SHADOW->CANARY->RAMP->AUTO-ROLLBACK  <-- recommended prod design
  4 risk-tiered policy gates (human sign-off for high blast radius)  <-- extension

SHADOW: mirror live traffic, log+discard. Buys: real latency, pred-dist diff, SKEW detect. Blind to outcomes.
CANARY: 1% sticky, outputs returned, real outcomes vs guardrails, bounded blast radius.
RAMP:   1->5->25->50->100 with holds + auto guardrail checks.
ROLLBACK: router maps name->active_version; flip pointer; warm prev; auto-trigger on guardrail/dist/error.

3 PATHS: serving (router->features->forward->post) | data (feature store parity) | feedback (log+labels->retrain/drift)

SKEW (signature problem): (1)compute skew (2)point-in-time leakage (3)dist shift
  fix 1&2 with feature store. #1 = top cause of OFFLINE-UP/ONLINE-DOWN.

OFFLINE!=ONLINE checklist: skew -> leakage -> latency/fallback -> drift -> feedback loop -> wrong proxy
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x the teams?** Self-service becomes mandatory (no human in the common deploy path), multi-tenant isolation and quotas matter more, and you invest in templates/golden-path SDKs so teams cannot misconfigure rollout. The registry and feature store shard; the router stays central.
- **How do you handle a model that has no online label for weeks (e.g., 90-day fraud chargebacks)?** Lean harder on shadow (prediction-distribution diff) and proxy metrics for the canary, accept that true performance confirmation is delayed, and keep the feedback path joining labels as they arrive to confirm or trigger retrain. (Cross-link: file 09 delayed labels.)
- **How do you pick rollout percentages and hold times?** By blast radius and statistical power: enough traffic per step to detect the guardrail-breaking effect size, enough wall-clock to cover a business cycle, larger steps only after clean reads. High-risk tiers ramp slower.
- **What if offline metrics improve but online metrics drop?** Run the §7.2 ordered checklist: skew → leakage → latency/fallback → distribution shift → feedback loop → wrong proxy.
- **How would you debug a bad launch?** Pin the version, identify the slice and version from the immutable registry bundle + traces, roll back via pointer, root-cause offline against the parked artifact, then add a gate so it cannot recur.
- **How do you prevent feedback loops?** Reserve exploration/holdout traffic that the model does not influence, log with randomization where you can, and prefer A/B (which measures reality) over offline replay (which is biased by what past models showed).
- **How is this different from regular software CI/CD?** You cannot assert model correctness before shipping; the rollout ladder *is* your test suite, and it runs in production.

---

## 13. Common mistakes

- Treating it as "serve a model behind an endpoint" and never mentioning reproducibility, skew, or rollback. (The case is about *safety at fleet scale*, not serving one model — that is file 08.)
- Never saying "a model is not code." Without that frame, the gates look like arbitrary bureaucracy.
- Ignoring train-serve skew and point-in-time leakage — the two failure modes that actually cause silent production disasters.
- Describing rollback as "redeploy the old version" instead of an instant, automatic pointer flip.
- Uniform gates for all models — either too slow for the safe 90% or too loose for the dangerous 10%. Risk-tier them.
- Forgetting the feature store is the real latency/cost bottleneck at scale.
- Listing offline metrics with no shadow/canary/A-B story and no offline-online-gap diagnosis.
- Making the platform own model *correctness* (bottleneck) instead of owning *mechanics* while teams own correctness.

---

## 14. Transfer: what this case unlocks

- **File 08 (LLM serving platform):** this case is the *control plane* (registry, rollout, rollback); file 08 is the *data plane* (KV cache, batching, throughput). Together they are a full ML platform answer.
- **File 20 (monitoring/drift/retraining):** the feedback path and drift monitors here are the *trigger*; file 20 owns the retraining *policy* and drift detection depth.
- **File 11 (ads/experimentation):** the A/B machinery the platform runs as a service — power, CUPED, guardrails, ramp — is file 11's home turf.
- **File 09 (fraud) and 02/03 (ranking):** every one of those models would *ship through this platform*; the skew/leakage discipline here is what keeps their offline wins from evaporating online.
- **General skill:** the "expose to reality gradually, measure against incumbent, undo instantly" pattern is the universal answer to deploying anything whose correctness you cannot prove in advance.

---

## 15. Sources

Original guides (kept):
- [IGotAnOffer ML System Design Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [Exponent ML System Design Interview Guide](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)
- [Hello Interview ML/System Design Learning](https://www.hellointerview.com/learn)
- [Designing Machine Learning Systems, Chip Huyen](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)

Added canonical references (verify titles; these are well-established works):
- [Hermann & Del Balso, "Meet Michelangelo: Uber's Machine Learning Platform" (2017) — the foundational ML-platform writeup](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015 — why ML systems rot (glue code, CACE, skew)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Feast (Feature Store) documentation — online/offline parity and point-in-time joins](https://docs.feast.dev/)
- [TFX (TensorFlow Extended) paper, Baylor et al., KDD 2017 — production ML pipeline gates and validation](https://dl.acm.org/doi/10.1145/3097983.3098021)
- [Breck et al., "The ML Test Score: A Rubric for ML Production Readiness," IEEE Big Data 2017](https://research.google/pubs/pub46555/)
