# 19. Notification Optimization / Bandit System

**Company tags:** Meta, Uber, LinkedIn, Amazon, DoorDash, any consumer app with push
**Interview frequency:** Medium
**Why it matters:** Notifications are the canonical *push* problem: you interrupt the user, every send costs attention even when ignored, the payoff is delayed retention rather than an immediate click, and over-sending churns people. It is the cleanest interview test of causal incrementality, long-horizon reward, fatigue as a budget, and guardrail-heavy experimentation. Optimize the obvious metric (open rate) and you actively harm the product.

---

## 0. How to use this doc

Two passes.

**Pass 1 (intuition).** Read sections 1-6 once. The one idea everything hangs on: *a notification is not a recommendation, it is an interruption with a cost.* You are not asking "what will the user click," you are asking "will sending this *cause* a useful action that would not have happened anyway, and is that incremental value worth spending a slice of this user's finite tolerance for being interrupted?" Hold that — incremental value minus fatigue cost, under a per-user budget — and uplift modeling, the suppress decision, the delayed reward, and the bandit constraints all follow.

**Pass 2 (active recall).** Cover the page. On a whiteboard, write the objective as a constrained optimization (maximize incremental long-term value subject to a per-user volume/fatigue budget), then explain why a click-prediction model sends exactly the wrong notifications. Run the section 9 transcript as a simulation: answer each INTERVIEWER line before reading YOU. If you cannot explain why open-rate optimization is harmful and how you would measure incrementality without a clean control, you have not learned it yet.

The reusable scaffold (same across this whole set):

> **Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor**

This file leans hardest on Frame (it is causal, not predictive), Data/Labels (the label is a counterfactual you cannot directly observe), and Eval (you cannot A/B an attack, but here you *can* and *must* run holdouts to measure incrementality and long-term harm).

---

## 1. Clarify: the questions that change the design

| Question | Why it changes the design |
|---|---|
| "Is the objective immediate action (open/click) or long-term retention?" | This is the whole design. Optimizing opens sends notifications people would have acted on anyway and burns tolerance for zero incremental gain. Optimizing long-term retention forces uplift + a delayed reward. |
| "Are these *triggered/transactional* (ride arrived, payment failed) or *discretionary/marketing* (re-engagement, recommendations)?" | Transactional are near-mandatory, low-volume, latency-critical, and mostly bypass the optimizer. Discretionary are where fatigue, suppression, and bandits live. Separate the two pipelines. |
| "What is the per-user volume budget, and is it a hard cap or a learned soft constraint?" | The budget is the user's attention. A hard daily/weekly cap is the simplest guardrail; a learned per-user fatigue budget is the senior version. Either way the decision is allocation under scarcity, not independent per-notification scoring. |
| "What counts as a 'useful action' vs a vanity open?" | Open rate is a vanity metric. The label must be a downstream useful action (completed order, meaningful session) and ideally an *incremental* one. |
| "What's the cost of over-notifying — mute, uninstall, push-permission revocation?" | Losing the push channel entirely (OS-level revoke) is catastrophic and irreversible. This makes opt-out/mute/uninstall a hard guardrail, not a secondary metric. |
| "Do we control timing (send-time optimization) or only the send/suppress decision?" | Send-time is a large, separable lever (per-user best hour). Pin whether it is in scope. |
| "Quiet hours, timezones, regulatory constraints (e.g., marketing-consent regimes)?" | Quiet-hours and consent are hard eligibility filters that run before any model. Getting these wrong is a compliance incident. |
| "Multiple senders/teams competing for the same user?" | If many teams can push, you need a central allocation/arbitration layer with a shared per-user budget, or every team individually "optimizes" and collectively spams the user to death (tragedy of the commons). |

The two answers that dominate: **immediate vs long-term objective** (decides predictive-vs-causal) and **triggered vs discretionary** (decides which pipeline you are even designing). Get those on the board first.

---

## 2. Numbers up front (carry these through)

Realistic large-consumer-app scale.

- **Users:** 200M MAU. Discretionary-notification-eligible at any time: ~100M.
- **Candidate notifications/day:** each user generates many candidate notifications from many senders — say ~20 candidate notifications/user/day across all senders -> **~2B candidate decisions/day.** Most must be *suppressed*.
- **Volume budget:** target ~2-3 discretionary notifications/user/day, hard cap ~5/day and ~10/week. So we send well under 10% of candidates. **The default decision is suppress.**
- **Decision latency:** discretionary sends are not interactive — they are decided in near-real-time-to-batch. Triggered/transactional (payment failed, ride arrived) need p99 < a few hundred ms; discretionary can tolerate seconds-to-minutes and run on a streaming/batch path. State this split explicitly.
- **Reward horizon:** the outcome that matters (did this send help or hurt retention) resolves over **days to weeks**, not seconds. This delayed, partly-negative reward is the crux.
- **Guardrail magnitudes:** mute/opt-out rate baseline ~0.X%/send; push-permission revoke is rare but terminal. A +0.1pp opt-out lift on 100M users is 100K users losing a channel — treat as serious.
- **Cost of a wasted send:** near-zero infra cost, but a real fatigue/attention cost. The binding constraint is attention, not compute. Say this out loud — it reframes the whole problem as constrained allocation of a non-monetary budget.
- **Exploration budget:** a small holdout (e.g., 1-5%) gets randomized send/suppress + randomized timing to (a) measure incrementality and (b) get unbiased labels for off-policy learning.

The headline reframe: **infra is cheap, attention is the scarce budget, and the default action is to send nothing.** Most candidates design a system that sends more; the senior move is a system that mostly suppresses.

---

## 3. The conceptual spine: incrementality, delayed reward, fatigue budget

Four ideas. Everything in the architecture is one of them.

**(1) Incrementality, not response.** A click-prediction model ranks notifications by P(action | sent). But the users most likely to act are exactly the ones who would have come back on their own — you spend their attention budget to "cause" something that was going to happen. The right target is **uplift / incremental treatment effect**: P(action | sent) − P(action | not sent). You want to notify the *persuadables*, not the *sure things* (waste) and not the *do-not-disturbs* (the people a notification annoys into leaving). This is the Persuadables / Sure-things / Lost-causes / Sleeping-dogs framing from uplift modeling, and the "sleeping dogs" (negative uplift — a send makes them *less* likely to stay) are the ones that make naive optimization dangerous.

**(2) The reward is delayed and can be negative.** The value of a send is not the open; it is the change in long-term engagement/retention, realized over days/weeks, and a poorly-targeted send has *negative* long-term value via fatigue, mute, and uninstall. A bandit or model that optimizes immediate clicks is optimizing a biased, short-horizon proxy of a long-horizon, sometimes-negative reward. This is why you need long-term holdouts (section 7), not just click A/Bs.

**(3) Fatigue is a per-user state and a budget.** Each send raises the probability the next one annoys. The decision is therefore not per-notification-independent: it is **allocation of a scarce per-user attention budget across competing candidates over time.** Formally, maximize sum of incremental value subject to a per-user volume/fatigue constraint — a constrained optimization (think a per-user "knapsack" over the day/week, or a Lagrangian where the multiplier prices a send against fatigue). Treating each notification as an independent send/no-send is the central junior mistake.

**(4) You are mostly choosing to stay silent, and learning requires exploration you cannot fully observe.** Since the default is suppress, you rarely observe what would have happened on the roads not taken. Off-policy/counterfactual estimation and a randomized exploration holdout are not optional niceties; they are the only way to learn under a policy that mostly does nothing.

Hold these four and the rest is mechanics.

---

## 4. Data and labels: the label is a counterfactual

**Sources/features.** User activity & recency, timezone & historical active hours, recent notification history (how many, when, responded?), current fatigue/saturation state, content type & urgency, sender, predicted relevance of the content, historical per-user response by content type and time.

**The label problem (this is the hard part).**
- The naive label "user opened the notification" is a **vanity metric and a biased target** (point 1 above). It rewards sending to sure-things.
- The label you actually want — *incremental* useful action caused by the send — is a **counterfactual**: you cannot observe both "sent" and "not sent" for the same user at the same moment. You estimate it. Two clean ways:
  - **Randomized holdout:** a fraction of eligible (user, candidate) pairs are randomly suppressed (control) vs sent (treatment). The difference in downstream useful-action and retention is the causal uplift, and these logs are the gold training/eval data.
  - **Uplift modeling on observational data** (two-model / class-transformation / uplift trees) when you cannot randomize everything — but with explicit assumptions (overlap, no unobserved confounders) and validated against the randomized holdout.
- **Delayed labels:** the retention outcome resolves over weeks. Train on a short proxy (e.g., 1-7 day incremental action) but *validate* the proxy against the long-horizon holdout, because a proxy that diverges from long-term retention is worse than useless.
- **Negative outcomes are labels too:** mute, opt-out, uninstall, push-revoke must be attributed back to the send(s) that caused them and enter the reward with a large negative weight. A model trained only on positive engagement will happily churn users.
- **Logging for off-policy:** log the *propensity* (probability the policy chose to send) with every decision, so you can do inverse-propensity / doubly-robust off-policy evaluation later. Without logged propensities you cannot honestly evaluate a new policy offline.

**Cold start.** New user / new content / new send-time: no per-user history. Fall back to segment priors and global send-time distributions, open a wider exploration budget early to learn the user's tolerance fast, and start conservative on volume (under-notify a new user rather than risk an early uninstall).

---

## 5. Baseline -> why it breaks -> the next rung

**Rung 0 — Rules + global caps.** Eligibility filters (consent, quiet hours), fixed per-segment schedules, hard volume caps. Simple, safe, explainable, compliant.
*Why it breaks:* no personalization; sends to people who would have come anyway (no incrementality); ignores per-user fatigue differences; misses good timing.

**Rung 1 — Response-prediction ranking + caps.** Rank candidates by P(useful action | sent), send the top ones up to the cap.
*Why it breaks:* it optimizes correlation, not cause. It systematically targets sure-things (high P(action) but low uplift) and wastes budget, while never identifying sleeping-dogs it should suppress. Open rate goes up, incremental retention does not, fatigue does.

**Rung 2 — Uplift model + constrained allocation (the production default).** Model incremental effect P(action|sent) − P(action|not sent) per candidate, then **allocate the per-user budget to the highest-uplift candidates** (suppress everything else), with hard caps, quiet hours, and a fatigue penalty as the constraint. Send-time optimization as a separable per-user model.
*Why it breaks (if pushed):* it is still myopic per-decision-window and depends on the randomized data staying representative; it does not actively explore to reduce its own uncertainty, and it treats the day's sends as independent of next week's.

**Rung 3 — Constrained contextual bandit / RL for long-term value.** A contextual bandit (e.g., Thompson sampling / LinUCB over context) learns send/suppress/timing online while *exploring*, with the constraints from rung 2 as hard guardrails; or a longer-horizon RL formulation where the reward is long-term retention and the action sequence over days matters (sending today affects tolerance tomorrow).
*When to actually go here:* when you have the holdout infrastructure to keep exploration safe, when rung 2's lack of exploration is a measured ceiling, and when you can stomach the operational complexity. Start with **offline replay (off-policy evaluation) and a small randomized bucket** before any broad online learning. Never let a bandit explore freely against the fatigue budget — it will over-send during exploration and churn users.

State explicitly: **rung 2 (uplift + constrained allocation) is the answer.** Rung 3 (bandit/RL) is the extension you reach for with evidence and strong guardrails, introduced gradually via offline replay then small exploration buckets.

---

## 6. One architecture, explained to the floor

Three paths drawn separately (serving/decision, data, feedback/learning).

### 6a. Decision path

```
Candidate notifications (many senders) for user U
  -> ELIGIBILITY FILTER: consent, quiet hours, channel valid, dedupe, regulatory   [hard gate]
  -> per-candidate UPLIFT score: P(action|send) - P(action|suppress)
  -> per-candidate SEND-TIME score (best hour for U)
  -> CENTRAL ALLOCATION across all candidates under U's budget:
        maximize sum(uplift) s.t. volume cap + fatigue penalty + sender fairness
        (Lagrangian: send only if uplift > price-of-a-send(fatigue state))
  -> for survivors: schedule at optimal time (or send now if urgent/transactional)
  -> else SUPPRESS (the default)
  -> log decision + PROPENSITY for off-policy eval
```

Key decisions and why:
- **Eligibility is a hard pre-model gate** (consent, quiet hours). Compliance and channel-preservation cannot be a soft score.
- **Central allocation, not per-notification independence.** A shared per-user budget arbitrated across all senders is the only defense against the multi-sender tragedy of the commons. The Lagrangian framing — "send only if the candidate's uplift exceeds the current price of spending a send, where the price rises with fatigue" — is the clean way to say this on a whiteboard.
- **Transactional bypass.** Urgent/transactional notifications (payment failed, ride arrived) skip the optimizer (or get near-infinite priority) and run on the low-latency path. Do not let the bandit suppress "your house is on fire."
- **Suppress is the default action.** Most candidates are dropped. Design the system as a filter, not a firehose.
- **Log propensities** so the feedback path can do honest off-policy evaluation.

### 6b. Data path

```
Notification logs + downstream actions + negative events (mute/opt-out/uninstall)
  -> attribute outcomes to sends (and to suppressions via the holdout)
  -> join randomized-holdout treatment/control -> uplift labels
  -> build per-user fatigue/saturation features, send-time histograms
  -> train: uplift model, send-time model, fatigue model
```

### 6c. Feedback / learning path

```
Randomized holdout (1-5%): randomized send/suppress + randomized timing
  -> unbiased incrementality + unbiased labels for off-policy learning
  -> off-policy evaluation (IPS / doubly-robust) of candidate new policy
  -> if bandit: update posterior (Thompson/LinUCB) within guardrails
  -> long-horizon holdout (global, never-notified or capped) measures true retention effect
  -> retrain / promote policy -> canary -> ramp
```

**Objective / "loss."** The uplift model is trained to predict the conditional treatment effect (two-model difference, or a single uplift-tree / class-transformation loss). The allocation is a constrained optimization (knapsack/Lagrangian) maximizing total uplift under the fatigue budget. The reward used for the bandit/long-term evaluation is **incremental useful action minus a large penalty for mute/opt-out/uninstall**, measured over the holdout horizon.

---

## 7. Evaluation: the offline/online gap, incrementality, and a real experiment

**Offline metrics.**
- **Uplift metrics:** Qini coefficient / Qini curve, area under the uplift curve, uplift-at-k. *Not* plain AUC on response — a model can have great response-AUC and useless uplift.
- **Calibrated treatment-effect estimates**, validated against the randomized holdout.
- **Off-policy value estimate** (IPS / doubly-robust) of the new policy vs current, using logged propensities.
- **Predicted volume / suppression rate** under the new policy (a policy that quietly wants to send 3x more is a red flag before you ever ship it).
- Slice by cohort: new vs tenured users, heavy vs light users, content type, sender, timezone.

**The offline-up / online-down trap (enumerated causes).**
1. **You optimized response, not uplift:** offline response-AUC improved, but the new sends went to sure-things; online incremental retention is flat and fatigue rose.
2. **Proxy/horizon mismatch:** the short proxy label (1-day action) improved, but long-term retention dropped because the extra sends fatigued users — the proxy diverged from the true objective.
3. **Off-policy estimate was high-variance/biased:** IPS blew up because the new policy put weight where the old one rarely sent (poor overlap), so the offline value estimate was simply wrong.
4. **Ignored negative outcomes:** offline eval scored engagement only; online, mute/opt-out climbed and you lost the channel.
5. **Novelty effect:** a new notification type spikes opens for a week then decays below baseline — offline snapshot misses it.
6. **Allocation interaction:** each candidate looked good in isolation, but the per-user budget meant sending the new high-uplift type crowded out an even better one; per-notification offline eval missed the allocation interaction.
7. **Train-serve skew** on the fatigue-state feature (computed over different windows offline vs online).

**A fully-specified experiment (note the long-term holdout).**
- **Hypothesis:** the uplift-based allocation policy increases *incremental* 4-week retention and useful actions vs the response-ranking policy, without raising opt-out/mute or violating volume caps.
- **Design:** user-level randomization, sticky. **Crucially, maintain a global long-term holdout** (a small % of users who get the *minimal/baseline* notification policy for the whole quarter) to measure the absolute incremental value of the entire notification program over a long horizon — this is the only honest read on whether notifications help retention at all, and it catches slow fatigue damage that short A/Bs miss.
- **Primary metric:** incremental useful actions per user (treatment − concurrent control), and 4-week retention.
- **Secondary:** useful-action rate, notification-driven sessions, send volume per user.
- **Guardrails (auto-pause on breach):** opt-out rate, mute rate, uninstall rate, push-permission-revoke rate, quiet-hours violations = 0, per-user volume cap respected, sender-fairness within bounds.
- **Minimum runtime:** weeks, not days — the reward horizon is the constraint. Short tests over-credit novelty and under-count fatigue.
- **Rollback trigger:** any negative guardrail movement (especially opt-out/uninstall) or negative long-term retention flips the policy back. CUPED on pre-period engagement to cut variance and read faster.

---

## 8. Deployment, monitoring, incident response

**Three surfaces.**
- *Decision policy (uplift model + allocator + bandit posterior):* versioned bundle; deploy via pointer flip; rollback = flip back. The allocator's budget/cap config is part of the versioned artifact.
- *Data pipeline:* validate attribution joins and holdout integrity before retrain (a broken holdout silently destroys your ability to measure incrementality).
- *Feedback/learning:* scheduled retrains for stable behavior; triggered retrains on drift; bandit posterior updates gated by guardrails.

**Rollout discipline.** Shadow (run the new policy's decisions in parallel, log what it *would* send, send nothing) -> canary (1% sticky) -> guarded ramp -> auto-rollback. Always preserve the long-term holdout untouched across rollouts.

**Monitoring (what pages someone).**
- **Opt-out / mute / uninstall / push-revoke rate** — the terminal guardrails. Any sustained rise pages immediately.
- **Per-user send volume distribution** — watch the tail (the over-notified power-users who are about to churn), not just the mean.
- **Suppression rate** — a sudden drop means the policy started over-sending (often a bug or runaway exploration).
- **Incremental action vs the holdout** — is the program still net-positive?
- **Send-time and quiet-hours violations** — compliance.
- **Off-policy / drift signals** — has user behavior or content mix shifted enough that the model is extrapolating?
- **Sender fairness** — is one team monopolizing budgets?

**Fallback ladder.** If the uplift model/allocator is unhealthy: **fall back to rules + hard caps** (rung 0) — safe and compliant. If the bandit is misbehaving (over-exploring): **freeze exploration, serve the greedy/exploit policy** within caps. If the fatigue/feature store is down: **clamp to conservative global caps** (under-notify rather than risk churn).

**Incident response.** Notification storm / spike in opt-outs: immediately clamp global volume caps (kill switch on discretionary sends), freeze the policy version, compare against the previous version's decision logs, and inspect whether a candidate-generation bug flooded the allocator. Because over-notifying can permanently lose the push channel, the bias is toward **suppress-on-uncertainty**: when in doubt, send less.

---

## 9. One-hour interview, transcribed

**INTERVIEWER:** Design a system that decides which notifications to send, when, and when to suppress.

**YOU:** Two clarifications first, because they change everything. One: are we optimizing immediate opens or long-term retention? Two: are these transactional notifications — ride arrived, payment failed — or discretionary re-engagement? Transactional are near-mandatory, low-volume, latency-critical, and mostly bypass any optimizer. The interesting problem is discretionary, and I'll assume the objective is long-term retention, not opens.

**INTERVIEWER:** Correct on both — discretionary, long-term retention. Why does that distinction matter so much?

**YOU:** Because it makes this a causal problem, not a prediction problem. If I optimize open rate, I rank by probability the user acts given I send. But the users most likely to act are the ones who'd have come back anyway. I'd spend their limited tolerance for interruptions to "cause" something that was already going to happen — zero incremental value, real fatigue cost. The right target is uplift: P(action if sent) minus P(action if not sent). I want to find the persuadables, skip the sure-things, and especially avoid the sleeping-dogs — people a notification actually annoys into leaving. Those have negative uplift, and a click-optimizer can't even see them.

**INTERVIEWER:** So how do you get a label for "uplift"? You can't both send and not send.

**YOU:** Right, it's a counterfactual. The clean way is a randomized holdout: for a fraction of eligible candidates, randomly suppress some and send others. The difference in downstream useful action and retention between those groups is the causal uplift, and those logs are my gold training and eval data. Where I can't randomize everything, I use uplift modeling on observational data — two-model or uplift trees — but validate it against the randomized holdout, and I log the send propensity on every decision so I can do honest off-policy evaluation later.

**INTERVIEWER:** Let me write down some numbers. What scale?

**YOU:** 200M MAU, ~100M eligible. Across all senders, maybe 20 candidate notifications per user per day — so ~2B candidate decisions daily. But the budget is ~2-3 sends per user per day, hard cap maybe 5. So I'm sending under 10% of candidates. The key reframe: infra is cheap, the scarce budget is the user's attention, and **the default action is to send nothing.** I'm designing a filter, not a firehose.

**INTERVIEWER:** Walk me through the decision for one user.

**YOU:** First a hard eligibility gate — consent, quiet hours, valid channel, dedupe, regulatory. Then for each surviving candidate I score uplift and a best send-time. Then — and this is the part people miss — I don't decide each notification independently. I run a central allocation across all of that user's candidates under their budget: maximize total uplift subject to the volume cap and a fatigue penalty. The clean framing is a Lagrangian: send a candidate only if its uplift exceeds the current price of spending a send, where that price rises as the user's fatigue state rises. Survivors get scheduled at their best time; everything else is suppressed. And I log the propensity.

**INTERVIEWER:** Why central allocation instead of each team deciding?

**YOU:** Because if every team independently optimizes its own notifications against the same user, they collectively spam the user to death — a tragedy of the commons on a shared attention budget. A central arbiter with one per-user budget and sender-fairness constraints is the only defense. Transactional notifications bypass it with high priority, so the bandit never suppresses "your payment failed."

**INTERVIEWER:** You mentioned a bandit. Where does that come in?

**YOU:** As an extension, not the starting point. My default production design is the uplift model plus constrained allocation. A contextual bandit — Thompson sampling or LinUCB — adds active exploration to reduce uncertainty about each user's tolerance and response, with the caps and quiet hours as hard guardrails. But I'd never let it explore freely against the fatigue budget; unconstrained exploration means over-sending, which churns users. So I'd introduce it gradually: offline replay with off-policy evaluation first, then a small randomized exploration bucket, then wider. And the reward it optimizes is incremental action minus a heavy penalty for mute, opt-out, and uninstall — because losing the push channel is terminal.

**INTERVIEWER:** Suppose offline uplift metrics improve but online retention drops. Debug it.

**YOU:** Several suspects. First: did I actually improve uplift, or just response? If response-AUC went up but Qini didn't, I targeted sure-things. Second, and most likely here: proxy-horizon mismatch — my short proxy label, like 1-day action, improved, but long-term retention fell because extra sends fatigued people. The proxy diverged from the real objective. Third: my off-policy estimate was high-variance because the new policy sends where the old one rarely did — poor overlap, so IPS lied to me offline. Fourth: I scored engagement offline but ignored negative outcomes, and online the opt-out rate climbed. Fifth: allocation interaction — each candidate looked good alone but crowded out a better one under the budget.

**INTERVIEWER:** How do you run the experiment to trust the result?

**YOU:** User-level sticky randomization, ramp 1-5-20-50. Primary metric is incremental useful actions and 4-week retention, treatment minus concurrent control. Critically, I keep a global long-term holdout — a small percent of users on a minimal baseline policy for the whole quarter — to measure the absolute incremental value of the entire notification program over a long horizon. That's the only honest read on whether notifications help retention at all, and it catches slow fatigue damage a one-week A/B can't see. Guardrails that auto-pause: opt-out, mute, uninstall, push-revoke, quiet-hours violations, volume cap, sender fairness. Runtime is weeks, because the reward horizon demands it. CUPED to cut variance. Rollback on any negative guardrail, especially opt-out and uninstall.

**INTERVIEWER:** Last one — opt-outs spike after a launch. What do you do?

**YOU:** Opt-out and uninstall are terminal guardrails — losing the OS push permission is irreversible — so I act fast and conservatively. Clamp the global volume caps immediately, effectively a kill switch on discretionary sends, freeze the policy version, and flip to the previous one. Then root-cause: compare decision logs against the prior version, check whether a candidate-generation bug flooded the allocator or the bandit ran away exploring. My standing bias is suppress-on-uncertainty: when the system is unsure, it should send less, not more. I'd frame the whole post-mortem around the experiment and guardrail design — what holdout or guardrail should have caught this earlier — and tighten that, rather than just patching the one bad notification.

**Why this transcript works:**
- Reframes the problem as causal (uplift) within the first two answers and explains *why* open-rate optimization is harmful, including sleeping-dogs.
- Handles the counterfactual label honestly (randomized holdout + logged propensities), which is the technical crux.
- Reframes the scarce resource as attention and states that the default action is suppress — the senior tell.
- Insists on central allocation under a shared budget and explains the multi-sender tragedy of the commons.
- Defers the bandit with an explicit, safety-aware rollout (offline replay -> small bucket -> wider) and never lets it explore against the budget.
- The offline-up/online-down answer leads with proxy-horizon mismatch and off-policy overlap — the notification-specific failure modes.
- Specifies a *long-term* holdout, not just a one-week A/B, showing understanding of the delayed reward.
- Closes on experimentation and guardrail framing (per the brief), with suppress-on-uncertainty and a process-level post-mortem, not a project tie-in.

---

## 10. Junior vs senior answer

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Objective | "Maximize opens / clicks." | "Maximize incremental long-term retention minus fatigue cost; the default action is suppress." |
| Framing | Prediction: P(action \| sent). | Causal: uplift = P(action\|send) − P(action\|suppress); target persuadables, avoid sleeping-dogs. |
| Label | "User opened it." | A counterfactual estimated via randomized holdout; negative events (opt-out/uninstall) weighted heavily. |
| Decision unit | Per-notification send/no-send. | Central allocation of a shared per-user budget across all candidates (Lagrangian / knapsack). |
| Multi-sender | Each team decides. | Central arbiter + sender fairness, else tragedy of the commons. |
| Bandit | "Use a bandit to learn online." | Uplift + constrained allocation default; bandit as guarded extension via offline replay then small buckets; never explore against the budget. |
| Reward horizon | Immediate click. | Days-to-weeks; proxy validated against long-term holdout. |
| Eval | Response AUC, open rate. | Qini/uplift curves, off-policy value, long-term holdout; enumerates offline-up/online-down causes. |
| Guardrails | "Add a cap." | Opt-out/mute/uninstall/push-revoke as terminal guardrails; suppress-on-uncertainty; kill switch. |

---

## 11. One-page whiteboard cheat sheet

```
NOTIFICATIONS = interruption with a COST. Default action = SUPPRESS.
Objective: max INCREMENTAL long-term retention - fatigue cost, under a budget.

CLARIFY: immediate vs long-term? transactional vs discretionary?
  hard cap vs learned budget? useful action vs vanity open? multi-sender?
  quiet hours/consent? send-time in scope?

NUMBERS: 200M MAU, ~2B candidate decisions/day, budget ~2-3/day cap ~5
  send <10% of candidates. infra cheap, ATTENTION is the budget.
  reward horizon = DAYS-WEEKS, can be NEGATIVE (fatigue/uninstall)

SPINE:
  1. UPLIFT not response: P(act|send)-P(act|suppress). persuadables, not sure-things;
     beware SLEEPING-DOGS (negative uplift)
  2. reward DELAYED + can be NEGATIVE -> need long-term holdout
  3. FATIGUE = per-user budget -> constrained allocation (knapsack/Lagrangian),
     NOT independent per-notification decisions
  4. default=suppress -> need exploration + off-policy (logged propensities)

LADDER:
  0 rules+caps -> no incrementality, no perso
  1 response-ranking -> targets sure-things, fatigue up, retention flat
  2 UPLIFT + constrained allocation   <-- THE ANSWER
  3 constrained bandit / RL (long-term reward) -> via offline replay then small buckets

DECISION PATH: candidates -> ELIGIBILITY gate (consent/quiet hrs) ->
  uplift + send-time score -> CENTRAL ALLOCATION under budget+fatigue+fairness
  (send iff uplift > price-of-send) -> schedule or SUPPRESS -> log propensity
  (transactional BYPASSES optimizer)

LABEL: incremental useful action (randomized holdout); opt-out/uninstall = big negative

EVAL: Qini/uplift curve, off-policy value (IPS/DR), predicted volume
  offline-up/online-down: response!=uplift | proxy-horizon mismatch | IPS overlap
    | ignored negatives | novelty | allocation interaction
  A/B: user-sticky + GLOBAL LONG-TERM HOLDOUT (quarter), weeks runtime, CUPED
  guardrails: opt-out/mute/uninstall/push-revoke = TERMINAL

DEPLOY: versioned policy+budget config, pointer-flip rollback
  fallback: model down -> rules+caps | bandit runaway -> freeze exploration
  INCIDENT: opt-out spike -> clamp caps (kill switch) + flip back; suppress-on-uncertainty
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 100M+ users / many senders?** The central allocator becomes the bottleneck and the most important component; shard by user, run discretionary decisions on a streaming/batch path (seconds-to-minutes is fine), keep transactional on a separate low-latency path, async logging, and enforce one shared budget per user across all senders.
- **How do you handle cold start?** Segment priors + global send-time distributions, wider early exploration, and start conservative on volume — under-notify a new user rather than risk an early uninstall.
- **How do you pick the volume cap / budget?** It's a guardrailed business choice; tune the cap and the fatigue-price multiplier via the long-term holdout, watching the opt-out/retention tradeoff, not a single offline curve.
- **Offline up, online down?** Lead with proxy-horizon mismatch and off-policy overlap; then response-vs-uplift and ignored negative outcomes.
- **How do you prevent feedback loops?** Randomized exploration holdout for unbiased labels, logged propensities for off-policy correction, and the long-term holdout to detect slow fatigue damage the online policy can't self-observe.
- **Send-time optimization?** Separable per-user model over historical active hours; respect quiet hours as a hard filter; for time-insensitive content, defer to the user's best window rather than sending now.
- **Why not just a bandit end-to-end?** Unconstrained exploration over-sends and churns users; the reward is delayed so immediate-reward bandits optimize a biased proxy; introduce exploration only inside hard guardrails and via offline replay first.

---

## 13. Common mistakes

- Optimizing open/click rate (a vanity metric) instead of incremental long-term value — sending to sure-things and burning attention for zero gain.
- Predicting response instead of estimating uplift, so the system never identifies sleeping-dogs it should suppress.
- Treating each notification as an independent send/no-send instead of allocating a shared per-user budget.
- Letting multiple senders each "optimize" with no central arbiter — collective spam.
- Ignoring the delayed, sometimes-negative reward; validating only on short-horizon proxies and missing fatigue damage.
- Letting a bandit explore freely against the fatigue budget and churning users during exploration.
- Forgetting that opt-out/uninstall/push-revoke are terminal — losing the channel is irreversible — and not treating them as hard guardrails.
- No randomized holdout and no logged propensities, leaving you unable to measure incrementality or honestly evaluate a new policy offline.

---

## 14. Transfer: what this case unlocks

- **File 11 (ads CTR / experimentation):** shares pacing/budget and guardrailed experimentation, but ads spends *advertiser money* for immediate revenue, while notifications spend *user attention* for delayed retention — contrast incrementality and the long-term holdout explicitly.
- **File 02 (feed ranking):** feed is pull (rank what the user came to see); notifications are push (you interrupt, every send costs). Same position-bias/feedback-loop machinery, opposite cost structure.
- **File 15 (conversational rec):** the explore/exploit-under-friction-budget framing is the same value-of-information idea; here the budget is attention over time.
- **File 20 (drift / retraining):** the delayed-label, holdout-driven monitoring here is a concrete instance of long-horizon performance tracking.
- **The reusable muscle:** causal/uplift thinking, constrained allocation of a scarce non-monetary budget, off-policy evaluation with logged propensities, and long-horizon holdouts — applies to any system that *intervenes* on users rather than just predicting them.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Rules of Machine Learning: https://developers.google.com/machine-learning/guides/rules-of-ml
- Vowpal Wabbit contextual bandits: https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html

Added (canonical, for the techniques cited above):
- Gutierrez & Gérardy, "Causal Inference and Uplift Modelling: A Review of the Literature" (2017): https://proceedings.mlr.press/v67/gutierrez17a.html
- Radcliffe & Surry, "Real-World Uplift Modelling with Significance-Based Uplift Trees" (2011): https://www.stochasticsolutions.com/pdf/sig-based-up-trees.pdf
- Li, Chu, Langford & Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation" (LinUCB, WWW 2010): https://arxiv.org/abs/1003.0146
- Chapelle & Li, "An Empirical Evaluation of Thompson Sampling" (NeurIPS 2011): https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
- Dudík, Langford & Li, "Doubly Robust Policy Evaluation and Learning" (off-policy evaluation, ICML 2011): https://arxiv.org/abs/1103.4601
- Lattimore & Szepesvári, "Bandit Algorithms" (2020, textbook): https://tor-lattimore.com/downloads/book/book.pdf
- Deng, Xu, Kohavi & Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED, WSDM 2013): https://www.microsoft.com/en-us/research/publication/improving-the-sensitivity-of-online-controlled-experiments-by-utilizing-pre-experiment-data-cuped/
