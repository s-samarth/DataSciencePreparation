# 07. AI Agent for Customer Support / Ticket Resolution

**Company tags:** Amazon, Salesforce, ServiceNow, Intercom, Zendesk, Decagon, Sierra, AI startups
**Interview frequency:** High for LLM / applied-AI roles
**Why it matters:** A support agent that can *take actions* (refunds, account changes, cancellations) is the case where a wrong move costs real money and trust. It tests whether you understand that **the agent's job is not to resolve every ticket — it's to safely resolve the majority it can and route the rest, under a hard ceiling on wrong actions.**

---

## 0. How to use this doc

Built two ways; read it twice.

1. **As a thinking guide.** The headers are the whiteboard order. Internalize the *triggers* for each ladder rung.
2. **As a worked transcript.** Section 11 is a full timestamped hour. Cover the `YOU:` lines and answer from memory.

The one idea to carry out: **autonomy must be bound to two things — the *reversibility* of the action and the *calibrated confidence* of the decision.** A read-only lookup can be fully autonomous; a $500 refund cannot. The senior framing is selective automation: maximize the safely-contained fraction subject to a hard wrong-action ceiling, and escalate everything else. Resolving 100% is not the goal; resolving the safe 45% with near-zero harm is.

Scaffold (identical across all cases):

```
Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor
```

Note on scope vs adjacent cases: the deep **control-plane mechanics and permission model** live in case 04 (Enterprise Copilot); **trace-based agent eval** lives in case 06. This case owns the **resolve-vs-escalate economics, action-reversibility tiers, confidence calibration, and safe action execution**. I'll reference the others rather than re-derive them.

---

## 1. The reusable scaffold, stated once

| Phase | The question |
|---|---|
| Clarify | What can the agent *do*, to whom, and what's the cost of a wrong action? |
| Frame | What's the learnable target, and the non-ML baseline? |
| Data / Labels | Where does signal come from? (Hint: the human-handled tickets.) |
| Baseline | Simplest shippable thing, and what breaks it? |
| Model | The agent loop + the autonomy/escalation policy, explained to the floor. |
| Eval | Containment vs wrong-action curve; the offline/online gap; one A/B. |
| Deploy | Three paths; rollout by reversibility tier. |
| Monitor | What pages someone; the fallback. |

---

## 2. Clarify requirements (scripted)

| Question | Why it changes the design |
|---|---|
| "Does the agent *take actions*, or only suggest replies to a human agent?" | Suggest-only is a RAG assistant (case 05) with no action risk. Taking actions — refunds, cancellations, account edits — introduces irreversible blast radius and is the real version of this problem. I'll assume it acts, with guards. |
| "What's the cost of a wrong action vs a wrong escalation?" | Wildly asymmetric. A wrong refund is real money and abuse exposure; a wrong escalation just costs a human's time. This asymmetry sets every threshold in the system. |
| "What action types are in scope, and which are irreversible?" | I need the action taxonomy by reversibility: read-only, reversible writes, irreversible/high-value. Autonomy is granted per tier, not globally. |
| "Synchronous chat or async email/ticket?" | Chat needs sub-3s first response and multi-turn memory. Async tickets relax latency but need long-horizon state. Changes the latency budget, not the safety model. |
| "Is there a human support team to escalate to, and what's their capacity?" | Escalation is only a valid fallback if humans can absorb it. Containment-rate targets must respect human capacity, and escalated tickets are my best training labels. |
| "What does the business optimize — cost saved, CSAT, or SLA?" | Sets the north star. I'll treat it as: maximize **safe containment rate** (tickets resolved without a human) subject to wrong-action and CSAT guardrails. |

**Numbers I'll commit to and carry through:**

- **Volume:** ~100K tickets/day.
- **Target containment:** autonomously resolve ~45% within a quarter, ramping by reversibility tier (start near 0% on irreversible actions).
- **Hard wrong-action ceiling:** < 0.1% of automated actions are wrong or harmful. This is the constraint, not a soft metric.
- **Latency (chat):** first response < 3s (TTFT), per-turn p95 < 5s.
- **Cost:** < $0.20 per resolved ticket, all-in.
- **Escalation precision/recall:** catch ≥ 95% of cases that *should* go to a human (recall on "needs human").

### Latency budget for one agent turn (chat), derived out loud

```
Intent + context load (account state, history)   ~200 ms
Retrieval over policy/KB (case 05 machinery)      ~300 ms
Planner LLM step (decide: answer/tool/escalate)   ~600 ms TTFT
Tool call (if any; e.g. order lookup)             ~300 ms
Policy/guard validation before any write          ~50 ms
-----------------------------------------------------------
First useful response well under 3s; multi-turn
sessions stream and can take several turns.
```

The safety checks (policy validation, permission, idempotency) are cheap (~50ms) and **non-negotiable** — they never get dropped to save latency.

### Storage / scale note

100K tickets/day × multi-turn traces (~20KB each with tool logs) ≈ **2 GB/day** of trace data — small, but every action must be **audit-logged immutably** for dispute resolution and abuse investigation. The audit log is a compliance artifact, not just telemetry.

---

## 3. Frame as an ML problem

- **Framing:** a guarded agent loop — perceive ticket + account state, plan the next step (answer / call a tool / escalate), act through validated tools, and *decide at each step whether it is confident and authorized enough to proceed autonomously.*
- **The real learnable target is two-part:** (1) the *action policy* — given ticket + context, what's the right resolution; and (2) the **selective-prediction / deferral decision** — should the agent act or hand off? Part (2) is the one candidates miss and the one that makes this safe.
- **Why this framing wins:** it reframes the goal from "resolve everything" (impossible safely) to "resolve what you can prove is safe, defer the rest." That's selective prediction, and it's the senior signal.
- **Non-ML baseline:** keyword routing + canned macros + "everything else goes to a human." Resolves the trivial 15% and escalates the rest. Surprisingly hard to beat on safety, and it's your honest starting point.

---

## 4. Data and labels — the human-handoff flywheel

The domain's defining feature: **your best labels are the tickets humans already resolve.** Every escalated ticket that a support rep handles is a (problem, context, correct-resolution, correct-actions) tuple. The system that learns fastest is the one that turns escalations into training signal.

Signal sources, increasing cost/quality:

1. **Historical resolved tickets (the goldmine):** millions of past tickets with the resolution and actions a human took. These are your demonstrations — for retrieval ("similar past ticket"), for the action policy, and for offline replay eval. Biased toward how humans *did* resolve, including their mistakes and inconsistencies.
2. **Implicit outcomes:** did the ticket reopen within N days? Did the customer reply "that didn't work"? CSAT survey? Reopen rate is a strong, if delayed, correctness signal.
3. **Human review of agent actions:** sample the agent's autonomous resolutions and have reps grade them. This calibrates confidence and catches silent wrong-actions before reopen data arrives.
4. **The escalation stream (the flywheel):** tickets the agent deferred, now resolved by humans, become new labeled demonstrations that *expand the safe-automation frontier* over time. Today's escalation is tomorrow's automated resolution.

### Biases to name out loud

- **Past-human bias:** historical resolutions encode human inconsistency and policy drift; learning to imitate them blindly reproduces their errors and any unfair treatment across customer tiers. Audit for disparate handling.
- **Survivorship in reopen labels:** customers who give up don't reopen; "no reopen" overstates success. Pair it with CSAT and explicit follow-up.
- **Delayed labels:** reopen/CSAT arrive days later, so you can't gate a real-time decision on them — you act on calibrated confidence now and reconcile later. (Same delayed-feedback structure as the ads case.)

---

## 5. Baseline -> why it breaks -> next rung

| Rung | What it is | Why it breaks -> trigger to climb |
|---|---|---|
| 0 | Keyword routing + canned macros; humans do the rest. | Can't handle novel phrasing or multi-step issues; resolves only the trivial slice. Trigger: most tickets fall through to humans. |
| 1 | RAG reply assistant — suggests grounded answers, human sends them. | No autonomous resolution; human is still in every loop. Trigger: humans rubber-stamp the suggestions, proving the agent could act. |
| 2 | **Guarded agent that acts on read-only + reversible tiers autonomously, with confidence-gated escalation; irreversible actions always human-approved.** | This is the production default. Trigger to extend: you've earned trust on lower tiers and want to expand autonomy or handle harder workflows. |
| 3 | Multi-agent / specialist routing, learned escalation policy, autonomy expanded into higher tiers with proven calibration. | More infra and eval burden. Trigger: measured headroom — a large population of safely-automatable tickets still being escalated. |

Earn rung 2 by explaining why suggest-only (rung 1) wastes the human and why you can't jump straight to full autonomy (irreversible blast radius). The whole craft is in rung 2's autonomy/escalation policy.

---

## 6. The architecture, explained to the floor

```
   Ticket + customer + account state + product telemetry
                          |
                    Planner / router (LLM)
        decides per step: answer | call tool | escalate
                          |
        +-----------------+------------------+
        |                 |                  |
   Retrieve (case 05)  Tool call        Escalate to human
   policy + similar    (GUARDED)        (with full context
   past tickets             |            + suggested action)
                          GUARD CHAIN before any write:
                          1. action allowlist (is this tool permitted?)
                          2. reversibility tier lookup
                          3. confidence >= tier threshold?  --no--> escalate
                          4. permission/auth (does customer own this order?)
                          5. policy validator (deterministic rules:
                             refund <= cap, within window, not flagged)
                          6. idempotency key (no double-refund on retry)
                          7. [tier 3] human approval required
                          8. execute -> immutable audit log
                          |
                  State update + verify effect
                          |
            Final response (grounded, cited) OR escalation
```

### The autonomy/escalation policy — the heart of this case

Two independent dials, ANDed together:

**Dial 1 — Reversibility tier (the action's blast radius):**

| Tier | Examples | Default autonomy |
|---|---|---|
| T0 read-only | order status, policy lookup, account info | Fully autonomous. |
| T1 reversible write | resend confirmation, update notification prefs, reschedule | Autonomous if confidence clears threshold. |
| T2 bounded-value | refund/credit ≤ a low cap, within policy window | Autonomous only with high confidence + deterministic policy pass; else escalate. |
| T3 irreversible / high-value | large refund, account deletion, plan downgrade, anything flagged abuse | **Always human approval**, agent only proposes. |

**Dial 2 — Calibrated confidence (selective prediction):** the agent emits an action *and* a confidence. Act only if confidence ≥ the tier's threshold; otherwise defer. The threshold rises with the tier because the cost of a wrong action rises. This is **selective prediction**: the agent is allowed to say "I'm not sure — human, please." Coverage (fraction it handles) trades against risk, and you tune the operating point against the wrong-action ceiling.

**Why calibration matters:** raw LLM confidence is poorly calibrated — it's confidently wrong. So you calibrate: hold out historical tickets, bin the agent's stated/derived confidence, and check empirical correctness per bin (a reliability diagram). Set thresholds from *empirical* correctness, not the model's self-reported number. Uncalibrated confidence makes the whole safety argument collapse.

### Safe action execution — the engineering that prevents disasters

- **Two-phase: propose then commit.** The planner *proposes* a structured action; a deterministic validator checks it against hard policy (refund ≤ cap, within window, account in good standing) before execution. The LLM never writes directly to systems of record.
- **Idempotency keys.** Every action carries a key so a retry (timeout, agent loop) can't issue a refund twice. This is the single most important reliability guard and candidates always forget it.
- **Allowlist + least privilege.** The agent can only call an explicit set of tools, each scoped to the current customer's resources. No open-ended execution.
- **Recursion / cost budgets.** Hard caps on steps and token spend per ticket; exceeding them auto-escalates rather than looping forever.
- **Verify the effect.** After acting, confirm the state changed as intended (refund posted, ticket status updated) — grade against environment state, not the model's claim (the case-06 principle).

### Memory and multi-turn

Per-ticket working state (what's been tried, tool results) plus customer long-term context (tier, history, prior issues). Keep working memory bounded and summarized so the planner isn't drowned; persist the audit trail separately and immutably.

### Canonical references (verified)

- ReAct (reason + act interleaving) — Yao et al., 2022: https://arxiv.org/abs/2210.03629
- Toolformer (learning to call tools) — Schick et al., 2023: https://arxiv.org/abs/2302.04761
- Selective prediction / "knowing when to abstain" survey context — Geifman & El-Yaniv, 2017: https://arxiv.org/abs/1705.08500
- Anthropic, Building Effective Agents: https://www.anthropic.com/research/building-effective-agents
- OpenAI, A Practical Guide to Building Agents: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

---

## 7. Evaluation — the containment/wrong-action operating curve

The single most important picture: **containment rate (x) vs wrong-action rate (y).** Every threshold setting is a point on this curve. The senior framing: pick the point that *maximizes containment subject to wrong-action ≤ 0.1%*, per reversibility tier. This is a selective-prediction risk-coverage curve in disguise.

- **Offline metrics:** on historical tickets — action-policy accuracy (did it choose the resolution the human did / a policy-valid one), escalation precision/recall ("needs human" detection), tool-call success, and **calibration** (reliability diagram of confidence vs correctness).
- **Online metrics:** containment rate (resolved without human), **wrong-action rate** (the guardrail ceiling), reopen rate, CSAT, average handle time, and escalation quality (did escalations actually need a human, or did we over-defer?).
- **Guardrail / safety:** refund/credit abuse rate, policy-violation rate, hallucination rate on customer-facing text, human-override rate.

### The offline-to-online gap, including the classic trap

**"Offline replay said the agent resolves 60% correctly, but online containment is 40% and reopens spiked."** Causes, ordered:

1. **Replay can't model the customer's reaction.** Offline, a resolution that *looks* complete scores as success. Online, the customer replies "that didn't fix it" and reopens — multi-turn dynamics aren't in the replay.
2. **Distribution shift.** Live tickets include issues, products, and policies not in the historical set; the agent over-acts on them.
3. **Past-human label noise.** Offline "correct" meant "matched what a human did," but humans were inconsistent, so the offline target was wrong.
4. **Calibration drift.** Confidence calibrated on old tickets is miscalibrated on new traffic, so the agent acts when it should defer.
5. **Delayed-label optimism.** Offline you don't see reopens (they hadn't happened); online they arrive and correctness drops.

### One fully specified A/B test

- **Hypothesis:** enabling autonomous T1+T2 resolution (vs suggest-only) raises containment without breaching the wrong-action ceiling or CSAT.
- **Unit:** customer (sticky), so a customer's experience is consistent across a multi-turn ticket.
- **Arms:** control = RAG suggest-only (human sends); treatment = guarded autonomous agent on T0-T2, T3 always escalated.
- **Primary:** safe containment rate.
- **Guardrails (auto-stop):** wrong-action rate > 0.1%, CSAT drop, reopen-rate increase, refund-abuse spike, p95 latency.
- **Ramp by tier, not just by traffic:** enable T0 → T1 → T2 sequentially, each at 1→5→25→50% traffic, watching wrong-action at every step. Never turn on all tiers at full traffic at once.
- **Runtime:** ≥ 2 weeks because reopen/CSAT labels are delayed; don't call it early on containment alone.
- **Rollback:** any guardrail breach → drop the offending tier back to escalate-only.

### Error analysis ritual

Maintain a bank of wrong-actions (the expensive failures) and near-misses (acted at borderline confidence). After every policy/prompt change, re-run the bank. Wrong-actions are reviewed individually like incidents — each one either tightens a threshold, adds a policy rule, or moves an action to a higher tier.

---

## 8. Deployment — three paths

- **Serving path (synchronous):** ticket intake → planner → retrieval/tools (guarded) → response or escalation. Guards are inline and deterministic so they can't be skipped under load.
- **Data path:** trace + audit logging of every step and action (immutable), feeding offline replay, calibration, and the historical-ticket store. Reopen/CSAT labels join asynchronously (delayed-feedback reconciliation, like the ads case).
- **Feedback path (the flywheel):** escalated tickets → human resolution → new demonstrations → expand retrieval corpus, refine the action policy, and re-calibrate confidence → *raise autonomy on tiers that have earned it.* This loop is how containment grows safely over time.

### Rollout discipline — by reversibility tier

The unique rollout axis here: **ramp autonomy per tier, gated by measured wrong-action rate.** Start fully escalating everything (shadow the agent's proposed actions against what humans do), then enable T0 (read-only, zero risk), then T1, then T2 only after T2 calibration is proven. T3 may never become autonomous — and that's a correct design, not a failure.

### Monitoring and fallback

- **What pages someone:** wrong-action rate breaching the ceiling, refund/abuse spike, reopen-rate jump, calibration drift (confidence no longer predicts correctness), escalation queue overflow (agent deferring too much → humans drowning), runaway cost/loop.
- **Fallback ladder:** on any safety breach, **demote the affected tier to escalate-only** (degrade autonomy, not availability). If tools are down, the agent answers from RAG and escalates anything needing an action. If the agent is fully untrusted, fall back to RAG suggest-only (rung 1), then to canned-macro routing (rung 0). Always have a human path.
- **Incident response:** freeze the agent/policy version, pull the audit trail for affected tickets, **reverse reversible wrong-actions** where possible (this is why tiering matters), diff policy/prompt/model versions, and roll back the offending tier. Every irreversible wrong-action becomes a postmortem that tightens the tier boundary.

---

## 9. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Goal | "Resolve all tickets automatically." | Maximize *safe* containment subject to a hard wrong-action ceiling; escalate the rest. |
| Autonomy | "The agent calls tools." | Autonomy bound to reversibility tier AND calibrated confidence (selective prediction). |
| Confidence | "Use the model's confidence." | Calibrates against empirical correctness; raw LLM confidence is unreliable. |
| Actions | "Agent issues the refund." | Two-phase propose→validate→execute; LLM never writes to systems of record. |
| Reliability | Unaware of retries. | Idempotency keys, allowlists, recursion/cost budgets, verify-effect. |
| Labels | "Train on tickets." | Human-handoff flywheel — escalations become demonstrations that expand autonomy. |
| Eval | "Measure resolution rate." | Containment-vs-wrong-action operating curve, per tier; escalation precision/recall. |
| Offline/online | "Replay predicts prod." | Replay can't model customer reaction/reopens; calibration drift; delayed labels. |
| Rollout | "Ship the agent." | Ramp autonomy per reversibility tier gated on wrong-action; T3 may stay human-only. |
| Incident | "Roll back." | Reverse reversible wrong-actions; tier boundary is the blast-radius control. |

---

## 10. One-page whiteboard cheat sheet

```
NUMBERS: 100K tickets/day | target ~45% safe containment
         WRONG-ACTION CEILING < 0.1% (hard constraint)
         chat TTFT<3s, p95<5s | <$0.20/ticket | escalation recall >=95%

BIG IDEA: don't resolve everything. Resolve the SAFE majority,
          escalate the rest. Autonomy = f(reversibility, calibrated confidence).

TWO DIALS (ANDed):
  reversibility tier         x   calibrated confidence >= tier threshold
  T0 read-only  -> auto
  T1 reversible -> auto if conf
  T2 bounded $  -> auto if HIGH conf + policy pass
  T3 irreversible-> ALWAYS human approval

GUARD CHAIN (before any write):
  allowlist -> tier -> confidence -> permission -> policy validator
  -> idempotency key -> [T3 approval] -> execute -> immutable audit

LADDER: macros -> RAG suggest-only -> GUARDED AGENT (T0-T2 auto, T3 human)
        -> multi-agent + learned escalation + expanded autonomy

LABELS: historical resolved tickets (goldmine) | reopen/CSAT (delayed)
        | human review (calibration) | ESCALATION FLYWHEEL (expands autonomy)

EVAL: CONTAINMENT vs WRONG-ACTION curve, per tier | escalation P/R | calibration diagram
OFFLINE-UP/ONLINE-DOWN: replay misses customer reaction & reopens
                        | dist shift | past-human label noise | calibration drift | delayed labels

A/B: unit=customer | primary=safe containment | guard=wrong-action 0.1%, CSAT, reopen
     RAMP BY TIER (T0->T1->T2), gated on wrong-action | >=2wk (delayed labels)

DEPLOY: serving/data/feedback | ramp autonomy per tier
        fallback: demote tier -> suggest-only -> macros (never no-human-path)
        reverse reversible wrong-actions on incident
```

---

## 11. Full interview transcript (one hour)

**[00:00] INTERVIEWER:** Design an AI agent that triages and resolves customer support tickets.

**[00:30] YOU:** First, clarifying questions, because they change the design a lot. Does the agent actually take actions — refunds, account changes — or just suggest replies to a human? What's the cost of a wrong action versus a wrong escalation? Which actions are irreversible? Is it synchronous chat or async email? And is there a human team to escalate to, with capacity?

**[01:10] INTERVIEWER:** It takes actions, including refunds and account changes. Synchronous chat. There's a human team but it's expensive, so we want to deflect load. Some actions are clearly irreversible.

**[01:30] YOU:** Then the central design idea is this: the agent's job is *not* to resolve every ticket. It's to safely resolve the majority it can prove it can handle, and escalate the rest — under a hard ceiling on wrong actions, say 0.1% of automated actions. A wrong refund is real money and abuse exposure; a wrong escalation just costs a human a few minutes. That asymmetry drives everything. Let me commit to numbers: 100K tickets a day, target ~45% safe containment within a quarter, wrong-action under 0.1%, chat first response under 3 seconds, under 20 cents a ticket.

**[02:30] INTERVIEWER:** How do you decide when the agent acts versus escalates?

**[02:40] YOU:** Two dials, ANDed together. First, the reversibility tier of the action — its blast radius. T0 is read-only, like order status; fully autonomous. T1 is reversible writes, like resending a confirmation; autonomous if confident. T2 is bounded-value, like a refund under a low cap within policy window; autonomous only with high confidence and a deterministic policy pass. T3 is irreversible or high-value — large refunds, account deletion; the agent only *proposes*, a human always approves. Second dial: calibrated confidence. The agent emits an action and a confidence, and acts only if confidence clears that tier's threshold; otherwise it escalates. The threshold rises with the tier because the cost of being wrong rises.

**[04:00] INTERVIEWER:** You said calibrated confidence. Why not just use the model's confidence?

**[04:10] YOU:** Because raw LLM confidence is poorly calibrated — models are often confidently wrong, and if I gate a refund on a self-reported 0.9 that's really 70% accurate, my safety argument collapses. So I calibrate empirically: take held-out historical tickets, bin the agent's confidence, and measure actual correctness per bin — a reliability diagram. I set the thresholds from empirical correctness, not the model's number. And I monitor calibration drift in production, because confidence calibrated on old tickets goes stale on new traffic.

**[05:20] INTERVIEWER:** Walk me up from the simplest baseline.

**[05:30] YOU:** Rung 0: keyword routing plus canned macros, humans do the rest. Resolves the trivial slice, safe, doesn't scale. Rung 1: a RAG assistant that suggests grounded replies a human sends — but if humans are just rubber-stamping the suggestions, that's wasted human effort and a signal the agent could act. That triggers rung 2: the guarded agent that autonomously handles T0 through T2 with confidence gating, while T3 always gets human approval. Rung 3, later: multi-agent specialists and a learned escalation policy, once I've earned trust and measured headroom.

**[06:40] INTERVIEWER:** Let's talk about the agent actually issuing a refund. What worries you?

**[06:50] YOU:** Several things, and each gets a guard. First, the LLM should never write to a system of record directly — it *proposes* a structured action, and a deterministic validator checks it against hard policy: refund under the cap, within the window, account in good standing, not flagged for abuse. Two-phase: propose then commit. Second, idempotency keys on every action, so if the agent loops or a call times out and retries, I don't issue the refund twice — that's the guard people always forget. Third, an allowlist scoped to this customer's resources, least privilege. Fourth, recursion and cost budgets, so it auto-escalates instead of looping forever. And after acting, I verify the effect actually happened in the environment rather than trusting the agent's claim.

**[08:30] INTERVIEWER:** Where do your training labels come from?

**[08:40] YOU:** The goldmine is historical resolved tickets — millions of (problem, context, resolution, actions) tuples from what humans did. Those feed retrieval, the action policy, and offline replay. Then implicit outcomes: reopen rate and CSAT, which are strong but delayed correctness signals. Then human review of a sample of the agent's autonomous actions, to calibrate confidence. But the most important one strategically is the escalation flywheel: every ticket the agent defers and a human resolves becomes a new demonstration that lets me *expand* safe automation over time. Today's escalation is tomorrow's autonomous resolution.

**[09:50] INTERVIEWER:** Any bias concerns in those labels?

**[10:00] YOU:** Yes. Historical resolutions encode human inconsistency and any unfair treatment across customer tiers, so imitating them blindly reproduces those errors — I'd audit for disparate handling. Reopen labels have survivorship bias: customers who give up don't reopen, so "no reopen" overstates success; I pair it with CSAT and explicit follow-up. And the labels are delayed — reopens land days later — so I can't gate a real-time decision on them; I act on calibrated confidence now and reconcile correctness later.

**[11:10] INTERVIEWER:** How do you evaluate the system?

**[11:20] YOU:** The key picture is the containment-versus-wrong-action curve. Every threshold setting is a point on it, and I pick the point that maximizes containment subject to wrong-action staying under 0.1%, per tier. It's a selective-prediction risk-coverage curve. Offline, on historical tickets: action-policy accuracy, escalation precision and recall, tool success, and calibration. Online: containment rate, wrong-action rate as the hard guardrail, reopen rate, CSAT, and escalation quality — am I over-deferring tickets that didn't need a human.

**[12:40] INTERVIEWER:** Offline replay says 60% correct, you launch, containment is 40% and reopens spike. Why?

**[12:50] YOU:** Ordered. One, replay can't model the customer's reaction — offline, a resolution that *looks* complete scores as success, but online the customer says "that didn't work" and reopens; multi-turn dynamics aren't in replay. Two, distribution shift — live tickets have products and policies not in the historical set. Three, past-human label noise — "correct" offline meant "matched a human," but humans were inconsistent, so the target was wrong. Four, calibration drift — confidence calibrated on old tickets is off on new traffic, so the agent acts when it should defer. Five, delayed-label optimism — offline I didn't see reopens because they hadn't happened yet.

**[14:10] INTERVIEWER:** Design the A/B test.

**[14:20] YOU:** Randomize by customer, sticky, so a multi-turn ticket is consistent. Control is RAG suggest-only; treatment is the guarded agent on T0 through T2, T3 always escalated. Primary metric is safe containment rate. Guardrails that auto-stop: wrong-action above 0.1%, CSAT drop, reopen increase, refund-abuse spike, p95 latency. The crucial part: I ramp by tier, not just by traffic — enable T0, then T1, then T2, each at 1, 5, 25, 50 percent, watching wrong-action at every step. I never turn on all tiers at full traffic at once. Run at least two weeks because reopen and CSAT are delayed, and don't call it early on containment alone. Rollback drops the offending tier to escalate-only.

**[15:50] INTERVIEWER:** The agent issues a batch of wrong refunds in production. What happens?

**[16:00] YOU:** It pages immediately — wrong-action rate is a guardrail. First action: demote the affected tier, say T2 refunds, to escalate-only, so I degrade autonomy without taking the system down. Then I pull the immutable audit trail for the affected tickets. Because refunds are a reversible-ish, bounded action, I reverse what I can — this is exactly why tiering matters; I kept the truly irreversible stuff human-gated. Then diff the policy, prompt, and model versions to find what changed, roll back the offending tier, and every wrong-action becomes a postmortem that either tightens a threshold, adds a policy rule, or moves the action up a tier.

**[17:30] INTERVIEWER:** When would you expand autonomy into T3?

**[17:40] YOU:** Maybe never for the truly irreversible ones, and I'd say that explicitly — keeping account deletion human-gated is a correct design, not a limitation. For the rest, only when calibration on that action class is proven over a long window, wrong-action stays well under ceiling, and the escalation flywheel has given me enough demonstrations that confidence is trustworthy. I expand autonomy as something the system *earns* tier by tier, gated on measured wrong-action, never as a default.

**[18:40] INTERVIEWER:** What dominates cost and how do you keep it under 20 cents?

**[18:50] YOU:** LLM tokens across multi-turn planning. I tier models — a cheap model for triage and simple T0/T1 turns, escalating to a stronger model only for ambiguous or T2 decisions. I cap steps and tokens per ticket with a budget that auto-escalates on breach, which also prevents runaway loops. And retrieval caching for common issues. I monitor cost-per-ticket as a guardrail because a planner that loops would quietly blow the budget.

**[19:40] YOU:** This maps directly onto the Seller Copilot work I did — tool orchestration with human-in-the-loop on an enterprise workflow. The lesson that shaped how I think here: the hard part was never the happy path, it was deciding *when not to act*. We built explicit guard chains and human-approval gates for the high-blast-radius actions, and the reliability win came from idempotency and validating actions deterministically rather than trusting the model. That's why I lead with reversibility tiers and selective prediction instead of "the agent resolves tickets."

**[20:40] INTERVIEWER:** That's exactly the framing I wanted.

### Why this transcript works

- **Reframes the goal** from "resolve everything" to "safely contain the majority, escalate the rest" — the senior move.
- **Two-dial autonomy policy** (reversibility × calibrated confidence) with a concrete tier table.
- **Insists on empirical calibration** and explains why raw LLM confidence is dangerous.
- **Names the action-safety engineering** — two-phase, idempotency, allowlist, verify-effect — that candidates skip.
- **Centers the human-handoff flywheel** as the label source that grows autonomy.
- **Uses the containment/wrong-action curve** as the eval frame and handles the replay-vs-reality trap.
- **Ramps autonomy by reversibility tier**, and reverses reversible wrong-actions on incident.
- **Closes on real Seller Copilot tool-orchestration + human-in-the-loop experience.**

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x volume?** Tighten model tiering and caching, shard the trace/audit store, and watch human escalation capacity — if containment can't keep up, the human queue overflows, so containment targets are bounded by staffing.
- **How do you handle a brand-new product line (cold start)?** No historical tickets, so start fully escalating (T0 read-only only), let humans handle it, harvest those resolutions into demonstrations, then ramp autonomy as calibration data accumulates.
- **How do you set the confidence threshold per tier?** From the calibration diagram and the cost asymmetry — pick the operating point on the containment/wrong-action curve where wrong-action sits safely under the ceiling for that tier.
- **Offline up, online down — what do you check?** Replay can't model customer reaction/reopens, distribution shift, past-human label noise, calibration drift, delayed-label optimism. (Section 7.)
- **How do you debug a bad launch?** Audit trail for failing tickets, diff policy/prompt/model versions, separate retrieval failures from action-policy failures from calibration failures, demote the offending tier, reverse reversible actions.
- **How do you prevent abuse (refund farming)?** Per-customer action-rate limits, abuse flags as a hard policy gate that forces escalation, anomaly detection on refund patterns (ties to the fraud case), and lower autonomy for flagged accounts.
- **Suggest-only vs autonomous — when each?** Suggest-only when actions are high-stakes/irreversible or trust isn't yet earned; autonomous on low-tier actions with proven calibration. Most mature systems run a mix, per tier.

---

## 13. Common mistakes

- Framing the goal as **"resolve 100% of tickets"** instead of safe containment with escalation.
- Granting autonomy **globally** rather than per **reversibility tier**.
- Trusting **raw model confidence** instead of empirically calibrating it.
- Letting the **LLM write directly** to systems of record instead of propose→validate→execute.
- Forgetting **idempotency keys**, so retries double-charge or double-refund.
- No **recursion/cost budget**, so the agent loops forever.
- Treating escalations as pure loss instead of the **flywheel** that expands autonomy.
- Measuring **resolution rate** without the **wrong-action** guardrail or the operating curve.
- Ramping **all tiers at once** instead of tier-by-tier gated on wrong-action.
- No **human path** in the fallback ladder.

---

## 14. Transfer — what this case unlocks

- **04 Enterprise AI Copilot:** shares the guarded-tool control plane and permission model; that case owns the control-plane internals, this one owns the autonomy/escalation economics.
- **06 LLM Eval & Monitoring:** trace-based agent eval + environment-state grading is exactly how you'd score this agent.
- **05 Production RAG:** the "look it up" retrieval inside each turn is the RAG case's machinery.
- **09 Fraud / Anomaly Detection:** refund-abuse detection and per-customer anomaly limits are a direct borrow.
- **11 Ads / Experimentation:** delayed-feedback reconciliation and ramped A/B discipline are the same backbone.
- **13 LLM Safety Gateway:** the customer-facing text guard and policy validation share the safety-gating pattern.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- OpenAI, A Practical Guide to Building Agents: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Anthropic, Building Effective Agents: https://www.anthropic.com/research/building-effective-agents

Added (verified canonical):
- ReAct (Yao et al., 2022): https://arxiv.org/abs/2210.03629
- Toolformer (Schick et al., 2023): https://arxiv.org/abs/2302.04761
- Selective prediction with reject option (Geifman & El-Yaniv, 2017): https://arxiv.org/abs/1705.08500
