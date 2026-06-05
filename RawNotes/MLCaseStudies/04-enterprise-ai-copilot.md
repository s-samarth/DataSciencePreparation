# 04. Enterprise AI Copilot — ML System Design Case Study

**Company tags:** Microsoft (Copilot), Google, Salesforce (Agentforce), ServiceNow, Glean, AI startups
**Interview frequency:** Very high for LLM/AI-engineer roles
**Why it matters:** This is the case that tests whether you can turn production LLM experience into engineering. It is *not* a QPS/scale problem like recsys — an enterprise copilot serves tens of thousands of seats, not 100M users. The hard parts are different: **grounding answers in authorized data without hallucinating**, **filtering every retrieval by the calling user's permissions**, **orchestrating tool calls reliably**, and **evaluating quality when there is no ground-truth label**. If you treat it like "RAG + an agent loop," you fail. This document goes to the floor on the control plane, permission-aware retrieval, and label-free evaluation.

---

## How to use this document

Built two ways: a **thinking guide** (Sections 0–8) and a **worked one-hour interview transcript** (Section 9). Read the guide, then watch each transcript line map back to it.

> **The single most important habit:** every decision traces back to a number or constraint you stated up front. But for a copilot the binding numbers are *cost per task, latency per task, and grounding/leakage rates*, not QPS. Say that out loud — recognizing this isn't a throughput problem is itself a senior signal.

> **The copilot-specific habit:** never describe an agent as a "reasoning loop." Make the **control plane** explicit — planner, guarded tool calls with permission checks, state, a validated answer — and treat **grounding** and **permissions** as first-class. "An LLM with tools" is junior. "A planner over a registry of permission-checked tools, with retrieval trimmed to the caller's ACLs, citation-enforced generation, and a groundedness validator before the user sees anything" is senior.

---

## Section 0: The reusable scaffold (learn this ONCE)

```text
1. Clarify        -> turn an ambiguous product goal into a scoped problem + numbers
2. Frame as ML    -> what exactly are we predicting/producing? what's "success"? non-LLM baseline?
3. Data & grounding -> where does truth come from, and WHY do answers go wrong?
4. Baseline       -> simplest shippable thing, then name what breaks
5. System         -> climb the ladder; explain the CONTROL PLANE to the floor
6. Evaluation     -> offline (no ground truth!), online, and the gap between them
7. Deploy         -> serving path / data (indexing) path / feedback path
8. Monitor        -> hallucination, leakage, cost, latency, drift, incident response
```

**The three-path mantra, copilot flavor:** **serving path** (query → route/plan → permission-checked retrieval + tools → grounded generation → validation → answer; latency- and cost-bounded). **Data/indexing path** (enterprise systems → connectors → chunk + embed + attach ACLs → indices; batch + incremental). **Feedback path** (accepts, edits, escalations, thumbs → eval sets + fine-tuning/router signals; *where biased acceptance enters*).

---

## Section 1: Clarify requirements (and pin down NUMBERS)

### Interview prompt
> "Design an enterprise AI copilot that helps sales or operations users answer questions and take actions across internal systems."

### The clarifying questions that actually change the design

| Question | Why it changes the design |
|---|---|
| **Read-only (answer questions) or does it take actions?** | Answering is RAG + grounding. *Taking actions* (update a CRM deal, file a ticket) adds tool-calling, permissions, idempotency, and an "are you sure" / approval step. The blast radius of a wrong action is far bigger than a wrong sentence. |
| **What systems must it reach?** CRM, wikis, tickets, data warehouse? | Each is a connector with its own schema, auth, freshness, and **access-control model**. The number and heterogeneity of systems is the real complexity, not user count. |
| **What's the permission model?** | This is *the* enterprise question. Users must never see/cite data they aren't authorized for. Retrieval must be filtered per-caller. Get this wrong and you have a data-leak incident, not a bug. |
| **What's the cost and latency budget per task?** | LLM tokens cost money; a multi-step agent makes several calls. "< $0.10/task, p95 < 8s, first token < 1.5s" drives model tiering and how much retrieval/iteration you can afford. |
| **What's the tolerance for a wrong answer?** | A sales-prep summary tolerates more than a quoted contract number. This sets how hard you push grounding, citation enforcement, and abstention ("I don't know"). |
| **How many seats and query volume?** | Sets scale — but note it's modest (tens of thousands of seats, single-digit-to-tens QPS), so this is a *quality/cost* problem, not a throughput one. |

> **Junior move:** "I'll build a RAG chatbot with an agent."
> **Senior move:** "Does it take actions or just answer? What's the permission model, and what's my cost/latency budget per task? Because this isn't a QPS problem — it's a grounding, permissions, and cost problem, and those three decide the architecture."

### Pin the numbers (carry these through the ENTIRE answer)

```text
Seats:               50,000 employees
Usage:               ~10 copilot interactions / active user / day
                     50K * 10 / 28800 (8hr workday)  ~= 17 QPS average, ~50 peak
                     -> THROUGHPUT IS TRIVIAL. The constraints are cost & quality.
Corpus:              ~10M internal docs across CRM, wikis, tickets, drive
Tools:               ~20 actions (CRM read/write, ticketing, analytics queries)
Budget per task:     cost < ~$0.10 ; p95 latency < 8s ; first token < 1.5s
Quality bars:        groundedness >= target ; PII/unauthorized-access leakage ~= 0
```

**The cost/latency budget split** (derive out loud — for a multi-step task):
```text
8s p95 budget for a multi-step answer:
  intent routing / planning (1 cheap-model call)      ~0.8s
  permission-aware retrieval (hybrid + ACL trim)       ~0.4s
  tool calls (e.g. 2 CRM/analytics lookups)            ~2.5s  (external APIs dominate)
  grounded generation (big model, streamed)            ~3.0s  (first token < 1.5s)
  groundedness / citation validation (cheap model)     ~0.8s
  overhead                                             ~0.5s

COST: ~4-6 LLM calls per task. Use a CHEAP model for routing/validation and the
      BIG model only for final synthesis -> "model tiering" keeps $/task in budget.
```
Now decisions have reasons: "Routing on a frontier model would blow the cost budget, so a small model routes and a big model only synthesizes; retrieval keeps the context small, which controls both latency and token cost."

---

## Section 2: Frame it as an ML problem

- **Framing:** a **grounded, permission-aware, tool-using agent** — RAG for knowledge, tools for actions, an orchestration control plane on top, and validation before the user sees anything.
- **What is "success"?** There is no single clean label. Define a basket: the user **accepts** the answer/action, completes the task without escalation, no policy/permission violation, and (the hard one) the answer is **factually grounded in authorized sources**. Naming that success is multi-faceted and partly unobservable is itself the senior framing.
- **The non-LLM baseline (always name one):** federated keyword search across systems with deep-links ("here are 5 docs and the CRM page"). It can't synthesize or act, but it's safe, cheap, and a real fallback — and it sets the bar the copilot must beat.

> **Why this framing matters (say it):** separating *knowledge* (retrieval + grounding) from *actions* (tools + permissions) from *orchestration* (the control plane) from *verification* (validators) gives four independently testable surfaces. The biggest failure of naive designs is collapsing all four into "one big prompt," which is impossible to debug, secure, or evaluate.

---

## Section 3: Grounding, permissions, and why answers go wrong (the intellectual core)

In recsys, "labels lie" because of position bias. Here, the analog is **the model is confidently wrong**, and the two root causes are *insufficient/incorrect retrieval* and *the model's willingness to answer anyway*. Plus a constraint recsys never has: **per-user authorization**.

### Where "truth" comes from
- **Retrieved enterprise documents** (the grounding source) — the answer must be supported by these, with citations.
- **Tool/API results** (live CRM fields, analytics queries) — structured ground truth for actions and numbers.
- **The model's parametric knowledge** — useful for language/format, *dangerous for facts* (it's stale and ungrounded). The discipline is: facts come from retrieval/tools, fluency from the model.

### Problem 1 — hallucination & grounding (the headline)
LLMs produce fluent, plausible, wrong answers, especially when retrieval misses. Concrete controls, in order:
- **Retrieval grounding:** put authoritative chunks in context; instruct the model to answer *only* from them.
- **Citation enforcement:** require every factual claim to cite a retrieved chunk; post-process to verify each cited span actually supports the claim (a lightweight NLI / "is this claim entailed by this source?" check). Drop or flag unsupported claims.
- **Abstention:** if retrieval is insufficient, the correct answer is **"I don't know / here's who to ask"**, not a guess. This connects to Google's "sufficient context" work: a large share of RAG errors are the model answering when the retrieved context does *not* contain the answer. Detecting insufficient context and abstaining is a top-leverage fix.
- **Groundedness validation:** a second (cheap) model scores whether the answer is entailed by the cited sources before the user sees it; below threshold → regenerate or abstain.

### Problem 2 — permissions / data leakage (the enterprise-defining constraint)
Every user has different access. The copilot must **never retrieve, reason over, or cite a document the calling user can't access** — even if it's the best answer.
- **Permission-aware retrieval (security trimming):** the ACL filter is applied *inside* the retrieval query (pre-filter), not after, so unauthorized docs never enter the candidate set or the LLM context. Post-filtering is dangerous because the content already touched the model and can leak via the answer.
- **Index-time ACLs:** store each chunk's access groups at indexing time; at query time, intersect with the caller's groups. Handle **ACL freshness** — when someone loses access, stale index ACLs can leak; reconcile periodically and check live for sensitive sources.
- **Tool permissions:** each tool call is authorized as the *user*, not a service account, so the copilot can't do what the user can't.

### Problem 3 — tool-use reliability
Agents loop, call the wrong tool, pass malformed arguments, or leak retrieved data into a tool call. Controls: tool **allowlists**, **schema validation** on arguments, **recursion/step limits**, **cost budgets** per task, **timeouts/retries**, **idempotency keys** for write actions, and an **audit log** of every tool call.

> **Junior says:** "RAG grounds the model so it won't hallucinate, and I'll check permissions."
> **Senior says:** "Grounding isn't binary — I enforce citations, verify each claim is entailed by its source, and *abstain* when context is insufficient (the dominant RAG error). Permissions are applied as a pre-filter inside retrieval so unauthorized docs never reach the model, with index-time ACLs reconciled for freshness. Tools run as the user with schema validation and audit logs."

---

## Section 4: Baseline first — then name exactly what breaks

```text
RUNG 0: Federated keyword search + deep links
   -> search across systems, return docs + links. Safe, cheap, no synthesis.
   BREAKS: no synthesis, no multi-step, no actions; user does the work.
   TRIGGER: users want a synthesized, grounded answer.

RUNG 1: Single-shot RAG (retrieve -> stuff context -> generate with citations)
   -> good for "answer a question from the KB."
   BREAKS: can't take actions; can't decompose multi-step tasks; no tool use;
           one retrieval can't serve a question that needs CRM + wiki + a calc.
   TRIGGER: tasks need actions and multiple systems.

RUNG 2: Single agent with a permission-checked tool registry   <-- PRODUCTION DEFAULT
   -> one LLM plans, calls RAG + tools (each guarded), updates state, validates answer.
   FIXES: multi-step, actions, multiple systems, with guardrails.
   BREAKS / costs: a single agent's planning gets unreliable as tool count grows;
           harder to eval; can loop or overuse tools.
   TRIGGER (only if pushed): many specialized domains -> a planner-with-specialists.

RUNG 3: Supervisor + specialist sub-agents (advanced)
   -> a router/planner delegates to search-agent, CRM-agent, analytics-agent, policy-agent.
   USE WHEN: domains are distinct enough that one prompt/tool-set can't hold them.
   COST: orchestration complexity, more eval surface, higher latency/cost.
```

> **Say this:** "I'd ship Rung 1 single-shot RAG for read-only Q&A first — it's most of the value at the least risk. I move to Rung 2, a single agent over a permission-checked tool registry, when tasks need actions and multiple systems. I only go to Rung 3 multi-agent when the domains are distinct enough that one agent can't hold them — and I'd justify the added latency/cost/eval surface, because multi-agent is often over-engineering."

The senior move here is *resisting* premature multi-agent. Many candidates jump to "supervisor with specialists" to sound sophisticated; the strong answer earns it.

---

## Section 5: The control plane — ONE design, to the floor

Depth on one thing beats naming four. The thing to explain to the floor for a copilot is **the orchestration control plane and how a single tool call is guarded end to end.**

### 5.1 The agent loop, as engineering (not magic)

```text
   USER GOAL
      |
   [ Router / Planner ]  (cheap model): classify intent, decompose into steps,
      |                   decide retrieve vs tool vs answer vs escalate
   [ Step executor ]  loop, bounded by step-limit + cost-budget:
      |   - retrieve (permission-aware) OR
      |   - call a tool (guarded, below) OR
      |   - synthesize
   [ State ]  updated after each step (scratchpad of results, citations, costs)
      |
   [ Grounded synthesis ]  (big model): answer ONLY from gathered context, with citations
      |
   [ Validators ]  groundedness/entailment check, citation check, PII scan, policy check
      |
   ANSWER  (or ABSTAIN / ESCALATE to human)
```
- **ReAct-style** interleaving (reason → act → observe) is the common pattern; name it but keep it bounded. The reliability comes from the *bounds and guards*, not the loop.
- **Why a cheap router + big synthesizer:** cost. Routing/validation on a small model, synthesis on the big one, keeps $/task in budget (the model-tiering point from Section 1).

### 5.2 A single guarded tool call (the part to draw)

```text
   intent to call tool T with args A
        |
   [ allowlist check ]   is T permitted for this user/agent?
        |
   [ permission check ]  authorize as the USER (not a service account)
        |
   [ schema validation ] are args A valid per T's JSON schema? (reject malformed)
        |
   [ dry-run / approval ] for WRITE actions: confirm with user or policy gate
        |
   [ execute w/ timeout + retry + idempotency key ]
        |
   [ audit log ]  who, what, when, args, result -> traceable + reversible
        |
   observation -> back into state
```
Saying "every tool call passes allowlist → user-permission → schema-validation → (approval for writes) → execute-with-idempotency → audit log" is the single most senior thing you can say in this case. It reframes the agent from "prompt magic" to "a guarded distributed system."

### 5.3 Permission-aware retrieval (drawn)

```text
   query + caller's identity/groups
        |
   [ Hybrid retrieval (BM25 + dense) WITH ACL pre-filter: chunk.acl ∩ caller.groups ]
        |   <- unauthorized chunks NEVER enter the candidate set
   [ Rerank ]  (cross-encoder on top-K)
        |
   authorized, relevant chunks -> context
```
(The retrieval internals — chunking, hybrid, rerank — are the subject of case 05; here the *new* idea is the ACL pre-filter.)

### 5.4 The full architecture diagram (draw this)

```text
                         USER GOAL (+ identity)
                                |
                    [ Router/Planner (cheap model) ]
                                |
           +--------------------+--------------------+
           |                    |                    |
  [ Permission-aware RAG ]  [ Guarded tools ]   [ Escalate to human ]
   (ACL pre-filter,          (allowlist, user-auth,
    hybrid, rerank)           schema, approval, audit)
           |                    |
           +-------- State (scratchpad, citations, cost) --------+
                                |
                    [ Grounded synthesis (big model) ]
                                |
        [ Validators: groundedness/entailment, citations, PII, policy ]
                                |
                     ANSWER / ABSTAIN / ESCALATE
                                |
              (accepts, edits, escalations, thumbs, traces logged)
                                |
        [ Feedback path -> eval sets + router/grounding improvements ]
```

---

## Section 6: Evaluation — without ground truth (the hard part)

There is no labeled "correct answer" for open-ended enterprise questions. This is the section that separates LLM-experienced candidates from the rest.

### Offline evaluation (no ground truth)
- **Curated golden sets:** experts write representative tasks with rubric-based acceptance criteria (and, where possible, reference answers/sources). Small, high-quality, version-controlled.
- **Component metrics (decompose the pipeline):**
  - *Retrieval:* context precision/recall, "did the answer-bearing chunk get retrieved?" (RAGAS-style context metrics).
  - *Grounding:* **faithfulness/groundedness** — is every claim entailed by the cited context? (RAGAS faithfulness; NLI checks.)
  - *Answer:* answer-relevance, correctness vs rubric.
  - *Agent/trace:* did each step succeed? did tool args validate? was the right tool chosen? **trace-based eval** over the full trajectory, not just the final string.
- **LLM-as-judge (the workhorse, with caveats):** a strong model scores answers against a rubric or compares two answers pairwise. Name the known biases — **position bias** (judge favors the first answer), **verbosity bias** (favors longer), **self-preference** (favors its own family) — and the mitigations: randomize order, use pairwise + rubric prompts, calibrate the judge against human labels on a sample, and human-audit a slice. Cite MT-Bench.

### Online metrics
- **Product:** task-completion / accepted-answer rate, edit distance on accepted answers, follow-up rate, **escalation/deflection rate**, active usage.
- **Guardrails (must hold):** sampled **hallucination rate** (human-audited), **PII / unauthorized-access leakage** (target ~0 — a hard gate), **cost/task**, p95 latency, **unauthorized/failed tool-call rate**.

### The offline→online gap (THE talking point)

> **Classic question: "Offline groundedness/faithfulness went up but online task-completion dropped (or escalations rose). Why?"**

1. **Judge miscalibration** — your LLM-judge rewards a style users don't actually find useful; offline "faithful" ≠ online "helpful."
2. **Eval-set vs real-query distribution mismatch** — the golden set covers clean questions; real users ask messy, multi-system ones the retrieval can't serve.
3. **Automation/sycophancy bias** — users *accept* confident wrong answers (acceptance is a biased label), so a model that sounds more confident scores better online while being less correct.
4. **Latency/cost regression** — a more thorough pipeline got slower; users abandon, completion drops.
5. **Abstention tuned wrong** — more "I don't know" raises offline faithfulness but online feels unhelpful → escalations rise.

**The lesson:** with no ground truth, *every* metric is a proxy. Triangulate: component metrics + LLM-judge (calibrated) offline as a filter, human-audited guardrails, and online behavior as the arbiter — while remembering acceptance itself is biased.

### A concrete A/B test (fully specified)

```text
Hypothesis:        The agentic copilot (Rung 2) raises task-completion and lowers
                   escalation vs single-shot RAG, without raising hallucination,
                   leakage, cost/task, or latency past thresholds.
Unit of randomization: user (consistent experience; teams share context, watch leakage).
Control / Treatment:   single-shot RAG / agent-with-tools.
Primary metric:    accepted-task-completion rate (no escalation, no edit-to-fix).
Guardrails (HARD): PII/unauthorized-access leakage (~0, any breach = halt), sampled
                   hallucination rate (human-audited), cost/task, p95 latency,
                   failed/unauthorized tool-call rate.
Ramp:              internal dogfood -> 1% -> 5% -> 20% -> 50%; security review at each.
Sample size / MDE: usage is low, so expect a LONGER run to reach power; pre-register MDE.
Duration:          >= 2-4 weeks (low volume + learning effects as users adapt to the tool).
Decision/rollback: ship if completion up, escalation down, ALL guardrails hold;
                   leakage breach = immediate halt, not a gradual rollback.
```

Note two copilot-specific points: leakage is a **hard halt** guardrail (not a "regression"), and low volume means experiments take **longer** to reach significance.

---

## Section 7: Deployment & serving (the three paths)

### Serving path (latency- and cost-bounded)
```text
goal -> router (cheap) -> [permission-aware RAG | guarded tools] (looped, bounded)
     -> grounded synthesis (big) -> validators -> answer/abstain/escalate
```
- Model tiering (cheap router/validator, big synthesizer) to hit $/task.
- Stream the answer (first token < 1.5s) while validators run on the buffered output before final commit (or validate-then-stream for high-risk).
- Stateless services + a session/state store; per-task step and cost budgets enforced.

### Data / indexing path (batch + incremental)
```text
enterprise systems -> connectors -> clean -> chunk -> embed -> attach ACLs + metadata
   -> inverted + vector indices -> incremental sync (freshness) + ACL reconciliation
```
- **ACLs attached at index time** and reconciled (the freshness risk in Section 3).
- Incremental connectors so a new wiki edit / CRM update is searchable quickly.

### Feedback path (where biased acceptance enters)
```text
accepts / edits / escalations / thumbs / full traces -> eval-set curation
   -> router & retrieval tuning, prompt/grounding fixes, fine-tuning candidates
```
- Traces (not just final answers) are logged so failures are debuggable step-by-step.

### Rollout discipline
```text
internal dogfood -> shadow (run new pipeline, don't show) -> canary -> A/B (ramp)
   with a SECURITY review gate at each step (leakage is unacceptable, not just bad)
```

---

## Section 8: Monitoring, retraining, incident response

- **Monitor:** sampled hallucination/groundedness rate, **leakage / unauthorized-access attempts** (alert immediately), cost/task and token usage, p95 latency, tool-call success/failure and which tools, escalation rate, retrieval coverage, and **prompt-injection attempts** (a doc or tool result trying to hijack the agent — sanitize/quarantine retrieved content).
- **"Retraining":** mostly *not* model weights — it's **prompt/router iteration, retrieval tuning, eval-set growth**, and occasionally fine-tuning the router or a small grounding-check model. Treat prompts and the tool registry as versioned, tested artifacts.
- **Fallback:** on LLM/tool failure or low groundedness → abstain and **degrade to federated search + deep links** (Rung 0). Never emit an ungrounded answer to a high-risk query; "here are the sources" beats a confident guess.
- **Incident response:** for a *leakage* incident, halt the feature, not just roll back; audit the trace log to scope exposure. For quality regressions, diff prompts/model/retrieval versions against the last release and replay the bad-example trace bank.

---

## Section 9: The worked one-hour interview (full transcript)

---

**[00:00 — The prompt]**

**INTERVIEWER:** Design an enterprise AI copilot for sales or ops users that answers questions and takes actions across internal systems.

**YOU:** A few scoping questions first, because they change the architecture more than model choice does. First, is it read-only or does it take actions like updating a CRM deal — actions add tool-calling, permissions, and an approval step, and the blast radius of a wrong action is much larger than a wrong sentence. Second, what's the permission model — because users must never see or cite data they aren't authorized for, and that decides how retrieval is filtered. Third, what's my cost and latency budget per task. And fourth, how many systems must it reach. Notice I'm not asking about QPS — at enterprise seat counts this isn't a throughput problem, it's a grounding, permissions, and cost problem.

**INTERVIEWER:** It answers and takes some actions. Strict per-user permissions. Budget say 10 cents and 8 seconds per task. CRM, wikis, tickets, a data warehouse. 50K seats.

**YOU:** Good.
```
50K seats, ~10 tasks/user/day -> ~17 QPS avg, ~50 peak  (throughput trivial)
~10M docs, ~20 tools, cost < $0.10/task, p95 < 8s, first token < 1.5s
leakage ~= 0 (hard gate), groundedness above target
```
The binding constraints are cost, latency, grounding, and leakage. So my design centers on a control plane with permission-aware retrieval and validation, and model tiering to hit the cost budget.

---

**[00:07 — Framing & the baseline ladder]**

**YOU:** I frame it as a grounded, permission-aware, tool-using agent: RAG for knowledge, tools for actions, an orchestration layer, and validators before the user sees anything. My non-LLM baseline is federated search with deep links — safe, cheap, and my fallback. Then I climb: single-shot RAG for read-only Q&A, which is most of the value at least risk; then a single agent over a permission-checked tool registry when tasks need actions and multiple systems; and only if domains are truly distinct, a supervisor with specialist sub-agents. I'd actually resist multi-agent unless pushed — it's often over-engineering that adds latency, cost, and eval surface for little gain.

**INTERVIEWER:** Why not go multi-agent from the start? It's more capable.

**YOU:** Capability isn't free. Every extra agent multiplies the eval surface, adds orchestration latency, and makes failures harder to trace. A single well-instrumented agent over a good tool registry handles most enterprise tasks. I'd only split into specialists when one prompt and tool-set genuinely can't hold the domains, and I'd justify the cost. Earning the complexity is the point.

---

**[00:15 — Grounding & permissions]**

**INTERVIEWER:** How do you stop it from hallucinating and from leaking data?

**YOU:** Two separate problems. Hallucination: grounding isn't binary. I retrieve authoritative chunks and instruct the model to answer only from them; I enforce citations and verify each claim is actually entailed by its cited source with a lightweight NLI check, dropping unsupported claims; and crucially I make abstention first-class — if retrieved context is insufficient, the right answer is "I don't know, here's who to ask," not a guess. Google's sufficient-context work shows a large share of RAG errors are the model answering when the context doesn't contain the answer, so detecting insufficiency and abstaining is top leverage. Then a cheap second model scores groundedness before the user sees anything.

Leakage is the enterprise-defining constraint. I apply the permission filter as a pre-filter inside the retrieval query — unauthorized chunks never enter the candidate set or the model's context. Post-filtering is dangerous because the content already touched the model and can leak through the answer. ACLs are attached at index time and reconciled for freshness, because when someone loses access, stale index ACLs would leak. And every tool runs authorized as the user, not a service account, so the copilot can't do what the user can't.

**INTERVIEWER:** Walk me through one tool call end to end.

**YOU:** Allowlist check — is this tool permitted for this user. Permission check — authorize as the user. Schema validation — are the arguments valid per the tool's JSON schema, reject if malformed. For write actions, a dry-run or approval gate. Execute with a timeout, retry, and an idempotency key so a retry doesn't double-write. Then an audit log of who, what, when, args, and result, so it's traceable and reversible. The reliability comes from those guards and the step and cost budgets, not from the LLM's reasoning.

---

**[00:28 — Evaluation without ground truth]**

**INTERVIEWER:** There's no labeled correct answer. How do you evaluate this?

**YOU:** Right, that's the hard part. Offline, I curate small expert golden sets with rubric acceptance criteria, and I decompose the pipeline: retrieval context precision and recall, grounding faithfulness via entailment checks, answer-relevance, and trace-based eval over the whole trajectory — did each step succeed, were tool args valid, was the right tool chosen. The workhorse is LLM-as-judge, but I'd name its biases — position, verbosity, self-preference — and mitigate with randomized-order pairwise comparison, rubric prompts, calibration against human labels on a sample, and a human-audited slice. Online, primary is accepted task completion and escalation rate, with hard guardrails on sampled hallucination rate, leakage, cost, and latency.

**INTERVIEWER:** Suppose offline faithfulness goes up but online completions drop and escalations rise. Why?

**YOU:** A few likely causes. My judge may reward faithful-but-unhelpful answers — offline faithful isn't online helpful. The golden set may be cleaner than real messy multi-system queries. Users may be accepting confident wrong answers, so acceptance is itself a biased label. Or I tuned abstention too conservatively, so more "I don't know" raised faithfulness offline but feels unhelpful online and pushes escalations up. I'd recalibrate the judge against humans, check eval-versus-real query distribution, and retune the abstention threshold. The meta-point is that without ground truth every metric is a proxy, so I triangulate rather than trust one number.

---

**[00:40 — A/B, serving, monitoring]**

**INTERVIEWER:** Design the experiment and tell me what's in production.

**YOU:** Randomize by user, control is single-shot RAG, treatment is the agent. Primary metric accepted task completion with escalation down. Guardrails are hard: leakage near zero — any breach halts the experiment, it's not a gradual rollback — plus sampled hallucination, cost, latency, and failed-tool-call rate. Because usage volume is low, the test runs longer to reach power, two to four weeks, and I ramp through internal dogfood with a security review at each step.

In production, three paths. Serving: cheap router, permission-aware retrieval and guarded tools in a bounded loop, big-model grounded synthesis, validators, then answer, abstain, or escalate — model-tiered to hit the cost budget, streamed for first-token latency. Indexing path: connectors chunk and embed docs, attach ACLs at index time, incremental sync plus ACL reconciliation. Feedback path: accepts, edits, escalations, and full traces feed eval sets and prompt and retrieval tuning. Monitoring watches hallucination, leakage and unauthorized-access attempts with immediate alerts, cost, latency, tool success, escalation, and prompt-injection attempts in retrieved content. Fallback degrades to federated search and deep links — never an ungrounded answer to a high-risk query.

---

**[00:53 — The close]**

**INTERVIEWER:** Anything to add?

**YOU:** To restate: this isn't a scale problem, it's grounding, permissions, cost, and evaluation. I optimize accepted task completion while holding leakage at zero, hallucination low, and cost and latency in budget, via a guarded control plane, permission pre-filtered retrieval, citation-enforced grounding with abstention, and triangulated label-free evaluation. Concretely, this is exactly what I built with Seller Copilot — a multi-agent system over enterprise tools with RAG and a real evaluation harness — so the control-plane guards, the grounding-versus-hallucination tradeoffs, and the no-ground-truth eval problem are production ground I've actually operated, not whiteboard theory.

**INTERVIEWER:** Strong answer.

---

> **Why this transcript works (study these moves):**
> 1. **Reframed away from QPS** — named cost/grounding/permissions as the real constraints.
> 2. **Resisted premature multi-agent** — earned complexity instead of cargo-culting it.
> 3. **Grounding as a pipeline, not a property** — citations, entailment, abstention, validation.
> 4. **Permission pre-filter** — the enterprise-defining constraint, with ACL freshness.
> 5. **A tool call to the floor** — allowlist→auth→schema→approval→idempotency→audit.
> 6. **Eval without ground truth** — component metrics + calibrated LLM-judge + traces.
> 7. **Leakage as a hard halt** — not a "regression"; security review gates.
> 8. **Closed on real shipped work (Seller Copilot)** — led with it.

---

## Section 10: Junior vs Senior — the highest-leverage contrast

| Decision | Junior answer | Senior answer |
|---|---|---|
| The problem | "Build a RAG chatbot with an agent." | "Not a QPS problem — grounding, permissions, cost, and label-free eval are the constraints." |
| Architecture | "Multi-agent supervisor with specialists." | "Single-shot RAG → single guarded agent; earn multi-agent only when domains truly diverge." |
| Hallucination | "RAG grounds it." | "Enforce citations, verify entailment, *abstain* on insufficient context (the dominant RAG error)." |
| Permissions | "Check the user's role." | "ACL *pre-filter* inside retrieval so unauthorized docs never reach the model; index-time ACLs reconciled for freshness." |
| Tools | "The agent calls tools." | "Allowlist → user-auth → schema-validate → approval(writes) → idempotent execute → audit log." |
| Cost | (ignores) | "Model tiering — cheap router/validator, big synthesizer — to hit $/task." |
| Eval | "Measure accuracy." | "No ground truth — component metrics + calibrated LLM-judge + trace-based eval; acceptance is biased." |
| Guardrails | "Track quality." | "Leakage ~0 is a hard halt, not a regression; security review at each ramp." |
| Reliability | "Add retries." | "Step/cost budgets, recursion limits, idempotency, and traces so failures are debuggable." |

---

## Section 11: One-page cheat sheet (whiteboard recall)

```text
SCAFFOLD: Clarify -> Frame -> Grounding/Perms -> Baseline -> Control plane -> Eval -> Deploy -> Monitor

NUMBERS:  seats (NOT QPS) -> throughput trivial. Binding = $/task, latency/task, grounding, leakage.
          ~4-6 LLM calls/task -> MODEL TIERING (cheap router/validator + big synthesizer).
          budget split: route 0.8s | retrieve 0.4s | tools 2.5s | gen 3s | validate 0.8s.

LADDER:   federated search -> single-shot RAG -> single guarded agent -> (earn) multi-agent.
          RESIST premature multi-agent.

GROUNDING (core): facts from retrieval/tools, fluency from model.
          enforce citations -> verify each claim ENTAILED by source -> ABSTAIN if context
          insufficient (the dominant RAG error) -> groundedness validator before user sees it.

PERMISSIONS (enterprise core): ACL PRE-FILTER inside retrieval (not post) ->
          unauthorized chunks never reach the model. index-time ACLs, reconcile freshness.
          tools run AS THE USER.

TOOL CALL (to the floor): allowlist -> user-auth -> schema-validate -> approval(writes)
          -> execute(timeout/retry/idempotency) -> audit log. + step & cost budgets.

EVAL (no ground truth): golden sets + component metrics (context P/R, faithfulness, answer-rel)
          + LLM-as-judge (BIASES: position/verbosity/self-pref; mitigate: pairwise+rubric+calibrate)
          + TRACE-based agent eval. online: completion, escalation; guardrails: hallucination,
          LEAKAGE(~0 hard gate), cost, latency.
          offline-up/online-down: judge miscalibration, eval!=real dist, automation bias, abstain mis-tuned.

3 PATHS:  serving (router->retrieve/tools->synth->validate) | indexing (chunk+embed+ACL, reconcile)
          | feedback (accepts/edits/escalations/TRACES -> eval sets + tuning)

MONITOR:  hallucination, LEAKAGE + injection attempts, cost, latency, tool success, escalation.
          fallback: abstain -> federated search + links. NEVER ungrounded high-risk answer.
          leakage incident = HALT, audit traces to scope exposure.
```

---

## Section 12: Follow-up questions the interviewer may ask

- **What scales / what's the real bottleneck?** Not QPS — cost/task and quality. Control token cost via model tiering and tight retrieval; cache embeddings and frequent answers; the corpus + connector heterogeneity is the complexity.
- **How do you handle a brand-new system/connector (cold start)?** Start it as a retrieval-only source with ACLs; add tools once schemas/permissions are modeled and tested; dogfood before exposing.
- **How do you set the abstention threshold?** Tune the groundedness/sufficient-context score against the cost of a wrong answer for that task type; high-risk (numbers, contracts) abstains earlier.
- **Offline up, online down — why?** Judge miscalibration, eval-vs-real distribution gap, automation/sycophancy bias in acceptance, latency/cost regression, abstention mis-tuned.
- **How do you prevent prompt injection?** Treat retrieved docs and tool outputs as untrusted; sanitize/quarantine, never let retrieved text issue tool calls, separate system instructions from data, and monitor for injection attempts.
- **How do you keep it reliable?** Step/cost budgets, recursion limits, schema validation, idempotency for writes, timeouts/retries, and full traces for debugging.
- **When multi-agent vs single?** Only when domains are distinct enough that one prompt/tool-set can't hold them; justify the extra latency/cost/eval surface.

---

## Section 13: Common mistakes (anti-patterns to avoid)

- Treating it as a QPS/scale problem instead of a cost/grounding/permissions/eval problem.
- Jumping to multi-agent to sound sophisticated, without earning the complexity.
- Describing the agent as a "reasoning loop" instead of a guarded control plane with bounds.
- Treating grounding as binary ("RAG fixes hallucination") — no citation enforcement, entailment check, or abstention.
- Post-filtering permissions (content already reached the model) instead of pre-filtering inside retrieval.
- Forgetting ACL freshness, idempotency for writes, audit logs, and prompt-injection defense.
- Evaluating with a single accuracy number, ignoring no-ground-truth reality and LLM-judge biases.
- Treating acceptance as truth (automation/sycophancy bias).
- Treating deployment as "an endpoint" instead of serving/indexing/feedback paths with a security gate.

---

## Section 14: Transfer — what mastering the copilot unlocks

| Problem | What changes vs copilot | What stays identical |
|---|---|---|
| **Production RAG (case 05)** | the retrieval *internals* (chunking, hybrid, rerank) are the focus | grounding, citation, abstention, RAG eval |
| **LLM eval platform (case 06)** | the eval *is* the product | LLM-as-judge, no-ground-truth metrics, traces |
| **Support agent (case 07)** | external customers, deflection/escalation economics | control plane, tool guards, trace eval |
| **LLM serving (case 08)** | infra/throughput is the focus | the cost/latency budget mindset |
| **Safety gateway (case 13)** | the guardrails *are* the system | injection defense, validators, abstention |
| **Document intelligence (case 14)** | structured extraction over docs | grounding/citation, confidence + human review |

The leverage: **the control plane + grounding + label-free eval recur in every LLM case.** Master them here and 05, 06, 07, and 13 are reconfigurations of the same parts.

---

## Sources
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- OpenAI, A Practical Guide to Building Agents: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Anthropic, Building Effective Agents: https://www.anthropic.com/research/building-effective-agents
- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022): https://arxiv.org/abs/2210.03629
- RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023): https://arxiv.org/abs/2309.15217
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023, judge biases): https://arxiv.org/abs/2306.05685
- Google Research, Deeper Insights into RAG: the role of sufficient context (abstention): https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/
