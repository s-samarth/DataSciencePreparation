# 06. LLM Evaluation and Monitoring Platform

**Company tags:** Google, Anthropic, OpenAI-style teams, Microsoft, Scale, AI startups
**Interview frequency:** Very high for LLM / applied-AI infra roles
**Why it matters:** Every LLM team eventually drowns in "is the new model better?" with no trustworthy way to answer. This case tests whether you can build the *measurement system itself* — and whether you understand its deepest trap: **your evaluator is also a model that can be wrong, biased, drift, and be gamed.**

---

## 0. How to use this doc

Built two ways; read it twice.

1. **As a thinking guide.** The headers are the whiteboard order. Internalize the *order* and the *triggers* for climbing each ladder rung.
2. **As a worked transcript.** Section 11 is a full timestamped hour with a pushing interviewer. Cover the `YOU:` lines and answer from memory.

The one idea to carry out: **"evaluate the evaluator." An LLM-judge is a model with its own error rate, biases, and drift. A senior eval platform is mostly the machinery that makes the judge trustworthy — calibration against humans, bias controls, versioning, and CI gating — not the judge prompt itself.** If you treat the judge as ground truth, you have built a confident lie generator.

Scaffold (identical across all cases in this set):

```
Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor
```

For this case there is a delicious recursion: the "Model" is the judge, and "Eval" means evaluating the judge against humans. Keep that straight and you will sound like someone who has run this in anger.

---

## 1. The reusable scaffold, stated once

| Phase | The question you are answering |
|---|---|
| Clarify | What are we measuring, for whom, to gate what decision? |
| Frame | What is the learnable target, and what is the non-ML baseline? |
| Data / Labels | Where does ground truth come from, and how scarce is it? |
| Baseline | Simplest thing that ships, and what breaks it? |
| Model | The judge: how it scores, and how we trust it. |
| Eval | Judge-vs-human agreement; the offline/online gap; one A/B. |
| Deploy | Three paths: serving, data, feedback. Plus CI gating. |
| Monitor | What pages someone, and the fallback. |

---

## 2. Clarify requirements (scripted)

Spend three minutes here. Each question changes the design.

| Question | Why it changes the design |
|---|---|
| "Are we gating *releases* (offline CI), monitoring *production* live, or both?" | Offline gating is a batch CI problem with curated sets and strong judges. Online monitoring is a sampling + async-scoring + alerting problem with cheap judges. They are different subsystems; most real platforms need both. I'll design both. |
| "Single-turn answers, or multi-step *agent* traces?" | Single-turn: grade the final output. Agents: grade the *trajectory* — tool calls, intermediate decisions, recovery from failure. Agent eval needs trace storage and replay, a much bigger lift. |
| "Do we have human labels, and how many?" | This sets how much we can trust and *calibrate* the judge. With zero human labels you cannot validate the judge at all, which is a red flag I'd raise. |
| "Are we comparing model *versions* (A vs B) or scoring absolute quality?" | Comparison favors pairwise / Elo ranking, which is more reliable than absolute scores. Absolute scoring favors rubric scores with thresholds. Most "is the new model better" questions are really pairwise. |
| "What decision does a bad eval cause?" | If a green eval ships a model to 5M users, the cost of a **false pass** is enormous and I weight precision of the gate accordingly. This sets the guardrail philosophy. |
| "What's the eval budget?" | LLM-judge calls cost money and time. At 10M calls/day I cannot judge everything; I must sample. The budget sets sample rate and judge-model tiering. |

**Numbers I will commit to and carry through:**

- **Scope:** ~40 LLM-powered apps across the org, one shared platform.
- **Production volume:** ~10M LLM calls/day (traces) aggregate.
- **Online judging:** sample ~2% → ~200K judge calls/day, scored async.
- **Golden sets:** 200-2000 curated, versioned examples per app.
- **Judge trust target:** judge-human agreement (Cohen's kappa) > 0.6 ("substantial") before a judge is allowed to gate.
- **CI gate:** offline eval suite runs in < 10 min on a PR and blocks merge on regression.
- **Online alert lag:** production quality regression surfaced within ~5 min.

### "Latency" budget, derived out loud

This platform is mostly *not* user-facing, so the budgets are turnaround budgets:

```
OFFLINE (CI gate):  full golden-set eval must finish < 10 min
  -> parallelize judge calls; cap golden set size per app; cache
     judgments for unchanged (prompt, output) pairs.

ONLINE (monitoring): trace -> async queue -> sampled judge -> metric store -> alert
  -> end-to-end < 5 min so a bad deploy is caught within one coffee.
  Judging is OFF the user's serving path entirely (async), so it
  never adds latency to the product.
```

The senior point: **eval must never sit on the user's request path.** It samples from logs and scores asynchronously. If you put the judge inline you've coupled product latency to your eval infra.

### Storage back-of-envelope

10M traces/day × ~5KB/trace (prompt + output + tool calls + context refs + metadata) = **~50 GB/day**. Retain 90 days hot for regression triage = ~4.5 TB; archive older to cold storage. Agent traces are larger (full trajectories), so budget 2-3x for agent-heavy apps. This is a logging/storage system as much as an ML one.

---

## 3. Frame as an ML problem

- **Framing:** learn (or prompt) a scorer that predicts a *human's* judgment of task success, quality, and policy compliance — then wrap it in machinery that keeps it honest.
- **The target is the human label.** The judge is an *approximation* of expert human judgment. That reframing is the whole game: the judge is a model with a measurable error rate against humans, not an oracle.
- **Why this framing wins:** it forces you to ask "how often does my judge agree with a human?" — which is the question that makes the platform trustworthy. Candidates who skip this build a fast way to be confidently wrong.
- **Non-ML baseline:** a human-QA spreadsheet — sample N outputs before each launch, have experts grade them. Slow, narrow, expensive, but it is the **source of truth you calibrate the judge against**, so it never fully goes away; it shrinks to an audit.

---

## 4. Data and labels — the scarcity problem, head-on

The domain's hard problem: **high-quality human labels are scarce and expensive, but they are the only anchor of truth.** Everything else (judges, heuristics) is a way to *amplify* scarce labels, and amplification without calibration is just confident error.

Sources of signal, increasing cost and quality:

1. **Deterministic checks (free, narrow):** schema valid? JSON parses? required citation present? forbidden PII absent? These catch format and policy *invariants* with zero ambiguity. Run them on 100% of traffic.
2. **Implicit production signal (cheap, biased):** thumbs up/down, retries, escalation to human, conversation abandonment. Abundant but biased — most users never rate, and "no thumbs" is not "good."
3. **LLM-as-judge (scalable, needs calibration):** a strong model scores outputs against a rubric. This is the amplifier — it turns 500 human labels into judgments on millions of traces. **Only trustworthy after calibration (section 7).**
4. **Expert human labels (expensive, the anchor):** the golden set. A few hundred to a few thousand examples per app, with rubric scores or pairwise preferences from people who know the domain. This is ground truth; guard it, version it, and refresh it.

### The label biases to name out loud

- **Selection bias in implicit signal:** people who rate are not representative (extremes rate more). Don't optimize the judge toward thumbs.
- **Golden-set staleness:** the set was built on an old product/corpus; the live distribution drifted, so a green CI no longer predicts production. Refresh the golden set from real recent traffic on a schedule.
- **Label contamination:** if your golden answers leak into a model's training data, evals are inflated. Keep a held-out, never-published set for the highest-stakes gates.

---

## 5. Baseline -> why it breaks -> next rung

| Rung | What it is | Why it breaks -> trigger to climb |
|---|---|---|
| 0 | Manual human QA on a sampled spreadsheet before launch. | Doesn't scale, slow, narrow coverage. Trigger: more than a couple of releases per week, or production incidents between releases. |
| 1 | Deterministic checks (schema, citation present, banned-word, latency/cost thresholds) in CI. | Can't judge *nuanced quality* — a well-formatted answer can still be wrong or unhelpful. Trigger: regressions that pass all format checks. |
| 2 | **LLM-as-judge with rubrics, calibrated against a human golden set, wired into CI as a release gate + sampled online.** | Judge has biases and drifts; absolute scores are noisy. Trigger to refine: you need reliable A-vs-B model comparison, or the judge disagrees with humans too often. |
| 3 | **Pairwise / Elo model ranking + trace-based agent eval + continuous judge re-calibration.** | More infra (replay, trace store, Elo bookkeeping). Trigger: comparing many model versions, or grading multi-step agents where the final answer alone is insufficient. |

Earn rung 2 by showing why deterministic checks (rung 1) miss quality. Earn rung 3 by showing why absolute judge scores are unreliable for ranking and why agents need trajectory grading.

---

## 6. The architecture, explained to the floor

```
        Releases (CI)                         Production (live)
            |                                      |
   PR triggers eval                      10M calls/day -> trace log
            |                                      | sample 2%
   Golden set (versioned)                  async judge queue
            |                                      |
   +--------+---------+                    +-------+--------+
   | deterministic    |                    | deterministic  | (100%)
   | checks           |                    | checks         |
   +--------+---------+                    +-------+--------+
            |                                      |
   LLM-judge (strong model)               LLM-judge (cheap model)
   rubric scores + pairwise vs            rubric scores, sampled
   current production model                       |
            |                                      |
   Elo / aggregate vs baseline            metric store (per app,
            |                              per cohort, per version)
   RELEASE GATE: block merge                       |
   if regression on quality,              drift + regression alerts
   safety, latency, or cost                        |
            |                              dashboards + paging
   ship -> canary -> ramp  <----- feedback ------- |

   CALIBRATION LOOP (the part that makes any of this trustworthy):
   sample judge outputs -> route to human reviewers -> measure
   judge-human agreement (kappa) -> if it drops, re-prompt / re-tune
   the judge or pull it from gating duty.
```

### The judge — how it scores, and how I make it trustworthy

**Scoring modes:**
- **Rubric (absolute):** the judge scores an output on named dimensions — groundedness, relevance, completeness, safety, tone — each with explicit criteria and ideally a 1-5 anchor or pass/fail. Decompose; never ask for one blurry "quality" number.
- **Pairwise (relative):** show the judge output A and output B, ask which is better. **Pairwise is more reliable than absolute** because "is this a 4 or a 5?" is noisier than "is A better than B?" — humans and LLM-judges are both better at comparisons than at calibrated absolute scores.
- **Reference-based vs reference-free:** with a golden answer, the judge checks equivalence (offline). Without one, it checks self-consistency / faithfulness against provided context (works live). (The RAGAS-style faithfulness/relevance decomposition from the RAG case is the reference-free toolkit.)

### The judge's biases — name them, then control them

This is the section that signals seniority. LLM-judges have systematic, documented biases:

- **Position bias:** the judge favors whichever answer is shown first (or last). **Control:** run each pairwise comparison both orders and average; if the verdict flips, it's a tie.
- **Verbosity bias:** the judge rewards longer answers regardless of quality. **Control:** length-balance the rubric, penalize unsupported padding, monitor score-vs-length correlation.
- **Self-preference bias:** a judge favors outputs from its own model family. **Control:** use a judge from a different family than the model under test where stakes are high; cross-check with humans.
- **Sycophancy / leniency drift:** judges tend to be too generous. **Control:** calibrate the threshold against human pass rates, not against the raw judge distribution.

### The calibration loop — "evaluate the evaluator"

The judge is only allowed to gate releases if it **agrees with humans** at a measured rate. Concretely:

- Maintain a human-labeled subset. Score it with the judge. Compute **agreement**: Cohen's kappa (chance-corrected) for categorical pass/fail, or correlation / pairwise-accuracy for rankings. Require kappa > ~0.6 ("substantial") before the judge gates anything important.
- **Continuously re-sample** judge outputs to humans (a small ongoing %). If agreement decays — because the judge model was upgraded, the rubric drifted, or the input distribution shifted — alert and pull the judge from gating until re-validated. This is judge drift, and it is as real as data drift.

### Elo for model ranking

When comparing many model versions, run pairwise judge battles between versions on a fixed prompt set and maintain an **Elo rating** (or Bradley-Terry) per version, like a chess ladder. This gives a single stable, transitive ranking that's robust to the absolute-score noise, and it's exactly how public LLM leaderboards rank models. New candidate model? Play it against the incumbent on the eval set; promote only if Elo clears the incumbent by a margin beyond noise.

### Trace-based agent eval

For agents, the final answer is not enough — a correct answer reached by a dangerous tool call, or by luck after three wrong turns, should not pass. Store the full trajectory: prompt, retrieved context, every tool call + output, intermediate decisions, final response, cost, latency. Then grade: **task success** (did the env reach the goal state?), **tool correctness** (right tools, right args), **policy compliance** (no forbidden action), **efficiency** (steps/cost), and **recovery** (did it recover from a failed tool call?). Where possible grade against **environment state** (did the ticket actually get refunded in the sandbox?) rather than the model's claim — state-based grading is far more trustworthy than judging the transcript.

### Canonical references (verified)

- LLM-as-judge + MT-Bench/Chatbot Arena, judge biases (position/verbosity/self-enhancement) — Zheng et al., 2023: https://arxiv.org/abs/2306.05685
- RAGAS (reference-free generation eval) — Es et al., 2023: https://arxiv.org/abs/2309.15217
- HELM (holistic, multi-metric eval) — Liang et al., 2022: https://arxiv.org/abs/2211.09110
- G-Eval (LLM-judge with chain-of-thought + form-filling) — Liu et al., 2023: https://arxiv.org/abs/2303.16634
- Cohen's kappa (inter-rater agreement) — overview: https://en.wikipedia.org/wiki/Cohen%27s_kappa
- Anthropic, Demystifying Evals for AI Agents: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- OpenAI Evals: https://github.com/openai/evals

---

## 7. Evaluation — yes, you evaluate the eval platform

The meta-level: how do you know the *platform* is good? Its job is to catch regressions and avoid false passes. So measure it like a classifier of "is this release good."

- **False pass rate (the headline guardrail):** releases the gate approved that caused a production incident. Minimize this above all; a false pass ships a bad model to everyone.
- **False fail rate:** good releases the gate blocked. Too high and engineers stop trusting/using the gate (and route around it), which is its own failure.
- **Judge-human agreement (kappa):** the trust metric for every judge in the system, tracked over time per app.
- **Coverage:** fraction of production query types represented in the golden set. Gaps here are where regressions hide.

### The offline-to-online gap, including the classic trap

**"The CI eval was green, but production quality dropped after launch."** Enumerate causes:

1. **Golden-set staleness / coverage gap.** The new failure mode isn't in the golden set, so CI couldn't see it. Most common cause.
2. **Judge-distribution shift.** The judge was calibrated on old-style outputs; the new model writes differently (e.g., more concise) and the judge mis-scores it — sometimes punishing a genuine improvement (verbosity bias inverted).
3. **Judge-user mismatch.** The rubric optimizes thoroughness; users wanted brevity. CI rewards the wrong thing.
4. **Contamination.** Golden answers leaked into training; CI is inflated and doesn't generalize.
5. **Metric-not-the-goal.** You gated on judge score, but the business metric is deflection/task-success, and they decoupled. The proxy improved while the target fell.
6. **Sampling bias online.** Your 2% online sample over-represents one cohort, masking a regression in the rest.

### One fully specified A/B test

The platform's own claim — "this gate prevents incidents" — deserves a test, but the more common interview ask is the A/B that the *gate itself authorizes*. I'll give that:

- **Hypothesis:** model v2 (passed the CI gate) increases task success without raising unsafe-output rate.
- **Unit:** user, sticky.
- **Arms:** control = v1 (current prod), treatment = v2.
- **Primary:** task success (resolved without escalation).
- **Guardrails (auto-stop):** unsafe-output rate (live judge + deterministic safety checks), p95 latency, cost/call, thumbs-down rate.
- **Ramp:** 1 → 5 → 25 → 50%, guardrail check at each step.
- **Runtime:** ≥ 1 week for day-of-week mix; CUPED on pre-period task success to cut variance.
- **Rollback:** any guardrail breach, or task success not trending positive by 50%.
- **Tie to the gate:** the *same* live judges that gate offline produce the online safety/quality guardrail signals, closing the loop — and I monitor judge-human agreement *during* the experiment so a drifting judge can't silently green-light a bad arm.

### Error analysis ritual

Keep a standing bank of "gate got it wrong" cases — both false passes and false fails. After each judge/rubric change, re-run it on the bank. The judge prompt is *code*: version it, regression-test it, and never edit it in production without re-running calibration.

---

## 8. Deployment — three paths

- **Serving path (the eval execution):** offline runner (CI-triggered, parallel judge calls, golden set, Elo battles, gate decision) and the online runner (sampler → async judge queue → metric store → alerting). Both are *off* the product's request path.
- **Data path:** trace ingestion — every production LLM call logs prompt, output, context refs, tool calls, cost, latency, model version, user cohort — into the trace store that both feeds online judging and supplies fresh examples to refresh golden sets.
- **Feedback path:** the calibration loop (judge outputs → human review → agreement metric → judge update) plus harvesting production failures into the golden set. Without this path the judge silently rots.

### Rollout discipline (for the platform's own changes)

Treat the judge prompt and rubric as versioned artifacts. New judge version runs in **shadow** against the current one on the golden set, you compare their human-agreement, and only then promote it to gating duty. Yes — shadow-deploy your evaluator before trusting it.

### Monitoring and fallback

- **What pages someone:** false-pass detected post-release; judge-human agreement (kappa) drops below threshold; unsafe-output rate spikes in production; eval cost spikes (runaway judge calls); CI eval latency blows the 10-min budget (blocks all releases).
- **Fallback ladder:** if the judge is untrustworthy (agreement collapsed), **demote it from gating to advisory** and fall back to deterministic checks + mandatory human QA for releases (rung 0/1) until re-calibrated. If the trace pipeline is down, deterministic checks still run on 100% inline. Degrade to stricter human oversight, never to "no eval."
- **Incident response:** freeze the offending model version, diff against last-good (model, prompt, judge version, golden set version), pull traces for the failing cohort, and crucially **check whether the judge or the model failed** — re-score the failures with humans to localize. Roll back the model, or roll back the judge, depending on which broke.

---

## 9. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Core idea | "Use an LLM to judge outputs." | "Evaluate the evaluator" — judge is a model with measurable error/bias/drift. |
| Metric | "Score quality." | Decomposed rubric + pairwise; absolute scores treated as noisy. |
| Trust | Assumes the judge is right. | Calibrates against human labels (kappa > 0.6) before letting it gate. |
| Judge bias | Unaware. | Names position/verbosity/self-preference + concrete controls. |
| Comparison | "Score A and B, pick higher." | Pairwise Elo/Bradley-Terry, both-orders to kill position bias. |
| Agents | Grades final answer. | Grades the trajectory + environment state, not just the transcript. |
| CI | "Run evals." | Eval-as-CI gate with false-pass as the headline guardrail; judge prompt is versioned code. |
| Offline/online | "CI predicts prod." | Enumerates staleness/contamination/judge-drift; refreshes golden set from live traffic. |
| Deploy | "Run the judge." | Shadow-deploys the *judge*; demote-to-advisory fallback; never degrades to no-eval. |

---

## 10. One-page whiteboard cheat sheet

```
NUMBERS: ~40 apps | 10M calls/day | sample 2% -> 200K judge/day
         golden sets 200-2000/app | kappa>0.6 to gate
         CI < 10 min | online alert < 5 min | judging is ASYNC, off serving path

BIG IDEA: EVALUATE THE EVALUATOR. judge = model w/ error+bias+drift.

LADDER: human QA -> deterministic checks -> CALIBRATED LLM-JUDGE (gate+online)
        -> pairwise Elo + trace agent eval + continuous re-calibration
                                            ^ earn it

LABELS: deterministic (100%, free) | implicit thumbs (biased)
        | LLM-judge (amplifier, needs calibration) | human golden set (the anchor)

JUDGE BIASES -> CONTROLS:
  position   -> run both orders, average
  verbosity  -> length-balance rubric, watch score~length corr
  self-pref  -> judge from different model family
  leniency   -> calibrate threshold to human pass rate

SCORING: rubric (absolute, decomposed) | pairwise (more reliable)
         Elo/Bradley-Terry for ranking many versions

AGENTS: grade TRAJECTORY + ENV STATE: task success, tool correctness,
        policy, efficiency, recovery -- not just final answer

PLATFORM METRICS: FALSE PASS (headline) | false fail | kappa | coverage

OFFLINE-GREEN/ONLINE-BAD: stale golden set | judge drift | judge!=user
                          | contamination | proxy decoupled | sample bias

DEPLOY: shadow-deploy the JUDGE | fallback: demote judge to advisory
        + human QA | never degrade to no-eval
```

---

## 11. Full interview transcript (one hour)

**[00:00] INTERVIEWER:** Design an evaluation and monitoring platform for LLM applications in production.

**[00:30] YOU:** Clarifying questions first, because they split the design. One: are we gating releases offline, monitoring production live, or both? Two: single-turn outputs or multi-step agents? Three: how many human labels do we have? Four: are we comparing model versions, or scoring absolute quality? The answers change everything downstream.

**[01:10] INTERVIEWER:** Both offline and online. Mix of single-turn and agents. We have a small human-labeled set, maybe a few hundred per app. And yes, we constantly compare model versions.

**[01:30] YOU:** Good. Let me commit to numbers: ~40 apps on one shared platform, ~10M LLM calls a day, I'll sample about 2% — 200K — for online judging, scored asynchronously off the serving path. Golden sets of a few hundred to a couple thousand per app. And here's the load-bearing constraint I'll keep coming back to: a judge is itself a model. It has an error rate, biases, and it drifts. So before any judge is allowed to gate a release, I require it to agree with humans at a measured rate — say Cohen's kappa above 0.6. The platform is mostly the machinery that keeps the judge honest.

**[02:30] INTERVIEWER:** Most people just say "use GPT as a judge." Why all the caveats?

**[02:40] YOU:** Because an uncalibrated judge is a fast way to be confidently wrong at scale. LLM-judges have documented systematic biases. Position bias — they favor whichever answer comes first. Verbosity bias — they reward longer answers regardless of quality. Self-preference — they favor outputs from their own model family. And general leniency. If I gate releases on a biased judge, I'll ship a model that games the judge, not one that helps users. So every one of those biases gets a control, and the judge gets calibrated against humans before I trust it.

**[03:40] INTERVIEWER:** Give me the controls.

**[03:50] YOU:** Position bias: run every pairwise comparison in both orders and average; if the verdict flips, call it a tie. Verbosity: length-balance the rubric and monitor the correlation between score and answer length — if it's high, the judge is rewarding length, not quality. Self-preference: for high-stakes gates, use a judge from a different model family than the model under test. Leniency: calibrate the pass threshold against the human pass rate, not the judge's own distribution. And underneath all of it, the calibration loop: continuously sample judge outputs to human reviewers, measure agreement, and if it decays, pull the judge from gating until re-validated.

**[05:10] INTERVIEWER:** Walk me up from the simplest thing.

**[05:20] YOU:** Rung 0 is human QA on a spreadsheet before launch — trustworthy but doesn't scale past a couple releases a week. Rung 1 adds deterministic checks in CI: schema valid, required citation present, no banned content, latency and cost under threshold. Those run on 100% of traffic and catch format and policy invariants with zero ambiguity — but they can't judge nuance; a perfectly formatted answer can still be wrong. That's the trigger for rung 2: the calibrated LLM-judge with rubrics, wired into CI as a release gate and sampled online. Rung 3, when I need reliable version comparison or I'm grading agents: pairwise Elo ranking and trace-based agent eval.

**[06:40] INTERVIEWER:** Why pairwise and Elo? Why not just score each model?

**[06:50] YOU:** Because absolute scores are noisy — "is this a 4 or a 5" is a hard, unstable judgment for both humans and LLM-judges. "Is A better than B" is much more reliable. So for version comparison I run pairwise judge battles on a fixed prompt set and maintain an Elo rating per version, like a chess ladder. It gives a stable, transitive ranking that's robust to absolute-score noise — it's literally how the public LLM leaderboards rank models. A new candidate model plays the incumbent; I promote only if its Elo clears the incumbent beyond noise.

**[08:00] INTERVIEWER:** You mentioned agents. The final answer is correct — isn't that enough?

**[08:10] YOU:** No, and that's a key point. An agent can reach a correct answer through a dangerous tool call, or by luck after three wrong turns. If I only grade the final answer, I pass a trajectory I should fail. So I store the full trace — prompt, retrieved context, every tool call and its output, intermediate decisions, final response, cost, latency — and grade task success, tool correctness, policy compliance, efficiency, and recovery from failed tool calls. And wherever possible I grade against actual environment state — did the refund actually post in the sandbox — rather than judging the agent's claim that it did. State-based grading is far more trustworthy than judging the transcript.

**[09:40] INTERVIEWER:** How do you know your *platform* is any good?

**[09:50] YOU:** I measure it like a classifier of "is this release good." The headline guardrail is false pass rate — releases the gate approved that caused a production incident — because a false pass ships a bad model to everyone. I also watch false fail rate, because if the gate blocks good releases, engineers lose trust and route around it. Then judge-human agreement per app over time, and coverage — what fraction of real production query types are represented in the golden set, because gaps are where regressions hide.

**[11:00] INTERVIEWER:** Your CI eval is green, you ship, and production quality drops. What happened?

**[11:15] YOU:** Ordered list. Most likely: golden-set staleness or a coverage gap — the new failure mode isn't in my set, so CI literally couldn't see it. Second: judge-distribution shift — I calibrated the judge on old-style outputs, the new model writes more concisely, and the judge mis-scores it, sometimes punishing a real improvement via inverted verbosity bias. Third: judge-user mismatch — my rubric rewards thoroughness, users wanted brevity. Fourth: contamination — golden answers leaked into training, so CI was inflated. Fifth: I gated on judge score but the business metric is task success and they decoupled — I optimized a proxy. Sixth: my 2% online sample is biased toward one cohort and masks a regression elsewhere.

**[12:50] INTERVIEWER:** How do you fix the staleness one specifically?

**[13:00] YOU:** Refresh the golden set from real recent production traffic on a schedule — harvest failures and new query types out of the trace store and route them to human labeling. The golden set is a living artifact, not a one-time snapshot. And I keep one held-out, never-published set for the highest-stakes gate so I have a contamination-resistant anchor.

**[13:50] INTERVIEWER:** Design the A/B test that the gate authorizes.

**[14:00] YOU:** Model v2 passed the gate; now prove it online. Randomize by user, sticky. Control is v1 in prod, treatment is v2. Primary metric is task success — resolved without escalation. Guardrails that auto-stop the ramp: unsafe-output rate from the live judge plus deterministic safety checks, p95 latency, cost per call, and thumbs-down rate. Ramp 1, 5, 25, 50 with a guardrail check each step, at least a week for day-of-week mix, CUPED on pre-period task success to cut variance. Rollback on any guardrail breach or if task success isn't trending positive by the halfway point. And critically, I monitor judge-human agreement *during* the experiment, so a drifting judge can't silently green-light the worse arm.

**[15:40] INTERVIEWER:** Your judge starts disagreeing with humans mid-week. What do you do?

**[15:50] YOU:** That's judge drift, and I treat it like data drift — it pages someone. Immediate action: demote the judge from gating to advisory and fall back to deterministic checks plus mandatory human QA for any release in flight. Then localize: did the judge model get silently upgraded, did someone edit the rubric, or did the input distribution shift? The judge prompt is versioned code, so I diff it. I re-validate on the human set, and only re-promote it to gating once agreement is back above threshold. The principle is: degrade to stricter human oversight, never to no eval.

**[17:10] INTERVIEWER:** You said the judge prompt is versioned. Do you deploy it like code too?

**[17:20] YOU:** Yes — I shadow-deploy the evaluator. A new judge version runs in shadow against the current one on the golden set; I compare their human-agreement and their verdicts on my bank of known-hard cases. Only if the new judge agrees with humans at least as well do I promote it to gating duty. It sounds funny to shadow-deploy your own QA tool, but the judge is the most load-bearing model in the platform — if it's wrong, every downstream decision is wrong.

**[18:30] INTERVIEWER:** Last one — what dominates cost, and how do you control it?

**[18:40] YOU:** Judge calls. I can't afford to judge 10M traces a day with a frontier model. So I tier: deterministic checks on 100% for free, a cheap judge model on the 2% online sample, and the strong, expensive judge reserved for the offline CI gate and Elo battles where stakes and accuracy matter most. I cache judgments for unchanged prompt-output pairs so re-runs are cheap, and I monitor eval cost as a guardrail because a runaway judge loop could quietly burn the budget.

**[19:40] YOU:** This connects directly to what I built before — I set up Elo-based scoring to rank model versions, regression detection in CI, and functional-correctness metrics for our agentic flows. The lesson that stuck was exactly the one I led with: we once trusted a judge that had drifted, shipped on a green eval, and caught a regression only from user complaints. After that we instrumented judge-human agreement as a first-class, paged metric. That scar is why I treat the evaluator as the thing most in need of evaluation.

**[20:30] INTERVIEWER:** That's the answer I was looking for.

### Why this transcript works

- **Leads with the meta-insight** — the judge is a fallible model — and never lets go of it.
- **Names judge biases with concrete controls**, not just a list.
- **Justifies pairwise/Elo** over absolute scoring with a real reliability argument.
- **Distinguishes final-answer grading from trajectory + environment-state grading** for agents.
- **Treats the platform itself as measurable** (false pass as headline) and handles the green-CI/bad-prod trap with an ordered list.
- **Shadow-deploys the evaluator** and degrades to stricter human oversight, never to no-eval.
- **Closes on real Elo/regression/agent-correctness experience**, anchored by a specific scar.

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x volume (100M calls/day)?** Lower the online sample rate but stratify it (sample more from high-risk cohorts and new model versions), shard the trace store, and push more checks to the cheap/deterministic tier. The strong judge stays reserved for CI.
- **How do you handle a brand-new app with no golden set (cold start)?** Bootstrap with synthetic eval cases (LLM-generated from the spec), borrow rubrics from similar apps, run heavy human QA early, and harvest the first weeks of production traffic into the first real golden set.
- **How do you pick the pass threshold?** Calibrate against the human pass rate on the golden set, then choose the operating point by the cost asymmetry — for high-stakes apps, bias toward false fails (block more) over false passes (ship bad).
- **Offline green, online bad — what do you check?** Golden-set staleness/coverage, judge drift, judge-vs-user mismatch, contamination, proxy-target decoupling, online sample bias. (Section 7.)
- **How do you debug a bad launch?** Diff model/prompt/judge/golden-set versions; pull failing-cohort traces; re-score failures with humans to decide whether the *judge* or the *model* broke; roll back the right one.
- **How do you prevent gaming / feedback loops?** Don't optimize models directly against the judge that gates them without periodic human re-anchoring; rotate/refresh eval sets; keep a held-out set; watch for score-vs-length and self-preference creep.
- **When is LLM-judge the wrong tool?** When a cheap deterministic check suffices (format, exact match, policy keyword) — don't pay a judge to check JSON validity. Reserve the judge for genuinely subjective quality.

---

## 13. Common mistakes

- Treating the **LLM-judge as ground truth** instead of a fallible model to be calibrated against humans.
- Ignoring **judge biases** (position, verbosity, self-preference, leniency) and their controls.
- Using **absolute scores for version comparison** when pairwise/Elo is far more reliable.
- Grading only the **agent's final answer**, not the trajectory and environment state.
- Building the gate but never measuring its **false pass / false fail** rates.
- A **static golden set** that goes stale and stops predicting production.
- Putting the judge **inline on the serving path**, coupling product latency to eval infra.
- No **fallback** when the judge becomes untrustworthy — degrading to no-eval instead of to human oversight.
- Judging everything with the **most expensive model**, blowing the eval budget.

---

## 14. Transfer — what this case unlocks

- **05 Production RAG:** the RAGAS faithfulness/relevance decomposition is the reference-free judging toolkit this platform operationalizes at scale.
- **04 Enterprise AI Copilot:** eval-without-ground-truth and agent trace grading are the consumer of this platform; that case applies, this case builds the measurement system.
- **07 AI Agent Ticket Resolution:** trace-based grading + environment-state success is exactly how you'd evaluate that agent.
- **13 LLM Safety Gateway:** the safety judge and its calibration loop are the same machinery pointed at policy compliance.
- **20 ML Monitoring & Drift:** judge drift is a special case of model drift; the alerting/fallback discipline is shared.
- **11 Ads CTR / Experimentation:** the A/B rigor, guardrails, CUPED, and ramp discipline are the same experimentation backbone.

---

## 15. Sources

Originals (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Anthropic, Demystifying Evals for AI Agents: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- OpenAI Evals repository: https://github.com/openai/evals

Added (verified canonical):
- LLM-as-judge, MT-Bench/Chatbot Arena (Zheng et al., 2023): https://arxiv.org/abs/2306.05685
- G-Eval (Liu et al., 2023): https://arxiv.org/abs/2303.16634
- HELM (Liang et al., 2022): https://arxiv.org/abs/2211.09110
- RAGAS (Es et al., 2023): https://arxiv.org/abs/2309.15217
- Cohen's kappa (inter-rater agreement): https://en.wikipedia.org/wiki/Cohen%27s_kappa
