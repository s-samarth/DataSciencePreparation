# 15. Conversational Recommender

**Company tags:** Amazon, Netflix, Spotify, Booking/Expedia, Instacart, commerce/travel AI startups
**Interview frequency:** Medium-high for LLM/applied-AI roles
**Why it matters:** A normal recommender (files 02, 03) ranks items against a *fixed* query or context — the user's intent is a given. A *conversational* recommender has to **construct** the intent over several turns, and that flips the central problem from "rank well" to "**when do I ask a clarifying question versus when do I just recommend?**" That ask-vs-recommend decision, made under a friction budget, is the case. If you treat it as "LLM parses the query, then run a recommender," you have skipped the only hard part.

---

## 0. How to use this doc

Two passes.

**Pass 1 (intuition).** Read the prose and transcript. Lock the spine: **the conversation is a preference-elicitation process, and every turn is a decision under uncertainty** — the system holds a *belief* about what the user wants and must choose, each turn, to either *exploit* that belief (recommend) or *explore* it (ask a question that reduces uncertainty). Asking costs a turn of friction; recommending too early risks a bad rec. That is a value-of-information / explore-exploit problem, and it is what makes this case different from every other recommender in the set.

**Pass 2 (active recall).** Cover the page. Can you (a) state the ask-vs-recommend decision and what governs it, (b) explain why you must separate *hard constraints* from *soft preferences* and the silent failure that follows from confusing them, (c) explain why you cannot offline-replay a logged conversation against a new dialogue policy (the counterfactual/off-policy trap) and what you do instead, and (d) say how recommendations stay grounded in the real catalog with no hallucinated attributes? Those four are the case.

**The scaffold (shared across this set):**

> **Clarify → Frame → Data/Labels → Baseline → Model → Eval → Deploy → Monitor**

Bends here: "Frame" must name the explore/exploit dialogue decision; "Model" is belief-tracker + clarification-policy + constrained retrieval + ranker + grounded generation; "Eval" confronts the fact that dialogue is *interactive*, so logged data is off-policy and you need user simulators + online A/B; "Data/Labels" deals with in-session vs cross-session feedback.

**The senior tell, stated once:** say early, "the model isn't just a ranker — it's a policy that decides each turn whether to *ask* or *recommend*, trading clarification friction against recommendation relevance." That reframes the whole problem and is what separates someone who has built a dialogue product from someone who will bolt an LLM onto a recommender.

---

## 1. Clarify (scripted, with *why each answer changes the design*)

| Question | Why it changes the design |
|---|---|
| **What domain — products, movies, jobs, trips?** | High-consideration, low-frequency purchases (a trip, a job, a laptop) reward multi-turn elicitation; users tolerate questions because the stakes are high. Low-stakes, high-frequency (a song, a snack) punish friction — users want one good answer fast. This sets the *friction budget* and thus how often I'm allowed to ask. |
| **How big and structured is the catalog?** | A 10M-item commerce catalog with rich structured attributes (price, brand, specs) makes hard-constraint filtering reliable. A sparsely-attributed catalog forces more on embeddings/soft matching. |
| **Is there a logged-in user with history, or anonymous?** | History gives a personalization prior (cold-start mitigation); anonymous means the conversation *is* the only signal, raising the value of good elicitation. |
| **How many turns is acceptable?** | This is the friction budget, the core constraint. If the product wants resolution in ~3 turns, I can ask at most one or two clarifying questions, so each must maximize information gain. |
| **Latency and modality (chat, voice)?** | Conversational means each turn must feel responsive — TTFT under ~1s. Voice tightens it further and changes how many options I can present. |
| **What is a "good outcome" — click, purchase, satisfaction?** | The label and the objective. Purchase/booking is a strong delayed signal; in-session acceptance ("yes, that one") is a faster proxy. I need both. |

State assumptions and move: high-consideration commerce/travel, 10M structured catalog, logged-in with history, ~3–5 turn budget, chat, success = booking/purchase with in-session acceptance as proxy.

---

## 2. Numbers up front (carry them through)

- **Scale:** ~5M conversations/day, ~5 turns average = ~25M turns/day ≈ ~300 turn-requests/sec average, peak ~1.5K/sec. Catalog ~10M items with structured attributes.
- **Latency budget (derived out loud):** conversational, so per-turn end-to-end target ~2s, **TTFT < 1s**. A turn does: belief update + ask-vs-recommend decision + (if recommend) constrained retrieval + rank + grounded generation. Retrieval over 10M items must be a fast ANN + filter (tens of ms); the LLM generation dominates TTFT, so the recommend path streams. The *decision* (ask vs recommend) must be cheap (a small model / heuristic), not a second large-LLM call in series.
- **The friction budget (the headline constraint):** target **resolution in ≤ 3–4 turns**. This caps clarifying questions at ~1–2, so each question must be chosen to maximize information gain. Over-asking and under-asking are *both* failure modes — this is an operating curve, not "more questions = better."
- **Cost back-of-envelope:** if each turn is one LLM call at ~$0.005–0.02 and conversations average 5 turns, that is ~$0.025–0.10/conversation × 5M/day = $125K–500K/day. So turn count is a *cost* lever too, reinforcing the friction budget: fewer, better turns is both better UX and cheaper.
- **Quality targets:** in-session **acceptance rate** (user picks/saves a recommended item) and downstream **conversion**; guardrails = **zero hallucinated item attributes** (hard — a recommended item must exist with the stated specs), latency, and **clarification-turns-to-resolution** (minimize without hurting acceptance).

---

## 3. The conceptual spine: ask vs recommend, and constraints vs preferences

Two ideas carry the whole case. State both explicitly.

### 3.1 The dialogue is an explore/exploit decision under a friction budget
At each turn the system holds a **belief** over what the user wants — a set of constraints and a soft preference profile, both with uncertainty. The decision:
- **Recommend (exploit):** present items if the belief is confident enough that a top recommendation will likely be accepted, OR the candidate set has narrowed enough, OR the user signals impatience.
- **Ask (explore):** pose a clarifying question if the candidate set is still large and *ambiguous on a high-value attribute* — i.e., a question whose answer would split the candidate set most usefully (highest expected information gain / value of information).

The cost of asking is a turn of friction (and money); the cost of recommending too early is a wasted rec that erodes trust. Choosing well is the product. A senior frames it as **value of information**: ask only when the expected improvement in recommendation quality from the answer exceeds the friction cost of the question. This is the same explore/exploit logic as a bandit (file 19), applied to *questions* instead of *arms*.

### 3.2 Hard constraints vs soft preferences (and the silent failure of confusing them)
User utterances carry two fundamentally different kinds of signal:
- **Hard constraints** — "under $500," "available next week," "size 10," "remote only." These are **deterministic filters**: an item either satisfies them or it is disqualified. They must be parsed into structured slots and applied as a filter, never as a ranking nudge.
- **Soft preferences** — "kind of sporty," "something cozy," "I usually like indie films." These are **ranking signals**: they shift scores, they don't disqualify.

The classic, *silent* failure (the original file even hints at it): the LLM parser mis-classifies one as the other.
- A soft preference treated as a hard filter → the candidate set is silently over-constrained, sometimes to **zero**, and the system either returns nothing or, worse, drops the filter and recommends something off-base without telling the user.
- A hard constraint treated as a soft preference → the system recommends a $2,000 item to someone who said "under $500." Trust destroyed.

The senior design: **separate slot-filling (structured, hard, validated) from preference modeling (soft, learned)**, and **validate parsed hard constraints against catalog availability before committing** — if "under $500 + brand X + in stock" yields zero items, surface that to the user ("nothing from X under $500 — want to raise the budget or change brand?") rather than silently failing. Echo the file 13/05 stance: show the interpreted constraints, let the user correct them.

---

## 4. The data/label problem for *this* domain: in-session vs cross-session, and off-policy logs

Every case has a signature data problem. The conversational recommender's is twofold:

1. **Two feedback timescales.** *In-session* signals (user says "cheaper," "I like that one," clicks one of three options) update the belief *within* the conversation — this is online, per-turn learning, bandit-like. *Cross-session* signals (purchases, return visits, long-run satisfaction) train the underlying preference/recsys models. You need both, and you must not confuse the fast, noisy in-session signal with the slow, reliable outcome signal.

2. **Logged conversations are off-policy — the deep label problem.** This is the one to emphasize. A conversation is *interactive*: the questions the system asked shaped the answers the user gave, which shaped the recommendations. If you change the dialogue policy, it would have asked *different* questions and received *different* answers — so you **cannot replay a logged conversation against a new policy** the way you can replay a ranking model against logged impressions. The data is fundamentally counterfactual/off-policy. Consequences (§7): offline eval needs **user simulators** (today, LLM-based simulated users) plus **off-policy correction** where applicable, and the real verdict comes from **online A/B**. Saying this out loud is a strong senior signal — most candidates assume they can offline-replay dialogue and are wrong.

**Cold start** (the original file's good bone, kept and sharpened): anonymous or new users have no history, so the conversation carries all the signal — which *raises* the value of good elicitation. Use catalog-content embeddings, popularity priors within the elicited constraints, and onboarding questions as the cold-start bridge.

---

## 5. The baseline → why-it-breaks → next-rung ladder

**Rung 0 — Faceted search wizard.** A fixed decision tree of dropdowns/filters ("Budget? Brand? Size?").
- *Works:* deterministic, transparent, controllable; no hallucination; fine for simple catalogs.
- *Breaks:* rigid and unnatural; can't handle "something cozy for a rainy weekend"; the question order is fixed regardless of what the user already said. **Trigger:** users abandon because the wizard asks irrelevant questions or can't express nuance.

**Rung 1 — LLM intent parser + classic recommender.** LLM extracts slots/constraints from free text, then a deterministic filter + ranker runs.
- *Adds:* natural language in, structured query out; reuses a proven recommender.
- *Breaks:* parser errors *silently constrain the wrong set* (§3.2); it's single-shot — no real multi-turn elicitation, no decision about *whether to ask*; soft preferences get flattened into filters. **Trigger:** the first "zero results" or wildly off-base rec from a misparse, or users wanting back-and-forth refinement.

**Rung 2 — Stateful dialogue manager: belief tracker + clarification policy + constrained retrieval + ranker + grounded generation (recommended production design).** Maintains a belief over constraints+preferences across turns, decides ask-vs-recommend by information gain, retrieves under hard constraints, ranks by soft preference + personalization, and generates grounded explanations. Detailed in §6.
- *Adds:* real elicitation, controlled friction, constraint/preference separation, grounding.
- *Breaks:* mostly fine; for very open-ended exploratory tasks ("plan me a 2-week trip") you may want tools (search flights, check availability, compare). **Trigger:** tasks needing multi-step actions / external tools.

**Rung 3 — Agentic recommender with tools (extension).** Planner + tools (search, availability, compare, book) with the file-04/07 control-plane discipline: tool schemas, permission checks, recursion/cost budgets, validators, trace eval. Mention as the extension for high-complexity domains (travel planning), *not* the default — agentic loops add reliability risk and latency you don't need for "recommend me a jacket."

Meta-rule out loud: "I default to the stateful dialogue manager because the hard problem is *when to ask vs recommend* and *constraint vs preference separation*, neither of which a single LLM parse solves. I only add tools when the task genuinely requires multi-step external actions."

---

## 6. The architecture explained to the floor

```text
user turn -> [1 NLU / belief update] -> [2 ASK-vs-RECOMMEND policy]
                                              |-- ask --> generate clarifying question (max info gain)
                                              |-- recommend --> [3 hard-constraint FILTER] -> [4 soft-pref RANK]
                                                                    -> [5 GROUNDED generation: 2-3 items + reasons]
   in-session feedback ----------------------------------------------------------^ (updates belief)
```

### 6.1 Belief / state tracker
Maintains, across turns: **structured slots** (hard constraints, with confidence — "budget ≤ $500 (high conf)"), a **soft preference profile** (an embedding or attribute weighting blended with the user's history prior), and **dialogue state** (what's been asked, what the user reacted to). Updated every turn from the new utterance + in-session feedback. This is the memory that makes it a *conversation*, not a sequence of independent queries.

### 6.2 The ask-vs-recommend policy (the heart)
Given the current belief, decide. A practical formulation:
- Run the hard-constraint filter to get the candidate set size and its diversity on key attributes.
- **Recommend if:** candidate set is small/coherent enough that the top item's predicted acceptance is high, OR turn budget is nearly spent, OR the user signaled "just show me something."
- **Ask if:** the set is large *and* ambiguous on a high-value attribute — choose the question with the highest **expected information gain** (the attribute that most splits/narrows the viable set, weighted by how much it affects ranking). Value-of-information: ask only if expected quality gain > friction cost.
- Can start as a heuristic (set-size + entropy thresholds) and graduate to a learned policy trained on which choice led to acceptance with fewer turns. Keep it *cheap* — it's in the latency path.

This is the component that does not exist in a normal recommender and is what the interviewer most wants to hear.

### 6.3 Constrained retrieval + ranking
- **Hard filter** on structured attributes over the catalog (deterministic; this is where "under $500, in stock" lives). Validate non-empty *before* committing (§3.2).
- **Soft retrieval + rank** within the filtered set: ANN over item embeddings against the soft-preference profile, blended with collaborative/personalization signals and business objective. Same two-stage retrieve→rank funnel as files 02/03, but operating on the conversationally-elicited belief instead of a static query.

### 6.4 Grounded generation (no hallucinated attributes)
The LLM presents 2–3 items with reasons, but **only from the retrieved real items, using their real attributes** — never inventing a product, price, or spec. Implementation: pass the structured item records to the generator and constrain it to reference only those fields; post-validate that every attribute mentioned matches the catalog record (reject/regenerate if not). This is cite-or-abstain (file 05) for recommendations and the relevant guardrail ("hallucinated item attributes") from the original file. Explanations also build trust and double as *implicit clarification* ("I picked these because they're under $500 and sporty — want warmer instead?").

### 6.5 The three paths, named
- **Serving path:** the per-turn loop above, latency-budgeted, streaming the recommend path.
- **Data path:** catalog + attributes + item embeddings, user history priors, the preference/ranking models.
- **Feedback path:** in-session reactions (fast, belief update) + cross-session outcomes (slow, model training) + conversation logs for offline simulation and policy learning.

### 6.6 Costs
Turn count drives both UX and cost (§2), so the friction budget is also a cost control. Keep the ask-vs-recommend decision and belief update on cheap models; reserve the big LLM for generation. Cache item embeddings and the user history prior.

---

## 7. Evaluation

### 7.1 Metrics
- **Offline (component-level):** intent-slot accuracy (and crucially constraint-vs-preference *classification* accuracy — the silent-failure source), Recall@K / NDCG@K of the ranker on logged accepted items, grounding/faithfulness (does generation stick to real attributes), constraint-satisfaction rate (recommended items actually meet stated hard constraints).
- **Online (the real verdict):** in-session **acceptance rate**, downstream **conversion**, **clarification-turns-to-resolution** (the friction metric — minimize subject to acceptance), repeat usage/satisfaction.
- **Guardrails:** hallucinated-attribute rate (target zero), latency, over-personalization/filter-bubble (diversity), unsafe recommendations.

### 7.2 Why offline eval is genuinely hard here (the trap, in conversational form)
*"Your offline acceptance looked great — why did the online A/B drop?"* The dialogue-specific causes, in order:
1. **Off-policy / counterfactual logs (§4.2):** you scored the new policy by replaying old conversations, but the new policy would have asked different questions and gotten different answers. Offline replay of dialogue is fundamentally invalid for the *policy* (it can still validate the ranker on a fixed turn). **The #1 trap here.**
2. **User-simulator gap:** you evaluated the policy against an LLM-simulated user, and the simulator is more cooperative/predictable than real users (answers cleanly, never gets annoyed, never goes off-script).
3. **Friction not captured offline:** offline you measured relevance, not annoyance; the new policy asks one more question, relevance ticks up, but real users abandon. Acceptance-per-turn matters, not acceptance alone.
4. **Soft/hard misparse rate higher on real distribution** than on your curated offline set.
5. **Standard recsys feedback loops** — the system shows what it predicts you'll like, narrowing future data.

Cure: use simulators only for *directional* offline screening and component metrics; treat **online A/B as the source of truth** for the dialogue policy; measure friction (turns) as a first-class guardrail, not just relevance.

### 7.3 A fully-specified A/B test
- **Hypothesis:** the new ask-vs-recommend policy v3 raises conversion *without* increasing clarification turns vs v2.
- **Unit:** user (sticky, so a user gets a consistent experience), randomized; stratify by logged-in vs anonymous.
- **Primary metric:** conversion (or in-session acceptance as faster proxy). **Guardrails:** clarification-turns-to-resolution (must not rise materially), latency, hallucinated-attribute rate (zero tolerance), diversity, abandonment rate.
- **The subtlety:** acceptance and friction *trade off*, so the win condition is a *frontier* improvement (more acceptance at equal-or-fewer turns), not just acceptance. State this explicitly.
- **Runtime/ramp:** shadow → canary → ramp; ≥ a full business cycle. CUPED on pre-period user activity to cut variance.
- **Rollback trigger:** conversion drop, turn-count spike, any hallucinated-attribute incident, latency breach.

---

## 8. Deployment, monitoring, fallback, incident response

- **Rollout:** shadow the new dialogue policy (log its ask/recommend decisions on live turns without showing them) to estimate turn-count and decision distribution before enforcing; then canary/ramp. Catalog/attribute changes (new items, price updates) flow on the freshness path so the filter and grounding stay accurate.
- **Monitoring:** acceptance rate, conversion, clarification-turns distribution (a spike = the policy got ask-happy; a drop with low acceptance = it's recommending too early), hallucinated-attribute rate (alarm at any nonzero), zero-result-after-filter rate (the silent-failure canary from §3.2), latency, cost/conversation, diversity. Slice by domain/category and logged-in vs anonymous.
- **Fallback:** if the dialogue policy or LLM is degraded, fall back to rung 0/1 — faceted filters or single-shot parse + ranker — never to ungrounded free-form generation. Graceful degradation = less natural, never hallucinated.
- **Incident response:** freeze policy/model version, inspect conversation traces (every turn's belief state, decision, retrieved set, and generated text logged), identify whether the failure is parse (misclassified constraint), policy (asked wrong/too much), retrieval (empty set), or generation (hallucinated attribute), roll back, add the failing conversation to the simulator/eval set.
- **Safety/over-personalization:** monitor for filter bubbles and ensure exploration so the belief can be corrected; never let a wrong inferred preference lock a user out of the catalog.

---

## 9. Full one-hour interview transcript

**[0:00] INTERVIEWER:** Design a conversational assistant that recommends products, movies, or trips through dialogue.

**[0:30] YOU:** Let me scope it, because the friction budget swings the whole design. What domain — high-consideration like trips and laptops, or low-stakes like songs? Because for a trip users tolerate questions; for a song any question is annoying. How big and structured is the catalog? Is the user logged in with history or anonymous? And roughly how many turns is acceptable before they expect an answer?

**[1:15] INTERVIEWER:** Commerce, high-consideration, 10M structured catalog, logged-in users, aim to resolve in three or four turns.

**[1:30] YOU:** Good. Then let me name the framing, because I think it's the crux and it's different from a normal recommender: the conversation is a *preference-elicitation process*, and every turn is a decision under uncertainty. The system holds a belief about what the user wants, and each turn it has to choose — **recommend** if the belief is confident enough, or **ask a clarifying question** if it isn't. Asking costs a turn of friction; recommending too early risks a bad rec that burns trust. So the model isn't just a ranker, it's a *policy* that trades clarification friction against relevance. That ask-versus-recommend decision is the case. A normal feed or search recommender never faces it because the query is given.

**[3:00] INTERVIEWER:** How do you decide whether to ask?

**[3:10] YOU:** Value of information. I run the hard-constraint filter to see how big and how ambiguous the candidate set is. If it's narrowed to a coherent handful, or the user signaled "just show me something," or my turn budget is nearly spent, I recommend. If it's still large and ambiguous on a *high-value* attribute, I ask — and I pick the question with the highest expected information gain, the attribute that most splits the viable set weighted by how much it affects ranking. The rule is: ask only if the expected quality gain beats the friction cost. It's the same explore-exploit logic as a bandit, but over *questions* instead of arms. I'd start with a set-size-plus-entropy heuristic and graduate to a learned policy trained on which choice led to acceptance in fewer turns.

**[4:45] INTERVIEWER:** What's the hardest failure mode you're worried about?

**[4:55] YOU:** Confusing hard constraints with soft preferences, because it fails *silently*. "Under $500" is a hard filter — an item either satisfies it or it's out. "Kind of sporty" is a soft preference — it shifts ranking, it doesn't disqualify. If my LLM parser treats a soft preference as a hard filter, I silently over-constrain the candidate set, sometimes to zero, and then I either return nothing or quietly drop the filter and recommend something off-base. The reverse — treating "under $500" as a soft nudge — recommends a $2,000 item to a budget shopper and destroys trust. So I separate slot-filling, which is structured and hard and validated, from preference modeling, which is soft and learned. And critically, I validate the parsed hard constraints against catalog availability *before* committing — if "brand X under $500 in stock" is empty, I surface that: "nothing from X under $500, want to raise budget or change brand?" rather than failing silently.

**[7:00] INTERVIEWER:** Walk me through the architecture of one turn.

**[7:10] YOU:** Five steps. One, NLU and belief update — parse the utterance, update the structured slots and the soft-preference profile, fold in the user's history prior and any in-session feedback. Two, the ask-versus-recommend policy we discussed, kept cheap because it's in the latency path. If it's "ask," I generate the max-info-gain question and stop. If it's "recommend": three, the hard-constraint filter over the 10M catalog, deterministic, fast ANN plus structured filter, validated non-empty. Four, rank within that set by the soft preference profile blended with personalization and business objective — the same retrieve-then-rank funnel as a normal recommender, just operating on a conversationally-elicited belief. Five, grounded generation: present two or three real items with reasons.

**[9:30] INTERVIEWER:** How do you stop it from making up a product or a spec?

**[9:40] YOU:** Grounding, same principle as cite-or-abstain in RAG. The generator only sees the retrieved real item records and is constrained to reference only their actual fields — price, specs, availability. Then I post-validate: every attribute it mentions must match the catalog record, and if it doesn't, I reject and regenerate. Zero hallucinated attributes is a hard guardrail, not a nice-to-have, because a confidently wrong "yes this laptop has 32GB RAM" is a returned product and a lost customer. The explanations also do double duty as soft clarification — "I picked these because they're under $500 and sporty, want warmer instead?" lets the user correct my belief cheaply.

**[11:30] INTERVIEWER:** How do you evaluate this offline before you ship?

**[11:40] YOU:** This is where I have to be careful, because dialogue breaks normal offline eval. A conversation is interactive — the questions I asked shaped the answers I got. So if I change the policy, it would have asked different questions and gotten different answers, which means **I cannot replay a logged conversation against a new policy.** The logs are off-policy, fundamentally counterfactual. I can still validate the *ranker* on a fixed turn with Recall@K and NDCG, and I can validate component metrics like constraint-vs-preference classification accuracy. But for the *dialogue policy*, offline replay is invalid. So I use LLM-based **user simulators** for directional screening, knowing they're more cooperative than real users, and I treat **online A/B as the source of truth.**

**[13:30] INTERVIEWER:** Suppose offline acceptance looked great and the A/B dropped. Why?**

**[13:40] YOU:** Top cause: the off-policy replay problem I just described — I scored the new policy on old conversations it would never actually have produced. Second: user-simulator gap — the simulator answered cleanly and never got annoyed, real users do. Third, and this is subtle: offline I measured *relevance*, not *friction*. The new policy asks one more question, relevance ticks up, but real users abandon — so the right metric is acceptance *per turn*, not acceptance alone. Fourth, the soft/hard misparse rate is higher on the messy real distribution than on my curated offline set. The fix is to measure clarification-turns-to-resolution as a first-class guardrail and let the A/B decide.

**[15:30] INTERVIEWER:** So specify that A/B.**

**[15:40] YOU:** Hypothesis: the new ask-versus-recommend policy raises conversion without increasing clarification turns. Unit is the user, sticky so they get a consistent experience, stratified by logged-in versus anonymous. Primary metric conversion, with in-session acceptance as a faster proxy. Guardrails: clarification-turns-to-resolution must not rise, latency, hallucinated-attribute rate at zero tolerance, diversity, and abandonment. The key subtlety is that acceptance and friction trade off, so my win condition is a *frontier* improvement — more acceptance at equal or fewer turns — not just higher acceptance. Shadow, canary, ramp over a full business cycle, CUPED on prior activity to cut variance, roll back on a conversion drop, a turn-count spike, or any hallucination incident.

**[18:00] INTERVIEWER:** When would you make this agentic with tools?**

**[18:10] YOU:** Only when the task genuinely needs multi-step external actions — "plan me a two-week trip" needs to search flights, check availability, compare, maybe book. Then I'd add a planner with tools behind a real control plane: tool schemas, permission checks, recursion and cost budgets, deterministic validators, trace-based eval, human handoff for booking. But I would *not* make "recommend me a jacket" agentic — the loop adds reliability risk and latency for no benefit. The hard problem there is still ask-versus-recommend and constraint separation, which the stateful dialogue manager already solves. Default to the simpler design; earn the agent.

**[20:00] INTERVIEWER:** How do you handle a brand-new anonymous user?**

**[20:10] YOU:** No history means the conversation carries *all* the signal, which actually raises the value of good elicitation — so I lean more on asking, within the friction budget. I bootstrap with catalog-content embeddings and popularity priors *within* the elicited constraints, and a light onboarding question. As soon as in-session reactions arrive, the belief tracker updates and I'm personalizing within the session even with zero history.

**[22:00] INTERVIEWER:** What breaks in production and how do you catch it?**

**[22:10] YOU:** My canary metrics: a clarification-turns spike means the policy got ask-happy; a drop with low acceptance means it's recommending too early; any nonzero hallucinated-attribute rate pages immediately; and a rising zero-result-after-filter rate is my silent-failure alarm for constraint misparsing. Traces log every turn's belief, decision, retrieved set, and generated text, so I can localize a failure to parse, policy, retrieval, or generation, roll back, and add the bad conversation to my eval and simulator set. Fallback while degraded is faceted filters or single-shot parse plus ranker — never ungrounded free-form generation.

**[24:00] INTERVIEWER:** Wrap up.

**[24:10] YOU:** To close: this is preference elicitation, not just ranking, so the core is a policy that decides each turn whether to ask or recommend, trading friction against relevance under a turn budget. I separate hard constraints from soft preferences to avoid silent over-constraining, ground every recommendation in the real catalog with zero invented attributes, and — because dialogue logs are off-policy — I lean on user simulators for screening and online A/B for truth, with clarification-turns as a first-class guardrail. This bridges directly from the product-search and copilot-orchestration work I've done: turning a vague natural-language goal into a structured, grounded, ranked result with controlled back-and-forth.

### Why this transcript works
- **Reframes to preference elicitation / ask-vs-recommend** in the first 90 seconds — the senior insight.
- **Uses value-of-information / explore-exploit** to make "when to ask" concrete, not hand-wavy.
- **Nails the hard-vs-soft silent-failure** and the pre-commit availability validation that prevents it.
- **Owns the off-policy dialogue-eval trap** — the thing most candidates get wrong — and the simulator + online-A/B response.
- **Treats friction (turns) as a first-class metric** and defines the A/B win as a frontier, not a point.
- **Knows when *not* to go agentic** — disciplined scoping.
- **Closes by connecting to real product-search/copilot experience** without overclaiming.

---

## 10. Junior vs senior contrast

| Dimension | Junior | Senior |
|---|---|---|
| Framing | "LLM parses the query, then recommend." | "Preference elicitation: a policy that decides ask vs recommend under a friction budget." |
| When to ask | "Ask a few questions." | Value-of-information / max info gain; ask only if expected gain > friction cost. |
| Constraints | "Extract preferences." | **Hard constraints (filter) vs soft preferences (rank)** separated; validate non-empty before committing. |
| Silent failure | unaware | Misparsing a soft pref as a hard filter zeroes the set silently; surfaces and self-corrects. |
| Grounding | "LLM describes items." | Generate only from retrieved real items; post-validate every attribute; zero-hallucination guardrail. |
| Offline eval | "Replay logged conversations." | Dialogue logs are **off-policy**; simulators for screening, **online A/B is truth**. |
| Metrics | "Acceptance/conversion." | Acceptance **per turn**; clarification-turns as first-class guardrail; frontier win condition. |
| Agentic | "Make it an agent." | Earn the agent — only for multi-step external actions; default to the dialogue manager. |

---

## 11. One-page whiteboard cheat sheet

```text
SPINE: conversation = PREFERENCE ELICITATION. Each turn: ASK (explore) vs RECOMMEND (exploit) under friction budget.

NUMBERS: 5M convos/day, ~5 turns, 10M-item catalog; TTFT<1s, turn<2s; resolve in <=3-4 turns
         turn count = UX AND cost lever ($0.025-0.10/convo)

ASK vs RECOMMEND: recommend if set small/coherent OR budget spent OR user impatient
                  ask if set large & ambiguous on high-value attr -> max INFO GAIN; ask iff gain > friction cost

HARD vs SOFT (the silent failure):
  hard constraint ("<$500","in stock") = deterministic FILTER (disqualifies)
  soft preference ("sporty","cozy")    = ranking SIGNAL (shifts score)
  misparse -> silent over-constrain to ZERO, or $2000 rec to budget user
  FIX: separate slot-fill (hard, validated) from pref model (soft); validate non-empty BEFORE committing

TURN LOOP: NLU/belief-update -> ASK-vs-REC policy -> [filter -> rank -> GROUNDED gen (2-3 real items+reasons)]
           in-session feedback updates belief

GROUNDING: generate only from retrieved real items; post-validate attributes; ZERO hallucinated attrs (hard guardrail)

LABELS: in-session (fast, belief) vs cross-session (slow, train).  LOGS ARE OFF-POLICY (interactive).
EVAL: offline = component metrics + USER SIMULATORS (screening only).  ONLINE A/B = truth.
  win = FRONTIER (more acceptance at <= turns). guardrail = clarification-turns, hallucination=0, diversity.

OFFLINE!=ONLINE: off-policy replay -> simulator too cooperative -> friction not measured -> misparse rate -> feedback loop

LADDER: 0 faceted wizard -> 1 LLM parse+recsys (silent misparse, single-shot)
        -> 2 STATEFUL dialogue mgr (belief+policy+filter+rank+grounded) [default]
        -> 3 agentic+tools (only for multi-step actions e.g. trip booking)
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 100M users?** Separate retrieval/ranking services, cache item embeddings + user priors, keep the ask-vs-recommend decision and belief update on small fast models, shard the catalog index, define per-turn p95/p99 SLOs; turn-count discipline becomes a major cost lever.
- **How would you handle cold start?** Anonymous → conversation carries all signal → lean on elicitation within the friction budget, content embeddings + constrained popularity priors + a light onboarding question; personalize from in-session feedback immediately.
- **How do you decide how many items to show / when to stop asking?** On the acceptance-vs-friction frontier: stop asking when expected info gain < friction cost; show 2–3 (more is choice overload, fewer feels arbitrary), tuned online.
- **Offline metrics up, online down?** §7.2 list: off-policy replay invalidity, simulator gap, friction not captured offline, higher real misparse rate, feedback loops.
- **How would you debug a bad launch?** Traces give per-turn belief/decision/retrieval/generation; localize to parse vs policy vs retrieval vs generation; watch the zero-result-after-filter and turn-count canaries; roll back.
- **How do you prevent over-personalization / filter bubbles?** Maintain exploration so a wrong inferred preference can be corrected, monitor diversity, never let one elicited preference permanently lock out catalog regions.
- **When fine-tune vs prompt the LLM?** Prompt + grounding for the generation/NLU by default; fine-tune only to internalize format/tone or improve constraint-vs-preference classification if prompting plateaus.

---

## 13. Common mistakes

- Framing it as "parse the query then recommend" — skipping the ask-vs-recommend policy that *is* the case.
- Treating every utterance the same — not separating hard constraints (filter) from soft preferences (rank), causing silent over-constraining.
- Letting the LLM invent items/attributes — no grounding, no post-validation against the catalog.
- Assuming you can offline-replay logged conversations against a new dialogue policy — ignoring that dialogue is off-policy/counterfactual.
- Optimizing relevance/acceptance while ignoring friction (turn count) — the new policy "wins" offline and gets abandoned online.
- Over-asking (annoying) or under-asking (bad recs) — not recognizing it's an operating curve with a frontier win condition.
- Jumping to a tool-using agent when a stateful dialogue manager suffices — adding loop/latency/reliability risk for no benefit.
- Forgetting the non-LLM fallback (faceted filters) for when generation degrades.

---

## 14. Transfer: what this case unlocks

- **Files 02 / 03 (feed & search ranking):** the retrieve→rank funnel, personalization, and feedback loops are shared; this case adds the *interactive intent construction* on top — they are the "query is given" version of the same machinery.
- **File 19 (bandits):** the ask-vs-recommend explore/exploit decision and value-of-information are the bandit framework applied to dialogue questions.
- **File 05 (RAG):** grounding / cite-or-abstain becomes "recommend only real catalog items with real attributes."
- **Files 04 / 07 (agents):** the rung-3 agentic extension reuses their control-plane discipline (tool schemas, budgets, validators, handoff).
- **File 06 (LLM eval):** user simulators, the off-policy/counterfactual eval problem, and "online is truth" are eval-platform concepts applied here.
- **General skill:** "turn a vague natural-language goal into a structured, grounded, ranked result with controlled clarification" is the pattern behind every conversational product surface.

---

## 15. Sources

Original guides (kept):
- [IGotAnOffer ML System Design Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [Exponent ML System Design Interview Guide](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)
- [Hello Interview ML/System Design Learning](https://www.hellointerview.com/learn)
- [Designing Machine Learning Systems, Chip Huyen](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [OpenAI Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

Added canonical references (verify titles; well-established works):
- [Christakopoulou et al., "Towards Conversational Recommender Systems," KDD 2016](https://dl.acm.org/doi/10.1145/2939672.2939746)
- [Sun & Zhang, "Conversational Recommender System," SIGIR 2018](https://arxiv.org/abs/1806.03277)
- [Gao et al., "Advances and Challenges in Conversational Recommender Systems: A Survey," 2021](https://arxiv.org/abs/2101.09459)
- [Li et al., "Towards Deep Conversational Recommendations" (ReDial), NeurIPS 2018](https://arxiv.org/abs/1812.07617)
- [Lattimore & Szepesvári, "Bandit Algorithms" (value of information / explore-exploit), 2020](https://tor-lattimore.com/downloads/book/book.pdf)
