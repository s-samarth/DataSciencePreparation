# Hallucination Mitigation Beyond "Just Use RAG"

Everyone knows "use RAG." You differentiate in interviews by knowing the full stack. **Hallucinations cannot be fully eliminated at the model level** — this is mathematically proven (Xu et al.). Your job is to shrink the effective input space and add safety nets. Five layers: prevention, structural constraints, detection, correction, measurement.

!!! tip "Rapid Recall"
    Hallucinations are unsolvable at the model level (proven). Build layered defenses. **Prevention:** system-prompt engineering, RAG grounding, few-shot "I don't know" examples, retrieval-constrained CoT. **Structural constraints:** structured output / JSON schema (turn LLM calls into typed function calls — single biggest engineering lever), constrained decoding, tool-use forcing. **Detection:** self-consistency (N samples, agreement = high confidence), LLM-as-a-judge verification, NLI-based grounding checks, claim decomposition + per-fact verification. **Correction:** guardian agents, dedicated hallucination correctors (Vectara), human-in-loop escalation. **Measurement:** RAGAS faithfulness, HHEM, TruthfulQA. **Bare LLM: 15-25% hallucination rate. LLM + RAG: 5-10%. Full stack: <1% on bounded domains.**

## §1 The fundamental fact

**Hallucinations cannot be fully eliminated at the model level.** This is mathematically proven (Xu et al.'s theorems): over an infinite input space, any computable function will diverge from ground truth somewhere. Your job is to **shrink the effective input space** and **add safety nets** so that hallucinations that slip through do not cause harm.

## §2 The full mitigation stack (layered defense)

### 2.1 Layer 1 — Prevention (before generation)

**a) System prompt engineering**

- Explicitly forbidden behaviors ("do not invent policies").
- Uncertainty clause ("if you do not know, say 'I do not have that information'").
- Source-only clause ("answer only using the provided documents").

**b) RAG (grounding)**

The 60% of the fix on its own. Advanced variants:

- **RE-RAG**: confidence scores on retrieved docs; falls back to parametric knowledge if retrieval quality low.
- **C-RAG**: contrastive RAG, generates explanations for why retrieved context supports the answer.

**c) Few-shot examples** demonstrating correct "I don't know" behavior. Model learns the pattern from 3-5 examples in the prompt.

**d) Chain-of-thought constrained by retrieval.** Force each reasoning step to reference a specific retrieved passage.

### 2.2 Layer 2 — Structural constraints (during generation)

**a) Structured output / JSON schema**

- Enforce a schema (OpenAI structured outputs, Pydantic, outlines, guidance, llguidance).
- Model can only fill defined fields; cannot invent narrative text.
- **Turn LLM calls into typed function calls — this is the single biggest engineering lever** for production reliability.

**b) Constrained decoding**

- At each generation step, mask out tokens that would violate grammar/regex/JSON.
- Lossless quality impact for schema compliance.
- Libraries: outlines, guidance, lm-format-enforcer.

**c) Tool use forcing**

- When the model needs a fact (date, calculation, database lookup), force it to call a tool rather than generate from memory.
- E.g., do not let the model "remember" current stock prices — force an API call.

### 2.3 Layer 3 — Detection (after generation, before serving)

**a) Self-consistency** (Wei et al.)

- Generate N responses with temperature > 0.
- Compare them; agreement on critical fields = high confidence.
- Disagreement = hallucination signal → retry or escalate.
- Downside: N× cost.

**b) LLM-as-a-judge verification**

- Separate LLM call: "Is this response factually consistent with these retrieved documents?"
- Works well, adds 1 LLM call of latency and cost.
- 2026 standard: use a cheaper model as judge (GPT-5 mini verifying Opus output).

**c) NLI-based grounding checks**

- Run a small natural language inference model to check if each claim in the output is entailed by the retrieved context.
- NVIDIA NeMo Guardrails, Amazon Bedrock Contextual Grounding, Fiddler do this.

**d) Claim decomposition + individual fact-checking**

- Break response into atomic claims → verify each against source.
- Galileo, Vectara HCMBench, Guardrails AI use this pattern.

### 2.4 Layer 4 — Correction / remediation

- **Guardian agents** — separate agent monitors main agent, can rollback or retry.
- **Vectara Hallucination Corrector** — dedicated model trained to fix hallucinations post-hoc.
- **Human-in-the-loop escalation** — low-confidence outputs route to human review.

### 2.5 Layer 5 — Measurement

Hallucination metrics you should know.

- **Faithfulness** (RAGAS): does the response align with retrieved context?
- **Answer relevancy** (RAGAS): does the response address the question?
- **Context precision / recall** (RAGAS): is retrieval pulling the right stuff?
- **HHEM** (Vectara's Hughes Hallucination Evaluation Model) — standard open benchmark.
- **TruthfulQA** — adversarial factuality benchmark.

## §3 The stack in practice (2026 production pattern)

```
User Query
   ↓
[Semantic cache check] → cache hit → return
   ↓
[Router: complexity classifier]
   ↓
[RAG retrieval with reranking]
   ↓
[LLM with structured output + citation requirement]
   ↓
[NLI grounding check] → fail → regenerate with stricter prompt
   ↓
[Claim-level fact check] → fail → escalate to human
   ↓
[Output + citations + confidence score]
```

Reported effectiveness:

- **Bare LLM:** 15-25% hallucination rate.
- **LLM + RAG:** 5-10%.
- **Full stack above:** **<1% on bounded domains**.

## §4 Industry specifics

- **Finance / legal**: mandate source citations, human-in-loop above confidence threshold.
- **Healthcare**: multi-agent validation (common in 2026), regulatory rejection of "self-certified" AI outputs.
- **Customer support**: aggressive prompt constraints, escalation paths, monitoring dashboards.
- **Code**: compile/test loops are the best hallucination check — if it compiles and passes tests, the "hallucination" is moot.

## §5 The interview answer framing

When asked "how do you handle hallucinations?", structure your answer as:

1. Acknowledge it is fundamentally unsolvable at the model level.
2. Build layered defenses: prevention → constraints → detection → correction → measurement.
3. Pick layers based on cost, latency, and stakes.
4. Measure with RAGAS or equivalent; monitor drift in production.

This signals you understand the full stack, not just RAG as a silver bullet.

## Interview Questions

**Q1: You are asked "how do you handle hallucinations?" by a PM interviewer. Walk me through your answer.**

I would frame it as layered defense because the problem is mathematically unsolvable at the model level. Layer 1, prevention: system prompts forbidding invention, RAG for grounding, few-shot examples of "I don't know" behavior. Layer 2, structural constraints: structured output schemas, constrained decoding for JSON, tool use for factual lookups. Layer 3, detection: self-consistency across multiple generations, LLM-as-a-judge verification, NLI-based grounding checks. Layer 4, remediation: guardian agents, human-in-loop for low-confidence outputs. Layer 5, measurement: RAGAS faithfulness, HHEM, production monitoring. The pick depends on cost, latency, and stakes — healthcare needs all five layers, an internal coding assistant needs two. Reported effectiveness: bare LLM 15-25% hallucination rate; full stack under 1% on bounded domains.

**Q2: Why is structured output the single biggest engineering lever?**

Because it turns an open-ended generative task ("write something") into a constrained classification-style task ("fill these typed fields"). The model cannot invent narrative around facts because the schema does not have a field for narrative; it must produce values for the fields you specified. Combined with constrained decoding (which masks tokens that would violate the schema), it makes JSON-valid output near-100%. Most production hallucinations in shipping products are not "wrong fact" but "made-up fact in extra narrative"; structured output removes the extra narrative entirely.

**Q3: What is self-consistency and when is it worth the cost?**

Self-consistency generates N responses (typically 5-10) with temperature > 0, then compares them on critical fields. Agreement = high confidence; disagreement = hallucination signal that triggers retry or escalation. Cost is N× per query, so it is worth it for high-stakes domains (medical, legal, financial) where you can pay the cost and where false confidence is more dangerous than added latency. Skip it for low-stakes high-volume chat where the average user prefers fast wrong answers to slow right ones.

**Q4: How does NLI-based grounding check work?**

Run a small natural language inference (NLI) model — like DeBERTa-v3-MNLI or a custom-trained verifier — that takes a claim and a context document and outputs entailment / neutral / contradiction. Apply per-claim from the LLM output against the retrieved RAG context. Any contradiction or unsupported claim triggers a regenerate or escalate. NeMo Guardrails, Bedrock Contextual Grounding, and Fiddler ship this pattern. Cheaper than LLM-as-a-judge (small model, batchable) and explicit per-claim.

**Q5 (Trap): A customer wants "zero hallucinations." What do you tell them?**

That it is mathematically impossible at the model level. What you can deliver: <1% measured hallucination rate on bounded domains via the five-layer stack, plus an SLA-backed escalation path so that the <1% does not reach end users without human review. Reframing from "no hallucinations" to "managed hallucination rate with safety nets" is the right interview move; promising zero hallucinations is either lying or signaling you do not understand the math.
