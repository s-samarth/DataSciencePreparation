# 14. Document Intelligence Pipeline

**Company tags:** Microsoft, Google, AWS (Textract), fintech, legaltech, healthcare AI, insurtech
**Interview frequency:** High for applied AI roles
**Why it matters:** This case looks like "OCR plus an LLM" and is actually about *economics*: the entire business value is "auto-process X% of documents at an error rate below Y, and pay humans to review the rest." That makes it a **selective-prediction problem at field granularity** — calibrated confidence deciding straight-through vs human review under asymmetric cost — wrapped in a multi-stage cascade where errors compound. If you frame it as "extract fields with a transformer" you have missed the case. If you frame it as "maximize straight-through-processing at a fixed error budget," you have it.

---

## 0. How to use this doc

Two passes.

**Pass 1 (intuition).** Read the prose and transcript. Lock in the spine: **the product is not extraction, it is automation-with-a-trust-boundary.** A perfect extractor that you cannot trust auto-pilots nothing; a mediocre extractor with *calibrated confidence* and a clean human-review lane can automate the easy 75% and route the hard 25%, which is the whole ROI. Everything — grounding, calibration, validation, the review UI — exists to make the auto/review routing decision safe.

**Pass 2 (active recall).** Cover the page. Can you (a) explain why 95% OCR character accuracy can mean 60% field accuracy, and why you measure end-to-end not per-stage, (b) explain why every extracted value must carry a *source span* and what three things that buys, (c) derive the straight-through-processing economics and the per-field confidence threshold under asymmetric cost, and (d) name the deterministic validation backstop that catches what the model cannot? Those four are the case.

**The scaffold (shared across this set):**

> **Clarify → Frame → Data/Labels → Baseline → Model → Eval → Deploy → Monitor**

Bends here: "Model" is a *cascade* (classify → OCR → layout → extract → validate → route), not a single model; "Eval" must be end-to-end and field-level, not per-stage; "Deploy" is mostly an async/batch pipeline whose real-time decision is the *confidence gate*; "Data/Labels" confronts the human-review flywheel as the label engine.

**The senior tell, stated once:** say early that "this is selective prediction — the model's job is not just to extract but to *know when it doesn't know*, because a confident wrong value on a payment amount that auto-pays is far more expensive than abstaining to a human." That reframes accuracy into calibrated-accuracy-at-a-cost, which is the senior lens.

---

## 1. Clarify (scripted, with *why each answer changes the design*)

| Question | Why it changes the design |
|---|---|
| **What document types, and how variable?** | Fixed templates (one insurer's claim form) → rules/template-matching may suffice. High variability (every vendor's invoice looks different) → you need layout-aware ML. "Mixed, mostly semi-structured" is the realistic, hard answer. |
| **Digital-native PDFs or scanned images / photos?** | Digital PDFs have an embedded text layer — no OCR error. Scans/phone photos need OCR, which is the noisiest stage and *caps* downstream accuracy. The mix ratio decides how much of your error budget OCR eats. |
| **What is the cost of a wrong field vs a missed field?** | Asymmetric and field-dependent. A wrong payment amount that auto-pays is a financial loss; a wrong cosmetic field is noise. This sets *per-field* confidence thresholds, not one global one. |
| **Is there a human-review team, and what does a review cost?** | This is the economic core. If reviews cost ~$1 each and you do 1M docs/day, the difference between 75% and 85% straight-through is 100K reviews/day ≈ $100K/day. The whole design optimizes this. |
| **Compliance / auditability?** | Finance, legal, health need every extracted value traceable to a source location in the original document, and full audit. This forces **span-grounded extraction** — non-negotiable. |
| **Latency: interactive or batch?** | Most document processing is async (seconds to minutes is fine). Some is interactive (user uploads, waits). Mostly batch relaxes the latency budget and lets you use heavier models. |

State assumptions and move: mixed semi-structured docs, mix of digital + scanned, asymmetric per-field cost, a review team, compliance-grade audit, mostly async.

---

## 2. Numbers up front (carry them through)

- **Volume:** ~1M documents/day, ~20 fields per document = ~20M field extractions/day. Mix ~60% digital-native PDF, ~40% scanned/photo (OCR needed).
- **The economic target (the headline):** **straight-through-processing (STP) rate** = fraction of documents auto-processed with no human. Suppose baseline STP = 75%, so 250K docs/day go to review. At ~$1/review that is **$250K/day** of human cost. **Pushing STP 75% → 85% saves ~100K reviews/day ≈ $100K/day ≈ $36M/year.** That number is why this case exists — say it out loud.
- **The guardrail that bounds STP:** **critical-field error rate on auto-processed documents < 0.1%.** You cannot buy STP by lowering the bar; the constraint is "automate as much as possible *subject to* the error ceiling on what you auto-process." STP and error rate are a joint operating curve, exactly like fraud's recall-at-fixed-FPR (file 09) and the support agent's containment-at-bounded-wrong-action (file 07).
- **Latency:** async pipeline; per-document end-to-end budget seconds to a few minutes. OCR + a layout model + validation fits comfortably; an LLM extraction call adds ~1–5s and real cost.
- **Cost back-of-envelope:** if extraction uses a vision-LLM at ~$0.01–0.05/doc × 1M/day = $10K–50K/day in model cost — comparable to the review savings, so model choice is an economic decision, not just an accuracy one. Cheap layout transformers (run on your own GPUs) amortize far better at this volume; reserve the expensive LLM for hard/low-confidence docs.
- **Compounding-error math (the OCR insight):** 95% per-character OCR accuracy sounds great, but a 10-character field is correct only if *all 10* chars are right: 0.95^10 ≈ **60%** field-level accuracy. Errors compound multiplicatively across characters *and* across cascade stages. This is why you measure end-to-end, and why OCR quality on the scanned 40% is often the real bottleneck.

---

## 3. The cascade and why errors compound (the conceptual spine)

Draw the pipeline as a cascade and make the compounding-error point explicitly — it is the systems insight most candidates miss.

```text
[1 classify doc type] -> [2 OCR / text-layer extract] -> [3 layout understanding]
   -> [4 field/table extraction] -> [5 validation (deterministic rules)]
   -> [6 CONFIDENCE GATE]  --high-conf-->  straight-through
                            --low-conf-->  human review -> corrections (labels)
```

**Each stage's error caps everything downstream.** If OCR drops a digit, no extractor downstream can recover it. If document classification picks the wrong type, the wrong schema is applied and every field is suspect. End-to-end field accuracy is roughly the *product* of stage accuracies, so a chain of "pretty good" stages yields a mediocre whole. Consequences a senior states:
- **Measure end-to-end and per-field, never per-stage in isolation.** A 99% OCR and 98% extractor still gives ~97% per field, and on a 20-field document the chance *all* fields are right is 0.97^20 ≈ 54% — so "document fully correct" is a much harsher metric than "field correct," and it is the one that decides STP.
- **Invest where the cascade is weakest.** On the scanned 40%, OCR usually dominates the error budget; throwing a bigger extraction model at bad OCR is wasted money.
- **Keep stages separable and observable** so you can attribute an error to the stage that caused it — this is also your debugging path.

This is the same "don't optimize one stage, optimize the funnel" discipline as the retrieval funnel in file 05, applied to extraction.

---

## 4. The data/label problem for *this* domain: the human-review flywheel + span ground truth

Every case has a signature data problem. Document AI's is: **labeled data is expensive (someone must annotate field values *and their locations* on real documents), the document distribution has a long tail of templates you have never seen, and your best label source is the human-review lane itself.**

The senior move is to design the **review→label flywheel** as a first-class system, not an afterthought:
- Every document the confidence gate routes to a human gets *corrected* by that human. Those corrections are **ground-truth labels, for free, on exactly the hard distribution** (the docs the model was unsure about). This is the most valuable training data you can get.
- **Active learning falls out naturally:** you are already labeling the low-confidence / model-disagreement cases, which is precisely what active learning would *choose* to label. The review lane is an active-learning loop in disguise — say this.
- **Span-grounded labels.** Ground truth is not just "invoice_total = 1234.56"; it is "invoice_total = 1234.56, located at bbox (x,y,w,h) on page 2." This makes labels reusable, lets you train grounding, and lets reviewers verify by jumping to the location.
- **Label quality / contestation:** even humans disagree on messy scans and ambiguous fields; measure inter-annotator agreement on a sample, use adjudication for critical fields (cross-link file 06/10 on contested labels).

**Data quality risks specific here:** new templates (distribution shift → a vendor changes their invoice layout and extraction silently degrades), OCR quality drift (a new scanner, worse photos), and PII everywhere (documents are full of it → redaction + access control + audit are mandatory).

---

## 5. The baseline → why-it-breaks → next-rung ladder

**Rung 0 — Template rules + regex.** Per-template anchors ("the number to the right of 'Total:'"), regex for structured fields.
- *Works:* fixed, known forms; deterministic, cheap, perfectly auditable, trivially calibrated (it matched or it didn't).
- *Breaks:* any layout variation or unseen template; maintaining hundreds of templates is unsustainable; useless on the long tail. **Trigger:** the Nth vendor template, or a layout change that silently breaks extraction.

**Rung 1 — OCR + classical NLP extraction.** OCR to text, then NER / pattern models over the token stream.
- *Adds:* handles semi-structured docs without per-template rules.
- *Breaks:* throws away **layout** — it flattens a 2-D document into 1-D text, losing the spatial cues (this number is in the "Total" column, this is a table cell) that disambiguate fields; weak on tables and multi-column; OCR noise compounds. **Trigger:** tables and spatially-laid-out forms where text order ≠ reading order.

**Rung 2 — Layout-aware transformer extraction (recommended production default).** Models that jointly consume text + 2-D position + (optionally) image pixels: LayoutLMv3, Donut (OCR-free encoder-decoder), TILT. Token-classification or seq2seq into the schema, with table-extraction support (TableFormer-style).
- *Adds:* uses spatial structure → robust field + table extraction across templates; runs on your own GPUs (cheap at 1M/day); **outputs token-level scores you can calibrate** for the confidence gate.
- *Breaks:* needs annotated documents (fed by the flywheel); a brand-new document type with no labels is still hard. **Trigger:** zero-shot need on a new doc type, or a complex reasoning extraction ("net 30 from invoice date" = compute the due date).

**Rung 3 — Vision-LLM extraction with schema + grounding + validation (the flexible extension).** A multimodal LLM extracts to a JSON schema with a prompt, *required to emit a source span/quote for each value*, behind deterministic validation.
- *Adds:* zero/few-shot across unseen doc types, handles reasoning fields and free-form documents (contracts, letters).
- *Costs:* expensive at scale, **hallucinates** (will confidently invent a plausible value), and is **poorly calibrated** (its stated confidence is not trustworthy). So you only use it where it earns its cost — hard/low-confidence/new-type docs — and you *force grounding*: if it cannot point to a span in the source, reject the value. **Do not** make it the default for the high-volume known types; the layout transformer is cheaper and more calibratable there.

Meta-rule out loud: "I default to a layout-aware transformer for the high-volume known types because it is cheap, fast, and calibratable, and I route the long tail / hard cases to a grounded vision-LLM. The whole system is organized around the confidence gate, not around one model."

---

## 6. The architecture explained to the floor

Two mechanisms deserve real depth: **span-grounded extraction** and the **confidence-gated STP economics.**

### 6.1 Span-grounded extraction (why every value carries a location)
Each extracted value is `{field, value, source_span (page + bbox / character offsets), confidence}`. Requiring the span buys three things, all load-bearing:
1. **Hallucination defense.** If the model (especially an LLM) cannot localize a value to actual pixels/text in the document, the value is fabricated — reject or route to review. Grounding turns "trust the model" into "verify against the source."
2. **Review efficiency.** The reviewer UI jumps straight to the highlighted box; verifying a value takes seconds instead of hunting the page. This directly lowers per-review cost, which is the economic lever.
3. **Audit / compliance.** Every decision is traceable to where in the source it came from — the legal/finance requirement. This is the document-AI analogue of RAG's "cite-or-abstain" (file 05).

### 6.2 The validation layer (deterministic backstop, independent of the model)
Before the confidence gate, run **deterministic business rules** — cheap, high-precision, and crucially *independent of the model's confidence*:
- **Format/schema checks:** dates parse, amounts are numeric, IDs match expected patterns.
- **Checksums:** IBAN/credit-card check digits, invoice `total == sum(line_items) + tax`.
- **Cross-field consistency:** dates ordered, quantities × unit price = line total.
- **External lookups:** does this vendor/account exist in our system? Is this PO valid?

Validation catches errors the model is *confidently wrong* about (calibration's blind spot) and is often the difference between 0.1% and 1% critical-field error. A value that fails validation is forced to review regardless of model confidence. State this: validation is a second, orthogonal safety net to calibration.

### 6.3 The confidence gate and STP economics (the heart)
This is selective prediction at field/document granularity. Per field you have a **calibrated** confidence (see §7.1 — calibrated, not raw softmax/LLM self-report). The gate:
- A document is **straight-through** only if *all critical fields* clear their thresholds *and* pass validation; otherwise it routes to review (whole doc, or just the uncertain fields for partial review).
- **Thresholds are per-field and cost-driven**, not global. Critical fields (payment amount, account number, patient ID) get high thresholds → favor sending to human over auto-erroring. Low-stakes fields get low thresholds → auto-process to maximize STP. This is the cost-sensitive thresholding of file 09 applied field-by-field.
- The operating point is chosen on the **STP-vs-error curve**: for a target critical-field error ≤ 0.1%, pick the thresholds that maximize STP. Improving the *model* shifts the whole curve up (more STP at the same error); improving *calibration* lets you sit safely closer to the threshold.

### 6.4 The three paths, named (senior framing)
- **Serving/processing path:** the cascade §3, mostly async, with the confidence gate as the real-time routing decision.
- **Data path:** OCR/layout features, document-type schemas, the validation rule set, the labeled corpus.
- **Feedback path:** human-review corrections → labels → retrain (the flywheel §4) + drift/quality monitors.

### 6.5 Costs
At 1M docs/day, route by cost: digital-native + known-template through the cheap layout transformer; reserve the vision-LLM for scanned long-tail / low-confidence / new types. Cache OCR results (re-processing the same doc is common). The model-cost vs review-cost tradeoff is explicit: spending more on a better model is worth it only up to the review dollars it saves.

---

## 7. Evaluation

### 7.1 The metrics, and the calibration point
- **Field-level precision / recall / F1 and exact-match**, sliced by document type, field, and digital-vs-scanned. (Recall matters: a missed field is a failure too.)
- **Table-extraction accuracy** (cell-level + structure) — tables are their own hard sub-problem; report separately.
- **End-to-end document accuracy** ("all fields correct") — the harsh metric that actually governs STP (§3).
- **Critical-field error rate on auto-processed docs** — the guardrail; must stay < 0.1%.
- **STP rate** — the business metric.
- **Calibration (the senior detail):** the confidence gate is only as good as the calibration of the confidence. Plot a **reliability diagram**; raw softmax and especially LLM self-reported confidence are *not* calibrated. Apply temperature scaling / Platt / isotonic so that "confidence 0.95" really means ~95% correct. Without calibration your STP threshold is meaningless. (Cross-link file 07.)

### 7.2 The offline↔online gap (the trap)
*"Field F1 was 0.97 offline; why is the production correction rate higher?"* Causes, in order:
1. **New / shifted templates** — offline test set was the templates you had; production has new vendor layouts the model never saw. The #1 cause; document distributions have a fat long tail.
2. **OCR quality skew** — offline docs were clean scans; production has phone photos, skew, glare, low DPI. The compounding-error multiplier (§2) bites hardest here.
3. **Label/eval definition mismatch** — offline "exact match" vs what the business actually tolerates (e.g., date format normalization), so offline under- or over-states real accuracy.
4. **Miscalibration** — confidence looked fine offline but is overconfident on the live distribution, so the gate lets through low-quality auto-processed docs.
5. **Selection bias in the offline label set** — you labeled the easy docs; the hard tail is underrepresented, inflating offline numbers.

Cure: continuously refresh the eval set from the *production* distribution via the flywheel, monitor per-template and per-field, and watch calibration drift.

### 7.3 A fully-specified A/B test
You can A/B a new extraction model:
- **Hypothesis:** model v5 raises STP at the fixed 0.1% critical-field error ceiling vs v4.
- **Unit:** document (or document stream), randomized; for fairness, stratify by document type.
- **Primary metric:** STP rate at the fixed error ceiling. **Guardrails:** critical-field error rate (must not exceed 0.1%), reviewer correction rate, end-to-end latency, model cost/doc.
- **The subtlety:** for auto-processed docs you do not have ground truth (that is the point of auto-processing). So measure true error via a **random audit holdout** — a small fraction of auto-processed docs sent to human review anyway, purely to estimate the real auto-process error rate (this mirrors fraud's random holdout in file 09 and is the cleanest way to know your STP isn't quietly leaking errors).
- **Runtime/ramp:** shadow new model → canary → ramp; ≥ a full business cycle to cover document-mix seasonality.
- **Rollback trigger:** critical-field error breaches ceiling, correction-rate spike, or cost blowout.

---

## 8. Deployment, monitoring, fallback, incident response

- **Rollout:** shadow the new extraction model on live traffic (extract, log, do not auto-act), compare against current + the random-audit ground truth, then canary/ramp. Schema changes (new field, changed validation) deploy the same disciplined way.
- **Monitoring:** STP rate and correction rate per document type (the leading indicators), per-field error, OCR quality scores, **new-template / out-of-distribution detection** (embedding-distance or low-confidence clustering → a vendor changed layout), calibration drift, latency, model cost/doc. A correction-rate spike on one template = that template drifted; a global spike = OCR or model regression.
- **Fallback:** if the ML extractor or OCR is degraded, fall back to template rules for known types and route everything else to human review — never auto-process on a degraded model. Graceful degradation = lower STP, not higher error.
- **Incident response:** freeze model/schema version, identify the affected document type/field from the structured logs + spans, roll back, re-queue affected auto-processed docs for re-review if a wrong-field leak is suspected, add the failing template to the training set. Because every value is span-grounded and logged, forensics is tractable.
- **PII/compliance:** redact PII in logs, access-control documents per tenant/role, retain audit trail of every extraction and decision.

---

## 9. Full one-hour interview transcript

**[0:00] INTERVIEWER:** Design a system that ingests business documents and extracts structured information for downstream workflows.

**[0:30] YOU:** Let me scope it, because the design swings on a few things. What document types and how variable are they — one fixed form, or every vendor's invoice? Are they digital-native PDFs with a text layer, or scanned images and phone photos that need OCR? Is there a human-review team, and roughly what does one review cost? And do we need compliance-grade auditability?

**[1:15] INTERVIEWER:** Mixed semi-structured docs, lots of templates, about 60% digital and 40% scanned. There's a review team, reviews cost about a dollar each. Yes, finance-grade audit.

**[1:30] YOU:** Then let me state the framing up front, because I think it's the whole case: the product is not "extraction," it's *automation with a trust boundary*. The value is auto-processing as many documents as possible without a human, subject to an error ceiling on what we auto-process. So this is **selective prediction** — the model has to know when it doesn't know — and the central metric is straight-through-processing rate at a fixed critical-field error rate. Let me put numbers on it: say a million docs a day, baseline 75% straight-through means 250K reviews a day at a dollar each, $250K/day. Pushing straight-through from 75 to 85 percent saves 100K reviews a day, about $36M a year. That number is why the system exists, and it's bounded by a guardrail — critical-field error on auto-processed docs under 0.1%. I can't buy straight-through by lowering the bar.

**[3:30] INTERVIEWER:** Makes sense. How do you build the extraction itself?

**[3:40] YOU:** As a cascade, and I want to flag the compounding-error point because it drives where I invest. Classify doc type, then OCR or read the text layer, then layout understanding, then field and table extraction, then validation, then the confidence gate. Each stage caps everything downstream — if OCR drops a digit, no extractor recovers it. And errors compound multiplicatively: 95% per-character OCR sounds great, but a 10-character field needs all ten right, 0.95^10 is about 60% field accuracy. On a 20-field document, even 97% per field means only about half of documents are fully correct. So I measure end-to-end and per-field, never per-stage, and on the scanned 40% the OCR stage usually eats most of my error budget — that's where I'd invest before buying a fancier extractor.

**[5:30] INTERVIEWER:** What model do you use to extract?

**[5:40] YOU:** I climb a ladder. Rung zero, template rules and regex — great for fixed forms, deterministic and perfectly calibrated, but unmaintainable across hundreds of templates and dead on the long tail. Rung one, OCR plus classical NLP — but it flattens a 2-D document to 1-D text and throws away layout, which is exactly the signal that says "this number is in the Total column." So my production default is rung two: a **layout-aware transformer** like LayoutLMv3 or Donut that jointly consumes text, 2-D position, and the image. It's robust across templates, runs cheaply on my own GPUs at this volume, handles tables, and — importantly — gives me token-level scores I can *calibrate* for the gate. For the long tail and reasoning fields, rung three: a vision-LLM with a schema prompt. It's flexible zero-shot but expensive, it hallucinates, and it's poorly calibrated — so I only send hard or new-type docs to it, and I *force grounding*.

**[8:00] INTERVIEWER:** Say more about grounding.

**[8:10] YOU:** Every extracted value carries a source span — page plus bounding box. That buys three things. One, hallucination defense: if the model can't point to actual pixels in the document, the value is invented, so I reject it or route to review. Two, review efficiency: the reviewer UI jumps straight to the highlighted box, so verifying takes seconds, which directly lowers my per-review cost — the economic lever. Three, audit: every value is traceable to where it came from, which is the finance requirement. It's the document-AI version of cite-or-abstain.

**[9:30] INTERVIEWER:** How do you decide auto-process versus review?

**[9:40] YOU:** Two orthogonal gates. First, a deterministic **validation** layer, independent of the model: format checks, checksums — invoice total equals sum of line items, IBAN check digits — cross-field consistency, and external lookups like "does this vendor exist." This catches errors the model is *confidently wrong* about, which is calibration's blind spot, and it's often the difference between 0.1% and 1% error. Anything that fails validation goes to review regardless of confidence. Second, the **confidence gate**: a document goes straight-through only if all critical fields clear their thresholds and pass validation. And the thresholds are **per-field and cost-driven** — payment amount and account number get high thresholds so I'd rather send to a human than auto-error, while cosmetic fields get low thresholds to maximize straight-through. I pick the operating point on the straight-through-versus-error curve for the 0.1% ceiling.

**[12:00] INTERVIEWER:** You keep saying calibrated confidence. Why does that matter?

**[12:10] YOU:** Because the gate is only as good as the calibration. Raw softmax, and especially an LLM's self-reported confidence, are not calibrated — "0.95" doesn't mean 95% correct. So I plot a reliability diagram and apply temperature scaling or isotonic regression so the number means what it says. Without that, my straight-through threshold is meaningless and I'm either leaking errors or reviewing too much. Same discipline as a deferral system in the support-agent case.

**[13:30] INTERVIEWER:** Where do your training labels come from?

**[13:40] YOU:** The human-review lane *is* my label engine — that's the flywheel and I'd design it as first-class. Every doc the gate sends to a human gets corrected, and those corrections are free ground-truth labels on exactly the hard distribution the model was unsure about. That's active learning for free — I'm already labeling the low-confidence and disagreement cases, which is what active learning would pick. I capture labels with spans so they're reusable, and for critical fields I adjudicate disagreements between reviewers. Then scheduled and drift-triggered retrains feed off that corpus.

**[15:30] INTERVIEWER:** Offline field F1 is 0.97 but the production correction rate is higher. Why?

**[15:45] YOU:** In priority order: new or shifted templates — a vendor changed their invoice layout and my offline test set never had it; document distributions have a fat long tail, so this is the usual culprit. Second, OCR-quality skew — offline scans were clean, production has phone photos with glare and skew, and the compounding-error multiplier bites hardest there. Third, label-definition mismatch — offline exact-match versus what the business actually tolerates. Fourth, miscalibration on the live distribution, so the gate is overconfident and leaks low-quality auto-processed docs. Fifth, selection bias — I labeled the easy docs, so offline is inflated. The fix is to refresh the eval set continuously from the production distribution via the flywheel and to monitor per-template and calibration drift, not just the aggregate.

**[18:00] INTERVIEWER:** How do you even know your auto-processed error rate, if you didn't review those docs?**

**[18:10] YOU:** A random audit holdout — I send a small random fraction of *auto-processed* docs to human review anyway, purely to estimate the true auto-process error rate. Same idea as a random holdout in fraud, where approved transactions are otherwise unlabeled. Without it I'm blind to exactly the population I most need to trust. That's also my A/B primary instrument: to ship a new model I require it to raise straight-through at the fixed 0.1% ceiling, with the audit holdout measuring the real error, and correction rate, latency, and cost-per-doc as guardrails.

**[20:00] INTERVIEWER:** A new model regresses in production. What happens?**

**[20:10] YOU:** Monitoring catches it as a correction-rate spike — global means OCR or model regression, single-template means that template drifted. Freeze the model and schema version, identify the affected type and field from the structured logs and spans, roll back. If I suspect a wrong-field leak got auto-processed, I re-queue the affected auto-processed docs for re-review. Fallback while fixing is graceful degradation: fall back to template rules for known types and route the rest to humans — lower straight-through, never higher error. Then add the failing template to the training set so it can't recur.

**[22:00] INTERVIEWER:** How do you manage cost at a million docs a day?**

**[22:10] YOU:** Route by cost. Digital-native, known-template docs go through the cheap layout transformer on my own GPUs. I reserve the vision-LLM — which might be 1 to 5 cents a doc, so $10K–50K/day if I ran it on everything — for the scanned long tail, low-confidence, and new types. I cache OCR results since the same document gets reprocessed often. And I treat model spend and review spend as one budget: a better model is worth it only up to the review dollars it saves.

**[24:00] INTERVIEWER:** Wrap up.

**[24:10] YOU:** To close: the product is automation with a trust boundary, so I optimize straight-through-processing at a fixed critical-field error ceiling. I extract with a calibrated layout-aware transformer for the common case and a grounded vision-LLM for the tail, ground every value to a source span for hallucination defense, review efficiency, and audit, back it with a deterministic validation layer, gate auto-vs-review on per-field cost-sensitive calibrated thresholds, and turn the review lane into my label flywheel. This is exactly the invoice-and-document extraction workflow I've built — field-level confidence routing and human-in-the-loop correction — generalized to a platform.

### Why this transcript works
- **Leads with the economics** (STP at a fixed error ceiling) and quantifies it — reframes "extraction" as selective prediction.
- **Names the compounding-error cascade** and uses it to decide where to invest (OCR on scans), a real systems insight.
- **Two orthogonal safety nets** — calibrated confidence *and* deterministic validation — and explains why both.
- **Span grounding** with its three concrete payoffs, including the LLM-hallucination defense.
- **Knows you can't measure auto-process error without a random audit holdout** — the subtle, senior point.
- **Designs the review→label flywheel** as the data engine and connects it to active learning.
- **Closes on real invoice/document experience** without overclaiming.

---

## 10. Junior vs senior contrast

| Dimension | Junior | Senior |
|---|---|---|
| Framing | "Extract fields with a transformer / LLM." | "Selective prediction: maximize straight-through at a fixed critical-field error ceiling." |
| Pipeline | "OCR then a model." | Cascade with **compounding errors**; measure end-to-end + per-field; invest in the weakest stage. |
| Model choice | "Use GPT-4V for everything." | Calibrated layout transformer for the common case; grounded vision-LLM for the tail only (cost + hallucination + calibration). |
| Trust | "The model outputs the value." | **Span-grounded** values + **deterministic validation** backstop; reject ungrounded outputs. |
| Routing | "Use the model's confidence." | **Calibrated** (reliability diagram), **per-field cost-sensitive** thresholds; STP-vs-error operating curve. |
| Labels | "Annotate a training set." | Human-review **flywheel** = free labels on the hard distribution = active learning. |
| Eval | "Field F1." | End-to-end doc accuracy, critical-field error guardrail, **random audit holdout** to measure auto-process error. |
| Ops | "Retrain periodically." | New-template/OOD detection, calibration drift, per-template correction-rate alarms. |

---

## 11. One-page whiteboard cheat sheet

```text
SPINE: product = AUTOMATION w/ a trust boundary -> SELECTIVE PREDICTION
       maximize STP rate s.t. critical-field error <= 0.1%

NUMBERS: 1M docs/day, ~20 fields; 60% digital / 40% scanned
         STP 75%->85% saves ~100K reviews/day ~= $36M/yr  <-- why this case exists
         compounding: 0.95^10 char ~= 60% field; 0.97^20 fields ~= 54% doc fully-correct

CASCADE: classify -> OCR/text-layer -> layout -> extract -> VALIDATE -> CONFIDENCE GATE -> STP | review
         each stage CAPS downstream; measure end-to-end + per-field; invest in weakest stage (OCR on scans)

LADDER: 0 template/regex -> 1 OCR+NLP (loses layout) -> 2 LAYOUT TRANSFORMER (LayoutLMv3/Donut, calibratable) [default]
        -> 3 vision-LLM w/ schema+GROUNDING (tail/new-type only; expensive, hallucinates, miscalibrated)

GROUNDING: every value -> source span. buys: (1) anti-hallucination (2) review speed (3) audit

TWO GATES (orthogonal):
  VALIDATION (deterministic): checksums, total=sum(lines), cross-field, external lookup  <- catches confident-wrong
  CONFIDENCE (calibrated, per-field cost-driven thresholds): critical fields high thresh

CALIBRATION: reliability diagram; raw softmax / LLM self-confidence NOT calibrated -> temp scaling / isotonic

LABELS: human-review lane = FREE labels on hard distribution = active learning flywheel; capture spans

EVAL: field P/R/F1 + exact, table acc, END-TO-END doc acc, critical-field error (guardrail), STP rate
      auto-process error -> RANDOM AUDIT HOLDOUT (can't know it otherwise)

OFFLINE!=ONLINE: new templates -> OCR skew (photos) -> label-def mismatch -> miscalibration -> selection bias
```

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x volume?** Cost routing gets stricter (more on cheap layout models, LLM only on the tail), OCR caching and batch GPU inference matter, and the review team becomes the bottleneck — so STP improvements have the highest ROI.
- **How do you handle a brand-new document type with no labels?** Route to the grounded vision-LLM (zero-shot) + heavy human review initially; the review corrections bootstrap labels for the cheap layout model, then graduate the type to the cheap path. Flywheel in action.
- **How do you pick confidence thresholds?** Per-field, on the STP-vs-error curve, at the cost ratio of (wrong auto-processed field) vs (human review). Critical fields get conservative thresholds; require calibration first.
- **Offline F1 high, production correction rate high — why?** §7.2 list: new templates, OCR skew on scans, label-definition mismatch, miscalibration, offline selection bias.
- **How would you debug a quality regression?** Correction-rate alarms localize it (global = OCR/model, per-template = drift); structured logs + spans pin the field; roll back; re-queue suspect auto-processed docs.
- **How do you prevent silently auto-processing errors?** Random audit holdout to measure true auto-process error, calibration monitoring, and the deterministic validation backstop.
- **Tables specifically?** Treat as a separate sub-problem (structure + cell extraction, TableFormer-style), eval cell-level + structure separately — tables are where most pipelines silently fail.

---

## 13. Common mistakes

- Framing it as "extract fields" instead of "maximize straight-through-processing at a fixed error ceiling" — missing the economics that *is* the case.
- Ignoring the compounding-error cascade and measuring per-stage accuracy that looks great but yields a mediocre whole.
- Reaching for a vision-LLM for everything — ignoring cost at scale, hallucination, and that it is poorly calibrated for the confidence gate.
- No span grounding — can't defend against hallucination, can't make review efficient, can't pass audit.
- Using raw model confidence for the gate without calibration; one global threshold instead of per-field cost-sensitive thresholds.
- Forgetting the deterministic validation backstop (checksums, totals, cross-field) that catches confidently-wrong values.
- No plan to measure auto-process error (no random audit holdout) — flying blind on the population you most need to trust.
- Treating the human-review team as a cost center instead of the label flywheel / active-learning loop.

---

## 14. Transfer: what this case unlocks

- **File 07 (support agent) & 09 (fraud):** the calibrated-confidence + cost-sensitive-threshold + selective-prediction machinery is identical; here it routes documents, there it routes actions/transactions. The random-audit-holdout trick comes straight from fraud's censored-label problem.
- **File 05 (RAG):** span grounding is cite-or-abstain; the "separate retrieval quality from generation quality" discipline maps to "separate OCR/extraction quality from validation."
- **File 06 (LLM eval):** calibration (reliability diagrams), contested labels / inter-annotator agreement, and "shadow the model" transfer directly.
- **File 12 (deployment platform):** shadow→canary→ramp, drift/OOD detection (new templates), and the feedback path are the same MLOps spine.
- **General skill:** "automate the confident majority, route the uncertain tail to humans, and turn that human work into your next training set" is the universal pattern for any human-in-the-loop ML product.

---

## 15. Sources

Original guides (kept):
- IGotAnOffer ML System Design Guide: https://igotanoffer.com/en/advice/machine-learning-system-design-interview
- Exponent ML System Design Interview Guide: https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide
- Hello Interview ML/System Design Learning: https://www.hellointerview.com/learn
- Designing Machine Learning Systems, Chip Huyen: https://huyenchip.com/machine-learning-systems-design/toc.html
- Google Research: RAG context sufficiency: https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/
- OpenAI text generation guide: https://platform.openai.com/docs/guides/text-generation

Added canonical references (verify titles; well-established works):
- Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," 2022: https://arxiv.org/abs/2204.08387
- Kim et al., "OCR-free Document Understanding Transformer (Donut)," 2021: https://arxiv.org/abs/2111.15664
- Xu et al., "LayoutLM: Pre-training of Text and Layout for Document Image Understanding," KDD 2020: https://arxiv.org/abs/1912.13318
- Nassar et al., "TableFormer: Table Structure Understanding with Transformers," CVPR 2022: https://arxiv.org/abs/2203.01017
- Guo et al., "On Calibration of Modern Neural Networks" (temperature scaling), ICML 2017: https://arxiv.org/abs/1706.04599
- Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks," NeurIPS 2017: https://arxiv.org/abs/1705.08500
