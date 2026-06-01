# 10. Content Moderation / Policy Enforcement System

**Company tags:** Meta, Google/YouTube, TikTok, Reddit, OpenAI-style platforms
**Interview frequency:** High
**Why it matters:** Moderation looks like "classify bad posts" but the real problem is subtler: you are minimizing harmful *exposure* across billions of views, under a review budget that can touch only a sliver of content, against labels that humans themselves disagree on. The senior signal is measuring the right thing (prevalence) and spending scarce review where expected harm is highest.

---

## 0. How to use this doc

Built two ways; read it twice.

1. **As a thinking guide.** The headers are the whiteboard order. Internalize the *triggers* for each rung.
2. **As a worked transcript.** Section 11 is a full timestamped hour. Cover the `YOU:` lines and answer from memory.

The one idea to carry out: **moderation is not "classify each post." It is "minimize harmful exposure (prevalence) under finite review capacity, where harm ≈ severity × reach, and where the ground-truth labels are themselves contested." You measure success on a random sample of *impressions*, not on the flagged pile, and you spend review capacity on the highest expected-harm content.** Say that and you stop sounding like someone who'd train a toxicity classifier and call it done.

Scaffold (identical across all cases):

```
Clarify -> Frame -> Data/Labels -> Baseline -> Model -> Eval -> Deploy -> Monitor
```

This case is won in **Eval** (prevalence vs flagged-precision) and **Data/Labels** (contested ground truth). Spend time there.

---

## 1. The reusable scaffold, stated once

| Phase | The question |
|---|---|
| Clarify | What harms, what surfaces, what enforcement actions, what's the latency-vs-reach race? |
| Frame | What's the target, given policies are ambiguous and reviewers disagree? |
| Data / Labels | Where do labels come from, and how contested/biased are they? |
| Baseline | Simplest shippable thing, and what breaks it? |
| Model | Hash-match + multimodal classifiers + triage, explained to the floor. |
| Eval | Prevalence (sampled impressions), severity-weighted recall; the offline/online gap. |
| Deploy | Three paths; the virality race; the appeals loop. |
| Monitor | What pages someone; the fallback; reviewer wellness. |

---

## 2. Clarify requirements (scripted)

| Question | Why it changes the design |
|---|---|
| "Which harms, and is CSAM/terrorism in scope?" | The most severe classes (CSAM, credible threats) get *zero-tolerance, pre-publish, hash-based* handling with legal obligations and almost no precision/recall trade allowed. Other classes (spam, mild harassment) tolerate softer enforcement. Severity tiers drive everything. |
| "What content types — text, image, video, audio, livestream?" | Multimodal. Video and livestream are the hardest: livestream can't be fully pre-reviewed, so you need fast in-stream detection and the harm is realized in real time. |
| "What enforcement actions are available?" | Not just remove/keep. Remove, **demote (reduce distribution)**, age-gate, label/interstitial, restrict the account, escalate to legal. A graduated ladder lets you act on borderline content without binary removal. |
| "Proactive (scan before/at upload) or reactive (on report)?" | Proactive is the goal — catch harm before it's seen — but it can't be perfect, so user reports remain a recall safety net. The metric is how much you action *before* exposure. |
| "What's the review capacity?" | The binding constraint. With ~1B uploads/day and tens of thousands of reviewers, humans touch a tiny fraction, so triage (what to send them) is the core optimization. |
| "Fairness/regulatory scope (languages, regions, EU DSA)?" | Enforcement parity across languages/regions is both an ethical and legal requirement; low-resource languages are where models are weakest and harm hides. |

**Numbers I'll commit to and carry through:**

- **Scale:** ~2B users, ~1B pieces of content/day (posts, comments, images, video).
- **Review capacity:** tens of thousands of reviewers → on the order of a few million reviews/day, i.e., **< 1% of content** can be human-reviewed.
- **Latency:** known-bad **hash match inline, pre-publish, < 100ms** (must block before it's seen). ML classifiers score **async within seconds** of upload — the race is against virality, not a strict request SLO.
- **North-star metric:** **prevalence** — violating views per 10,000 views — driven *down*, measured on a random impression sample.
- **Guardrail:** false-positive rate on benign content (over-enforcement destroys the product and free expression) + enforcement parity across languages.

### The latency-vs-reach race, derived out loud

There's no single user-facing SLO; the real clock is **distribution speed**:

```
Pre-publish (blocking): perceptual-hash match vs known-bad DB    < 100 ms
                        (CSAM/terror: must never be shown -> inline)
At-upload (async):      multimodal classifiers score             ~1-5 s
In-distribution:        re-score as a post gains velocity; a borderline
                        post going viral gets re-evaluated + prioritized
                        for review BEFORE it reaches millions
```

The senior insight: **action value = severity × reach-prevented.** Catching a violating post after 10 views is nearly worthless; catching it before it goes viral is the whole game. So the system races virality, not a fixed latency budget.

### Scale / storage note

1B items/day × multimodal embeddings + decisions (~10KB with image/video features) = **~10 TB/day**, plus the hash database of known-bad content and an immutable enforcement+appeal audit log (legally required, retained long). Video dominates storage and compute (frame sampling, transcription).

---

## 3. Frame as an ML problem

- **Framing:** per-policy multi-label classification with severity, feeding a **triage/prioritization** system that allocates automated action vs scarce human review to minimize expected harmful exposure.
- **The target:** "violates policy class C at severity S" — assigned by trained reviewers against a written policy. The honest complication: this label is **contested** — reviewers disagree, policies are ambiguous and evolve, and context (satire, news, counter-speech) flips the label. Ground truth is a *negotiated, noisy* signal, not a fact of nature.
- **Why this framing wins:** it forces the two senior questions — (1) how do I measure success when I can't review everything (→ prevalence on sampled impressions), and (2) how do I get trustworthy labels when humans disagree (→ policy taxonomy, reviewer calibration, adjudication).
- **Non-ML baseline:** keyword blocklists + perceptual-hash matching of known-bad media + user-report queues. Catches known/exact violations and obvious slurs; misses context, novelty, and evasion. Never removed — hashing is the precise floor for the worst content.

---

## 4. Data and labels — contested ground truth, head-on

The defining label problem (different from fraud's *censored* labels): here labels are **subjective and contested**.

- **Policies are ambiguous and evolving.** "Hate speech" has edge cases; satire, reclaimed slurs, news reporting, and counter-speech all complicate it. The same image is a violation in one context and journalism in another.
- **Reviewers disagree** (low inter-annotator agreement on borderline content). If humans only agree 70% of the time, no model can exceed that ceiling, and "model accuracy" is meaningless without first measuring **label reliability**.

Mitigations to name:
- **A precise policy taxonomy with severity tiers and examples** — the spec the model is really learning. Investing in the policy/labeling guidelines often beats investing in the model.
- **Multi-reviewer labeling with adjudication** for borderline cases; measure inter-annotator agreement (Cohen's/Fleiss' kappa) and treat low-agreement classes as inherently hard, with wider review bands.
- **A golden set of expert-adjudicated examples** as the stable benchmark, refreshed as policy changes (policy drift invalidates old labels).
- **Provenance:** keep label version tied to policy version, so when the policy changes you know which labels went stale.

### Label sources, increasing cost/quality

1. **Hash matches / known-bad lists** — exact, high-precision, for the worst content (CSAM via PhotoDNA-style perceptual hashing, terror content via shared industry hash databases).
2. **User reports** — high recall on what users *notice*, but biased (brigading, false reports, misses harm in private/niche communities) and noisy.
3. **Reviewer decisions** — the core labels, contested as above.
4. **Appeals overturns** — a *quality signal on the system itself*: a high overturn rate means over-enforcement (your precision is worse than you thought).

### Features and the evasion problem

- **Multimodal:** text transformer embeddings; image model; sampled video frames + OCR (text baked into images to evade text filters) + audio transcript; plus metadata (poster history, account age, virality, network).
- **Adversarial evasion:** leetspeak, coded language ("dog whistles"), text-in-image, slight media perturbations to dodge hashes, breaking content across posts. This makes moderation adversarial like fraud — perceptual hashing (robust to crops/recompression) and continual retraining on new evasions are required.
- **Train-serve skew & context:** the same content needs the same features online and offline; and crucially, **context features** (caption + image together, thread context) matter — judging a comment without its parent flips labels.

---

## 5. Baseline -> why it breaks -> next rung

| Rung | What it is | Why it breaks -> trigger to climb |
|---|---|---|
| 0 | Keyword blocklists + perceptual-hash match of known-bad + report queue. | No context, trivially evaded (leetspeak, new media), no novel-harm detection. Trigger: evasion and novel content slip through. |
| 1 | Per-policy text/image classifiers + rules, fixed thresholds, report-driven review. | Ignores severity × reach (treats a viral threat like a dead post), misses multimodal/context, can't prioritize finite review. Trigger: harmful viral content actioned too late; review capacity wasted on low-reach items. |
| 2 | **Multimodal per-policy classifiers with severity + a triage layer that prioritizes by expected harm (P × severity × reach), graduated enforcement, human-in-the-loop.** | Production default. Trigger to extend: livestream, cross-post coordination, low-resource languages. |
| 3 | Real-time/in-stream detection, cross-modal foundation models, coordinated-behavior graph detection, LLM-as-classifier for nuanced policy reasoning. | Heavy infra, calibration, and explainability burden. Trigger: measured gaps rung 2 can't close (livestream harm, organized campaigns, subtle policy nuance). |

Earn rung 2 by explaining the severity×reach triage — that's the core idea. LLM-as-classifier (rung 3) is genuinely useful for *nuanced, contextual* policy calls (it can reason about satire vs hate) but is too expensive to run on all 1B items, so it's reserved for borderline/high-reach triage, not the firehose.

---

## 6. The architecture, explained to the floor

```
   Upload
     |
   [Pre-publish, inline <100ms]
   Perceptual-hash match vs known-bad DB (CSAM/terror)
     |  hit -> BLOCK before publish + legal escalation
     |  miss
   Publish (with limited initial distribution for new/low-trust accounts)
     |
   [Async, ~seconds] Multimodal classifiers per policy:
     text transformer | image model | video frames+OCR | audio transcript
     -> per-policy violation prob + severity, fused with poster/context features
     |
   TRIAGE: expected harm = P(violation) x severity x predicted reach
     |
   +----------------+------------------+-------------------+
   high conf + high  | uncertain OR     | low conf /        |
   severity          | high-reach       | low reach         |
   -> auto-enforce   | -> HUMAN REVIEW  | -> allow + sample-monitor
                       (capacity spent
                        where harm is highest)
     |
   GRADUATED ENFORCEMENT:
     allow | label/interstitial | age-gate | DEMOTE (reduce distribution)
     | remove | restrict/ban account | escalate to legal
     |
   Reason codes + immutable audit + APPEAL path
     |
   Outcomes (reviewer verdicts, appeals overturns, prevalence samples) -> labels
```

### Why each piece exists

- **Perceptual hash matching first:** for the worst, known content you don't want a probabilistic model — you want an exact/near-exact match that blocks pre-publish with ~100% precision. Perceptual (not cryptographic) hashing survives crops, recompression, minor edits. Industry-shared hash databases (e.g., for CSAM/terror) extend coverage. This is non-negotiable and legally mandated for some classes.
- **Multimodal per-policy classifiers:** different harms need different signals and thresholds, and ownership is cleaner per policy. Multimodal because evasion moves harm into images/video/audio; OCR and transcription pull text out of media. Each emits a **calibrated** probability (so the triage math is meaningful) and a **severity**.
- **The triage layer (the heart):** with <1% reviewable, you cannot review by probability alone. You rank by **expected harm = P(violation) × severity × predicted reach.** A borderline post about to go viral outranks a clearly-violating post with 3 views. This is where finite human capacity gets spent, and articulating it is the senior signal. (Predicted reach borrows the feed/virality models from case 02.)
- **Graduated enforcement:** binary remove/keep is too blunt. **Demotion** (reduce distribution) lets you handle borderline or "awful but lawful" content by limiting reach without censoring — the same borderline-demotion idea from the feed-integrity case, here as a first-class enforcement tier. Labels/interstitials add context for misinformation. The action scales with confidence × severity.
- **Human-in-the-loop + appeals:** humans handle the uncertain and high-stakes; their verdicts retrain the models; appeals overturns measure over-enforcement and feed back as precision corrections.

### LLM-as-classifier (rung 3, used surgically)

A strong LLM can reason about context — satire vs hate, news vs glorification — far better than a small classifier, and can cite the specific policy clause. But at $/call it can't touch 1B items. So it's deployed on the **borderline/high-reach slice the triage layer surfaces**, as a smarter reviewer-assist or pre-review filter, not on the firehose. (Its own outputs are evaluated via the case-06 eval platform, since judging policy compliance is exactly that problem.)

### Canonical references (verified)

- Perceptual hashing for known-bad media (PhotoDNA overview): https://www.microsoft.com/en-us/photodnacloudservice
- CLIP (image-text multimodal embeddings) — Radford et al., 2021: https://arxiv.org/abs/2103.00020
- Whisper (robust audio transcription) — Radford et al., 2022: https://arxiv.org/abs/2212.04356
- Inter-annotator agreement / kappa: https://en.wikipedia.org/wiki/Fleiss%27_kappa
- Meta on prevalence as the key metric (Community Standards Enforcement Report): https://transparency.fb.com/reports/community-standards-enforcement/
- EU Digital Services Act overview: https://commission.europa.eu/strategy-and-policy/priorities-2019-2024/europe-fit-digital-age/digital-services-act_en

---

## 7. Evaluation — prevalence, not flagged-precision

This is the section that separates levels. The naive metric is precision/recall on flagged content. The *right* metric is **prevalence**.

- **Prevalence (north star):** of all content **views**, what fraction are of violating content? Measured by drawing a **random sample of impressions** and having experts label them. Why this and not flagged-precision: flagged content is a biased sample (it's what the model already caught), so metrics on it tell you nothing about what you're *missing* in what people actually see. Prevalence on sampled views is the unbiased truth, and it's view-weighted, so it correctly counts harm by exposure.
- **Severity-weighted recall:** not all misses are equal — missing a credible threat ≫ missing mild spam. Weight recall by severity tier.
- **Precision / appeals-overturn rate:** the over-enforcement signal. A rising overturn rate means you're removing benign content — a free-expression and trust failure. Track per policy and per language.
- **Proactive rate:** fraction of violating content actioned *before* a user reports it (and before significant views). High proactive rate = catching harm early.
- **Per-class calibration:** the triage math (P × severity × reach) needs calibrated P, so monitor reliability per policy.
- **Fairness:** enforcement-error parity across languages and regions; low-resource languages are where models are weakest and harm concentrates.

### The offline-to-online gap, including the classic trap

**"Offline classifier precision/recall improved, but prevalence didn't move."** Causes, ordered:

1. **Biased eval set (the signature cause).** You evaluated on flagged/reported content, which over-represents what you already catch. Prevalence (random impressions) didn't move because the new model is better at the easy stuff and still misses the viral/novel content driving actual exposure.
2. **Reach-blind optimization.** You improved recall on low-reach content (lots of items, few views) while the high-reach violations slipped — prevalence is view-weighted, so item-level recall can rise while exposure stays flat.
3. **Adversarial shift.** Bad actors evolved evasions between train and serve; offline (historical) looks better than live.
4. **Label drift.** Policy changed; old labels are stale, so offline gains are measured against an outdated target.
5. **Label-reliability ceiling.** Reviewers only agree ~X%; offline "gains" inside the disagreement band are noise, not real improvement.

### One fully specified A/B test

- **Hypothesis:** the multimodal+triage model (vs per-policy classifiers) reduces prevalence without raising the appeals-overturn rate.
- **Unit:** randomize by **viewer** (or by content cohort) — but note moderation has **network/exposure interference** (removing content changes what everyone sees), so a content-level holdout (don't enforce on a random sample of content, measure resulting exposure) is often cleaner for measuring true recall.
- **Arms:** control = current enforcement; treatment = new model + triage.
- **Primary:** prevalence (violating views per 10K), measured on sampled impressions.
- **Guardrails (auto-stop):** appeals-overturn rate (over-enforcement), benign-content false-positive rate, per-language parity, time-to-action on high-severity, reviewer queue health.
- **Ramp:** 1 → 5 → 25 → 50%, guardrail check each step; **high-severity classes ramp most conservatively** (or not via A/B at all — you don't run an experiment that knowingly leaves CSAM up).
- **Runtime:** long enough for prevalence sampling to reach significance and to span virality cycles.
- **Rollback:** any guardrail breach, especially an overturn-rate spike.

### Error analysis ritual

Review false negatives that went viral (highest harm) and false positives that were overturned (over-enforcement). Maintain banks of both, segmented by policy and language. Reviewer disagreement on the bank tells you which policies need clearer guidelines, not a better model.

---

## 8. Deployment — three paths

- **Serving path:** pre-publish hash match (inline) → async multimodal scoring → triage → enforcement → audit + appeal. New/low-trust accounts get limited initial distribution so the async scorer has time to act before wide reach.
- **Data path:** multimodal feature/embedding pipelines (shared offline/online to avoid skew), the known-bad hash DB (continuously updated from confirmed cases + industry sharing), reach/virality signals, and the immutable enforcement+appeal log.
- **Feedback path:** reviewer verdicts and appeals overturns retrain classifiers and recalibrate thresholds; newly confirmed bad media is hashed and added to the pre-publish DB; new evasions feed adversarial retraining; prevalence samples track whether any of it actually reduced exposure.

### Rollout discipline

Shadow new models (score, no enforcement) and compare against current on a *prevalence* sample, not just flagged precision. Canary by traffic/content %, ramp with guardrails, high-severity classes most conservatively. Policy changes are themselves "releases" — when policy shifts, re-label the golden set and re-baseline metrics.

### Monitoring and fallback

- **What pages someone:** prevalence spike (harm getting through — possibly a coordinated campaign or a new evasion), appeals-overturn spike (over-enforcement incident), high-severity time-to-action breach, hash-DB or feature-pipeline failure, per-language parity breach, review-queue overflow, **reviewer-wellness** signals (exposure to graphic content is a real human cost — rotate, support, and minimize unnecessary human exposure to the worst content).
- **Fallback ladder:** under model failure, **fall back to hashes + rules + report-driven review** (the floor), and for high-severity classes **fail safe toward more restriction** (when uncertain about CSAM/terror, hold/escalate rather than allow). For lower-severity classes, fail toward allowing + sampling to avoid mass over-enforcement. The fail-safe direction *depends on severity* — that nuance is senior.
- **Incident response:** a prevalence spike is often an *attack/campaign*, so treat like fraud — tighten rules/thresholds immediately as a stopgap (faster than retraining), hash the new bad media, identify the evasion, retrain. Preserve audit trails for regulators.

---

## 9. Junior vs senior contrast

| Dimension | Junior answer | Senior answer |
|---|---|---|
| Goal | "Classify and remove bad posts." | Minimize harmful *exposure* (prevalence) under finite review capacity. |
| Metric | "Precision/recall on flagged content." | Prevalence on *sampled impressions*; flagged-precision is a biased sample. |
| Prioritization | "Threshold the classifier." | Triage by expected harm = P × severity × reach; race virality. |
| Labels | "We have moderation labels." | Labels are contested; measure inter-annotator agreement; invest in policy taxonomy. |
| Enforcement | "Remove or keep." | Graduated ladder incl. demotion (reduce distribution) for borderline content. |
| Worst content | "The classifier handles it." | Perceptual hash, pre-publish, ~100% precision, legal escalation, zero-tolerance. |
| Modality/evasion | Text only. | Multimodal + OCR + transcript; perceptual hashing robust to evasion; adversarial retraining. |
| Eval gap | "Offline predicts online." | Biased eval set + reach-blind recall explain why prevalence didn't move. |
| Fallback | "Roll back." | Fail-safe direction depends on severity; hashes/rules floor; reviewer wellness. |
| Cost/nuance | "Use an LLM on everything." | LLM-as-classifier only on the borderline/high-reach slice; too costly for the firehose. |

---

## 10. One-page whiteboard cheat sheet

```
NUMBERS: 2B users | 1B items/day | <1% human-reviewable
         hash match inline <100ms | classifiers async ~secs | RACE VIRALITY

BIG IDEA: minimize harmful EXPOSURE (prevalence) under finite review.
  harm ~ severity x reach. catch BEFORE viral, not after.
  ground truth is CONTESTED (reviewers disagree -> measure kappa).

LADDER: blocklist+hash -> per-policy classifiers
        -> MULTIMODAL + SEVERITY + TRIAGE(PxSevxReach) + graduated enforce + HITL
        -> in-stream/livestream, coordinated-graph, LLM-as-classifier (borderline only)

PIPELINE: hash-match pre-publish (CSAM/terror, ~100% prec, legal)
          -> multimodal classifiers (text/image/video+OCR/audio) calibrated P+severity
          -> TRIAGE expected harm -> auto-enforce / HUMAN REVIEW / allow+sample
          -> enforce ladder: label/age-gate/DEMOTE/remove/ban/legal
          -> reason codes + audit + APPEAL

EVAL: PREVALENCE (random IMPRESSION sample) = north star  [not flagged-precision!]
      severity-weighted recall | appeals-overturn = over-enforcement
      proactive rate | per-language fairness | per-class calibration

OFFLINE-UP/PREVALENCE-FLAT: biased eval set (flagged != viewed)
      | reach-blind recall | adversarial shift | policy/label drift | kappa ceiling

A/B: content-level holdout (network interference!) | primary=prevalence
     guard=overturn rate, benign FP, per-lang parity, high-sev time-to-action
     high-severity ramps most conservatively (never "experiment" w/ CSAM)

DEPLOY: serving/data(hash DB+multimodal)/feedback(verdicts+appeals+new hashes)
        FALLBACK direction depends on SEVERITY (high-sev fail to restrict)
        prevalence spike = likely campaign -> tighten rules now, hash, retrain
        REVIEWER WELLNESS is a real constraint
```

---

## 11. Full interview transcript (one hour)

**[00:00] INTERVIEWER:** Design a system to detect and enforce policy violations in user-generated content.

**[00:30] YOU:** Clarifying questions first. Which harms, and is CSAM or terrorism in scope, since those change the design? What content types — text, image, video, livestream? What enforcement actions can I take beyond remove? Proactive scanning or reactive on reports? And what's the human review capacity?

**[01:00] INTERVIEWER:** All harms, including CSAM. Text, image, and video. You can remove, demote, age-gate, or restrict accounts. We want proactive detection. Review capacity is tens of thousands of people.

**[01:20] YOU:** Then let me frame the real problem, because it's not "classify bad posts." With a billion uploads a day and capacity to review under one percent, the goal is to minimize harmful *exposure* — prevalence — under a tight review budget. Harm is roughly severity times reach: catching a violating post after ten views is almost worthless; catching it before it goes viral is the whole game. And the ground truth is contested — reviewers disagree on borderline content, so I have to measure label reliability before I can even talk about model accuracy. Those three things — prevalence, severity-times-reach triage, and contested labels — drive the whole design.

**[02:50] INTERVIEWER:** Start with the most severe content — CSAM. How do you handle it?

**[03:00] YOU:** Not with a probabilistic classifier as the front line. For known CSAM and terror content I use perceptual hash matching against a known-bad database — including industry-shared hash sets — and I block it pre-publish, inline under 100ms, before it's ever shown. Perceptual hashing, not cryptographic, so it survives crops, recompression, and minor edits. It's near-100% precision, it's legally mandated, and it's zero-tolerance. The ML models are for *novel* violating content the hashes don't know yet, and confirmed new cases get hashed and added to the pre-publish DB.

**[04:10] INTERVIEWER:** For the novel content, what's the model architecture?

**[04:20] YOU:** Multimodal per-policy classifiers. Text transformer embeddings, an image model, sampled video frames with OCR to pull text baked into images, and audio transcription — because evasion moves harm into media. Each policy class gets its own calibrated classifier emitting a violation probability and a severity, fused with poster history, context, and virality features. Calibration matters because of the next piece — the triage layer. I can't review by probability alone with under one percent capacity, so I rank by expected harm: probability of violation times severity times predicted reach. A borderline post about to go viral outranks a clearly-violating post with three views. That's where I spend human reviewers.

**[05:50] INTERVIEWER:** Why per-policy classifiers instead of one big model?

**[05:40] YOU:** Cleaner ownership and per-class thresholds — hate speech and spam have very different severity and tolerance for false positives — and different classes need different signal weighting. I'd share the multimodal backbone embeddings across them for efficiency, but keep per-policy heads and thresholds. For genuinely nuanced calls — satire versus hate, news versus glorification — I'd use an LLM-as-classifier that can reason about context and cite the policy clause, but only on the borderline, high-reach slice the triage surfaces, never on the full billion-item firehose, because it's too expensive.

**[07:00] INTERVIEWER:** You keep saying enforcement isn't just remove. Explain.

**[07:10] YOU:** Binary remove-or-keep is too blunt and it forces a bad precision/recall trade. I want a graduated ladder: allow, add a label or interstitial for misinformation, age-gate, *demote* — reduce distribution without removing — remove, restrict or ban the account, and escalate to legal. Demotion is the key middle tier: for borderline or "awful but lawful" content, I can limit reach without censoring, which both reduces harm and protects free expression. The action scales with confidence times severity. High-confidence, high-severity gets removed automatically; borderline gets demoted and queued for review.

**[08:30] INTERVIEWER:** How do you know if any of this is working?

**[08:40] YOU:** Prevalence, measured the right way. The naive metric is precision and recall on flagged content — but flagged content is a biased sample of what the model already catches, so it tells me nothing about what I'm missing in what people actually see. The correct metric is prevalence: draw a *random sample of impressions* — actual views — and have experts label them, then measure violating views per ten thousand. It's unbiased and it's view-weighted, so it counts harm by exposure. Alongside it: severity-weighted recall, because missing a credible threat is not the same as missing spam; appeals-overturn rate as my over-enforcement signal; proactive rate — how much I catch before a user reports it; and per-language fairness, because low-resource languages are where my models are weakest and harm hides.

**[10:10] INTERVIEWER:** Suppose offline precision and recall improve, you ship, and prevalence doesn't move. Why?

**[10:20] YOU:** The signature cause: I evaluated on flagged or reported content, which over-represents what I already catch, so the new model got better at easy stuff while still missing the viral, novel content that actually drives exposure. Second, reach-blind optimization — I improved recall on lots of low-reach items but the high-reach violations slipped, and prevalence is view-weighted, so item-level recall rose while exposure stayed flat. Third, adversarial shift — evasions evolved between training and serving. Fourth, policy drift — the policy changed and my offline labels are stale. Fifth, the label-reliability ceiling — if reviewers only agree 70% of the time, gains inside that disagreement band are just noise.

**[11:50] INTERVIEWER:** Tell me about those contested labels. How do you get trustworthy ground truth?

**[12:00] YOU:** I treat the labeling system as seriously as the model. A precise policy taxonomy with severity tiers and worked examples — that spec is what the model is really learning, and improving the guidelines often beats improving the model. Multi-reviewer labeling with adjudication for borderline cases, and I measure inter-annotator agreement with kappa per class; low-agreement classes are inherently hard and get wider review bands rather than aggressive auto-enforcement. A golden set of expert-adjudicated examples as the stable benchmark, versioned to the policy version so I know which labels go stale when policy changes. Without measuring label reliability first, "model accuracy" is meaningless.

**[13:40] INTERVIEWER:** Design the A/B test.

**[13:50] YOU:** Primary metric is prevalence on sampled impressions. One subtlety: moderation has exposure interference — removing content changes what everyone sees — so rather than a pure viewer split, a clean design is a content-level holdout: don't enforce on a small random sample of content and measure the resulting true exposure, which also gives me an unbiased recall estimate. Control is current enforcement, treatment is the new model plus triage. Guardrails that auto-stop: appeals-overturn rate, benign false-positive rate, per-language parity, and high-severity time-to-action. I ramp 1, 5, 25, 50 — but high-severity classes ramp most conservatively, and honestly I would not run an experiment that knowingly leaves CSAM up, so the worst classes aren't A/B'd at all; they're always max enforcement. Rollback on any guardrail breach, especially an overturn spike.

**[15:30] INTERVIEWER:** Prevalence suddenly spikes in production. What do you do?

**[15:40] YOU:** I treat it like an attack or coordinated campaign, similar to fraud. Rules and thresholds are my fast lever — faster than retraining — so I tighten them around the affected segment immediately as a stopgap, hash any new bad media into the pre-publish DB, identify the evasion technique, and then retrain on it. I preserve the audit trail for regulators. The reflex is the same as fraud: a fast, auditable stopgap that doesn't require a model deploy, then a durable fix.

**[16:50] INTERVIEWER:** When your model is uncertain, do you fail toward removing or allowing?

**[17:00] YOU:** It depends on severity — and naming that dependency is the point. For high-severity classes like CSAM or credible threats, I fail safe toward restriction: when uncertain, hold or escalate rather than allow, because the cost of exposure is catastrophic. For lower-severity classes, I fail toward allowing plus sampling, because failing toward removal there would mean mass over-enforcement that destroys the product and free expression. A single global "block when unsure" policy would be wrong; the fail-safe direction is severity-dependent.

**[18:10] INTERVIEWER:** Anything about the humans in this system?

**[18:20] YOU:** Reviewer wellness is a real constraint, not a footnote. Reviewers exposed to graphic content suffer real harm, so I minimize unnecessary human exposure to the worst material — let hashing handle confirmed CSAM so humans don't re-view it, blur/grayscale tooling, rotation, and support. It's also a system-design lever: the triage that sends humans only the highest expected-harm, genuinely-ambiguous cases both spends capacity well and reduces needless exposure.

**[19:00] YOU:** I'd be honest that I haven't worked directly in trust-and-safety, so I'd connect this to what I have done — evaluation, fairness, and monitoring systems. The prevalence-versus-flagged-precision distinction is exactly the biased-eval-set problem I've dealt with in eval platforms: measuring on the sample your model already catches flatters you and hides what you miss. And measuring inter-annotator agreement before trusting labels, plus per-cohort fairness monitoring, is the same discipline. So I lead with measurement integrity and triage rather than claiming moderation domain expertise I don't have.

**[19:50] INTERVIEWER:** That's a thoughtful answer.

### Why this transcript works

- **Reframes the goal as minimizing exposure (prevalence)**, not classifying posts.
- **Handles the worst content with hashing, pre-publish**, distinct from probabilistic models.
- **Centers the severity×reach triage** as how finite review capacity is spent.
- **Treats labels as contested**, measuring inter-annotator agreement and investing in the policy taxonomy.
- **Uses graduated enforcement** including demotion, not binary remove/keep.
- **Nails the prevalence-vs-flagged-precision trap** in the offline/online gap.
- **Makes the fail-safe direction severity-dependent** — a genuinely senior nuance.
- **Raises reviewer wellness** as a real constraint, and **honestly scopes the candidate's experience** to eval/fairness/monitoring as the prompt requires.

---

## 12. Follow-up questions the interviewer may ask

- **What changes at 10x scale / livestream?** Livestream can't be fully pre-reviewed, so you need fast in-stream detection (sampled frames + transcript in near-real-time), aggressive reach-limiting for new streams, and rapid human escalation; harm is realized live, so latency matters more there than for static posts.
- **How do you handle a brand-new harm type (cold start)?** No labels, so start with reports + anomaly/novelty detection + human review to bootstrap a labeled set and a policy, then train a classifier as labels accrue (mirrors fraud's anomaly layer).
- **How do you set thresholds?** Per policy from severity and the over-enforcement budget (overturn-rate ceiling), constrained by review capacity; high-severity gets recall-favoring thresholds, low-severity precision-favoring.
- **Offline up, prevalence flat — what do you check?** Biased eval set, reach-blind recall, adversarial shift, policy/label drift, label-reliability ceiling. (Section 7.)
- **How do you debug?** Slice prevalence by policy/language/reach, inspect viral false negatives and overturned false positives, check feature/hash-pipeline freshness, and check whether reviewer disagreement (not the model) is the bottleneck.
- **How do you handle appeals and over-enforcement?** Appeals overturns are a first-class precision signal fed back to retraining and threshold tuning; a rising overturn rate triggers a precision review per policy/language.
- **Fairness across languages?** Monitor enforcement-error parity; invest labeling and model capacity in low-resource languages where harm hides and models are weakest; avoid English-centric thresholds.

---

## 13. Common mistakes

- Framing it as **"classify each post"** instead of minimizing exposure (prevalence) under finite review.
- Measuring **flagged-precision** instead of **prevalence on sampled impressions**.
- Ignoring **severity × reach** — wasting review on low-reach content and missing viral harm.
- Treating labels as **objective** when reviewers disagree; skipping inter-annotator agreement and the policy taxonomy.
- Binary **remove/keep** with no demotion or graduated ladder.
- Using a probabilistic classifier (not **hashing**) as the front line for the worst content.
- **Text-only**, ignoring multimodal evasion (text-in-image, audio).
- A single global **fail-safe direction** instead of severity-dependent.
- Ignoring **reviewer wellness** and the human cost of the queue.

---

## 14. Transfer — what this case unlocks

- **02 News Feed Ranking:** borderline-content demotion and integrity signals are the same idea; predicted-reach models come from there.
- **13 LLM Safety Gateway:** input/output guardrails for *generative* models share the policy-enforcement structure; that case owns jailbreaks/prompt-injection, this one owns UGC at platform scale.
- **09 Fraud Detection:** adversarial dynamics, the rules-as-fast-lever incident response, and cost-asymmetric thresholds transfer directly.
- **06 LLM Eval & Monitoring:** the biased-eval-set lesson, inter-annotator agreement, and LLM-as-classifier evaluation are shared.
- **17 Spam / Bot Detection:** coordinated-behavior and graph detection overlap for campaign-style abuse.
- **11 Experimentation:** content-level holdouts and interference-aware A/B design come from the experimentation backbone.

---

## 15. Sources

Originals (kept):
- [IGotAnOffer ML System Design Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [Exponent ML System Design Interview Guide](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)
- [Hello Interview ML/System Design Learning](https://www.hellointerview.com/learn)
- [Designing Machine Learning Systems, Chip Huyen](https://huyenchip.com/machine-learning-systems-design/toc.html)
- [OpenAI Agents SDK Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

Added (verified canonical):
- [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Microsoft PhotoDNA (perceptual hashing)](https://www.microsoft.com/en-us/photodnacloudservice)
- [Fleiss' kappa (inter-annotator agreement)](https://en.wikipedia.org/wiki/Fleiss%27_kappa)
- [Meta Community Standards Enforcement Report (prevalence)](https://transparency.fb.com/reports/community-standards-enforcement/)
- [EU Digital Services Act](https://commission.europa.eu/strategy-and-policy/priorities-2019-2024/europe-fit-digital-age/digital-services-act_en)
