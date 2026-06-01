# Distillation and the Cat-and-Mouse Game

Take a big expensive "teacher" model. Ask it a lot of questions. Train a small cheap "student" model to mimic. The student often punches way above its weight class. This page covers Hinton's classic recipe (logits + temperature), why it died for frontier API models, the modern sequence-level (text-output) playbook that DeepSeek used to distill R1 traces, and the defense layers labs deploy to slow attackers down.

!!! tip "Rapid Recall"
    **Classic distillation** (Hinton 2015): train student on KL between student logits and teacher logits, with temperature softening to amplify "dark knowledge" — the small probabilities. Requires logits. **Sequence-level distillation** (modern, what everyone does): query teacher with millions of prompts, collect text output, train student via standard SFT on those responses. Weaker per-example but compensated by scale. **Why reasoning models made distillation worse for defenders:** a bare answer is ~5 tokens of teacher behavior; a chain-of-thought trace is ~80 tokens that teach the *procedure*. **Defenses (watermarking, trace obfuscation, behavioral detection, refusal hardening, rate limits)** raise cost, not impossibility — buy ~1 year of lead. The structural asymmetry: any defense that preserves utility for legit users also preserves enough signal for distillers.

## §1 Classic distillation (Hinton 2015)

Hard labels say "dog (1.0)." A teacher's soft distribution says "dog 0.85, wolf 0.10, cat 0.04" — the **dark knowledge** of class-similarity structure. Train the student on KL against the teacher's logits, with **temperature** `softmax(logits/T)` to amplify the small probabilities.

\[ L_{\text{distill}} = \alpha \cdot L_{CE}(y_{\text{hard}}, \hat y) + (1-\alpha) \cdot T^2 \cdot L_{KL}\!\left(\text{softmax}(z_t/T), \text{softmax}(z_s/T)\right) \]

Where \(z_t\) and \(z_s\) are teacher and student logits, T is **temperature** (higher T = softer distribution = more information), α balances hard and soft labels. Typical T = 2-5.

**Requires the full logit vector.** This is the part that killed it for frontier APIs.

## §2 Two flavors

| Flavor | What you have | Loss | Used by |
|---|---|---|---|
| **White-box** | Full logits from teacher | Weighted CE on hard label + KL on soft distributions | Internal lab pipelines |
| **Black-box / API** | Only text outputs | Standard SFT on teacher outputs | "What everyone accuses everyone else of doing" |

**This is what everyone accuses everyone else of doing.** OpenAI's Feb 2026 memo to Congress accused DeepSeek of black-box distilling GPT outputs via obfuscated third-party routers. Legally gray, technically trivial.

## §3 Why the classic playbook is dead for frontier APIs

Frontier APIs do not expose logits — text out, period. So attacks shifted to **sequence-level distillation**: query the teacher with millions of prompts, collect the text, train the student on next-token prediction over those responses. The teacher's *output* becomes the hard label. Weaker per-example, but compensated by scale (the Anthropic-DeepSeek case: 16M+ exchanges via ~24,000 fake accounts).

## §4 Why reasoning models made it worse for defenders

!!! abstract "The reasoning-trace problem"
    A bare answer is ~5 tokens of teacher behavior. A chain-of-thought trace is ~80 tokens that teach the **procedure**, not just the answer. Reasoning traces transmit how the teacher thinks — vastly richer distillation signal. This is the R1 flashpoint. The model not only learns *what* the teacher would answer, but *how* it would reason there.

This is why the Constitutional AI summaries Anthropic returns instead of full extended-thinking traces are a meaningful defense even at minor UX cost.

## §5 The DeepSeek-R1 distillation play (important in 2026)

DeepSeek distilled R1's reasoning traces into **six smaller dense models** (1.5B, 7B, 8B, 14B, 32B, 70B) based on Qwen and Llama backbones. These distilled models "surpass the performance of their original instruction-tuned counterparts" per the R1 paper. This is why you see so many "DeepSeek-R1-Distill-Qwen-7B" style model names on HuggingFace — it is a recipe anyone can now reproduce.

Why it matters:

- **Cost.** A distilled 7B model at 99% of teacher quality is 10× cheaper to serve.
- **Latency.** Small models = lower TTFT.
- **Deployment.** Fits on edge / consumer GPUs.
- **Legal concerns.** API TOS typically forbid using outputs to train competing models. Does not stop anyone but introduces legal exposure.

## §6 The real attack pipeline

1. Prompt diversification (cover the capability space).
2. Querying through obfuscation (residential proxies, fake accounts, third-party routers).
3. Output collection and cleaning (strip refusals, disclaimers, watermark signals).
4. Synthetic-data augmentation (launder via intermediate models).
5. Standard SFT + RL on the cleaned set.

The pipeline is not technically novel; it is operationally disciplined. The hard parts are (1) and (4), not the model training.

## §7 The defense layers

| Defense | Mechanism | Weakness |
|---|---|---|
| Watermarking | bias toward "green-list" tokens; detect via z-score | paraphrasing scrambles it |
| Trace obfuscation | summarize reasoning (Anthropic summarizes ~95% of extended-thinking traces) | degrades legit UX too |
| Behavioral detection | flag burst/edge-case query patterns | attackers spread and throttle |
| Refusal hardening | refuse highest-value queries | hurts real users |
| Rate / geo limits | blunt access control | proxies and accounts |
| Coalition intel | Frontier Model Forum signature sharing | reactive |

!!! warning "The structural asymmetry"
    Any defense that preserves utility for legit users **also** preserves enough signal for distillers — the floor is the model's real capability. Defenses raise *cost*, not impossibility; they buy ~1 year of lead. That is why the SOTA-shipping race *is* the moat, and why the policy response (chip export controls) targets the compute distillers need rather than the information leak itself.

## §8 The KL framing

Distillation training is fundamentally **KL divergence minimization** between the student and the teacher. For the choice of which direction:

- **Forward KL** \(D_{KL}(P_t \| P_s)\) — mode-covering, "where the teacher has mass, the student must have mass." Standard distillation choice.
- **Reverse KL** \(D_{KL}(P_s \| P_t)\) — mode-seeking, would collapse the student onto the teacher's argmax and discard "dark knowledge." Wrong for distillation.

The forward KL is exactly the cross-entropy loss with teacher distribution as the target. Same recipe as classification (see [MLE and MAP backbone](../sft/mle-map-backbone.md)), different target distribution.

## §9 Legitimate distillation pipelines

There are entirely above-board distillation use cases.

- **In-house distillation.** Lab uses its own teacher model to train smaller versions. No TOS issues.
- **Open-source teachers.** Distilling Llama, Qwen, DeepSeek (open-weight under permissive licenses) is fine.
- **Licensed teacher access.** Some providers offer paid distillation access (you train against their model under contract).
- **Sequence-level distillation in production.** Distill your own fine-tuned model into a smaller one for inference cost reduction.

The legal gray area is specifically using outputs of *other companies'* APIs to train competing models when their TOS forbids it.

## Interview Questions

**Q1: What is the difference between classic Hinton distillation and modern sequence-level distillation?**

Classic distillation requires logits: train the student to match the teacher's full softmax distribution via KL, with temperature softening to amplify the small probabilities ("dark knowledge"). Modern sequence-level distillation only needs text outputs: query the teacher with millions of prompts, collect responses, train the student via standard SFT on those responses. Sequence-level is weaker per-example (you lose the rich distributional signal) but compensated by scale and is the only option when targeting closed APIs that do not expose logits.

**Q2: Why did reasoning models make distillation worse for defenders?**

A bare answer is ~5 tokens of teacher behavior. A chain-of-thought trace is ~80 tokens that teach the *procedure*, not just the answer. The student learns not only what the teacher says but how it reasons there. Reasoning traces transmit vastly more distillation signal per query. This is exactly the R1 distillation flashpoint, and is why Anthropic summarizes ~95% of extended-thinking traces before returning them — to reduce the trace surface available to attackers, at minor legit UX cost.

**Q3: Explain the DeepSeek-R1 distillation play.**

DeepSeek collected R1's reasoning traces and used them to fine-tune six smaller models (1.5B, 7B, 8B, 14B, 32B, 70B) on Qwen and Llama backbones. The distilled models "surpass the performance of their original instruction-tuned counterparts" — small dense models punching way above their weight class. Operational implications: a 7B distill at ~99% of R1's reasoning quality is 10× cheaper to serve and fits on consumer GPUs. This is why HuggingFace is full of "DeepSeek-R1-Distill-Qwen-7B" style models — the recipe is open and reproducible.

**Q4: Why are defenses against API distillation always temporary?**

The structural asymmetry: any defense that preserves utility for legit users also preserves enough signal for distillers — the floor is the model's real capability. Watermarking gets paraphrased away. Trace obfuscation degrades real UX. Behavioral detection gets spread and throttled. Refusal hardening hurts real users. Defenses raise *cost*, not impossibility; they buy roughly a year of lead. That is why the SOTA-shipping race is the actual moat, and why policy response (chip export controls) targets the compute distillers need rather than the information leak itself.

**Q5 (Trap): If sequence-level distillation works, why train the teacher at all? Just iterate distillation forever.**

Because distillation is bounded by the teacher's capability — you cannot distill a 70B-level model into a 1B and exceed the teacher. To advance the frontier you still need to train better teachers (pretrain + post-train), which is where the original cost lives. Distillation is a cost-amortization trick for serving capability you already developed, not a capability-creation trick. Frontier labs invest heavily in teacher training because the distill economy depends on them having something to distill from.
