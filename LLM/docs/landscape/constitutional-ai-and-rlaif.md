# Constitutional AI and RLAIF

Anthropic's alignment recipe (CAI) is a specific flavor of the broader pattern: **replace human preference labelers with an AI judge.** That broader pattern is RLAIF. The humans shift from labeler (bottleneck, expensive, inconsistent) to legislator (write the rules once, scale infinitely). Every major lab uses some AI-feedback step in 2026.

!!! tip "Rapid Recall"
    **Constitutional AI** writes down a constitution (principles like "be helpful, do not deceive, do not help with illegal activities") and uses the model to critique and revise its own responses against that constitution. **SL-CAI** (supervised): prompt → harmful response → AI self-critique against a randomly selected principle → AI revision → SFT on revised pairs. **RL-CAI**: generate two responses, AI judges which is better per constitution, train RM on AI preferences, run PPO/DPO. **RLAIF** is the general case: replace human preference labelers with a strong AI model. **Tradeoff:** AI judges are ~1000× cheaper than humans and more consistent, but inherit the judge model's biases as systematic blind spots, and have a quality ceiling at the judge's judgment. **Capability floor:** CAI's self-critique requires a model strong enough to detect the harm; weaker models fail at the critique step.

## §1 The simple story

Instead of paying humans to write preference labels for every pair of responses ("A is better than B"), **write down a constitution** — a set of principles like "be helpful, do not help with illegal activities, do not deceive." Then let the model critique and revise its own responses against that constitution. The AI itself generates the preference data.

## §2 Mental model

Traditional RLHF: `human writes principle → human labels preferences → reward model → RL`.

Constitutional AI: `human writes principle → AI self-critiques using principle → AI labels preferences → reward model → RL`.

The humans' role shifts from **labeler** (bottleneck, expensive, inconsistent) to **legislator** (write the rules once, scale infinitely).

## §3 The two phases

### 3.1 Phase 1 — Supervised Constitutional (SL-CAI)

1. Prompt the model with a harmful request, get a harmful response.
2. Ask the model to **critique its own response** against a randomly selected principle from the constitution.
3. Ask the model to **revise its response** to address the critique.
4. Fine-tune on these (prompt, revised-response) pairs with standard SFT.

This phase is just SFT on AI-generated revised data — no RL.

### 3.2 Phase 2 — RL from AI Feedback (RL-CAI)

1. Generate two responses to a prompt from the SL-CAI model.
2. Ask a separate AI judge which is better according to the constitution.
3. Use those AI preferences as labels for a reward model (just like RLHF).
4. Run PPO or DPO (see [RL from zero and PPO](../alignment/rl-from-zero-and-ppo.md), [DPO](../alignment/dpo.md)).

This phase is RLHF, except the preference labels come from an AI judge instead of human labelers.

## §4 The constitution itself

Anthropic's published constitution draws from the UN Declaration of Human Rights, Apple's ToS, DeepMind's Sparrow rules, etc. Example principles:

- "Choose the response that is most helpful, honest, and harmless."
- "Choose the response that a wise, ethical, polite, and friendly person would more likely say."

The constitution is the only human-written artifact in the pipeline. Once written, the AI generates every downstream label.

## §5 Why it works (and sometimes does not)

The 2025 paper "How Effective Is Constitutional AI in Small LLMs?" showed CAI's self-critique works well on models with **strong reasoning capability** (DeepSeek-R1's explicit-thinking made it the best at CAI-style harm detection), but **weaker models fail to detect harm during critique**. This is a **capability floor** — you cannot use CAI on a 1B model to align it; the critic needs to be strong enough to identify issues.

This means CAI scales with model capability: as base models get smarter, the CAI pipeline becomes more effective because the critic is more discerning.

## §6 Why it matters in 2026

Constitutional AI (or variants) is the closest thing to a **scalable, transparent alignment process**. Every major lab now uses some AI-feedback step to reduce human labeling cost.

- **Anthropic's version** is called CAI; the constitution is public.
- **OpenAI's** is essentially RLAIF without a published constitution.
- **DeepMind** uses similar in Gemini.
- **Open-source** projects use simplified CAI variants for community-driven alignment.

Know it as the recipe behind Claude specifically; know RLAIF as the general pattern.

## §7 RLAIF — the general case

Same as RLHF, but replace the human preference labelers with a strong AI model (typically GPT-4-class or better) that does the "which of these two responses is better" ranking. Constitutional AI is a specific flavor of RLAIF.

RLHF's bottleneck is humans. One labeler gets through maybe 500 preference pairs a day, and labelers disagree with each other ~20% of the time. An AI judge can do 500 pairs in a few minutes at 1/1000 the cost, and is more consistent (though potentially more biased).

### 7.1 The tradeoff

| | RLHF | RLAIF |
|---|---|---|
| Cost | $$$ (humans) | $ (API calls) |
| Scale | Slow | Fast |
| Consistency | Variable | High |
| Bias | Human biases | Judge model's biases (inherited) |
| Ceiling | Labelers' judgment | Judge model's judgment |
| Regulatory | Defensible ("humans in the loop") | Harder to defend in high-stakes settings |

### 7.2 When to use which in 2026

- **High-stakes alignment** (safety, harmful content): still mostly RLHF, or hybrid.
- **Preference tuning for style / helpfulness**: RLAIF dominant.
- **Domain-specific tuning**: RLAIF with a domain-expert model as judge.
- **Research settings**: RLAIF for speed and reproducibility.

### 7.3 Limitation you must know

**Reward model collapse:** if the AI judge is itself imperfect, the policy will learn to exploit the judge's biases. Same reward-hacking problem as RLHF but worse because AI judges have more systematic blind spots than human crowds. Mitigations: ensemble judges, rotating judge models, human spot-checking.

## §8 Why this is the scalable alignment story

Human preference labeling does not scale to the frontier. Frontier post-training in 2026 needs millions of preference pairs across thousands of nuanced behaviors (style, helpfulness, refusal correctness, format compliance, tool use). The labor for that with humans is infeasible. RLAIF in some form is the only path that scales, which is why every lab uses it.

The catch: RLAIF inherits the judge model's biases. If your judge model thinks longer answers are better, your policy will learn to write longer answers. Constitutional AI mitigates this by making the bias *legible* (it is in the constitution) rather than implicit (it is in the labelers' heads).

## Interview Questions

**Q1: What is the difference between RLHF, RLAIF, and Constitutional AI?**

RLHF uses humans to label preference pairs, trains a reward model, then PPO-optimizes the policy against it. RLAIF replaces the human labelers with an AI judge — much cheaper and faster, but inherits the judge model's biases. Constitutional AI is Anthropic's specific version of RLAIF: write a constitution of principles once, then use the AI to self-critique and revise its own responses against those principles, then use those revised pairs as preference data. The key progression is humans shifting from **labeler** (bottleneck) to **legislator** (scalable). Every major lab now uses AI feedback in some form because human labeling does not scale.

**Q2: Why does CAI need a strong base model?**

The self-critique step asks the model to identify whether its own response violates a principle. A weak model that does not understand the principle cannot identify the violation; the critique becomes shallow or wrong, and the downstream "revised response" does not actually fix anything. The 2025 paper on small-LLM CAI showed this clearly: strong reasoning models (DeepSeek-R1) excel at the critique; small models (under ~7B) often miss harm. CAI scales with capability.

**Q3: What is the main risk of RLAIF over RLHF?**

Reward model collapse via judge bias. Humans disagree with each other, so the noise is roughly random; the RM averages out and the policy learns the "median preference." AI judges have systematic biases (length, sycophancy, specific phrasings) that all point the same direction; the RM amplifies those biases and the policy exploits them. Mitigations: ensemble multiple judge models, rotate judges over training, human spot-checking, and explicit constitutions that make biases legible. None of these fully eliminate the problem.

**Q4: Why is Constitutional AI considered more transparent than implicit RLHF?**

Because the alignment criteria are written down. With RLHF you cannot easily explain why the model behaves a certain way — labelers had their own biases and those got compiled into the reward model opaquely. With CAI, the constitution is public; the principles guiding behavior are auditable. When the model refuses something or rephrases an answer, you can point to the principle that drove the decision. This is the regulatory and audit story Anthropic emphasizes.

**Q5: When should you still pick RLHF over RLAIF in 2026?**

High-stakes alignment where regulatory defensibility matters — medical, legal, content moderation that might require human-in-the-loop justification. Also for behaviors that AI judges systematically misjudge (e.g., tone calibration in sensitive customer service). Hybrid is common: RLAIF for most preference data, RLHF for the highest-stakes subset where you cannot afford the judge model's blind spots.
