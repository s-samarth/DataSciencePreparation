# The GRPO Hype, Decoded

The claim on the timeline: "GRPO is so revolutionary it will train models from scratch — no SFT, no pretraining." Half-true, mostly hype. This page is the calibrated version: what R1-Zero actually proved, what it did NOT prove, and why GRPO does not replace any stage of the 2026 pipeline.

!!! tip "Rapid Recall"
    R1-Zero proved that **pure RL via GRPO can elicit reasoning from a strong base model without SFT** — chain-of-thought, self-verification, and reflection emerged from RL alone because they earned reward. AIME 2024 jumped from ~40% to 71%. But R1-Zero is **not deployable** (endless repetition, language mixing, poor readability), and shipped R1 reintroduces SFT — its production pipeline is **2 SFT stages + 2 RL stages**. GRPO cannot replace pretraining (RL needs a reasonable starting policy, sparse rewards do not create knowledge), cannot replace SFT in practice (readability collapse, no instruction-following on cold start), and cannot replace RLHF for subjective tasks (no verifier for tone). **The real revolution:** RL became *capability-eliciting*, not just behavior-polishing. Long-form reasoning was in no training set; the model invented it because it earned reward. Post-training shifted from a refinement phase into a capability-generation phase.

## §1 What R1-Zero actually proved

DeepSeek applied RL *directly to the base model* with no SFT — the first open validation that reasoning can be incentivized purely through RL. The model discovered chain-of-thought, self-verification, and reflection on its own. AIME 2024 jumped from ~40% (base V3) to 71% (R1-Zero).

This is genuinely the most important post-training advance of 2024-2025. RL was previously seen as "polish" applied after SFT; R1-Zero proved RL can *create new capabilities*.

## §2 What R1-Zero did NOT prove

Several claims that bubbled up in the wake of the paper are not what the paper actually showed.

- **It did not skip pretraining.** R1-Zero started from DeepSeek-V3-Base — a ~671B MoE trained on 14.8T tokens. All knowledge came from pretraining; GRPO only sculpted what was already latent.
- **It was not deployable.** R1-Zero had endless repetition, poor readability, language mixing. The shipped R1 reintroduced SFT — its production pipeline is **2 SFT stages + 2 RL stages**.
- **It only worked because the base was excellent.** "If the base is strong enough, RL can elicit reasoning without SFT" is conditional, not universal. Try the same recipe on a weak 7B base and you do not get the same emergence.

## §3 Why pure GRPO cannot replace SFT (in practice)

- **Cold-start.** Base model cannot follow instructions, so nothing it produces is verifiable. RL needs at least some structure to grade.
- **Format and readability collapse.** Without SFT teaching `<think>...</think><answer>...</answer>` (or equivalent), the model invents idiosyncratic formats that pass verifier but humans cannot read.
- **Mode collapse under sparse reward.** Binary rewards on a base model often collapse to one or two stereotyped response patterns.
- **Verifiable domains only.** RLVR has no signal on creative writing, dialogue style, or subjective preferences.
- **50 to 100× costlier per example.** RL rollouts are expensive vs SFT's single forward-backward per example.

## §4 Why GRPO cannot replace pretraining (fundamentally)

- **RL needs a reasonable starting policy.** A randomly initialized network produces only gibberish, which means no rewards, which means no gradient signal.
- **RL shapes behavior; it does not create knowledge.** You cannot RL your way to "France's capital is Paris" — that fact has to come from pretraining data.
- **The compute math is infeasible.** Pretraining's dense next-token loss gives a gradient on every token; RL rewards are sparse, rollouts expensive. To accumulate the same gradient signal you would burn orders of magnitude more compute.
- **There is no objective to optimize "France's capital is Paris" against.** Factual recall is not a verifiable reward in the RLVR sense.

## §5 The calibrated table

| Claim | Verdict |
|---|---|
| GRPO can replace SFT entirely | **Partially true** — possible on strong bases; production still uses SFT |
| GRPO can replace pretraining | **False** — RL operates on a pretrained substrate |
| GRPO can replace RLHF | **Domain-dependent** — yes for verifiable, no for open-ended |
| Most important post-training advance of 2024-2025 | **True** |
| Frontier labs abandoned SFT | **False** — every shipped model uses it, R1 included |

## §6 The real revolution

!!! abstract "The actual shift"
    Not "GRPO replaces stages." It is that **RL became capability-eliciting, not just behavior-polishing**. Long-form reasoning and self-verification were in no training set — the model invented them because they earned reward. That shifted post-training from a refinement phase into a capability-generation phase, which is why post-training compute budgets are exploding in 2026.

The mental model shift to internalize: pre-2025 we thought of post-training as "make the pretrained model nicer." Post-2025 we think of it as "post-training is where new capabilities are added on top of the pretrained foundation." Pretraining gives knowledge; SFT teaches format and instruction-following; RFT cheaply boosts capability; GRPO+RLVR drives reasoning; RLHF/DPO handles safety and style.

## §7 The interview-grade phrasing

!!! note "Polished one-paragraph version"
    "R1-Zero proved pure RL via GRPO can elicit reasoning from a strong base without SFT, but production R1 reintroduced SFT for readability and a better RL starting policy. The 2026 pipeline is multi-stage: pretraining gives knowledge, SFT gives instruction-following and format, RFT cheaply boosts capability, GRPO+RLVR drives reasoning, and RLHF/DPO handles safety and style. GRPO does not replace a stage; it is the most powerful new tool added to the stack."

This is the version to deliver when an interviewer asks "is GRPO going to replace SFT?" or "what makes DeepSeek-R1 so important?"

## Interview Questions

**Q1: Can GRPO replace pretraining?**

No. RL shapes behavior over a pretrained substrate; it cannot create knowledge, and the compute and signal math is infeasible. A randomly initialized network produces gibberish, no rewards, no gradient signal. Factual recall has no verifier reward in the RLVR sense, so there is no way for RL to teach the model that France's capital is Paris.

**Q2: The real significance of R1-Zero?**

RL became capability-eliciting (it discovered reasoning patterns that were in no dataset), not merely behavior-polishing. Pre-2025 we thought of post-training as "make the pretrained model nicer." Post-2025 we think of it as where new capabilities are added. Long-form reasoning, self-verification, and reflection emerged from RL on a strong base because they earned reward, even though no SFT data taught them.

**Q3: Why does production R1 still use SFT if R1-Zero showed pure RL works?**

R1-Zero has endless repetition, poor readability, and language mixing. It is technically impressive but not shippable. Production R1 uses 2 SFT stages and 2 RL stages: SFT teaches format and instruction-following; RL teaches reasoning. The combination ships; either alone does not.

**Q4: Will GRPO eventually replace SFT?**

Probably not, even at the frontier. SFT remains the cheapest way to teach format, instruction-following, and tool use; RL is the expensive way to teach genuinely new behaviors. The 2026 pipeline uses both because each is good at different things. RFT (rejection-sampling fine-tuning on verifier-passing traces) is the cheap way to combine them.

**Q5 (Trap): A timeline post claims "GRPO is so good it will train frontier models from scratch by 2027." How do you respond?**

It will not. Three reasons. (1) RL needs a reasonable starting policy; random initialization produces gibberish with no rewards. (2) RL shapes behavior, it does not create knowledge — factual recall has no verifier. (3) The compute math is infeasible: dense pretraining gives gradient on every token; sparse RL rewards plus expensive rollouts cannot match that signal even at huge cost. GRPO is the most powerful tool added to the post-training stack since RLHF, but it operates on a pretrained substrate, not in place of it.
