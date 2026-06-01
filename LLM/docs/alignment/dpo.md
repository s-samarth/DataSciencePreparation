# DPO — Direct Preference Optimization

PPO's pipeline is heavy: train an RM, then run RL with four models. DPO's insight is that *the policy itself implicitly defines a reward function*, so you can rearrange the entire RLHF objective into one supervised loss on preference pairs. No reward model, no RL loop, no critic, no separate KL term. Just \((x, y_w, y_l)\) tuples and a single sigmoid-of-log-ratios loss.

!!! tip "Rapid Recall"
    DPO derives the closed-form optimal policy of the KL-constrained RLHF objective: `π* ∝ π_ref · exp(r/β)`. Take log, rearrange to express `r(x,y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)`. Plug into Bradley-Terry. **The intractable Z(x) cancels** because BT uses only the reward *difference* for the same prompt. The result is a supervised contrastive loss: `L = -log σ(β · log[π_θ(y_w)/π_ref(y_w)] - β · log[π_θ(y_l)/π_ref(y_l)])`. Two models in memory (policy + frozen reference), four log-prob computations per example, no rollouts, no critic. β controls how far the policy can drift from reference: typical 0.1, lower β = more drift, more alignment tax risk.

## §1 The core insight

**The RLHF objective has a closed-form optimal policy.** You do not actually need to run PPO. You can solve for what the optimal \(\pi_\theta\) looks like in terms of the reward function, then invert the relationship to get a reward function in terms of \(\pi_\theta\), then plug that into the Bradley-Terry loss — and now you are just doing supervised learning on preference pairs. No reward model, no PPO, no four models in memory. Just the policy and the reference.

Let us derive it step by step.

## §2 Step 1 — The RLHF objective has a closed-form solution

Start with the KL-constrained reward maximization objective:

\[ \pi^*(y \mid x) = \arg\max_\pi \mathbb{E}_{y \sim \pi} [r(x, y)] - \beta \, \mathrm{KL}(\pi \,\Vert\, \pi_{\text{ref}}) \]

This is a standard constrained optimization. You can solve it with Lagrangians or, more simply, recognize it as a distribution matching problem. The optimal policy is:

\[ \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left( \frac{1}{\beta} r(x, y) \right) \]

Where \(Z(x)\) is a partition function (normalizer) summing over all possible responses \(y\):

\[ Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\!\left( \frac{1}{\beta} r(x, y) \right) \]

**Intuition.** The optimal policy is the reference policy **reweighted by exponential of the reward**. High-reward responses get upweighted, low-reward responses get downweighted. β controls how sharply.

## §3 Step 2 — Invert to get reward in terms of policy

Here is the trick. Take the log of the equation above:

\[ \log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x) \]

Rearrange to solve for \(r(x, y)\):

\[ r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) \]

**This is the magic step.** We have expressed the reward entirely in terms of the (unknown) optimal policy \(\pi^*\) and the known reference policy \(\pi_{\text{ref}}\) — no neural network reward model required. The \(\log Z(x)\) term is just a function of \(x\), not \(y\).

## §4 Step 3 — Plug into Bradley-Terry

Remember the Bradley-Terry loss for preference modeling from [Reward Models](reward-models-bradley-terry.md):

\[ \mathcal{L}_{RM} = -\log \sigma(r(x, y_w) - r(x, y_l)) \]

We only need the **difference** \(r(x, y_w) - r(x, y_l)\). And watch what happens to the \(Z(x)\) term when we subtract:

\[ r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \beta \log Z(x) \]

The \(Z(x)\) terms **cancel**:

\[ r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \]

## §5 Step 4 — The DPO loss

Substitute into the Bradley-Terry loss. Now we are optimizing the policy \(\pi_\theta\) directly (pretending it is \(\pi^*\)):

\[ \boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)\right]} \]

**That is DPO.** A single supervised-style loss function. No reward model. No RL. No PPO. No value model. Just two models in memory (policy + frozen reference) and a sigmoid-of-log-ratios loss that you can optimize with AdamW.

## §6 What the loss is actually doing

Look at the term inside the sigmoid:

\[ \underbrace{\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}}_{\text{how much more likely } y_w \text{ is under } \pi_\theta \text{ vs } \pi_{\text{ref}}} - \underbrace{\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}}_{\text{how much more likely } y_l \text{ is under } \pi_\theta \text{ vs } \pi_{\text{ref}}} \]

Gradient descent pushes this difference to be large and positive, which means:

- **Increase the policy's probability of \(y_w\) relative to reference** (make the winning response more likely).
- **Decrease the policy's probability of \(y_l\) relative to reference** (make the losing response less likely).

It is a contrastive loss. Pull winners closer, push losers away, with the reference model as the anchor so you do not drift too far.

!!! note "Implicit reward framing"
    Define the implicit reward \(\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\). The DPO loss is then *identical in form to the Bradley-Terry RM loss* from [Reward Models](reward-models-bradley-terry.md), only now applied to the policy's own implicit rewards. Gradients boost the winner's probability over the reference and push the loser's below it. The reference-ratio structure bakes the KL constraint in automatically, which is why DPO needs no separate KL term.

## §7 The implementation in 10 lines

```python
def dpo_step(policy, ref, batch, beta=0.1):
    # 4 log-prob computations per example
    logp_pol_c = sequence_logprob(policy, batch.chosen_ids,   batch.prompt_len)
    logp_pol_r = sequence_logprob(policy, batch.rejected_ids, batch.prompt_len)
    with torch.no_grad():
        logp_ref_c = sequence_logprob(ref, batch.chosen_ids,   batch.prompt_len)
        logp_ref_r = sequence_logprob(ref, batch.rejected_ids, batch.prompt_len)

    logits_chosen   = beta * (logp_pol_c - logp_ref_c)
    logits_rejected = beta * (logp_pol_r - logp_ref_r)
    return -F.logsigmoid(logits_chosen - logits_rejected).mean()
```

For the runnable from-scratch version on the vowel-count toy task, see [Alignment Walkthrough §6](../build-from-scratch/alignment-walkthrough.md).

## §8 The hyperparameter β

β is inherited from the KL penalty coefficient. Higher β = stay closer to reference (safer, less drift). Lower β = move further from reference (stronger preference learning, more alignment-tax risk). Typical values: 0.1 to 0.5. Default in TRL: 0.1.

## §9 DPO's known limitations (2026 view)

DPO is not a free lunch.

- **Alignment tax.** DPO with low β can degrade capabilities on MMLU/HumanEval. Monitor benchmarks before/after.
- **Ceiling below PPO on hard tasks.** "Is DPO Superior to PPO?" (ICML 2024) showed PPO still wins on code generation and competitive math with enough compute. This is why OpenAI/Anthropic still use PPO-style RL for frontier models, while open-source ecosystems (Mistral, Llama fine-tunes, Qwen tunes) default to DPO.
- **Needs high-quality preference data.** Garbage pairs → garbage model. DPO has no reward model to average out noise.
- **Length bias.** Vanilla DPO can still learn to prefer longer responses spuriously. Variants like length-normalized DPO fix this.

## §10 DPO variants you should know in 2026

- **IPO** (Identity Preference Optimization): replaces the sigmoid with a squared loss to avoid DPO's tendency to overfit on easy pairs.
- **KTO** (Kahneman-Tversky Optimization): works with single-signal "good/bad" labels instead of pairwise. Huge unlock because pairwise data is expensive to collect.
- **ORPO** (Odds Ratio Preference Optimization): combines SFT + preference optimization in one step. Skip SFT entirely.
- **SimPO**: reference-model-free DPO. One less model in memory.
- **GRPO** (Group Relative Policy Optimization): see [GRPO](grpo.md). Used in DeepSeek-R1.

For interviews, know DPO deeply and mention KTO + GRPO as modern variants you would evaluate.

## Interview Questions

**Q1: Derive the DPO loss.**

Start with the KL-constrained reward maximization objective. Solve for the optimal policy in closed form: \(\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp(r(x,y)/\beta)\). Take log, rearrange to express reward as \(r(x,y) = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z(x)\). Plug into the Bradley-Terry preference loss \(-\log \sigma(r(y_w) - r(y_l))\). The partition function \(Z(x)\) cancels because it appears in both reward terms with the same sign for a fixed prompt. Result: a supervised-style loss on log-probability ratios between policy and reference, no reward model needed.

**Q2: Why is DPO simpler than RLHF?**

RLHF needs 4 models in memory (policy, reference, reward model, value model), PPO which is unstable with many hyperparameters, and a separate reward-modeling stage that creates an extra point of failure. DPO collapses this to a single supervised loss with 2 models (policy + reference frozen), using AdamW. No RL, no reward model. Training is more stable, uses less compute, and is easier to debug. The tradeoff: DPO has a lower ceiling on complex tasks like code generation, and is more sensitive to preference data quality because there is no reward model to smooth noise.

**Q3 (Trap): Why does Z(x) cancel in DPO?**

The Z(x) term is the normalizer from the closed-form optimal policy solution; it sums over all possible responses y for a given x, which is intractable. The magic of the DPO derivation is that Z(x) depends only on x, not y. When you plug the reward expression into Bradley-Terry, you compute r(x, y_w) − r(x, y_l), and Z(x) appears with the same sign in both terms. It **cancels**. This is the entire reason DPO is tractable. Without the cancellation, you would need to estimate Z(x) via sampling (which brings back RL-style difficulties).

**Q4: What is the tradeoff of lowering β in DPO?**

Lower β means weaker KL penalty, so the policy can drift further from the reference SFT model. Pros: stronger preference learning, better alignment to the preference data. Cons: alignment tax — the model can forget capabilities (MMLU, HumanEval scores drop), overfit to preference data idiosyncrasies, or develop repetitive verbose outputs. Standard default is β = 0.1. Monitor benchmark performance before and after; if capabilities drop more than 1 to 2%, raise β.

**Q5: Is DPO doing RL?**

No, pure supervised learning. The brilliance is showing the RLHF objective has a closed form that lets you skip RL entirely. The reference-ratio structure bakes the KL constraint in automatically, so you do not need a separate KL term either — just AdamW on the contrastive sigmoid loss.

**Q6: Why does DPO sometimes decrease the chosen response's probability?**

It optimizes the gap, not absolute probabilities. If lowering both (but lowering rejected more) is the easier descent direction, the gradient takes it. This is the motivation behind IPO (squared loss reduces this tendency) and a known failure mode where good-output probability collapses while loss looks fine.

**Q7: Can you do DPO without a reference model?**

SimPO does, using length-normalized log-probs directly. It loses theoretical grounding but works empirically and halves memory. Useful when memory is tight or when the reference model is already weak enough that anchoring to it adds little value.
