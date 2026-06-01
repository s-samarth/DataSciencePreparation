# GRPO — Group Relative Policy Optimization

A crucial correction up front: GRPO is *not* a DPO variant despite the chatter. It is a **PPO variant that eliminates the critic.** It is still RL, with rollouts and rewards. The naming just confuses people. The trick is to use the group mean reward as the baseline instead of a learned value function.

!!! tip "Rapid Recall"
    For each prompt, sample a **group of G responses** (typically 8 to 16). Compute each one's reward, then normalize within the group: `A_i = (R_i - mean(R)) / std(R)`. Same advantage is used for every token in the response. Plug into PPO's clipped surrogate with KL added directly to the loss (not folded into per-token rewards). Result: drops the critic (~25% memory savings, no critic instability), keeps PPO's structure. **Failure mode: advantage collapse** when all G samples in a group get the same reward — std = 0, no learning signal. Real implementations mask zero-variance groups out. **Dr. GRPO** drops std-normalization (which biases toward short responses); **DAPO** adds dynamic sampling and decoupled clip ratios. Every open-source reasoning model trained since DeepSeek-R1 uses GRPO or a variant.

## §1 The core trick

PPO's critic is expensive, hard to train on sparse rewards, and a source of bias. GRPO's insight: instead of learning \(V(s)\), sample several responses to the same prompt and use the **group's mean reward as the baseline**.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="GRPO group-relative advantage">
<text x="40" y="40" font-family="Newsreader,serif" font-size="15" fill="#1a1410">One prompt → sample G responses → score each → advantage = standardized deviation from the group mean</text>
<g font-family="JetBrains Mono,monospace" font-size="11">
<rect x="60"  y="90" width="50" height="110" fill="#2c6e63" fill-opacity="0.25" stroke="#2c6e63"/>
<rect x="140" y="70"  width="50" height="130" fill="#2c6e63" fill-opacity="0.4" stroke="#2c6e63"/>
<rect x="220" y="140" width="50" height="60"  fill="#b4451f" fill-opacity="0.3" stroke="#b4451f"/>
<rect x="300" y="60"  width="50" height="140" fill="#2c6e63" fill-opacity="0.5" stroke="#2c6e63"/>
<rect x="380" y="160" width="50" height="40"  fill="#b4451f" fill-opacity="0.3" stroke="#b4451f"/>
<rect x="460" y="110" width="50" height="90"  fill="#2c6e63" fill-opacity="0.2" stroke="#2c6e63"/>
<text x="85"  y="216" text-anchor="middle">y&#8321;</text>
<text x="165" y="216" text-anchor="middle">y&#8322;</text>
<text x="245" y="216" text-anchor="middle">y&#8323;</text>
<text x="325" y="216" text-anchor="middle">y&#8324;</text>
<text x="405" y="216" text-anchor="middle">y&#8325;</text>
<text x="485" y="216" text-anchor="middle">y&#8326;</text>
</g>
<line x1="50" y1="120" x2="540" y2="120" stroke="#c98a2b" stroke-width="2" stroke-dasharray="6 4"/>
<text x="548" y="124" font-family="JetBrains Mono,monospace" font-size="12" fill="#9a6a1e">group mean</text>
<text x="548" y="100" font-family="Newsreader,serif" font-style="italic" font-size="13" fill="#2c6e63">above → A&gt;0</text>
<text x="548" y="155" font-family="Newsreader,serif" font-style="italic" font-size="13" fill="#b4451f">below → A&lt;0</text>
</svg>
<figcaption>Advantage = (r_i - mean) / std. The baseline is the group, not a learned value function.</figcaption>
</figure>

## §2 The advantage formula

For each token in response \(i\) of group \(g\):

\[ A_{g,i,t} = \frac{R_{g,i} - \mu_g}{\sigma_g + \epsilon} \]

where \(\mu_g\) and \(\sigma_g\) are the mean and std of rewards \(\{R_{g,1}, \dots, R_{g,G}\}\) for the group. Note: **the same advantage is used for every token in the response.** This is a deliberate simplification; no per-token credit assignment.

## §3 The GRPO loss

That advantage feeds the same PPO clipped objective, plus a KL term, usually inside the objective directly rather than folded into per-token rewards:

\[ \mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac{1}{G}\sum_{g,i}\frac{1}{|y_{g,i}|}\sum_t \min\big(r_t(\theta) A_{g,i,t}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_{g,i,t}\big)\right] + \beta \cdot \mathrm{KL}\big(\pi_\theta \,\Vert\, \pi_{\text{ref}}\big) \]

Same clipped objective as PPO. Same importance ratio. Just a different (cheaper) advantage estimator plus KL added directly to the loss instead of folded into per-token rewards.

### Key differences from PPO

- **No critic.** One fewer model.
- **One shared advantage per response.** Versus per-token GAE.
- **No reward model needed for verifiable tasks.** Rule-based checks suffice; see [RLVR](rlvr.md).

This is how DeepSeek-R1-Zero ran: no RM, just verifier rewards.

## §4 Why GRPO shines for reasoning

- **Verifiable rewards** (math/code). Reward is binary or sparse, exactly the regime where a learned critic suffers most.
- **Cheap parallel sampling.** Modern inference servers (vLLM, SGLang) batch G samples in a single forward pass.
- **Informative cross-sample variance.** When some samples succeed and others fail, the advantage signal is sharp.

## §5 Where it breaks

- **G× sampling cost.** Generating 8 to 16 responses per prompt is expensive even with batched sampling.
- **Advantage collapse** when all G samples in a group get the same reward. Std → 0, no signal. Common on too-easy prompts (everyone gets it right) or too-hard prompts (everyone gets it wrong). Mask zero-variance groups out (a `valid_group` flag).
- **No per-token credit.** Every token in a response shares the same advantage, even tokens that did not contribute.
- **Still needs full RL infrastructure.** Rollouts, KL bookkeeping, multi-model memory (policy + reference).

## §6 The implementation skeleton

```python
def grpo_rollout(policy, ref, reward_fn, n_prompts, G):
    """Sample G responses per prompt. Group-normalize advantages."""
    prompts = make_prompts(n_prompts)
    prompts_rep = prompts.repeat_interleave(G, dim=0)
    with torch.no_grad():
        full_ids = policy.generate(prompts_rep, max_new_tokens=16, temperature=1.0)
        rewards = reward_fn(full_ids, prompt_len)             # [N*G]

        # Group-normalize: reshape to [N, G], compute (R - mean) / std
        r_grp = rewards.view(n_prompts, G)
        mu    = r_grp.mean(dim=1, keepdim=True)
        sigma = r_grp.std (dim=1, keepdim=True)
        valid = (sigma.squeeze(1) > 1e-6)                     # mask zero-var groups
        adv = torch.zeros_like(r_grp)
        adv[valid] = (r_grp[valid] - mu[valid]) / (sigma[valid] + 1e-6)
        advantages = adv.view(n_prompts * G)                  # broadcast to tokens
    return advantages
```

For the full rollout-plus-update including action masks and KL, see [Alignment Walkthrough §7](../build-from-scratch/alignment-walkthrough.md).

## §7 Refinements (Dr. GRPO, DAPO)

- **Dr. GRPO** drops std-normalization, which biases toward short responses (short responses have less variability, so they get artificially inflated advantages). Use mean-centered advantages only.
- **DAPO** adds dynamic sampling (more samples for high-uncertainty prompts) and decoupled clip ratios (different ε for the positive and negative advantage cases).

These are minor tweaks; the core GRPO recipe is what matters.

## §8 Hyperparameters

DeepSeek-R1's setup as reference:

- `G = 16` outputs per prompt.
- Temperature = 1.0 during rollout.
- KL coefficient β = 0.001 (much smaller than PPO's typical 0.05 because KL is added once to the loss, not per token).
- Clip ratio ε = 10 (huge, because the group advantage absorbs scale variation).
- Reward signal: **rule-based** (math answer correctness, code passing tests, format compliance via `<think>` / `<answer>` tags).
- **No SFT before RL in R1-Zero** — pure RL on the base model.

## Interview Questions

**Q1: Why does GRPO not need a critic?**

The group mean serves as the baseline for the advantage, which is exactly the job the critic did in PPO. Sampling G responses gives an unbiased Monte Carlo estimate of the expected reward for a given prompt, which is what V(s) was estimating. Dropping the critic saves one model in memory (~25% reduction) and removes the critic-instability failure mode where the value head fails to learn on sparse rewards.

**Q2: GRPO vs REINFORCE-with-baseline?**

Conceptually close (both use a sample-based baseline), but GRPO adds PPO's clipped ratio and an explicit KL term to a reference model. REINFORCE has no clipping, so updates can drift arbitrarily far per step. GRPO is "REINFORCE with PPO's stability tricks plus a trust region to a reference," not just "REINFORCE."

**Q3: Why group responses by prompt?**

To get a meaningful baseline. Comparing rewards across different prompts is apples-to-oranges (some prompts are inherently harder); grouping isolates "how does this response compare for THIS prompt." It is the same variance-reduction argument as PPO's critic, but Monte Carlo instead of learned.

**Q4: When does std-normalization hurt?**

When group std is dominated by length or format differences rather than quality, you can inflate advantages for responses that happen to have low variance neighbors. Dr. GRPO removes std-normalization for this reason and uses mean-centered advantages only.

**Q5: When does GRPO fail catastrophically?**

Zero-variance groups (all responses get the same reward). The advantage becomes 0 for every sample and there is no gradient signal. This happens on prompts that are too easy (everyone solves it) or too hard (everyone fails). Fixes: filter zero-variance groups before computing loss, increase sampling temperature, or curriculum the prompts so the group sees a mix of solvable and challenging items.

**Q6: Is GRPO a DPO variant?**

No, this is the trap question and the most common confusion. GRPO is a PPO variant that drops the critic and uses the group mean as a baseline. Still RL, still rollouts, still online. DPO is supervised and offline; GRPO is reinforcement learning. The two are not in the same family.
