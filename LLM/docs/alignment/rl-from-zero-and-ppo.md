# RL From Zero and PPO

The full classic RLHF stack, derived from scratch. RL vocabulary, the non-obvious step where an LLM is already a policy, why naive REINFORCE fails, the clipped surrogate, the KL penalty, and the famous four models in memory. Once you can name what each of the four models does, the rest of post-training organizes itself around what each subsequent method removes.

!!! tip "Rapid Recall"
    **An LLM is already a policy:** the softmax over next-token logits *is* \(\pi(a|s)\). Generating a response is rolling out a trajectory. The RLHF reward is sparse — one scalar at the very end. Naive policy gradient (REINFORCE) has crushing variance, off-policy issues, and destructive single-step updates. **PPO** layers three fixes: (1) **advantage estimation** with a learned critic V(s) for variance reduction; (2) **clipped surrogate objective** with importance ratio `π_θ / π_θ_old` clipped to `[1-ε, 1+ε]` for a trust region; (3) **KL penalty** to a frozen reference model to prevent reward hacking. Cost: **four models in memory** — policy (trained), reference (frozen SFT), reward model (frozen), value/critic (trained).

## §1 Reinforcement learning, from zero

No labeled "right answer" per step — just an agent trying things and learning from outcomes. The vocabulary: an **agent** observes a **state** \(s\), picks an **action** \(a\) via its **policy** \(\pi\); the **environment** returns a **reward** \(r\) and a new state. Loop. Goal: a policy that maximizes total expected reward.

A policy is literally the agent's strategy, a function from state to action (or a distribution over actions).

## §2 Mapping RL onto LLMs

The non-obvious translation that unlocks everything: **an LLM is already a policy.** The softmax over next-token logits *is* \(\pi(a \mid s)\).

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mapping RL concepts onto LLM generation">
<defs><marker id="rlarr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#b4451f"/></marker></defs>
<rect x="20" y="30" width="150" height="230" rx="10" fill="#ece1cd" stroke="#d4c4a8"/>
<text x="95" y="55" text-anchor="middle" font-family="Fraunces, serif" font-weight="600" font-size="16" fill="#8a3115">RL concept</text>
<text x="95" y="92" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="#1a1410">policy &#960;</text>
<text x="95" y="128" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="#1a1410">state s&#7511;</text>
<text x="95" y="164" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="#1a1410">action a&#7511;</text>
<text x="95" y="200" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="#1a1410">reward r</text>
<text x="95" y="236" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="#1a1410">episode</text>
<line x1="175" y1="87" x2="345" y2="87" stroke="#b4451f" stroke-width="1.5" marker-end="url(#rlarr)"/>
<line x1="175" y1="123" x2="345" y2="123" stroke="#b4451f" stroke-width="1.5" marker-end="url(#rlarr)"/>
<line x1="175" y1="159" x2="345" y2="159" stroke="#b4451f" stroke-width="1.5" marker-end="url(#rlarr)"/>
<line x1="175" y1="195" x2="345" y2="195" stroke="#b4451f" stroke-width="1.5" marker-end="url(#rlarr)"/>
<line x1="175" y1="231" x2="345" y2="231" stroke="#b4451f" stroke-width="1.5" marker-end="url(#rlarr)"/>
<rect x="350" y="30" width="330" height="230" rx="10" fill="#2c6e63" fill-opacity="0.08" stroke="#2c6e63" stroke-opacity="0.4"/>
<text x="515" y="55" text-anchor="middle" font-family="Fraunces, serif" font-weight="600" font-size="16" fill="#1d4a42">LLM equivalent</text>
<text x="515" y="92" text-anchor="middle" font-family="Newsreader, serif" font-size="14.5" fill="#1a1410">the LLM itself (softmax over tokens)</text>
<text x="515" y="128" text-anchor="middle" font-family="Newsreader, serif" font-size="14.5" fill="#1a1410">prompt + tokens generated so far</text>
<text x="515" y="164" text-anchor="middle" font-family="Newsreader, serif" font-size="14.5" fill="#1a1410">the next token</text>
<text x="515" y="200" text-anchor="middle" font-family="Newsreader, serif" font-size="14.5" fill="#1a1410">RM score, at the END of the response</text>
<text x="515" y="236" text-anchor="middle" font-family="Newsreader, serif" font-size="14.5" fill="#1a1410">one full generation: prompt → response</text>
</svg>
<figcaption>Generating a response is rolling out a trajectory. Crucially, the reward in RLHF is sparse — one scalar at the very end, not per token.</figcaption>
</figure>

## §3 Why naive REINFORCE fails

The obvious move — gradient ascent on expected reward via the policy gradient theorem — works, but suffers from:

- **High variance.** A single trajectory's reward is a noisy estimate of the gradient direction.
- **Sample inefficiency.** On-policy: each sample is valid for one update step.
- **No baseline.** Every token in a high-reward trajectory gets reinforced equally, even the ones that did not help.
- **Destructive updates.** One bad step can collapse the model into gibberish with no path back, especially with sparse rewards.

PPO exists to fix that last problem above all.

## §4 The four models in PPO

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The four models in PPO RLHF">
<rect x="20"  y="40" width="155" height="160" rx="10" fill="#b4451f" fill-opacity="0.1" stroke="#b4451f"/>
<rect x="195" y="40" width="155" height="160" rx="10" fill="#2c6e63" fill-opacity="0.1" stroke="#2c6e63"/>
<rect x="370" y="40" width="155" height="160" rx="10" fill="#c98a2b" fill-opacity="0.13" stroke="#c98a2b"/>
<rect x="545" y="40" width="135" height="160" rx="10" fill="#6a3a5e" fill-opacity="0.1" stroke="#6a3a5e"/>
<text x="97"  y="72" text-anchor="middle" font-family="Fraunces,serif" font-weight="600" font-size="15" fill="#8a3115">Policy</text>
<text x="97"  y="92" text-anchor="middle" font-family="Newsreader,serif" font-style="italic" font-size="12.5" fill="#5a4f42">(actor)</text>
<text x="97" y="128" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">the LLM being</text>
<text x="97" y="146" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">trained</text>
<text x="97" y="178" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10.5" fill="#b4451f">TRAINED</text>
<text x="272" y="72" text-anchor="middle" font-family="Fraunces,serif" font-weight="600" font-size="15" fill="#1d4a42">Reference</text>
<text x="272" y="92" text-anchor="middle" font-family="Newsreader,serif" font-style="italic" font-size="12.5" fill="#5a4f42">(frozen SFT)</text>
<text x="272" y="128" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">KL penalty</text>
<text x="272" y="146" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">anchor</text>
<text x="272" y="178" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10.5" fill="#2c6e63">FROZEN</text>
<text x="447" y="72" text-anchor="middle" font-family="Fraunces,serif" font-weight="600" font-size="15" fill="#9a6a1e">Reward</text>
<text x="447" y="92" text-anchor="middle" font-family="Newsreader,serif" font-style="italic" font-size="12.5" fill="#5a4f42">(frozen RM)</text>
<text x="447" y="128" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">scores the</text>
<text x="447" y="146" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">full response</text>
<text x="447" y="178" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10.5" fill="#c98a2b">FROZEN</text>
<text x="612" y="72" text-anchor="middle" font-family="Fraunces,serif" font-weight="600" font-size="15" fill="#6a3a5e">Value</text>
<text x="612" y="92" text-anchor="middle" font-family="Newsreader,serif" font-style="italic" font-size="12.5" fill="#5a4f42">(critic)</text>
<text x="612" y="128" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">estimates V(s)</text>
<text x="612" y="146" text-anchor="middle" font-family="Newsreader,serif" font-size="13" fill="#1a1410">for advantages</text>
<text x="612" y="178" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10.5" fill="#6a3a5e">TRAINED</text>
</svg>
<figcaption>Four models, the reason PPO for LLMs is so expensive. Two are trained (policy, critic), two are frozen (reference, reward).</figcaption>
</figure>

## §5 PPO — three machineries

### 5.1 Idea 1 — advantage, and why you need a critic

Instead of "was this response good?" ask "was it *better than expected* from this state?" That is the advantage: \(A_t = R_t - V(s_t)\). If every response scores ~+5, raw reward is uninformative; advantage centered near zero gives a clean low-variance signal.

\(V(s)\), the expected reward from state \(s\) onward, is not known analytically, so you **learn it with a neural network: the critic.** Same trick as the RM: SFT backbone plus a scalar value head, trained by regression on observed returns:

\[ \mathcal{L}_{\text{value}} = (V_\phi(s_t) - R_t)^2 \]

Policy plus critic training together is the **actor-critic** setup. This is precisely why the value model exists: the RM gives the terminal reward; the critic gives per-state baselines.

### 5.2 Idea 2 — the clipped objective (the "proximal" part)

Define the probability ratio:

\[ r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \]

PPO maximizes:

\[ \mathcal{L}^{\text{PPO}} = \mathbb{E}_t\Big[\min\big(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big] \]

With \(\epsilon = 0.2\), the ratio is clipped to \([0.8, 1.2]\). The `min` creates a trust region: when \(A_t > 0\) you increase probability but stop benefiting past +20%; when \(A_t < 0\) you keep suppressing the bad action. The policy moves only modestly from \(\pi_{\text{old}}\) — no catastrophic jumps. Because the clip keeps you near the data-generating policy, you can take *several* gradient steps on the same batch, which is why PPO is more sample-efficient than vanilla policy gradient.

### 5.3 Idea 3 — KL penalty and the reference model

Even with clipping, the policy can reward-hack — drift into RM blind spots. So subtract a penalty for diverging from the frozen SFT model (the **reference model**):

\[ r_{\text{final}}(x, y) = R_\phi(x, y) - \beta \, D_{KL}\big(\pi_\theta(\cdot \mid x) \,\Vert\, \pi_{\text{ref}}(\cdot \mid x)\big) \]

!!! abstract "Two different jobs"
    The **clip ratio** constrains update size *per step*. The **KL penalty** constrains total drift from the original SFT model across *all* of training. They are not redundant.

## §6 One PPO iteration, end to end

1. Sample prompts; **generate** responses with \(\pi_\theta\) (save log-probs as \(\pi_{\text{old}}\)).
2. **Score** responses with the frozen RM.
3. Build per-token rewards: KL penalty on every token, RM reward added only on the last token.
4. **Compute advantages** via the critic (usually with GAE smoothing).
5. **Update policy** with the clipped objective (multiple steps per batch).
6. **Update critic** by MSE to observed returns. Repeat.

For the runnable rollout-plus-update implementation, see [Alignment Walkthrough §5](../build-from-scratch/alignment-walkthrough.md).

## §7 Why RLHF is hard, in five bullets

1. **Four models in memory** — expensive. Need big clusters. For a 70B base model that is crushing.
2. **RL is unstable** — PPO has many hyperparameters (LR, β, ε, GAE λ), and they interact. Reward goes up, then crashes. Loops diverge.
3. **Reward hacking** — the policy finds ways to game the RM. Classic example: if RM prefers longer answers, policy learns to produce rambling verbose outputs.
4. **RM quality ceiling** — your final model can only be as good as your RM. If the RM misjudges a preference, the policy learns that misjudgment.
5. **Compute** — one full RLHF run on a 70B model takes days on a multi-node cluster.

This is why OpenAI and Anthropic can do it (they have the infra and RL experts), but a 10-person startup usually cannot. Enter [DPO](dpo.md).

## Interview Questions

**Q1: Why is the reward only at the end, not per token?**

The RM was trained on complete responses, so it only knows how to score finished outputs. Per-token rewards need per-token reward models, which are harder to get from human preferences. The KL penalty is added at every token, but the RM score is added only at the final response token; GAE smooths this terminal reward backward across the trajectory.

**Q2: What happens if β = 0 (no KL penalty)?**

Reward hacking. The policy exploits RM blind spots and drifts into verbose, sycophantic, or repetitive outputs that score high but are useless. The KL penalty is what keeps the policy in the RM's "trust region."

**Q3: What happens if β is too high?**

The policy never moves; RL becomes useless because any change from SFT is penalized away. You get exactly the SFT model back, which is fine but defeats the point of running RLHF.

**Q4: Why multiple gradient steps per batch (vs one for vanilla PG)?**

The clip keeps the policy near the data-generating policy, so re-using the batch a few times is approximately valid. Vanilla policy gradient re-use would be off-policy and biased; PPO's clipping bounds the off-policy error so multiple updates remain trustworthy.

**Q5: What is the role of the critic if you already have a reward model?**

Different jobs. The RM scores complete responses after generation; the critic estimates expected future reward from any mid-generation state, used for advantages. The critic gives advantages (variance reduction); the RM gives the reward signal.

**Q6: Why advantage instead of raw return?**

Variance reduction. Subtracting any state-only baseline does not bias the gradient (provable from the policy gradient theorem), but sharply reduces variance. Without a baseline, all trajectories with positive reward get reinforced equally regardless of which actions were actually responsible, which makes credit assignment slow and noisy.

**Q7 (Trap): Why is PPO "on-policy" if we use an importance ratio?**

It is *approximately* on-policy. The clipping keeps us close enough to the data-generating policy that the off-policy correction (the ratio) does not blow up. Truly off-policy methods (DQN) do not need this constraint because they have explicit value functions and bootstrapping. PPO sits in the middle: re-use a batch a few times, but only if you stay near the policy that produced it.
