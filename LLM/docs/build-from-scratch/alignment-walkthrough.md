# Alignment Walkthrough — RM, PPO, DPO, GRPO, RLVR

Four alignment methods on one toy task so they are directly comparable: a tiny transformer pretrained to produce plausible letter sequences, then nudged to emit more vowels. Reward = `count_vowels(continuation)`. Verifiable, deterministic, instant. All in pure PyTorch on a Colab T4.

!!! tip "Rapid Recall"
    Every alignment method optimizes the same RLHF objective `E[r(x,y)] − β·KL(π‖π_ref)` but trades different things. **RM** trains a Bradley-Terry classifier on preference pairs (`loss = -logσ(r_chosen - r_rejected)`). **PPO** rolls out from the policy, scores with the RM, uses a critic for advantages, clips the importance ratio, and adds KL to the reward — four models in memory. **DPO** uses the closed-form optimal policy to rewrite the RLHF loss as a supervised contrastive loss on preference pairs — no RM, no rollouts, two models. **GRPO** keeps PPO's rollouts and clipped surrogate but kills the critic by using the group mean reward as the baseline. **RLVR** is GRPO with a programmatic verifier as the reward function — DeepSeek-R1's recipe.

## §1 The toy task

- Vocab: 26 letters + `<bos>` (26), `<eos>` (27), `<pad>` (28).
- Prompts are random 3-letter strings; model continues up to 16 tokens.
- Reward = number of vowels (`a, e, i, o, u`) in the continuation only (prompt excluded).

**Why this works for teaching RLHF:**

- Reward is a pure function: `count_vowels(continuation)`. No human labeler, no LLM judge.
- We can use this reward two ways: (1) as ground-truth to train a reward model (then "forget" it during RLHF), (2) as a verifier directly (RLVR — no RM at all).
- A 3M-param transformer can learn to inflate vowels in ~200 RL steps. You see it work.

```python
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
VOWELS  = set("aeiou")
VOCAB   = LETTERS + ['<bos>', '<eos>', '<pad>']
STOI    = {ch: i for i, ch in enumerate(VOCAB)}
ITOS    = {i: ch for i, ch in enumerate(VOCAB)}
BOS_ID, EOS_ID, PAD_ID = STOI['<bos>'], STOI['<eos>'], STOI['<pad>']
VOWEL_IDS = set(STOI[v] for v in VOWELS)

def count_vowels_in_ids(ids):
    return sum(1 for i in ids if i in VOWEL_IDS)
```

## §2 A tiny transformer (~3M params)

RLHF requires keeping multiple model copies in memory. PPO needs: policy, reference, value head, sometimes old-policy snapshot → 3-4× model size. GRPO needs: policy, reference. DPO needs: policy, reference. A 3M-param model means we can hold 4 copies in ~50 MB.

```python
class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x))
```

After pretraining on English-frequency-weighted random letters for ~1500 steps, the model produces plausible (English-ish) letter strings. Save its state so each alignment method starts from the same checkpoint: `PRETRAINED_STATE = copy.deepcopy(model.state_dict())`.

## §3 Generating synthetic preference pairs

RM training, PPO, and DPO all need pairs: (prompt, response A, response B, which won). In real RLHF, humans pick the winner. Here we use a "ground-truth preference oracle": the response with more vowels wins. Ties are dropped.

```python
@torch.no_grad()
def build_preference_pairs(model, n_pairs=2000, prompt_len=3, gen_len=16, temperature=1.2):
    """Generate (prompt, chosen, rejected) triples where chosen has more vowels."""
    pairs = []
    while len(pairs) < n_pairs:
        prompts = make_prompts(64, prompt_len=prompt_len)
        _, conts_a, rew_a = generate_and_score(model, prompts, max_new=gen_len, temperature=temperature)
        _, conts_b, rew_b = generate_and_score(model, prompts, max_new=gen_len, temperature=temperature)
        for prompt, ca, cb, ra, rb in zip(prompts.tolist(), conts_a, conts_b, rew_a, rew_b):
            if ra == rb: continue
            chosen, rejected = (ca, cb) if ra > rb else (cb, ca)
            pairs.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
            if len(pairs) >= n_pairs: break
    return pairs
```

## §4 Method 1 — Reward Model (Bradley-Terry)

A reward model `r_φ(x, y)` takes a prompt-response and returns a scalar.

**Bradley-Terry assumption:** the probability of preferring `y_w` over `y_l` is the sigmoid of the reward difference:

\[ P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) \]

**Training loss** (negative log-likelihood):

\[ \mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\right] \]

In code: `loss = -F.logsigmoid(r_chosen - r_rejected).mean()`. The gradient pushes the gap between chosen and rejected to grow.

**Architecture trick:** RM is the same transformer as the policy, with a single scalar output instead of vocab-size logits. Take the SFT model, replace `lm_head` with `Linear(n_embd, 1)`, run.

```python
class RewardModel(nn.Module):
    """A transformer with a scalar reward head instead of LM head."""
    def __init__(self, cfg, backbone_state_dict=None):
        super().__init__()
        # ... transformer body, copied from TinyLM ...
        self.reward_head = nn.Linear(cfg.n_embd, 1, bias=False)
        if backbone_state_dict is not None:
            own = self.state_dict()
            for k, v in backbone_state_dict.items():
                if k in own and 'head' not in k: own[k] = v
            self.load_state_dict(own, strict=False)

    def forward(self, idx, attention_mask=None):
        """Reward per sequence: pool at the last non-pad token."""
        # ... body forward ...
        per_token_reward = self.reward_head(x).squeeze(-1)   # [B, T]
        if attention_mask is None: return per_token_reward[:, -1]
        last_idx = attention_mask.sum(dim=1) - 1
        return per_token_reward[torch.arange(B), last_idx]


def train_reward_model(pairs, n_steps=600, batch_size=64, lr=3e-4):
    rm = RewardModel(cfg, backbone_state_dict=PRETRAINED_STATE).to(DEVICE)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    for step in range(n_steps):
        idx = np.random.choice(len(pairs), batch_size, replace=False)
        batch = [pairs[i] for i in idx]
        c_ids, c_mask, r_ids, r_mask = make_rm_batch(batch)
        r_chosen   = rm(c_ids, c_mask)
        r_rejected = rm(r_ids, r_mask)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()        # Bradley-Terry loss
        opt.zero_grad(); loss.backward(); opt.step()
```

For the math intuition and trap questions, see [Reward Models and Bradley-Terry](../alignment/reward-models-bradley-terry.md).

## §5 Method 2 — PPO

The four models in PPO: policy (trained), old-policy snapshot (frozen each iter for the importance ratio), reference (frozen pretrained, for KL), value head (trained alongside policy).

**Total reward:**

\[ R(x, y) = r_\phi(x, y) - \beta \cdot \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big) \]

**PPO clipped surrogate:**

\[ \mathcal{L}^{\text{CLIP}}(\theta) = -\mathbb{E}_t\left[\min\big(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\cdot A_t\big)\right] \]

where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the importance ratio. Clipping is the "proximal" in PPO.

**The rollout (skipped: GAE bookkeeping, padding):**

```python
def ppo_rollout(policy, ref_policy, rm, ppo_cfg):
    prompts = make_prompts(ppo_cfg.rollout_batch)
    with torch.no_grad():
        full_ids = policy.lm.generate(prompts, max_new_tokens=ppo_cfg.max_new_tokens,
                                       temperature=ppo_cfg.sample_temperature)
        # Log-probs of generated tokens under current policy AND reference
        _, logits_pi = policy.hidden_and_logits(full_ids)
        logp_pi  = gather_logprobs(logits_pi, full_ids)
        logits_ref = ref_policy(full_ids)
        logp_ref = gather_logprobs(logits_ref, full_ids)
        # Per-token KL penalty; RM reward at the LAST response token only
        per_token_kl = (logp_pi - logp_ref) * action_mask
        kl_reward = -ppo_cfg.kl_coef * per_token_kl
        token_rewards = kl_reward + last_response_idx * rewards_seq.unsqueeze(1)
        # GAE: walk backward to get advantages and returns
        # ... (omitted: standard GAE math) ...
    return {'full_ids': full_ids, 'old_logp': logp_pi.detach(),
            'advantages': advantages_normalized.detach(),
            'returns': returns.detach(), 'action_mask': action_mask.detach()}
```

**The update — this is the core PPO loss:**

```python
for epoch in range(ppo_cfg.ppo_epochs):
    for mb in mini_batches:
        _, logits = policy.hidden_and_logits(mb_ids)
        new_logp  = gather_logprobs(logits, mb_ids)
        _, values = policy(mb_ids)
        values    = values[:, :-1]

        # Importance ratio
        ratio = (new_logp - mb_old_logp).exp()
        # PPO clipped surrogate (negation: we MINIMIZE)
        surr1 = ratio * mb_adv
        surr2 = torch.clamp(ratio, 1 - ppo_cfg.clip_eps, 1 + ppo_cfg.clip_eps) * mb_adv
        policy_loss = -(torch.min(surr1, surr2) * mb_mask).sum() / mb_mask.sum().clamp(min=1)
        # Value loss (MSE) + entropy bonus
        value_loss  = ((values - mb_ret)**2 * mb_mask).sum() / mb_mask.sum().clamp(min=1)
        loss = policy_loss + ppo_cfg.value_coef * value_loss - ppo_cfg.entropy_coef * entropy
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

For full math (including GAE), see [RL from Zero and PPO](../alignment/rl-from-zero-and-ppo.md).

## §6 Method 3 — DPO

**The insight (Rafailov et al. 2023):** the KL-constrained RLHF objective has a closed-form optimal policy:

\[ \pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp\left(\frac{1}{\beta} r(x, y)\right) \]

Rearranging gives `r(x, y) = β log[π*(y|x) / π_ref(y|x)] + β log Z(x)`. The partition function `Z(x)` cancels when we form a Bradley-Terry preference loss over PAIRS. Substituting into BT gives the DPO loss:

\[ \mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\!\left(\beta \log \tfrac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \tfrac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right] \]

**Why this is huge:** no reward model, no rollouts, no value function. Standard supervised-style training loop on (chosen, rejected) pairs.

**The core loss in code is ~10 lines:**

```python
def train_dpo(pairs, n_steps=400, batch_size=32, lr=5e-6, beta=0.1):
    policy = TinyLM(cfg).to(DEVICE); policy.load_state_dict(PRETRAINED_STATE)
    ref    = TinyLM(cfg).to(DEVICE); ref.load_state_dict(PRETRAINED_STATE)
    for p in ref.parameters(): p.requires_grad = False
    ref.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    PROMPT_LEN = 4   # BOS + 3 letters
    for step in range(n_steps):
        batch = [pairs[i] for i in np.random.choice(len(pairs), batch_size, replace=False)]
        c_ids, r_ids = make_dpo_batch(batch, prompt_len=PROMPT_LEN)

        logp_pol_c = sequence_logprob(policy, c_ids, PROMPT_LEN)
        logp_pol_r = sequence_logprob(policy, r_ids, PROMPT_LEN)
        with torch.no_grad():
            logp_ref_c = sequence_logprob(ref, c_ids, PROMPT_LEN)
            logp_ref_r = sequence_logprob(ref, r_ids, PROMPT_LEN)

        logits_chosen   = beta * (logp_pol_c - logp_ref_c)
        logits_rejected = beta * (logp_pol_r - logp_ref_r)
        loss = -F.logsigmoid(logits_chosen - logits_rejected).mean()      # the DPO loss
        opt.zero_grad(); loss.backward(); opt.step()
```

For the full step-by-step derivation, see [DPO](../alignment/dpo.md).

## §7 Method 4 — GRPO

**Motivation:** PPO needs a value function. That is an extra model, extra memory, extra hyperparameters. GRPO's key idea: for each prompt, sample a GROUP of G responses. Use the group's reward statistics as the baseline.

**Advantage for response i in group g:**

\[ A_{g,i} = \frac{R_{g,i} - \mu_g}{\sigma_g + \epsilon} \]

Same advantage is used for every token in the response — a deliberate simplification.

**The rollout:**

```python
def grpo_rollout(policy, ref, reward_fn, grpo_cfg):
    """Sample G responses per prompt. Group-normalize advantages."""
    prompts = make_prompts(grpo_cfg.n_prompts)
    # Repeat each prompt G times so we have G samples per prompt in a single batch
    prompts_rep = prompts.repeat_interleave(grpo_cfg.group_size, dim=0)
    with torch.no_grad():
        full_ids = policy.generate(prompts_rep, max_new_tokens=grpo_cfg.max_new_tokens,
                                    temperature=grpo_cfg.sample_temperature)
        logp_pi  = gather_logprobs(policy(full_ids),  full_ids)
        logp_ref = gather_logprobs(ref(full_ids), full_ids)

        # Reward via the supplied reward function (RM or verifier)
        rewards = reward_fn(full_ids, prompt_len)

        # Group-normalize: reshape to [N, G], compute (R - mean) / std
        N, G = grpo_cfg.n_prompts, grpo_cfg.group_size
        r_grp = rewards.view(N, G)
        mu, sigma = r_grp.mean(dim=1, keepdim=True), r_grp.std(dim=1, keepdim=True)
        # Zero-variance groups: no learning signal (advantage collapse)
        valid = (sigma.squeeze(1) > 1e-6)
        adv = torch.zeros_like(r_grp)
        adv[valid] = (r_grp[valid] - mu[valid]) / (sigma[valid] + 1e-6)
        advantages = adv.view(N * G)
        per_token_adv = advantages.unsqueeze(1) * action_mask
    return {'full_ids': full_ids, 'old_logp': logp_pi, 'ref_logp': logp_ref,
            'advantages': per_token_adv, 'action_mask': action_mask, 'rewards': rewards}
```

**The update — PPO's clipped surrogate plus a direct KL term:**

```python
ratio = (new_logp - mb_oldlogp).exp()
surr1 = ratio * mb_adv
surr2 = torch.clamp(ratio, 1 - grpo_cfg.clip_eps, 1 + grpo_cfg.clip_eps) * mb_adv
policy_loss = -(torch.min(surr1, surr2) * mb_mask).sum() / mb_mask.sum().clamp(min=1)
# KL penalty (k1 estimator: mean over response tokens)
kl = ((new_logp - mb_reflogp) * mb_mask).sum() / mb_mask.sum().clamp(min=1)
loss = policy_loss + grpo_cfg.kl_coef * kl
```

**Failure mode:** if all G samples in a group get the same reward, advantage = 0 for every sample → no gradient. This is the "advantage collapse" / "all-negative-sample group" problem; real implementations mask zero-variance groups out (`valid_group` above).

## §8 Method 4b — RLVR

**RLVR is not a new algorithm.** It is GRPO (or PPO) where the reward is computed by a *verifier function* instead of a learned reward model. The verifier is just code:

- For math: parse the model's answer, compare to ground truth.
- For code: run unit tests, count passing tests.
- For our toy task: count vowels directly.

**The only thing we change:** swap `reward_fn_rm` for `reward_fn_verifier`.

```python
def reward_fn_verifier(full_ids, prompt_len):
    rewards = []
    for row in full_ids:
        cont = row[prompt_len:].tolist()
        for stop in (EOS_ID, PAD_ID):
            if stop in cont: cont = cont[:cont.index(stop)]; break
        rewards.append(count_vowels_in_ids(cont))
    return torch.tensor(rewards, dtype=torch.float, device=full_ids.device)

# Everything else stays the same — just pass this into train_grpo.
rlvr_policy, rlvr_rewards = train_grpo(reward_fn_verifier, label="GRPO+RLVR")
```

Most "reasoning model" training in 2025-2026 is RLVR. DeepSeek-R1's `<think>...</think>` reasoning was learned by GRPO with answer-correctness as reward — no reward model, no human preferences during the RL stage.

## §9 Side-by-side comparison

| Method | Needs RM? | Needs rollouts? | Needs value fn? | Best for | Key downside |
|---|---|---|---|---|---|
| **PPO** | Yes | Yes (online) | Yes | High-quality alignment when compute is available | Most complex, most memory, hyperparameter-sensitive |
| **DPO** | No | No (offline) | No | Cheap fine-tuning from preference pairs | Cannot use new generations during training; sometimes weaker than PPO |
| **GRPO** | Yes (usually) or No (RLVR) | Yes (online) | **No** | Reasoning tasks; modern reasoning models | Stalls on zero-variance groups; needs G samples per prompt |
| **RLVR** | **No** | Yes (online) | No (uses GRPO) | Verifiable tasks: math, code, tool use, structured output | Only works when reward is programmable |

## Interview-ready talking points

**"What is the difference between RLHF and RLAIF?"**
RLHF = humans provide preferences (slow, expensive). RLAIF = an LLM judges preferences (fast, scalable, biased toward judge model's preferences). Same algorithms underneath; only the labeler changes.

**"Why did the field move from PPO to DPO to GRPO?"**
PPO works but is engineering-heavy: value function, GAE, multiple model copies, careful hyperparameter tuning. DPO removed the RM and the rollouts (huge simplification). GRPO removed the value function while keeping the online RL structure (best of both worlds for reasoning).

**"Walk me through the DPO loss."**
DPO derives an implicit reward `r(x,y) = β log[π/π_ref]` from the PPO optimal-policy closed form. Plug into Bradley-Terry. Sigmoid-of-log-ratios loss. Train like a classifier on (chosen, rejected) pairs. No RM, no rollouts.

**"When does GRPO fail?"**
Zero-variance groups (all responses get same reward). The advantage becomes 0 and there is no learning signal. Fixes: filter zero-variance groups, increase exploration temperature, or use task-specific reward shaping to break ties.

**"Why is the KL term so important?"**
Without it, the policy can drift arbitrarily far from the reference. The RM (or verifier) only covers a thin slice of behavior space — outside that slice, reward predictions are unreliable, and you get **reward hacking**: outputs that score high on the RM but are nonsense to humans. KL acts as a regularizer keeping the policy in the RM's "trust region."

## Stretch goals

1. **Mixed reward:** combine RM + verifier. E.g., RM for fluency + verifier for correctness.
2. **Iterative DPO:** generate new pairs from your DPO'd model, label with RM (or verifier), retrain DPO. Bridges the offline/online gap.
3. **Scale up:** take a real ~1B model from the [SFT walkthrough](sft-walkthrough.md), define a verifier (e.g., answer-extraction regex on GSM8K), run GRPO. This is the actual DeepSeek-R1 recipe in miniature.
4. **Process reward:** instead of one reward per response, give per-step rewards.
5. **REINFORCE++:** popular GRPO variant that drops the importance ratio. Useful for understanding what each component contributes.
