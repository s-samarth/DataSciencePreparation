# Reward Models and Bradley-Terry

The first stage of classic RLHF. Goal: a function that assigns a scalar reward `r(x, y)` to *any* prompt-response pair, including responses the policy has not generated yet. The Bradley-Terry preference model from 1952 is the statistical backbone, and the loss is plain Maximum Likelihood Estimation. Once you see this, DPO's "closed form of RLHF" stops looking like magic.

!!! tip "Rapid Recall"
    A reward model `r_φ(x, y)` is a neural network (usually the SFT model with the LM head swapped for a scalar head) trained on preference pairs `(x, y_w, y_l)`. The **Bradley-Terry assumption**: probability of preferring chosen over rejected is `σ(r(y_w) - r(y_l))`. The loss is **negative log-likelihood under Bradley-Terry**: `L = -log σ(r_chosen - r_rejected)`. The gradient carries a clean `(1 - σ(Δ))` factor — "how wrong am I right now" — large when wrong, near-zero when confidently right. Bradley-Terry is invariant to additive shifts in reward, so the absolute reward magnitude is meaningless; only the gap between chosen and rejected matters. This same Bradley-Terry loss is the backbone of both RLHF (via PPO) and DPO.

## §1 Why a neural network?

You collected ~50K to ~500K pairwise human comparisons. A lookup table only covers pairs you have seen. Hand-crafted rules are brittle. Raw labels do not exist for novel responses. The neural network is a **function approximator that generalizes** from your finite comparisons to any text the policy might produce. That is its entire job.

It is almost always initialized from the SFT model (so it already understands language), with the LM head swapped for a linear layer emitting a single scalar from the final token's hidden state.

```python
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        self.transformer = base_lm.transformer    # reuse pretrained weights
        self.reward_head = nn.Linear(hidden_dim, 1)  # NEW, replaces LM head

    def forward(self, input_ids):
        hidden = self.transformer(input_ids)
        return self.reward_head(hidden[:, -1, :])  # scalar from last token
```

For the runnable from-scratch version on the vowel-count toy task, see [Alignment Walkthrough §4](../build-from-scratch/alignment-walkthrough.md).

## §2 Bradley-Terry: the modeling assumption

A 1952 model for ranking from pairwise comparisons. The probability that the winning response beats the loser is a softmax over their underlying rewards, which simplifies to a sigmoid of the reward difference:

\[ P(y_w \succ y_l) = \frac{e^{r(y_w)}}{e^{r(y_w)} + e^{r(y_l)}} = \sigma\big(r(y_w) - r(y_l)\big) \]

Then it is pure MLE (see [MLE and MAP backbone](../sft/mle-map-backbone.md)): write the likelihood of every observed preference, take the negative log, minimize. This *is* where the "expectation" in the papers comes from — it is just an average over the dataset.

\[ \mathcal{L}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\Big] \]

In code, this is the entire RM training loss:

```python
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

## §3 What the gradient is doing

Let \(\Delta = r_\theta(y_w) - r_\theta(y_l)\). The loss \(-\log\sigma(\Delta)\) pushes the winner's reward up and the loser's down.

- Large positive \(\Delta\) → \(\sigma(\Delta) \to 1\) → \(\log(1) = 0\) → near-zero loss.
- Negative \(\Delta\) → \(\sigma(\Delta) \to 0\) → \(\log(\to 0) = -\infty\) → exploding loss.

The gradient carries a clean \((1 - \sigma(\Delta))\) factor — "how wrong am I right now" — large when wrong, near-zero when confidently right. The optimizer focuses gradient flow on pairs the RM is currently getting wrong, which is exactly what you want.

## §4 The shift invariance trick

Bradley-Terry is invariant to additive shifts: \(r\) and \(r + c\) give identical preferences because the constant cancels in the difference. Only differences matter. Two consequences.

- The absolute reward magnitude is meaningless. You cannot say "this response got reward +3.7" in any absolute sense.
- Normalizing reward to zero mean per batch stabilizes downstream PPO without changing what the RM expresses. Frontier training pipelines do this routinely.

## §5 What pooling strategy?

"Scoring a response" means we run the model on `[prompt + response]`, take the hidden state at the position of the LAST token (typically the EOS), and project to a scalar. That is the reward for the (prompt, response) pair. The last-token pooling reflects the autoregressive view: by the last token, the model has seen everything and produces its summary.

The PEFT-style alternative is to take the hidden state from a special `<|reward|>` token appended at the end, but in practice the last-EOS pooling works as well.

## §6 The trap question on dimensionality

!!! warning "Why a scalar, not a distribution?"
    Because Bradley-Terry assumes a 1D utility function. Multi-dim preferences (helpfulness, harmlessness, conciseness, …) need separate RMs or a multi-head RM that produces a vector of rewards. Frontier labs do train multi-objective RMs and combine them at inference, but the core Bradley-Terry assumption is one scalar per response. Asking the RM to express conflicting preferences as a single scalar is what creates many of the alignment-tax behaviors you see (sycophancy, verbosity bias, helpfulness/harmlessness tension).

## §7 RM training in five lines

```python
def rm_loss(rm, batch):
    r_chosen   = rm(batch.chosen_ids)
    r_rejected = rm(batch.rejected_ids)
    return -F.logsigmoid(r_chosen - r_rejected).mean()

for batch in loader:
    loss = rm_loss(rm, batch)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

That is the whole story. The RM is just a Bradley-Terry classifier with a transformer body.

## Interview Questions

**Q1: Why a learned RM instead of optimizing preferences directly?**

That is exactly what DPO does (see [DPO](dpo.md)) — it derives a closed form that bypasses the RM, while still assuming Bradley-Terry. The RM exists as a separate stage in classic RLHF because PPO needs a reward signal at every rollout, and computing pairwise preferences online during PPO would require humans-in-the-loop in real time. The RM amortizes the preference data into a function that can be queried any time during PPO.

**Q2: What is reward hacking and how does the RM enable it?**

The RM is a finite-data approximation. The policy can find responses exploiting RM blind spots — verbosity, sycophancy, specific phrasings — that score high on the RM but are nonsense to humans. The KL penalty mitigates but never eliminates it; see [RL from zero and PPO §5](rl-from-zero-and-ppo.md).

**Q3: Why shift rewards to zero mean per batch?**

Bradley-Terry is invariant to additive shifts: \(r\) and \(r+c\) give identical preferences. Only differences matter, so normalizing absolute scale stabilizes PPO without changing what the RM expresses. Centered rewards keep advantage estimates near zero on average, which makes variance reduction more effective.

**Q4: Can you trust the RM out of distribution?**

No. As the policy improves and drifts from the RM's training data, RM reliability degrades, which is why RLHF is done in multiple rounds of re-collecting preferences and retraining the RM. The 2024-2026 frontier pattern is "iterative RLHF": collect, train RM, run PPO, sample new responses, collect new preferences on those, retrain RM, repeat.

**Q5 (Trap): Why a scalar reward, not a distribution over rewards?**

Because Bradley-Terry assumes a 1D utility function — the preference probability is a sigmoid of one scalar difference, which only makes sense if reward is a scalar. Multi-objective preferences need either separate RMs (one per objective) or a multi-head RM producing a reward vector combined at inference. Asking one scalar to express "helpfulness OR harmlessness OR conciseness" is what creates many alignment-tax behaviors.
