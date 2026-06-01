# RLVR — Reinforcement Learning from Verifiable Rewards

RLHF's bottleneck is the reward model: subjective, expensive, hackable, and unreliable off-distribution. RLVR replaces the learned RM with an external **verifier** — a program that objectively checks correctness. The DeepSeek-R1 recipe; reasoning models in general. The 2026 frontier shift in one acronym.

!!! tip "Rapid Recall"
    RLVR is not a new algorithm. It is GRPO (or PPO) where the reward function is a **deterministic verifier**, not a learned RM. For math, parse the model's answer and compare to ground truth. For code, run unit tests and count passes. For our toy vowel task, count vowels directly. The only code change vs GRPO+RM is swapping `reward_fn_rm` for `reward_fn_verifier`. **Why it matters:** scales with compute not labels, eliminates reward hacking in verifiable domains (verifier = ground truth), and elicits emergent reasoning behaviors (backtracking, self-correction, verification) that were never explicitly taught. **Failure modes:** verifier noise (38%+ false negatives in one analysis), sparse binary rewards stalling on all-correct or all-wrong batches, and process vs outcome — checking only final answers lets models reach right answers via bad reasoning.

## §1 The one-line definition

**RLVR is GRPO with a programmatic reward.** Same algorithm, different reward source.

| Axis | RLHF | RLVR |
|---|---|---|
| Reward source | Learned RM from preferences | Programmatic verifier |
| Signal | Continuous scalar | Usually binary (correct/incorrect) |
| Data | Pairwise human preferences | Problems with verifiable answers |
| Cost per example | Human label ($) | Programmatic check (~free) |
| Reward hacking | Significant | Largely solved (verifier = ground truth) |
| Best for | Open-ended generation | Math, code, logic, agent/tool tasks |

## §2 The code change is trivial

For the runnable version on the vowel-count toy task, see [Alignment Walkthrough §8](../build-from-scratch/alignment-walkthrough.md). The only thing that changes from GRPO+RM is the reward function:

```python
def reward_fn_verifier(full_ids, prompt_len):
    rewards = []
    for row in full_ids:
        cont = row[prompt_len:].tolist()
        for stop in (EOS_ID, PAD_ID):
            if stop in cont: cont = cont[:cont.index(stop)]; break
        rewards.append(count_vowels_in_ids(cont))   # the verifier
    return torch.tensor(rewards, dtype=torch.float, device=full_ids.device)

# Everything else stays the same.
rlvr_policy, rlvr_rewards = train_grpo(reward_fn_verifier, label="GRPO+RLVR")
```

The same machinery (rollouts, KL penalty, clipped surrogate, group advantage) plugs into a different reward function.

## §3 Why it is the 2026 hype

1. **Reasoning models broke through with it.** DeepSeek-R1 showed GRPO + RLVR on a base model produces o1-competitive reasoning. No RM, no human preferences.
2. **It scales with compute, not labels.** As long as you have verifiable problems, the training signal is effectively unlimited.
3. **It elicits emergent reasoning.** Backtracking, self-correction, verification — behaviors never explicitly taught. The model invents them because they earn reward.
4. **The "verifier problem" is being solved.** RLVRR (ICLR 2026) extends verifiable signals to open-ended tasks via high-quality reference outputs, reportedly beating SFT trained on 10× more data and generalizing across domains.

!!! abstract "Is it 'better' than RLHF?"
    For verifiable tasks: **yes, decisively** (cost, scale, robustness, capability). For genuinely subjective tasks (writing, dialogue, safety): **not yet**; you still need a preference signal. The frontier reality is "RLVR wherever you can verify, RLHF/DPO for the rest."

## §4 Verifiable domains in practice

What counts as a verifiable reward?

- **Math.** Parse the model's answer (regex or symbolic), compare to ground truth. Tolerances for floating point.
- **Code.** Run unit tests, count passing tests. Bonus: weight tests by difficulty.
- **JSON / structured output.** Validate against a schema; binary reward for valid + matching expected fields.
- **Tool calls.** Did the tool execute? Did it return the expected result?
- **Theorem proving.** Lean / Coq proof checker as the verifier.
- **Format compliance.** Regex match against `<think>...</think><answer>...</answer>` structure.

The common thread: a deterministic function from response text to a scalar.

## §5 Failure modes

### 5.1 Verifier noise destroys training

One analysis found **more than 38% of responses flagged incorrect by a rule-based system were actually correct**. False negatives starve gradients; false positives reward hackable patterns. The verifier is a piece of code; like any code, it has bugs. Garbage in, garbage policy.

### 5.2 Sparse binary rewards are unstable

All-correct or all-wrong batches kill signal (advantage collapse, see [GRPO §5](grpo.md)). Mitigations: curriculum the prompts so the group sees a mix of difficulties, or use ranked verifiers that produce partial credit.

### 5.3 Process vs outcome

Checking only final answers lets models reach right answers via bad reasoning. The motivation for **Process Reward Models** (which are themselves learned RMs and reintroduce the supervision problem) or for verifying reasoning steps as well as final outputs.

### 5.4 Reward hacking still exists

The verifier is not ground truth; it is your code. Models can learn to hardcode visible test cases, exploit equivalence-checker quirks, or produce outputs that pass the format check while saying nothing. The hacking just moves from the RM's blind spots to the verifier's bugs.

## §6 Where it fits in the 2026 pipeline

For a typical frontier model in 2026:

1. Pretrain (knowledge).
2. Mid-training (annealing on high-quality data).
3. SFT (instruction following + format + tool use).
4. RFT (rejection sampling on verifier-passing traces).
5. **RLVR via GRPO** for reasoning, math, code, agents.
6. RLHF or DPO or RLAIF for safety, style, helpfulness, refusals.

RLVR does not replace any other stage. It is a powerful new tool inserted between SFT and final alignment. See [The GRPO hype, decoded](grpo-hype-decoded.md) for the misconception that "GRPO replaces SFT."

## Interview Questions

**Q1: One-line RLHF vs RLVR?**

RLHF rewards via a learned model of human preference; RLVR rewards via a programmatic verifier of correctness. Same GRPO or PPO machinery underneath; only the reward function differs.

**Q2: Why did RLVR break through in 2025-2026?**

Because reasoning models broke through. Math correctness and code-passes-tests are clean verifiable signals, and they happen to be exactly the domains where you most need reasoning. Once DeepSeek-R1 showed that pure RL on a base model with verifier rewards plus GRPO elicits chain-of-thought, self-verification, and reflection that no SFT data ever taught, every open-source lab pivoted.

**Q3: When does RLVR not work?**

Subjective tasks: creative writing, dialogue style, safety judgments, helpfulness tradeoffs. No deterministic verifier exists for "is this answer tactful?" so you still need RLHF or DPO. The frontier rule of thumb is "RLVR where you can verify, RLHF for the rest."

**Q4: What is the biggest practical failure mode of RLVR?**

Verifier noise. In one analysis more than 38% of responses flagged incorrect were actually correct, and the same kind of false-positive rate exists for hackable verifiers. Garbage verifier → garbage policy. Investing in verifier quality (multiple checks, partial credit, tolerance handling) often matters more than tweaking GRPO hyperparameters.

**Q5: Why is the verifier sometimes wrong but the reward model usually less so?**

Reward models smooth over noisy human labels by learning a continuous function. A verifier is a piece of code, and code has bugs — strict string matching misses semantically equivalent answers, regex misses formatting variations, equivalence checks can be fooled. The verifier feels "objective" but is only as good as its implementation, and small bugs translate directly into systematic training noise.

**Q6 (Trap): If RLVR scales with compute not labels, can we skip SFT entirely?**

Sometimes. DeepSeek-R1-Zero did exactly that: pure RL on the base model with verifier rewards, no SFT. But the production R1 reintroduced SFT for readability, format, and a stable starting policy. RLVR works on a strong base but produces ugly outputs (endless repetition, language mixing, no readability); SFT before RLVR gives the RL policy a sane starting point. See [The GRPO hype, decoded](grpo-hype-decoded.md).
