# Decoding and Speculative Decoding

Autoregressive generation is one token at a time, and decode is memory-bandwidth-bound (see [Prefill vs decode](../inference-arch/prefill-vs-decode.md)). Speculative decoding asks: what if we could generate multiple tokens per forward pass? The answer is yes, if we are willing to sometimes throw results away — and the beautiful part is the output distribution stays mathematically identical to the target model's. This page covers sampling basics (top-k, top-p, temperature) and the four speculative variants you need to name.

!!! tip "Rapid Recall"
    Sampling: **greedy** (top-1, deterministic, repetitive), **top-k** (sample from k most probable), **top-p / nucleus** (sample from smallest set covering p=0.9 probability — the chat default), **temperature** (divides logits before softmax; low T = deterministic, high T = chaotic). **Speculative decoding** uses a small fast draft model to predict K tokens, then the big target model verifies all K in a single parallel forward pass. **It is mathematically lossless** because the acceptance criterion is rejection sampling: accept token x with probability `min(1, q(x)/p(x))` where q is target and p is draft. **EAGLE-3** is the 2026 production default (~3× speedup, on by default in vLLM/SGLang/TensorRT-LLM). Medusa needs no separate model (extra heads on target). PLD (prompt lookup) just reuses n-grams from the prompt — free for code completion and summarization.

## §1 Basic decoding strategies

- **Greedy decoding.** Always pick top-1 token. Deterministic, can be repetitive.
- **Top-k sampling.** Sample from top-k tokens by probability. Filters out tail nonsense.
- **Top-p (nucleus) sampling.** Sample from the smallest set covering cumulative probability p (e.g., p = 0.9). Standard for chat. Adapts to the distribution: if the model is confident, the set is small; if not, it expands.
- **Temperature.** Divides logits before softmax. Low T (< 1) sharpens (more deterministic, more repetitive). High T (> 1) flattens (more random, more diverse).
- **Beam search.** Maintain K candidate sequences, pick best. **Almost never used for chat LLMs** — kills diversity, encourages bland safe outputs. Still used for machine translation.
- **Constrained decoding.** Force outputs to match a grammar (JSON schema, regex). Critical for function calling. Libraries: `outlines`, `guidance`, `lm-format-enforcer`, `llguidance`.

Typical chat default: `temperature = 0.7`, `top_p = 0.9`, no top-k.

## §2 Why generation is slow

Inference has two phases:

1. **Prefill** — process the input prompt all at once. Compute-bound — one big matmul over all prompt tokens in parallel. Fast per token.
2. **Decode** — generate output tokens one at a time. Each token requires a full forward pass through the model. Memory-bound — the GPU cannot utilize compute because it is waiting on weight loads.

Speculative decoding attacks the decode phase.

## §3 The core idea of speculative decoding

**What if we could generate multiple tokens per forward pass?** We can, if we are willing to sometimes throw results away.

The trick: use a **small fast draft model** to predict N tokens ahead, then use the **big target model** to verify all N in a single parallel forward pass. Because the target model's forward pass processes all positions in parallel anyway, verifying N tokens is almost the same cost as generating 1 token.

```
Without speculation:
[Big model]→ t1 → [Big model]→ t2 → [Big model]→ t3 → [Big model]→ t4
(4 full forward passes of the big model)

With speculation (K=4 draft tokens):
[Small draft]→ t1', t2', t3', t4' (fast, sequential but cheap)
[Big model]→ verify all 4 in ONE forward pass
  → if all match: accept all 4 (4× speedup)
  → if first 2 match, 3rd fails: accept 2, try again from token 3
```

## §4 Why output distribution does NOT change

This is the beautiful part. Speculative decoding is **mathematically lossless** — the output distribution is **identical** to the target model generating tokens one at a time.

The mechanism: **rejection sampling**. For each draft token `t'`:

- Let `p` = target's probability of `t'`.
- Let `q` = draft's probability of `t'`.
- Accept `t'` with probability `min(1, p/q)`.
- If rejected, sample a new token from the adjusted distribution `(p - q)_+` (normalized).

This is provably equivalent to direct sampling from the target. So you get speedup **without quality degradation**. This is why spec decoding is default-on in vLLM, SGLang, and TensorRT-LLM.

## §5 The four speculative variants

### 5.1 Vanilla speculative decoding (Leviathan et al., 2023)

Separate draft model, separate target model. Draft is typically ~10× smaller (Llama-3.2-1B as draft for Llama-3.3-70B target). Works, but adds another model and memory pressure.

### 5.2 Medusa (Cai et al., 2024)

**No separate draft model.** Add multiple extra output heads to the target model, each trained to predict tokens 2, 3, 4, ... ahead. Simpler deployment (one model, one checkpoint), but lower acceptance rate than a dedicated draft.

### 5.3 EAGLE-3 (2025, current SOTA)

Uses a **lightweight autoregressive head** attached to the target model's internal hidden states, instead of predicting from raw tokens. Because it gets to "peek at" the target's internal features, acceptance rates are much higher — 2 to 6× speedup, often 3× in production. **The 2026 production default.** Supported in vLLM, SGLang, TensorRT-LLM.

### 5.4 N-gram / PLD (Prompt Lookup Decoding)

The cheapest possible draft: look for repeated n-grams in the prompt and propose them as draft tokens. Works great for tasks where outputs reuse prompt content (code completion, summarization, doc Q&A). No draft model at all. Typical speedup: 1.5 to 2× for those tasks, 0 otherwise.

## §6 Acceptance rate math

Speedup depends on the **acceptance rate α** — the fraction of draft tokens that get accepted by the target. Expected tokens per round with K draft tokens:

\[ E[\text{accepted}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} \]

| α | K = 4 | Effective speedup |
|---|---|---|
| 0.4 | ~1.5 tokens | barely worth it |
| 0.6 | ~2.2 tokens | ~2× |
| 0.8 | ~3.4 tokens | ~3.4× |
| 0.9 | ~4.1 tokens | ~4× |

**Tasks where acceptance is high** (good fit): code completion, structured output, templated responses, RAG answers.

**Tasks where acceptance is low** (bad fit): creative writing, open-ended chat, highly fine-tuned domain-specific models.

## §7 The implementation sketch

```python
def speculative_decode(draft, target, prompt, K=4, max_tokens=100):
    """Speculative decoding with rejection sampling."""
    tokens = prompt.clone()
    while len(tokens) - len(prompt) < max_tokens:
        # Step 1: Draft generates K candidate tokens.
        draft_tokens, draft_probs = [], []
        for _ in range(K):
            logits = draft(tokens)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            draft_tokens.append(token)
            draft_probs.append(probs[0, token.item()])
            tokens = torch.cat([tokens, token], dim=-1)

        # Step 2: Target scores all K+1 positions in ONE parallel forward pass.
        all_logits = target(tokens)

        # Step 3: Accept/reject each draft via rejection sampling.
        for i in range(K):
            q = torch.softmax(all_logits[:, -(K+1-i), :], dim=-1)
            p = draft_probs[i]
            acceptance_prob = min(1.0, (q[0, draft_tokens[i].item()] / p).item())
            if torch.rand(1).item() >= acceptance_prob:
                # Reject: sample from corrected distribution (p - q)_+, discard rest
                correction = torch.clamp(q - p, min=0)
                sampled = torch.multinomial(correction / correction.sum(), 1)
                tokens = torch.cat([tokens[:-(K-i)], sampled], dim=-1)
                break
    return tokens
```

In production you let vLLM or SGLang handle this; the snippet is for understanding.

## §8 When speculative decoding hurts

!!! warning "Trap"
    You enable spec decoding and throughput goes down. Two likely causes. First, **acceptance rate too low** — if α < ~0.4, you waste cycles on rejected drafts that more than offset the parallelism gain. Happens when the draft model is misaligned with the target (e.g., generic Llama-3.2 draft but heavily fine-tuned domain target). Second, **batch size already high** — spec decoding's benefit comes from the decode phase being memory-bound. At large batch sizes, decode becomes compute-bound (weights amortize across batch), and spec decoding adds overhead without speedup. Spec decoding helps small/medium-batch latency-sensitive workloads, not already-saturated high-throughput ones.

## Interview Questions

**Q1: Why is speculative decoding guaranteed to not change the output distribution?**

The acceptance criterion is rejection sampling: accept draft token x with probability `min(1, q(x)/p(x))` where p is the draft distribution and q is the verifier's distribution. This is exactly how rejection sampling works: the accepted samples follow q's distribution regardless of p's proposals. Mathematically equivalent to sampling directly from q, just faster. The draft model is purely a speed optimization; it cannot affect what the final distribution looks like.

**Q2: Walk me through the four speculative decoding variants.**

Vanilla (Leviathan 2023): separate draft and target models, draft ~10× smaller. Medusa: no separate draft — add multiple output heads to the target, each predicts 2, 3, 4 steps ahead. Lower acceptance but one model. EAGLE-3 (2025 SOTA): a small autoregressive head attached to target hidden states; gets to peek at internal features, much higher acceptance, 2 to 6× speedup. PLD (prompt lookup): the cheapest — propose n-grams from the prompt as draft tokens. Free for code completion, summarization, RAG; useless for creative writing.

**Q3: When does speculative decoding hurt throughput?**

Two scenarios. (1) Low acceptance rate: if α < ~0.4, rejected drafts waste compute that exceeds the parallelism gain. Common when the draft is misaligned with a heavily fine-tuned target. (2) Already-high batch size: spec decoding's benefit comes from breaking the memory-bound decode floor. At high batches, decode becomes compute-bound (weights amortize across many sequences), and spec adds overhead without speedup. Spec is for small/medium-batch latency-sensitive workloads.

**Q4: Why is EAGLE-3 better than vanilla speculative decoding?**

Vanilla uses a separate draft model that sees only tokens. EAGLE-3 attaches a small autoregressive head to the target's internal hidden states, so the draft has access to richer features than just the previous tokens. Higher predictive power → higher acceptance rate → larger effective speedup. Plus one less model to load and serve. The tradeoff: EAGLE-3 has to be trained for each target model; vanilla can use any compatible pair.

**Q5: Top-k vs top-p — when each?**

Top-k samples from the k most probable tokens regardless of distribution shape. Top-p (nucleus) samples from the smallest set covering cumulative probability p. Top-p is adaptive: in confident moments the set is tiny, in ambiguous moments it expands. For chat, top-p ≈ 0.9 with temperature 0.7 is the modern default. Top-k is mostly historical; you rarely want a fixed k across very different token distributions.

**Q6 (Trap): What is beam search and why do chat LLMs not use it?**

Beam search keeps the top-K candidate sequences at each step and picks the best at the end. Great for machine translation where there is one right answer. Disastrous for chat because it concentrates probability mass on bland safe sequences and crushes diversity — every output starts with "I think..." or "As an AI..." Chat needs sampling to get personality and creativity; top-p with reasonable temperature is the right call.
