# Failure Modes

Two specific ways transformers break at scale, with diagnostics and fixes for each. Every trainable architecture is a tightrope walk between gradient flow and softmax behavior; once you have seen these two failure patterns you will recognize them in any new architecture you read about.

!!! tip "Rapid Recall"
    Two failure modes worth knowing cold. First, gradient flow: each layer multiplies back-flowing gradients by ~its Jacobian; over 80 layers tiny deviations explode (1.1⁸⁰ ≈ huge) or vanish (0.9⁸⁰ ≈ zero), and you see loss spike then NaN. Fixes: pre-norm, residuals, depth-scaled init (`1/√(2L)`). Second, softmax dilution: at long N where no key matches the query, softmax must spread weight (it cannot output zero), so attention becomes a meaningless uniform average; the *fix the model learns on its own* is attention sinks (dumping excess weight on the BOS token), which is why StreamingLLM keeps the first ~4 tokens when sliding a context window.

## §1 Gradient flow through a deep stack

Each layer multiplies back-flowing gradients by ~its Jacobian. Compounded over 80 layers, tiny deviations from 1.0 explode or vanish.

- **Detection:** loss spikes (2.5 → 200) then NaN. Once any tensor is NaN it propagates everywhere.
- **Fixes:** pre-norm (clean residual highway), residual connections (the highway itself), depth-scaled init (`1/√(2L)` output-projection scaling so variance does not accumulate).

For the residual-highway diagram and the pre-norm vs post-norm rationale, see [Block and Residual](block-and-residual.md).

## §2 Softmax attention dilution at long N

Softmax must sum to 1; it can never say "nothing is relevant." Over 100K tokens where none match, it smears ~1/100K weight everywhere, giving a meaningless near-uniform average.

- **Detection:** attention entropy approaches `log(N)`.

### Attention sinks — the surprising fix the model invents on its own

Trained transformers spontaneously dump excess attention onto the first token(s). The BOS gets 30 to 70% weight even with zero semantic relevance. It is a *release valve* for "nothing to say."

**StreamingLLM exploits this:** never drop the first ~4 tokens when sliding a context window, or the model breaks.

**Differential Transformer** (newer) subtracts a noise-attention map from a signal map (like a differential amplifier) to cancel dilution at the source.

### Other fixes

- The original `√d_k` scaling (controls per-token sharpness; see [Attention](attention.md)).
- Temperature tuning on softmax.
- Sparse-activation variants (sparsemax, entmax) that can output exact zeros.

## §3 Companion table: assumptions and failure modes

| Assumption | What breaks if violated | Detection | Fix |
|---|---|---|---|
| Sequence fits in context | O(N²) attention blows up memory at long N | OOM at training, slow inference past trained length | Flash Attention, RoPE scaling (NTK, YaRN), sliding window, GQA + long-context training |
| Tokens have consistent semantics | Tokenizer splits rare words weirdly → weird embeddings | Check tokenizer output, inspect rare-token perplexity | BPE/tiktoken with larger vocab, byte-level fallback |
| Positional info is injected somehow | Without it, transformer is permutation-invariant (bag of words) | Shuffle input, same output = broken | Sinusoidal / RoPE / ALiBi |
| Gradient flows through deep stack | Vanishing/exploding grads in 80-layer model | Training loss NaN, spikes | Pre-norm, residuals, depth-scaled init |
| Softmax attention is well-calibrated | At very long N, attention gets diluted and focuses on nothing | Attention entropy too high, model ignores relevant tokens | Scale `sqrt(d_k)`, temperature tuning, attention sinks (first-token bias), Differential Transformer |

## Interview Questions

**Q1: What does it mean when training loss spikes and then NaNs?**

It is almost always a gradient-flow problem. Either an exploding gradient blew through the loss (per-layer multiplicative factor compounded over depth) or a denominator inside a norm or softmax went to zero. Once any tensor is NaN it propagates. Fixes: pre-norm, gradient clipping, depth-scaled init for residual output projections, and lowering the LR until the first few hundred steps stabilize.

**Q2: Why do trained transformers put 30 to 70% of attention weight on the BOS token?**

Softmax must sum to 1; it cannot say "nothing is relevant." When the model has no real signal to attend to (or has computed everything it needs), it dumps the excess on the BOS as a release valve. StreamingLLM exploits this: keep the first ~4 tokens when sliding a long context window or the model breaks.

**Q3: What is softmax dilution and how does Flash Attention or differential attention help?**

At very long sequence lengths, when no key strongly matches a query, softmax spreads weight ~uniformly across N positions and the resulting weighted average becomes meaningless. Flash Attention does not fix the dilution itself; it fixes the *memory cost* so long context is even feasible. Differential Transformer subtracts a noise attention map from a signal map (like a differential amplifier) to cancel the diluted background.

**Q4: How would you diagnose softmax dilution in a trained model?**

Look at attention entropy. If entropy approaches `log(N)` for many heads, those heads are essentially uniform and not doing useful work. Pair this with attention-pattern visualization: if a head puts most weight on BOS or spreads evenly, it is either an attention sink or diluted.
