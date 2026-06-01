# Foundations: Transformer Architecture

The bottom layer of the stack. Every page in this section breaks down one of the pieces that make a modern transformer work, from the attention operation itself up through the design choices that distinguish a 2017 paper transformer from a 2026 production LLM.

!!! tip "Rapid Recall"
    A transformer takes a sequence of tokens and re-mixes them so every token gets enriched with information from every other token via attention. Stack 32 to 80 of these blocks with residuals, RMSNorm, SwiGLU, and RoPE, and you get GPT-style language models. The two non-obvious wins are pre-norm residual connections (so gradients flow at depth) and rotary position embeddings (so relative position falls out of the attention dot product). Everything else is plumbing around `softmax(QKᵀ/√dk)·V`.

## What to read in what order

- Start with [Attention](attention.md) to get the soft-retrieval framing of Q, K, V right.
- [Multi-head and softmax](multi-head-and-softmax.md) explains why we split heads and what the softmax actually buys.
- [Normalization](normalization.md) and [Block and residual](block-and-residual.md) cover the engineering that makes deep stacks trainable.
- [Positional encodings](positional-encodings.md) tracks the sinusoidal → RoPE → ALiBi arc.
- [Failure modes](failure-modes.md) is the catalog of how all of this breaks at scale (vanishing gradients, softmax dilution, attention sinks).
- [Architecture families](architecture-families.md) is the ViT / BERT / CLIP / Whisper map plus the bi-encoder vs cross-encoder pattern.

## The five things to never forget

1. **Attention = soft dictionary lookup.** `softmax(QKᵀ/√d_k)·V`. Q, K, V are learned projections playing retrieval roles; the √d_k normalizes dot-product variance.
2. **O(N²) is a memory problem first.** Cost is N²·d (inner dim is d, not N, so not N³). Flash Attention tiles to kill the N² materialization.
3. **Pre-norm + residual = trainable depth.** The residual stream is a clean gradient highway; norm sits on the branch, not the highway.
4. **RoPE won positional encoding** by rotating Q, K so relative position emerges from the angle difference, and its clean math enables 4K → 128K context extension.
5. **Pick architecture by I/O shape.** Encoder-only for understand-then-label, decoder-only for generate or general purpose, encoder-decoder for source-to-different-target. Compose them for real systems (bi-encoder retrieve → cross-encoder rerank → LLM generate).

For the runnable, line-by-line version of all of this, see [Pretraining on TinyStories](../build-from-scratch/pretraining-tinystories.md) in the Build From Scratch section.
