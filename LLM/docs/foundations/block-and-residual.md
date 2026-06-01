# Block Structure: Pre-Norm and the Residual Highway

The modern transformer block puts norm on the *branch*, not on the residual highway. That tiny rearrangement is the difference between training 12 layers fine and training 80 layers without warmup. Pair it with weight tying and scaled init and you have everything you need to instantiate a working LLM.

!!! tip "Rapid Recall"
    The modern block is `x → LN → Attention → +x → LN → FFN → +x → out`. Pre-norm keeps LayerNorm on the branch, not on the residual highway, so gradients flow back through pure addition. Post-norm (original 2017) renormalizes the highway every layer and dies past ~20 layers without elaborate warmup. Modern blocks also tie the input embedding to the output projection (`lm_head.weight = tok_emb.weight`) for a 3M+ parameter saving and free regularization, and scale residual sublayer output projections by `1/√(2L)` so the residual stream variance does not accumulate with depth.

## §1 The block in one line

```
x → LN → Attention → +x → LN → FFN → +x → out
```

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 210" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace" font-size="12">
<line x1="40" y1="40" x2="660" y2="40" stroke="#b5462a" stroke-width="4"/>
<text x="350" y="28" text-anchor="middle" fill="#b5462a" font-weight="700">residual stream — clean addition, gradient flows straight back</text>
<line x1="160" y1="40" x2="160" y2="95" stroke="#1a1410" stroke-width="1.5"/>
<rect x="110" y="95" width="100" height="34" rx="4" fill="#1f5e5b" stroke="#1a1410" stroke-width="1.3"/>
<text x="160" y="117" text-anchor="middle" fill="#f4ede2" font-size="11">LN→Attn</text>
<line x1="160" y1="129" x2="160" y2="155" stroke="#1a1410" stroke-width="1.5"/>
<circle cx="160" cy="40" r="9" fill="#f4ede2" stroke="#b5462a" stroke-width="2"/>
<text x="160" y="44" text-anchor="middle" font-size="12" font-weight="700">+</text>
<line x1="160" y1="155" x2="270" y2="155" stroke="#1a1410" stroke-width="1.5"/>
<line x1="270" y1="155" x2="270" y2="49" stroke="#1a1410" stroke-width="1.5"/>
<line x1="460" y1="40" x2="460" y2="95" stroke="#1a1410" stroke-width="1.5"/>
<rect x="410" y="95" width="100" height="34" rx="4" fill="#c08a2d" stroke="#1a1410" stroke-width="1.3"/>
<text x="460" y="117" text-anchor="middle" fill="#1a1410" font-size="11">LN→FFN</text>
<line x1="460" y1="129" x2="460" y2="155" stroke="#1a1410" stroke-width="1.5"/>
<circle cx="460" cy="40" r="9" fill="#f4ede2" stroke="#b5462a" stroke-width="2"/>
<text x="460" y="44" text-anchor="middle" font-size="12" font-weight="700">+</text>
<line x1="460" y1="155" x2="570" y2="155" stroke="#1a1410" stroke-width="1.5"/>
<line x1="570" y1="155" x2="570" y2="49" stroke="#1a1410" stroke-width="1.5"/>
<text x="40" y="195" font-size="11" fill="#3a2e22">Pre-norm: LN sits on the BRANCH, not the highway → gradients never pass through a norm on the way back → 80-layer models train without warmup.</text>
</svg>
<figcaption>The residual stream as a "gradient highway." Pre-norm keeps norms off the highway so gradients flow back through pure addition.</figcaption>
</figure>

## §2 Pre-norm vs post-norm

The original 2017 Transformer used post-norm: `x = LN(x + sublayer(x))`. Modern LLMs use pre-norm: `x = x + sublayer(LN(x))`.

Pre-norm keeps the residual stream clean (no normalization on the skip path), which means gradients flow straight through without passing through layernorm. This is critical for very deep models. Post-norm transformers past ~20 layers are almost impossible to train without warmup tricks. Every modern LLM (LLaMA, GPT-4, Claude) uses pre-norm.

## §3 Two sub-blocks per layer

Attention (cross-token mixing) and FFN (per-token transform) are fundamentally different ops. Each gets its own modular "read highway → compute → write back" unit.

## §4 Weight tying

The input embedding (`[vocab_size, n_embd]`) and the output projection to logits (`[n_embd, vocab_size]`) share weights. This:

- Saves `vocab_size × n_embd` parameters. For a 30M-param model with vocab 8K and d 384, that is ~3M params, or 10% of the total.
- Acts as regularization.
- Is theoretically justified: the model is using the same "semantic space" to read and write tokens.
- Is standard since GPT-2.

```python
self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
self.lm_head.weight = self.tok_emb.weight    # tie input embedding to output projection
```

## §5 Scaled initialization

Residual connections add up over layers. If each sublayer has unit-variance output, after `n_layer` layers the residual stream has variance `n_layer`. To prevent this, the GPT-2 paper scales the *output projection* of each sublayer by `1/sqrt(2 * n_layer)`. We follow that.

```python
for name, p in self.named_parameters():
    if name.endswith('proj.weight') or name.endswith('w_down.weight'):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
```

## §6 Putting one block together

```python
class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm SwiGLU MLP, with residuals."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp   = SwiGLU(cfg)

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x
```

For the runnable full-model wrapper (including embeddings, positional cache, final norm, and the tied LM head), see [Pretraining on TinyStories §10](../build-from-scratch/pretraining-tinystories.md).

## Interview Questions

**Q1: What is the difference between pre-norm and post-norm, and why does everyone use pre-norm now?**

Post-norm (original transformer): `x + LN(Attn(x))`. Pre-norm (modern): `x + Attn(LN(x))`. Pre-norm keeps the residual stream clean (no normalization on the skip path), which means gradients flow straight through without passing through layernorm. This is critical for very deep models. Post-norm transformers past ~20 layers are almost impossible to train without warmup tricks. Every modern LLM uses pre-norm.

**Q2: What is weight tying and what does it buy you?**

The input embedding `[V, d]` and the output projection `[d, V]` share the same weight matrix (`lm_head.weight = tok_emb.weight`). Saves vocab × d_model parameters (often ~10% of a small model's total), acts as regularization, and is theoretically justified because the model reads and writes tokens through the same semantic space. Standard since GPT-2.

**Q3: Why scale residual output projections by 1/sqrt(2L)?**

Each residual sublayer adds its output to the stream. If outputs are unit-variance and you stack L layers, the residual stream variance grows like L. Scaling each sublayer's output projection by `1/sqrt(2L)` keeps the stream variance roughly constant with depth, which is what makes deep stacks trainable without warmup hacks.

**Q4: Walk me through one transformer block, from input to output.**

Input x of shape (batch, seq, d). RMSNorm it. Pass through multi-head attention: project to Q, K, V per head; compute scaled dot-product attention with causal mask; apply RoPE to Q and K before the dot product; concatenate heads; project back to d. Add the input back (residual). RMSNorm again. Pass through SwiGLU FFN (d → 2.67d gated → d). Add residual again. That is one block. Stack 32 to 80 of these, with embeddings at the bottom and a linear plus softmax head at the top.
