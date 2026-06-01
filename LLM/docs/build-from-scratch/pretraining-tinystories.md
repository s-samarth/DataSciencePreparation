# Building a Decoder-Only LLM From Scratch on Colab

End-to-end pretraining of a ~30M-parameter GPT-style language model on TinyStories using a free Colab T4. Every line is yours: BPE tokenizer, packed dataloader, RoPE attention, RMSNorm, SwiGLU, weight tying, AdamW with 8-bit and paged variants, cosine LR with warmup, mixed precision, and Drive checkpointing.

!!! tip "Rapid Recall"
    Pretraining is just next-token cross-entropy at scale. The structural choices that matter are: BPE tokenizer (custom 8K vocab beats reusing GPT-2's 50K for a narrow domain), packed-sequences dataloader (concatenate every doc with EOT, sample random `block_size` windows, zero padding waste), RoPE positional encoding (relative position emerges from the dot product), RMSNorm + SwiGLU + pre-norm + weight tying (the modern block), AdamW with weight-decay on 2D tensors only, linear warmup → cosine decay LR, mixed precision (bf16 on Ampere+, fp16 on T4 for hardware speed). Validation loss starts near `ln(vocab) ≈ 9` and lands around 2.0 after a few thousand steps.

## §1 What we will build, end to end

1. Custom BPE tokenizer (trained on our data, not borrowed).
2. Streaming dataset + DataLoader with packed sequences.
3. Decoder-only Transformer (RMSNorm, RoPE or sinusoidal, SwiGLU, weight tying).
4. Training loop with AdamW, cosine LR schedule + linear warmup, gradient accumulation, gradient clipping, mixed precision.
5. Optional: 8-bit and paged optimizers via `bitsandbytes`.
6. Validation loss tracking + periodic text generation samples.
7. Checkpoint save/resume to Google Drive.
8. Inference: autoregressive sampling with temperature + top-k.

**Runtime expectations:** Setup ~10 min, BPE training ~5 min, tokenization ~10 min, model training ~2-4 hours for coherent output.

## §2 Config dataclass

Why a single config dataclass: every hyperparameter in one place. Easier to tweak, log, and checkpoint.

Why these specific dimensions: `n_embd=384`, `n_layer=6`, `n_head=6` gives ~30M params and fits T4 with batch=12 ctx=512 in mixed precision. `head_dim = n_embd / n_head = 64` is the standard GPT-2 head size and works cleanly with RoPE. `block_size=512` covers most TinyStories docs. `vocab_size=8192` from BPE keeps the embedding table small.

```python
@dataclass
class Config:
    # --- Model architecture ---
    vocab_size: int = 8192      # set after BPE training; placeholder
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384           # head_dim = 64
    block_size: int = 512
    dropout: float = 0.0
    pos_encoding: str = 'rope'  # 'rope' or 'sinusoidal'

    # --- Training ---
    batch_size: int = 12        # per-step micro-batch (fits T4 at ctx=512)
    grad_accum_steps: int = 4   # effective batch = 12 * 4 = 48
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4 # peak LR
    min_lr: float = 3e-5        # 10% of peak, standard
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95         # Karpathy uses 0.95, not 0.999, for LLM pretraining
    grad_clip: float = 1.0

    optimizer: str = 'adamw'    # 'adamw' | 'adamw8bit' | 'paged_adamw8bit'

    compile_model: bool = True  # torch.compile for ~20-30% speedup
    mixed_precision: str = 'bf16'  # 'bf16' | 'fp16' | 'fp32'
```

## §3 Custom BPE tokenizer

**Why custom (not tiktoken):** tiktoken's GPT-2 vocab is 50,257 tokens, overkill for TinyStories. A custom 8192-token vocab trained on our data is 6× smaller embedding table, better compression on our domain, and demonstrates you understand tokenization end-to-end.

**BPE algorithm in one paragraph:** start with characters as the base vocabulary. Count every adjacent pair. Merge the most frequent pair into a new token. Repeat until you hit target vocab size. Each merge becomes a rule. To encode new text, apply merges greedily.

**ByteLevel pre-tokenizer:** converts text to UTF-8 bytes first, then runs BPE on byte-level. Every possible input encodes safely (no UNK), handles emojis, and is what GPT-2 does.

**Special tokens:** `<|endoftext|>` marks story boundaries during training so the model learns documents are independent; `<|pad|>` is reserved but rarely used because we pack sequences.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)
tokenizer.train_from_iterator(bpe_corpus_iter(), trainer=trainer, length=N_BPE_SAMPLES)
```

## §4 Packed sequences + memory-mapped dataloader

**Why upfront tokenization (not on-the-fly):** tokenization is the slow part. Doing it inside `__getitem__` would bottleneck training on CPU. Tokenize once, save as a flat numpy array of token IDs, memory-map for fast random access.

**Why a single flat array (not per-document):** the standard "packed sequences" trick. Concatenate every story separated by `<|endoftext|>` into one giant token stream. Training samples random `block_size`-length windows from it. Means zero padding waste (every position contributes to loss), easy random sampling, and the model learns document boundaries from the EOT token.

**uint16 vs int32:** our vocab is 8192 which fits in 13 bits, so uint16 (max 65535) works and halves the disk/RAM cost.

```python
class TokenWindowDataset(Dataset):
    """Samples random fixed-length windows from a flat token array.
    Each example: x = tokens[i:i+block_size], y = tokens[i+1:i+block_size+1].
    """
    def __init__(self, bin_path, block_size, length=None):
        self.bin_path = bin_path
        self.block_size = block_size
        self.length = length or 100_000

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Re-open memmap per call (cheap, ensures fork-safety).
        data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')
        i = np.random.randint(0, len(data) - self.block_size - 1)
        x = torch.from_numpy(data[i:i+self.block_size].astype(np.int64))
        y = torch.from_numpy(data[i+1:i+1+self.block_size].astype(np.int64))
        return x, y
```

This IS teacher forcing — at every position, we feed the ground-truth previous token rather than the model's prediction. There is no separate flag; it is how causal LM training works.

## §5 Positional encodings: Sinusoidal AND RoPE

Why positional encoding exists at all: attention is permutation-invariant. Without position info, "dog bites man" and "man bites dog" look identical to the model.

**Two approaches, both implemented, swap via `cfg.pos_encoding`.**

**Sinusoidal** (Vaswani 2017): add a fixed sinusoidal pattern to token embeddings. Each position gets a unique sine/cosine fingerprint. Simple, parameter-free. But position info dilutes through layers and attention scores do not directly encode relative distance.

**RoPE** (Su et al. 2021, used by Llama, Qwen, Gemma): instead of adding to embeddings, *rotate* the query and key vectors by an angle proportional to position. The dot product `q · k` then naturally encodes relative position. Applied inside attention at every layer.

```python
def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device='cuda'):
    """Precompute cos/sin tables for RoPE. Done once at startup."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """x: [B, n_head, T, head_dim], cos/sin: [T, head_dim/2].
    Rotates each (x_even, x_odd) pair by angle theta_pos."""
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos = cos[:x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(-2)].unsqueeze(0).unsqueeze(0)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd  = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
```

In interviews: RoPE is the answer for "what positional encoding would you use today?" — dominant choice in 2024-2026 open LLMs.

## §6 Multi-head causal self-attention

**Why we use `F.scaled_dot_product_attention` (SDPA):** PyTorch's built-in SDPA automatically dispatches to FlashAttention on supported GPUs. T4 also gets memory-efficient kernels. Writing the math by hand is good for understanding, but for actual training you want the optimized kernel.

**Causal mask:** `is_causal=True` tells SDPA to apply the lower-triangular mask internally, preventing position `t` from attending to `t+1, t+2, …`.

**RoPE applied where:** after projection to Q and K, before the dot product. We do NOT rotate V (only the "what to attend to" signal is positionally aware).

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd
        self.use_rope = (cfg.pos_encoding == 'rope')

        # Single linear that produces Q, K, V concatenated.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape to [B, n_head, T, head_dim] for parallel multi-head attention.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        # SDPA is faster, more memory-efficient, and uses FlashAttention when available.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))
```

## §7 RMSNorm and SwiGLU MLP

**RMSNorm** (Zhang & Sennrich 2019, used by Llama 2/3): LayerNorm computes both mean and variance, then re-centers and re-scales. RMSNorm drops the mean (just re-scales by root-mean-square). Slightly faster, comparable quality.

**SwiGLU** (Shazeer 2020, used by Llama/PaLM): the "MLP" block. Original GPT uses `Linear → GELU → Linear`. SwiGLU uses a gated variant: `SwiGLU(x) = (Linear_a(x) * silu(Linear_b(x))) → Linear_out`. The element-wise gate lets the model selectively pass information.

**Param count math:** plain MLP at 4× expansion = `n_embd · 4·n_embd · 2 = 8 · n_embd²` params. SwiGLU at the *same* total params uses `(2/3) · 4 = 8/3` expansion so the three linears together match.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Match plain-MLP param count by using 8/3 expansion instead of 4.
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)   # round to multiple of 64 for GPU efficiency
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
```

## §8 Transformer block and full GPT model

**Pre-norm vs post-norm:** original used post-norm. Modern LLMs use pre-norm: `x = x + sublayer(LN(x))`. Pre-norm gives more stable training at depth — the residual path is "clean" and gradients flow through it directly.

**Weight tying:** the input embedding `[vocab_size, n_embd]` and the output projection `[n_embd, vocab_size]` share weights. Saves ~3M params for a 30M model, acts as regularization, standard since GPT-2.

**Scaled initialization:** residual connections add up over layers. To prevent variance from accumulating, scale the *output projection* of each sublayer by `1/sqrt(2 * n_layer)`.

```python
class Block(nn.Module):
    """Pre-norm attention + pre-norm SwiGLU MLP, with residuals."""
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


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_rope = (cfg.pos_encoding == 'rope')

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if not self.use_rope:
            self.pos_emb = SinusoidalPositionalEmbedding(cfg.n_embd, max_len=cfg.block_size)
        else:
            head_dim = cfg.n_embd // cfg.n_head
            cos, sin = precompute_rope_cache(head_dim, cfg.block_size, device='cpu')
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_final = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: input embedding == output projection.
        self.lm_head.weight = self.tok_emb.weight

        # Init + scaled residual projection init.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w_down.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if not self.use_rope: x = self.pos_emb(x)
        x = self.drop(x)
        rope_cos = self.rope_cos if self.use_rope else None
        rope_sin = self.rope_sin if self.use_rope else None
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)
        logits = self.lm_head(self.norm_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling. NO teacher forcing here — model sees its own outputs."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

## §9 Optimizer: AdamW with proper param groups

**Critical detail many people miss:** weight decay should NOT apply to:

- Bias terms.
- LayerNorm/RMSNorm weights.
- Embeddings (debatable, but Karpathy's nanoGPT decays them; we follow).

Split parameters into two groups: decay (2D+ tensors like Linear weights) vs no-decay (1D tensors like norms, biases).

### Three optimizer variants

1. **`adamw`** (default): standard `torch.optim.AdamW`. Memory: ~8 bytes/param for optimizer state (fp32 momentum + variance).
2. **`adamw8bit`**: `bnb.optim.AdamW8bit`. Quantizes optimizer state to int8 (~2 bytes/param). Numerical quality preserved via block-wise quantization.
3. **`paged_adamw8bit`**: same plus CPU paging under memory pressure. Useful for bigger models on a small GPU.

```python
def get_optimizer(model, cfg):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    params = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in params.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in params.items() if p.dim() <  2]

    optim_groups = [
        {'params': decay_params,   'weight_decay': cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(optim_groups, lr=cfg.learning_rate,
                                  betas=(cfg.beta1, cfg.beta2), fused=(DEVICE == 'cuda'))
    elif cfg.optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    elif cfg.optimizer == 'paged_adamw8bit':
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
```

## §10 LR schedule: linear warmup → cosine decay

The standard recipe since GPT-3. Steps 0 → `warmup_iters` linearly ramp from 0 to peak LR. Steps `warmup_iters` → `max_iters` cosine decay from peak LR to `min_lr`.

**Why warmup:** Adam's variance estimate is unreliable in the first few steps (small sample size). Warmup prevents catastrophic early updates. Without it, you sometimes see a loss spike the model never recovers from.

```python
def get_lr(step, cfg):
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.warmup_iters
    if step >= cfg.max_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)
```

## §11 Mixed precision

**Mixed precision in one paragraph:** store master weights in fp32. Run forward + backward in bf16 (or fp16). Cast loss/gradients back to fp32 for optimizer update. Result: ~2× speedup, ~half memory, near-zero quality loss.

**bf16 vs fp16 on T4:** bf16 has the same exponent range as fp32, so no `GradScaler` needed. But **T4 is Turing arch and bf16 is software-emulated on it** — works correctly but no perf benefit. fp16 has narrower range and needs `GradScaler` to prevent gradient underflow, but it has **native hardware support on T4** — this is the fast path. Use `cfg.mixed_precision = 'fp16'` for max T4 speed.

## §12 The training loop

Every concept in one loop: LR schedule, gradient accumulation, mixed precision, gradient clipping, teacher forcing (built into causal LM), periodic eval + sample generation, checkpoints.

```python
for step in range(start_step, cfg.max_iters):
    # 1. Set LR for this step.
    lr = get_lr(step, cfg)
    for pg in optimizer.param_groups: pg['lr'] = lr

    # 2. Periodic eval BEFORE training step.
    if step % cfg.eval_interval == 0:
        val_loss = estimate_loss(model, val_loader, max_batches=cfg.eval_iters)
        # ... generate sample, save checkpoint ...

    # 3. Gradient accumulation loop.
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(cfg.grad_accum_steps):
        x, y = next(train_iter)
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps   # so sum matches a true bigger batch
        if scaler is not None: scaler.scale(loss).backward()
        else: loss.backward()

    # 4. Gradient clipping (after backward, before step).
    if scaler is not None: scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # 5. Optimizer step.
    if scaler is not None:
        scaler.step(optimizer); scaler.update()
    else:
        optimizer.step()
```

### Reading the loss numbers

For TinyStories with 8K vocab:

- Start: ~9.0 (random init, ≈ `ln(vocab)`).
- After 500 steps: ~4.5-5.0 (learned token frequencies).
- After 2000 steps: ~3.0-3.5 (real structure emerging).
- After 5000 steps: ~2.0-2.5 (coherent short stories).

## §13 Stretch goals

Things to try once the base notebook works:

1. **Scale up**: `n_layer=8, n_embd=512, block_size=1024` → ~80M params. Train longer. Switch to FineWeb-Edu sample-10BT.
2. **Compare positional encodings**: train one with `pos_encoding='sinusoidal'`, one with `'rope'`. Plot val loss.
3. **Compare optimizers**: train with `adamw` vs `adamw8bit`. Measure peak VRAM (`torch.cuda.max_memory_allocated()`).
4. **Fine-tune on a downstream task**: SFT on a small instruction dataset. See [SFT Walkthrough](sft-walkthrough.md).
5. **Implement KV cache for inference**: the current `generate()` recomputes attention over the whole context each token. KV caching makes inference O(n) instead of O(n²).

For interview prep: be ready to explain every block, every flag, and every hyperparameter choice. The value is in being able to defend the design, not in the model artifact itself.
