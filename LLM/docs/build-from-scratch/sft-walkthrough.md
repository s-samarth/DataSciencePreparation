# SFT From Scratch — Pure PyTorch on Colab

SFT done two ways so you internalize what changes with model size: full SFT on Qwen2.5-0.5B (494M, every weight trainable, fp16 on T4), and LoRA SFT on TinyLlama-1.1B (1.1B, base frozen, ~0.5% of params trainable). One shared training loop. Dataset: Dolly-15K (15K human-written instruction/response pairs).

!!! tip "Rapid Recall"
    SFT is the same next-token cross-entropy as pretraining, but with three structural changes: (1) the data is instruction-response pairs wrapped in the model's chat template (use `tokenizer.apply_chat_template`, never hand-roll), (2) the loss is **masked on the prompt** so the model only learns to predict the assistant's tokens (`labels[i] = -100` for user/system/padding), and (3) the LR is 10-100x smaller than pretraining (1e-5 for full SFT, 2e-4 for LoRA because adapters need to move more aggressively). LoRA freezes the base model and trains tiny low-rank adapters injected into Q, K, V, O — same training loop, just `requires_grad=True` only on the adapters.

## §1 The five SFT-specific concepts

1. **Chat template formatting** (instruction → model's expected prompt format).
2. **Loss masking on prompt tokens** — the single most important SFT concept.
3. Padding + attention masks for variable-length sequences.
4. LoRA injection (Part B).
5. Generation-based qualitative eval (perplexity alone is not enough).

## §2 Load and inspect Dolly-15K

Why dolly-15k specifically:

- **Human-written** (Databricks employees), not GPT-generated → no model collapse worries.
- **Diverse categories** → model learns multiple task types.
- **15K is the sweet spot** → big enough to teach generalization, small enough to train fast on Colab.
- **Commercial license** (CC BY-SA 3.0) → safe for portfolio/demo work.

```python
raw_ds = load_dataset("databricks/databricks-dolly-15k", split='train')
shuffled = raw_ds.shuffle(seed=SEED)
val_raw = shuffled.select(range(500))
train_raw = shuffled.select(range(500, len(shuffled)))
```

## §3 Chat template formatting — the first thing people get wrong

Pretrained instruct models learn a SPECIFIC prompt format. Qwen uses ChatML-style with `<|im_start|>` / `<|im_end|>` tags. Llama uses `[INST]...[/INST]`. TinyLlama uses ChatML. **If you SFT in the wrong format, you fight the model's existing knowledge** instead of building on it — the model has to unlearn its prompt format before it can learn your task.

**The fix:** use the model's built-in chat template via `tokenizer.apply_chat_template(messages, ...)`. This converts a list of `{"role": ..., "content": ...}` dicts into the exact string format the model was trained on.

```python
def example_to_messages(ex):
    """Convert one dolly row into the standard chat messages format."""
    system = "You are a helpful AI assistant. Answer questions accurately and concisely."
    if ex['context']:
        user = f"Context:\n{ex['context']}\n\nQuestion: {ex['instruction']}"
    else:
        user = ex['instruction']
    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": ex['response']},
    ]

msgs = example_to_messages(train_raw[0])
formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
```

Output (Qwen format):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

The model sees this entire string during training, but **we only compute loss on the assistant portion**. That is the next section.

## §4 Loss masking — the single most important SFT concept

**The core idea:** during SFT, the model sees `[system + user + assistant]` tokens. But we only want to TRAIN it to produce the assistant tokens. Computing loss on the user tokens would teach the model to predict the user's question, which is wrong — at inference time the user provides their own question, the model does not generate it.

**How:** PyTorch's `cross_entropy` accepts an `ignore_index` (default `-100`). Any target position equal to that index contributes zero to the loss. So we set:

- `labels[i] = -100` for system + user tokens (and padding).
- `labels[i] = input_ids[i]` for assistant tokens.

**The tricky bit — finding the assistant span:** tokenize the full conversation, then re-tokenize just the prefix (system + user portion with `add_generation_prompt=True`) to find where the assistant tokens begin. Mask everything up to that index.

**Two common bugs:**

1. **Off-by-one on the boundary.** Some templates put the role tag (`<|im_start|>assistant\n`) right before the assistant content. **We mask it.** The model should learn to produce the content *after* the role tag, not the role tag itself.
2. **Forgetting to mask padding tokens.** PAD tokens have `input_ids[i] = pad_id`. Their labels MUST be `-100`, otherwise we train the model to predict padding.

```python
def tokenize_and_mask(example, tokenizer, max_length=1024):
    """Returns input_ids, attention_mask, labels (with prompt tokens masked to -100)."""
    msgs = example_to_messages(example)

    full_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    # Prompt only (system + user + assistant role header but no answer):
    prompt_only = msgs[:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100   # mask everything in the prompt

    return {'input_ids': full_ids, 'labels': labels, 'length': len(full_ids)}
```

**Verification:** print one example showing exactly which tokens are kept vs masked. If you see user-text tokens in the kept positions, the masking is broken.

## §5 Padding collator (dynamic, not fixed)

**Why dynamic padding to the longest example in the batch:** if most examples in a batch are 200 tokens long and one is 800, padding to 1024 wastes 80% of compute on padding. Dynamic padding cuts wasted compute roughly in half.

```python
def make_collator(pad_token_id):
    def collate(batch):
        max_len = max(ex['length'] for ex in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            L = ex['length']
            input_ids[i, :L] = torch.tensor(ex['input_ids'], dtype=torch.long)
            labels[i, :L]    = torch.tensor(ex['labels'],    dtype=torch.long)
            attn_mask[i, :L] = 1
        return {'input_ids': input_ids, 'attention_mask': attn_mask, 'labels': labels}
    return collate
```

## §6 The shared training loop

Full SFT and LoRA SFT differ in *which parameters get trained*, not in HOW the training loop runs. Write the loop once and pass in different models.

**Key training mechanics:**

- Mixed precision (fp16 with GradScaler).
- Gradient clipping at 1.0.
- Cosine LR schedule with warmup.
- Gradient accumulation.

**Important LR difference vs pretraining:**

| Stage | Typical LR |
|---|---|
| Pretraining | 3e-4 |
| Full SFT | 1e-5 to 5e-5 |
| LoRA SFT | 1e-4 to 3e-4 (higher: only adapters move) |

```python
def train_sft(model, tokenizer, train_ds, val_ds, args, ckpt_prefix, sample_prompt):
    model.to(DEVICE)
    model.train()

    collate = make_collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2, pin_memory=True)

    # Only optimize parameters with requires_grad=True. This is the magic that makes
    # the same loop work for full SFT (everything trainable) and LoRA (only adapters)
    # without changing the loop.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                out = model(**batch)
                loss = out.loss / args.grad_accum_steps     # HF models return loss directly when labels passed
            scaler.scale(loss).backward()
            if micro_step % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer); scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
```

## §7 Part A — Full SFT on Qwen2.5-0.5B

**Why this model:** 494M params, modern architecture, ships with ChatML template. We use the BASE variant (not Instruct) to make the impact of SFT visible. All parameters trainable. With fp16 + AdamW: ~6 GB for model + grads + optimizer state. Fits T4.

**What to expect:**

- Loss starts ~2.0-2.5.
- Drops to ~1.3-1.7 after 500 steps.
- Generated samples shift from rambly base-model output → concise, on-topic instruction following.

**Common pitfall on T4:** the model loads in fp32 by default. Load directly with `torch_dtype=torch.float16` to save peak memory during load.

```python
model_a = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float16,
    device_map=None,
)
args_a = TrainArgs(
    epochs=1, batch_size=4, grad_accum_steps=4,
    learning_rate=1e-5,    # gentle: every param moves
    warmup_ratio=0.03, eval_every=100, save_every=200,
)
history_a = train_sft(model_a, tokenizer_a, train_ds_a, val_ds_a, args_a,
                      ckpt_prefix='qwen05b_fullsft')
```

## §8 Part B — LoRA SFT on TinyLlama-1.1B

**Why LoRA at all:** full SFT on a 1.1B model needs ~13 GB just for model + grads + AdamW state. Add activations + batch and you OOM on a 16 GB T4. LoRA solves this by:

1. **Freezing the base model.** No gradients flow through frozen weights. Optimizer state and gradient buffers are 0 for frozen params.
2. **Adding small trainable adapters.** For each target layer (typically attention's Q and V projections), insert two low-rank matrices `A` and `B` such that the layer's effective weight becomes `W + B·A`. `W` is frozen; `A` and `B` are tiny (rank `r`, typically 8-16) and trained.

Result: trainable params drop from 1.1B to ~5M (~0.5%). Memory drops ~10×.

**The LoRA equation:**

```
output = (W + B·A) · x      where W ∈ R^(d_out × d_in)
                                  A ∈ R^(r × d_in), B ∈ R^(d_out × r),  r << min(d_in, d_out)
```

**Key hyperparameters:**

- `r` (rank): 8 = light, 16 = standard, 32+ = heavy. We use 16.
- `lora_alpha`: scaling. Effective scale = `alpha / r`. Standard: `alpha = 2 × r`.
- `target_modules`: which layers to adapt. Original paper: `q_proj`, `v_proj`. 2026 best practice: all attention + FFN projections.
- `lora_dropout`: 0.05-0.1.

**Why higher LR (2e-4 vs 1e-5):** the base model cannot move, so adapters need to move aggressively to reach a good solution.

```python
from peft import LoraConfig, get_peft_model, TaskType

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model_b = get_peft_model(base_model, lora_config)
model_b.print_trainable_parameters()
# trainable params: ~5M || all params: 1.1B || trainable%: ~0.5%

args_b = TrainArgs(epochs=1, batch_size=4, grad_accum_steps=4,
                   learning_rate=2e-4,    # 20x higher than full SFT
                   warmup_ratio=0.03)
history_b = train_sft(model_b, tokenizer_b, train_ds_b, val_ds_b, args_b,
                      ckpt_prefix='tinyllama_lora')
```

## §9 The deployment pattern: save, reload, merge

In production you ship a small adapter file (~30 MB). Users load the base model from HF, then attach your adapter.

```python
# Save adapter only (the small artifact).
model_b.save_pretrained('/path/tinyllama_lora_adapters')

# At deploy: reload base from HF, then load adapter from disk.
from peft import PeftModel
base_reloaded = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16,
).to(DEVICE)
model_with_adapter = PeftModel.from_pretrained(base_reloaded,
                                               '/path/tinyllama_lora_adapters').to(DEVICE)

# Optional: merge adapter into base for a single-file model with zero inference overhead.
merged = model_with_adapter.merge_and_unload()
```

## §10 The production shortcut: `trl.SFTTrainer`

Everything above can be done in ~30 lines with TRL. The point of building from scratch first is to understand WHAT is happening; here is the production version.

**What TRL handles for you:**

- Chat template application.
- Loss masking (via `DataCollatorForCompletionOnlyLM`).
- Padding + attention masks.
- LoRA injection (just pass `peft_config`).
- Mixed precision, gradient accumulation, schedulers.
- Logging, eval, checkpointing.

**What you give up:** visibility. When something goes wrong (loss explodes, NaN gradients, wrong template) you have less control to debug. That is why building from scratch first is worth it.

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir='/path/trl_run',
    num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=2e-4, warmup_ratio=0.03, lr_scheduler_type='cosine',
    fp16=True, max_seq_length=1024, dataset_text_field='text',
)
trainer = SFTTrainer(
    model=base_for_trl, train_dataset=trl_train_ds, eval_dataset=trl_val_ds,
    peft_config=peft_cfg, args=sft_config, tokenizer=tokenizer_b,
)
trainer.train()
```

## §11 What you now understand (interview-ready)

1. **How is SFT different from pretraining?** Same loss (next-token cross-entropy), but loss is MASKED on user/system tokens. LR is 10-100× smaller. Data is instruction-response pairs in a chat template.
2. **What is LoRA and why does it work?** Freeze the base model, inject low-rank `B·A` adapters into select linear layers. Train only the adapters. Saves memory because optimizer state and gradient buffers shrink ~99%. Works because pretrained models have low-intrinsic-dimension fine-tuning updates (see [Why low-rank](../peft/why-low-rank.md)).
3. **What is the difference between LoRA and QLoRA?** QLoRA quantizes the frozen base to 4-bit (NF4) so it fits in even less VRAM, then runs LoRA on top. Same training quality as LoRA, ~4× less memory for the base model.
4. **How do you pick LoRA rank?** Start with r=16, alpha=32. Increase rank if val loss plateaus too high. Diminishing returns past r=64.
5. **Why does training loss drop fast then plateau?** Model is at a good optimum from pretraining. SFT moves it to a *nearby* optimum. Most learning happens in the first few hundred steps; rest is refinement.
6. **What is a chat template and why does it matter?** Format the pretrained model expects. Wrong template = model fights its own training. Always use `tokenizer.apply_chat_template`.

## Stretch goals

1. **Compare LoRA configs:** train with r=8, r=16, r=32. Plot val loss. Which wins?
2. **Add MLP layers to target_modules:** `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`. Bigger adapter, better quality, more VRAM.
3. **Try QLoRA:** load base with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. See [QLoRA Assembled](../peft/qlora-assembled.md).
4. **DPO:** take your SFT'd model and run preference fine-tuning on top. See [Alignment Walkthrough](alignment-walkthrough.md).
5. **Eval on a real benchmark:** run on a slice of MMLU or AlpacaEval for non-loss-based metrics.
