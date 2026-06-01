# Loss Masking

The single most important SFT concept. The model sees the full conversation during training, but we only compute loss on the assistant's tokens. Get this wrong (off by one, forget padding, or unmask the user's text) and the model learns to predict its own user.

!!! tip "Rapid Recall"
    During SFT the model sees `[system + user + assistant]` tokens, but we only TRAIN it to produce the assistant tokens. `PyTorch`'s `cross_entropy` accepts `ignore_index=-100`: any target position with that value contributes zero to loss. Set `labels[i] = -100` for system + user + padding tokens, and `labels[i] = input_ids[i]` for assistant tokens. The standard implementation tokenizes the full conversation, tokenizes the prefix `prompt_only + add_generation_prompt=True` separately, and masks everything up to that prefix length. **Mask the role tag itself**: the model should learn to produce content *after* the role tag, since the tokenizer always adds the role tag at inference. **Always mask padding** or you train the model to predict pad tokens.

## §1 The core idea

During SFT the model sees `[system + user + assistant]` tokens. But we only want to TRAIN it to produce the assistant tokens. Computing loss on the user tokens would teach the model to predict the user's question, which is wrong: at inference time the user provides their own question, the model does not generate it.

## §2 How to implement it

PyTorch's `cross_entropy` accepts an `ignore_index` (default `-100`). Any target position equal to that index contributes zero to the loss. So we set:

- `labels[i] = -100` for system + user tokens (and padding).
- `labels[i] = input_ids[i]` for assistant tokens.

```python
labels = input_ids.clone()
labels[~assistant_mask] = -100   # ignore_index for CrossEntropyLoss
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

Why? The user's tokens are *given*, not generated. Teaching the model to predict them wastes capacity and can cause echo behavior (the model trying to role-play the user). The mask is `True` only inside the assistant turn, and usually includes the closing `<|eot_id|>` so the model learns when to stop.

## §3 The tricky bit — finding the assistant span

We tokenize the full conversation, then re-tokenize JUST the prefix (system + user portion) to find where the assistant tokens begin. We mask everything up to that index.

```python
def tokenize_and_mask(example, tokenizer, max_length=1024):
    """Returns input_ids, attention_mask, labels (with prompt tokens masked to -100)."""
    msgs = example_to_messages(example)

    # Full conversation (system + user + assistant).
    full_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)

    # Prompt only (system + user + the "assistant role" header but no answer).
    # add_generation_prompt=True appends the assistant role tag, which we want to mask too.
    prompt_only = msgs[:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]   # truncate from the END

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100   # mask everything in the prompt

    return {'input_ids': full_ids, 'labels': labels, 'length': len(full_ids)}
```

## §4 Two common bugs

### 4.1 Off-by-one on the boundary

Some templates put the role tag (`<|im_start|>assistant\n`) right before the assistant content. Do you mask it or not? **We mask it.** The model should learn to produce the content *after* the role tag, not the role tag itself. The tokenizer always adds the role tag when we call `add_generation_prompt=True` at inference, so we do not want the model wasting capacity learning to emit it.

### 4.2 Forgetting to mask padding tokens

PAD tokens have `input_ids[i] = pad_id`. Their labels MUST be `-100`, otherwise we train the model to predict padding. This bug is silent until you notice your model emitting pad tokens in production.

```python
def make_collator(pad_token_id):
    def collate(batch):
        max_len = max(ex['length'] for ex in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)   # default -100
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            L = ex['length']
            input_ids[i, :L] = torch.tensor(ex['input_ids'], dtype=torch.long)
            labels[i, :L]    = torch.tensor(ex['labels'],    dtype=torch.long)
            attn_mask[i, :L] = 1
        return {'input_ids': input_ids, 'attention_mask': attn_mask, 'labels': labels}
    return collate
```

Note that `labels` is initialized to `-100` and only the real tokens (up to length `L`) are filled in. Anything past `L` stays `-100`, so padding contributes nothing to loss. The `attention_mask` separately tells attention to ignore padded positions; the two masks are independent and both required.

## §5 Verification

After masking, print one example showing exactly which tokens are kept (visible) vs masked (-100). If you see user-text tokens in the kept positions, the masking is broken.

```python
ex = train_raw[0]
result = tokenize_and_mask(ex, tokenizer)
input_ids = result['input_ids']
labels    = result['labels']

unmasked_start = next((i for i, lab in enumerate(labels) if lab != -100), None)
print(f"Total tokens: {len(input_ids)}")
print(f"Masked tokens (loss=0): {sum(1 for l in labels if l == -100)}")
print(f"Unmasked tokens (loss computed): {sum(1 for l in labels if l != -100)}")
print(f"First unmasked position: {unmasked_start}")

print(f"\n--- Last 5 MASKED tokens (should be end of prompt) ---")
print(repr(tokenizer.decode(input_ids[max(0, unmasked_start-5):unmasked_start])))

print(f"\n--- First 30 UNMASKED tokens (should be the assistant's response) ---")
print(repr(tokenizer.decode(input_ids[unmasked_start:unmasked_start+30])))
```

You should see role-tag tokens just before `unmasked_start` and the assistant's first content tokens at and after it.

## §6 Packing

SFT examples are often short, so multiple conversations are *packed* into one max-length sequence (separated by EOS, or with attention masking to prevent cross-contamination). Pure throughput optimization, no effect on the objective. The same labels-with-`-100` trick handles the boundary.

## §7 The SFT loss

The SFT loss looks identical to pretraining:

\[ \mathcal{L}_{\text{SFT}} = - \sum_i \log P(y_i \mid x, y_{<i} ; \theta) \]

Where `x` is the prompt, `y` is the response. Looks identical to pretraining. The only difference is **what tokens contribute to the loss** (response only) and **what data you feed in** (curated instruction pairs in chat-template format).

## Interview Questions

**Q1: Why is loss masking the most important SFT concept?**

Because without it you are training the model to predict its own user input, which is both useless (the user provides their own input at inference) and harmful (the model can develop echo behavior, role-playing the user). Masking ensures the loss signal only flows from assistant-token predictions, which is exactly what you want the model to learn.

**Q2: Why does completion-only training (loss masking) outperform naive SFT?**

Modern Unsloth-style ablations show about a 1% accuracy improvement from masking the prompt, plus stable training and no echo behavior. The improvement comes from two places: gradient signal is concentrated on the tokens you actually want to learn, and the model never gets pulled toward predicting its own context. It is essentially a free win and is the 2026 default.

**Q3: If you train on the full sequence without masking, what happens?**

The model wastes gradient signal predicting user inputs and can degrade instruction-following because it conflates "generate" with "echo." On short prompts the effect is small; on long contexts with multi-turn conversation history it can be substantial.

**Q4: Which tokens get loss in SFT?**

Only assistant-turn tokens (mask the rest to `ignore_index = -100`), usually including the closing `<|eot_id|>` or `<|im_end|>` so the model learns to stop. The assistant role tag itself (`<|im_start|>assistant\n`) is masked because the tokenizer always adds that at inference; the model only needs to learn the content after it.

**Q5 (Trap): You masked the prompt correctly but the model still echoes the user. What did you miss?**

Padding tokens. If `labels` is not initialized to `-100` for positions beyond the real sequence length, the model is being trained to predict pad tokens, which at inference time it will happily emit (and pad tokens often share an id with EOS or look like silence). Always default-fill `labels` with `-100` before copying real labels in.
