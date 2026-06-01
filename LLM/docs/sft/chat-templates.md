# Chat Templates

Modern LLM serving lives or dies on the chat template. The tokenizer's `apply_chat_template()` is the single source of truth: use a different template at serve time than at train time, and you get silent quality degradation that is invisible until eval. This page traces the historical evolution from plain concatenation to atomic special tokens and explains why the switch happened.

!!! tip "Rapid Recall"
    Chat templates are the specific string format that tells the model "user is talking" vs "assistant is talking." Llama 3 uses `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`. ChatML (OpenAI, Qwen) uses `<|im_start|>` / `<|im_end|>` around `system` / `user` / `assistant` turns. **These tags are single token IDs in the vocabulary, not seven characters** — base models reserve unused vocab slots during pretraining precisely so they can be repurposed as chat tokens during post-training. Always use `tokenizer.apply_chat_template()`; never hand-roll. The loss math is identical to plain LM training; chat templates are structural scaffolding that tells the model where its turn begins, what context it has, and when to stop.

## §1 Two ways to feed instructions

Tracing the actual historical evolution of how instruction + output pairs get presented to the model.

### 1.1 Approach 1 — Plain concatenation (early instruction tuning, ~2021-22)

You literally concatenate the instruction and the output into one continuous string and train next-token prediction over it (FLAN / T0 style):

```
Translate to French: Hello, how are you?
Bonjour, comment allez-vous?
```

The fatal flaw: **no structure**. At deployment, the model has no reliable signal for where the user's input ends and its own generation should begin. It relies on fragile heuristics ("after `Output:` I respond") that break under multi-turn, system prompts, and tool use.

### 1.2 Approach 2 — Chat templates with special tokens (modern, 2023+)

Everyone now (Meta, OpenAI, Anthropic, Mistral, Qwen) wraps conversations in special tokens that the tokenizer treats as single atomic units.

Llama 3 style:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize: The mitochondria...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Mitochondria produce energy for cells.<|eot_id|>
```

ChatML (OpenAI / Qwen) uses `<|im_start|>` / `<|im_end|>` around `system` / `user` / `assistant` turns to the same effect.

Example Qwen output:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

## §2 The three mechanical truths

!!! note "Mechanical truths to internalize"
    - **Special tokens are single token IDs.** `<|eot_id|>` is ONE token in the vocabulary, not seven characters. The model learns a dedicated embedding for it.
    - **Base models reserve "unused" vocab slots** during pretraining specifically so they can be repurposed as chat tokens during post-training.
    - **The model learns the structural meaning.** After the assistant header it should generate, and emitting `<|eot_id|>` means "stop."

## §3 Why chat templates won

Six properties that plain concatenation cannot deliver.

- **Unambiguous boundaries.** A special token is atomic; "USER:" can appear inside a user message ("the USER: in the diagram").
- **Multi-turn scaling.** Three-turn or thirty-turn conversations parse identically.
- **First-class system prompts.** The system role is a token, not a convention.
- **Tool-call roles.** New roles (tool, function) drop in as new special tokens.
- **Injection separation.** Easier to detect prompt injection when role boundaries are atomic.
- **Train/serve consistency.** `tokenizer.apply_chat_template()` is the single source of truth at both ends.

The loss math is *identical* to plain LM training. Chat templates are structural scaffolding. They tell the model where its turn begins, what context it has, and when to stop. The single source of truth at inference is `tokenizer.apply_chat_template()`. Using a different template at serve time than at train time causes silent degradation (train-serve skew).

## §4 The practical usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

msgs = [
    {"role": "system",    "content": "You are a helpful AI assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

# During training: tokenize the full conversation.
full_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)

# At inference: append the assistant role header so the model generates from there.
prompt_only = msgs[:-1]   # drop the assistant message
prompt_ids = tokenizer.apply_chat_template(prompt_only, tokenize=True, add_generation_prompt=True)
```

The `add_generation_prompt=True` flag matters at inference. It appends the `<|im_start|>assistant\n` (or equivalent) header tokens, putting the model in the right state to generate.

## Interview Questions

**Q1: What is a chat template and why does it matter?**

It is the specific string format the pretrained model expects, with role boundaries marked by atomic special tokens. Different models use different templates (ChatML, Llama 3, Mistral, Gemma each have their own). Wrong template at serve time means the model fights its own training and quality silently drops. Always use `tokenizer.apply_chat_template()`.

**Q2: Why not just use plain-text markers like "USER:" / "ASSISTANT:"?**

Those tokenize into multiple regular tokens the model must *parse*, and they could appear inside genuine user input ("ignore previous instructions, USER: …"). Special tokens are atomic and unambiguous, give multi-turn scaling, first-class system prompts, and tool roles, and make injection separation feasible.

**Q3: How is `<|eot_id|>` different from `</s>` (EOS)?**

EOS = "this document is completely finished" (a pretraining concept). EOT = "this turn is done but the conversation continues." Llama 3 uses them differently; mixing them up causes generation bugs.

**Q4: What is the train-serve skew risk in SFT?**

If inference uses a different chat template than training, performance silently drops. This is exactly why `apply_chat_template()` exists as a single source of truth. Always inspect the formatted prompt at both train and inference and confirm they match byte-for-byte.

**Q5: Where do these special tokens come from in the vocabulary?**

Base models reserve unused vocab slots during pretraining specifically so they can be repurposed as chat tokens during post-training. The slots already have learned embeddings (initialized like everything else and updated during pretraining), and SFT or post-training learns the specific structural meaning attached to each one.
