# The Training Loop

Every training step in every neural network ever trained does the same six things in the same order. Memorize this. Skipping any one of them is a real bug that experienced engineers still make.

!!! tip "Rapid Recall"
    The six steps, in order: zero gradients, forward pass, compute loss, backward pass, (optionally) clip gradients, update weights. PyTorch accumulates gradients by default, so forgetting `optimizer.zero_grad()` silently diverges training. Dropout and BatchNorm behave differently in `train()` versus `eval()`, so forgetting `model.eval()` at inference is a classic production bug. Wrap evaluation forward passes in `torch.no_grad()` to skip building the graph and save memory.

## §1 The six steps

**Step 1 · Zero gradients.** Clear gradient memory from the previous step.

```python
optimizer.zero_grad()
```

PyTorch *accumulates* gradients by default. Every backward pass *adds* the new gradients to whatever was already in each parameter's `.grad` field. Forget this and gradients accumulate across batches, a silent bug where training slowly diverges into noise. (This accumulation is useful when intentional, for simulating larger batch sizes by accumulating gradients across multiple small batches before stepping.)

**Step 2 · Forward pass.** Run input through the network to produce predictions.

```python
output = model(x)
```

PyTorch silently records every operation in a computational graph as it runs. This graph is what makes the backward pass possible.

**Step 3 · Compute loss.** Measure how wrong the prediction was.

```python
loss = criterion(output, y)
```

A single scalar tensor, one number summarizing batch error. Exact computation depends on the task.

**Step 4 · Backward pass.** Walk backwards through the graph, computing every parameter's gradient.

```python
loss.backward()
```

After this line, every parameter has a populated `.grad` field of the same shape as itself, telling the optimizer how that parameter contributed to the loss. The mechanics are covered in [Backpropagation](backpropagation.md).

**Step 5 · Clip gradients (optional but standard).** Cap gradient magnitude to prevent catastrophic updates.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Rescales gradients if their total L2 norm exceeds the threshold. Mandatory for RNN training, standard for transformer pretraining, cheap insurance for everything else. See [Exploding Gradients & Clipping](../gradient-flow/exploding-gradients.md).

**Step 6 · Update weights.** The optimizer reads each `.grad` and applies its update rule.

```python
optimizer.step()
```

For SGD: subtract learning rate times gradient. For Adam: a more involved formula covered in [Adaptive Methods](../optimization/adaptive-methods.md). Either way, weights have changed after this line.

## §2 Train mode versus eval mode

Two specific layer types behave differently depending on the mode. Switching modes is mandatory.

| Layer | `model.train()` | `model.eval()` |
| --- | --- | --- |
| Dropout | Active, zeros random units | Disabled, passes activations unchanged |
| BatchNorm | Uses current batch statistics; updates running averages | Uses stored running averages (not current batch) |

Forgetting to switch modes is one of the most common production bugs. A model that worked great in training mysteriously performs poorly in production because `model.eval()` wasn't called, dropout is still zeroing neurons and BatchNorm is using batch-of-size-1 statistics.

### torch.no_grad() at evaluation

For evaluation, wrap forward passes in `with torch.no_grad():` or decorate with `@torch.no_grad()`. This tells PyTorch not to build the computational graph. Saves roughly half the memory, runs slightly faster. Always use it for any forward pass that won't be followed by a backward pass.

!!! warning "The three classic loop bugs"
    **1.** Forgetting `optimizer.zero_grad()`, gradients accumulate, updates become noise, training diverges silently. **2.** Forgetting `model.eval()` at evaluation time, dropout still active, BatchNorm wrong, metrics are stochastic and incorrect. **3.** Forgetting `torch.no_grad()` at evaluation, build the graph for nothing, waste memory, run out of memory on a constrained GPU.

## Interview Questions

**Q1: What is the train/eval mode distinction in PyTorch, and why does it matter?**

Two layer types behave differently. Dropout is active in train mode (zeros random units) and disabled in eval mode (passes activations through). BatchNorm uses current batch statistics in train mode (and updates running averages), and uses stored running averages in eval mode. Forgetting `model.eval()` at inference is a classic bug. Dropout will still be zeroing neurons (so predictions are stochastic, different on every call), and BatchNorm will be using batch statistics computed from your inference batch (possibly batch size 1), which gives garbage outputs.
