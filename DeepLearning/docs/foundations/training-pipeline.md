# The Complete Training Pipeline

The full end-to-end PyTorch workflow: synthetic data → Dataset → DataLoader → Model → Loss + Optimizer + Scheduler → Training loop → Evaluation → Saving. Walking through this once gives you the template for any project, and it ties together every concept from the rest of this section.

!!! tip "Rapid Recall"
    Split into train/val/test *before* computing normalization statistics, and compute those stats from training data only or you leak. A `Dataset` is just `__init__`, `__len__`, `__getitem__`; the `DataLoader` batches and shuffles (train only). The model outputs raw logits and `CrossEntropyLoss` applies softmax internally. The loop is the canonical six steps; track the best model by validation loss, then evaluate on test exactly once. Save the `state_dict` for research and TorchScript for production, and keep softmax at the serving layer so you can recalibrate without retraining.

## §1 Step 1: Generate synthetic data

```python
import numpy as np

# 3-class classifier, 10D inputs, nonlinear true function
N_SAMPLES = 5000
N_FEATURES = 10
N_CLASSES = 3

X = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)

# Nonlinear signal — quadratic in features, plus noise
signal = (X[:, 0] * X[:, 1]
        + X[:, 2] ** 2
        - X[:, 3] * X[:, 4]
        + 0.3 * np.random.randn(N_SAMPLES))

# Threshold into 3 balanced classes
thresholds = np.percentile(signal, [33.33, 66.67])
y = np.zeros(N_SAMPLES, dtype=np.int64)
y[signal > thresholds[0]] = 1
y[signal > thresholds[1]] = 2

# dtype matters: float32 for features, int64 for labels (CrossEntropyLoss requirement)
```

## §2 Step 2: Split into train/val/test BEFORE normalization

This is critical. Compute normalization statistics from training data only, never use val/test statistics or it's data leakage.

```python
# 70/15/15 split
n_train = int(0.70 * N_SAMPLES)
n_val   = int(0.15 * N_SAMPLES)
n_test  = N_SAMPLES - n_train - n_val

indices = np.random.permutation(N_SAMPLES)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# Compute normalization stats from TRAINING data ONLY
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)

# Apply to all splits using training statistics
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std
```

## §3 Step 3: PyTorch Dataset class

A `Dataset` is just a class with three methods: `__init__`, `__len__`, `__getitem__`. "How to get one sample."

```python
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy → tensors ONCE at init
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()      # long = int64
        # requires_grad defaults to False — data never carries gradients

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return one (features, label) pair
        return self.X[idx], self.y[idx]

train_dataset = SyntheticDataset(X_train, y_train)
val_dataset   = SyntheticDataset(X_val,   y_val)
test_dataset  = SyntheticDataset(X_test,  y_test)
```

## §4 Step 4: DataLoaders

`DataLoader` handles batching, shuffling, parallel loading. Shuffle training only, never shuffle val/test.

```python
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # shuffle train
    num_workers=0,
    drop_last=True         # drop incomplete last batch for consistent sizes
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

## §5 Step 5: Model definition

Output raw logits, no softmax. `CrossEntropyLoss` applies softmax internally for numerical stability.

```python
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=10, n_classes=3, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)    # weight (64,10), bias (64,)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(dropout)
        # All Linear params have requires_grad=True automatically
        # Dropout has no learnable parameters

    def forward(self, x):
        # x: (B, 10)
        x = self.fc1(x)         # (B, 64), now in compute graph
        x = F.relu(x)            # (B, 64)
        x = self.dropout(x)     # (B, 64) — active in train mode only
        x = self.fc2(x)         # (B, 32)
        x = F.relu(x)            # (B, 32)
        x = self.fc3(x)         # (B, 3) — RAW LOGITS
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
```

## §6 Step 6: Loss, Optimizer, Scheduler

```python
criterion = nn.CrossEntropyLoss()    # takes raw logits + integer labels

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)

NUM_EPOCHS = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
```

The choices here are covered in detail elsewhere: [AdamW](../optimization/adaptive-methods.md), [weight decay](../regularization/regularization.md), and [cosine annealing](../regularization/lr-schedules.md).

## §7 Step 7: The training loop (with shape tracking)

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()      # IMPORTANT: dropout active, BN uses batch stats

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        # x: (B, 10), y: (B,)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()       # Step 1: clear .grad fields

        logits = model(x)            # Step 2: forward, logits (B, 3)
                                    #         requires_grad=True in train mode

        loss = criterion(logits, y)  # Step 3: scalar tensor

        loss.backward()             # Step 4: populate .grad on all params

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Step 5

        optimizer.step()             # Step 6: update weights

        # Tracking — .item() breaks the graph, safe to use
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()       # IMPORTANT: dropout off, BN uses stored stats
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)           # NO graph built (no_grad context)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples

# Main loop: save the best model by validation loss
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
```

## §8 Step 8: Final evaluation

Load the best checkpoint (from the best validation epoch), evaluate on the test set ONCE, never iterate on the test set or it's data leakage.

```python
model.load_state_dict(torch.load("best_model.pt"))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final test accuracy: {test_acc:.4f}")
```

## §9 Saving and serving the model

Three approaches to persisting a trained model, with tradeoffs. Plus a complete inference wrapper that loads the model once and serves predictions.

| Method | What it saves | Pros | Cons |
| --- | --- | --- | --- |
| **state_dict** (recommended for research) | Just the learned parameters as a Python dict | Smallest file. Version-control friendly. Most flexible. | Requires the model class definition to load. |
| Entire model (`torch.save(model, ...)`) | Pickles the whole Python object | Simplest to load. | Brittle. Breaks if class definition changes. Avoided in practice. |
| **TorchScript** (recommended for production) | Compiled, serializable intermediate representation | Runs WITHOUT the original Python class. Runs in C++ (libtorch). Slight speed boost. | Some Python features unsupported (must be script-friendly). |

### state_dict approach

```python
# Save:
torch.save(model.state_dict(), "best_model.pt")

# Load (in another script — requires the NeuralNetwork class to be in scope):
model = NeuralNetwork(input_dim=10, n_classes=3)   # rebuild architecture
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
```

### TorchScript approach (production-grade)

```python
model.eval()   # MUST be in eval mode before scripting

# Two ways to script:
# - torch.jit.script(): handles control flow, more flexible
# - torch.jit.trace(): records one execution, faster but no dynamic control flow

scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# Also save preprocessing stats — needed at inference!
np.save("norm_mean.npy", mean)
np.save("norm_std.npy", std)

# Load in production (does NOT need NeuralNetwork class):
model = torch.jit.load("model_scripted.pt")
```

### A complete inference server wrapper

```python
class ModelServer:
    """Loads model + preprocessing once. Exposes predict()."""

    def __init__(self, model_path, mean_path, std_path, device="cpu"):
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.device = device

    @torch.no_grad()
    def predict(self, x_raw):
        """
        x_raw: numpy array (n_samples, n_features) OR (n_features,)
        Returns: predicted classes and probabilities
        """
        # Handle single sample case
        if x_raw.ndim == 1:
            x_raw = x_raw[None, :]   # add batch dim

        # Preprocess — apply same normalization as training
        x_norm = (x_raw - self.mean) / self.std

        # Convert to tensor
        x_tensor = torch.from_numpy(x_norm).float().to(self.device)

        # Forward pass — get logits
        logits = self.model(x_tensor)

        # Apply softmax for probabilities (kept out of model for flexibility)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        return preds.cpu().numpy(), probs.cpu().numpy()

# Usage:
server = ModelServer("model_scripted.pt", "norm_mean.npy", "norm_std.npy")
preds, probs = server.predict(new_data)
```

Always sanity-check that the serialized model gives identical outputs to the original. Floating-point should be exactly identical for the same inputs.

```python
# Get a test batch, run through both models
with torch.no_grad():
    out_original = original_model(x)
    out_scripted = scripted_model(x)

max_diff = (out_original - out_scripted).abs().max().item()
assert max_diff < 1e-6, "Models diverged!"
```

!!! note "The serving rule"
    Always keep the model emitting logits and convert to probabilities at the serving layer. Why? Flexibility. You can change probability calibration, apply temperature scaling, or threshold differently without retraining. If softmax is baked into the model, you have to retrain to change behavior.

## §10 Tensor shape reference

For the classifier above, here is every tensor with its shape, whether it carries gradients, and a note on its role. Internalize this, it is the mental model PyTorch operates under.

| Tensor | Shape | requires_grad | Note |
| --- | --- | --- | --- |
| Single sample x | `(10,)` | False | Data, never carries gradients |
| Single label y | `()` scalar | False | Target, never carries gradients |
| Batch x | `(B, 10)` | False | DataLoader stacks samples |
| Batch y | `(B,)` | False | Integer class indices |
| fc1.weight | `(64, 10)` | True | Learnable parameter, leaf tensor |
| fc1.bias | `(64,)` | True | Learnable parameter, leaf tensor |
| fc2.weight | `(32, 64)` | True | Learnable parameter |
| fc3.weight | `(3, 32)` | True | Learnable parameter |
| $z_1$ (after fc1) | `(B, 64)` | True (in graph) | Intermediate, not a leaf, no .grad stored |
| $a_1$ (after ReLU) | `(B, 64)` | True (in graph) | Intermediate |
| logits $z_2$ | `(B, 3)` | True in train, False in eval/no_grad | Raw output scores |
| loss | `()` scalar | True | Where .backward() starts |
| fc1.weight.grad | `(64, 10)` | N/A (it IS a gradient) | Populated by .backward(); same shape as weight |

The gradient mental model:

- **Leaf tensors with requires_grad=True** are model parameters. The optimizer updates these. Their `.grad` attribute gets populated by `.backward()`.
- **Data tensors** never have requires_grad=True. They flow through the graph but no gradient is computed for them (we don't want to update the data).
- **Intermediate activations** are part of the computational graph in training mode but are NOT leaf tensors. Their `.grad` is never populated, they are just nodes the gradient passes through.
- **`@torch.no_grad()` and `with torch.no_grad():`** break the graph. Operations don't get recorded. Use during eval/inference to save memory and speed up the forward pass.
- **`.item()` and `.detach()`** also break the graph. Use `.item()` when tracking metrics (extracts a Python scalar), `.detach()` when you want a tensor copy that won't backprop.

!!! note "The realization"
    Only leaf tensors get `.grad` populated. Intermediate tensors are in the graph but the gradient just flows through them. This is why you cannot directly access "the gradient of an intermediate activation", there is no place to store it. If you need it, register a hook or use `retain_grad()`.

## §11 Complexity reference

The time, memory, and parallelism characteristics for every technique on this site. Useful for systems-design discussions and back-of-envelope calculations. Symbols: $P$ = parameters, $B$ = batch size, $L$ = layers, $d$ = layer width, $C$ = channels, $H, W$ = spatial dims, $T$ = sequence length, $D$ = model dimension.

### Forward and backward pass

| Operation | Time | Memory | Parallelizable? |
| --- | --- | --- | --- |
| Forward pass | $O(L \cdot d^2)$ per sample | $O(L \cdot d)$ activations (stored for backward) | Yes, GPU matmul |
| Backward pass | ~2-3× forward pass | $O(L \cdot d^2)$ gradients (size of model) | Yes |

### Optimizer step

| Optimizer | Time per step | Extra memory | Notes |
| --- | --- | --- | --- |
| SGD | $O(P)$ | 0 | 1× param memory total |
| SGD + Momentum | $O(P)$ | $O(P)$ | 2× param memory (v) |
| RMSProp | $O(P)$ | $O(P)$ | 2× param memory (v) |
| Adam / AdamW | $O(P)$ | $O(2P)$ | 3× param memory (m, v) |
| Lion | $O(P)$ | $O(P)$ | 2× param memory |

### Normalization

|  | Forward time | Extra memory | Params | Parallelizable |
| --- | --- | --- | --- | --- |
| BatchNorm (CNN) | $O(B \cdot C \cdot H \cdot W)$ | cached activations + $O(C)$ stats | $2C + 2C$ buffers | Across batch (SyncBN for multi-GPU) |
| LayerNorm | $O(B \cdot T \cdot D)$ | cached + $O(B \cdot T)$ stats | $2D$ | Trivially per-sample |
| RMSNorm | $O(B \cdot T \cdot D)$, ~0.75× LN | cached | $D$ | Trivially |

### Other techniques

| Technique | Time | Memory | Notes |
| --- | --- | --- | --- |
| Gradient clipping | $O(P)$ | $O(1)$ | One pass for norm, one to rescale |
| Weight decay | $O(P)$ | 0 | One elementwise multiply per step |
| LR schedule | $O(1)$ | $O(1)$ | Just update a number |
| Dropout (training) | $O(\text{activation size})$ | $O(\text{activation size})$ for mask | Bernoulli mask, elementwise multiply |
| Dropout (inference) | 0 (disabled) | 0 |  |
| Residual connection | $O(\text{activation size})$ | $O(\text{activation size})$ for stored x | Just an add, basically free |
| LSTM cell (vs RNN) | ~4× RNN cost | 4× state size | 4 gates instead of 1 |

!!! warning "Memory bottlenecks for large models"
    **1. Activations dominate.** The backward pass needs all forward activations to compute gradients. For a 70B parameter model, activations can be hundreds of GB. This is why *gradient checkpointing* exists (recompute activations instead of storing them, trading compute for memory). **2. Adam optimizer state is 3× the model size.** A 7B model in fp32 = 28GB weights + 28GB m + 28GB v = 84GB optimizer memory. This forces mixed precision (bf16/fp16), 8-bit Adam, or sharded optimizers (ZeRO, FSDP) at scale.
