# Initialization and Gradient Flow

A is initialized random-normal, B is initialized to zero. The standard explanation ("A can't be zero or no gradient flows") is right in conclusion but usually wrong in reasoning. This page is the actual proof: it is **B's** gradient that dies if A = 0, not A's own gradient.

!!! tip "Rapid Recall"
    LoRA initializes `A ~ N(0, σ²)` and `B = 0`. Setting B = 0 makes `ΔW = BA = 0` at step 0 so the model begins at its pretrained state, no disruption on the first forward pass. The subtle part: A's gradient does NOT depend on A's value — it depends on B and the upstream gradient. So A = 0 would NOT kill A's own gradient. What it kills is **B's gradient**, because B's gradient flows through `A·x`, and if A = 0 then `A·x = 0`, so `∂L/∂B = 0` forever. B never moves, ΔW stays stuck at 0, adapter dead. The asymmetry: zero B (safe), never zero A (kills B).

## §1 The setup

- **A** ~ N(0, σ²), where σ is a small constant. The LoRA paper uses σ = 1/√r; some libraries use a fixed 0.02. You do not tune it; set once and forget.
- **B** = 0, so that `ΔW = BA = 0` at step 0 and the model starts exactly at its pretrained state (no disruption on the first forward pass).

## §2 Why A cannot also be zero — the real reason

The output is `output = (α/r) · B · A · x`. Walk the gradients.

\[ \frac{\partial L}{\partial A} = (\alpha/r) \cdot B^T \cdot \frac{\partial L}{\partial \text{output}} \cdot x^T \]

A's own gradient does NOT depend on A. So **A = 0 does not kill A's gradient.**

!!! warning "The common misconception"
    People say "A = 0 means A gets no gradient." False. Look at the formula above. A's gradient is independent of A's value. The real problem is what A = 0 does to **B's** gradient.

\[ \frac{\partial L}{\partial B} = (\alpha/r) \cdot \frac{\partial L}{\partial \text{output}} \cdot (A \cdot x)^T \]

B's gradient depends on `A·x`. If A = 0, then `A·x = 0`, so `∂L/∂B = 0` forever.

## §3 The asymmetry, stated cleanly

| Init choice | Consequence | Verdict |
|---|---|---|
| B = 0, A = random | ΔW = 0 at start (good). A·x ≠ 0, so B receives gradient and starts moving. | works |
| A = 0, B = random | A·x = 0 → B's gradient = 0 → B never updates → ΔW stuck at 0 → adapter dead. | dead |
| both = 0 | No gradient path opens at all. | dead |
| both random | ΔW ≠ 0 at start → perturbs pretrained model immediately. | disruptive |

!!! abstract "The crisp takeaway"
    A being zero does not kill A's own gradient — it kills **B's** gradient (because B's gradient flows through A·x). Dead B means ΔW never moves off zero. That is why exactly one of them is zeroed, and it is B.

## §4 What the LoRA paper actually does

```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False

        d_out, d_in = base_layer.weight.shape
        # LoRA matrices: random A, zero B
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # random init
        self.B = nn.Parameter(torch.zeros(d_out, r))          # zero init → ΔW=0 at start
```

In the LoRA paper the initialization for A is Gaussian with std `1/√r`. The PEFT library uses Kaiming-uniform for A by default. The choice does not matter much in practice; what matters is that A is *not* zero and B is *exactly* zero.

## §5 What happens in step 1, step 2, step 3

A useful exercise: trace the first few optimization steps.

- **Step 0:** A random, B = 0. ΔW = 0. Forward pass: `h = Wx`. Loss is the base model's loss.
- **Step 1 backward:** ∂L/∂B is nonzero (depends on `A·x` and upstream gradient). ∂L/∂A is zero (depends on B^T which is still zero from step 0). Wait — A does not move at step 1? Correct. But B does.
- **Step 1 update:** B becomes small but nonzero.
- **Step 2 backward:** Now both ∂L/∂A and ∂L/∂B are nonzero. ΔW = BA is small but nonzero, so the forward pass starts diverging from the base model.
- **Step 2 update:** A and B both move.

So A actually does not move until step 2 — but it does not matter, because B moves at step 1 and that opens the door for A to move at step 2. The whole training run gets bootstrapped.

## Interview Questions

**Q1: Why is B zero and A random — and what really breaks if you swap them?**

B = 0 makes ΔW = 0 at start so the model begins at its pretrained state. The subtle part: A's gradient does NOT depend on A, so A = 0 would not kill A's own gradient. The real issue is that B's gradient flows through `A·x` — if A = 0 then `A·x = 0`, so B's gradient is zero forever, B never moves, and ΔW stays stuck at zero. So you zero B (safe) and never A (kills B).

**Q2: If both A and B are random, why is that bad?**

Then ΔW = BA is nonzero at step 0, which means the model immediately diverges from its pretrained state before training has even begun. You lose the initialization advantage of starting from a good base model and effectively run with random adapters bolted on, which destabilizes training and often diverges before any useful signal accumulates.

**Q3: Walk me through the first three optimizer steps under the standard A-random, B-zero init.**

Step 0: ΔW = 0, forward equals the base model. Step 1 backward: ∂L/∂B is nonzero (depends on `A·x` which is nonzero), ∂L/∂A is zero (depends on B which is zero). Only B updates. Step 2 backward: B is now small but nonzero, so ∂L/∂A is also nonzero. Both update. From step 2 onward, both A and B move, and the adapter starts to do real work.

**Q4 (Trap): Someone proposes initializing A = 0 and B = N(0, σ²) instead. Will it work?**

No, for the symmetric reason. `output = B·A·x` and the gradient is `∂L/∂A = (α/r)·Bᵀ·grad·xᵀ`, which is fine (depends on B, which is nonzero). But `∂L/∂B = (α/r)·grad·(A·x)ᵀ`, and with A = 0 we get `A·x = 0`, so B gets zero gradient forever despite being random. Same dead-adapter failure mode, just from the other side. The asymmetry is: whichever matrix you zero, its partner needs that matrix in its gradient path to be nonzero. So you zero the one whose partner does not need it.
