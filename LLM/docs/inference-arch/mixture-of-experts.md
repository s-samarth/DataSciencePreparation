# Mixture of Experts and Routing

The tension MoE solves: more parameters = smarter model, but every param costs memory *and* compute-per-token. In a dense model, predicting "the" still multiplies all 70B weights — most irrelevant. MoE asks: get the capacity of a giant model while activating only a slice per token. DeepSeek V3 = 671B total, 37B active. Same knowledge, ~5% of the compute.

!!! tip "Rapid Recall"
    MoE replaces the dense FFN with a router plus N parallel expert FFNs, top-K activated per token. The FFN is ~2/3 of a transformer's params and holds most of its "knowledge," so this is where the saving lives. Specialization is **learned end-to-end** — no human assigns "this expert = math." **Routing collapse** is the failure mode: one lucky expert gets picked more → improves → gets picked more, while others starve. **Aux-loss-free balancing (DeepSeek V3)** maintains a per-expert bias added to the routing score; nudge it up for under-used experts and down for over-used. LM loss stays uncontaminated. **Fine-grained experts plus a shared expert** is the 2026 default: 256 small experts top-8 give ~4×10¹⁴ combinations vs 28 for top-2 of 8 (compositional specialization), plus 1 always-on shared expert for universal knowledge.

## §1 The tension it solves

More parameters = smarter model, but every param costs memory *and* compute-per-token. In a dense model, predicting "the" still multiplies all 70B weights — most irrelevant. MoE asks: **get the capacity of a giant model while activating only a slice per token.** DeepSeek V3 = 671B total, 37B active.

## §2 Origin

Jacobs / Jordan / Nowlan / Hinton 1991 → Shazeer 2017 (sparsely-gated, top-K) → GShard / Switch 2020-21 (into transformers) → Mixtral 8×7B Dec 2023 (mainstream open weights) → DeepSeek V3 Dec 2024 (aux-loss-free + fine-grained).

## §3 Where it lives — it replaces the FFN, not attention

```
x → Attention → Add&Norm → [ Router + N parallel FFN experts ] → Add&Norm
```

The FFN is ~2/3 of a transformer's params and holds most of its "knowledge." Attention is the lookup mechanism; the FFN is the memory bank. MoE is **FFN parallelism with sparse activation.**

## §4 Routing math

```
gate_logits = W_router · x            # [num_experts]
probs       = softmax(gate_logits)
idx, w      = top_k(probs, k=2)       # pick best K
output      = Σ  w[i] · Expert_i(x)   # only selected experts compute
```

Specialization is **learned end-to-end** — no human assigns "this expert = math." A correct prediction sends gradient through the chosen expert (improving it) and the router (reinforcing the choice). The rich-get-better feedback loop creates emergent specialization.

DeepSeek V3 uses sigmoid affinity instead of softmax:

\[ s_{i,t} = \sigma(u_t^T \cdot e_i) \quad \text{(DeepSeek V3)} \]

Pick top-k experts by score. Normalize their scores, use as weights to blend expert outputs:

\[ y_t = \sum_{i \in \text{top-k}} \frac{s_{i,t}}{\sum_j s_{j,t}} \cdot \text{FFN}_i(u_t) \]

## §5 The hard part — routing collapse and load balancing

Left alone, one lucky expert gets picked more → improves → gets picked more, while others starve. You end up with a 671B model effectively using one expert.

| Fix | How | Cost |
|---|---|---|
| Auxiliary loss (old) | penalty term for uneven usage | competes with LM loss → hurts quality |
| Aux-loss-free (DeepSeek V3) | per-expert **bias** nudged up/down by observed load; bias only affects top-K selection, not the probability used in the weighted sum | LM loss stays uncontaminated — now standard |

The DeepSeek V3 routing trick is one of the most important MoE advances of 2024-2025 and is the modern default.

## §6 Fine-grained experts plus shared expert

Many small experts (256) instead of few big ones (8). Top-8 of 256 gives ~4×10¹⁴ combinations vs 28 for top-2 of 8 → **compositional** specialization. Plus 1 always-on **shared expert** for universal knowledge (grammar, common functions), freeing routed experts to specialize.

This is the 2026 frontier pattern: many fine-grained routed experts, one or two shared experts, top-K = 8 routed plus 1 shared activated per token.

## §7 The implementation sketch

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(n_experts)])
        self.top_k = top_k
        # DeepSeek V3: per-expert bias for aux-loss-free balancing.
        self.expert_bias = nn.Parameter(torch.zeros(n_experts), requires_grad=False)

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(-1, D)
        scores = torch.sigmoid(self.router(x_flat)) + self.expert_bias
        top_k_scores, top_k_idx = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_scores, dim=-1)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                token_weights = weights[mask][top_k_idx[mask] == i]
                out[mask] += token_weights.unsqueeze(-1) * expert(x_flat[mask])
        return out.view(B, N, D)
```

Real MoE implementations use much more efficient scatter-gather kernels (Megablocks, GroupedGEMM). This snippet is for understanding only.

## §8 Training vs inference nuance

!!! warning "The serving asymmetry"
    **Training:** better quality-per-FLOP. V3 trained 671B for ~$5.6M because each token activates 37B. **Inference:** you get the compute saving but **not** the memory saving — all experts must be resident in VRAM since routing is unpredictable per token. MoE serving works far better at high batch sizes, where routing variance averages out and most experts get used anyway. Small-batch serving of a 671B MoE is wasteful.

## §9 Why MoE makes sense at scale only

For models under ~30B parameters, the routing overhead and expert specialization do not pay off. MoE is used when you want capacity over ~100B without paying for it at inference (DeepSeek V3 671B / 37B, Mixtral 8×7B, Grok, likely GPT-5).

## Interview Questions

**Q1: DeepSeek V3 has 671B parameters but only activates 37B per token. How?**

Mixture of Experts in the FFN layers. Each MoE layer has 256 routed experts plus 1 shared expert always on. A small router network scores each expert for each token and picks top-8 routed experts. Only those 8 plus the shared expert run. Out of 256 experts, 3.1% are activated per token. Combined with 5.5% total activation rate across the model, you get 37B active from 671B. The architectural innovation is auxiliary-loss-free load balancing: instead of adding a loss that forces even expert usage (which distorts routing), they maintain a per-expert bias term that is nudged up for underused experts and down for overused ones, added to the routing score. This balances load without sacrificing routing quality.

**Q2: Why does MoE replace the FFN, not attention?**

The FFN holds ~2/3 of a transformer's parameters and is believed to be where factual knowledge is stored. Attention is the lookup mechanism (cheap, parameter-light); the FFN is the memory bank (expensive, parameter-heavy). Sparsifying the FFN gives the biggest capacity-per-FLOP win because that is where the parameters live. Sparsifying attention is harder and gives smaller gains.

**Q3: What is auxiliary-loss-free routing and why is it better than traditional aux loss?**

Traditional MoE adds a loss term penalizing uneven expert usage, but that loss competes with the LM loss and distorts what the router learns. DeepSeek V3 instead maintains a per-expert bias added to the routing score before top-K selection: if expert i is underutilized, increment b_i; if overutilized, decrement. The bias affects which experts get picked but is removed from the probability used to weight expert outputs. LM loss stays uncontaminated, load balance is enforced, quality is preserved.

**Q4: Why does MoE save compute at training but not memory at inference?**

At training, you only run the experts the router picks, so compute scales with activated parameters (37B for DeepSeek V3) not total (671B). But all 671B experts must be in VRAM at inference because the router can pick any of them at any time, unpredictably per token. There is no way to keep only a subset resident in advance. The compute saving is real; the memory saving is not.

**Q5: When does MoE not pay off?**

Below ~30B total parameters. Routing overhead, expert specialization, and load-balancing pain do not amortize over small capacity. Also small-batch serving — at batch 1, you activate top-K experts for one token, but you still hold all experts in VRAM, so cost-per-token is high. MoE is for scale (>100B capacity) and high-batch serving where routing variance averages out.

**Q6 (Trap): If MoE activates only 5% of parameters per token, is it 20× faster to serve?**

No, only on compute. Memory bandwidth still has to load the activated experts plus the routing computation plus the always-resident KV cache plus the shared expert. At small batch sizes, the per-token cost is dominated by loading the activated experts (similar to dense model decode), and the speedup is closer to 2 to 4×, not 20×. At large batch sizes where many tokens hit the same experts, the routing averages out and the activation efficiency closer to its theoretical limit.
