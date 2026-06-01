# Alignment

After SFT your model answers questions, but it may answer them rudely, verbosely, unsafely, or just averagely. Alignment teaches it which of two answers humans prefer. The pages in this section walk the full historical arc: Bradley-Terry reward models, PPO with all four models, the DPO derivation that killed three of them, GRPO which killed the fourth, and RLVR which kills the reward model entirely for verifiable domains.

!!! tip "Rapid Recall"
    Alignment optimizes the constrained objective `E[r(x,y)] − β·KL(π‖π_ref)`. **RM + PPO** is the classical RLHF stack: Bradley-Terry RM, clipped surrogate, four models in memory (policy, reference, RM, value). **DPO** uses the closed-form optimal policy to rewrite the objective as supervised contrastive loss; no RM, no rollouts, two models. **GRPO** keeps PPO's rollouts and clipping but drops the critic by using the group mean reward as baseline; it is the post-2024 reasoning-model default. **RLVR** is GRPO with a programmatic verifier in place of the RM, the DeepSeek-R1 recipe. **The 2026 frontier pipeline** chains pretrain → SFT → RFT → RLVR via GRPO → RLHF/DPO. GRPO does not replace stages; it is a powerful new tool added to the stack.

## The 2026 pipeline

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The 2026 frontier model training pipeline">
<defs><marker id="alarr" markerWidth="10" markerHeight="10" refX="5" refY="8" orient="auto"><path d="M0,0 L10,0 L5,9 Z" fill="#8a3115"/></marker></defs>
<g font-family="Fraunces, serif">
<rect x="200" y="20"  width="300" height="56" rx="9" fill="#1a1410"/>
<text x="350" y="44" text-anchor="middle" font-weight="600" font-size="16" fill="#f4ede0">PRE-TRAINING</text>
<text x="350" y="64" text-anchor="middle" font-family="Newsreader,serif" font-size="12.5" fill="#d4c4a8">10-15T tokens, knowledge lives here</text>
<rect x="200" y="98"  width="300" height="50" rx="9" fill="#2c6e63" fill-opacity="0.85"/>
<text x="350" y="120" text-anchor="middle" font-weight="600" font-size="15" fill="#f4ede0">MID-TRAINING</text>
<text x="350" y="138" text-anchor="middle" font-family="Newsreader,serif" font-size="12" fill="#e8f0ed">synthetic + high-quality annealing</text>
<rect x="200" y="170" width="300" height="50" rx="9" fill="#2c6e63" fill-opacity="0.6"/>
<text x="350" y="192" text-anchor="middle" font-weight="600" font-size="15" fill="#1a1410">SFT</text>
<text x="350" y="210" text-anchor="middle" font-family="Newsreader,serif" font-size="12" fill="#1a1410">instruction following + tool use + format</text>
<rect x="200" y="242" width="300" height="50" rx="9" fill="#c98a2b" fill-opacity="0.55"/>
<text x="350" y="264" text-anchor="middle" font-weight="600" font-size="15" fill="#1a1410">RFT (rejection sampling)</text>
<text x="350" y="282" text-anchor="middle" font-family="Newsreader,serif" font-size="12" fill="#1a1410">keep verifier-passing traces, 1-3 rounds</text>
<rect x="200" y="314" width="300" height="50" rx="9" fill="#b4451f" fill-opacity="0.85"/>
<text x="350" y="336" text-anchor="middle" font-weight="600" font-size="15" fill="#f4ede0">RLVR via GRPO</text>
<text x="350" y="354" text-anchor="middle" font-family="Newsreader,serif" font-size="12" fill="#f6e3da">reasoning, math, code, agents</text>
<rect x="200" y="386" width="300" height="50" rx="9" fill="#6a3a5e" fill-opacity="0.8"/>
<text x="350" y="408" text-anchor="middle" font-weight="600" font-size="15" fill="#f4ede0">RLHF / DPO / RLAIF</text>
<text x="350" y="426" text-anchor="middle" font-family="Newsreader,serif" font-size="12" fill="#f0e6ee">safety, style, helpfulness, refusals</text>
<line x1="350" y1="76"  x2="350" y2="96"  stroke="#8a3115" stroke-width="2" marker-end="url(#alarr)"/>
<line x1="350" y1="148" x2="350" y2="168" stroke="#8a3115" stroke-width="2" marker-end="url(#alarr)"/>
<line x1="350" y1="220" x2="350" y2="240" stroke="#8a3115" stroke-width="2" marker-end="url(#alarr)"/>
<line x1="350" y1="292" x2="350" y2="312" stroke="#8a3115" stroke-width="2" marker-end="url(#alarr)"/>
<line x1="350" y1="364" x2="350" y2="384" stroke="#8a3115" stroke-width="2" marker-end="url(#alarr)"/>
<text x="180" y="50"  text-anchor="end" font-family="JetBrains Mono,monospace" font-size="11" fill="#5a4f42">knowledge</text>
<text x="520" y="345" text-anchor="start" font-family="JetBrains Mono,monospace" font-size="11" fill="#5a4f42">capability</text>
<text x="520" y="411" text-anchor="start" font-family="JetBrains Mono,monospace" font-size="11" fill="#5a4f42">alignment</text>
</g>
</svg>
<figcaption>Each stage solves a different problem; the order matters. GRPO is inserted as a powerful new stage — it does not remove any of the others.</figcaption>
</figure>

## The alignment problem

After SFT your model answers questions. But it may answer them rudely, verbosely, unsafely, or just averagely. How do you teach it to give **preferred** responses?

Key insight: humans find it much easier to compare two answers than to write the ideal answer from scratch. So instead of collecting more (prompt, ideal response) pairs, you collect (prompt, response A, response B, which was better) **preference pairs**. Now the question is how to train on preference data.

## Pages in this section

- **[Reward models and Bradley-Terry](reward-models-bradley-terry.md)** — the 1952 statistical model that backs both RLHF and DPO.
- **[RL from zero and PPO](rl-from-zero-and-ppo.md)** — RL vocabulary, mapping onto LLMs, the four PPO models, clipped surrogate, KL penalty.
- **[DPO](dpo.md)** — the closed-form derivation that removed the RM and the RL loop.
- **[GRPO](grpo.md)** — group-relative advantages and why DeepSeek-R1 used it.
- **[RLVR](rlvr.md)** — verifiable rewards in math, code, and tool use.
- **[The GRPO hype, decoded](grpo-hype-decoded.md)** — what R1-Zero proved, what it did not, and the calibrated take.

For the runnable implementation of every method on a toy task, see [Alignment Walkthrough](../build-from-scratch/alignment-walkthrough.md).
