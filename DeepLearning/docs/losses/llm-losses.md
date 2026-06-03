# LLM & Generation Losses

The losses behind language model pretraining and preference alignment. They are mostly cross-entropy in disguise, plus the preference-optimization family that aligns a model to human feedback.

!!! tip "Rapid Recall"
    Next-token cross-entropy is categorical cross-entropy applied at every position, predicting the next token; it is the entire foundation of GPT-style pretraining and equals maximum likelihood of the text. MLM is cross-entropy on masked positions (BERT). RLHF/PPO maximizes a learned reward while a KL penalty keeps the policy near a reference. DPO reformulates RLHF as a classification on preference pairs with no reward model. GRPO drops PPO's value function and normalizes rewards within a group of completions, used in recent reasoning models.

## §1 Next-Token Cross-Entropy (Language Modeling Loss)

| Symbol | Meaning |
| --- | --- |
| $x_t$ | Token at position $t$ in the sequence. |
| $x_{<t}$ | All tokens before position $t$ (the context). |
| $T$ | Sequence length. |

$$
L = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

Exactly categorical cross-entropy applied at every position in the sequence, predicting the next token from the vocabulary. Sum or mean over sequence positions. This is the entire foundation of GPT-style pretraining. It is equivalent to maximum likelihood estimation of $P(\text{text})$. Every modern LLM (GPT, LLaMA, Claude, Gemini) is trained with this loss at scale. The architectures that consume it are covered on the [architectures and attention](../sequences/architectures-attention.md) page.

## §2 Masked Language Modeling (MLM) Loss

Cross-entropy on the masked positions only. The foundation of BERT-style models. Less common now, most modern LLMs use causal LM, not MLM. Used in encoder-only models for representation learning.

## §3 RLHF / PPO Losses

Reinforcement Learning from Human Feedback. Maximize reward (from a reward model trained on human preferences) while penalizing KL divergence from a reference policy:

$$
L = -\mathbb{E}[r(x, y)] + \beta \cdot \mathrm{KL}(\pi \,\Vert\, \pi_{\text{ref}})
$$

Where $r$ is a learned reward model, $\pi$ is the current policy (the LLM being trained), and $\pi_{\text{ref}}$ is a frozen reference (typically the supervised fine-tuned model before RLHF). $\beta$ controls how far the policy can drift from the reference.

## §4 DPO (Direct Preference Optimization)

Reformulates RLHF as a classification problem on preference pairs. No reward model needed, it directly optimizes on preference data.

| Symbol | Meaning |
| --- | --- |
| $y_w$ | Preferred response (winning). |
| $y_l$ | Dispreferred response (losing). |
| $\pi$ | Current policy being optimized. |
| $\pi_{\text{ref}}$ | Reference policy. |
| $\beta$ | Strength of the KL constraint. |

$$
L_{\text{DPO}} = -\log\sigma\!\left(\beta \log\frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)
$$

Equivalent to RLHF but simpler, no separate reward model, no PPO machinery. Standard for modern preference learning.

## §5 GRPO (Group Relative Policy Optimization)

Used in DeepSeek-R1 and other recent reasoning models. Removes the value function from PPO. Normalizes rewards within a group of samples (multiple completions for the same prompt). Simpler than PPO, comparable results, much less memory.
