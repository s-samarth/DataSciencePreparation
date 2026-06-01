# MLE and MAP — the Hidden Engine

Almost every loss function in machine learning is maximum likelihood estimation in disguise. Once this clicks, half of ML stops looking like a pile of unrelated formulas and starts looking like one recipe applied over and over. This is the conceptual keystone for everything downstream: SFT, reward modeling, DPO, even diffusion.

!!! tip "Rapid Recall"
    Every classification loss you have ever written is negative log-likelihood under some assumption. MSE = NLL with Gaussian noise. Binary cross-entropy = NLL with Bernoulli labels. Cross-entropy = NLL with a categorical distribution over the vocabulary. Reward-model loss = NLL with a Bradley-Terry preference model. The recipe is always the same: pick a probabilistic model of your data, write its likelihood, take the negative log, minimize with SGD. **MAP = MLE + a prior.** L2 regularization is MAP with a Gaussian prior on weights; L1 / Lasso is MAP with a Laplace prior. Every regularizer is a prior in disguise.

## §1 Probability vs likelihood

The distinction most textbooks butcher.

| | Probability | Likelihood |
|---|---|---|
| **Question** | Given known parameters, how likely is some data? | Given observed data, how plausible is each parameter value? |
| **Fixed** | The parameter \(\theta\) | The data |
| **Varies** | The data | The parameter \(\theta\) |
| **Notation** | \(P(\text{data} \mid \theta)\) | \(L(\theta \mid \text{data})\) |

For a coin with \(P(H)=0.7\), the probability of HHTH is \(0.7 \cdot 0.7 \cdot 0.3 \cdot 0.7 = 0.1029\). The likelihood that \(p=0.7\) given you observed HHTH is the *same number*, but you are asking a different question. The data is fixed; \(p\) is the variable.

!!! note "Critical distinction"
    Likelihood is **not** a probability distribution over \(\theta\). Integrate \(L(\theta)\) over all \(\theta\) and it does *not* equal 1. It is just a function scoring how plausible each parameter makes the observed data.

## §2 Worked example — 10 flips, 7 heads

The likelihood function is \(L(p) = p^7(1-p)^3\). The curve below shows that it peaks exactly at \(p = 0.7\), the empirical frequency. **MLE = pick the parameter at the peak.**

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 700 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Likelihood curve peaking at p=0.7">
<defs>
<linearGradient id="lfill" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="#b4451f" stop-opacity="0.35"/>
<stop offset="100%" stop-color="#b4451f" stop-opacity="0.02"/>
</linearGradient>
</defs>
<line x1="70" y1="280" x2="660" y2="280" stroke="#5a4f42" stroke-width="1.5"/>
<line x1="70" y1="40" x2="70" y2="280" stroke="#5a4f42" stroke-width="1.5"/>
<path d="M70,280 L100,279.7 L130,278.5 L160,275.8 L190,270.6 L220,262 L250,249 L280,231 L310,208 L340,180 L370,149 L400,118 L430,92 L460,77 L478,73 L490,75 L520,98 L550,143 L580,200 L610,256 L640,279 L660,280 Z" fill="url(#lfill)" stroke="#b4451f" stroke-width="2.5"/>
<line x1="478" y1="73" x2="478" y2="280" stroke="#2c6e63" stroke-width="1.5" stroke-dasharray="5 4"/>
<circle cx="478" cy="73" r="5" fill="#2c6e63"/>
<text x="478" y="62" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="#1d4a42" font-weight="600">MLE: p&#770; = 0.7</text>
<text x="70" y="300" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="#5a4f42">0.0</text>
<text x="247" y="300" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="#5a4f42">0.3</text>
<text x="365" y="300" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="#5a4f42">0.5</text>
<text x="478" y="300" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="#5a4f42">0.7</text>
<text x="660" y="300" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="#5a4f42">1.0</text>
<text x="365" y="325" text-anchor="middle" font-family="Newsreader, serif" font-style="italic" font-size="14" fill="#1a1410">candidate value of p (probability of heads)</text>
<text x="30" y="160" text-anchor="middle" font-family="Newsreader, serif" font-style="italic" font-size="14" fill="#1a1410" transform="rotate(-90 30 160)">L(p) = p&#8311;(1-p)&#179;</text>
</svg>
<figcaption>The likelihood as a function of the parameter. Detective logic: which value of p makes the data we actually saw least surprising?</figcaption>
</figure>

This is why "maximize the probability of observing the data we observed" is *not* circular. It does not mean "set probability to 1 because it happened." It means: **of all parameter values I could believe in, which one would have made the observed data the most expected?** Inference to the best explanation, formalized — like seeing a wet umbrella and concluding "it rained" because that hypothesis makes the observation least surprising.

## §3 The log-likelihood trick and the universal recipe

Products of probabilities underflow numerically, so we take logs (monotonic, so the argmax is unchanged) and flip the sign to get a loss to minimize — the **negative log-likelihood (NLL)**:

\[ \mathcal{L}(\theta) = -\sum_i \log P(x_i \mid \theta) \]

!!! abstract "Every loss is MLE in disguise"
    | Loss function | = NLL under this assumption |
    |---|---|
    | **MSE** (regression) | Gaussian noise on the targets |
    | **Binary cross-entropy** | Bernoulli labels |
    | **Cross-entropy** (LLM next-token) | Categorical distribution over the vocabulary |
    | **Reward-model loss** | Bradley-Terry preference model |

Same recipe each time: (1) pick a probabilistic model of your data, (2) write its likelihood, (3) take negative log, (4) minimize with SGD. Different assumption → different loss. That is the whole trick.

## §4 MAP and the L2 = Gaussian-prior insight

MLE overfits on small data: 3 flips of HHH gives \(\hat p = 1.0\), confidently wrong. **MAP** fixes this by injecting a prior via Bayes' rule, maximizing the posterior instead of the likelihood:

\[ \hat\theta_{MAP} = \arg\max_\theta \big[\, \log P(\text{data}\mid\theta) + \log P(\theta) \,\big] \]

**MAP = MLE + a prior term.** The canonical insight to internalize.

!!! note "L2 regularization IS MAP with a Gaussian prior"
    Start with MLE regression (gives MSE). Add a prior \(w \sim \mathcal{N}(0, \tau^2 I)\) — "weights should be small." Its log is \(-\frac{1}{2\tau^2}\lVert w\rVert^2\). The MAP objective becomes:

    \[ \mathcal{L}_{MAP}(w) = \sum_i (y_i - w^\top x_i)^2 + \lambda \lVert w\rVert^2, \qquad \lambda = \tfrac{\sigma^2}{\tau^2} \]

    That second term is ridge regression. Likewise **L1 / Lasso = MAP with a Laplace prior**. Every regularizer is a prior in disguise.

The hierarchy in one breath: **MLE** trusts the data alone; **MAP** trusts the data but regularizes toward prior belief (MLE is MAP with a uniform prior); **full Bayesian** keeps the entire distribution over \(\theta\) instead of one point estimate.

## §5 Where this shows up in the rest of the stack

Every loss in the alignment chapter has the same MLE shape underneath.

- [Reward Models](../alignment/reward-models-bradley-terry.md) — NLL under a Bradley-Terry preference model.
- [DPO](../alignment/dpo.md) — NLL under Bradley-Terry, but with the reward expressed in terms of the policy itself via the closed-form RLHF solution.
- [PPO clipped surrogate](../alignment/rl-from-zero-and-ppo.md) — not pure MLE, but advantage-weighted log-probabilities; the policy gradient theorem is itself a likelihood-ratio result.
- LLM next-token cross-entropy = NLL under a categorical distribution over the vocabulary.

If you ever feel like a loss came out of nowhere, ask "what distributional assumption makes this NLL?" and the answer is usually one line away.

## Interview Questions

**Q1: Is likelihood a probability?**

No. It is a function of \(\theta\) with the data fixed, and it does not integrate to 1 over \(\theta\). Its numerical value equals \(P(\text{data} \mid \theta)\), but the interpretation and the free variable differ.

**Q2: Why log-likelihood instead of likelihood?**

Three reasons. Numerical stability (avoids underflow on products of small probabilities), turns products into sums (easier to differentiate), and many exponential-family distributions have natural log forms.

**Q3: When does MLE fail?**

Small data (overfits with high variance), misspecified models (wrong distribution assumed), and unbounded likelihoods. MAP and regularization help the first; nothing saves you from a wrong model.

**Q4: What is the relationship between cross-entropy and KL divergence?**

Cross-entropy = entropy of the true distribution + KL(true ‖ model). Entropy is constant with respect to the parameters, so minimizing cross-entropy = minimizing KL = doing MLE. Three views, one optimization.

**Q5: Why is L2 a Gaussian prior and not Laplace?**

The quadratic penalty \(\lVert w \rVert^2\) matches the quadratic term in a Gaussian log-density; the absolute-value penalty \(\lVert w \rVert_1\) matches the linear term in a Laplace log-density. Same Bayesian story, different prior shape.

**Q6 (Trap): If MAP = MLE + prior, what prior gives ordinary MLE?**

A uniform prior over the parameter space. \(\log P(\theta) = \text{const}\) drops out of the argmax, leaving pure log-likelihood. So MLE is the special case of MAP with no prior information.
