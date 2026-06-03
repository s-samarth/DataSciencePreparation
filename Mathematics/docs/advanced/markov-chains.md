# Markov Chains

A Markov chain is a system that hops between states where only the present matters: the future is
conditionally independent of the past given the current state. That single property turns a tangled sequence
problem into clean matrix algebra, and its long-run equilibrium is a left eigenvector.

!!! tip "Rapid Recall"
    The Markov property says the next state depends only on the current state, so past and future are
    conditionally independent given the present. A chain is specified by its states and a transition matrix
    $Q$ whose rows sum to 1. Evolution is matrix multiplication: one step is $sQ$, $m$ steps is $Q^m$. The
    stationary distribution solves $sQ=s$, which is a left-eigenvector equation with eigenvalue 1, the bridge
    from probability to linear algebra. For irreducible aperiodic chains it exists, is unique, and the chain
    converges to it from any start. PageRank is the stationary distribution of a giant web chain.

## §10 Markov Chains

A system hops between *states* over time, with one rule: the **future depends only on the present, not the
path that got there.** The present fully summarizes the past. That is the *Markov property*.

$$P(X_{n+1}=j \mid X_n=i,\ X_{n-1}=i_{n-1},\dots,X_0=i_0) = P(X_{n+1}=j \mid X_n=i)$$

In casual terms: past and future are **conditionally independent given the present**. If the rule does not
depend on time, the chain is *homogeneous* with transition probabilities $q_{ij} = P(X_{n+1}=j\mid X_n=i)$.

### Building blocks

- **States**, the situations the system can be in (nodes).
- **Transition matrix $Q$**, entry $(i,j)$ is $q_{ij}$; **every row sums to 1** (you must go somewhere).

States plus $Q$ fully specify the chain. Diagram and matrix carry the same information. **Why bother:** the
Markov property turns a tangled sequence problem into clean matrix algebra.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 560 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two-state Markov chain Working and Broken">
  <circle cx="150" cy="100" r="46" fill="#1b2029" stroke="#79c073" stroke-width="2"/>
  <text x="150" y="98" fill="#79c073" font-family="Fraunces, serif" font-size="16" text-anchor="middle">Working</text>
  <text x="150" y="116" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">W</text>
  <circle cx="410" cy="100" r="46" fill="#1b2029" stroke="#e0686f" stroke-width="2"/>
  <text x="410" y="98" fill="#e0686f" font-family="Fraunces, serif" font-size="16" text-anchor="middle">Broken</text>
  <text x="410" y="116" fill="#7c8593" font-family="monospace" font-size="10" text-anchor="middle">B</text>
  <path d="M192,82 Q280,40 368,82" fill="none" stroke="#e0a23b" stroke-width="1.6" marker-end="url(#ah)"/>
  <text x="280" y="48" fill="#e0a23b" font-family="monospace" font-size="11" text-anchor="middle">0.1</text>
  <path d="M368,118 Q280,160 192,118" fill="none" stroke="#54c7b8" stroke-width="1.6" marker-end="url(#ah)"/>
  <text x="280" y="168" fill="#54c7b8" font-family="monospace" font-size="11" text-anchor="middle">0.6</text>
  <path d="M120,62 Q110,12 150,30 Q165,38 150,54" fill="none" stroke="#79c073" stroke-width="1.4" marker-end="url(#ah)"/>
  <text x="108" y="30" fill="#79c073" font-family="monospace" font-size="11" text-anchor="middle">0.9</text>
  <path d="M440,62 Q450,12 410,30 Q395,38 410,54" fill="none" stroke="#e0686f" stroke-width="1.4" marker-end="url(#ah)"/>
  <text x="452" y="30" fill="#e0686f" font-family="monospace" font-size="11" text-anchor="middle">0.4</text>
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#a9b1bd"/>
    </marker>
  </defs>
</svg>
<figcaption>A 2-state chain (Working / Broken). Rows of Q sum to 1: W to {0.9, 0.1}, B to {0.6, 0.4}.</figcaption>
</figure>

### How the system evolves

From a distribution $s$ (a row vector), one step is multiplication by $Q$; $m$ steps is $Q^m$:

$$\text{dist of } X_{n+1} = sQ, \qquad P(X_{n+m}=j\mid X_n=i) = (Q^m)_{ij}$$

To find the chance of state $j$ next, sum over current states: $\sum_i q_{ij}s_i = sQ$. The Markov property
collapses multi-step history into a matrix power.

### Stationary distribution, the equilibrium

$$sQ = s$$

If the distribution ever reaches $s$, it stays forever, the long-run equilibrium. Crucially, $sQ=s$ is an
**eigenvector equation** ($s$ is a left eigenvector of $Q$ with eigenvalue 1), the bridge from probability to
linear algebra. This raises four natural questions: (1) does a stationary distribution exist for every chain?
(2) is it unique? (3) does the chain converge to it? (4) how do you compute it efficiently? For irreducible,
aperiodic chains the answers are: yes, unique, yes it converges from any start, and you solve the eigenvector
equation. See [Special Matrices & Definiteness](../linear-algebra/special-definite.md) for the Markov-matrix
view.

### Examples

- **Weather**, states {sunny, rainy}, transitions per day given the previous day. The stationary distribution is the long-run fraction of sunny versus rainy days.
- **Google PageRank**, the original killer app. States are web pages, transitions are clicking a random link. The stationary distribution is the long-run probability a random surfer lands on each page, the page's importance score. PageRank *is* the stationary distribution of a giant chain over the web.
- **Board games** (Monopoly, Snakes and Ladders), your square is the state, dice rolls are transitions. Which squares get landed on most? The stationary distribution.
- **Text generation**, states are words, transitions are which word tends to follow. A word-level Markov chain is the great-grandparent of language models, though modern **LLMs deliberately break the Markov property** by attending to long history.

### How to spot a Markov-chain question

- A system moves among a **finite set of distinct states**.
- Transitions are **probabilistic and given per-step**.
- The next step depends **only on the current state** (rules phrased "where you are now"). If it depends on the last two states, redefine the state.
- The question asks: state after $k$ steps (gives $Q^k$), long-run fractions (gives stationary $sQ=s$), or expected hitting or return time.

A worked two-step example:

$$Q = \begin{pmatrix} 0.9 & 0.1 \\ 0.6 & 0.4 \end{pmatrix}$$

W to W to W: $0.9\times 0.9 = 0.81$; W to B to W: $0.1\times 0.6 = 0.06$.

$$(Q^2)_{WW} = 0.81 + 0.06 = 0.87$$

And the stationary distribution, $sQ=s$ with $s=(\pi_W,\pi_B)$, $\pi_W+\pi_B=1$:

$$0.9\pi_W + 0.6\pi_B = \pi_W \;\Rightarrow\; 0.6\pi_B = 0.1\pi_W \;\Rightarrow\; \pi_W = 6\pi_B$$
$$6\pi_B + \pi_B = 1 \;\Rightarrow\; \pi_B = \tfrac17,\ \pi_W = \tfrac67$$

Finite states plus per-step probabilities plus a long-run question equals stationary distribution. As a second
check, if all 3 pads are interchangeable then the stationary distribution is uniform: check
$s=(\tfrac13,\tfrac13,\tfrac13)$, each component equals $\tfrac12\cdot\tfrac13+\tfrac12\cdot\tfrac13=\tfrac13$.

**Where Markov chains live.** **MDPs and RL** (a chain with actions plus rewards; value functions and the
Bellman equation rest on it), **Hidden Markov Models** (speech, POS-tagging), **MCMC** (design a chain whose
stationary distribution is the posterior, then sample: Gibbs, Metropolis-Hastings), **diffusion models**
(forward noising is a Markov chain). PageRank is the stationary distribution of the web.

## Interview Questions

**Q1: State the Markov property in words and in symbols.**
The Markov property says the next state depends only on the current state, not on the full history, so the
present summarizes the past. Symbolically,
$P(X_{n+1}=j\mid X_n=i, X_{n-1}=\dots) = P(X_{n+1}=j\mid X_n=i)$. Equivalently, past and future are
conditionally independent given the present.

**Q2: How do you compute the distribution after $k$ steps from an initial distribution?**
Represent the initial distribution as a row vector $s$ and multiply by the transition matrix once per step, so
after $k$ steps it is $sQ^k$, and the $k$-step transition probability from $i$ to $j$ is $(Q^k)_{ij}$. The
Markov property is what lets multi-step evolution collapse into a matrix power.

**Q3: What is a stationary distribution and how is it connected to linear algebra?**
A stationary distribution $s$ satisfies $sQ=s$, meaning once the chain reaches it the distribution never
changes. This is exactly a left-eigenvector equation for $Q$ with eigenvalue 1, so finding the long-run
behavior of a chain reduces to an eigenvector computation, which is how PageRank scores web pages.

**Q4: How do you recognize that a problem is a Markov chain, and what if the future depends on the last two states?**
Look for a finite set of states, probabilistic per-step transitions, and a next step that depends only on the
current state. If the dynamics depend on the last two states, you redefine the state to be the pair, which
restores the Markov property at the cost of a larger state space.
