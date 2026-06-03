# Paradoxes & Random Walks

The famous paradoxes are interview favorites because they punish memorized formulas and reward thinking
about the process that generated the data. This page works Monty Hall and Simpson's paradox, then turns to
random walks: gambler's ruin with its first-step analysis, and Polya's theorem on whether an infinite walk
ever returns home.

!!! tip "Rapid Recall"
    Monty Hall: switching wins two thirds of the time because the host's knowledge injects information into
    the other door; how the data was generated matters. Simpson's paradox: a doctor can win every operation
    yet lose overall when a confounder skews case mix, so always segment and hunt for confounders. Gambler's
    ruin: a first-step analysis gives a difference equation whose solution is $p_i=(1-(q/p)^i)/(1-(q/p)^n)$
    for a biased game and $p_i=i/n$ for a fair one, and a tiny edge repeated forever is destiny. Random walks
    are recurrent in 1D and 2D but transient in 3D and above (Polya), which is why a drunk man finds home but
    a drunk bird may not.

## §11 The Monty Hall Problem

!!! note "Setup"
    Three doors: one car, two goats. You pick a door. The host, who **knows** where the car is, opens a
    different door revealing a goat, then offers a switch. Should you switch?

**Yes, switching wins $\tfrac{2}{3}$ of the time; staying wins only $\tfrac{1}{3}$.**

**Intuition 1, your first pick was probably wrong, and that never changes.** Your pick was right with
probability $\tfrac{1}{3}$, wrong with $\tfrac{2}{3}$, fixed at choice time. The host was *always* going to
open a goat door (he knows where the car is), so his reveal gives *no* new info about *your* door. Your door
stays $\tfrac{1}{3}$. Since probabilities sum to 1, the entire remaining $\tfrac{2}{3}$ concentrates onto the
one other unopened door.

**Intuition 2, the 100-door amplifier.** 100 doors, one car. You pick one ($\tfrac{1}{100}$ chance). The
host opens 98 goats, leaving your door and one other. Your door was a wild 1-in-100 guess; the other is the
lone survivor of deliberate filtering. Obviously switch, it is the car $\tfrac{99}{100}$ of the time.

**Formal solve, law of total probability.** You picked Door 1. Let $S$ = "switching wins," $D_i$ = "car
behind door $i$," each $P(D_i)=\tfrac{1}{3}$.

- $P(S\mid D_1)=0$, car is behind your door; switching leaves it.
- $P(S\mid D_2)=1$, host must open Door 3; switching lands on the car.
- $P(S\mid D_3)=1$, host must open Door 2; switching lands on the car.

$$P(S)=0\cdot\tfrac{1}{3}+1\cdot\tfrac{1}{3}+1\cdot\tfrac{1}{3}=\tfrac{2}{3}$$

**Answer:** $P(\text{switch wins})=\tfrac{2}{3}$. Proven.

> **Key takeaway.** **How the data was generated matters.** The host's choice is constrained by knowledge,
> he never reveals the car. That constraint injects information into the other door. If he opened a door
> *randomly* and happened to show a goat, it genuinely would be 50/50. Same observation, different
> probability, because the *process* differs. You cannot compute $P(\cdot)$ without knowing the mechanism that
> produced your observation.

## §12 Simpson's Paradox

!!! note "Setup"
    Two doctors, two operations. Hibbert beats Nick at *each* operation, yet Nick has the better *overall*
    success rate. Both are true at once.

**Hibbert**

|  | heart | band-aid |
|---|---|---|
| success | 70 | 10 |
| failure | 20 | 0 |

**Nick**

|  | heart | band-aid |
|---|---|---|
| success | 2 | 81 |
| failure | 8 | 9 |

**Per operation, Hibbert wins both:**

- Heart: Hibbert $70/90=77.8\%$ vs Nick $2/10=20\%$.
- Band-aid: Hibbert $10/10=100\%$ vs Nick $81/90=90\%$.

**Overall, Nick wins:**

- Hibbert: $80/100=80\%$.
- Nick: $83/100=83\%$.

Hibbert is better at every operation, yet worse overall. The paradox is real, not arithmetic error.

**Why, the confounder.** Hibbert did mostly **heart surgeries** (90 of 100), hard, low success. Nick did
mostly **band-aids** (90 of 100), trivial, high success. Nick's gaudy overall number reflects easy cases,
not skill. **Operation type is a confounder** $C$, a hidden variable affecting both which doctor you get and
the success rate. Aggregating secretly compares "mostly-hard Hibbert" versus "mostly-easy Nick."

**In conditional-probability language.** With $A$ = success, $B$ = treated by Nick, $C$ = heart surgery:

$$P(A\mid B,C)<P(A\mid B^{c},C)\quad\text{and}\quad P(A\mid B,C^{c})<P(A\mid B^{c},C^{c})$$

$$\text{but}\quad P(A\mid B)>P(A\mid B^{c})$$

*Within each operation Nick loses; aggregated, the inequality flips. $C$ is the confounder.*

> **Key takeaway.** A constant production threat. In **A/B tests**, a variant can win overall but lose in
> every segment if traffic was split unevenly. In **fairness**, a model can look unbiased in aggregate while
> discriminating in every subgroup. **Defenses:** always segment; hunt for confounders before trusting any
> comparison; remember the fix is *causal*, not statistical, and which number is "right" depends on the
> question (choosing a heart surgeon? use the per-operation number). You cannot resolve it with more math,
> only by understanding how the data arose.

> **Key takeaway (shared with Monty Hall).** The numbers alone never tell you the answer; you must understand
> the process that generated the data. "Think about the data-generating process" is the single most valuable
> reflex in probabilistic reasoning, and exactly what interviewers probe with these two problems.

## §13 Random Walk & Gambler's Ruin

A **1D random walk** is a sequence of steps on the number line: move **+1 with probability $p$** or **-1
with probability $q=1-p$**. Repeated coin flips pushing left or right.

**Gambler's Ruin framing.** Players A and B bet \$1 per round. A wins a round with probability $p$ (takes
\$1 from B), loses with probability $q$. A starts with \$$i$, B with \$$(n-i)$, total \$$n$. Given A starts
with $i$, what is $P(\text{A wins everything})$?

**Setting up, first-step analysis.** Let $p_i=P(\text{A eventually wins}\mid\text{A has } i)$. Condition on
the next round (law of total probability): A wins it ($p$, to state $i+1$) or loses it ($q$, to state
$i-1$):

$$p_i=p\,p_{i+1}+q\,p_{i-1}$$

*A* **difference equation** *(discrete cousin of a differential equation). Boundaries: $p_0=0$ (A ruined),
$p_n=1$ (A won).*

**Solving it:**

- Guess $p_i=x^i$. Substitute and divide by $x^{i-1}$: $x=px^2+q$, i.e. $px^2-x+q=0$.
- Quadratic formula: $x=\dfrac{1\pm\sqrt{1-4pq}}{2p}$. Since $q=1-p$, the discriminant $1-4pq=(2p-1)^2$ is a perfect square, giving roots $x=1$ and $x=\tfrac{q}{p}$.
- Two distinct roots, so general solution $p_i=A\cdot 1^i+B\left(\tfrac{q}{p}\right)^i$ (for $p\neq q$).
- Apply boundaries. $p_0=0\Rightarrow B=-A$. $p_n=1\Rightarrow A\left(1-(q/p)^n\right)=1$.

**Result, biased game ($p\neq q$):**

$$p_i=\frac{1-(q/p)^i}{1-(q/p)^n}$$

**Result, fair game ($p=q=\tfrac{1}{2}$):** Here $q/p=1$ gives $0/0$; an L'Hopital limit of
$\dfrac{1-x^i}{1-x^n}$ as $x\to 1$ yields:

$$p_i=\frac{i}{n}$$

Your chance of winning everything is just your share of the money. 30% of the chips means 30% chance.

**The devastating table, $p=0.49$, equal start ($i=n/2$):**

| Total stake $N$ | $P(\text{A wins})$ |
|---|---|
| 20 | 0.40 |
| 100 | 0.12 |
| 200 | 0.02 |

A microscopic edge (49% vs 51%), starting with exactly half the money, yet at \$200 total, A wins only
**2%** of the time.

> **Key takeaway.** **A small edge, repeated forever, is destiny.** Over many steps the drift dominates the
> noise, the bias never sleeps. Counterintuitively, playing *longer* makes the disadvantaged player do
> *worse*: more rounds means more chances for the bias to express. This is why "I will keep playing until I am
> up" is a guaranteed losing strategy against any house edge. Used in: risk-of-ruin in trading
> (fees/slippage = the edge), insurance reserves, startup runway versus burn, genetic drift (allele fixation
> = $i/n$), and randomized-algorithm analysis.

## §14 Walks in 1D, 2D, 3D & Beyond

Drop the absorbing boundaries and ask of an *infinite* walk: **if you wander forever, are you guaranteed to
return to the start?** The answer flips with dimension, Polya's theorem.

| Dimension | Return probability | Behavior |
|---|---|---|
| 1D (line) | 1 | Recurrent, certain to return (infinitely often) |
| 2D (grid) | 1 | Recurrent, still certain, surprisingly |
| 3D (space) | $\approx 0.34$ | Transient, about 66% chance of never returning |
| $d\geq 4$ | keeps dropping | Transient, increasingly so |

> **Intuition.** The one-liner (Polya / Kakutani): *"A drunk man will find his way home, but a drunk bird may
> get lost forever."* The 2D/3D break is the deep fact, low dimensions trap you in recurrence, high
> dimensions set you free into transience. The boundary sits exactly between 2 and 3.

**Real-world cases by dimension:**

- **1D:** stock price or account equity, diffusion in a tube, queue length, inventory level, single-allele genetic drift, startup cash runway.
- **2D:** animal foraging, robot-vacuum random coverage (recurrence is *why* random bouncing eventually covers a floor), molecule diffusing on a membrane or catalyst surface, disease spread on a map.
- **3D:** gas molecule or perfume dispersing (transience is why smells do not reconcentrate), pollutant in air or ocean, neutron transport in a reactor, light scattering through fog or tissue (medical imaging, volumetric graphics), polymer-chain configuration.
- **High-d ($d\geq 4$), most ML-relevant:** **SGD is a biased random walk** through million-dimensional parameter space (gradient = drift, minibatch noise = kicks), and high-dimensional transience helps escape saddle points. **MCMC sampling** explores high-dimensional posteriors; random-walk Metropolis degrades with dimension, motivating HMC and NUTS. The **curse of dimensionality**, sparse vast spaces where points rarely revisit, is the same geometry.

**How the math changes.** Bounded 1D gives the clean quadratic and difference-equation solve above. For
return probability you instead sum over all ways to return and ask whether that series converges (transient)
or diverges (recurrent): it converges for $d\geq 3$, diverges for $d\leq 2$. Bounded higher-dimensional
versions analyze hitting probabilities of absorbing *surfaces* and typically need PDE methods (the continuous
limit is the heat or diffusion equation).

> **Key takeaway.** The reflex: whenever something explores a space randomly, money, molecules, robots,
> gradients, samplers, ask *"what dimension is this effectively in?"* That alone predicts whether it gets
> **trapped and revisits** (low-d) or **escapes and disperses** (high-d), and that qualitative split governs
> an enormous range of real systems.

## Interview Questions

**Q1: In Monty Hall, why is switching better, and what assumption makes it so?**
Switching wins two thirds of the time. Your first pick is right only one third of the time, and the host, who
knows where the car is, always reveals a goat, so his action gives no information about your door but
concentrates the remaining two thirds onto the other unopened door. The key assumption is that the host
knowingly avoids the car; if he opened a door at random, the problem would become 50/50.

**Q2: How can a doctor have a higher success rate on every operation yet a lower overall rate?**
Through a confounder in the case mix. If the better doctor mostly takes hard, low-success cases and the other
mostly takes easy, high-success cases, the aggregate rate reflects difficulty rather than skill, so the
per-segment winner can lose overall. The defense is to always segment and to identify confounders before
trusting an aggregate comparison.

**Q3: Set up and solve gambler's ruin for a fair game.**
A first-step analysis gives the difference equation $p_i=p\,p_{i+1}+q\,p_{i-1}$ with boundaries $p_0=0$ and
$p_n=1$. For a fair game $p=q=\tfrac12$ the solution is the linear $p_i=i/n$, so your probability of winning
everything equals your starting share of the money. For a biased game it is
$p_i=(1-(q/p)^i)/(1-(q/p)^n)$.

**Q4: Why does a tiny disadvantage become near-certain ruin over many rounds?**
Because drift accumulates while noise averages out: with $p=0.49$ and an equal start, the win probability
collapses from 0.40 at a stake of 20 to 0.02 at a stake of 200. The longer you play, the more the small bias
expresses itself, which is the mathematical reason a house edge eventually wins.

**Q5: What does Polya's theorem say about random walks across dimensions, and why does it matter for ML?**
A symmetric random walk is recurrent in one and two dimensions (it returns to the origin with probability
one) but transient in three or more dimensions (it may never return; the 3D return probability is about
0.34). This matters because stochastic gradient descent and MCMC samplers are high-dimensional walks, where
transience helps escape saddle points but also means points rarely revisit, the same geometry as the curse of
dimensionality.
