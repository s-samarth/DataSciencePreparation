# Random Variables & Discrete Families

A random variable turns a messy outcome into the single number you actually care about, and that move
unlocks all of probabilistic arithmetic. This page defines random variables and random processes, then
builds the first members of the discrete family: Bernoulli, binomial, the probability mass function,
indicator variables, and the hypergeometric distribution.

!!! tip "Rapid Recall"
    A random variable is a function $X:S\to\mathbb{R}$, and the notation $X=1$ denotes the event of all
    outcomes mapping to $1$. Bernoulli is the atom, a single 0/1 trial with $P(X=1)=p$; the binomial counts
    successes in $n$ independent Bernoulli trials with PMF $\binom{n}{k}p^k(1-p)^{n-k}$. The PMF must be
    nonnegative and sum to one. An indicator is a Bernoulli flag for an event, and writing a binomial as a
    sum of indicators makes $E[X]=np$ fall out of linearity. The hypergeometric counts successes drawn
    without replacement, with PMF favorable over total, $\binom{w}{k}\binom{b}{n-k}/\binom{w+b}{n}$.

## §1 Random Variables & Random Processes

A random variable is a function

$$X : S \rightarrow \mathbb{R}$$

where $S$ is the sample space (all possible raw outcomes) and $\mathbb{R}$ is the real line. Plain English:
**a random variable reads an outcome and reports a number summarizing the part you care about.**

**Why we need it.**

- **You care about a number, not the full outcome.** Flipping 3 coins, you want "how many heads," not the exact sequence. $X=$ "number of heads" discards ordering and keeps the summary.
- **Math needs numbers.** Expectation, variance, every operation is arithmetic.
- **It compresses the sample space.** 8 raw outcomes of 3 flips collapse to 4 values (0,1,2,3).

**The mapping is fixed; the input is random.** Flip a fair coin twice, $S=\{HH,HT,TH,TT\}$, and let $X=$
number of heads:

$$X(HH)=2,\quad X(HT)=1,\quad X(TH)=1,\quad X(TT)=0$$

**Notation: "$X=1$" is secretly a set of outcomes.** We write $X=1$ to denote the *event*, the set of
outcomes that map to 1:

$$\{s \in S : X(s)=1\} = X^{-1}\{1\}$$

For the coin example, $X=1$ is the event $\{HT,TH\}$. So $P(X=1)$ asks "what is the probability of the set
of outcomes producing the value 1?" The $X^{-1}\{1\}$ is the *preimage*: "work backwards from the number to
the outcomes."

### Random processes, one draw versus a movie of draws

A random variable gives one number from one experiment. A **random process** (stochastic process) is a whole
family of random variables, usually indexed by time or step:

$$\{X_t : t \in T\}$$

One random variable is a snapshot; a random process is a movie. A *random walk* is the canonical example:
each step flips a coin, your position after $n$ steps is $X_n$, and the sequence $X_1,X_2,X_3,\dots$ is the
process.

## §2 Bernoulli Distribution

$X$ has the Bernoulli distribution if it takes only values 0 and 1, with some $p$ such that

$$P(X=1)=p \qquad P(X=0)=1-p$$

written $X \sim \text{Bern}(p)$. The symbol $\sim$ reads "is distributed as." Here $p$ is the probability of
"success" (which event counts as success is *your* modeling choice), and $1-p$ is everything else.

- **Logistic regression** outputs a probability $p$ and models the label as $\text{Bern}(p)$. Binary cross-entropy *is* the Bernoulli likelihood, that is why it is the loss.
- **Any binary classification target** is Bernoulli per example.
- **Dropout**: each neuron is kept with prob $p$, dropped with $1-p$, a Bernoulli mask per neuron.
- **A/B testing**: whether one user converts is a Bernoulli trial.

## §3 Binomial Distribution

The number of successes in $n$ independent $\text{Bern}(p)$ trials:

$$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}, \qquad 0\le k\le n$$

written $X \sim \text{Bin}(n,p)$.

**Reading the formula.** "Probability of exactly $k$ successes = (ways to arrange $k$ successes among $n$
slots) times (probability of any one such arrangement)."

- $p^k$, $k$ successes, each prob $p$, multiplied (independence).
- $(1-p)^{n-k}$, the remaining $n-k$ must be failures.
- $\binom{n}{k}$, counts the arrangements; all share the same probability, so multiply.

$$\binom{n}{k}=\frac{n!}{k!\,(n-k)!}$$

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 640 260" role="img" aria-label="Binomial PMF bar charts for two parameter settings">
      <g transform="translate(40,20)">
        <text x="130" y="0" fill="#5ad1c4" font-family="'Trebuchet MS',sans-serif" font-size="13" text-anchor="middle">Bin(10, 0.5) — symmetric</text>
        <line x1="0" y1="180" x2="260" y2="180" stroke="#2a323d"/>
        <g fill="#ffb347">
          <rect x="6"  y="178" width="18" height="2"/>
          <rect x="30" y="166" width="18" height="14"/>
          <rect x="54" y="135" width="18" height="45"/>
          <rect x="78" y="92"  width="18" height="88"/>
          <rect x="102" y="62" width="18" height="118"/>
          <rect x="126" y="51" width="18" height="129"/>
          <rect x="150" y="62" width="18" height="118"/>
          <rect x="174" y="92" width="18" height="88"/>
          <rect x="198" y="135" width="18" height="45"/>
          <rect x="222" y="166" width="18" height="14"/>
          <rect x="246" y="178" width="18" height="2"/>
        </g>
        <text x="15" y="196" fill="#7d8a98" font-size="10" font-family="sans-serif">0</text>
        <text x="131" y="196" fill="#7d8a98" font-size="10" font-family="sans-serif">5</text>
        <text x="250" y="196" fill="#7d8a98" font-size="10" font-family="sans-serif">10</text>
      </g>
      <g transform="translate(360,20)">
        <text x="130" y="0" fill="#5ad1c4" font-family="'Trebuchet MS',sans-serif" font-size="13" text-anchor="middle">Bin(10, 0.2) — right-skewed</text>
        <line x1="0" y1="180" x2="260" y2="180" stroke="#2a323d"/>
        <g fill="#9d8cff">
          <rect x="6"  y="73"  width="18" height="107"/>
          <rect x="30" y="46"  width="18" height="134"/>
          <rect x="54" y="78"  width="18" height="102"/>
          <rect x="78" y="123" width="18" height="57"/>
          <rect x="102" y="156" width="18" height="24"/>
          <rect x="126" y="172" width="18" height="8"/>
          <rect x="150" y="178" width="18" height="2"/>
          <rect x="174" y="179" width="18" height="1"/>
          <rect x="198" y="180" width="18" height="0.5"/>
          <rect x="222" y="180" width="18" height="0.3"/>
          <rect x="246" y="180" width="18" height="0.2"/>
        </g>
        <text x="15" y="196" fill="#7d8a98" font-size="10" font-family="sans-serif">0</text>
        <text x="131" y="196" fill="#7d8a98" font-size="10" font-family="sans-serif">5</text>
        <text x="250" y="196" fill="#7d8a98" font-size="10" font-family="sans-serif">10</text>
      </g>
    </svg>
<figcaption>The binomial PMF piles probability on each possible count of successes. A fair coin (p=0.5) is symmetric around the mean np=5; a rare event (p=0.2) skews left toward np=2.</figcaption>
</figure>

Worked example: fixed $n=5$, independent, same $p=0.5$, count successes gives $X\sim\text{Bin}(5,0.5)$, want
$P(X=3)$.

$$P(X=3)=\binom{5}{3}(0.5)^3(0.5)^{2}$$
$$\binom{5}{3}=\frac{5!}{3!\,2!}=\frac{120}{12}=10$$
$$(0.5)^3(0.5)^2=(0.5)^5=\tfrac{1}{32}$$
$$P(X=3)=10\times\tfrac{1}{32}=\tfrac{5}{16}=0.3125$$

And $X\sim\text{Bin}(8,0.2)$, want $P(X=2)$:

$$P(X=2)=\binom{8}{2}(0.2)^2(0.8)^{6}$$
$$\binom{8}{2}=\frac{8\times7}{2}=28$$
$$(0.2)^2=0.04,\qquad (0.8)^6=0.262144$$
$$P(X=2)=28\times0.04\times0.262144\approx0.2936$$

Setting $n=1$ recovers Bernoulli:

$$P(X=k)=\binom{1}{k}p^k(1-p)^{1-k},\quad k\in\{0,1\}$$

gives $\binom{1}{1}p^1(1-p)^0=p$ and $\binom{1}{0}p^0(1-p)^1=1-p$, exactly Bernoulli.

## §4 Probability Mass Function (PMF)

$$f_X(x)=P(X=x)$$

Feed it a value $x$; it returns the probability $X$ equals exactly that. The subscript labels which
variable's PMF it is.

**Two properties every PMF must satisfy:**

$$f_X(x)\ge 0 \quad\text{for all }x \qquad\text{and}\qquad \sum_x f_X(x)=1$$

Probabilities cannot be negative; all the mass adds to the full kilogram (the variable takes *some* value).

The PMFs of our distributions, explicit. For $X\sim\text{Bern}(p)$:

$$f_X(x)=\begin{cases}p & x=1\\ 1-p & x=0\end{cases}$$

For $X\sim\text{Bin}(n,p)$:

$$f_X(k)=\binom{n}{k}p^k(1-p)^{n-k},\quad k=0,1,\dots,n$$

## §5 Indicator Random Variables

$$X_i=\begin{cases}1 & \text{if the } i\text{th trial succeeds}\\ 0 & \text{otherwise}\end{cases}$$

An indicator *is* a $\text{Bern}(p)$. "Indicator" stresses its role (flagging an event); "Bernoulli"
stresses its distribution. Same object.

**The key trick: binomial as a sum of indicators.**

$$X = X_1 + X_2 + \dots + X_n$$

where each $X_i$ is i.i.d. $\text{Bern}(p)$. **i.i.d.** means independent and identically distributed. Each
$X_i$ contributes a 1 exactly when trial $i$ succeeds, so the sum *is* the count of successes.

$$E[X]=E[X_1]+\dots+E[X_n]=\underbrace{p+\dots+p}_{n}=np$$

- **Accuracy is a mean of indicators**: $X_i=1$ if prediction $i$ correct, so accuracy $=\frac1n\sum X_i$. Precision, recall, hit rate, conversion rate are all indicator averages.
- **0/1 loss** is an indicator (1 if misclassified).
- **One-hot encoding** is a vector of indicators.
- **Counting via expectation**: expected collisions, expected users who do X, write as a sum of indicators, use linearity.

**Two quick results.** The binomial PMF sums to 1:

$$\sum_{k=0}^{n}\binom{n}{k}p^k q^{n-k}=(p+q)^n=1$$

with $q=1-p$, so $p+q=1$ and $1^n=1$. Some number of successes definitely happens; the binomial theorem
makes it automatic. And the sum of independent same-$p$ binomials is binomial:

$$X\sim\text{Bin}(n,p),\ Y\sim\text{Bin}(m,p)\ \text{independent}\ \Rightarrow\ X+Y\sim\text{Bin}(n+m,p)$$

Story proof: pool $n$ flips and $m$ flips at the same $p$ into $n+m$ flips. The $p$ must match. As a
preview of expectation under dependence, let $X_i=1$ if card $i$ is an ace and $X=X_1+\dots+X_5$. By
symmetry any drawn card is an ace with prob $\tfrac{4}{52}=\tfrac1{13}$, so $E[X_i]=\tfrac1{13}$ and

$$E[X]=5\times\tfrac{1}{13}=\tfrac{5}{13}\approx 0.385$$

## §6 Hypergeometric Distribution

$w$ white and $b$ black marbles; draw $n$ without replacement; $X=$ number of white drawn:

$$P(X=k)=\frac{\dbinom{w}{k}\dbinom{b}{n-k}}{\dbinom{w+b}{n}}$$

written $X\sim\text{HGeom}(w,b,n)$, valid for $0\le k\le w$ and $0\le n-k\le b$.

**Reading the formula, favorable over total.**

- $\binom{w}{k}$, ways to choose $k$ whites from $w$.
- $\binom{b}{n-k}$, ways to choose the remaining $n-k$ blacks from $b$; multiply since both must happen.
- $\binom{w+b}{n}$, total ways to draw any $n$ (the full sample space).

The card example ($X$ = number of aces in a 5-card hand). Aces are "white" ($w=4$), non-aces "black"
($b=48$), draw $n=5$:

$$P(X=k)=\frac{\dbinom{4}{k}\dbinom{48}{5-k}}{\dbinom{52}{5}},\quad k=0,1,2,3,4$$

It is a valid PMF by [Vandermonde's identity](../foundations/counting.md#2-vandermondes-identity):

$$\sum_{k=0}^{w}\frac{\binom{w}{k}\binom{b}{n-k}}{\binom{w+b}{n}}=\frac{\binom{w+b}{n}}{\binom{w+b}{n}}=1$$

Worked example: drawing without replacement, $\text{HGeom}(w{=}4,b{=}8,n{=}3)$, want $k=1$:

$$P(X=1)=\frac{\dbinom{4}{1}\dbinom{8}{2}}{\dbinom{12}{3}}$$
$$\binom{4}{1}=4,\quad \binom{8}{2}=28,\quad \binom{12}{3}=220$$
$$P(X=1)=\frac{4\times28}{220}=\frac{112}{220}=\frac{28}{55}\approx0.509$$

When the population dwarfs the sample, the hypergeometric is well approximated by the binomial. Drawing
without replacement from a finite labeled pool $\text{HGeom}(3000,7000,100)$: the population (10,000) is 100
times the sample, far above the roughly 20 times rule, so removing 100 barely shifts the 30%. Treat it as
$\text{Bin}(100,0.3)$; the means are identical ($np=30$) and the hypergeometric variance is only slightly
smaller (the finite-population correction).

## Interview Questions

**Q1: What is a random variable, formally, and what does $P(X=1)$ really mean?**
A random variable is a function $X:S\to\mathbb{R}$ from the sample space to the reals that summarizes an
outcome as a number. The expression $X=1$ denotes the event $\{s\in S: X(s)=1\}$, the preimage of the value
under $X$, so $P(X=1)$ is the probability of that set of outcomes. The mapping is fixed; the randomness lives
in which outcome occurs.

**Q2: Why is the expected value of a binomial $np$, and what makes the argument so clean?**
Write the binomial as a sum of $n$ i.i.d. indicator variables, one per trial, each with expectation $p$. By
linearity of expectation, $E[X]=\sum E[X_i]=np$, with no need to manipulate the binomial coefficient.
Linearity holds regardless of dependence, which is why the same indicator trick computes expected counts even
when the indicators are correlated.

**Q3: When can you approximate a hypergeometric distribution by a binomial, and why?**
When the population is large relative to the sample, roughly twenty times or more, removing drawn items
barely changes the success proportion, so sampling without replacement behaves almost like sampling with
replacement. The means match exactly, and the hypergeometric variance is only slightly smaller due to the
finite-population correction.

**Q4: What is the difference between an indicator variable and a Bernoulli variable?**
They are the same object viewed two ways. "Indicator" emphasizes the role of flagging whether an event
occurred, returning 1 or 0, while "Bernoulli" emphasizes the distribution with success probability $p$. The
indicator of an event $A$ is distributed $\text{Bern}(P(A))$, and its expectation equals $P(A)$.
