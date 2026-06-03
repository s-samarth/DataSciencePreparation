# Discrete Distributions

Outcomes you can list and count. Probability lives in a PMF (probability **mass** function): the height at
each value *is* the probability of that value. This grid lines up the eight integer-outcome distributions by
parameters, story, use case, formula, and moments.

!!! tip "Rapid Recall"
    Bernoulli is the atom (one 0/1 trial), and the binomial counts successes across independent repeats with
    replacement. The hypergeometric is its without-replacement twin, where each draw changes the odds. The
    geometric waits for the first success and is discrete-memoryless; the negative binomial waits for the
    $r$-th and doubles as the over-dispersed count model. The Poisson counts rare events at a known rate with
    mean equal to variance. The discrete uniform encodes maximum ignorance over a finite set, and the
    multinomial generalizes the binomial to many categories, the likelihood behind softmax.

## The discrete grid

| Distribution | Parameters | What it models | When and why you use it | PMF / CDF | $E[X]$ | $\mathrm{Var}(X)$ |
|---|---|---|---|---|---|---|
| **Bernoulli** <br> $X\sim\text{Bern}(p)$ | $p$ = probability of success, $0\le p\le1$ | A single yes/no trial. One coin flip, one click-or-not. | The **atom** of probability, every other counting distribution is built from it. Use whenever an event has exactly two outcomes. <br> A binary label, a single binary feature, a dropout keep/drop mask. | $P(X=k)=p^k(1-p)^{1-k},\ k\in\{0,1\}$, i.e. $P(1)=p,\ P(0)=1-p$ | $p$ | $p(1-p)$ |
| **Binomial** <br> $X\sim\text{Bin}(n,p)$ | $n$ = number of independent trials; $p$ = success prob per trial | Count of successes in $n$ independent identical Bernoulli trials. "How many heads in $n$ flips." | Counting successes over **independent repeats** (sampling *with* replacement). <br> Correct predictions out of $n$; conversion counts in an A/B test. | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$, with $\binom{n}{k}=\frac{n!}{k!(n-k)!}$ | $np$ | $np(1-p)$ |
| **Hypergeometric** <br> $X\sim\text{HGeom}(N,K,n)$ | $N$ = population size; $K$ = successes in population; $n$ = draws (no replacement) | Successes in $n$ draws **without replacement**. "Draw n cards, how many aces?" | Sampling **without replacement** from a finite pool, where each draw changes the odds. The without-replacement twin of the Binomial. <br> Enrichment or over-representation tests (gene-set, feature significance). | $P(X=k)=\dfrac{\binom{K}{k}\binom{N-K}{\,n-k}}{\binom{N}{n}}$ | $n\dfrac{K}{N}$ | $n\dfrac{K}{N}\!\left(1-\dfrac{K}{N}\right)\dfrac{N-n}{N-1}$ (last factor is the finite-population correction) |
| **Geometric** <br> $X\sim\text{Geom}(p)$ | $p$ = success prob per trial | Number of trials up to and **including** the first success. "Flips until first head." | Waiting for the **first** success. The discrete memoryless distribution, past failures do not change future odds. <br> Run-lengths, discrete time-to-event, retry modeling. | $P(X=k)=(1-p)^{k-1}p,\ k=1,2,\dots$ (alt convention: failures *before* 1st success, $k=0,1,2,\dots$) | $\dfrac{1}{p}$ | $\dfrac{1-p}{p^2}$ |
| **Negative Binomial** <br> $X\sim\text{NB}(r,p)$ | $r$ = number of successes to wait for; $p$ = success prob per trial | Number of trials until the $r$-th success. Generalizes Geometric (which is $r=1$). | Waiting for $r$ successes. Also the go-to **over-dispersed count model** (variance > mean), where Poisson is too rigid. <br> Negative-binomial regression for count data with extra spread. | $P(X=k)=\binom{k-1}{r-1}p^r(1-p)^{k-r},\ k=r,r+1,\dots$ | $\dfrac{r}{p}$ | $\dfrac{r(1-p)}{p^2}$ |
| **Poisson** <br> $X\sim\text{Pois}(\lambda)$ | $\lambda$ = average events per interval (the rate) | Count of rare independent events in a fixed window of time or space, given a known average rate. "Emails per hour." | Counting events when you know the **average rate** and events are independent and rare. The limit of Binomial as $n\to\infty,\ p\to0$ with $np=\lambda$. Signature: **mean = variance**. <br> Poisson regression, arrival/queue modeling, count features. | $P(X=k)=\dfrac{e^{-\lambda}\lambda^{k}}{k!},\ k=0,1,2,\dots$ | $\lambda$ | $\lambda$ |
| **Discrete Uniform** <br> $X\sim\text{Unif}\{1..n\}$ | $n$ = number of equally likely outcomes | $n$ equally likely outcomes. A fair die. | Maximum ignorance over a **finite** set of equally-likely options, no reason to prefer any. <br> Random choice baselines, uniform sampling of indices. | $P(X=k)=\dfrac{1}{n}$ for each outcome | $\dfrac{n+1}{2}$ | $\dfrac{n^2-1}{12}$ |
| **Multinomial** (multivariate) <br> $X\sim\text{Mult}(n; p_1..p_k)$ | $n$ = trials; $p_i$ = prob of category $i$ ($\sum p_i=1$); $k$ = number of categories | Generalizes Binomial to $k>2$ outcomes. "Roll a die $n$ times, count each face." Output is a vector of counts. | Counts spread across **multiple categories**. <br> The likelihood behind softmax / multiclass classification; bag-of-words count vectors; topic models. | $P=\dfrac{n!}{x_1!\cdots x_k!}\,p_1^{x_1}\!\cdots p_k^{x_k}$ | $E[X_i]=np_i$ | $\mathrm{Var}(X_i)=np_i(1-p_i)$, $\mathrm{Cov}(X_i,X_j)=-np_ip_j$ |

See [More Discrete Distributions](../distributions/discrete-families.md) for the derivations behind the
geometric, negative binomial, and Poisson rows, and [Big Ideas](big-ideas.md) for the threads that tie them
together.

## Interview Questions

**Q1: What single switch turns a binomial into a hypergeometric, and what does it change?**
Removing replacement. The binomial assumes independent trials with a fixed success probability, while the
hypergeometric draws from a finite pool so each draw changes the remaining odds. The means coincide, but the
hypergeometric variance carries an extra finite-population correction factor $\frac{N-n}{N-1}$ that shrinks
it.

**Q2: Which discrete distribution has equal mean and variance, and why is that diagnostic?**
The Poisson, where both equal the rate $\lambda$. If you measure a count whose sample variance is close to
its mean, Poisson is a natural model; if the variance noticeably exceeds the mean, that over-dispersion points
you to the negative binomial instead.

**Q3: How are the geometric and negative binomial related?**
The geometric counts trials until the first success, and the negative binomial counts trials until the $r$-th
success, so the geometric is exactly the negative binomial with $r=1$. Equivalently, a negative binomial is a
sum of $r$ independent geometrics.

**Q4: Where does the multinomial show up in machine learning?**
It is the likelihood model behind multiclass classification: softmax outputs a category probability vector,
and the multinomial scores observed category counts against it. It also underlies bag-of-words count vectors
and topic models.
