# Counting & the Sampling Table

Combinatorics is the engine room of elementary probability: nearly every elementary question reduces to
counting favorable outcomes and dividing by the total. This page builds the four sampling regimes from a
single logic spine, proves Vandermonde's identity by double counting, and develops inclusion-exclusion as
the formula for "at least one."

!!! tip "Rapid Recall"
    The master counting question is "how many ways to pick $k$ from $n$," answered by two switches: does
    order matter, and is replacement allowed. Ordered with replacement gives $n^k$; ordered without gives
    $n!/(n-k)!$; unordered without gives $\binom{n}{k}$; unordered with replacement gives $\binom{n+k-1}{k}$
    by stars and bars. Vandermonde's identity, $\binom{n+m}{k}=\sum_i \binom{n}{i}\binom{m}{k-i}$, falls out
    of counting one committee two ways. Inclusion-exclusion adds singles, subtracts pairs, adds triples, and
    so on, and it is the formula for the probability of at least one event.

## §1 The Sampling Table: How Many Ways?

Almost all of elementary probability reduces to **counting**: how many ways can a thing happen, divided by
how many things can happen in total. The master question is: *how many ways can you pick $k$ items from $n$
items?* The answer depends on two yes/no switches, does **order matter**, and can you **reuse** an item
(replacement)?

| Picking $k$ from $n$ | Ordered | Unordered |
|---|---|---|
| **With replacement** | $n^k$ | $\binom{n+k-1}{k}$ |
| **No replacement** | $\dfrac{n!}{(n-k)!}$ | $\binom{n}{k}=\dfrac{n!}{k!\,(n-k)!}$ |

Running example throughout: pick $k=2$ from $n=3$ items $\{A,B,C\}$.

### Ordered, with replacement → nᵏ

Pick $k$ times in sequence. Every pick has all $n$ items available (you put each back), and order matters
so $AB \neq BA$. Each of the $k$ slots independently takes any of $n$ values:

$$n \times n \times \cdots \times n = n^k$$

*Example: $3^2 = 9$, giving AA, AB, AC, BA, BB, BC, CA, CB, CC.*

> **Intuition.** A $k$-digit number in base $n$. This is also the size of a sequence space, for example
> the number of length-$k$ strings over an $n$-symbol vocabulary.

### Ordered, no replacement → n! / (n−k)!

Order matters, but each item is used up once picked. The choices shrink each step: $n$, then $n-1$, then
$n-2$, down to $(n-k+1)$:

$$n(n-1)(n-2)\cdots(n-k+1)=\frac{n!}{(n-k)!}$$

*We want only the first $k$ factors of $n!$, so we divide out the tail $(n-k)!$. Example: $\tfrac{3!}{1!}=6$,
giving AB, AC, BA, BC, CA, CB. This is the* **permutation** *count.*

### Unordered, no replacement → "n choose k"

Now $\{A,B\}=\{B,A\}$; we care only about *which set*. Start from the ordered case and remove the
over-counting: each unordered set of $k$ items was counted $k!$ times (once per ordering of those $k$
items). Divide it out:

$$\frac{1}{k!}\cdot\frac{n!}{(n-k)!}=\frac{n!}{k!\,(n-k)!}=\binom{n}{k}$$

*Example: $\tfrac{3!}{2!\,1!}=3$, giving $\{A,B\},\{A,C\},\{B,C\}$. The* **binomial coefficient**, *"n choose k."*

### Unordered, with replacement → "n+k−1 choose k"

Order off, but repeats allowed (AA, BB valid; AB = BA). The slick trick is **stars and bars**. Reframe:
instead of "which items," ask *how many times was each item picked?* A selection becomes a list of counts
summing to $k$.

Picture $k$ identical **stars** (the picks) and $n-1$ **bars** (dividers splitting them into $n$ bins, one
per item type):

```
★ ★ | |     →  (2,0,0) = AA
★ | ★ |     →  (1,1,0) = AB
| | ★ ★     →  (0,0,2) = CC
```

You have $k$ stars plus $(n-1)$ bars, so $(n+k-1)$ symbols. A selection is just a choice of *which positions*
are stars:

$$\binom{n+k-1}{k}$$

*Example: $\binom{4}{2}=6$, giving AA, BB, CC, AB, AC, BC.*

> **Key takeaway.** The logic spine: start at $n^k$ (most permissive). Kill replacement, so factors shrink,
> giving $n!/(n-k)!$. Kill order, so divide by $k!$ redundant arrangements, giving $\binom{n}{k}$. Bring
> replacement back without order, so reframe as counts, giving $\binom{n+k-1}{k}$. The two you will actually
> use in ML: $\binom{n}{k}$ (combinations everywhere) and $n^k$ (size of a sequence space).

## §2 Vandermonde's Identity

$$\binom{n+m}{k}=\sum_{i=0}^{k}\binom{n}{i}\binom{m}{k-i}$$

Proved by the **double-counting** technique: if two expressions count the same set of objects, they are
equal. Setup: a group of $n+m$ people, say $n$ men and $m$ women, from which we form a committee of size
$k$. Count the number of possible committees two ways.

**Way 1, the left-hand side.** Ignore the split. Choose $k$ from the whole pool of $n+m$: that is
$\binom{n+m}{k}$ by definition.

**Way 2, the right-hand side.** Split by how many men, $i$, are on the committee. $i$ ranges $0$ to $k$;
these cases do not overlap and cover everything. For a fixed $i$: choose $i$ men from $n$, which is
$\binom{n}{i}$, and the remaining $k-i$ seats are women from $m$, which is $\binom{m}{k-i}$. Multiply
(independent choices), then sum over all $i$:

$$\sum_{i=0}^{k}\binom{n}{i}\binom{m}{k-i}$$

Both ways count the same committees, so they are equal.

> **Intuition.** No factorials were expanded; the identity falls out purely from describing one set of
> objects from two angles. Impossible terms self-zero: $\binom{n}{i}=0$ when $i>n$, so the sum's bounds
> never need babysitting. Any time you see a sum of products of binomial coefficients, suspect someone
> partitioned a counting problem into cases.

## §3 The Inclusion-Exclusion Principle

To find the size (or probability) of a **union** of overlapping sets: add the singles, subtract the
pairwise overlaps, add back the triples, subtract the quadruples, and so on, alternating signs until you
run out of levels. It is the formula for "**probability of at least one**."

**Two sets:**

$$P(A\cup B)=P(A)+P(B)-P(A\cap B)$$

*Naively $P(A)+P(B)$ counts the overlap twice, so subtract it once.*

**Three sets:**

$$P(A\cup B\cup C)=P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)$$

**General form ($n$ sets):**

$$P\!\left(\bigcup_{i=1}^{n}A_i\right)=\sum_i P(A_i)-\sum_{i<j}P(A_i\cap A_j)+\sum_{i<j<k}P(A_i\cap A_j\cap A_k)-\cdots+(-1)^{n+1}P\!\left(\bigcap_{i=1}^{n}A_i\right)$$

> **Intuition.** The alternating signs are just bookkeeping: each correction over-corrects the next level,
> so you keep flipping the sign to fix the previous fix. For three sets, the triple-overlap center is added
> three times by the singles and subtracted three times by the pairs, netting zero, so it must be added back
> once. Add singles, subtract pairs, add triples, and so on.

### Worked problem: deMontmort's matching problem

!!! note "Question"
    Cards labeled $1,\dots,n$ are shuffled into random order. A "match" occurs at position $i$ if card $i$
    sits in slot $i$. What is the probability of **at least one match**?

Let $A_i$ = "card $i$ is in its correct slot." We want $P(A_1\cup\cdots\cup A_n)$.

- **Single match.** Pin card $i$; the other $n-1$ shuffle freely: $P(A_i)=\tfrac{(n-1)!}{n!}=\tfrac{1}{n}$. There are $n$ such terms, summing to $1$.
- **Two matches.** Pin two cards: $P(A_i\cap A_j)=\tfrac{(n-2)!}{n!}=\tfrac{1}{n(n-1)}$. Number of pairs $\binom{n}{2}=\tfrac{n(n-1)}{2!}$, so the pair-sum is $\binom{n}{2}\cdot\tfrac{1}{n(n-1)}=\tfrac{1}{2!}$. The $n(n-1)$ cancels, and this clean collapse is the magic.
- **General $k$.** $\binom{n}{k}\cdot\tfrac{(n-k)!}{n!}=\tfrac{1}{k!}$. Every level collapses to $\tfrac{1}{k!}$ regardless of $n$.
- **Assemble with alternating signs.**

$$P\!\left(\bigcup A_i\right)=\frac{1}{1!}-\frac{1}{2!}+\frac{1}{3!}-\cdots+(-1)^{n+1}\frac{1}{n!}$$

- **Recognize the series.** Since $e^{-1}=1-\tfrac{1}{1!}+\tfrac{1}{2!}-\tfrac{1}{3!}+\cdots$, the expression is $1-e^{-1}$ (truncated):

$$P\!\left(\bigcup A_i\right)\approx 1-e^{-1}\approx 0.632$$

**Answer:** about $1-\tfrac{1}{e}\approx 63.2\%$, and it converges there fast (locked by $n\approx 7$).

> **Key takeaway.** Whether you shuffle 10 cards or 10 million, the chance that at least one lands in its own
> slot stays about $63\%$. More cards means more chances to match, but each match gets rarer; the two effects
> almost perfectly cancel, parking the answer at $1-1/e$ forever.

## Interview Questions

**Q1: A problem says "order does not matter and items cannot repeat." Which counting formula applies, and why?**
That is the unordered, no-replacement regime, so the count is $\binom{n}{k}=\tfrac{n!}{k!(n-k)!}$. You start
from the ordered no-replacement count $n!/(n-k)!$ and divide by $k!$ because each unordered set of $k$ items
was counted once for every ordering of those items. This is the combination, "n choose k," and it is the one
that appears most often in machine learning.

**Q2: Derive Vandermonde's identity in one sentence of reasoning.**
Count the committees of size $k$ drawn from $n$ men and $m$ women two ways: directly as $\binom{n+m}{k}$, or
by splitting on the number of men $i$ and summing $\binom{n}{i}\binom{m}{k-i}$ over $i$. Both count the same
objects, so they are equal. No factorials need to be expanded.

**Q3: Why is the probability of "at least one match" in a shuffled deck essentially constant in $n$?**
By inclusion-exclusion, every level of the sum collapses to $1/k!$ independent of $n$, leaving the
alternating series $1 - 1/2! + 1/3! - \cdots$, which is the truncated expansion of $1 - e^{-1}$. More cards
add more chances to match, but each individual match becomes proportionally rarer, and the two effects cancel,
fixing the answer near $63.2\%$.

**Q4: When would you reach for inclusion-exclusion rather than a direct count?**
Use it whenever you need the probability or size of a union of overlapping events, especially "at least one"
questions where direct counting double counts the overlaps. Add the singles, subtract the pairwise overlaps,
add back the triples, and continue with alternating signs. If the complement "none occur" is easier, compute
that instead and subtract from one.
