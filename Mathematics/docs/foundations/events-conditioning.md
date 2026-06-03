# Events, Independence & Conditioning

Once you can count, the next layer is reasoning about events: how to translate a word problem into set
operations, when two events are independent, and how learning that one event occurred reshapes the
probability of another. Conditioning is the grammar of machine learning, so this page is the hinge between
counting and everything that follows.

!!! tip "Rapid Recall"
    Word problems map to three set operations: "at least one" is a union, "none" is the complement of a
    union, and "all" is an intersection (often rewritten with De Morgan). The complement flip,
    $P(\text{at least one}) = 1 - P(\text{none})$, is the single most useful trick. Two events are
    independent when $P(A\cap B)=P(A)P(B)$, but joint independence of many events needs every subset to
    factorize, not just pairs. Conditional probability $P(A\mid B)=P(A\cap B)/P(B)$ shrinks your world to
    $B$. The multiplication and chain rules build joint events one factor at a time, and the law of total
    probability assembles $P(B)$ from a partition, which is exactly the denominator Bayes needs.

## §4 Translating English into Set Language

Real questions come in three flavors. Spotting which one you are asked is half the battle, because each
maps to a different operation.

- **"At least one occurs"** maps to the **union**: $P(A_1\cup\cdots\cup A_n)$. This is the inclusion-exclusion case.
- **"None occur"** maps to the complement of "at least one": $P(\text{none})=1-P(A_1\cup\cdots\cup A_n)$.
- **"All occur"** maps to the **intersection**, with a De Morgan rewrite: $P(A_1\cap\cdots\cap A_n)=1-P(A_1^{C}\cup\cdots\cup A_n^{C})$. ("All happen" equals "not (at least one fails)".)

> **Key takeaway.** The single most useful trick in probability is the **complement flip**:
> $P(\text{at least one})=1-P(\text{none})$. Computing "at least one" directly is painful; "none" is usually
> trivial. Compute the easy side, subtract from 1.

## §5 Independence of Events

Events $A$ and $B$ are **independent** if

$$P(A\cap B)=P(A)\,P(B)$$

Knowing that $A$ happened tells you *nothing* about $B$. The probability of both is just the product.

!!! warning "Interview trap"
    For many events to be jointly independent, it is *not* enough that every pair is independent. Every
    subset must factorize: every pair, every triple, all the way up. **Pairwise independence does not imply
    full independence.** Clean cases (separate dice, separate flips) satisfy full independence automatically.
    Notation: $P(A\cap B)$ is often written $P(A,B)$, where the comma means "and," seen everywhere in ML as
    joint probability $P(x,y)$.

### Worked problem: Newton-Pepys (1693)

!!! note "Question"
    Which is most likely? **(1)** at least one 6 in 6 dice, **(2)** at least two 6's in 12 dice, **(3)** at
    least three 6's in 18 dice. Pepys thought all equal (ratios 1/6, 2/12, 3/18). Newton disagreed.

Per die: $P(6)=\tfrac{1}{6}$, $P(\text{not }6)=\tfrac{5}{6}$. Dice independent, so multiply freely. Use the
complement flip each time.

**Case 1, at least one 6 in 6 dice:**

$$P(A)=1-\left(\tfrac{5}{6}\right)^{6}\approx 0.665$$

**Case 2, at least two 6's in 12 dice** (subtract "zero" and "exactly one"):

$$P(B)=1-\left(\tfrac{5}{6}\right)^{12}-12\cdot\tfrac{1}{6}\left(\tfrac{5}{6}\right)^{11}\approx 0.619$$

**Case 3, at least three 6's in 18 dice** (subtract zero, one, two, each a binomial term):

$$P(C)=1-\sum_{k=0}^{2}\binom{18}{k}\left(\tfrac{1}{6}\right)^{k}\left(\tfrac{5}{6}\right)^{18-k}\approx 0.597$$

**Answer:** $P(A)\approx 0.665 > P(B)\approx 0.619 > P(C)\approx 0.597$. The first bet wins. Newton was right.

> **Intuition.** Keeping the ratio (sixes needed)/(dice) constant does *not* keep the probability constant.
> "At least $k$" is a tail question, and the tail tightens as you scale up, a baby law of large numbers. More
> dice makes the *fraction* of sixes more predictable, shaving the upside probability.

## §6 Conditional Probability

The probability of $A$ *given* that $B$ occurred:

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}\qquad (P(B)>0)$$

> **Intuition.** Conditional probability is about **shrinking your world**. The moment you learn "$B$
> happened," you throw away every outcome where $B$ did not happen. Your universe shrinks to just $B$ (the
> denominator). Then you ask: what fraction of this smaller world is also $A$ (the numerator $P(A\cap B)$,
> the part of $A$ surviving inside $B$)?

**Concrete picture.** 100 people; 30 own a dog ($B$); of those, 12 also own a cat ($A\cap B$). Then
$P(\text{cat}\mid\text{dog})=\tfrac{12}{30}=0.4$: forget the 70 non-dog-owners; within the 30 dog-owners, 12
have cats. The "/100" cancels, and the world shrank from 100 to 30.

> **Key takeaway.** Almost all of ML is conditional. A classifier computes $P(\text{label}\mid\text{features})$.
> A language model computes $P(\text{next word}\mid\text{context})$. The whole enterprise is *updating beliefs
> given evidence*, and conditional probability is its grammar.

## §7 Multiplication & Chain Rule

Rearrange the conditional formula by multiplying through by $P(B)$. You can condition in either direction,
so there are two equivalent forms:

$$P(A\cap B)=P(B)\,P(A\mid B)=P(A)\,P(B\mid A)$$

*"Both happen" equals "B happens, then A given B" or "A happens, then B given A." Pick whichever direction
you have numbers for.*

**General chain rule:**

$$P(A_1\cap\cdots\cap A_n)=P(A_1)\,P(A_2\mid A_1)\,P(A_3\mid A_1,A_2)\cdots P(A_n\mid A_1,\dots,A_{n-1})$$

> **Intuition.** Build the joint event one piece at a time. This is exactly how autoregressive language
> models factorize a sentence: $P(\text{sentence})=P(w_1)P(w_2\mid w_1)P(w_3\mid w_1,w_2)\cdots$. The chain
> rule *is* the math behind next-token prediction.

## §8 Law of Total Probability

If $A_1,\dots,A_n$ **partition** the sample space (non-overlapping, covering everything), then for any
event $B$:

$$P(B)=\sum_{i=1}^{n}P(B\mid A_i)\,P(A_i)=P(B\mid A_1)P(A_1)+\cdots+P(B\mid A_n)P(A_n)$$

> **Intuition.** **Divide and conquer for probability.** $P(B)$ may be hard directly but easy *if you knew
> which scenario you were in*. Split the world into all scenarios $A_i$, compute $B$'s probability inside
> each, then average, weighted by how likely each scenario is. For example,
> $P(\text{rain})=P(\text{rain}\mid\text{summer})P(\text{summer})+P(\text{rain}\mid\text{winter})P(\text{winter})+\cdots$

> **Key takeaway.** This law builds $P(B)$, which is exactly the denominator in Bayes' theorem. That is the
> partnership: Bayes needs a denominator, total probability is how you compute it. A matched set. See
> [Bayes' Theorem](bayes.md) for the other half.

## Interview Questions

**Q1: What is the complement flip, and why is it so useful?**
It is the identity $P(\text{at least one}) = 1 - P(\text{none})$. "At least one" questions are painful to
compute directly because the events overlap, but "none occur" is usually a simple product of independent
complements. Compute the easy side and subtract from one.

**Q2: Does pairwise independence imply joint independence?**
No. Joint independence requires every subset of the events to factorize, every pair, every triple, and so on
up to the full set. It is possible to construct events that are pairwise independent yet not jointly
independent, which is a classic trap. Clean physical setups like separate dice or separate coin flips satisfy
full independence automatically.

**Q3: Explain conditional probability as "shrinking your world."**
Learning that $B$ occurred discards every outcome where $B$ did not happen, so the sample space shrinks to
$B$, which becomes the denominator. The numerator is $P(A\cap B)$, the part of $A$ that survives inside $B$.
The ratio $P(A\cap B)/P(B)$ is the fraction of the shrunken world that is also $A$.

**Q4: How does the chain rule connect to language models?**
The chain rule factorizes a joint probability into a product of conditionals, each one predicting the next
element given all previous elements. An autoregressive language model uses exactly this:
$P(\text{sentence})=P(w_1)P(w_2\mid w_1)P(w_3\mid w_1,w_2)\cdots$, which is next-token prediction written as
probability.

**Q5: How do the law of total probability and Bayes' theorem work together?**
The law of total probability assembles the marginal $P(B)$ by summing $P(B\mid A_i)P(A_i)$ over a partition
of the sample space. That marginal is precisely the denominator in Bayes' theorem, so total probability is
the tool that makes the Bayes update computable.
