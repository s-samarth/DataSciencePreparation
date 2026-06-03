# Moment-Generating Functions

A moment-generating function is a distribution's fingerprint: one compact function that stores every moment
and turns the convolution of independent sums into simple multiplication. It is the cleanest path to proving
the central limit theorem.

!!! tip "Rapid Recall"
    The MGF is $M(t)=E(e^{tX})$, and Taylor-expanding $e^{tX}$ shows the moments sit inside it as
    coefficients. Differentiate $n$ times and set $t=0$ to extract the $n$-th moment, $E(X^n)=M^{(n)}(0)$. Two
    superpowers: MGFs are unique, so matching MGFs means matching distributions, and independent sums multiply
    their MGFs, $M_{X+Y}=M_X M_Y$. That converts painful density convolutions into multiplication, which is why
    sums of independent normals are normal and sums of Poissons are Poisson, and it is the engine of the MGF
    proof of the CLT.

## §3 Moment-Generating Functions (MGF)

A distribution's fingerprint that stores every moment. The MGF of $X$ is defined wherever this is finite near
$t=0$.

$$M(t) = E(e^{tX})$$

**Intuition, a fingerprint packed with moments.** The *moments* $E(X),\ E(X^2),\ E(X^3),\dots$ collectively
pin down a distribution's shape. The MGF is one compact function that secretly contains **all** of them.
Taylor-expand $e^{tX}$ and take expectations:

$$M(t) = E\!\left(\sum_{n=0}^{\infty}\frac{X^n t^n}{n!}\right) = \sum_{n=0}^{\infty}\frac{E(X^n)\,t^n}{n!}$$

The moments sit there as coefficients, hence "moment-generating."

### The extraction rule

Differentiate $n$ times and set $t=0$ to peel off the $n$-th moment:

$$E(X^n) = M^{(n)}(0)$$

Once gives the mean. Twice gives $E(X^2)$. Setting $t=0$ kills every term except the one whose coefficient is
$E(X^n)$.

### Two jobs

- **Extract moments without integrals.** The MGF is often a clean closed form; differentiating it beats grinding $\int x^2 f(x)\,dx$.
- **Identify distributions and prove sums.** Two facts make this the real superpower:

| Fact | Statement | Use |
|---|---|---|
| Uniqueness | same MGF $\Rightarrow$ same CDF | match an MGF to a known one to *identify* a distribution |
| Independent sums multiply | $M_{X+Y}(t) = M_X(t)\,M_Y(t)$ | find a sum's distribution by multiplying, then recognizing |

$$M_{X+Y}(t) = E(e^{t(X+Y)}) = E(e^{tX})E(e^{tY}) = M_X M_Y$$

The middle step needs **independence** (the $E(XY)=E(X)E(Y)$ property). This converts a painful density
convolution into simple multiplication, the reason "sum of independent normals is normal" and "sum of Poissons
is Poisson" are easy.

> **The one-paragraph mental summary.** The MGF is a distribution's unique fingerprint that stores every moment
> inside it. Differentiate it and set $t=0$ to pull out moments (easier than integrals). Because it is unique,
> matching MGFs means matching distributions. And because independent sums turn into MGF products, you prove
> "sum of these is that" by multiplying and recognizing, instead of convolving densities. That is the entire
> reason it earns its place in derivations.

**Payoff later.** The cleanest CLT proof rides on MGFs: show the standardized sum's MGF converges to
$e^{t^2/2}$ (the $\mathcal{N}(0,1)$ MGF), and by uniqueness the sum must be becoming normal. See
[Law of Large Numbers & CLT](lln-clt.md#9-the-central-limit-theorem) for the full argument.

## Interview Questions

**Q1: What is a moment-generating function and how do you extract the mean from it?**
The MGF is $M(t)=E(e^{tX})$, a single function whose Taylor coefficients are the moments of $X$. To extract
the $n$-th moment you differentiate $n$ times and evaluate at $t=0$, so the mean is $M'(0)$ and the second
moment is $M''(0)$. Setting $t=0$ isolates exactly the coefficient you want.

**Q2: Why are MGFs so useful for sums of independent random variables?**
Because the MGF of a sum of independent variables is the product of their MGFs, $M_{X+Y}=M_X M_Y$, which
follows from $E(e^{t(X+Y)})=E(e^{tX})E(e^{tY})$ under independence. This replaces convolving densities with
multiplying functions, making results like "sum of independent normals is normal" immediate.

**Q3: How does uniqueness of the MGF help identify a distribution?**
If two random variables share the same MGF, they share the same CDF, so an MGF determines the distribution
uniquely. In practice you compute the MGF of an unknown quantity, simplify it, and recognize it as a known
distribution's MGF, which identifies the distribution without further work.

**Q4: Sketch how MGFs prove the central limit theorem.**
Standardize the variables and form the sum's MGF, which by independence is the per-term MGF raised to the
$n$-th power. Taking logs and expanding shows that only the first two moments survive the $\sqrt{n}$ rescaling,
and the limit of the MGF is $e^{t^2/2}$, the MGF of a standard normal. By uniqueness, the standardized sum
converges to a normal.
