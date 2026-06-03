# Matrices as Machines

A matrix is a machine that takes a vector in, processes it linearly, and hands a vector back. That single
picture, plus the fact that a linear map is fixed by where it sends the basis vectors, explains why matrices
exist and why their multiplication rule is forced rather than arbitrary.

!!! tip "Rapid Recall"
    A matrix is a linear machine: it obeys additivity and homogeneity, and any linear map on finite vectors is
    fully described by a finite grid of numbers. An $m\times n$ matrix maps $\mathbb{R}^n$ to $\mathbb{R}^m$,
    so shape reads as (out, in). Because a linear map is determined by where it sends the basis vectors, those
    landing spots stacked as columns *are* the matrix. Matrix-vector multiplication has two equal pictures, a
    linear combination of columns and a dot product per row, and matrix-matrix multiplication is either
    composition (do B then A) or batching.

## §1 Matrices as machines

A matrix is a machine that takes a vector in, processes it, and hands a vector back, with the one restriction
that the processing is *linear*. Start from an ordinary function. $f(x) = 2x$ takes a number, doubles it,
returns a number. Now let the input be a *list* of numbers (a vector) and the output also a list. A matrix is
exactly that machine, restricted to be **linear**, meaning it obeys two rules, "the machine plays no tricks":

$$f(u + v) = f(u) + f(v) \quad\text{and}\quad f(c\cdot v) = c\cdot f(v)$$

We obsess over linear machines because they are the simplest non-trivial functions, they are everywhere
(rotations, scalings, projections, the $Wx$ inside every neural-net layer), and, the punchline, **any linear
machine on finite vectors is fully described by a finite grid of numbers.** That grid is the matrix. We do not
invent matrices; they fall out as the minimum data needed to pin down a linear map.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Matrix as a machine mapping R-n to R-m">
      <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1d5b6e"/></marker></defs>
      <text x="60" y="40" font-family="monospace" font-size="13" fill="#444a5c">input vector</text>
      <rect x="40" y="55" width="70" height="110" rx="6" fill="#efe9da" stroke="#d9cfba"/>
      <text x="75" y="100" font-family="serif" font-size="18" fill="#1f2330" text-anchor="middle">x</text>
      <text x="75" y="125" font-family="monospace" font-size="11" fill="#444a5c" text-anchor="middle">n numbers</text>
      <line x1="120" y1="110" x2="245" y2="110" stroke="#1d5b6e" stroke-width="2" marker-end="url(#ar)"/>
      <rect x="250" y="45" width="150" height="130" rx="10" fill="#b1542f" />
      <text x="325" y="100" font-family="serif" font-size="26" fill="#fff" text-anchor="middle" font-style="italic">A</text>
      <text x="325" y="128" font-family="monospace" font-size="11" fill="#f7e6dc" text-anchor="middle">m × n  (out × in)</text>
      <text x="325" y="35" font-family="monospace" font-size="12" fill="#444a5c" text-anchor="middle">the linear machine</text>
      <line x1="402" y1="110" x2="525" y2="110" stroke="#1d5b6e" stroke-width="2" marker-end="url(#ar)"/>
      <rect x="530" y="62" width="70" height="96" rx="6" fill="#efe9da" stroke="#d9cfba"/>
      <text x="565" y="105" font-family="serif" font-size="18" fill="#1f2330" text-anchor="middle">Ax</text>
      <text x="565" y="128" font-family="monospace" font-size="11" fill="#444a5c" text-anchor="middle">m numbers</text>
    </svg>
<figcaption>The matrix is an operator: a vector enters, a (possibly different-dimensional) vector leaves.</figcaption>
</figure>

### Shape is the dimensions it maps between

An $m \times n$ matrix accepts an $n$-dimensional input and returns an $m$-dimensional output. It maps
$\mathbb{R}^n \to \mathbb{R}^m$.

$$(\text{rows}, \text{columns}) = (\text{out}, \text{in}) \quad\cdot\quad \text{columns eat the input, rows build the output}$$

That is the "changes dimensions" idea: a $3072 \times 768$ weight matrix takes a 768-dim token and blows it up
to 3072 dims.

### Why a matrix is "a collection of vectors"

This is the master key. A linear machine is **completely determined by where it sends the basis vectors**,
because every vector is a combination of basis vectors and linearity lets you push the machine through the
combination:

$$v = x\cdot e_1 + y\cdot e_2 \quad\Longrightarrow\quad T(v) = x\cdot T(e_1) + y\cdot T(e_2)$$

So if you know where $e_1$ and $e_2$ land, you know where *every* vector lands. Store $T(e_1)$ as column 1,
$T(e_2)$ as column 2. That is the whole matrix.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Basis vectors transformed into the columns of A">
      <g transform="translate(20,20)">
        <text x="120" y="-2" font-family="monospace" font-size="12" fill="#444a5c" text-anchor="middle">input space</text>
        <line x1="0" y1="200" x2="240" y2="200" stroke="#d9cfba"/>
        <line x1="0" y1="0" x2="0" y2="200" stroke="#d9cfba"/>
        <g stroke="#e7ddc8"><line x1="60" y1="0" x2="60" y2="200"/><line x1="120" y1="0" x2="120" y2="200"/><line x1="180" y1="0" x2="180" y2="200"/><line x1="0" y1="60" x2="240" y2="60"/><line x1="0" y1="120" x2="240" y2="120"/><line x1="0" y1="60" x2="240" y2="60"/></g>
        <line x1="0" y1="200" x2="60" y2="200" stroke="#b1542f" stroke-width="3"/>
        <line x1="0" y1="200" x2="0" y2="140" stroke="#1d5b6e" stroke-width="3"/>
        <text x="34" y="218" font-family="serif" font-size="14" fill="#b1542f">e₁</text>
        <text x="6" y="160" font-family="serif" font-size="14" fill="#1d5b6e">e₂</text>
      </g>
      <text x="330" y="135" font-family="serif" font-size="22" fill="#1f2330" text-anchor="middle">A&#160;⟶</text>
      <g transform="translate(380,20)">
        <text x="120" y="-2" font-family="monospace" font-size="12" fill="#444a5c" text-anchor="middle">output space</text>
        <line x1="0" y1="200" x2="240" y2="200" stroke="#d9cfba"/>
        <line x1="0" y1="0" x2="0" y2="200" stroke="#d9cfba"/>
        <g stroke="#e7ddc8"><line x1="60" y1="0" x2="60" y2="200"/><line x1="120" y1="0" x2="120" y2="200"/><line x1="180" y1="0" x2="180" y2="200"/><line x1="0" y1="60" x2="240" y2="60"/><line x1="0" y1="120" x2="240" y2="120"/></g>
        <line x1="0" y1="200" x2="120" y2="140" stroke="#b1542f" stroke-width="3"/>
        <line x1="0" y1="200" x2="60" y2="80" stroke="#1d5b6e" stroke-width="3"/>
        <text x="120" y="135" font-family="serif" font-size="13" fill="#b1542f">col&#8201;1 = A&#8201;e₁</text>
        <text x="64" y="76" font-family="serif" font-size="13" fill="#1d5b6e">col&#8201;2 = A&#8201;e₂</text>
      </g>
    </svg>
<figcaption>Each axis is dragged to a new spot. Those landing spots, stacked, are the columns of A.</figcaption>
</figure>

## §2 Matrix multiplication, both methods

### Matrix times vector

Because the columns are landing spots, the multiplication rule is *forced*, not arbitrary.

**Method 1, column picture (linear combination of columns).** The output is the input's entries used as
**weights on the columns of A**.

$$A\cdot v = v_1\cdot(\text{col }1) + v_2\cdot(\text{col }2) + \cdots$$

**Method 2, row picture (dot product per output coordinate).** Each output entry is the dot product of a
**row of A** with $v$. This is the "system of equations" view: each row is one equation, one measurement.

The column view tells you what outputs are reachable (the span of the columns). The row view is the lens for
solving systems. Both must agree, same operation, two angles.

### Matrix times matrix

Two distinct reasons you need it:

- **Composition**, "do B, then A" is itself one machine, and its matrix is $AB$: $(AB)v = A(Bv)$.
- **Batching**, stack many input vectors as columns of B; then $AB$ applies the same transform to all of them at once (this is $XW$ over a batch).

$$(m \times k)(k \times n) = (m \times n) \quad\cdot\quad \text{order matters: } AB \neq BA \text{ in general}$$

The inner dimensions must match.

## Interview Questions

**Q1: What makes a function "linear," and why does that pin it down to a grid of numbers?**
Linearity means the map satisfies additivity, $f(u+v)=f(u)+f(v)$, and homogeneity, $f(cv)=cf(v)$. Because every
vector is a combination of basis vectors, linearity lets you push the map through that combination, so the map
is completely determined by where it sends the basis. Storing those landing spots as columns gives the finite
grid of numbers that is the matrix.

**Q2: How do you read the shape of an $m\times n$ matrix?**
As (out, in): it accepts an $n$-dimensional input and returns an $m$-dimensional output, mapping
$\mathbb{R}^n$ to $\mathbb{R}^m$. The columns eat the input and the rows build the output, so a $3072\times768$
weight matrix takes a 768-dimensional vector to a 3072-dimensional one.

**Q3: Explain the column picture and the row picture of matrix-vector multiplication.**
In the column picture, the output is a linear combination of the matrix's columns weighted by the input
entries, which reveals the reachable outputs as the span of the columns. In the row picture, each output
coordinate is the dot product of a row with the input, which is the system-of-equations view used for solving.
Both produce the same result.

**Q4: Why is matrix multiplication associative but not commutative?**
It represents composition of linear machines, "do B then A," so chaining is associative because applying maps
in sequence does not depend on grouping. It is not commutative because the order of transformations matters:
rotating then scaling generally differs from scaling then rotating, so $AB\neq BA$ in general.
