# Eigenvalues, Pivots & Rank

Eigenvectors are the special directions a matrix only stretches, pivots are the anchors of row reduction, and
rank is the count of genuinely independent directions a matrix carries. Together they describe what a matrix
really does underneath its apparent complexity.

!!! tip "Rapid Recall"
    An eigenvector satisfies $Av=\lambda v$: it keeps its direction and is only scaled, and repeated
    application gives $A^k v=\lambda^k v$, which governs Markov chains, PageRank, and stability. Eigenvalues
    come from the characteristic equation $\det(A-\lambda I)=0$. Pivots are the leading nonzero entries in
    echelon form, and their count is the rank. Rank is the matrix's true working dimension, equal across
    independent columns, independent rows, pivots, and nonzero singular values. A sum of $r$ rank-1 outer
    products has rank at most $r$, so low rank means buildable from a few underlying factors.

## §3 Eigenvalues & eigenvectors

Most vectors get rotated *and* stretched by a matrix. Eigenvectors are the special directions that only get
stretched, no turning. Picture stretching a rubber sheet diagonally. Most painted arrows swing to point
somewhere new. The arrows lying along the pure stretch and squeeze axes just get longer or shorter without
turning. Those are the eigenvectors; the stretch factors are the eigenvalues.

$$A\cdot v = \lambda\cdot v \quad (v \neq 0)$$

An eigenvector is really a **direction**: if $v$ works, so does any multiple of it, with the same $\lambda$.
That is why we find a line (and often normalize it).

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 640 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Generic vectors rotate, eigenvector only scales">
      <g transform="translate(40,20)">
        <text x="110" y="-2" font-family="monospace" font-size="12" fill="#444a5c" text-anchor="middle">before A</text>
        <circle cx="110" cy="100" r="3" fill="#1f2330"/>
        <line x1="110" y1="100" x2="190" y2="60" stroke="#9aa0ad" stroke-width="2.5"/>
        <line x1="110" y1="100" x2="60" y2="40" stroke="#9aa0ad" stroke-width="2.5"/>
        <line x1="110" y1="100" x2="185" y2="148" stroke="#b1542f" stroke-width="3"/>
        <text x="188" y="165" font-family="serif" font-size="13" fill="#b1542f">eigenvector</text>
      </g>
      <text x="320" y="110" font-family="serif" font-size="20" fill="#1f2330" text-anchor="middle">A ⟶</text>
      <g transform="translate(380,20)">
        <text x="110" y="-2" font-family="monospace" font-size="12" fill="#444a5c" text-anchor="middle">after A</text>
        <circle cx="110" cy="100" r="3" fill="#1f2330"/>
        <line x1="110" y1="100" x2="205" y2="95" stroke="#9aa0ad" stroke-width="2.5"/>
        <line x1="110" y1="100" x2="95" y2="30" stroke="#9aa0ad" stroke-width="2.5"/>
        <text x="150" y="60" font-family="serif" font-size="11" fill="#777" >turned</text>
        <line x1="110" y1="100" x2="215" y2="167" stroke="#b1542f" stroke-width="3"/>
        <text x="150" y="180" font-family="serif" font-size="13" fill="#b1542f">same line, scaled by λ</text>
      </g>
    </svg>
<figcaption>Grey vectors change direction; the eigenvector stays on its own line and is only rescaled.</figcaption>
</figure>

### Why we care

Eigenvectors are the matrix's **natural axes**, the coordinate system in which the machine stops mixing and is
pure stretching. Concretely:

- **Repeated application is trivial:** $A^k v = \lambda^k v$. Governs Markov chains, PageRank, dynamics, the largest $|\lambda|$ dominates ($>1$ blow-up, $<1$ decay, $=1$ steady state).
- **Stability and conditioning:** the Hessian's eigenvalues are curvatures; the ratio largest/smallest is the **condition number** that makes gradient descent zig-zag.
- **PCA** is the eigendecomposition of the covariance matrix, eigenvectors are the directions of maximum variance, eigenvalues are how much.

### How to find them, step by step

$$A\cdot v = \lambda\cdot v \;\Longrightarrow\; (A - \lambda I)\cdot v = 0$$

We need a **nonzero** $v$ crushed to zero by $(A - \lambda I)$. A matrix that crushes a nonzero vector is
**singular**, which happens exactly when its determinant is zero (the determinant is the volume-scaling
factor; zero means a dimension collapsed):

$$\det(A - \lambda I) = 0$$

- Form $A - \lambda I$.
- Set its determinant to zero, giving a polynomial in $\lambda$.
- Solve for $\lambda$ (the eigenvalues).
- For each $\lambda$, solve $(A - \lambda I)v = 0$ for the eigenvector(s).

## §4 Pivots

A completely separate idea from eigenvalues. When you row-reduce a matrix into staircase (echelon) form, the
**pivot** is the leading nonzero entry of each row, the anchor you eliminate with. Columns with a pivot are
**pivot columns**; columns without one are **free columns**.

**Where pivots are used.**

- **Number of pivots equals rank**, the true working dimension.
- **Pivot columns** form a basis for the column space; **free columns** count the null space.
- **Solvability:** unique, infinite, or no solution depending on pivots and free columns.
- **Invertibility:** an $n\times n$ matrix is invertible if and only if it has $n$ pivots.
- **Determinant** equals the product of pivots (with sign for row swaps); pivots are the diagonal of U in LU.

## §5 Rank & low rank

Rank is the number of genuinely independent directions a matrix carries, the dimension of its column space. A
$3\times3$ matrix whose three columns all lie in a plane has rank 2: it is a 2D machine wearing a 3D costume.
Rank is the matrix's *true* working dimension. All of these are the same number:

- independent columns,
- independent rows,
- pivots,
- nonzero singular values,
- dimension of the column space.

**Full rank** is the maximum $\min(m, n)$. For a square matrix that means invertible, nonzero determinant, no
zero eigenvalues. **Low rank (rank deficient)** means some columns are redundant; the matrix collapses
dimensions and loses information.

### How a matrix becomes low rank, the outer product

The key fact: **a sum of r rank-1 outer products has rank at most r.** So "low rank" literally means
"buildable from a few underlying factors."

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Low rank as outer product">
      <rect x="40" y="35" width="40" height="80" rx="4" fill="#b1542f"/>
      <text x="60" y="132" font-family="serif" font-size="14" text-anchor="middle" fill="#1f2330">u</text>
      <text x="98" y="80" font-family="serif" font-size="22" fill="#444a5c">×</text>
      <rect x="120" y="55" width="120" height="22" rx="4" fill="#1d5b6e"/>
      <text x="180" y="98" font-family="serif" font-size="14" text-anchor="middle" fill="#1f2330">vᵀ</text>
      <text x="262" y="80" font-family="serif" font-size="22" fill="#444a5c">=</text>
      <rect x="295" y="35" width="120" height="80" rx="4" fill="none" stroke="#d9cfba"/>
      <g fill="#e7c9b8"><rect x="295" y="35" width="120" height="80" opacity="0.5"/></g>
      <text x="355" y="80" font-family="serif" font-size="13" text-anchor="middle" fill="#1f2330">rank-1 block</text>
      <text x="470" y="62" font-family="monospace" font-size="12" fill="#444a5c">a full grid…</text>
      <text x="470" y="84" font-family="monospace" font-size="12" fill="#444a5c">…carrying only</text>
      <text x="470" y="106" font-family="monospace" font-size="12" fill="#b1542f">one direction</text>
    </svg>
<figcaption>One column times one row fills a whole matrix that still holds just a single direction.</figcaption>
</figure>

## Interview Questions

**Q1: What is an eigenvector, and why does it make repeated application easy?**
An eigenvector $v$ satisfies $Av=\lambda v$, so the matrix only scales it without turning, and any multiple of
$v$ is also an eigenvector with the same $\lambda$. Repeated application gives $A^k v=\lambda^k v$, turning a
hard matrix power into simple scalar powers, which is why eigenvalues govern Markov chains, PageRank, and
dynamical stability where the largest magnitude eigenvalue dominates.

**Q2: How do you compute eigenvalues from scratch?**
Rewrite $Av=\lambda v$ as $(A-\lambda I)v=0$ and require a nonzero solution, which forces $A-\lambda I$ to be
singular. That means its determinant is zero, giving the characteristic equation $\det(A-\lambda I)=0$, a
polynomial whose roots are the eigenvalues. For each eigenvalue you then solve $(A-\lambda I)v=0$ for the
eigenvectors.

**Q3: List several quantities that all equal the rank of a matrix.**
The rank equals the number of independent columns, the number of independent rows, the number of pivots in
echelon form, the number of nonzero singular values, and the dimension of the column space. They are five
views of the same underlying count of genuinely independent directions.

**Q4: What does it mean for a matrix to be low rank, and why is that useful?**
A low-rank matrix has fewer independent directions than its size suggests, so it can be written as a sum of a
few rank-1 outer products. This means it is buildable from a small number of underlying factors, which is the
basis of compression, PCA, and low-rank adaptation, since you store and compute with far fewer numbers.
