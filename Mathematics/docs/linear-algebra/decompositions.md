# Decompositions (LU, Eigen, SVD)

A matrix tangles rotation, stretch, shear, and collapse together. A decomposition factors that one machine
into a product of simpler machines, each doing one clean thing. This page covers why decomposition matters,
LU for solving systems, diagonalization for powers, and the singular value decomposition that unifies them
all.

!!! tip "Rapid Recall"
    Decomposition factors a matrix into simpler machines for easy solving, easy powers, transparency,
    compression, and stability. LU is Gaussian elimination recorded as a product, turning a solve into two
    triangular solves. Diagonalization $A=PDP^{-1}$ reads right to left as change into the eigenbasis, scale,
    change back, which makes $A^k=PD^kP^{-1}$. The singular value decomposition $A=U\Sigma V^\top$ exists for
    every matrix: rotate, stretch, rotate, with singular values $\sigma_i=\sqrt{\lambda_i}$ of $A^\top A$.
    Truncating it gives the best rank-r approximation (Eckart-Young), the engine of PCA, compression, and
    denoising.

## §8 Why decompose a matrix at all

A matrix tangles rotation, stretch, shear and collapse together. A decomposition factors that one machine into
a product of simpler machines, each doing one clean thing. The analogy is number factoring: $60 = 2^2\cdot3\cdot5$
instantly reveals structure. Factoring a matrix gives **easy solving** (triangular/diagonal), **easy powers**,
**transparency** of what it does, **compression**, and **numerical stability**.

## §9 LU decomposition

$A = LU$: L lower-triangular (1s on the diagonal), U upper-triangular. It is just **Gaussian elimination
recorded as a product**, U is the staircase, L stores the multipliers used to clear each column. The diagonal
of U is the pivots; their product is the determinant.

Payoff: solving $Ax = b$ becomes two trivial triangular solves, set $Ux = y$, solve $Ly = b$ top-down (forward
substitution), then $Ux = y$ bottom-up (back substitution). Each is $O(n^2)$ and reusable for any new b. With
row swaps it becomes $PA = LU$.

## §10 Diagonalization, eigenvalue decomposition

$$A = P D P^{-1} \quad\text{(P = eigenvectors as columns, D = eigenvalues on the diagonal)}$$

**Why this form is forced (derivation).**

$$AP = A[v_1|\dots|v_n] = [Av_1|\dots|Av_n] = [\lambda_1 v_1|\dots|\lambda_n v_n] = P D \;\Longrightarrow\; A = P D P^{-1}$$

**Read it right-to-left as a three-step machine.**

- $P^{-1}$, change coordinates *into* the eigenbasis (express input as amounts of each eigenvector).
- $D$, scale each eigen-coordinate by its eigenvalue (pure, independent stretching).
- $P$, change back to the standard basis.

The complexity of A was an illusion of the coordinate system; in the eigenbasis it is just scaling. The payoff,
powers collapse:

$$A^k = P D^k P^{-1} \quad\text{(all the inner } P^{-1}P \text{ cancel; } D^k \text{ just powers each diagonal entry)}$$

## §11 Singular value decomposition

SVD does for *every* matrix what the spectral theorem does for symmetric ones, rectangular, singular,
non-diagonalizable, anything.

$$A = U \Sigma V^\top \quad\text{(U, V orthogonal; } \Sigma \text{ holds the singular values } \sigma_1 \ge \sigma_2 \ge \dots \ge 0)$$

**The geometry, circle becomes ellipse.** Every linear map, however ugly, does exactly three things (right to
left): **$V^\top$** a rigid rotation of the input, **$\Sigma$** an axis-aligned stretch, **$U$** a rigid
rotation of the output. Rotate, stretch, rotate.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SVD maps a unit circle to an ellipse">
      <g transform="translate(90,110)">
        <circle cx="0" cy="0" r="60" fill="none" stroke="#1d5b6e" stroke-width="2.5"/>
        <line x1="0" y1="0" x2="42" y2="-42" stroke="#b1542f" stroke-width="2.5"/>
        <line x1="0" y1="0" x2="-42" y2="-42" stroke="#a8852c" stroke-width="2.5"/>
        <text x="-60" y="92" font-family="monospace" font-size="12" fill="#444a5c">unit circle</text>
        <text x="20" y="-50" font-family="serif" font-size="12" fill="#b1542f">v₁</text>
        <text x="-58" y="-50" font-family="serif" font-size="12" fill="#a8852c">v₂</text>
      </g>
      <text x="300" y="115" font-family="serif" font-size="16" fill="#1f2330" text-anchor="middle">A = UΣVᵀ ⟶</text>
      <g transform="translate(500,110)">
        <ellipse cx="0" cy="0" rx="95" ry="40" transform="rotate(-22)" fill="none" stroke="#1d5b6e" stroke-width="2.5"/>
        <line x1="0" y1="0" x2="88" y2="-36" stroke="#b1542f" stroke-width="3"/>
        <line x1="0" y1="0" x2="-15" y2="-37" stroke="#a8852c" stroke-width="3"/>
        <text x="60" y="-44" font-family="serif" font-size="12" fill="#b1542f">σ₁u₁</text>
        <text x="-44" y="-40" font-family="serif" font-size="12" fill="#a8852c">σ₂u₂</text>
        <text x="-50" y="78" font-family="monospace" font-size="12" fill="#444a5c">ellipse</text>
      </g>
    </svg>
<figcaption>V picks the circle directions; the sigma are the ellipse semi-axis lengths; U gives the ellipse axes in output space.</figcaption>
</figure>

**How it is built on eigendecomposition.**

$$A^\top A = (U\Sigma V^\top)^\top(U\Sigma V^\top) = V \Sigma^\top(U^\top U)\Sigma V^\top = V \Sigma^2 V^\top$$

That is exactly the eigendecomposition of the symmetric PSD matrix $A^\top A$: the right singular vectors
**V are its eigenvectors**, and **$\sigma_i = \sqrt{\lambda_i}$** (real and $\ge 0$ because $A^\top A$ is PSD).
Similarly $AA^\top = U\Sigma^2 U^\top$ gives U.

- Form $A^\top A$; eigendecompose to get V and $\lambda_i$.
- $\sigma_i = \sqrt{\lambda_i}$, sorted descending, gives $\Sigma$.
- $u_i = A v_i / \sigma_i$ gives U.

$$A = \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \dots + \sigma_r u_r v_r^\top$$

Truncating after r terms gives the provably **best rank-r approximation** (Eckart-Young). Big $\sigma$ is
signal, small $\sigma$ is detail/noise. This is the engine of PCA, image compression, LSA, recommendation, and
denoising. (LoRA assumes the update is one of these short sums from the start.)

### SVD versus eigendecomposition

|  | Eigendecomposition $A = PDP^{-1}$ | SVD $A = U\Sigma V^\top$ |
|---|---|---|
| Exists for | square and diagonalizable only | **every** matrix, any shape |
| Bases | P generally skewed | U, V both orthonormal (rigid) |
| Frames | one shared basis (P, $P^{-1}$) | two bases, V in, U out |
| Values | eigenvalues, can be $\pm$/complex | singular values, always real $\ge 0$ |
| Stability | can be unstable | numerically very stable |
| Geometry | rotate, scale, un-rotate | rotate, stretch, rotate (circle to ellipse) |

They coincide for a symmetric PSD matrix ($U = V = Q$, $\Sigma = \Lambda$). SVD is the spectral theorem
extended to all matrices.

## Interview Questions

**Q1: Why decompose a matrix at all?**
Because a single matrix tangles rotation, stretch, shear, and collapse, while a decomposition factors it into
simpler machines that each do one clean thing. The benefits are easy solving (triangular or diagonal factors),
easy powers, transparency about what the matrix does, compression, and numerical stability, the same way
factoring an integer reveals its structure.

**Q2: How does LU decomposition speed up solving a linear system?**
LU records Gaussian elimination as a product of a lower- and an upper-triangular matrix, so $Ax=b$ becomes two
triangular solves, forward substitution for $Ly=b$ then back substitution for $Ux=y$. Each is $O(n^2)$ and the
factorization is reused for any new right-hand side, which is far cheaper than re-eliminating each time.

**Q3: What does the singular value decomposition do geometrically, and why does it always exist?**
It writes any matrix as $U\Sigma V^\top$, a rigid input rotation, an axis-aligned stretch, and a rigid output
rotation, mapping a unit circle to an ellipse. It always exists because it is built from the eigendecomposition
of the symmetric positive semidefinite matrix $A^\top A$, whose eigenvectors give $V$ and whose square-rooted
eigenvalues give the nonnegative singular values, which works for any shape of matrix.

**Q4: Why is the SVD the tool behind PCA and compression?**
Because truncating the SVD after the largest $r$ singular values gives the provably best rank-$r$ approximation
by the Eckart-Young theorem, with large singular values capturing signal and small ones capturing noise.
Keeping only the top terms compresses the matrix while preserving its dominant structure, which is exactly what
PCA, image compression, and low-rank adaptation exploit.
