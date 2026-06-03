# Special Matrices & Definiteness

A handful of matrix types recur so often that their properties are worth memorizing: identity, singular,
symmetric, orthogonal, projection, reflection, and Markov. Positive (semi)definite matrices then add a sign
condition that makes them the algebraic language of a bowl, the foundation of convex optimization.

!!! tip "Rapid Recall"
    The identity does nothing; a singular matrix has zero determinant and loses information. Symmetric
    matrices ($A=A^\top$) have real eigenvalues, orthogonal eigenvectors, and the spectral factorization
    $A=Q\Lambda Q^\top$. Orthogonal matrices preserve lengths and angles ($Q^\top Q=I$), projections are
    idempotent ($P^2=P$) with the regression hat matrix as the key example, reflections square to the
    identity, and Markov matrices have top eigenvalue 1 whose eigenvector is the stationary distribution.
    Positive definite means $v^\top Av>0$ for all nonzero $v$, equivalently all eigenvalues positive, which is
    a bowl curving up in every direction.

## §6 Special matrices

### Identity matrix (I)

The do-nothing machine: $Iv = v$. Multiplicative identity, full rank, every eigenvalue 1, every vector an
eigenvector. Shows up as residual connections ($x + f(x)$) and the $\lambda I$ in ridge regression.

### Singular matrix

Square but **not invertible**: $\det = 0$, rank $< n$, has 0 as an eigenvalue, nonempty null space. Information
is lost. The failure mode behind collinear features wrecking regression.

### Symmetric matrix (A = Aᵀ), the important one

The matrix of **mutual relationships and second-order structure**: covariances, Gram matrices $X^\top X$,
kernels, Hessians, undirected-graph adjacencies. Mutuality forces symmetry; symmetry pays back the cleanest
eigenstructure:

- All eigenvalues are **real**.
- Eigenvectors are **orthogonal** (a full orthonormal set exists).
- **Spectral theorem:** $A = Q\Lambda Q^\top$ with Q orthogonal, $\Lambda$ diagonal.

$$A = Q \Lambda Q^\top \;=\; \text{rotate into the eigenbasis} \cdot \text{scale} \cdot \text{rotate back (rigidly, no distortion)}$$

### Orthogonal matrix (QᵀQ = I)

Inverse equals transpose; columns orthonormal; preserves all lengths and angles (a rigid motion). Every
eigenvalue has magnitude 1. Parent of rotations and reflections.

### Projection matrix (P² = P)

Squashes vectors onto a subspace. Projecting twice equals once (idempotent). Eigenvalues only 0 (killed) and 1
(kept). Mental image: a shadow on the ground.

$$P = A (A^\top A)^{-1} A^\top \quad\text{(the "hat matrix" that produces }\hat y)$$

This orthogonal projection onto the column space of A *is* linear regression.

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 520 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Projection of a vector onto a line">
      <line x1="40" y1="170" x2="470" y2="60" stroke="#1d5b6e" stroke-width="2"/>
      <text x="450" y="55" font-family="serif" font-size="13" fill="#1d5b6e">subspace</text>
      <line x1="60" y1="170" x2="300" y2="40" stroke="#b1542f" stroke-width="3"/>
      <text x="305" y="38" font-family="serif" font-size="14" fill="#b1542f">b</text>
      <line x1="60" y1="170" x2="270" y2="116" stroke="#a8852c" stroke-width="3"/>
      <text x="275" y="132" font-family="serif" font-size="14" fill="#a8852c">Pb = ŷ</text>
      <line x1="300" y1="40" x2="270" y2="116" stroke="#9aa0ad" stroke-width="1.6" stroke-dasharray="5 4"/>
      <text x="312" y="92" font-family="serif" font-size="12" fill="#777">residual</text>
      <circle cx="60" cy="170" r="3" fill="#1f2330"/>
    </svg>
<figcaption>Regression projects the target b onto what the features can reach; the leftover is the residual.</figcaption>
</figure>

### Reflection matrix (R² = I)

A mirror. Reflect twice goes back to start, so $R = R^{-1}$. Eigenvalues $+1$ (on the mirror) and $-1$
(perpendicular, flipped). Determinant $-1$ (orientation reversed). Householder reflections power QR
decomposition.

### Markov (stochastic) matrix

Entries $\ge 0$, each column sums to 1 (probability conserved). Largest eigenvalue is **exactly 1**, and its
eigenvector is the **stationary distribution**; all other $|\lambda| \le 1$, so iteration converges. This is
[Markov chains](../advanced/markov-chains.md), HMMs, and PageRank.

## §7 Positive definite & positive semidefinite

These are symmetric matrices with a sign condition. They are the algebraic language of "a bowl," which is why
they run convex optimization.

$$\text{PD: } v^\top A v > 0 \text{ for all } v \neq 0 \qquad \text{PSD: } v^\top A v \ge 0 \text{ for all } v$$

The quantity $v^\top Av$ is a bowl-shaped function of $v$. PD means the bowl curves strictly up in every
direction (single lowest point). PSD means up or flat. Indefinite means a saddle. Rotate into the eigenbasis
and the reason is obvious:

$$v^\top A v = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2$$

- all $\lambda > 0$ means always positive, so **PD**
- some $\lambda = 0$ means flat along that axis, so **PSD**
- some $\lambda < 0$ means curves down, so **indefinite / saddle**

<figure class="diagram diagram-light" markdown="0">
<svg viewBox="0 0 560 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PD bowl versus saddle">
      <g transform="translate(20,10)">
        <path d="M20,140 Q110,-10 200,140" fill="none" stroke="#2f7d4f" stroke-width="3"/>
        <path d="M40,150 Q110,30 180,150" fill="none" stroke="#9fc9ad" stroke-width="2"/>
        <text x="110" y="178" font-family="serif" font-size="14" fill="#2f7d4f" text-anchor="middle">PD — true bowl (min)</text>
      </g>
      <g transform="translate(320,10)">
        <path d="M20,40 Q110,150 200,40" fill="none" stroke="#a8392b" stroke-width="2" opacity="0.6"/>
        <path d="M20,140 Q110,30 200,140" fill="none" stroke="#a8392b" stroke-width="3"/>
        <text x="110" y="178" font-family="serif" font-size="14" fill="#a8392b" text-anchor="middle">indefinite — saddle</text>
      </g>
    </svg>
<figcaption>Up in every direction (PD) versus up one way and down another (indefinite).</figcaption>
</figure>

**How to test PD / PSD.**

- all eigenvalues $> 0$ (PD) or $\ge 0$ (PSD),
- all pivots positive,
- leading minors $> 0$ (Sylvester's criterion),
- writable as $B^\top B$.

## Interview Questions

**Q1: What makes symmetric matrices so well-behaved?**
A symmetric matrix equals its transpose, which forces all eigenvalues to be real and guarantees a full
orthonormal set of eigenvectors. This gives the spectral theorem factorization $A=Q\Lambda Q^\top$ with $Q$
orthogonal, a rigid rotate-scale-rotate with no distortion. Covariances, Gram matrices, kernels, and Hessians
are all symmetric, which is why this structure appears everywhere.

**Q2: What is a projection matrix, and how does it relate to regression?**
A projection matrix is idempotent, $P^2=P$, squashing vectors onto a subspace with eigenvalues only 0 and 1.
The orthogonal projection onto the column space of $A$ is $P=A(A^\top A)^{-1}A^\top$, the hat matrix that
produces the fitted values $\hat y$. Linear regression literally projects the target onto what the features can
reach, leaving the residual.

**Q3: What characterizes a positive definite matrix, and why does it matter for optimization?**
A symmetric matrix is positive definite when $v^\top A v>0$ for all nonzero $v$, equivalently when all its
eigenvalues are positive. In the eigenbasis the quadratic form becomes a sum of $\lambda_i y_i^2$, so positive
eigenvalues make it a bowl curving up in every direction with a single minimum. That is exactly the condition
that makes a quadratic objective convex with a unique optimum.

**Q4: What is special about the eigenvalues of a Markov matrix?**
Its largest eigenvalue is exactly 1, and the corresponding eigenvector is the stationary distribution, while
all other eigenvalues have magnitude at most 1 so iteration converges. This is why repeatedly applying a
transition matrix drives any starting distribution toward the stationary one, the principle behind PageRank and
hidden Markov models.
