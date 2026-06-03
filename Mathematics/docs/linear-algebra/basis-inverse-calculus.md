# Basis, Inverse & Matrix Calculus

The closing thread of linear algebra: a basis is the language for describing vectors, invertibility is the
single statement that a machine collapses nothing, the pseudo-inverse is the best-effort inverse when it does,
the null space is what a matrix crushes, and the matrix calculus cheat sheet powers every backward pass.

!!! tip "Rapid Recall"
    A basis is the smallest spanning, independent set, so every vector has a unique recipe, and its size is the
    dimension. Invertibility bundles many equivalent facts: full rank, independent columns, zero null space,
    nonzero determinant, and $n$ pivots, all saying nothing collapsed. The pseudo-inverse generalizes the
    inverse via SVD, $A^+=V\Sigma^+U^\top$, giving least-squares for tall matrices and minimum-norm for wide
    ones. The null space is the set of inputs crushed to zero, and rank plus nullity equals the number of
    columns. The matrix calculus workhorse is $\partial\|Ax-b\|^2/\partial x=2A^\top(Ax-b)$, which yields the
    normal equations.

## §12 Basis

A basis is the smallest set of building-block directions from which every vector is built in **exactly one
way**. Two conditions produce that uniqueness:

- **Spanning**, they reach everything.
- **Independence**, no redundancy.

Too few means unreachable vectors; too many (dependent) means non-unique recipes. The count of basis vectors is
the **dimension**, and it is fixed for a space. Any $n$ independent vectors in $\mathbb{R}^n$ form a basis. A
basis is a **language** for describing vectors, and coordinates are basis-dependent, changing basis is
translation (exactly what $P^{-1}$ and $V^\top$ did). The pivot columns of a matrix form a basis for its column
space, so basis, span, independence, rank, and dimension are five views of one structure. PCA is literally the
act of choosing a better basis, the one aligned with variance.

## §13 Inverse & invertibility

The inverse $A^{-1}$ is the machine that **undoes** A: $A^{-1}A = I$. You can run a machine backwards only if
it is a perfect one-to-one correspondence, every output came from exactly one input. For a square $n\times n$
matrix these are all the *same* fact ("nothing collapsed"):

- full rank $n$,
- columns independent (a basis),
- columns span $\mathbb{R}^n$,
- null space $=\{0\}$,
- unique solution for every b,
- $\det \neq 0$,
- no zero eigenvalues,
- no zero singular values,
- $n$ pivots.

**Why the determinant test works:** det is the signed volume-scaling factor. $\det \neq 0$ means volume
survived, no dimension flattened, reversible. $\det = 0$ means volume collapsed, a dimension was crushed, not
reversible. A rectangular matrix has no true two-sided inverse (it changes the dimension of the space), which
is why we need the pseudo-inverse.

## §14 Pseudo-inverse

The best-effort inverse for matrices that are not invertible (rectangular or singular). Two cases:

- **Overdetermined** (tall, more equations than unknowns): usually no exact solution, the pseudo-inverse returns the **least-squares** solution, the x minimizing $\|Ax - b\|^2$. This is linear regression; $AA^+$ projects b onto the column space.
- **Underdetermined** (wide, infinitely many solutions): it picks the **minimum-norm** solution.

$$A^+ = V \Sigma^+ U^\top \quad\text{invert each nonzero } \sigma \;(\sigma \to 1/\sigma)\text{, leave zeros as zeros}$$

Intuition straight from the rotate-stretch-rotate picture: to undo A, reverse-rotate ($U^\top$), un-stretch the
directions that *were* stretched ($\Sigma^+$), reverse-rotate ($V$). A direction with $\sigma = 0$ was crushed
flat and cannot be un-crushed (no $1/0$), so you zero it, that dropped term is the source of "best effort." If
A is invertible, $A^+ = A^{-1}$, the pseudo-inverse strictly generalizes the inverse. The regression normal
equations $X^\top X \beta = X^\top y$ are just $(X^\top X)^{-1}X^\top$ written out.

## §15 Null space

The null space of A is the set of all vectors v with $Av = 0$, the inputs the matrix crushes to zero.

**How to get it:** solve $Av = 0$ by elimination; non-pivot columns become free variables; solve the pivot
variables in terms of them; each free variable yields one basis vector of the null space.

$$\text{rank} + \text{nullity} = \text{number of columns}$$

**Why it matters:** a nonzero null space means lost information and no inverse (two inputs collide); it is the
set of directions your features cannot distinguish (collinearity), the directions a pseudo-inverse discards,
and the source of non-unique solutions that regularization resolves.

## §16 Matrix calculus cheat sheet

Convention: the gradient has the same shape as the variable (ML / denominator layout). $x, a$ are vectors, $A$
a matrix; scalar output unless noted.

| Expression | Derivative / result | Used in | Caveat |
|---|---|---|---|
| $a^\top x$ or $x^\top a$ | $a$ | linear layers, dot-product scores | the simplest building block |
| $x^\top x = \|x\|^2$ | $2x$ | L2 penalties, squared distance | this is the *squared* norm |
| $\|x\|_2$ | $x/\|x\|$ | normalization, cosine sim | undefined at x = 0 |
| $\|x\|_1$ | $\text{sign}(x)$ | Lasso / L1 | subgradient; non-diff at 0 |
| $x^\top A x$ | $(A + A^\top)x \to 2Ax$ if symmetric | Mahalanobis, energy, SVM/GP | 2Ax only when symmetric |
| $A x$ (vector out) | Jacobian $= A$ | backprop through linear layers | a Jacobian, not a gradient |
| $\|Ax - b\|^2$ | $2A^\top(Ax - b)$ | least squares to normal eqns | set to 0 to solve; the workhorse |
| $a^\top X b$ | $a b^\top$ | bilinear forms, attention scores | outer-product result |
| $\text{tr}(AX)$ | $A^\top$ | trace objectives | tr makes a scalar |
| $\text{tr}(X^\top A X)$ | $(A + A^\top)X$ | covariance / quadratic losses | 2AX if symmetric |
| $\log\det X$ | $X^{-\top}$ | Gaussian log-lik, covariance MLE, flows | needs X invertible (PD) |
| $\det X$ | $\det(X)\cdot X^{-\top}$ | Jacobian-determinant terms | prefer log det in practice |
| softmax + cross-entropy (out p, target y) | $\partial L/\partial z = p - y$ | every classifier's backward pass | holds for softmax+CE paired |
| sigmoid $\sigma(z)$ | $\sigma(z)(1 - \sigma(z))$ | logistic regression, gates | saturates, vanishing gradient |
| $\tanh(z)$ | $1 - \tanh^2(z)$ | RNN / legacy activations | also saturates |
| ReLU $\max(0,z)$ | 1 if z > 0 else 0 | default hidden activation | subgradient at 0; dead units |
| Hessian of $x^\top A x$ | $A + A^\top \to 2A$ if symmetric | 2nd-order opt, convexity check | PD Hessian means local min |
| chain rule (vectors) | $\partial z/\partial x = (\partial z/\partial y)(\partial y/\partial x)$ | all of backprop | mind Jacobian shapes/order |

The one that earns its keep most: $\partial\|Ax-b\|^2/\partial x = 2A^\top(Ax-b)$, set it to zero and out fall
the normal equations and the whole least-squares machinery.

## §17 The single thread

Matrices are machines whose columns are basis landing spots. Matrix multiplication chains machines or runs one
over a batch. Eigenvectors are a matrix's natural stretch axes; rank counts its real independent directions.
Symmetric matrices factor into a rigid rotate-scale-rotate, and "symmetric with non-negative eigenvalues" (PSD)
is the algebraic signature of a bowl, the language of convex optimization. SVD unifies all of it: every matrix
is a rotation, a stretch, and a rotation, with the stretches ordered so you can keep the big ones and throw the
rest away. Invertibility is the single statement that a machine collapses nothing; the pseudo-inverse is what
you reach for when it does.

## Interview Questions

**Q1: What two conditions define a basis, and why do both matter?**
A basis must span the space, so every vector is reachable, and be linearly independent, so there is no
redundancy. Spanning without independence gives non-unique recipes, while independence without spanning leaves
some vectors unreachable. Together they guarantee that every vector has exactly one coordinate representation,
and the number of basis vectors is the dimension.

**Q2: List several equivalent conditions for a square matrix to be invertible.**
Full rank, columns linearly independent, columns spanning the space, null space containing only the zero
vector, a unique solution for every right-hand side, nonzero determinant, no zero eigenvalues, no zero singular
values, and $n$ pivots. They are all the same statement that the matrix collapses no dimension and is therefore
reversible.

**Q3: What does the pseudo-inverse compute for tall versus wide matrices?**
For a tall, overdetermined matrix it returns the least-squares solution that minimizes $\|Ax-b\|^2$, which is
linear regression. For a wide, underdetermined matrix it returns the minimum-norm solution among the infinitely
many. Computed via the SVD as $A^+=V\Sigma^+U^\top$, it inverts the nonzero singular values and zeros the
crushed directions, reducing to the ordinary inverse when one exists.

**Q4: Why is $\partial\|Ax-b\|^2/\partial x=2A^\top(Ax-b)$ the most important matrix-calculus identity?**
Because setting this gradient to zero gives $A^\top A x=A^\top b$, the normal equations, which are the closed-form
solution to least squares and hence to linear regression. It is the workhorse behind fitting linear models and
appears throughout optimization wherever a squared residual is minimized.
