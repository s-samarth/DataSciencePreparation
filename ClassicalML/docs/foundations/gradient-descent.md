# Gradient Descent & the LMS Rule

When the closed form is too expensive, you walk to the bottom of the loss bowl instead of jumping there. This page covers gradient descent and its three flavors, the self-correcting LMS (Widrow-Hoff) update that every flavor is built from, and the convexity argument (the Hessian and positive semi-definiteness) that guarantees the walk reaches the global minimum.

!!! tip "Rapid Recall"
    Gradient descent steps against the gradient, which points uphill, with the learning rate setting stride length. The only difference between batch, stochastic, and mini-batch is how many samples estimate the gradient before each step: all of them, one, or 32 to 256. The per-example update is the LMS rule, error times feature, which is self-correcting and only adjusts the features that drove the prediction. Linear regression's loss is exactly quadratic, so its Hessian \(2X^\top X\) is constant and positive semi-definite, which makes the loss convex and any stationary point the global minimum.

## §1 Gradient descent and its three flavors

Instead of jumping to the bottom of the bowl, **walk** there. The gradient \(\nabla_{\mathbf w}L\) points uphill (steepest increase), so step against it:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla_{\mathbf{w}} L(\mathbf{w})$$

\(\eta\) is the learning rate (stride length). Too small → crawls forever. Too large → oversteps the valley and bounces / diverges.

> **Foggy hillside.** You are on a hill in fog (no closed form visible), but you can feel the slope at your feet. Step in the steepest downhill direction, reassess, repeat. \(\eta\) is your stride; too big and you leap clear over the valley onto the far slope.

### The gradient for linear regression (vectorized)

With error vector \(\mathbf e = X\mathbf w - \mathbf y\):

$$\nabla_{\mathbf{w}} L = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y})$$

In words: **the gradient for feature \(j\) is the average of (prediction error × that feature's value).** Setting it to zero recovers the Normal Equation — so GD and the closed form chase the same point; GD just crawls there. Because the loss is convex, GD with a sane \(\eta\) is guaranteed to reach the global minimum.

### The three flavors: same algorithm, one dial

The only thing that changes is **how many samples are used to estimate the gradient before each step**:

| Flavor | Samples / step | Character |
| --- | --- | --- |
| **Batch** (\(B=n\)) | all \(n\) | True gradient, smooth path, but one step touches the whole dataset. 1 update / epoch. |
| **Stochastic (SGD)** (\(B=1\)) | one random | Cheap, noisy, zig-zags; buzzes around the minimum (decay \(\eta\) to settle). \(n\) updates / epoch. |
| **Mini-batch** (\(1\!<\!B\!<\!n\)) | 32–256 | Best of both; GPU-friendly matrix multiply. The practical default. \(n/B\) updates / epoch. |

One epoch = one full pass over the data. Note: PyTorch's `optim.SGD` is really mini-batch — define all three precisely by \(B\) in an interview.

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="Contour paths of batch, mini-batch and SGD descent" role="img" viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg">
<!-- contours -->
<g fill="none" stroke="#d8cfb8" stroke-width="1.4">
<ellipse cx="470" cy="130" rx="150" ry="92"></ellipse>
<ellipse cx="470" cy="130" rx="112" ry="68"></ellipse>
<ellipse cx="470" cy="130" rx="74" ry="44"></ellipse>
<ellipse cx="470" cy="130" rx="38" ry="22"></ellipse>
</g>
<circle cx="470" cy="130" fill="#7c2d12" r="4.5"></circle>
<text fill="#7c2d12" font-family="monospace" font-size="11" text-anchor="middle" x="470" y="118">min</text>
<!-- batch: smooth arc -->
<path d="M70,40 Q250,70 470,130" fill="none" stroke="#1d4d4f" stroke-width="2.6"></path>
<circle cx="70" cy="40" fill="#1d4d4f" r="4"></circle>
<text fill="#1d4d4f" font-family="monospace" font-size="12" x="60" y="32">Batch</text>
<!-- mini-batch: slightly wobbly -->
<path d="M70,110 Q180,150 250,120 T390,150 Q440,140 470,130" fill="none" stroke="#9a7b2e" stroke-width="2.2"></path>
<circle cx="70" cy="110" fill="#9a7b2e" r="4"></circle>
<text fill="#9a7b2e" font-family="monospace" font-size="12" x="60" y="103">Mini-batch</text>
<!-- sgd: drunk stumble -->
<path d="M70,210 L120,170 L100,200 L165,160 L150,195 L220,150 L205,185 L280,150 L270,170 L340,140 L330,165 L400,140 L395,150 L450,135 L460,140 L470,130" fill="none" stroke="#7c2d12" stroke-width="1.7"></path>
<circle cx="70" cy="210" fill="#7c2d12" r="4"></circle>
<text fill="#7c2d12" font-family="monospace" font-size="12" x="60" y="232">SGD</text>
</svg>
<figcaption>Contour map of the loss bowl. Batch takes a clean arc (few costly steps); SGD is a cheap drunk stumble that orbits the minimum; mini-batch is the purposeful middle path.</figcaption>
</figure>

## §2 The LMS / Widrow-Hoff rule

Per-example cost \(J(\theta)=\tfrac12(h_\theta(x)-y)^2\) with \(h_\theta(x)=\theta^\top x\). Chain rule (the 2 and \(\tfrac12\) cancel; \(\partial h/\partial\theta_j = x_j\)):

$$\frac{\partial J}{\partial \theta_j} = \big(h_\theta(x) - y\big)\,x_j$$

**The gradient for weight \(j\) is the prediction error times the \(j\)-th feature.** Plugging into the GD step gives the *LMS / Widrow-Hoff update*:

$$\theta_j := \theta_j + \alpha\big(y^{(i)} - h_\theta(x^{(i)})\big)x_j^{(i)}$$

!!! note "Why it is intuitive"
    The update is *self-correcting and proportional to error*: near-right prediction → tiny step; badly wrong → big step; the sign auto-handles direction; and it is scaled by \(x_j\) so only the feature that actually drove the prediction gets corrected. This single-example atom is what Batch GD (sum over all \(m\)) and SGD (one at a time) are built from.

> **Cricket.** A batsman adjusting after each ball: misjudge the length badly, big technical correction; almost middled it, leave technique alone; and only adjust the part of your game that was involved (no point fixing pull-shot footwork after misjudging a cover drive).

## §3 Convexity, the Hessian and positive semi-definite

From \(\nabla_{\mathbf w}L = -2X^\top\mathbf y + 2X^\top X\mathbf w\), differentiate again:

$$H = \nabla^2_{\mathbf{w}} L = 2 X^\top X$$

The Hessian is **constant** — the loss is exactly quadratic, so curvature is identical everywhere (a true paraboloid). Convexity ⇔ \(X^\top X\) is positive semi-definite (PSD).

> **PSD proof.** For any vector \(\mathbf z\): \(\;\mathbf{z}^\top (X^\top X) \mathbf{z} = (X\mathbf{z})^\top (X\mathbf{z}) = \|X\mathbf{z}\|^2 \geq 0.\) A squared length is never negative — which is the definition of PSD. Hence \(H = 2X^\top X \succeq 0\), the loss is convex, and any stationary point is a global minimum.

### What PSD actually means

- **Matrix version of "non-negative":** just as \(a\ge 0\) means \(az^2\ge0\) (upward parabola), PSD means \(\mathbf z^\top A\mathbf z\ge0\) — the quadratic surface opens upward in *every* direction. That's why "Hessian PSD ⇔ convex."
- **Eigenvalue view:** PSD ⇔ all eigenvalues \(\ge 0\) (the curvatures along the principal axes). A negative eigenvalue = a downhill direction = a saddle.
- **PD vs PSD:** strictly positive (PD, unique minimum) iff \(X\) has full column rank. Collinear features make some \(X\mathbf z = 0\) → a flat trough → \(X^\top X\) singular → the closed form breaks. *This is exactly what [Ridge](regularization.md)'s \(+\lambda I\) fixes.*

## Interview questions

**Q1: What actually differs between batch, stochastic, and mini-batch gradient descent?**
Only the number of samples used to estimate the gradient before each step. Batch uses all n for the true gradient and a smooth path but one update per epoch; SGD uses one random sample, which is cheap and noisy and buzzes around the minimum; mini-batch uses 32 to 256, the GPU-friendly default. PyTorch's `optim.SGD` is actually mini-batch, so define each precisely by the batch size B.

**Q2: How does the learning rate affect convergence?**
The learning rate is the stride length applied against the gradient. Too small and the descent crawls and takes forever; too large and it oversteps the valley, bounces, or diverges. For SGD you typically decay the learning rate over time so the iterate settles into the minimum rather than buzzing around it.

**Q3: Why is linear regression guaranteed to find the global minimum?**
Its loss is exactly quadratic, so the Hessian \(2X^\top X\) is constant and positive semi-definite, since \(\mathbf z^\top X^\top X\mathbf z=\|X\mathbf z\|^2\ge0\). A positive semi-definite Hessian means the surface opens upward in every direction, so the loss is convex and any point where the gradient is zero is the global minimum. Gradient descent with a sane learning rate therefore converges to it.

**Q4: What does the LMS update tell you intuitively?**
The per-weight gradient is the prediction error times that feature's value, so the update is proportional to how wrong you were and scaled by the feature that drove the prediction. A near-correct prediction barely moves the weights, a badly wrong one moves them a lot, and the sign handles direction automatically. Batch GD sums this atom over all examples and SGD applies it one at a time.
