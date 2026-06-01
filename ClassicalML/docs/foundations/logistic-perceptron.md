# Logistic Regression & the Perceptron

Classification needs an output bounded in zero to one that stops caring once a point is confidently on the right side. This page covers logistic regression and the sigmoid, why its loss is cross-entropy, Newton's method as the fast curvature-aware optimizer, and the perceptron, which copies the update form but sits outside the probabilistic framework.

!!! tip "Rapid Recall"
    Logistic regression squashes a linear score through the sigmoid into a probability, with a linear decision boundary at \(\theta^\top x=0\). Its loss is cross-entropy, which is exactly the negative Bernoulli log-likelihood, just as squared error was the Gaussian one. The gradient is character-for-character the LMS rule, error times feature, because logistic regression is a GLM. Newton's method converges in under ten iterations using the Hessian but costs \(O(n^3)\) in parameters, so deep nets use SGD instead. The perceptron mimics the update but its hard step is not a probability, so it has no MLE interpretation and lives outside the GLM framework.

## §1 Logistic regression

Linear regression fails at classification: outputs are unbounded, and outliers drag the line and swing the threshold. We need a hypothesis bounded in \([0,1]\) that *saturates* (stops caring once a point is confidently on the right side).

$$h_\theta(x) = g(\theta^\top x) = \frac{1}{1+e^{-\theta^\top x}}, \qquad g(z) = \frac{1}{1+e^{-z}}$$

The sigmoid squashes the real line into \((0,1)\); \(h_\theta(x)\) is now \(P(y=1\mid x)\). Decision boundary at \(\theta^\top x=0\) (still linear in \(x\)).

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="The sigmoid function" role="img" viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg">
<g fill="#4a4438" font-family="monospace" font-size="12">
<line stroke="#d8cfb8" x1="40" x2="520" y1="200" y2="200"></line>
<line stroke="#d8cfb8" x1="280" x2="280" y1="30" y2="215"></line>
<line stroke="#d8cfb8" stroke-dasharray="4,4" x1="40" x2="520" y1="115" y2="115"></line>
<text x="46" y="60">1.0</text><text x="46" y="128">0.5</text><text x="46" y="214">0.0</text>
<text x="500" y="218">z = θᵀx</text>
<path d="M40,198 C160,196 220,180 280,115 C340,50 400,34 520,32" fill="none" stroke="#7c2d12" stroke-width="2.6"></path>
<circle cx="280" cy="115" fill="#1d4d4f" r="4.5"></circle>
<text fill="#1d4d4f" x="290" y="110">boundary (P=0.5)</text>
</g>
</svg>
<figcaption>The sigmoid bends the linear score into a probability. Its slope is highest at z=0 and vanishes at the extremes, that flat tail is the saturation that gives outlier-robustness.</figcaption>
</figure>

### Key derivative

$$g'(z) = g(z)\big(1 - g(z)\big)$$

This makes the gradient collapse cleanly, and its vanishing at large \(|z|\) *is* the saturation: confidently-classified points have ≈0 gradient and stop pulling on the parameters.

### The Bernoulli model and log-likelihood

$$P(y\mid x;\theta) = h_\theta(x)^y\,(1-h_\theta(x))^{1-y}$$

$$\ell(\theta) = \sum_i \Big[ y^{(i)}\log h(x^{(i)}) + (1-y^{(i)})\log(1-h(x^{(i)})) \Big]$$

!!! note "Cross-entropy = negative Bernoulli log-likelihood"
    "Maximize log-likelihood under Bernoulli" and "minimize binary cross-entropy" are literally the same objective. Gaussian gave squared error; Bernoulli gives cross-entropy. Same machinery, swapped distribution.

### The gradient: same form as LMS

$$\frac{\partial\ell}{\partial\theta_j} = \big(y - h_\theta(x)\big)x_j \quad\Rightarrow\quad \theta_j := \theta_j + \alpha\big(y^{(i)} - h_\theta(x^{(i)})\big)x_j^{(i)}$$

Character-for-character the Widrow-Hoff rule — the *only* difference is that here \(h_\theta(x)=g(\theta^\top x)\) (squashed) rather than \(\theta^\top x\) (raw). Not a coincidence: both are [GLMs](glms-generative.md).

## §2 Newton's method

Gradient ascent takes many small steps; Newton takes huge curvature-aware steps and converges in <10 iterations for logistic regression. Root-finding intuition: at a guess, follow the **tangent line** to where it crosses zero — the tangent is the best local linear approximation, so each step leaps most of the remaining distance.

$$\theta := \theta - \frac{f(\theta)}{f'(\theta)}$$

To *maximize* \(\ell\), find where \(\ell'=0\), i.e. run root-finding on \(f=\ell'\):

$$\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)} \qquad\xrightarrow{\text{vector form}}\qquad \theta := \theta - H^{-1}\nabla_\theta\ell(\theta)$$

The first derivative says which way and how steeply; the second (curvature, the Hessian \(H\)) sizes the step.

!!! note "Two facts"
    *Quadratic convergence* — correct digits roughly double each step (error 0.01 → 0.0001 → 1e-8). But the *Hessian inverse is \(O(n^3)\)* in parameter count — prohibitive for deep nets (millions of params), which is why they use SGD/Adam. Newton applied to logistic regression is called *Fisher Scoring*.

## §3 The perceptron

Replace the sigmoid with a hard step:

$$g(z) = \begin{cases}1 & z\ge0\\ 0 & z<0\end{cases} \qquad \theta_j := \theta_j + \alpha\big(y^{(i)} - h_\theta(x^{(i)})\big)x_j^{(i)}$$

Same update form (only moves on mistakes), but a fundamentally different algorithm.

!!! warning "Interview trap"
    The perceptron looks "cosmetically similar" to logistic/linear regression but the step function is *not a probability distribution* — so the perceptron has *no probabilistic interpretation and cannot be derived as an MLE*. There's no likelihood it maximizes. It sits outside the GLM framework — a conceptual stepping-stone, not a probabilistic model.

## Interview questions

**Q1: Why use the sigmoid for binary classification rather than a straight line?**
A line produces unbounded outputs and is dragged by outliers, swinging the decision threshold. The sigmoid squashes the linear score into the open interval zero to one so the output reads as \(P(y=1\mid x)\), and it saturates: its slope vanishes at the extremes, so confidently classified points have near-zero gradient and stop pulling on the parameters. The decision boundary stays linear at \(\theta^\top x=0\).

**Q2: What loss does logistic regression minimize and where does it come from?**
Binary cross-entropy, which is exactly the negative log-likelihood of a Bernoulli model. Maximizing the Bernoulli log-likelihood and minimizing cross-entropy are the same objective, mirroring how Gaussian noise gives squared error. This is why the gradient comes out as error times feature, identical in form to the LMS rule.

**Q3: When is Newton's method worth it over gradient descent?**
When the parameter count is small and the problem is convex, as in logistic regression, where it converges in under ten iterations with quadratic convergence, doubling correct digits each step. It is not worth it for deep nets because inverting the Hessian costs \(O(n^3)\) in parameters, so those use SGD or Adam. Newton applied to logistic regression is called Fisher Scoring.

**Q4: Why is the perceptron not a probabilistic model even though its update looks identical?**
Because its activation is a hard step, which is not a probability distribution, so there is no likelihood for it to maximize and it cannot be derived as an MLE. Logistic regression's sigmoid is a Bernoulli probability, placing it inside the GLM framework, whereas the perceptron only mimics the error-times-feature update and sits outside it as a conceptual stepping stone.
