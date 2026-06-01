# GLMs, Softmax & Generative Learning

This is the keystone page: the framework that reveals linear, logistic, softmax, and Poisson regression as one recipe, and the generative counterpart that approaches classification from the opposite direction yet lands provably back on the logistic form. If the earlier pages were instances, this is the pattern behind them.

!!! tip "Rapid Recall"
    A GLM picks an exponential-family distribution for the output, sets its natural parameter to a linear score \(\eta=\theta^\top x\), and predicts the mean. The link function is forced by the distribution: identity for Gaussian gives linear regression, sigmoid for Bernoulli gives logistic, softmax for multinomial gives softmax regression. Every member trains with the same error-times-feature update. Generative learning flips the question: model \(P(x\mid y)\) and the prior \(P(y)\), then invert with Bayes. GDA assumes shared-covariance Gaussians per class and is provably equivalent to logistic regression in the posterior, but via stronger assumptions, which is why Gaussian-true small data favors GDA and everything else favors logistic.

## §1 Generalized Linear Models: the keystone

Linear regression, logistic regression, softmax, Poisson regression are all **one recipe**: pick an exponential-family distribution for the output, set its natural parameter \(=\theta^\top x\), and predict the mean.

### The exponential family

$$P(y;\eta) = b(y)\,\exp\big(\eta^\top T(y) - a(\eta)\big)$$

- **\(\eta\)** — natural (canonical) parameter; the GLM links this to the linear model.
- **\(T(y)\)** — sufficient statistic (usually \(T(y)=y\)).
- **\(a(\eta)\)** — log partition function; \(e^{-a(\eta)}\) is the normalizing constant forcing \(\int P=1\).
- **\(b(y)\)** — base measure.

Members: Bernoulli (binary), Gaussian (continuous), Poisson (counts), Gamma/Exponential (waiting times), Beta/Dirichlet (distributions over probabilities), multinomial (categories).

### Bernoulli is exponential-family, and the sigmoid appears

$$\phi^y(1-\phi)^{1-y} = \exp\Big(y\log\tfrac{\phi}{1-\phi} + \log(1-\phi)\Big)$$

So \(\eta = \log\frac{\phi}{1-\phi}\) (the log-odds). Invert it:

$$e^\eta = \frac{\phi}{1-\phi} \;\Rightarrow\; \phi = \frac{1}{1+e^{-\eta}}$$

!!! note "Why the sigmoid is forced"
    The sigmoid is precisely the function mapping the Bernoulli's natural parameter \(\eta\) back to its probability \(\phi\). Logistic regression doesn't "choose an S-curve" — the Bernoulli's structure produces the sigmoid automatically.

### Gaussian is exponential-family too

With \(\sigma^2=1\): \(\eta=\mu\) (the natural parameter *is* the mean), \(T(y)=y\), \(a(\eta)=\tfrac12\eta^2\). That's why linear regression connects \(\theta^\top x\) directly to the prediction with no squashing — its link is the identity.

### Properties (why GLMs are well-behaved)

1. MLE w.r.t. \(\eta\) is concave — the NLL is **convex** (no bad local optima).
2. \(\mathbb E[y;\eta] = \dfrac{\partial a}{\partial\eta}\) — the mean is the first derivative of \(a\).
3. \(\mathrm{Var}(y;\eta) = \dfrac{\partial^2 a}{\partial\eta^2}\) — the variance is the second derivative (Hessian if \(\eta\) is a vector).

### The three GLM assumptions

1. \(y\mid x;\theta \sim\) ExponentialFamily\((\eta)\) — pick the member matching the output type.
2. **Design choice:** \(\eta = \theta^\top x\) (the "linear" in GLM).
3. **Prediction:** output \(\mathbb E[y\mid x]\).

**Recipe in action.** Gaussian → \(h(x)=\mathbb E[y\mid x]=\mu=\eta=\theta^\top x\) → *linear regression*. Bernoulli → \(h(x)=\phi=\frac{1}{1+e^{-\theta^\top x}}\) → *logistic regression*, sigmoid for free.

<figure class="diagram diagram-light" markdown="0">
<svg aria-label="GLM parameterization chain" role="img" viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg">
<g fill="#23201a" font-family="monospace" font-size="13">
<rect fill="#eef4f1" height="50" rx="4" stroke="#1d4d4f" stroke-width="1.5" width="120" x="30" y="100"></rect>
<text fill="#1d4d4f" text-anchor="middle" x="90" y="122">model θ</text>
<text font-size="10" text-anchor="middle" x="90" y="140">(we learn this)</text>
<rect fill="#f7ede9" height="50" rx="4" stroke="#7c2d12" stroke-width="1.5" width="120" x="260" y="100"></rect>
<text fill="#7c2d12" text-anchor="middle" x="320" y="128">natural η</text>
<rect fill="#f3ede0" height="50" rx="4" stroke="#9a7b2e" stroke-width="1.5" width="120" x="490" y="100"></rect>
<text fill="#9a7b2e" text-anchor="middle" x="550" y="122">mean</text>
<text font-size="10" text-anchor="middle" x="550" y="140">φ / μ / λ</text>
<line marker-end="url(#ah)" stroke="#4a4438" stroke-width="1.6" x1="150" x2="258" y1="125" y2="125"></line>
<text font-size="11" text-anchor="middle" x="204" y="116">η = θᵀx</text>
<line marker-end="url(#ah)" stroke="#4a4438" stroke-width="1.6" x1="380" x2="488" y1="125" y2="125"></line>
<text font-size="11" text-anchor="middle" x="434" y="116">response g</text>
<text fill="#7c2d12" font-size="10" text-anchor="middle" x="434" y="160">(sigmoid / identity / exp)</text>
<defs><marker id="ah" markerHeight="9" markerWidth="9" orient="auto" refX="7" refY="4.5">
<path d="M0,0 L9,4.5 L0,9 z" fill="#4a4438"></path></marker></defs>
<text fill="#1d4d4f" font-size="14" text-anchor="middle" x="320" y="40">The one chain behind every GLM</text>
</g>
</svg>
<figcaption>You learn θ; it produces η linearly; the canonical response function g maps η to the distribution's mean (sigmoid for Bernoulli, identity for Gaussian, exp for Poisson).</figcaption>
</figure>

### Unified training

Every GLM shares the identical update — the reason linear, logistic, and Poisson regression all use "error × feature":

$$\theta_j := \theta_j + \alpha\big(y^{(i)} - h_\theta(x^{(i)})\big)x_j^{(i)}$$

Only \(h_\theta(x)\) changes (identity vs sigmoid vs exp). Newton/Fisher Scoring is the common optimizer because GLMs are low-dimensional and convex. Terminology: **canonical response function** \(g(\eta)=\mathbb E[y;\eta]\) maps \(\eta\to\) mean; **canonical link** \(g^{-1}\) maps mean \(\to\eta\).

## §2 Softmax regression (multiclass)

For \(K\) classes, the matching family member is the multinomial → softmax. Each class gets a parameter vector \(\theta_k\); labels are one-hot. Compute \(K\) logits \(\theta_k^\top x\), then exponentiate (forces positivity) and normalize (forces sum to 1):

$$P(y=k\mid x) = \frac{e^{\theta_k^\top x}}{\sum_{i\in\text{classes}} e^{\theta_i^\top x}}$$

The loss is **cross-entropy** between the one-hot truth \(P\) and the predicted bars \(\hat P\):

$$\text{CrossEntropy}(P,\hat P) = -\sum_{y\in\text{classes}} P(y)\log\hat P(y)$$

Since \(P\) is one-hot, this collapses to \(-\log\hat P(\text{true class})\) — minimizing it maximizes the predicted log-probability of the correct class. \(K=2\) softmax reduces to the sigmoid; this is the same cross-entropy as the negative Bernoulli log-likelihood from [logistic regression](logistic-perceptron.md).

## §3 Generative learning and GDA

**Discriminative** models (everything so far) learn \(P(y\mid x)\) directly — where's the boundary. **Generative** models learn \(P(x\mid y)\) (what each class's data looks like) plus priors \(P(y)\), then invert with Bayes. Intuition: instead of "where's the line between cats and dogs," learn "what cats look like" and "what dogs look like" separately, then ask which model better explains a new point.

$$P(y\mid x) = \frac{P(x\mid y)P(y)}{P(x)}, \qquad \arg\max_y P(y\mid x) = \arg\max_y P(x\mid y)P(y)$$

(\(P(x)\) is constant across classes, so it drops out of the argmax — you never compute the denominator at prediction time.)

### Gaussian Discriminant Analysis

Assume each class's features are multivariate Gaussian:

$$P(x;\mu,\Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\!\Big(-\tfrac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\Big)$$

The exponent is the squared *Mahalanobis distance* (distance from the mean, stretched by the covariance). The 2-class model:

- \(y\sim\) Bernoulli\((\phi)\) — the class prior.
- \(x\mid y=0\sim\mathcal N(\mu_0,\Sigma)\), \(x\mid y=1\sim\mathcal N(\mu_1,\Sigma)\).

!!! note "Shared Σ is the key detail"
    Both classes share *one* covariance \(\Sigma\) (different means). This makes the decision boundary *linear* (and cheaper). Per-class \(\Sigma\) would give a quadratic boundary — that's QDA.

### Closed-form training (no gradient descent)

Maximize the joint log-likelihood \(\sum_i\log P(x^{(i)}\mid y^{(i)})P(y^{(i)})\); the estimates are intuitive "count and average":

$$\phi = \tfrac1m\sum_i \mathbb{1}\{y^{(i)}=1\}, \qquad \mu_c = \frac{\sum_i \mathbb{1}\{y^{(i)}=c\}\,x^{(i)}}{\sum_i \mathbb{1}\{y^{(i)}=c\}}$$

$$\Sigma = \tfrac1m\sum_i (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^\top$$

Priors by counting, class centers by averaging within class, shape by pooling deviations.

## §4 GDA and logistic regression

View the GDA posterior as a function of \(x\). Because \(\Sigma\) is shared, the quadratic-in-\(x\) terms cancel and you get:

$$P(y=1\mid x) = \frac{1}{1+\exp(-\theta^\top x)}$$

for a \(\theta\) built from \((\phi,\mu_0,\mu_1,\Sigma)\). **Exactly the logistic form.**

!!! warning "The asymmetry (top interview question)"
    \(P(x\mid y)\) Gaussian (shared Σ) \(\Rightarrow\) \(P(y\mid x)\) logistic — **TRUE**. \(P(y\mid x)\) logistic \(\not\Rightarrow\) \(P(x\mid y)\) Gaussian — **does NOT hold**. Logistic posteriors arise from *any* exponential-family class-conditional (Poisson too). Gaussian is just one road to logistic.

!!! note "Bias-variance reading (the practical takeaway)"
    GDA = stronger assumptions → more data-efficient *if the data really is Gaussian* (then GDA > logistic). Logistic = weaker assumptions → more robust (if not Gaussian, logistic > GDA). Corollary: *for larger datasets, prefer logistic* — with enough data you don't need GDA's efficiency boost, so you take logistic's robustness for free. GDA's niche is small data + believable Gaussian structure.

## Interview questions

**Q1: What is the GLM recipe and what unifies the family?**
Choose an exponential-family distribution for the output, set its natural parameter equal to a linear score \(\eta=\theta^\top x\), and predict the mean \(\mathbb E[y\mid x]\). The link function is forced by the distribution, identity for Gaussian, sigmoid for Bernoulli, softmax for multinomial, so the modeler does not pick the S-curve. Every member trains with the identical error-times-feature update and is convex, so Newton or Fisher Scoring works well.

**Q2: Why is the sigmoid not an arbitrary choice?**
Writing the Bernoulli in exponential-family form shows its natural parameter is the log-odds \(\eta=\log\frac{\phi}{1-\phi}\). Inverting that to recover the probability gives exactly \(\phi=1/(1+e^{-\eta})\), the sigmoid. So logistic regression's sigmoid falls out of the Bernoulli structure rather than being chosen.

**Q3: What is the difference between discriminative and generative models, and where does GDA sit?**
Discriminative models learn \(P(y\mid x)\) directly, modeling the boundary, while generative models learn \(P(x\mid y)\) and the prior \(P(y)\) and invert with Bayes, modeling what each class looks like. GDA is generative: it fits a shared-covariance Gaussian per class with closed-form count-and-average estimates. The shared covariance is what makes its decision boundary linear; per-class covariance would give QDA.

**Q4: GDA and logistic regression are linked, but the link is one-directional. Explain.**
If the class-conditionals are Gaussian with shared covariance, the posterior \(P(y=1\mid x)\) is provably logistic, so Gaussian implies logistic. The converse fails: a logistic posterior can arise from any exponential-family class-conditional, such as Poisson, so logistic does not imply Gaussian. Practically, GDA's stronger assumptions make it more data-efficient when the data really is Gaussian, while logistic is more robust otherwise, so larger datasets favor logistic.
