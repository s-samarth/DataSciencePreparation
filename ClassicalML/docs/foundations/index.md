# Foundations & Linear Models

This section is the supervised-learning spine: one continuous arc from fitting a straight line, through the optimizers and the regularizers that tame it, into logistic regression, the Generalized Linear Model framework that unifies them all, and the generative counterpart (GDA) that loops back to logistic. If you understand why squared error is a Gaussian assumption and why the sigmoid is forced rather than chosen, the rest of classical ML reads as variations on these themes.

!!! tip "Rapid Recall"
    Linear and logistic regression are not two tricks, they are two instances of one recipe. Pick an exponential-family distribution for the output (Gaussian gives linear, Bernoulli gives logistic, multinomial gives softmax), set the natural parameter to a linear score, and predict the mean. The link function is forced by the distribution, and every member trains with the same "error times feature" update. Squared error is the Gaussian MLE; cross-entropy is the Bernoulli MLE; the loss is secretly a noise assumption. Regularization adds a budget on weight size that fixes both overfitting and the explosion when features outnumber samples.

## What each page covers

- **[Linear Regression](linear-regression.md)** sets up the model, the squared-error loss, the closed-form normal equation (with the trace derivation), the probabilistic reason the loss is squared, and locally weighted regression.
- **[Gradient Descent & the LMS Rule](gradient-descent.md)** walks the optimizer: batch, stochastic, and mini-batch, the self-correcting Widrow-Hoff update, and why the loss is convex (the Hessian and positive semi-definiteness).
- **[Assumptions & Diagnostics](assumptions-diagnostics.md)** hangs the six assumptions on one scene and reads them off the residual plot and the Q-Q plot.
- **[Regularization](regularization.md)** explains Ridge, Lasso, and ElasticNet, the geometry of why L1 zeros weights, and why coefficients explode when features outnumber samples.
- **[Logistic Regression & the Perceptron](logistic-perceptron.md)** moves to classification: the sigmoid, cross-entropy, Newton's method, and why the perceptron sits outside the probabilistic framework.
- **[GLMs, Softmax & Generative Learning](glms-generative.md)** is the keystone: the exponential family, the GLM recipe, softmax, and Gaussian Discriminant Analysis with its provable link back to logistic regression.

## The whole arc in one paragraph

Linear and logistic regression are two instances of one recipe (GLMs): pick an exponential-family distribution for your output (Gaussian for linear, Bernoulli for logistic, multinomial for softmax, Poisson for Poisson regression), set the natural parameter \(\eta=\theta^\top x\), and predict \(\mathbb E[y\mid x]\). The link function (identity, sigmoid, softmax) is forced by the distribution, not chosen, and every one of them trains with the identical "error times feature" update because that is a structural GLM property. Squared error itself is the Gaussian MLE; cross-entropy is the Bernoulli MLE. Regularization adds a budget on weights (Ridge's circle shrinks all, Lasso's corners zero some), fixing both overfitting and the \(p>n\) weight-explosion by tie-breaking toward small weights. Newton's method (Fisher Scoring) is the fast curvature-aware optimizer for these low-dimensional convex problems; the perceptron mimics the update but sits outside the framework. Generative models attack from the opposite side, modeling \(P(x\mid y)\) and \(P(y)\) and inverting with Bayes, and GDA with shared-covariance Gaussians lands provably back on the logistic form.
