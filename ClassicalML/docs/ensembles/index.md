# Tree Ensembles

A single tree is a high-variance learner, so the whole field of tree ensembles is two answers to that problem: average many independent strong trees to kill variance (bagging and random forests), or add up many weak trees sequentially to kill bias (boosting, GBM, XGBoost). This section follows that chain from the variance algebra to the modern boosting libraries.

!!! tip "Rapid Recall"
    Bagging averages independent strong trees to cut variance; the variance of correlated trees is \(\rho\sigma^2+\frac{1-\rho}{B}\sigma^2\), so adding trees only kills the cheap term and the correlation \(\rho\) sets a hard floor. Random forests attack \(\rho\) directly with per-split feature subsampling. Boosting instead adds weak learners sequentially, each fixing the last one's errors, reducing bias: AdaBoost reweights misclassified points, gradient boosting fits the next tree to the negative gradient, and XGBoost uses a regularized second-order objective with a closed-form Newton leaf. Trees still own tabular data in 2026.

## What each page covers

- **[Variance, Bagging & OOB](bagging-oob.md)**: why a single tree overfits, the exact variance-reduction algebra, and bootstrapping with its free out-of-bag error estimate.
- **[Random Forests](random-forests.md)**: feature randomness as an attack on tree correlation, the three hyperparameter levers, and where RF still loses to boosting.
- **[Boosting & AdaBoost](boosting-adaboost.md)**: the sequential-correction contrast, AdaBoost step by step, why its weight is optimal under exponential loss, and the multiclass and regression extensions.
- **[Gradient Boosting](gradient-boosting.md)**: residuals as negative gradients, the regressor/classifier unification, tuning, and Newton's method.
- **[XGBoost & Modern Libraries](xgboost-libraries.md)**: the regularized second-order objective, parallelization and missing-value handling, and XGBoost versus LightGBM versus CatBoost.
- **[Trees vs Other Models](trees-vs-others.md)**: when boosted trees dominate, when other models win, and why trees need almost no preprocessing.

## One-page formula strip

- **Variance of averaged correlated models:** \(\rho\sigma^2+\frac{1-\rho}{B}\sigma^2\), floor \(\rho\sigma^2\) (lowering \(\rho\) is RF's whole goal).
- **OOB fraction left out:** \((1-\frac{1}{n})^n\to e^{-1}\approx 0.368\).
- **RF defaults:** \(m=\sqrt{d}\) (classification), \(m=d/3\) (regression).
- **AdaBoost weighted error:** \({\epsilon}_t=\sum_{i:\text{wrong}}w_i\).
- **AdaBoost learner say:** \({\alpha}_t=\frac{1}{2}\ln\frac{1-{\epsilon}_t}{{\epsilon}_t}\) (from minimizing exponential loss).
- **AdaBoost final:** \(H(x)=\text{sign}(\sum_t{\alpha}_t h_t(x))\); SAMME adds \(+\ln(K-1)\).
- **GBM pseudo-residual:** \(r_i=-\partial L/\partial F\big|_{F_{m-1}}\); squared error gives \(y-F\); log loss gives \(y-p\).
- **GBM update:** \(F_m=F_{m-1}+\nu\,h_m\); predict \(F_0+\nu\sum_m h_m(x)\) (classifier: sigmoid/softmax).
- **Newton step:** \(\theta-L'/L''\) (1-D), \(\theta-H^{-1}\nabla L\) (multi-D).
- **XGBoost g, h:** squared error \((\hat{y}-y,\,1)\); log loss \((p-y,\,p(1-p))\).
- **XGBoost leaf:** \(w_j^*=-\frac{G_j}{H_j+\lambda}\); **gain** \(\frac{1}{2}\big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\big]-\gamma\).
- **Reg objective:** \(\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_j w_j^2\) (γ per leaf, λ = L2; α = L1; min_child_weight = min \(\sum h_i\)).
