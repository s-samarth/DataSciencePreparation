# Bayesian Estimation & MAP

Bayesian estimation turns prior knowledge plus data into an updated belief, and maximum a posteriori picks the
single most probable parameter from that belief. The punchline for machine learning: MAP is exactly
regularized maximum likelihood, which is why regularization is everywhere.

!!! tip "Rapid Recall"
    Bayes combines a prior $P(\theta)$ with a likelihood $P(\text{data}\mid\theta)$ to give a posterior
    $P(\theta\mid\text{data})\propto P(\text{data}\mid\theta)P(\theta)$, where the denominator is just a
    normalizer. Use it when you have meaningful prior knowledge, scarce data, a need for full uncertainty, or
    continuous updating. MAP picks the posterior mode, $\arg\max_\theta P(\text{data}\mid\theta)P(\theta)$,
    identical to MLE except for the prior term. L2 regularization is MAP with a Gaussian prior and L1 with a
    Laplace prior, so the regularization strength is the strength of your prior. Three heads in three flips
    gives MLE 1.0 but a sensible MAP near 0.71.

## §12 Bayesian Estimation & MAP

**Prior, likelihood, posterior.**

- **Prior $P(\theta)$:** belief about the parameter *before* data. Where outside knowledge enters.
- **Likelihood $P(\text{data}\mid\theta)$:** how probable the observed data is for each $\theta$, the same likelihood as MLE.
- **Posterior $P(\theta\mid\text{data})$:** updated belief *after* data, the output.

$$P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \times P(\theta)$$

Your updated belief is your prior reweighted by how well each parameter value explains the data you actually
saw. Posterior is proportional to likelihood times prior (the denominator $P(\text{data})$ is just a
normalizing constant).

**Why and when we use it.**

- **Meaningful prior knowledge** (a hundred similar A/B tests landing near 10 to 15%). MLE throws that away; Bayesian updates from it.
- **Scarce data.** 3 heads in 3 flips: MLE says $P(\text{heads}) = 1.0$ ("never tails"). A reasonable prior pulls it back to sanity. Less data means the prior protects more; as data grows the prior fades.
- **Need uncertainty, not just a point.** The posterior is a full distribution, giving "95% probability the rate is 9 to 14%" (a credible interval).
- **Continuous updating.** Today's posterior is tomorrow's prior, powers bandits, spam filters, recommenders.

**MAP, maximum a posteriori.** The full posterior is rich, but often you want one best number. MAP asks: **what
single $\theta$ is most probable given data and prior?** the peak (mode) of the posterior.

$$\hat\theta_{\text{MAP}} = \arg\max_{\theta} P(\theta \mid \text{data}) = \arg\max_{\theta} P(\text{data} \mid \theta) P(\theta)$$

Compare with MLE, identical except MAP includes the prior term $P(\theta)$:

$$\hat\theta_{\text{MLE}} = \arg\max_{\theta} P(\text{data} \mid \theta)$$

**Why MAP is everywhere in ML.**

- **L2 (Ridge / weight decay) is MAP with a Gaussian prior** on the weights ("weights should be small, near zero"). The prior's belief shrinks them.
- **L1 (Lasso) is MAP with a Laplace prior.** The Laplace's sharp spike at zero is why L1 drives weights to *exactly* zero, sparsity and feature selection.

The regularization strength $\lambda$ **is** the strength of your prior: strong $\lambda$ means a confident
prior so data must fight harder; weak $\lambda$ means a vague prior so data dominates. Regularization is
everywhere in ML, and regularization is MAP, that is why MAP is everywhere. See
[Likelihood & MLE](../estimation/likelihood-mle.md) for the maximum-likelihood half.

A worked coin example, three heads in three flips. The MLE is degenerate:

$$\hat\theta_{\text{MLE}} = \frac{\text{heads}}{\text{total}} = \frac{3}{3} = 1.0$$

With a prior worth 2 heads and 2 tails of pseudo-data, the MAP is sane:

$$\hat\theta_{\text{MAP}} = \frac{\text{real heads} + \text{prior heads}}{\text{real total} + \text{prior total}} = \frac{3 + 2}{3 + 4} = \frac{5}{7} \approx 0.71$$

And a Gaussian prior-data blend lands between the two centers, pulled toward whichever side is more confident
(smaller variance):

$$\hat\theta_{\text{MAP}} = \frac{(\text{prec}_{\text{prior}} \times 100) + (\text{prec}_{\text{data}} \times 130)}{\text{prec}_{\text{prior}} + \text{prec}_{\text{data}}}$$

## Interview Questions

**Q1: Write Bayes' rule for estimation and name each term.**
$P(\theta\mid\text{data})\propto P(\text{data}\mid\theta)\,P(\theta)$, where $P(\theta)$ is the prior belief
before data, $P(\text{data}\mid\theta)$ is the likelihood of the observed data, and $P(\theta\mid\text{data})$
is the posterior, the updated belief. The omitted denominator $P(\text{data})$ is just a normalizing constant,
so the posterior is the prior reweighted by the likelihood.

**Q2: How does MAP differ from MLE?**
MAP maximizes the posterior, $P(\text{data}\mid\theta)P(\theta)$, while MLE maximizes only the likelihood
$P(\text{data}\mid\theta)$. They are identical except that MAP includes the prior term, so MAP equals MLE when
the prior is flat. The prior makes MAP more robust when data is scarce, for instance pulling three-heads-in-three
back from a degenerate 1.0.

**Q3: Why is MAP estimation equivalent to regularization?**
Because taking the log of the posterior splits into the log-likelihood plus the log-prior, and the log-prior
acts as a penalty on the parameters. A Gaussian prior gives an L2 penalty (ridge or weight decay) and a Laplace
prior gives an L1 penalty (lasso), so the regularization strength is exactly the strength of the prior. That is
why regularization, which is everywhere in ML, is MAP in disguise.

**Q4: When does the prior matter most, and when does it fade?**
The prior matters most when data is scarce, where it protects against degenerate or extreme estimates, and when
you have genuine outside knowledge to inject. As the sample grows, the likelihood dominates and the posterior
concentrates around the data-driven estimate, so the prior's influence fades and MAP approaches MLE.
