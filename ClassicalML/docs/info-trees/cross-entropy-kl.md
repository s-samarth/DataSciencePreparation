# Cross-Entropy, KL & Log-Loss

Entropy assumed one known distribution. Splitting it into truth versus belief gives cross-entropy, the loss that trains essentially every classifier and every language model. This page covers cross-entropy and its coding intuition, KL divergence and its asymmetry, why the logistic loss is cross-entropy rather than entropy, and how knowledge distillation falls out of the same machinery.

!!! tip "Rapid Recall"
    Cross-entropy \(H(p,q)=-\sum_i p_i\log q_i\) is the average surprise you pay when reality is p but you encoded for q, and with one-hot labels it collapses to \(-\log q_c\), the log-loss and negative log-likelihood. KL is the wasted bits, \(H(p,q)-H(p)\), non-negative and asymmetric, so it is not a distance. Because the label entropy \(H(p)\) is a constant, minimizing cross-entropy equals minimizing KL equals MLE, and only KL is the part the model can reduce. Knowledge distillation is just cross-entropy against a teacher's soft labels, using forward KL so the student covers the teacher's full distribution.

## §1 Cross-entropy

Entropy assumed one known distribution. Cross-entropy splits it into truth vs belief.

- \(p\) = the **true** distribution (reality, the actual labels).
- \(q\) = the **model's predicted** distribution.

!!! note "Definition"
    $$H(p,q)=-\sum_i p_i\log q_i$$
    Average surprise you *actually experience* when reality is \(p\) but you encoded expectations using \(q\). The log holds your belief \(q\); the weighting out front is what truly happens \(p\).

### Coding intuition (the "why we care" anchor)

Design a codebook optimized for \(q\), short codes for events you *think* are common. But events actually arrive per \(p\). Your average message length is \(H(p,q)\). If beliefs were perfect (\(q=p\)) you'd hit the floor \(H(p)\). Any mismatch costs extra bits, so always \(H(p,q)\ge H(p)\).

### Where we measure it: the classifier loss

In supervised classification the true label is one-hot: \(p=[0,\dots,1,\dots,0]\), all mass on correct class \(c\). Plug in:

$$H(p,q)=-\sum_i p_i\log q_i=-\log q_c$$

Every term dies except the correct class. So **cross-entropy loss = negative log-probability assigned to the right answer.** Confident-and-wrong is punished brutally (\(-\log q_c\to\infty\) as \(q_c\to 0\)), exactly the gradient signal you want.

| Setting | How CE appears |
| --- | --- |
| Image / tabular classification | One CE per example over the class softmax. |
| LLM training | *This is* the objective. Per token: distribution over vocab; true label = actual next token (one-hot); loss \(=-\log q(\text{next token})\), averaged over positions. |
| Perplexity | \(=e^{\text{cross-entropy}}\), same number, exponentiated into "effective vocabulary size." |

Minimizing cross-entropy = maximizing likelihood. \(-\log q_c\) summed over data is exactly the negative log-likelihood. "Train with cross-entropy" and "do MLE" are the same statement.

## §2 KL divergence

"Cross-entropy minus entropy," the extra surprise from using the wrong distribution.

!!! note "Definition"
    $$D_{KL}(p\parallel q)=H(p,q)-H(p)=\sum_i p_i\log\frac{p_i}{q_i}$$
    KL = cross-entropy minus entropy = the wasted bits. Cross-entropy is the total bill; entropy is the unavoidable floor; KL is the overpayment from your model being wrong.

| Property | Detail |
| --- | --- |
| Non-negative | \(D_{KL}\ge 0\), \(=0\) iff \(p=q\) (Gibbs' inequality, same Jensen argument as IG ≥ 0). |
| **Asymmetric** | \(D_{KL}(p\parallel q)\neq D_{KL}(q\parallel p)\). *Not* a distance, a frequent interview trap. |
| Forward KL | Mean-seeking / mass-covering. |
| Reverse KL | Mode-seeking. (Shows up in variational inference, RLHF.) |

Why classifiers minimize CE, not KL: \(H(p)\) depends only on fixed labels, not on \(\theta\), a constant during training. So \({\nabla}_{\theta}H(p,q)={\nabla}_{\theta}D_{KL}(p\parallel q)\): identical gradients. CE is just cheaper to compute, so that's what the loss says.

## §3 MLE, log-loss, and "are we optimizing entropy?"

The confusion: the logistic / NN classification loss *looks* like entropy. It isn't, it's cross-entropy, and that resolves everything.

### The formula you stare at

Logistic regression, one example, true label \(y\in\{0,1\}\), predicted probability \(\hat{y}\):

$$L=-[\,y\log\hat{y}+(1-y)\log(1-\hat{y})\,]$$

It *resembles* binary entropy \(H(p)=-[p\log p+(1-p)\log(1-p)]\). Look at the arguments:

|  | Weight out front | Inside the log | How many distributions |
| --- | --- | --- | --- |
| Entropy \(H(p)\) | \(p\) | \(p\) | One (measures its own disorder) |
| Your loss | \(y\) (truth) | \(\hat{y}\) (prediction) | **Two** → cross-entropy |

!!! note "Resolution"
    You are *not* optimizing the entropy of your predictions. You minimize the *cross-entropy between true labels and predictions*: \(p=[1-y,y]\), \(q=[1-\hat{y},\hat{y}]\).

### MLE produces exactly this

Bernoulli likelihood of one label: \(\,\mathrm{P}(y\mid\hat{y})={\hat{y}}^{\,y}(1-\hat{y})^{1-y}\) (check: \(y=1\to\hat{y}\); \(y=0\to1-\hat{y}\)). Negative log:

$$-\log\mathrm{P}(y\mid\hat{y})=-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$$

That is the loss, verbatim. **Negative log-likelihood and cross-entropy are the same expression**, one derived from "maximize probability of data," the other from "match two distributions."

### The decomposition that kills the confusion

Over the dataset, with \(p\) the true labels and \(q\) the predictions:

$$\underbrace{H(p,q)}_{\text{what you minimize}}=\underbrace{H(p)}_{\text{entropy of true labels (constant)}}+\underbrace{D_{KL}(p\parallel q)}_{\text{how wrong the model is}}$$

- \(H(p)\): entropy of the fixed labels, your weights can't change it. The irreducible noise floor.
- \(D_{KL}(p\parallel q)\): the only part parameters touch.

!!! note "Direct answer"
    The entropy of the *predictions* never appears. Minimizing the loss drives down \(D_{KL}(p\parallel q)\): pushing your predicted distribution toward the true one. Minimize cross-entropy = minimize KL = MLE. Three names, one optimization.

Sanity check: at the optimum with hard 0/1 labels and \(\hat{y}=y\), cross-entropy loss \(\to 0\), and the entropy of a hard (one-hot) label is also \(H(p)=0\). Both pieces of the decomposition vanish, they snap together.

## §4 Knowledge distillation (KD) loss

The payoff of the cross-entropy / KL machinery: it *is* "cross-entropy minus entropy," which is why KD belongs in this thread.

### Setup and the insight

A big accurate **teacher**, a small cheap **student** you want to mimic it. Naive training uses hard one-hot labels. KD's insight: the teacher's *full output distribution* carries far more, the "dark knowledge."

Teacher sees a dog photo → `{dog:0.9, wolf:0.08, cat:0.015, car:0.005}`. The one-hot label only says "dog." The soft distribution says *dog looks like wolf, a bit like cat, nothing like car*, relational structure you teach the student.

### The loss

$$L_{KD}=D_{KL}(q_T\parallel q_S)=\underbrace{H(q_T,q_S)}_{\text{cross-entropy}}-\underbrace{H(q_T)}_{\text{teacher entropy (fixed)}}$$

There is the "cross-entropy minus entropy." Teacher frozen implies \(H(q_T)\) constant implies **minimizing KD-KL = minimizing the cross-entropy between teacher and student.** The student does CE training with the teacher's soft probabilities as "labels."

### Temperature

Teacher probs are often too peaked (0.9 drowns the useful 0.08). Soften both with temperature \(T\):

$$q_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$$

Higher \(T\) flattens the distribution, amplifying small-logit (dark-knowledge) signal. Train at high \(T\), deploy at \(T=1\). KD gradients scale as \(1/T^2\), so the KD term is multiplied by \(T^2\) to stay comparable to the hard-label term.

### Full objective in practice

$$L=\alpha\,\underbrace{H(p_{\text{true}},q_S)}_{\text{hard-label CE}}+(1-\alpha)\,T^2\,\underbrace{D_{KL}(q_T^{(T)}\parallel q_S^{(T)})}_{\text{soft KD term}}$$

Learn from ground truth *and* mimic the teacher.

| When / why | Use case |
| --- | --- |
| Compression for deployment | Shrink a giant model for phone / edge / low-latency serving, the dominant use case. |
| Small-model ecosystem | Most "distilled" / small open-weight models (DistilBERT lineage, small variants of model families) are KD products, they beat same-size from-scratch models, since soft targets are richer signal. |
| Ensemble compression | Distill a 10-model ensemble into one network. |
| Label smoothing connection | A degenerate cousin: KD against a uniform "teacher." |

!!! warning "Why teacher-first KL, not student-first?"
    \(D_{KL}(q_T\parallel q_S)\) (forward, teacher first) is *mass-covering*: the student is penalized wherever the teacher puts mass but the student doesn't, it must reproduce the teacher's full distribution including the dark-knowledge tail. Reverse \(D_{KL}(q_S\parallel q_T)\) would be mode-seeking: the student could collapse onto the teacher's single peak and ignore the relational tail, defeating the point of distillation.

!!! note "Unifying model"
    Everything here is cross-entropy. Classifier training = CE against one-hot labels. Distillation = CE against a teacher's soft labels. KL = the part of CE your model can reduce, because the rest \(H(p)\) is a fixed floor.

## Interview questions

**Q1: How does cross-entropy become the standard classification loss?**
With a one-hot true label, \(H(p,q)=-\sum_i p_i\log q_i\) collapses to \(-\log q_c\), the negative log-probability of the correct class, since every other term has zero weight. Summed over data this is exactly the negative log-likelihood, so minimizing cross-entropy equals maximizing likelihood. It punishes confident-and-wrong predictions toward infinity, which is the gradient signal you want.

**Q2: Why is KL divergence not a distance?**
Because it is asymmetric: \(D_{KL}(p\parallel q)\neq D_{KL}(q\parallel p)\) in general, so it violates the symmetry a metric requires. It is still non-negative and zero only when the distributions match, by Gibbs' inequality. Forward KL is mass-covering and reverse KL is mode-seeking, a distinction that matters in variational inference and RLHF.

**Q3: The logistic loss looks like entropy. Why is it actually cross-entropy?**
Entropy weights and takes the log of the same single distribution, measuring its own disorder, whereas the logistic loss weights by the truth y and takes the log of the prediction \(\hat y\), so it involves two distributions. Decomposing it gives \(H(p,q)=H(p)+D_{KL}(p\parallel q)\), where the label entropy \(H(p)\) is a constant the weights cannot change, so training only reduces the KL term. The entropy of the predictions never appears.

**Q4: In knowledge distillation, why use the teacher-first KL and a temperature?**
Forward KL with the teacher first is mass-covering, forcing the student to reproduce the teacher's entire distribution including the dark-knowledge tail, whereas reverse KL is mode-seeking and would let the student collapse onto the top peak. Temperature softens both distributions so the informative small-logit probabilities are not drowned by the dominant class, and because KD gradients scale as \(1/T^2\) the soft term is multiplied by \(T^2\) to stay comparable to the hard-label term.
