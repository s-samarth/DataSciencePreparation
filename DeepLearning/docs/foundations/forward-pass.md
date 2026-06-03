# The Forward Pass

The forward pass is just function composition. Each layer is a linear transformation followed by a nonlinear activation. Understanding why we need both, and what each contributes, is the foundation everything else builds on.

!!! tip "Rapid Recall"
    A linear layer computes $z = Wx + b$: each output is a weighted sum of all inputs plus a learnable offset. Stacking linear layers without a nonlinearity between them collapses to a single linear layer, so activations are the entire reason depth helps. A 2-layer classifier is linear, then ReLU, then linear to logits; softmax turns logits into probabilities and cross-entropy scores them. In PyTorch you output raw logits and let `nn.CrossEntropyLoss` apply softmax internally for numerical stability.

## §1 What a linear layer does

| Symbol | Meaning |
| --- | --- |
| $x$ | Input vector for one sample, or matrix where each row is a sample for a batch. |
| $W$ | Weight matrix. Learnable. Shape (output_size, input_size) in PyTorch. |
| $b$ | Bias vector. Learnable. Shape (output_size,). |
| $z$ | Output of the linear transformation, before the activation. Called the *pre-activation* or *logit*. |

$$
\begin{aligned}
\text{For one sample:}\quad & z = W \cdot x + b \\[4pt]
\text{For a batch } X \text{ of shape } (B, \text{input\_size}):\quad & Z = X \cdot W^{\top} + b \quad \text{(shape } (B, \text{output\_size}))
\end{aligned}
$$

Intuitively, each output dimension is a weighted sum of all input dimensions, plus a learnable offset. Think of each row of $W$ as a "detector" that weighs the input features in some combination and produces one scalar score. With 64 output dimensions you have 64 such detectors, each looking at the input in a different way.

## §2 Why activations are essential, the proof

This is the most important property of neural networks. Stack two linear layers without an activation between them:

$$
\begin{aligned}
z_1 &= W_1 \cdot x + b_1 \\
z_2 &= W_2 \cdot z_1 + b_2 \quad \text{(no activation)} \\[4pt]
\text{Substitute } z_1 \text{ into } z_2: \quad
z_2 &= W_2 \cdot (W_1 \cdot x + b_1) + b_2 \\
    &= (W_2 \cdot W_1) \cdot x + (W_2 \cdot b_1 + b_2) \\
    &= W' \cdot x + b' \quad \leftarrow \text{ just ONE linear layer}
\end{aligned}
$$

The composition of linear functions is linear. Stacking 100 linear layers without activations is mathematically identical to a single linear layer with weight matrix $W' = W_{100} \cdot W_{99} \cdots W_1$. You haven't increased the model's capacity at all. You've just made the matmul more expensive.

Activations are the entire reason depth is useful. They introduce kinks, curves, and gates into the function the network represents. Stacking enough of them makes the network capable of approximating any continuous function (the universal approximation theorem). The [activation functions](../gradient-flow/activation-functions.md) page covers the specific choices and their gradient behavior.

## §3 The complete forward pass of a 2-layer classifier

| Symbol | Meaning |
| --- | --- |
| $x$ | Input of shape (B, 10). B is batch size, 10 is input features. |
| $W_1, b_1$ | First layer. $W_1$ shape (64, 10), $b_1$ shape (64,). |
| $z_1, a_1$ | Pre-activation and activation of first layer. Shape (B, 64). |
| $W_2, b_2$ | Second layer. $W_2$ shape (3, 64), $b_2$ shape (3,). |
| $z_2$ | Pre-activation of output layer. Shape (B, 3). These are the raw *logits*. |
| $\hat{y}$ | Predicted probabilities. Shape (B, 3). Each row sums to 1. |
| $y$ | True labels. Shape (B,) of integers, or (B, 3) one-hot. |
| $L$ | Loss. A single scalar. |

$$
\begin{aligned}
\textbf{First layer:}\quad
z_1 &= x \cdot W_1^{\top} + b_1 && \text{linear, shape } (B, 64) \\
a_1 &= \mathrm{ReLU}(z_1) = \max(0, z_1) && \text{activation, elementwise} \\[4pt]
\textbf{Second layer (output):}\quad
z_2 &= a_1 \cdot W_2^{\top} + b_2 && \text{shape } (B, 3) \text{ — LOGITS} \\[4pt]
\textbf{Softmax to probabilities:}\quad
\hat{y}_{i,j} &= \frac{\exp(z_{2,i,j})}{\sum_k \exp(z_{2,i,k})} && \text{row-wise normalization} \\[4pt]
\textbf{Cross-entropy:}\quad
L &= -\frac{1}{B}\sum_i \log \hat{y}_{i, y_i} && \text{mean neg log-prob of correct class}
\end{aligned}
$$

Notation: $\hat{y}_{i,j}$ means "predicted probability for sample $i$, class $j$." $y_i$ means "true class label for sample $i$." So $\hat{y}_{i, y_i}$ is "the predicted probability that sample $i$ belongs to its actual correct class."

### What softmax actually does

A linear layer's output is unbounded. We want a probability distribution. Softmax does two things at once:

- Exponentiate each value, making everything positive (and amplifying differences, larger inputs become disproportionately larger outputs).
- Normalize by the sum, ensuring outputs add to 1.

The "soft" in softmax means it is a smooth, differentiable version of the hard argmax. If one logit is much bigger than the others, softmax produces a near-one-hot output. If all logits are similar, softmax produces a near-uniform output.

### Cross-entropy intuition

For each sample, look at the predicted probability of the *correct* class. If the model is confident and correct (probability near 1), log is near 0, contributing little to loss. If wrong (probability near 0), log goes to negative infinity, contributing a huge penalty. The negative sign and the mean over the batch give a scalar to minimize.

!!! note "The elegant cancellation (memorize this)"
    Combining softmax and cross-entropy, and computing the derivative of $L$ with respect to the logits $z_2$: the exp from softmax and the log from cross-entropy cancel beautifully. The result is just $\partial L/\partial z_2 = (\hat{y} - y) / B$, prediction minus true label, divided by batch size. We derive this fully in the [backprop derivation](backprop-derivation.md).

### Why output layers do not apply softmax in PyTorch

PyTorch classification models output raw logits and let `nn.CrossEntropyLoss` apply softmax internally. The reason is numerical stability: separate softmax + log can overflow or underflow (exp of a large number is infinity; log of a tiny number is negative infinity). Combining them uses the log-sum-exp trick, which is numerically robust. At inference, apply softmax explicitly only if you need probabilities; if you just need the predicted class, argmax on logits gives the same answer as argmax on softmax output.

## Interview Questions

**Q1: If you remove all activation functions from a neural network, what do you get?**

A single linear transformation, regardless of how many layers you have. Composition of linear functions is always linear: $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$, which is just $Wx + b$ with different $W$ and $b$. Depth without nonlinearity is useless. Activation functions are what give neural networks their expressive power to approximate arbitrary functions. This is why removing them and adding more layers doesn't help, you're not increasing model capacity, you're just making the matrix multiply more expensive.
