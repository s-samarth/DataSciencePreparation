# Deep Learning Study Notes

A complete, math-first interview-prep reference covering the mechanics of neural networks and the architectures built on them: from the forward pass and backpropagation through optimization, normalization, regularization, and the full loss-function taxonomy, then into convolutional networks, vision and generative models, and the sequence-modeling arc from tokenization up to attention.

!!! tip "Rapid Recall"
    A neural network is a stack of learned transformations; training is a forward pass to predict and a backward pass to assign blame, repeated millions of times. Almost every concept fits one of five buckets: predicting (forward pass), assigning blame (backprop), taking the step (optimizers), staying stable (normalization, clipping, residuals, warmup), and not overfitting (dropout, weight decay, early stopping, label smoothing). The same gradient-highway idea recurs everywhere: the LSTM cell state through time, the residual connection through depth. And almost every loss is the negative log-likelihood of an assumed distribution.

## How this site is organized

The site is built from two dense study documents and split into eight sections, each an overview page plus focused sub-pages. Every page opens with a Rapid Recall callout for fast revision and ends with relevant interview questions where applicable.

### Neural network mechanics

- **[Foundations](foundations/index.md)** — what a network computes, the forward pass, the training loop, backpropagation, the full 2-layer derivation, and a complete training pipeline.
- **[Gradient Flow](gradient-flow/index.md)** — activation functions, vanishing and exploding gradients, and the LSTM and residual gradient highways.
- **[Optimization](optimization/index.md)** — the batch-size spectrum, SGD and momentum, the adaptive family up to AdamW, and modern optimizers.
- **[Normalization & Regularization](regularization/index.md)** — BatchNorm/LayerNorm/RMSNorm, weight decay, dropout, early stopping, label smoothing, and learning rate schedules.
- **[Loss Functions](losses/index.md)** — the unifying maximum-likelihood view, plus regression, classification, ranking, and LLM losses.

### Architectures

- **[Convolutional Networks](cnns/index.md)** — why CNNs exist, convolution mechanics, the output-size formula, pooling, and the architectural lineage to ResNet and beyond.
- **[Vision Tasks & Generative Models](vision-generative/index.md)** — segmentation and detection, GANs, and diffusion.
- **[Sequences & Attention](sequences/index.md)** — tokenization, embeddings, RNNs and gated cells, the encoder/decoder families, and attention as the bridge to the Transformer.

## The five buckets

If you can place any technique into one of these, you understand why it exists:

- **Making predictions (forward pass).** Layers of linear + activation; without activations the stack collapses to one linear layer.
- **Assigning blame (backpropagation).** The chain rule, organized so every gradient is computed once, in roughly the cost of one extra forward pass.
- **Taking the step (optimizers).** SGD, momentum, RMSProp, Adam, AdamW: smarter ways to turn gradients into updates.
- **Keeping it stable (normalization, clipping, residuals, init, warmup).** Everything that keeps gradients from vanishing or exploding through depth.
- **Not overfitting (dropout, weight decay, early stopping, label smoothing).** Trading a little training fit for generalization.
