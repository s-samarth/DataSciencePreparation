# Normalization & Regularization

This section covers two of the five buckets at once. Normalization keeps activations in a well-behaved range so gradients stay stable and you can use higher learning rates. Regularization trades a little training fit for better generalization, so a network with enough capacity to memorize the data learns the signal instead. Learning rate schedules sit alongside both, varying the single most important hyperparameter over training.

!!! tip "Rapid Recall"
    Normalize activations to a clean range: BatchNorm normalizes across the batch (per channel, standard for CNNs), LayerNorm across features within each sample (standard for transformers, works at any batch size), and RMSNorm drops the mean-centering for a cheaper LayerNorm (modern LLMs). Regularize to avoid overfitting: weight decay shrinks weights each step, dropout zeros random units to force redundancy, early stopping halts at the best validation loss, and label smoothing softens one-hot targets to curb overconfidence. Schedule the learning rate with warmup + cosine for transformers and cosine annealing for CNNs.

## §1 What is in this section

- [Normalization](normalization.md): BatchNorm, LayerNorm, and RMSNorm, what each normalizes over, when each breaks, and the comparison table.
- [Regularization](regularization.md): weight decay (L2), dropout, early stopping, and label smoothing.
- [Learning rate schedules](lr-schedules.md): step decay, cosine annealing, warmup + cosine, ReduceLROnPlateau, and one-cycle, with the recommendation matrix.

Normalization is the bridge from the [gradient flow](../gradient-flow/index.md) section (it is another attack on unstable gradients) into the regularization toolkit, and weight decay connects directly back to the [AdamW](../optimization/adaptive-methods.md) story.
