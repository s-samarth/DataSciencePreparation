# Convolutional Networks

Convolutional neural networks are a direct response to a failure: feeding raw images to dense layers explodes the parameter count, throws away spatial structure, and has no translation invariance. CNNs fix all three with local connectivity and weight sharing, and the rest of the architecture is mechanics built on those two ideas.

!!! tip "Rapid Recall"
    A dense layer on a 224×224×3 image needs ~150M parameters in the first layer alone, ignores that nearby pixels are correlated, and relearns a feature for every position. CNNs use local connectivity (a neuron sees a small patch) and weight sharing (the same filter slides everywhere), giving few parameters, preserved spatial structure, and translation equivariance. A convolution is a learnable pattern template you slide over the image, computing a dot product at each position to build a feature map. Everything else, channels, kernel sizes, padding, stride, pooling, and the architectural lineage, is detail on top.

## §1 Why CNNs exist at all

The architecture is a direct response to a failure. Take a modest color image: 224×224×3 = **150,528 numbers**. Feed it to a plain fully-connected layer with 1,000 hidden units. The weight matrix is 150,528 × 1,000 ≈ **150 million parameters in the first layer alone**. Three problems, all fatal:

- **Parameter explosion.** You overfit instantly and can't fit it in memory at scale.
- **No spatial awareness.** A dense layer flattens the image into a 1D vector. Pixel (0,0) and (0,1), neighbors in reality, become two arbitrary entries. It throws away the single most important fact about an image: nearby pixels are correlated, far pixels usually aren't.
- **No translation invariance.** Train a dense net to see a cat in the top-left; move the cat to the bottom-right and it activates entirely different weights. It has effectively never seen this input.

CNNs fix all three with two ideas. **Local connectivity**: a neuron looks at a small patch (for example 3×3), not the whole image, so spatial structure is preserved. **Weight sharing**: the *same* small filter slides across the entire image, one set of weights reused everywhere. This is what gives translation equivariance, a vertical-edge detector finds vertical edges *anywhere*, because it is literally the same weights applied at every location.

Local connectivity plus weight sharing give few parameters, preserved spatial structure, and translation equivariance. Everything else in CNNs is mechanics on top of these two ideas.

## §2 What is in this section

- [Convolution mechanics](convolution-mechanics.md): what a convolution computes, a fully worked numeric example, channels and the weight tensor, kernel sizes, and the 1×1 convolution.
- [Output size, pooling, and the forward pass](output-pooling-forward.md): the output-size formula they grill you on, padding and stride, and pooling.
- [CNN architectures](architectures.md): the LeNet → AlexNet → VGG → ResNet lineage, skip connections, the efficiency directions after ResNet, data augmentation, and a CNN in PyTorch.
