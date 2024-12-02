# Basic (Multiclass) CNN Classifier
Simple CNN classifier for the CIFAR-10 Dataset.

## Techniques it uses
- Early Stopping
- Most basic image augmentation techniques: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10699340
- Good set of augmentations found via forward selection
- Batchnormalization
- Weight Decay
- Cosine Learning rate Scheduling
- ResNet Blocks https://arxiv.org/pdf/1512.03385
- GELU activation function
- AdamW optimizer
- Squeeze and Exitation Layers https://arxiv.org/pdf/1709.01507
- Roughly follow scaling used by EfficientNet https://arxiv.org/pdf/1905.11946

## Accuracy
Test Acc: 0.822, increases with larger network sizes.