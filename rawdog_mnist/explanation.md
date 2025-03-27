# MNIST Dataset AI using NumPy

This document outlines the mathematical foundation for an AI model designed to classify the MNIST dataset. The model consists of two layers, leveraging ReLU and softmax activations. 

## General Equation of the System
Let $\text{Model}(x)$ represent the overall function of the system.

### Definitions:
- $\text{CL1}$: Cost of Layer 1
- $Y$: Training labels
- $\text{CM}$: Cost of the Model

## Layer 1
The first layer applies a ReLU activation to the weighted input plus bias:

$$ \text{L}_1(x_{(784 \times 1)}) = \text{relu}(W^{(1)}_{(32 \times 784)} x_{(784 \times 1)} + b^{(1)}_{(32 \times 1)}) $$

### Where:
- $W^{(1)}$ is the weight matrix for the first layer.
- $b^{(1)}$ is the bias vector for the first layer.
- $x$ is the input vector (a flattened 28x28 image).

## Layer 2
The second layer applies a softmax activation to the weighted input plus bias:

$$ \text{L}_2(x_{(32 \times 1)}) = \text{softmax}(W^{(2)}_{(10 \times 32)} x_{(32 \times 1)} + b^{(2)}_{(10 \times 1)}) $$

### Where:
- $W^{(2)}$ is the weight matrix for the second layer.
- $b^{(2)}$ is the bias vector for the second layer.
- $x$ is the output from the first layer.

## Model
Combining both layers, the model is defined as:

$$ \text{Model}(x) = \text{L}_2(\text{L}_1(x)) $$

## Differences and Gradients
### Difference Between Model Output and True Labels:
$$ \text{D2}_{(10 \times 1)} = \text{Model}(x) - Y $$

### Gradient with Respect to Weights of Layer 2:
$$ dW^{(2)}_{(10 \times 10)} = \frac{1}{m} \cdot \text{D2}_{(10 \times 1)} \cdot \text{L1}^T_{(1 \times 10)} $$

### Gradient with Respect to Bias of Layer 2:
$$ db^{(2)}_{(10 \times 1)} = \frac{1}{m} \cdot \sum \text{D2} $$

### Cost of Layer 1 and Backpropagation:
$$ \text{CL1} = dW^{(2)T} \cdot \text{D2} \cdot g'(\text{activation L2}) $$

### Gradient with Respect to Weights of Layer 1:
$$ dW^{(1)}_{(10 \times 784)} = \frac{1}{m} \cdot \text{CL1}_{(10 \times 1)} \cdot x^T_{(1 \times 784)} $$

### Gradient with Respect to Bias of Layer 1:
$$ db^{(1)}_{(10 \times 1)} = \frac{1}{m} \cdot \sum \text{CL1} $$

## Output variables
$$ W^{(1)} = W^{(1)} - \alpha \cdot dW^{(1)}$$
$$ b^{(1)} = b^{(1)} - \alpha \cdot db^{(1)}$$

## Summary of Variables:
- **$W^{(1)}$**: Weight matrix for the first layer (32x784).
- **$b^{(1)}$**: Bias vector for the first layer (32x1).
- **$W^{(2)}$**: Weight matrix for the second layer (10x32).
- **$b^{(2)}$**: Bias vector for the second layer (10x1).
- **$x$**: Input vector (784x1).
- **$Y$**: Training labels.
- **$\text{D2}$**: Difference between model output and true labels (10x1).
- **$\text{CL1}$**: Cost of Layer 1.
- **$dW^{(2)}$**: Gradient with respect to weights of Layer 2.
- **$db^{(2)}$**: Gradient with respect to bias of Layer 2.
- **$dW^{(1)}$**: Gradient with respect to weights of Layer 1.
- **$db^{(1)}$**: Gradient with respect to bias of Layer 1.

This breakdown provides a clear mathematical explanation of the MNIST classification model using NumPy.
