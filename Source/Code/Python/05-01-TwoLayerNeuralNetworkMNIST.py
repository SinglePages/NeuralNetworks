#!/usr/bin/env python

"""
TODO(AJC): 
- Change X,y
"""

from timeit import default_timer as timer

import torch

from utilities import format_duration_with_prefix, get_binary_mnist_one_batch


def compute_accuracy(prediction, target):
    valid_N = target.shape[0]
    return 1 - (torch.round(prediction) - target).abs().sum() / valid_N


# Get training and validation data loaders for classes A and B
data_dir = "../../Data"
classA, classB = 1, 7
flatten = True
train_X, train_y, valid_X, valid_y = get_binary_mnist_one_batch(
    data_dir, classA, classB, flatten
)

# Neural network layer sizes for MNIST
n0 = 28 * 28
n1 = 2
n2 = 1

# Network parameters
W1 = torch.randn(n1, n0)
b1 = torch.randn(n1)
W2 = torch.randn(n2, n1)
b2 = torch.randn(n2)


def model(A0):
    Z1 = A0 @ W1.T + b1
    A1 = torch.sigmoid(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = torch.sigmoid(Z2)
    return Z1, A1, Z2, A2.squeeze()


# Batch gradient descent hyper-parameters
num_epochs = 4
learning_rate = 0.01

# Compute initial accuracy (should be around 50%)
_, _, _, valid_preds = model(valid_X)
valid_accuracy = compute_accuracy(valid_preds, valid_y)
print(f"Accuracy before training: {valid_accuracy:.2f}")
print(valid_preds.shape)
print(valid_y.shape)