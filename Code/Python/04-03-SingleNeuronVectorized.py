#!/usr/bin/env python

import torch


# The number of examples in our dataset
N = 100

# Randomly generate some input data
nx = 4
X = torch.randn(N, nx)

# Generate random neuron parameters
w = torch.randn(nx)
b = 0

# Compute neuron output for each of the N examples
z = X @ w + b
yhat = torch.sigmoid(z)
