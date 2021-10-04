#!/usr/bin/env python

from math import exp
from random import gauss


def sigmoid(z: float) -> float:
    """The sigmoid/logistic activation function."""
    return 1 / (1 + exp(-z))


# The number of examples in our dataset
N = 100

# Randomly generate some input data
nx = 4
x1 = [gauss(0, 1) for _ in range(N)]
x2 = [gauss(0, 1) for _ in range(N)]
x3 = [gauss(0, 1) for _ in range(N)]
x4 = [gauss(0, 1) for _ in range(N)]

# Generate random neuron parameters
w1 = gauss(0, 1)
w2 = gauss(0, 1)
w3 = gauss(0, 1)
w4 = gauss(0, 1)
b = 0

# Compute neuron output for each of the N examples
for x1i, x2i, x3i, x4i in zip(x1, x2, x3, x4):
    zi = w1 * x1i + w2 * x2i + w3 * x3i + w4 * x4i + b
    ai = sigmoid(zi)
