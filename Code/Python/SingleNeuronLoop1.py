#!/usr/bin/env python

from math import exp
from random import uniform


def sigmoid(z: float) -> float:
    """The sigmoid/logistic activation function."""
    return 1 / (1 + exp(-z))


# The number of examples in our dataset
N = 100

# Randomly generate some input data
nx = 4
x1 = [uniform(-20, 40) for _ in range(N)]
x2 = [uniform(0, 1e6) for _ in range(N)]
x3 = [uniform(0, 24 * 60 * 60) for _ in range(N)]
x4 = [round(uniform(1, 365)) for _ in range(N)]

# Generate random neuron parameters
w1, w2, w3, w4 = [uniform(-1, 1) for _ in range(nx)]
b = uniform(-1, 1)

# Compute neuron output for each of the N examples
for x1i, x2i, x3i, x4i in zip(x1, x2, x3, x4):
    zi = w1 * x1i + w2 * x2i + w3 * x3i + w4 * x4i + b
    ai = sigmoid(zi)
