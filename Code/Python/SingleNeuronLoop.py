#!/usr/bin/env python

from random import random
from typing import List, Optional


def uniform(hilo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Generate a random number uniformly in the given range.

    Args:
        hilo (Optional[float]): If no arguments are given then hilo defaults to 0;
            if one argument is given then it acts as the upper limit and the lower
            limit defaults to 0; otherwise this argument is the lower limit.
        hi (Optional[float]): The upper limit. Defaults to 1.

    Returns:
        float: a random number in [lo, hi]
    """
    if hilo == None and hi == None:
        lo, hi = 0, 1
    elif hi == None:
        lo, hi = 0, hilo
    else:
        lo, hi = hilo, hi
    return random() * (hi - lo) + lo


def ReLU_Activation(z: float) -> float:
    """Rectified linear unit (ReLU) function.

    Args:
        z (float): any real-value number

    Returns:
        float: max(0, z)
    """
    return max(0, z)


def vector_dot(w: List[float], x: List[float]) -> float:
    """Compute the dot product between two vectors.

    Args:
        w (List[float]): A list of weight paramter values.
        x (List[float]): A list of input features

    Returns:
        float: the dot product
    """
    return sum(wk * xk for wk, xk in zip(w, x))


# The number of examples in our dataset
N = 100

# Randomly generate some data
x1 = [uniform(-20, 40) for _ in range(N)]  # Temperature
x2 = [uniform(1e6) for _ in range(N)]  # Illuminance
x3 = [uniform(24 * 60 * 60) for _ in range(N)]  # Time of day
x4 = [round(uniform(365))]  # Day of year

# Group all input features together
x = [x1, x2, x3, x4]

# Generate random parameters
w = [uniform(-1, 1) for _ in range(len(x))]
b = 0
# TODO: why can we start b at 0 by not w?

# Compute model output
z = vector_dot(w, x)
a = ReLU_Activation(z)
