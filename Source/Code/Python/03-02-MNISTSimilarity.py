#!/usr/bin/env python

from math import inf
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# Location in which to store downloaded data
data_dir = "../Data"

# I used torch.std_mean to find the values given to Normalize
# We will discuss normalization in section 4
mnist_xforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

# Load data files (training and validation partitions)
train_data = MNIST(root=data_dir, train=True, download=True, transform=mnist_xforms)
valid_data = MNIST(root=data_dir, train=False, download=True, transform=mnist_xforms)

# Data loaders provide an easy interface for interactive with data
train_loader = DataLoader(train_data, batch_size=len(train_data))
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

# This odd bit of code forces the train loader to give us all inputs and targets
X_train, y_train = next(iter(train_loader))
X_valid, y_valid = next(iter(valid_loader))

# Let's get the average for each digit based on all training examples
digit_averages = {}
for digit in range(10):
    digit_averages[digit] = X_train[y_train == digit].mean(dim=0).squeeze()


# Next up we need to compare "unknown" images to our average images
def get_most_similar(image: torch.Tensor, averages: dict):
    """Compare the image to each of the averaged images.

    Args:
        image (torch.Tensor): an image represented as a tensor
        averages (dict): a dictionary of averaged images

    Returns:
        the most similar label
    """
    closest_label = None
    closest_distance = inf
    for label in averages:
        distance = (image - averages[label]).abs().mean()
        if distance < closest_distance:
            closest_label = label
            closest_distance = distance
    return closest_label


# Now we can get the most similar label for each validation image
num_correct = 0
for image, label in zip(X_valid, y_valid):
    num_correct += label == get_most_similar(image, digit_averages)

print(f"Percent guessed correctly: {num_correct/len(X_valid)*100:.2f}%")
