#!/usr/bin/env python

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

# Let's start by simply printing out some basic information
print("Training input shape    :", X_train.shape)
print("Training target shape   :", y_train.shape)
print("Validation input shape  :", X_valid.shape)
print("Validation target shape :", y_valid.shape)
