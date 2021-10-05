from utilities import get_binary_mnist_one_batch, format_duration_with_prefix
from timeit import default_timer as timer
import torch


def compute_accuracy(yhat, y):
    valid_N = y.shape[0]
    return 1 - (torch.round(yhat) - y).abs().sum() / valid_N


# Get training and validation data loaders for classes A and B
data_dir = "../../Data"
classA, classB = 1, 7
flatten = True
train_X, train_y, valid_X, valid_y = get_binary_mnist_one_batch(
    data_dir, classA, classB, flatten
)

# Neuron parameters
nx = 28 * 28
w = torch.randn(nx) * 0.01
b = torch.zeros(1)

# Batch gradient descent hyper-parameters
num_epochs = 4
learning_rate = 0.01

# Compute initial accuracy (should be around 50%)
valid_yhat = torch.sigmoid(valid_X @ w + b)
valid_accuracy = compute_accuracy(valid_yhat, valid_y)
print(f"Accuracy before training: {valid_accuracy:.2f}")

# Learn values for w and b that minimize loss
for epoch in range(num_epochs):

    start = timer()

    # Make predictions given current paramters and then compute loss
    yhat = torch.sigmoid(train_X @ w + b)
    loss = -(train_y * torch.log(yhat) + (1 - train_y) * torch.log(1 - yhat))

    # Compute derivatives for w and b (dz is common to both derivatives)
    dz = yhat - train_y
    dw = (1 / train_y.shape[0]) * (dz @ train_X)
    db = dz.mean()

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Report on progress
    valid_yhat = torch.sigmoid(valid_X @ w + b)
    valid_accuracy = compute_accuracy(valid_yhat, valid_y)

    info = f"{epoch+1:>2}/{num_epochs}"
    info += f", Cost={loss.mean():0.1f}"
    info += f", Accuracy={valid_accuracy:.2f}"
    info += f", Time={format_duration_with_prefix(timer()-start)}"
    print(info)
