# Single Neuron

When our model is a single neuron we can only produce a single output. So, $n_y=1$ for this section. Sticking to our MNSIT digits example from above, we could train a single neuron to distinguish between two different classes (e.g., "1" vs "7", "0" vs "non-zero", etc.).

## Notation and Diagram

Here is a diagram representing a single neuron (as we'll see later, a neural network often refers to many of these neurons interconnected):

![A neuron model with separate nodes for linear and activation computations.](img/NeuronSeparate.svg)

The diagram represents the following equations (note that I removed the parenthesis superscript from the diagram to make it a bit easier to read):

\begin{align}
z^{(i)} &= \sum_{k=1}^{n_x} x_k^{(i)} w_k + b\\
a^{(i)} &= g(z^{(i)})
\end{align}

For these two equations:

- $x_k^{(i)}$ are the input features for the $i^{th}$ example (e.g., $k=76$ and $i=7436$ would denote pixel 76 of 784 for image 7436 of 60000)
- $w_k$ (weights) and $b$ (bias) are the **learned** parameters
- $z^{(i)}$ is a weighted sum of the input features plus the additional bias term
- $a^{(i)}$ is the output of a non-linear activation function $g(\mathord{\cdot})$ applied to $z^{(i)}$
- $\hat y^{(i)}$ (pronounced *"y hat"*) is the label we often give to the output ($a^{(i)} = \hat y^{(i)}$)


m4question([[Why do $w_k$ and $b$ not have superscripts?]], [[The parameters $w_k$ and $b$ do not change as the input $x_k^{(i)}$ changes. These parameters **are** the neuron, and they are used to produce the output $\hat y^{(i)}$ for any given input; we use the same parameter values regardless of input.]])


**For this model, we want to find parameters $w_k$ and $b$ such that the neuron outputs $\hat y^{(i)} \approx y$ for any input.** Before we discuss optimization we should take a moment to code up this single neuron model.

Before we continue I should show a more common representation of a neuron model. The image above separates the linear and activation components, but it is more common to show them together in a single node.

![A neuron model.](img/Neuron.svg)

## Neuron with Python Standard Libraries

This code does not include any "learning" (i.e., optimization), but it is worth showing just how simple it is to write a single neuron from scratch. Most of the code below is necessary only to create some faked input data.


m4code(Code/Python/04-01-SingleNeuronLoop.py)


## The Dot-Product

We compute $z^{(i)}$ above using a summation, but we can express this same bit of math using the dot-product from linear algebra.


$$
z^{(i)} = \sum_{k=1}^{n_x} x_k^{(i)} w_k + b = \mathbf{x}^{(i)T} \mathbf{w} + b
$$


The $\mathbf{x}^{(i)T} \mathbf{w}$ part of the equation computes the dot-product between $\mathbf{x}^{(i)T}$ and $\mathbf{w}$. We need to transpose $\mathbf{x}^{(i)}$ to make the dimensions work (i.e., we need to multiply a row vector by a column vector).

This not only turns out to be easier to write/type, but it is more efficiently computed by a neural network library. The code listing below uses [PyTorch](https://pytorch.org/) to compute $z^{(i)}$ (`zi`). Libraries like PyTorch and Tensorflow make use of both vectorized CPU instructions and graphics cards (GPUs) to quickly compute the output of matrix multiplications.


m4diff([[Code/Python/04-01-SingleNeuronLoop.py]], [[Code/Python/04-02-SingleNeuronDot.py]])


The code snippet above shows a [diff](https://en.wikipedia.org/wiki/Diff) between the previous code snippet and an updated one using the dot product. You will see many diffs throughout this document. The key points are that: (1) red indicates text or entire lines that have been removed and (2) green indicates updated or newly added lines.

We do not need to transpose `xi` in code because when we iteration through `X` we get row vectors. As it happens, we can improve efficiency even further.

## Vectorizing Inputs

In addition to using a dot-product in place of a summation, we can use a matrix multiplication in place of looping over all examples in the dataset. In the two equations below we perform a matrix multiplication that computes the output of the network for all examples at once. A neural network library can turn this into highly efficient CPU or GPU operations.


\begin{align}
\mathbf{z} &= X \mathbf{w} + b \\
\mathbf{a} &= g(\mathbf{z})
\end{align}


m4diff([[Code/Python/04-02-SingleNeuronDot.py]], [[Code/Python/04-03-SingleNeuronVectorized.py]])


m4question([[What are the dimensions of $\mathbf{z}$ and $\mathbf{a}$ (aka, $\mathbf{\hat y}$)?]], [[We are computing a single output value for each input, so, the shape of these vectors are $(N \times 1)$. PyTorch will treat these as arrays with $N$ elements instead of as column vectors.
\begin{align}
\mathbf{z} &= m4colvec("\mathbf{x}^{(row)T} \mathbf{w} + b", "N") \\
\mathbf{a} &= m4colvec("g(z^{(row)})", "N")
\end{align}
]])


In the code snippet above, a matrix multiplication is indicated in PyTorch using the `@` symbol (a `*` is used for element-wise multiplications). A key to understanding matrix math is to examine the shapes of all matrices involved. Above, $X$ has a shape of $(N \times n_x)$, $\mathbf{w}$ has a shape of $(n_x \times 1)$, and $b$ is a scalar.

Inner dimensions (the last dimension of the left matrix and the first dimension of the right matrix) must be the same for any valid matrix multiplication. The scalar, $b$, is added element-wise to every element in the final matrix due to [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) (this is a common library feature, not necessarily standard linear algebra).

So far, we have random parameters and we ignore the output. But what if we want to train the neuron so that the output mimics a real function or process? The next subsection tackles this very problem.

<!--

## Optimization with Batch Gradient Descent

You may have noticed that in the previous code listing I also introduced a specific activation function (aka squashing function) called `sigmoid` (aka the logistic function). In this section we'

Do this first

**Deeper dive:** TODO: something on activation functions.

https://nbviewer.jupyter.org/gist/joshfp/85d96f07aaa5f4d2c9eb47956ccdcc88/lesson2-sgd-in-action.ipynb

## Input Normalization

I provided *reasonable* ranges for values in the previous code example. For example, temperature values on Earth are typically in the range $[-20, 40]$ °C and illuminance in the range $[0, 1e6]$ Lux.

An NN can work with with values in these ranges, but it makes learning easier when you first scale values into the same range, typically $[-1, 1]$. TODO: why?




## Parameter Initialization

TODO: why can we start b at 0 by not \mathbf{w}?

## The Role of an Activation Function

- hidden neurons
    + default to relu
    + try/create others to solve/investigate specific issues
- output neurons
    + default to sigmoid for binary classification
    + default to softmax for multi-class classification
    + default to no activation for regression

## Vectorization with PyTorch

-->
