# Single Neuron

When our model is a single neuron we can only produce a single output. So, $n_y=1$ for this section. Sticking to our MNSIT digits example from above, we could train a single neuron to distinguish between two different classes (e.g., "1" vs "7", "0" vs "non-zero", etc.).

<!--
m4aside

perceptron, regression

 -->

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


**For this model, we want to find parameters $w_k$ and $b$ such that the neuron outputs $\hat y^{(i)} \approx y^{(i)}$ for any input.** Before we discuss optimization we should take a moment to code up this single neuron model.

Before we continue I should show a more common representation of a neuron model. The image above separates the linear and activation components, but it is more common to show them together in a single node.

![A neuron model.](img/Neuron.svg)

## Neuron with Python Standard Libraries

This code does not include any "learning" (i.e., optimization), but it is worth showing just how simple it is to write a single neuron from scratch. Most of the code below is necessary only to create some faked input data.


m4code(Code/Python/04-01-NeuronLoop.py)


In this code listing I use the `sigmoid` activation function (referred to as $g(\mathord{\cdot})$ in most equations.). This function is plotted below.


![Sigmoid activation function and its derivative.](img/Sigmoid.png)


Some nice properties of this function include:

- An output range of [0, 1] (all inputs are "squashed" into this range).
- An easy to compute derivative.
- Easy to interpret and understand.
- Well-known.

We often use sigmoid activation functions for binary classification (i.e., models trained to predict whether an input belongs to one of two classes). If the output is $≤0.5$ we say the neuron predicts class $A$ otherwise class $B$.


m4question([[Can you think of any downsides for this function (hint: look at the derivative curve)?]], [[While this function was once widely used, it has fallen out of favor because it can often lead to slower learning due to small derivative values for any input $z$ outside of the range [-4, 4]. [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is a more commonly used activation function for hidden layer neurons.]])


## The Dot-Product

We compute $z^{(i)}$ above using a summation, but we can express this same bit of math using the dot-product from linear algebra.


$$
z^{(i)} = \sum_{k=1}^{n_x} x_k^{(i)} w_k + b = \mathbf{x}^{(i)T} \mathbf{w} + b
$$


The $\mathbf{x}^{(i)T} \mathbf{w}$ part of the equation computes the dot-product between $\mathbf{x}^{(i)T}$ and $\mathbf{w}$. We need to transpose $\mathbf{x}^{(i)}$ to make the dimensions work (i.e., we need to multiply a row vector by a column vector).

This not only turns out to be easier to write/type, but it is more efficiently computed by a neural network library. The code listing below uses [PyTorch](https://pytorch.org/) to compute $z^{(i)}$ (`zi`). Libraries like PyTorch and Tensorflow make use of both vectorized CPU instructions and graphics cards (GPUs) to quickly compute the output of matrix multiplications.


m4diff([[Code/Python/04-01-NeuronLoop.py]], [[Code/Python/04-02-NeuronDot.py]])


The code snippet above shows a [diff](https://en.wikipedia.org/wiki/Diff) between the previous code snippet and an updated one using the dot product. You will see many diffs throughout this document. The key points are that: (1) red indicates text or entire lines that have been removed and (2) green indicates updated or newly added lines.

We do not need to transpose `xi` in code because when we iteration through `X` we get row vectors. As it happens, we can improve efficiency even further.

## Vectorizing Inputs

In addition to using a dot-product in place of a summation, we can use a matrix multiplication in place of looping over all examples in the dataset. In the two equations below we perform a matrix multiplication that computes the output of the network for all examples at once. A neural network library can turn this into highly efficient CPU or GPU operations.


\begin{align}
\mathbf{z} &= X \mathbf{w} + b \\
\mathbf{a} &= g(\mathbf{z})
\end{align}


m4diff([[Code/Python/04-02-NeuronDot.py]], [[Code/Python/04-03-NeuronVectorized.py]])


m4question([[What are the dimensions of $\mathbf{z}$ and $\mathbf{a}$ (aka, $\mathbf{\hat y}$)?]], [[We are computing a single output value for each input, so, the shape of these vectors are $(N \times 1)$. PyTorch will treat these as arrays with $N$ elements instead of as column vectors.
\begin{align}
\mathbf{z} &= m4colvec("\mathbf{x}^{(row)T} \mathbf{w} + b", "N") \\
\mathbf{a} &= m4colvec("g(z^{(row)})", "N")
\end{align}
]])


In the code snippet above, a matrix multiplication is indicated in PyTorch using the `@` symbol (a `*` is used for element-wise multiplications). A key to understanding matrix math is to examine the shapes of all matrices involved. Above, $X$ has a shape of $(N \times n_x)$, $\mathbf{w}$ has a shape of $(n_x \times 1)$, and $b$ is a scalar.

Inner dimensions (the last dimension of the left matrix and the first dimension of the right matrix) must be the same for any valid matrix multiplication. The scalar, $b$, is added element-wise to every element in the final matrix due to [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) (this is a common library feature, not necessarily standard linear algebra).

So far, we have random parameters and we ignore the output. But what if we want to train the neuron so that the output mimics a real function or process? The next subsection tackles this very problem.

## Optimization with Batch Gradient Descent

We must find values for parameters $\mathbf{w}$ and $b$ to make $\hat y^{(i)} \approx y^{(i)}$. As you might expect from the title of this subsection, we are going to use gradient descent to optimize the parameters. This means that we are going to need an objective function (something to minimize) and to compute some derivatives.

But what is an appropriate objective function (I'll refer to this as the *loss* function going forward)? How about the **mean-difference**?

$$ℒ(\hat{\mathbf{y}}, \mathbf{y}) = \sum_{i=1}^N \hat y^{(i)} - y^{(i)} \quad \color{red}{\text{Don't use this loss function.}}$$

m4question([[What is problematic about this loss function?]], [[

Let's start by looking at the output of the function for different values of the inputs.

 $\hat y^{(i)}$   $y^{(i)}$     ℒ
---------------- ----------- -------
   0.1                 0        0.1
   0.1                 1       -0.9
   0.9                 0        0.9
   0.9                 1       -0.1

The table indicates that loss can be positive or negative. But how should we interpret negative loss? The sign of loss is not helpful--as we'll see shortly, we will use the sign of the derivative.]])

A quick "fix" for the above loss function is to change it into the **mean-absolute-error** (MAE):

$$ℒ(\hat{\mathbf{y}}, \mathbf{y}) = \sum_{i=1}^N |\hat y^{(i)} - y^{(i)}| \quad \text{MAE works well with outliers.}$$

A common choice for a loss function when training a regression model is **Half mean-square-error** (Half-MSE):

$$ℒ(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{2N} \sum_{i=1}^N (\hat y^{(i)} - y^{(i)})^2$$


m4question([[Why might we compute the half-MSE instead of MSE?]], [[The \frac{1}{2} factor cancels out when we take the derivative. This scaling factor is unimportant since we will later multiply it by a learning rate.]])


The standard choice when performing classification with a neuron is **binary cross-entropy**:

$$ℒ(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{i=1}^N (y \log{\hat y^{(i)}} + (1 - y)\log{(1-\hat y^{(i)})})$$

m4question([[Take some time to examine this loss function. What happens for various values of $\hat y^{(i)}$, $y^{(i)}$?]], [[

 $\hat y^{(i)}$   $y^{(i)}$   $\log{\hat y^{(i)}}$   $\log{(1-\hat y^{(i)})}$     ℒ
---------------- ----------- ---------------------- -------------------------- -------
   0.1                 0           -2.3                   -0.1                   0.1
   0.1                 1           -2.3                   -0.1                   2.3
   0.9                 0           -0.1                   -2.3                   2.3
   0.9                 1           -0.1                   -2.3                   0.1

The tables shows that a larger difference between $\hat y^{(i)}$ and $y^{(i)}$ (rows 2 and 3) results in a larger loss, which is exactly what we'd like to see.
]])


Let's move forward using binary cross-entropy loss. We can only reduce loss by adjusting parameters. To determine **how** we should adjust parameters, we take the partial derivative of loss with respect to each parameter. We can do this using the chain rule in matrix form as follows:

\begin{align}
\frac{\partial ℒ}{\partial \mathbf{w}} &=
    \frac{\partial ℒ}{\partial \hat{\mathbf{y}}}
    \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}}
    \frac{\partial \mathbf{z}}{\partial \mathbf{w}} \\
&= \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y})X\\\\
\frac{\partial ℒ}{\partial b} &=
    \frac{\partial ℒ}{\partial \hat{\mathbf{y}}}
    \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}}
    \frac{\partial \mathbf{z}}{\partial b} \\
&= \frac{1}{N}\sum_{i=1}^N (\hat y^{(i)} - y^{(i)})
\end{align}

m4question([[Why is it necessary to apply the chain rule? And why did the chain rule appear as it does above?]], [[First, we cannot directly compute the partial derivative of $ℒ$ with respect to $\mathbf{w}$ (or $b$). Second, we only apply the chain rule to equations that have some form of dependency on the term in the first denominator ($\mathbf{w}$ and $b$).]])


m4question([[What do we do with the partial derivatives $\frac{\partial ℒ}{\partial \mathbf{w}}$ and $\frac{\partial ℒ}{\partial b}$?]], [[We use these terms to update parameters

\begin{align}
w &:= w - \alpha \frac{\partial ℒ}{\partial \mathbf{w}} \\
b &:= b - \alpha \frac{\partial ℒ}{\partial b}
\end{align}

]]).


With the two update equations shown in the previous answer we have everything we need to train our neuron model. Looking at these two equations you might wonder about the purpose of $\alpha$ (i.e., the "learning rate"). This factor enables us to tune how fast or slow we learn. If $\alpha$ is set too high we might not be able to learn, and it it is set too low we might learn prohibitively slowly.

We will go into more details on optimization in [@sec:opti].

## Neuron Batch Gradient Descent

m4code([[Code/Python/04-04-NeuronMNIST.py]])

m4question([[Which lines of code correspond to $\frac{\partial ℒ}{\partial \mathbf{w}}$ and $\frac{\partial ℒ}{\partial b}$?]], [[Lines 44 and 45.]])

m4question([[What is an epoch?]], [[It turns out that we might need to update our weights more than once to get useful results. Each time we update parameters based on all training examples we mark the end of an epoch. In the code above we iterate through four epochs.]])
