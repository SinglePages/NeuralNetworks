# Single Neuron

> A single neuron in the brain is an incredibly complex machine that even today we don’t understand. A single “neuron” in a neural network is an incredibly simple mathematical function that captures a minuscule fraction of the complexity of a biological neuron.
>
> -- [Andrew Ng](https://www.wired.com/2015/02/google-brains-co-inventor-tells-why-hes-building-chinese-neural-networks/)


When our model is a single neuron we can only produce a single output. So, $n_y=1$ for this section. Sticking to our MNSIT digits example from above, we could train a single neuron to distinguish between two different classes of digits (e.g., "1" vs "7", "0" vs "non-zero", etc.).

<!--
TODO: 

m4aside: perceptron, regression
 -->

## Notation and Diagram

Here is a diagram representing a single neuron (as we'll see later, some neural networks are just many of these neurons interconnected):

![A neuron model with separate nodes for linear and activation computations.](img/NeuronSeparate.svg)

The diagram represents the following equations (note that I removed the parenthesis superscript from the diagram to make it a bit easier to read):

\begin{align}
z\i &= \sum_{k=1}^{n_x} x_k\i w_k + b\\
a\i &= g(z\i)
\end{align}

For these two equations:

- $x_k\i$ are the input features for the $i^{th}$ example (e.g., $k=76$ and $i=7436$ would denote pixel 76 of 784 for image 7436 of 60000)
- $w_k$ (weights) and $b$ (bias) are the **learned** parameters
- $z\i$ is a weighted sum of the input features plus the additional bias term
- $a\i$ is the output of a non-linear activation function $g(\mathord{\cdot})$ applied to $z\i$
- $\yhat\i$ (pronounced *"y hat"*) is the label we often give to the output ($a\i = \yhat\i$)


m4question([[Why do $w_k$ and $b$ not have superscripts?]], [[The parameters $w_k$ and $b$ do not change as the input $x_k\i$ changes. These parameters **are** the neuron, and they are used to produce the output $\yhat\i$ for any given input; we use the same parameter values regardless of input.]])


**For this model, we want to find parameters $w_k$ and $b$ such that the neuron outputs $\yhat\i \approx y\i$ for any input.** Before we discuss optimization we should take a moment to code up this single neuron model.

(Below is a more common representation of a neuron model. The image above separates the linear and activation components into distinct nodes, but it is more common to show them together as below.)

![A neuron model.](img/Neuron.svg)

## Neuron with Python Standard Libraries

This code does **not** include any "learning" (i.e., optimization), but it is worth showing just how simple it is to write a single neuron from scratch. Most of the code below is necessary only to create some faked input data.


m4code(Source/Code/Python/04-01-NeuronLoop.py)


In this code listing I use the `sigmoid` activation function (when not using a specific activation function we use $g(\mathord{\cdot})$ in most equations). This function is plotted below.


![Sigmoid activation function and its derivative.](img/Sigmoid.png)


Some nice properties of this function include:

- An output range of [0, 1] (all inputs are "squashed" into this range).
- An easy to compute derivative.
- Easy to interpret and understand.
- Well-known.

We often use sigmoid activation functions for binary classification (i.e., models trained to predict whether an input belongs to one of two classes). If the output is $≤0.5$ we say the neuron predicts class $A$ otherwise class $B$.


m4question([[Can you think of any downsides for this function (hint: look at the derivative curve)?]], [[While this function was once widely used, it has fallen out of favor because it can often lead to slower learning due to small derivative values for any input $z$ outside of the range [-4, 4]. [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is a more commonly used activation function for hidden layer neurons.]])


## The Dot-Product

We compute $z\i$ using a summation, but we can express this same bit of math using the dot-product from linear algebra.


$$
z\i = \sum_{k=1}^{n_x} x_k\i w_k + b = \vx^{(i)T} \vw + b
$$


The $\vx^{(i)T} \vw$ part of the equation computes the dot-product between $\vx^{(i)T}$ and $\vw$. We need to transpose $\vx\i$ to make the dimensions work (i.e., we need to multiply a row vector by a column vector).

This not only turns out to be easier to write/type, but it is more efficiently computed by a neural network library. The code listing below uses [PyTorch](https://pytorch.org/) to compute $z\i$ (`zi`). Libraries like PyTorch and Tensorflow make use of both vectorized CPU instructions and graphics cards (GPUs) to quickly compute the output of matrix multiplications.


m4diff([[Source/Code/Python/04-01-NeuronLoop.py]], [[Source/Code/Python/04-02-NeuronDot.py]])


The code snippet above shows a [diff](https://en.wikipedia.org/wiki/Diff) between the previous code snippet and an updated one using the dot product. You will see many diffs throughout this document. The key points are that: (1) red indicates text or entire lines that have been removed and (2) green indicates updated or newly added lines.

We do not need to transpose `xi` in code because when we iteration through `X` we get row vectors. As it happens, we can improve efficiency even further.

## Vectorizing Inputs

In addition to using a dot-product in place of a summation, we can use a matrix multiplication in place of looping over all examples in the dataset. In the two equations below we perform a matrix multiplication that computes the output of the network for all examples at once. A neural network library can turn this into highly efficient CPU or GPU operations.


\begin{align}
\vz &= X \vw + \mathbf{1} b \\
\va &= g(\vz)
\end{align}


m4diff([[Source/Code/Python/04-02-NeuronDot.py]], [[Source/Code/Python/04-03-NeuronVectorized.py]])


m4question([[What are the dimensions of $\vz$ and $\va$ (aka, $\vyhat$)?]], [[We are computing a single output value for each input, so, the shape of these vectors are $(N \times 1)$. PyTorch will treat these as arrays with $N$ elements instead of as column vectors.
\begin{align}
\vz &= m4colvec("\vx^{(row)T} \vw + b", "N") \\
\va &= m4colvec("g(z^{(row)})", "N")
\end{align}
]])


In the code snippet above, a matrix multiplication is indicated in PyTorch using the `@` symbol (a `*` is used for element-wise multiplications). A key to understanding matrix math is to examine the shapes of all matrices involved. Above, $X$ has a shape of $(N \times n_x)$, $\vw$ has a shape of $(n_x \times 1)$, and $b$ is a scalar multiplied by an appropriately-shaped matrix of all ones (so that we can add $b$ to each element of the $X\vw$ result). Inner dimensions (the last dimension of the left matrix and the first dimension of the right matrix) must be the same for any valid matrix multiplication.

In the code snippet, the scalar $b$ is added element-wise to every element in the final matrix due to [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) (this is a common library feature, not necessarily standard linear algebra).

So far, we have random parameters and we ignore the output. But what if we want to train the neuron so that the output mimics a real function or process? The next subsection tackles this very problem.

## Optimization with Batch Gradient Descent

We must find values for parameters $\vw$ and $b$ to make $\yhat\i \approx y\i$. As you might expect from the title of this subsection, we are going to use gradient descent to optimize the parameters. This means that we are going to need an objective function (something to minimize) and to compute some derivatives.

But what is an appropriate objective function (I'll refer to this as the *loss* function going forward)? How about the **mean-difference**?

$$ℒ(\vyhat, \vy) = \sum_{i=1}^N (\yhat\i - y\i) \quad \color{red}{\text{Don't use this loss function.}}$$

m4question([[What is problematic about this loss function?]], [[

Let's start by looking at the output of the function for different values of the inputs.

 $\yhat\i$           $y\i$      ℒ
---------------- ----------- -------
   0.1                 0        0.1
   0.1                 1       -0.9
   0.9                 0        0.9
   0.9                 1       -0.1

The table indicates that loss can be positive or negative. But how should we interpret negative loss? We see that $ℒ$ is minimized in row 2 of the table, but this is not an ideal result. The sign of loss is not helpful---as we'll see shortly, we will use the sign of the derivative.]])

A quick "fix" for the above loss function is to change it into the **mean-absolute-error** (MAE):

\begin{align}
ℒ(\vyhat, \vy) &= \sum_{i=1}^N |\yhat\i - y\i|\\
  &= \mae
\end{align}

The second line shows a vectorized version using the L1-norm, which is the sum of the absolute values of the given vector. MAE is a good choice if your dataset includes outliers. MAE is also simple to interpret: it is the average deviation between your models guess and the correct answer.

A common choice for a loss function when training a regression model is **Half mean-square-error** (Half-MSE):

\begin{align}
ℒ(\vyhat, \vy) &= \frac{1}{2N} \sum_{i=1}^N (\yhat\i - y\i)^2\\
  &= \vhmse
\end{align}

We are again using the L1-norm, but this time the vector we are norming is the element-wise squared values of the difference between the vectors $\vyhat$ and $\vy$. Interpreting Half-MSE is a bit harder than MAE---you should multiply the result by two and then take the square-root.

m4question([[Why might we compute the half-MSE instead of MSE or sum-square-error (SSE)?]], [[The \frac{1}{2} factor cancels out when we take the derivative. This scaling factor is unimportant since we will later multiply it by a learning rate, and can use that to achieve whatever effect we want.]])


The standard choice when performing classification with a neuron is **binary cross-entropy** (BCE):

\begin{align}
ℒ(\vyhat, \vy) &= \vbce\\
  &= -\text{mean}_0\left(\vy \cdot \log{\vyhat} + (1 - \vy) \cdot \log{(1 - \vyhat)}\right)
\end{align}


In the vectorized version of BCE, I've used the non-standard $\text{mean}_0$ notation to indicate that we're taking the average across the rows, dimension zero. This is closer to the code that you'll actually write. Very rarely will you want to put a summation loop in your code.


m4question([[Take some time to examine this loss function. What happens for various values of $\yhat\i$, $y\i$?]], [[

 $\yhat\i$           $y\i$      $\log{\yhat\i}$        $\log{(1-\yhat\i)}$        ℒ
---------------- ----------- ---------------------- -------------------------- -------
   0.1                 0           -2.3                   -0.1                   0.1
   0.1                 1           -2.3                   -0.1                   2.3
   0.9                 0           -0.1                   -2.3                   2.3
   0.9                 1           -0.1                   -2.3                   0.1

The tables shows that a larger difference between $\yhat\i$ and $y\i$ (rows 2 and 3) results in a larger loss, which is exactly what we'd like to see.
]])


Let's move forward using binary cross-entropy loss and the sigmoid activation function.

We can only reduce loss by adjusting parameters. It doesn't make sense, for example, to minimize loss by changing the input values $X$ or the output targets $Y$. Take a look at the following fictitious loss landscape.


![The effect on loss $ℒ$ of adjusting parameter $w_k$.](img/LossLandscape.svg)

The diagram above shows a curve for loss as a function of a single parameter, $w_k$. For this figure, we'll momentarily ignore that we might have dozens (or thousands or millions) of parameters. We want to find a new value for $w_k$ such that loss is reduced. You might wonder why I said "loss is reduced" instead of "loss is minimized." You might be familiar with techniques for finding an **exact** answer using an analytical (aka closed-form) solution.

m4question([[What should we do if we wanted to **minimize** loss with respect to the parameter using an analytical solution?]], [[We should take the derivative, set it equal to zero, and then solve the set of linear equations. Here is an example using linear regression, which is very similar to our single neuron. Here is our model:

$$\vyhat = X \mathbf{θ},$$

where $θ$ is our vector of parameters. Here is our loss function (half-SSE):

$$ℒ(\vyhat, \vy) = \frac{1}{2} ||(\vyhat - \vy)^2||_1.$$

Now we can take the partial derivative of loss with respect to parameters $θ$. (Note that I substitute for $\vyhat$ on the third line.)

\begin{align}
\frac{∂ ℒ}{∂ \mathbf{θ}} &=
  \frac{∂ ||\frac{1}{2} (\vyhat - \vy)^2||_1}{∂ \mathbf{θ}} \\
&= ||\vyhat - \vy||_1 \frac{∂ \vyhat}{∂ \mathbf{θ}} \\
&= ||X \mathbf{θ} - \vy||_1 \frac{∂ X \mathbf{θ}}{∂ \mathbf{θ}} \\
&= ||X \mathbf{θ} - \vy||_1 X \\
&= X^T X \mathbf{θ} - X^T \vy
\end{align}

We can now set this derivative to zero and solve for $\mathbf{θ}$.

\begin{align}
\frac{∂ ℒ}{∂ \mathbf{θ}} &= 0 \\
X^T X \mathbf{θ} - X^T \vy &= 0 \\
X^T X \mathbf{θ} &= X^T \vy
\end{align}

And now assuming that $X^T X$ is invertible (that the columns are linearly independent).

$$\mathbf{θ}^* = (X^T X)^{-1} X^T \vy$$

We now have an optimal solution (called $\mathbf{θ}^*$) that minimizes loss. (See [Ordinary least squares - Wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares "Ordinary least squares - Wikipedia") for more details.)
]])

For complex models, such as a neural network, analytical solutions are sometimes too slow or complicated to compute. Instead, we use an iterative (aka numerical) solution. You can think of numerical solutions as finding a good enough approximate solution as opposed to the exact correct solution. Surprisingly, the numerical solution is often more general than the exact solution---we'll discuss this in later sections.

To determine **how** we should adjust parameters, we start the same way as finding the exact location and take the partial derivative of loss with respect to each parameter. Taking the single neuron, binary cross-entropy loss, and the sigmoid activation function the chain rule in matrix form is as follows.

\begin{align}
\frac{∂ ℒ}{∂ \vw} &=
  \frac{∂ ℒ}{∂ \vyhat}
  \frac{∂ \vyhat}{∂ \vz}
  \frac{∂ \vz}{∂ \vw} \\
  &= \frac{1}{N} X^T (\vyhat - \vy)\\[20pt]
\frac{∂ ℒ}{∂ b} &=
  \frac{∂ ℒ}{∂ \vyhat}
  \frac{∂ \vyhat}{∂ \vz}
  \frac{∂ \vz}{∂ b} \\
  &= \frac{1}{N} \sum_{i=1}^N (\yhat\i - y\i)
\end{align}

Don't worry too much about the derivations or notation at this stage. We'll go into more details when we follow the same process for a full neural network in the next section.

m4question([[Why is it necessary to apply the chain rule? And why did the chain rule appear as it does above?]], [[First, we cannot directly compute the partial derivative of $ℒ$ with respect to $\vw$ (or $b$). Second, we only apply the chain rule to equations that have some form of dependency on the term in the first denominator ($\vw$ and $b$). It is useful to look at the loss function when we substitute in values for $\yhat$ and then $z$.

$$ℒ(\vyhat, \vy) = -\frac{1}{N}\sum_{i=1}^N (y\i \log{σ(\vx^{(i)T}\vw\i + b)} + (1 - y\i)\log{(1-σ(\vx^{(i)T}\vw\i + b))})$$

In the above equation we can more easily see how the chain-rule comes into play. The parameter $\vw$ is nested within a call to $σ$ which is nested within a call to $\log$ when computing $\frac{∂ ℒ}{∂ \vw}$.
]])


m4question([[What do we do with the partial derivatives $\frac{∂ ℒ}{∂ \vw}$ and $\frac{∂ ℒ}{∂ b}$?]], [[We use these terms to update model parameters.

\begin{align}
\vw &:= \vw - η \frac{∂ ℒ}{∂ \vw} \\
b &:= b - η \frac{∂ ℒ}{∂ b}
\end{align}

]])


m4question([[What is the derivative of the sigmoid function, $σ$?]], [[

\begin{align}
\frac{d}{dz} σ(z) &= 
  \frac{d}{dz} \left(\frac{1}{1 + e^{-z}}\right) \\
&= \frac{d}{dz} \left(1 + e^{-z} \right)^{-1} \\
&= -(1 + e^{-z})^{-2}(-e^{-z}) \\
&= \frac{e^{-z}}{\left(1 + e^{-z}\right)^2} \\
&= \frac{1}{1 + e^{-z}\ } \cdot \frac{e^{-z}}{1 + e^{-z}}  \\
&= \frac{1}{1 + e^{-z}\ } \cdot \frac{e^{-z} + 1 - 1}{1 + e^{-z}}  \\
&= \frac{1}{1 + e^{-z}\ } \cdot \left( \frac{1 + e^{-z}}{1 + e^{-z}} - \frac{1}{1 + e^{-z}} \right) \\
&= \frac{1}{1 + e^{-z}\ } \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) \\
&= σ(z) \cdot (1 - σ(z))
\end{align}

]])


With the two update equations shown in the previous answer we have everything we need to train our neuron model. Looking at these two equations you might wonder about the purpose of $η$ (i.e., the "learning rate"). This factor enables us to tune how fast or slow we learn. If $η$ is set too high we might not be able to learn, and it it is set too low we might learn prohibitively slowly. We will go into more details on optimization in [@sec:opti].

## Neuron Batch Gradient Descent

Here is a complete example in which we train a neuron to classify images as either being of the digit 1 or the digit 7. Data processing details are hidden in the `get_binary_mnist_one_batch` function, but you can find that [code in the repository for this guide](https://github.com/SinglePages/NeuralNetworks/blob/767c4a3e357ba757b2e39767b489d7c51d1688c7/Source/Code/Python/utilities.py#L69).


m4code([[Source/Code/Python/04-04-NeuronMNIST.py]])


m4question([[Which lines of code correspond to $\frac{∂ ℒ}{∂ \vw}$ and $\frac{∂ ℒ}{∂ b}$?]], [[Lines 46 and 47.]])


m4question([[What is an epoch?]], [[It turns out that we might need to update our weights more than once to get useful results. Each time we update parameters based on all training examples we mark the end of an epoch. In the code above we iterate through four epochs.]])


m4question([[What do you expect to see for the output?]], [[

<pre class="code-block">
Accuracy before training: 0.54
 1/4, Loss=0.7, Accuracy=0.97, Time=5.5 ms
 2/4, Loss=0.5, Accuracy=0.96, Time=4.8 ms
 3/4, Loss=0.4, Accuracy=0.96, Time=4.6 ms
 4/4, Loss=0.3, Accuracy=0.96, Time=4.4 ms
</pre>
]])
