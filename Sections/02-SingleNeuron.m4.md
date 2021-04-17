# Single Neuron

When our model is a single neuron we can only produce a single output. So, $n_y=1$ for this section.

## Notation and Diagram

A diagram representing a single neuron (as we'll see later, a neural network often refers to many of these neurons interconnected):

<!-- TODO: use m4 here -->
esyscmd(dot -Tsvg Diagrams/SingleNeuron.dot | tail -n +4)

<script type="text/javascript" src="js/main.js"></script>
<link rel="stylesheet" href="css/main.css">

The diagram represents the following equations:

\begin{align}
z^{(i)} &= \sum_{k=1}^{n_x} w_k x_k^{(i)} + b\\
a^{(i)} &= g(z^{(i)})
\end{align}

The main points of this equation:

- $x_k^{(i)}$ are the input features for the $i^{th}$ example (e.g., temperature)
- $z^{(i)}$ is a linear combination of the input features
- $w_k$ (weights) and $b$ (bias) are the **learned** parameters (notice the lack of any superscript)
- $a^{(i)}$ is the output of a non-linear activation function $g(\mathord{\cdot})$ applied to $z^{(i)}$
- $\hat y^{(i)}$ is the label we often give to the output ($a^{(i)} = \hat y^{(i)}$)

**For this model, we want to find parameters $w_k$ and $b$ such that the neuron outputs $\hat y^{(i)} \approx y$ for any input.** Before we discuss optimization we should take a moment to code up this single neuron model.

## Code for Computing a Neuron's Output

This code does not containing any "learning" (i.e., optimization), but it is worth showing just how simple it is to write a single neuron from scratch. Nearly all code is used to create random input data.

~~~python
include(Code/Python/SingleNeuronLoop1.py)
~~~

In the example, we have random parameters and we ignore the output. But what if we want to train the neuron so that the output mimics a real function or process? The next subsection tackles this very problem.

## Optimization with Batch Gradient Descent

You may have noticed that in the previous code listing I also introduced a specific activation function (aka squashing function) called `sigmoid` (aka the logistic function). In this section we'

Do this first

**Deeper dive:** TODO: something on activation functions.

## Input Normalization

I provided *reasonable* ranges for values in the previous code example. For example, temperature values on Earth are typically in the range $[-20, 40]$ °C and illuminance in the range $[0, 1e6]$ Lux.

An NN can work with with values in these ranges, but it makes learning easier when you first scale values into the same range, typically $[-1, 1]$. TODO: why?

<div class="sourceCode">
<pre>
gendiff(Code/Python/SingleNeuronLoop1.py Code/Python/SingleNeuronLoop2.py)
</pre>
</div>

## Parameter Initialization

<!-- TODO: why can we start b at 0 by not w? -->

## The Role of an Activation Function

- hidden neurons
    + default to relu
    + try/create others to solve/investigate specific issues
- output neurons
    + default to sigmoid for binary classification
    + default to softmax for multi-class classification
    + default to no activation for regression

## Vectorization with PyTorch
