# Single Neuron

## Notation and Diagram

When our model is a single neuron we can only produce a single output. So, $n_y=1$ for this section. We'll modify our reference accordingly and say that we only care about predicting the latitude and ignore longitude and elevation.

<!-- TODO: use m4 here -->
esyscmd(dot -Tsvg Diagrams/SingleNeuron.dot | tail -n +4)

<script type="text/javascript" src="js/main.js"></script>
<link rel="stylesheet" href="css/main.css">

The diagram represents the following equations:

\begin{align}
z^{(i)} &= \sum_{k=1}^{n_x} w_k x_k^{(i)} + b\\
a^{(i)} &= g(z^{(i)}).
\end{align}

The main points of this equation:

- $x_k^{(i)}$ are the input features for the $i^{th}$ example (e.g., temperature)
- $z^{(i)}$ is a linear combination of the input features
- $w_k$ (weights) and $b$ (bias) are the **learned** parameters (**no superscript**)
- $a^{(i)}$ is the output of a non-linear activation function $g(\mathord{\cdot})$ applied to $z^{(i)}$
- $\hat y^{(i)}$ is the label we often give to the output ($a^{(i)} = \hat y^{(i)}$)

**We want to find parameters $w_k$ and $b$ such that the neuron outputs $\hat y^{(i)} \approx y$ for any input.** Before we discuss optimization we should take a moment to code up this single neuron model.

## Simple Example

<!-- TODO: get real data for this (and show it later) -->

~~~python
include(Code/Python/SingleNeuronLoop.py)
~~~


## Full Example





Some text


Some text

<div class="diff-example"><pre>
gendiff(Code/Python/1.py, Code/Python/2.py)
</pre></div>
