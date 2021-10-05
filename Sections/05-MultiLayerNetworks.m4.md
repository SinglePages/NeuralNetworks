# Multi-Layer Networks

*Being revised*

<!-- - mlp
- deep networks -->

![A two-layer neural network.](img/2LayerNetwork.svg)

<!-- - Layer 0 is the input (formerly X)
- Bracket superscript gives the layer
- Parentheses superscript gives the example index (removed for readability)
- Subscript gives neuron index, previous layer neuron index

Single neuron uses the same equations as above

Grouping neurons


\begin{align}
W^{[l]} &= m4matrix([["w_{row,col}^{[l]}"]], "n_l", "n_{l-1}") \\
\mathbf{b}^{[l]} &= m4colvec("b_{row}^{[l]}", "n_l")\end{align}


$$Z^{[l]} = A^{[l-1]} W^{[l]T}$$


m4question([[What is the shape of $Z^{[l]}$?]], [[$Z^{[l]}$ is $(N \times n_l)$.
$$Z^{[l]} = m4matrix("z_{col}^{[l](row)}", "N", "n_l")$$]])


$$A^{[l]} = g(Z^{[l]})$$


m4question([[What is the shape of $A^{[l]}$?]], [[$A^{[l]}$ is $(N \times n_l)$.
\begin{align}
A^{[l]} &= m4matrix("a_{col}^{[l](row)}", "N", "n_l") \\
\\
&= m4matrix("g(z_{col}^{[l](row)})", "N", "n_l") \\
\\
&= m4matrix([["g(\mathbf{a}^{[l-1](row)} \mathbf{w}_{col}^{[l]T} + b_{col}^{[l]})"]], "N", "n_l")\end{align}

You should also think about the shapes of $\mathbf{a}^{[l-1](i)}$ and $\mathbf{w}_{j}^{[l]}$.
]])
 -->

<!--

## Input Normalization

I provided *reasonable* ranges for values in the previous code example. For example, temperature values on Earth are typically in the range $[-20, 40]$ °C and illuminance in the range $[0, 1e6]$ Lux.


An NN can work with with values in these ranges, but it makes learning easier when you first scale values into the same range, typically $[-1, 1]$. TODO: why?


## Why "Deep" Neural Networks?

- Universal approximation theorem

## The Role of an Activation Function

- what if we remove activation functions? -> linear model only
- hidden neurons
    + default to relu
    + try/create others to solve/investigate specific issues
- output neurons
    + default to sigmoid for binary classification
    + default to softmax for multi-class classification
    + default to no activation for regression

## Parameter Initialization

TODO: why can we start b at 0 by not \mathbf{w}?

## Vanishing and Exploding Gradients

https://nbviewer.jupyter.org/gist/joshfp/85d96f07aaa5f4d2c9eb47956ccdcc88/lesson2-sgd-in-action.ipynb

-->
