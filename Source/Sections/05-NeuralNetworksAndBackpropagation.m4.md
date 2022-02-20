# Neural Networks and Backpropagation

> Once your computer is pretending to be a neural net, you get it to be able to do a particular task by just showing it a whole lot of examples.
>
> -- Geoffrey Hinton


Below is our first neural network (aka multi-layer perceptron). We'll start by using this diagram to formulate terminology and conventions.


![A two-layer neural network.](img/2LayerNetwork.svg)


Notation:

- Layer 0 is the input (we called this $X$ for a single Neuron)
- Square bracket superscripts denote the network layer
- Round parenthesis superscripts denote the example index
- $w$ parameter subscripts denote first the associated neuron in the current layer and second the associated neuron (or input) from the previous layer
- $b$, $z$, and $a$ subscripts denote an associated neuron

Notice how we have all the same components as we did for the single neuron. We've just added additional notation to distinguish among layers and neurons in the same layer.


m4question([[Given some hypothetical deep neural network, how would you denote the linear computation of the third neuron in the fifth layer for training example 6123?]], [[$$z_3^{[5](6123)}$$

- "$z$": linear computation
- "$[5]$" superscript: fifth layer
- "$(6123)$" superscript: example 6123
- "$3$" subscript: third neuron

]])


## Vectorized Equations For a Neural Network

Let's start with showing the notation for parameters from any layer $l = 1, 2, ..., L$ where $L$ is the number of layers in the network.

\begin{align}
W^{[l]} &= m4matrix([["w_{row,col}^{[l]}"]], "n_l", "n_{l-1}") \\
\mathbf{b}^{[l]} &= m4colvec("b_{row}^{[l]}", "n_l")\end{align}

Compare these equations to the diagram above. Notice how the top neuron in layer 1 would have its associated parameters in the first row of $W^{[1]}$ and the first value in $\mathbf{b}^{[1]}$.

Next we have the vectorized linear and activation equations for each neuron in a layer (these are for all training examples):


\begin{align}
Z^{[l]} &= A^{[l-1]} W^{[l]T} + \mathbf{1} \mathbf{b}^{[l]T}\\
A^{[l]} &= g^{[l]}(Z^{[l]})
\end{align}


m4question([[Why do we have $\mathbf{1} \mathbf{b}^{[l]T}$?]], [[This ensures that the dimensions are correct between the added matrices. Try this out in Python:
```python
import torch
N, nl = 10, 4
b = torch.randn(nl, 1)
ONE = torch.ones(N, 1)
print(ONE @ b.T)
```

Note that most neural network frameworks handle this for you in the form of [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html).
]])




m4question([[What is the shape of $Z^{[l]}$?]], [[$Z^{[l]}$ is $(N \times n_l)$.
$$Z^{[l]} = m4matrix("z_{col}^{[l](row)}", "N", "n_l")$$

We compute this matrix by multiplying a $(N \times n_{l-1})$ matrix by a $(n_{l-1}, n_l)$ matrix (the transposed parameter matrix) and adding an $(N \times n_l)$ matrix.
]])




m4question([[What is the shape of $A^{[l]}$?]], [[$A^{[l]}$ is $(N \times n_l)$.
\begin{align}
A^{[l]} &= m4matrix("a_{col}^{[l](row)}", "N", "n_l") \\
\\
&= m4matrix("g_{col}^{[l]}(z_{col}^{[l](row)})", "N", "n_l") \\
\\
&= m4matrix([["g_{col}^{[l]}(\mathbf{a}^{[l-1](row)} \mathbf{w}_{col}^{[l]T} + b_{col}^{[l]})"]], "N", "n_l")\end{align}

You should also think about the shapes of $\mathbf{a}^{[l-1](i)}$ and $\mathbf{w}_{j}^{[l]}$.
]])


## Backpropagation

Just like for the single neuron, we want to find values for $W^{[l]}$ and $\mathbf{b}^{[l]}$ (for $l = 1, 2, ..., L$) such that $A^{[L]} \approx Y$ ($A^{[L]}$ is another name for $\hat Y$). Instead of looking at a more general case, let's work through gradient descent for the two-layer network above where we

- have three inputs ($n_x=n_0=3$),
- have two neurons in layer 1 ($n_1=2$),
- have three neurons in layer 2 ($n_y=n_2=3$),
- are using sigmoid activations for all neurons, and
- are using the binary-cross-entropy (BCE) loss function.

<!-- TODO: multi-label in terms. -->

You can imagine that we are performing multi-label classification. For this network, we need to compute these partial derivatives:

$$
\frac{∂ℒ}{∂ W^{[1]}}^①,
\frac{∂ℒ}{∂ \mathbf{b}^{[1]}}^②,
\frac{∂ℒ}{∂ W^{[2]}}^③,
\frac{∂ℒ}{∂ \mathbf{b}^{[2]}}^④
$$

We are going to start at layer 2 and work backward through the network to layer 1. As we compute these derivatives answer for yourself "why do we work backward through the network?"

This process of computing derivatives backward through the network is why this process if referred to as backpropagation--we'll compute values and propagate them backward to earlier layers in the network. This is easier to see when viewing the compute graph.


![Compute graph for two-layer network.](img/ComputeGraph.svg)


Notice how the input flows forward from top-to-bottom, but gradients flow backward (from bottom-to-top). This image corresponds to the network above if you rotate it 90 degrees anti-clockwise. Let's start with the term labeled ④ above. The chain-rule requires us to derive three components.

$$
\frac{∂ ℒ}{∂ \mathbf{b}^{[2]}}^④ = 
	\frac{∂ ℒ}{∂ A^{[2]}}
	\frac{∂ A^{[2]}}{∂ Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ \mathbf{b}^{[2]}}
$$

Here are the derivations.

\begin{align}
\frac{∂ ℒ}{∂ A^{[2]}} &=
	\frac{-∂}{∂ A^{[2]}} ||Y \cdot \log{A^{[2]}} + (1 - Y) \cdot \log{(1 - A^{[2]})}||_1\\
	&= \frac{1-Y}{1-A^{[2]}} - \frac{Y}{A^{[2]}}\\[20pt]

\frac{∂ A^{[2]}}{∂ Z^{[2]}} &=
	\frac{∂}{∂ Z^{[2]}} σ(Z^{[2]})\\
	&= σ(Z^{[2]})(1 - σ(Z^{[2]}))\\
	&= A^{[2]}(1 - A^{[2]})\\[20pt]

\frac{∂ Z^{[2]}}{∂ \mathbf{b}^{[2]}} &=
	\frac{∂}{∂ \mathbf{b}^{[2]}} A^{[1]} W^{[2]T} + \mathbf{1} \mathbf{b}^{[2]T}\\
	&= \mathbf{1}
\end{align}

Substituting back in.

\begin{align}
\frac{∂ ℒ}{∂ \mathbf{b}^{[2]}}^④ &=
	\left(\frac{1-Y}{1-A^{[2]}} - \frac{Y}{A^{[2]}}\right)
	A^{[2]}(1 - A^{[2]})
	\mathbf{1}\\
&= (1-Y)A^{[2]} - Y(1 - A^{[2]})\\
&= A^{[2]} - YA^{[2]} - Y + YA^{[2]}\\
&= A^{[2]} - Y
\end{align}


Similarly, for ③ we have:

$$
\frac{∂ ℒ}{∂ W^{[2]}}^③ = 
	\frac{∂ ℒ}{∂ A^{[2]}}
	\frac{∂ A^{[2]}}{∂ Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ W^{[2]}}
$$

The first two terms are identical to our previous derivation. In fact, we'll see these again for the first layer; so, it makes sense to give them their own symbol, $∂_{Z^{[2]}}=\frac{∂ ℒ}{∂ A^{[2]}}\frac{∂ A^{[2]}}{∂ Z^{[2]}}$. We do still need to derive the third term.

\begin{align}
\frac{∂ Z^{[2]}}{∂ W^{[2]}} &=
	\frac{∂}{∂ W^{[2]}} A^{[1]} W^{[2]T} + \mathbf{1} \mathbf{b}^{[2]T}\\ 
	&= A^{[1]}
\end{align}

TODO:

- dw2, db2 (using dz)
- dw1, db1
- color code?
- norm instead of sum


\begin{align}
\frac{∂ ℒ}{∂ \mathbf{b}^{[1]}}^② &= 
	\frac{∂ ℒ}{∂ A^{[2]}}
	\frac{∂ A^{[2]}}{∂ Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ A^{[1]}}
	\frac{∂ A^{[1]}}{∂ Z^{[1]}}
	\frac{∂ Z^{[1]}}{∂ \mathbf{b}^{[1]}}\\
&= ∂_{Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ A^{[1]}}
	\frac{∂ A^{[1]}}{∂ Z^{[1]}}
	\frac{∂ Z^{[1]}}{∂ \mathbf{b}^{[1]}}
\end{align}


\begin{align}
\frac{∂ Z^{[2]}}{∂ A^{[1]}} &= 
	\frac{∂}{A^{[1]}} A^{[1]} W^{[2]T} + \mathbf{1} \mathbf{b}^{[2]T}\\
	&= W^{[2]}\\[20pt]

\frac{∂ A^{[1]}}{∂ Z^{[1]}} &= \frac{∂}{Z^{[1]}}\\
	&= σ(Z^{[1]})\\
	&= σ(Z^{[1]})(1-σ(Z^{[1]})\\
	&= A^{[1]}(1-A^{[1]})\\[20pt]

\frac{∂ Z^{[1]}}{∂ \mathbf{b}^{[1]}} &= 
	\frac{∂}{\mathbf{b}^{[1]}} A^{[0]} W^{[1]T} + \mathbf{1} \mathbf{b}^{[1]T}\\
	&= \mathbf{1}\\[20pt]
\end{align}



\begin{align}
\frac{∂ ℒ}{∂ W^{[1]}}^② &= 
	\frac{∂ ℒ}{∂ A^{[2]}}
	\frac{∂ A^{[2]}}{∂ Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ A^{[1]}}
	\frac{∂ A^{[1]}}{∂ Z^{[1]}}
	\frac{∂ Z^{[1]}}{∂ W^{[1]}}\\
&= ∂_{Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ A^{[1]}}
	\frac{∂ A^{[1]}}{∂ Z^{[1]}}
	\frac{∂ Z^{[1]}}{∂ W^{[1]}}
\end{align}


Blah blah is $∂_{Z^{[1]}}=∂_{Z^{[2]}}\frac{∂ Z^{[2]}}{∂ A^{[1]}}\frac{∂ A^{[1]}}{∂ Z^{[1]}}$

\begin{align}
\frac{∂ Z^{[1]}}{∂ W^{[1]}} &=
	\frac{∂}{∂ W^{[1]}} A^{[0]} W^{[1]T} + \mathbf{1} \mathbf{b}^{[1]T}\\ 
	&= A^{[0]}
\end{align}


We can now write our update equations for all network parameters.

\begin{align}
W^{[1]} &:= 
	W^{[1]} - α\frac{∂ℒ}{∂ W^{[1]}} \\
	&:=\\[20pt]
\mathbf{b}^{[1]} &:= 
	\mathbf{b}^{[1]} - α\frac{∂ℒ}{∂ \mathbf{b}^{[1]}} \\
	&:=\\[20pt]
W^{[2]} &:= 
	W^{[2]} - α\frac{∂ℒ}{∂ W^{[2]}} \\
	&:=\\[20pt]
\mathbf{b}^{[2]} &:= 
	\mathbf{b}^{[2]} - α\frac{∂ℒ}{∂ \mathbf{b}^{[2]}} \\
	&:=\\
\end{align}


## Neuron Batch Gradient Descent


m4code([[Source/Code/Python/05-01-TwoLayerNeuralNetwork.py]])


## Automatic Differentiation

<!-- Autodiff
- symbolic (apply a sequence of of rules)
- [Numerical differentiation - Wikipedia](https://en.wikipedia.org/wiki/Numerical_differentiation "Numerical differentiation - Wikipedia")
- [Computer algebra - Wikipedia](https://en.wikipedia.org/wiki/Computer_algebra "Computer algebra - Wikipedia")

- dual numbers
- forward mode
- reverse mode
- Wengert list
 -->

## Why "Deep" Neural Networks?

<!-- - Universal approximation theorem -->

## The Role of an Activation Function

<!-- - what if we remove activation functions? -> linear model only
- hidden neurons
    + default to relu
    + try/create others to solve/investigate specific issues
- output neurons
    + default to sigmoid for binary classification
    + default to softmax for multi-class classification
    + default to no activation for regression -->


