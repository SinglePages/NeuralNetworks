# Neural Networks and Backpropagation

> Once your computer is pretending to be a neural net, you get it to be able to do a particular task by just showing it a whole lot of examples.
>
> -- Geoffrey Hinton


Below is our first neural network (aka multi-layer perceptron, MLP). We'll start by using this diagram to formulate terminology and conventions.


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
\vb^{[l]} &= m4colvec("b_{row}^{[l]}", "n_l")\end{align}

Compare these equations to the diagram above. Notice how the top neuron in layer 1 would have its associated parameters in the first row of $W^{[1]}$ and the first value in $\vb^{[1]}$.

Next we have the vectorized linear and activation equations for each neuron in a layer (these are for all training examples):


\begin{align}
Z^{[l]} &= A^{[l-1]} W^{[l]T} + \mathbf{1} \vb^{[l]T}\\
A^{[l]} &= g^{[l]}(Z^{[l]})
\end{align}


m4question([[Why do we have $\mathbf{1} \vb^{[l]T}$?]], [[This ensures that the dimensions are correct between the added matrices. Try this out in Python:
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
&= m4matrix([["g_{col}^{[l]}(\va^{[l-1](row)} \vw_{col}^{[l]T} + b_{col}^{[l]})"]], "N", "n_l")\end{align}

You should also think about the shapes of $\va^{[l-1](i)}$ and $\vw_{j}^{[l]}$.
]])


## Backpropagation

Just like for the single neuron, we want to find values for $W^{[l]}$ and $\vb^{[l]}$ (for $l = 1, 2, ..., L$) such that $A^{[L]} \approx Y$ ($A^{[L]}$ is another name for $\hat Y$). Instead of looking at a more general case, let's work through gradient descent for the two-layer network above where we

- have three inputs ($n_x=n_0=3$),
- have two neurons in layer 1 ($n_1=2$),
- have three neurons in layer 2 ($n_y=n_2=3$),
- are using sigmoid activations for all neurons, and
- are using the binary-cross-entropy (BCE) loss function.

You can imagine that we are performing multi-label classification. For this network, we need to compute these partial derivatives:

$$
\frac{∂ℒ}{∂ W^{[1]}}^①,
\frac{∂ℒ}{∂ \vb^{[1]}}^②,
\frac{∂ℒ}{∂ W^{[2]}}^③,
\frac{∂ℒ}{∂ \vb^{[2]}}^④
$$

We are going to start at layer 2 and work backward through the network to layer 1. As we compute these derivatives answer for yourself **"why do we work backward through the network?"**

This process of computing derivatives backward through the network is why this process if referred to as backpropagation--we'll compute values and propagate them backward to earlier layers in the network. This is easier to see when viewing the compute graph. A compute graph depicts the flow of activations (during the forward pass) and gradients (during the backward pass) through the network.


![Compute graph for two-layer network.](img/ComputeGraph.svg)


Notice how the input flows forward from top-to-bottom in the compute graph, but gradients flow backward (from bottom-to-top). This image corresponds to the network above if you rotate it 90 degrees anti-clockwise (mostly just so we I had space for the image on this page).

### Layer 2 Parameters

Let's start with the terms labeled ③ and ④ above, which correspond to layer 2. The chain-rule requires us to derive three components.

\begin{align}
\frac{∂ℒ}{∂ W^{[2]}}^③ &= 
	\textcolor{blue}{\frac{∂ ℒ}{∂ A^{[2]}}}
	\textcolor{green}{\frac{∂ A^{[2]}}{∂ Z^{[2]}}}
	\frac{∂ Z^{[2]}}{∂ W^{[2]}}\\
\frac{∂ ℒ}{∂ \vb^{[2]}}^④ &= 
	\textcolor{blue}{\frac{∂ ℒ}{∂ A^{[2]}}}
	\textcolor{green}{\frac{∂ A^{[2]}}{∂ Z^{[2]}}}
	\frac{∂ Z^{[2]}}{∂ \vb^{[2]}}
\end{align}

These equations share the first two terms. In fact, we'll see these again for the first layer; so, it makes sense to give them their own symbol, $∂_{Z^{[2]}}=\textcolor{blue}{{\frac{∂ ℒ}{∂ A^{[2]}}}}\textcolor{green}{\frac{∂ A^{[2]}}{∂ Z^{[2]}}}$. (You might notice that I am leaving out the $\text{mean}_0$ operation from BCE; this is intentional as it will be handled below using a matrix multiplication for one of the partial derivatives below.)

\begin{align}
\textcolor{blue}{\frac{∂ ℒ}{∂ A^{[2]}}} &=
	-\frac{∂}{∂ A^{[2]}} \left(Y \cdot \log{A^{[2]}} + (1 - Y) \cdot \log{\left(1 - A^{[2]}\right)}\right)\\
	&= \left( \frac{1-Y}{1-A^{[2]}} - \frac{Y}{A^{[2]}} \right)\\[20pt]

\textcolor{green}{\frac{∂ A^{[2]}}{∂ Z^{[2]}}} &=
	\frac{∂}{∂ Z^{[2]}} σ(Z^{[2]})\\
	&= σ(Z^{[2]}) \cdot (1 - σ(Z^{[2]}))\\
	&= A^{[2]} \cdot (1 - A^{[2]})
\end{align}

Now substituting to solve for $∂_{Z^{[2]}}$.

\begin{align}
∂_{Z^{[2]}} &= 
	\left(\frac{1-Y}{1-A^{[2]}} - \frac{Y}{A^{[2]}}\right) A^{[2]} \cdot (1 - A^{[2]})\\
	&= (1-Y) \cdot A^{[2]} - Y \cdot (1 - A^{[2]})\\
	&= A^{[2]} - Y \cdot A^{[2]} - Y + Y \cdot A^{[2]}\\
	&= A^{[2]} - Y
\end{align}


Next we can solve the third terms in equations ③ and ④.


\begin{align}
\frac{∂ Z^{[2]}}{∂ W^{[2]}} &=
	\frac{∂}{∂ W^{[2]}} A^{[1]} W^{[2]T} + \mathbf{1} \vb^{[2]T}\\
	&= A^{[1]}\\[20pt]
\frac{∂ Z^{[2]}}{∂ \vb^{[2]}} &=
	\frac{∂}{∂ \vb^{[2]}} A^{[1]} W^{[2]T} + \mathbf{1} \vb^{[2]T}\\
	&= \mathbf{1}
\end{align}


And that leaves us with the following partial derivatives for ③ and ④.


\begin{align}
\frac{∂ℒ}{∂ W^{[2]}}^③ &= \frac{1}{N} ∂_{Z^{[2]}}^T A^{[1]}\\
\frac{∂ ℒ}{∂ \vb^{[2]}}^④ &= \text{mean}_0 (∂_{\vz^{[2]}})
\end{align}


### Layer 1 Parameters

Now we can continue to layer 1 and derive equations for terms ① and ②.

\begin{align}
\frac{∂ ℒ}{∂ W^{[1]}}^① &= 
	\textcolor{blue}{\frac{∂ ℒ}{∂ A^{[2]}}}
	\textcolor{green}{\frac{∂ A^{[2]}}{∂ Z^{[2]}}}
	\textcolor{cyan}{\frac{∂ Z^{[2]}}{∂ A^{[1]}}}
	\textcolor{lime}{\frac{∂ A^{[1]}}{∂ Z^{[1]}}}
	\frac{∂ Z^{[1]}}{∂ W^{[1]}}\\
&= ∂_{Z^{[2]}}
	\frac{∂ Z^{[2]}}{∂ A^{[1]}}
	\frac{∂ A^{[1]}}{∂ Z^{[1]}}
	\frac{∂ Z^{[1]}}{∂ W^{[1]}}\\[20pt]

\frac{∂ ℒ}{∂ \vb^{[1]}}^② &= 
	\textcolor{blue}{\frac{∂ ℒ}{∂ A^{[2]}}}
	\textcolor{green}{\frac{∂ A^{[2]}}{∂ Z^{[2]}}}
	\textcolor{cyan}{\frac{∂ Z^{[2]}}{∂ A^{[1]}}}
	\textcolor{lime}{\frac{∂ A^{[1]}}{∂ Z^{[1]}}}
	\frac{∂ Z^{[1]}}{∂ \vb^{[1]}}\\
&= ∂_{Z^{[2]}}
	\textcolor{cyan}{\frac{∂ Z^{[2]}}{∂ A^{[1]}}}
	\textcolor{lime}{\frac{∂ A^{[1]}}{∂ Z^{[1]}}}
	\frac{∂ Z^{[1]}}{∂ \vb^{[1]}}
\end{align}


All four derivations share the first two terms in common, $∂_{Z^{[2]}}$. The first layer parameters additional share the next two terms. We'll group the first four terms together just like we did for layer 2: $∂_{Z^{[1]}}=∂_{Z^{[2]}}\textcolor{cyan}{\frac{∂ Z^{[2]}}{∂ A^{[1]}}}\textcolor{lime}{\frac{∂ A^{[1]}}{∂ Z^{[1]}}}$.

Let's start by deriving this shared term.

\begin{align}
\textcolor{cyan}{\frac{∂ Z^{[2]}}{∂ A^{[1]}}} &= 
	\frac{∂}{A^{[1]}} (A^{[1]} W^{[2]T} + \mathbf{1} \vb^{[2]T})\\
	&= W^{[2]}\\[20pt]

\textcolor{lime}{\frac{∂ A^{[1]}}{∂ Z^{[1]}}} &= \frac{∂}{Z^{[1]}}\\
	&= σ(Z^{[1]})\\
	&= σ(Z^{[1]}) \cdot (1 - σ(Z^{[1]})\\
	&= A^{[1]} \cdot (1-A^{[1]})
\end{align}

Now substituting to solve for $∂_{Z^{[1]}}$.

$$∂_{Z^{[1]}} = ∂_{Z^{[2]}} W^{[2]} \cdot A^{[1]} \cdot (1 - A^{[1]})$$

We still have one term remaining for each of ① and ②.

\begin{align}
\frac{∂ Z^{[1]}}{∂ W^{[1]}} &=
	\frac{∂}{∂ W^{[1]}} A^{[0]} W^{[1]T} + \mathbf{1} \vb^{[1]T}\\ 
	&= A^{[0]}\\[20pt]
\frac{∂ Z^{[1]}}{∂ \vb^{[1]}} &= 
	\frac{∂}{\vb^{[1]}} A^{[0]} W^{[1]T} + \mathbf{1} \vb^{[1]T}\\
	&= \mathbf{1}\\[20pt]
\end{align}


And that leaves us with the following partial derivatives for ① and ②.


\begin{align}
\frac{∂ℒ}{∂ W^{[1]}}^① &= \frac{1}{N} ∂_{Z^{[1]}}^T A^{[0]}\\
\frac{∂ ℒ}{∂ \vb^{[1]}}^② &= \text{mean}_0 (∂_{\vz^{[1](i)}})
\end{align}


### Parameter Update Equations

We can now write our update equations for all network parameters.

\begin{align}
W^{[1]} &:= W^{[1]} - η \frac{∂ℒ}{∂ W^{[1]}} \\
	&:= W^{[1]} - η \frac{1}{N} ∂_{Z^{[1]}}^T A^{[0]} \\[20pt]
\vb^{[1]} &:= \vb^{[1]} - η \frac{∂ℒ}{∂ \vb^{[1]}} \\
	&:= \vb^{[1]} - η\;\text{mean}_0 (∂_{\vz^{[1](i)}}) \\[20pt]
W^{[2]} &:= W^{[2]} - η \frac{∂ℒ}{∂ W^{[2]}} \\
	&:= W^{[2]} - η \frac{1}{N} ∂_{Z^{[2]}}^T A^{[1]}\\[20pt]
\vb^{[2]} &:= \vb^{[2]} - η \frac{∂ℒ}{∂ \vb^{[2]}} \\
	&:= \vb^{[2]} - η\;\text{mean}_0 (∂_{\vz^{[2](i)}})
\end{align}


m4question([[Do these update equations need to be altered if we want to change the loss function, activation functions, or network architecture?]], [[Yes. Each of these factors play a part in the derivations above.]])

## Neuron Batch Gradient Descent

Let's put this together into an example similar to that shown in the single neuron.

m4code([[Source/Code/Python/05-01-TwoLayerNeuralNetworkMNIST.py]])


## Automatic Differentiation

Let's agree we should avoid computing those derivatives by hand. The process is time consuming and error prone. Instead let's rely on a technique known as *automatic differentiation*, which is built-in to PyTorch and most machine learning frameworks.

An automatic differentiation library:

1. Creates a compute graph from your tensor operations.
2. Performs a topological sort on the compute graph.
3. Compute gradients and back propagates them to all matrices.

Let's take a look at an example.

<!-- Wengert list -->

m4code([[Source/Code/Python/05-02-AutomaticDifferentiation.py]])


Two key points from the listing above, (1) `requires_grad=True` tells PyTorch to create the compute graph and compute partial derivatives with respect to the given tensor, and (2) I've ensured that each line of code contains a single operation, which makes it easier to match with the diagram below (I've provided this one in a bit more detail).


![Compute graph for two-layer network.](img/AutoDiffComputeGraph.svg)


This diagram is (often) constructed dynamically as operations are performed. Edges indicate the flow of gradients in the backward direction. We start at graph source nodes (e.g., the "loss" node) and compute partial derivatives with respect to their inputs until we reach graph sinks (e.g., parameters). This diagram (and the corresponding code) map directly to the hand-computed derivatives from the previous section. Take some time and see if you can see how they map to one another.

If you'd like to see *how* an automatic differentiation library is coded, please take a look at my simple [Match](https://github.com/anthonyjclark/match) library, which tries to closely mimic the PyTorch interface.

### Alternatives

In addition to the technique above known as reverse mode automatic differentiation, you might also hear about

- [forward mode automatic differentiation with dual numbers](https://mostafa-samir.github.io/auto-diff-pt1/),
- [numerical differentiation (Wikipedia)](https://en.wikipedia.org/wiki/Numerical_differentiation), and
- [symbolic differentiation (Wikipedia)](https://en.wikipedia.org/wiki/Computer_algebra).

These techniques are similar and have various deficiences and advantages. Most modern libraries implement reverse mode automatic differentiation.

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


<!-- 


*Sigmoid*: (sigmoid curve, logistic curve/function) a common activation function that is mostly used in the output layer of a binary classifier. Gradient is small whenever the input value is too far from 0.

<img class="syllabus-terms-img" src="images/sigmoid.png">

*Hyperbolic Tangent*: (tanh) another (formerly) common activation funtcion (better than sigmoid, but typically worse than ReLu). Gradient is small whenever the input value is too far from zero.

<img class="syllabus-terms-img" src="images/tanh.png">

*ReLu*: (rectified linear unit, rectifier) the most widely used activation function.

<img class="syllabus-terms-img" src="images/relu.png">

*Leaky ReLu*: a slightly modified version of ReLu where there is a non-zero derivative when the input is less than zero.

<img class="syllabus-terms-img" src="images/leaky-relu.png">




animation like in match?
 -->
