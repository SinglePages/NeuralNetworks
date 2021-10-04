# Multi-Layer Networks

<!-- - mlp
- deep networks -->

![A two-layer neural network.](img/2LayerNetwork.svg)

- Layer 0 is the input (formerly X)
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
