---
title:  "Neural Networks"
subtitle: "A terse neural network walk-through"
author: "Anthony J. Clark"
...

# Introduction {#sec:intro}

Goal: provide a concise walk through of all fundamental neural network (including modern deep learning) techniques.

I will not discuss every possible analogy, angle, or topic here. Instead, I will provide links to external resources so that you can choose which topics you want to investigate more closely.

**Useful prior knowledge:**

- matrix calculus (see [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) by Terence Parr and Jeremy Howard)
- programming skills (I will show examples in Python, but many languages will work)
- familiarity with computing tools (using a server, cloud-based services, the command line interface (CLI))

TODO: background

- AI/ML/NN
- applications
- terminology (nn, ann, mlp)
- ethics
- non-ml example

## Notation

For the sake of making this concrete, I am going to introduce a reference example: **predicting a person's location on Earth (latitude, longitude, and elevation) by looking at the temperature, illuminance, time of day, and day of year.**

Starting at the top,

$$
\mathcal{D} = \{X, Y\}
$$

is a dataset comprising input *features* $X$ and output *targets* $Y$. Although $X$ and $Y$ can come in many shapes, I am going to be opinionated here and force a specific convention. For the supervised learning case, we will always have the same number of input feature and output target sets. Let's use $N$ to denote the size of the paired dataset.

$X$ is a matrix (indicated by the capitalization) containing all features of all input examples. A single input example $\mathbf{x}^{(i)}$ ($\mathbf{x}$ is **bold** to indicate that it is a vector) is often represented as a *column* vector:

$$
\mathbf{x}^{(i)} =
\begin{bmatrix}
x^{(i)}_{1} \\
x^{(i)}_{2} \\
\vdots \\
x^{(i)}_{n_x-1} \\
x^{(i)}_{n_x} \\
\end{bmatrix}
$$

where $n_x$ is the number of input features ($n_x = 4$ in our reference example). For example, $\mathbf{x}^{(1)}$ would contain the temperature, illuminance, time of day, and day of year for the first person in our dataset. We do not always put the input features into a column vector, but it is a good starting place (see [@sec:cnns] for more information).

Each row in $X$ is a single input example (also referred to as an instance or sample), and when you stack all $N$ examples side-by-side, you end up with

$$
X =
\begin{bmatrix}
\mathbf{x}^{(1)T}\\
\mathbf{x}^{(2)T}\\
\vdots\\
\mathbf{x}^{(N)T}
\end{bmatrix}
=
\begin{bmatrix}
x^{(1)}_{1} & x^{(1)}_{2} & \cdots & x^{(1)}_{n_x-1} & x^{(1)}_{n_x}\\
x^{(2)}_{1} & x^{(2)}_{2} & \cdots & x^{(2)}_{n_x-1} & x^{(2)}_{n_x}\\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x^{(N-1)}_{1} & x^{(N-1)}_{2} & \cdots & x^{(N-1)}_{n_x-1} & x^{(N-1)}_{n_x}\\
x^{(N)}_{1} & x^{(N)}_{2} & \cdots & x^{(N)}_{n_x-1} & x^{(N)}_{n_x}\\
\end{bmatrix}
$$

<!-- TODO: insert equations using m4 -->

where we need to transpose each example column vector to make it fit correctly in the matrix as a row.

We say that $\mathbf{x}^{(i)} \in \mathcal{R}^{n_x}$ (each input example is $n_x$ real values) and $X \in \mathcal{R}^{N \times n_x}$ (the entire input is a $(N, n_x)$ matrix).

$Y$ contains the targets (also referred to as labels). Using our example, targets comprise the latitude, longitude, and elevation of a person.

$$
Y =
\begin{bmatrix}
\mathbf{y}^{(1)T}\\
\mathbf{y}^{(2)T}\\
\vdots\\
\mathbf{y}^{(N)T}
\end{bmatrix}
=
\begin{bmatrix}
y^{(1)}_{1} & y^{(1)}_{2} & \cdots & y^{(1)}_{n_y-1} & y^{(1)}_{n_y}\\
y^{(2)}_{1} & y^{(2)}_{2} & \cdots & y^{(2)}_{n_y-1} & y^{(2)}_{n_y}\\
\vdots & \vdots & \ddots & \vdots & \vdots \\
y^{(N-1)}_{1} & y^{(N-1)}_{2} & \cdots & y^{(N-1)}_{n_y-1} & y^{(N-1)}_{n_y}\\
y^{(N)}_{1} & y^{(N)}_{2} & \cdots & y^{(N)}_{n_y-1} & y^{(N)}_{n_y}\\
\end{bmatrix}
$$

Each $y^{(i)} \in \mathcal{R}^{n_y}$ (each target is $n_y$ real values, 3 in our example) and $Y \in \mathcal{R}^{N \times n_y}$ (the entire input is a $(N, n_y)$ matrix).
