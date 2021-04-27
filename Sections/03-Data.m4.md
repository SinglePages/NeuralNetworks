# Data {#sec:data}

Often considered the most important aspect of deep learning,

$$
\mathcal{D} = \{X, Y\}
$$

<!-- TODO: address training/validation/test sets -->

is a dataset comprising input *features* $X$ and output *targets* $Y$. Although $X$ and $Y$ can come in many shapes, I am going to be opinionated here and use a specific (and consistent) convention. Let's use $N$ to denote the size of the paired dataset. (Note, not all problems have output targets, but herein I am talking about supervised learning unless otherwise specified.)

$X$ is a matrix (indicated by capitalization) containing all features of all input examples. A single input example $\mathbf{x}^{(i)}$ is often represented as a *column* vector (indicated by boldface).

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

where $n_x$ is the number of input features. We do not always put the input features into a column vector (see [@sec:cnns] for more information), but it is a useful convention to remember.

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
\end{bmatrix}.
$$

<!-- TODO: insert equations using m4 -->

We need to transpose each example column vector (i.e., $\mathbf{x}^{(1)T}$) into a row vector so that the first dimension of $X$ is the number of examples $N$ and the second dimension is the number of features $n_{n_x}$. (This is not required, but it is the convention I will use for $X$.)

We say that $\mathbf{x}^{(i)} \in \mathcal{R}^{n_x}$ (each input example is $n_x$ real values) and $X \in \mathcal{R}^{N \times n_x}$ (the entire input is a $(N, n_x)$ matrix).

$Y$ contains the targets (also referred to as labels or the true/correct/expected output values).

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

Each $y^{(i)} \in \mathcal{R}^{n_y}$ (each target is $n_y$ real values) and $Y \in \mathcal{R}^{N \times n_y}$ (the entire input is a $(N, n_y)$ matrix).

For example, we might **predict a person's location on Earth in latitude, longitude, and altitude by looking at the temperature, illuminance, time of day, and day of year at their location**. In this example, $n_x$ and $n_y$ are $4$ (temperature, illuminance, time of day, and day of year) and $3$ (latitude, longitude, and altitude), respectively. And if we have $N=785$ example pairs, then $X$ and $Y$ are $(785, 4)$ and $(785, 3)$, respectively.
