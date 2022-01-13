# Data {#sec:data}

Perhaps the most important aspect of a neural network is the dataset. Let

$$\mathcal{D} = \{X, Y\}$$

denote a dataset comprising input *features* $X$ and output *targets* $Y$. Although $X$ and $Y$ can come in many shapes, I am going to be opinionated here and use a specific (and consistent) convention. Let's use $N$ to denote the size of the paired dataset. (Note, not all problems have output targets, but herein I am talking about supervised learning unless otherwise specified.)

We will frequently take a dataset and split it into examples used for training, validation, and evaluation. We'll discuss these terms near the end of this section.

$X$ is a matrix (indicated by capitalization) containing all features of all input examples. A single input example $\mathbf{x}^{(i)}$ is often represented as a *column* vector (indicated by boldface):


$$\mathbf{x}^{(i)} = m4colvec("x^{(i)}_{row}", "n_x")$$


where subscripts denote the feature index, $n_x$ is the number of features, and the superscript $i$ denotes that this is the $i^{\mathit{th}}$ training example. We do not always put the input features into a column vector (see [@sec:cnns] for more information), but it is fairly standard.

Each row in $X$ is a single input example (also referred to as an instance or sample), and when you stack all $N$ examples on top of each other (first transposing them into row vectors), you end up with:


$$X = m4colvec("\rule[.5ex]{1em}{0.4pt}\mathbf{x}^{(row)T}\rule[.5ex]{1em}{0.4pt}", "N") = m4matrix("x^{(row)}_{col}", "N", "n_x")$$


We transpose each example column vector (i.e., $\mathbf{x}^{(i)T}$) into a row vector so that the first dimension of $X$ corresponds to the number of examples $N$ and the second dimension is the number of features $n_x$. Compare the column vector above to each row in the matrix.

Let's denote matrix dimensions with $(r \times c)$ (the number of rows $r$ by the number of columns $c$ in the matrix). I will, in text and in code, refer to matrix dimensions as the "shape" of the matrix.


m4question([[What is the shape of $X$?]], [[We say that $\mathbf{x}^{(i)} \in \mathcal{R}^{n_x}$ (each input example is $n_x$ real values) and $X \in \mathcal{R}^{N \times n_x}$. Therefore, the shape of $X$ is $(N \times n_x)$.]])


$Y$ contains the targets (also referred to as labels or the true/correct/actual/expected output values). Here is a single target column vector:


$$\mathbf{y}^{(i)} = m4colvec("y^{(i)}_{row}", "n_y")$$


And here is the entire target matrix including all examples:


$$Y = m4colvec("\rule[.5ex]{1em}{0.4pt}\mathbf{y}^{(row)T}\rule[.5ex]{1em}{0.4pt}", "N") = m4matrix("y^{(row)}_{col}", "N", "n_y")$$


m4question([[What is the shape of $Y$?]], [[The shape of $Y$ is $(N \times n_y)$.]])


<!-- For example, we might **predict a person's location on Earth in latitude, longitude, and altitude by looking at the temperature, illuminance, time of day, and day of year at their location**. In this example, $n_x$ and $n_y$ are $4$ (temperature, illuminance, time of day, and day of year) and $3$ (latitude, longitude, and altitude), respectively. And if we have $N=785$ example pairs, then $X$ and $Y$ are $(785, 4)$ and $(785, 3)$, respectively. -->

Let's use the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) as an example. This dataset comprises a training partition including 60,000 images and a validation partition including 10,000 images. Each image is 28 pixels in height and 28 pixels in width for a total of 784 pixels. Each image depicts a single handwritten digit---a number in the range zero through nine). Here is a small sample of these images:


![MNIST Sample. Image from Wikipedia.](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)


m4question([[What is the shape of the training partition of the input $X_{train}$?]], [[$X_{train}$ is $(60000 \times 784$): $$X = m4matrix("x^{(row)}_{col}", "60000", "784")$$ The first row includes all 784 pixels of the first training image, and subsequent rows likewise contain pixel data for a single image.]])


m4question([[What is the shape of the training partition of the targets $Y_{train}$?]], [[$Y_{train}$ is $(60000 \times 10$): $$Y = m4matrix("x^{(row)}_{col}", "60000", "10")$$ Each row in this matrix is one-hot encoded, meaning that only one item in each row is "1" and all other items in a row are "0". Here is an example of a one-hot encoding target for an input image representing the digit "2" $$y^T = \begin{bmatrix} 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}$$ For efficiency sake, we often represent a one-hot encoded vector using just the index of the "hot" item. For example, the previous vector can be represented by the integer 2.]])


m4question([[What are the shapes of $X_{valid}$ and $Y_{valid}$?]], [[$X_{valid}$ and $Y_{valid}$ are $(10000 \times 784)$ and $(10000 \times 10)$, respectively.]])


You might now wonder why we split a dataset into training/validation/evaluation partitions. It is reasonable to think that we would be better off using all 70000 images to train a neural network. However, we need some method for *measuring* how well a model is performing. That is the purpose of the validation set--to measure performance.

If we measure performance directly on the training dataset, we might trick ourselves into thinking that the neural network will perform very well when it is eventually deployed as part of an application (for example, as a mobile app in which we convert an image of someones's handwritten notes into a text document), when in reality the network might only perform well specifically on the examples found in the training dataset. We will discuss this issue more in [@sec:generalization] when we cover overfitting and generalization.

Similarly, the evaluation partition is only used to compare performance after hyper-parameter tuning, which we'll discuss in [@sec:hyper].

## Loading MNIST Using PyTorch

We've discuss notation and general concepts, but how would we write this out in code? Here is an example how how to load the MNIST dataset using PyTorch.


m4code([[Source/Code/Python/03-01-LoadMNIST.py]])


m4question([[What do you expect to see as this program's output?]], [[
<pre class="code-block">
Training input shape    : torch.Size([60000, 1, 28, 28])
Training target shape   : torch.Size([60000])
Validation input shape  : torch.Size([10000, 1, 28, 28])
Validation target shape : torch.Size([10000])
</pre>

This is slightly different than what we discussed. PyTorch expects us to use this dataset with a convolutional neural network. When we get to [@sec:cnns] we'll make more sense of this data format.
]])


## A Non-ML Digit Classifier


<!-- - non-ml example -->
