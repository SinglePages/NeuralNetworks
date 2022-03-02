# Terminology {.unnumbered #sec:terms}

<!-- 
- transformer
- graph neural network
- objective
-->



**Machine Learning**

*Artificial Intelligence* (AI): computer systems that are capable of completing tasks that typically require a human. This is a moving bar--as something becomes easier for a computer, we tend to stop considering it as AI (how long until deep learning is not AI?).

*Machine Learning* (ML): learn a predictive model from data (e.g., deep learning and random forests). ML is related to data mining and pattern recognition.

*Deep Learning* (DL): learn a neural network model with two or more hidden layers.

*Supervised Learning*: learn a mapping from input features to output values using labeled examples (e.g., image classification).

*Unsupervised Learning*: extract relationships among data examples (e.g., clustering).

*Reinforcement Learning* (RL): learn a model that maximizes rewards provided by the environment (or minimize penalties).

*Hybrid Learning*: combine methods from supervised, unsupervised, and reinforcement learning (e.g., semi-supervised learning).

*Classification*: given a set of input features, produce a discrete output value (e.g., predict whether a written review is negative, neutral, or positive).

*Regression*: given a set of input features, produce a continuous output value (e.g., predict the price of a house from the square footage, location, etc.).

*Clustering*: a grouping of input examples such that those that are most similar are in the same group.

*Model*: (predictor, prediction function, hypothesis, classifier) a model along with its parameters.

*Example*: (instance, sample, observation, training pair) an input training/validation/testing input (along with its label in the case of supervised learning).

*Input*: (features, feature vector, attributes, covariates, independent variables) values used to make predictions.

*Channel*: subset of an input--typically refers to the red, green, or blue values of an image.

*Output*: (label, dependent variable, class, prediction) a prediction provided by the model.

*Linear Separability*: two sets of inputs can be divided a hyperplane (a line in the case of two dimensions). This is the easiest case for learning a binary classification.

*Parameter*: (weights and biases, beta, etc.) any model values that are learned during training.

*Hyperparameter* (learning rate, number of epochs, architecture, etc.): any value that affects training results but is not directly learned during training.

**Neural Network Terms**

*Neural Network* (NN): (multi-layer perceptron (MLP), artificial NN (ANN)) a machine learning model (very loosely) based on biological nervous systems.

*Perceptron*: a single layer, binary classification NN (only capable of learning linearly separable patterns).

*Neuron*: (node) a single unit of computation in a NN. A neuron typically refers to a linear (affine) computation followed by a nonlinear activation.

*Activation*: (activation function, squashing function, nonlinearity) a neuron function that provides a nonlinear transformation (see [this Stack Exchange Post for some examples and equations](https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons)).

*Layer*: many NNs are simply a sequence of layers, where each layer contains some number of neurons.

*Input Layer*: the input features of a NN (the first "layer"). These can be the raw values or scaled values--we typically normalize inputs or scale them to either [0, 1] or [-1, 1].

*Hidden Layer*: a NN layer for which we do not directly observe the values during inference (all layers that are not an input or output layer).

*Output Layer*: the final layer of a NN. The output of this layer is (are) the prediction(s).

*Architecture*: a specific instance of a NN, where the types of neurons and connectivity of those neurons are specified (e.g., VGG16, ResNet34, etc.). The architecture sometimes includes optimization techniques such as batch normalization.

*Forward Propagation*: the process of computing the output from the input.

*Training*: the process of learning model parameters.

*Inference*: (deployment, application) the process of using a trained model.

*Dataset*: (training, validation/development, testing) a set of data used for training a model. Typically a dataset is split into a set used for training (the training set), a set for computing metrics (the validation/development set), and a set for evaluation (the testing set).

*Convolutional Neural Network* (CNN): a NN using convolutional filters. These are best suited for problems where the input features have geometric properties--mainly images (see [3D Visualization of a Convolutional Neural Network](https://www.cs.ryerson.ca/~aharley/vis/conv/)).

*Filter*: a convolution filter is a matrix that can be used to detect features in an image; they will normally produce a two-dimensional output (see [Image Kernels Explained Visually](https://setosa.io/ev/image-kernels/), [Convolution Visualizer](https://ezyang.github.io/convolution-visualizer/index.html), and [Receptive Field Calculator](https://fomoro.com/research/article/receptive-field-calculator)). Filters will typically have a kernel size, padding size, dilation amount, and stride.

*Pooling*: (average-pooling, max-pooling, pooling layer) a pooling layer is typically used to reduce the size of a filter output.

*Autoencoder*: a common type of NN used to learn new or compressed representations.

*Recurrent Neural Network* (RNN): A NN where neurons can maintain an internal state or backward connections and exhibit with temporal dynamics. One type of RNN is a **recursive** neural network.

*Long Short-Term Memory* (LSTM): a common type of RNN developed in part to deal with the vanishing gradient problem (see [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) (YouTube)](https://www.youtube.com/watch?v=WCUNPb-5EYI)).

**Learning Terms**

*Loss*: (loss function) a function that we minimize during learning. We take the gradient of loss with respect to each parameter and then move down the slope. Loss is frequently defined as the error for a single example in supervised learning.

*Cost*: (cost function) similar to loss, this is a function that we try to minimize. Cost is frequently defined as the sum of loss for all examples.

*Generalization*: how well a model extrapolates to unseen data.

*Overfitting*: how much the model has memorized characteristics of the training input (instead of generalizing).

*Regularization*: a set of methods meant to prevent overfitting. Regularization reduces overfitting by shrinking parameter values (larger parameters typically means more overfitting).

*Bias*: when a model has larger-than-expected training and validation loss.

*Variance*: when model has a much larger validation error compared to the training error (also an indication of overfitting).

*Uncertainty*: some models can estimate a confidence in a given prediction.

*Embedding*: a vector representation of a discrete variable (e.g., a method for representing an English language word as an input feature).

**Activation Terms**

*Affine*: (affine layer, affine transformation) the combination of a linear transformation and a translation (this results in a linear transformation).

*Nonlinear*: a function for which the change in the output is not proportional to the change in the input.

*Sigmoid*: (sigmoid curve, logistic curve/function) a common activation function that is mostly used in the output layer of a binary classifier. Gradient is small whenever the input value is too far from 0.

![Sigmoid Activation Function](img/Sigmoid.png)

*Hyperbolic Tangent*: (tanh) another (formerly) common activation funtcion (better than sigmoid, but typically worse than ReLu). Gradient is small whenever the input value is too far from zero.

![Hyperbolic Tangent  Activation Function](img/Tanh.png)

*ReLU*: (rectified linear unit, rectifier) the most widely used activation function.

![ReLU Activation Function](img/ReLU.png)

*Leaky ReLU*: a slightly modified version of ReLU where there is a non-zero derivative when the input is less than zero.

![Leaky ReLU Activation Function](img/LeakyReLU.png)

*Softmax*: (softmax function, softargmax, log loss) is a standard activation function for the last layer of a multi-class NN classifier. It turns the outputs of several nodes into a probability distribution (see [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)).

**Learning Techniques**

*Data Augmentation*: the process of altering inputs each epoch thereby *increasing* the effective training set size.

*Transfer Learning*: use a trained model (or part of it) on an input from a different distribution. Most frequently this also involve fine-tuning.

*Fine-tuning*: training/learning only a subset of all parameters (usually only those nearest the output layer).

*Dropout*: a regularization technique in which neurons are randomly zeroed out during training.

*Batch Normalization*: is a technique that speeds up training by normalizing the values of hidden layers across input batches. Normalizing hidden neuron values will keep derivatives higher on average.

*Attention*: (attention mechanism, neural attention) is a technique that enables a NN to focus on a subset of the input data (see [Attention in Neural Networks and How to Use It](https://akosiorek.github.io/ml/2017/10/14/visual-attention.html)).

**Optimization**

*Gradient Descent* (GD): (batch GD (BGD), stochastic GD (SGD), mini-batch GD) a first-order optimization algorithm that can be used to learn parameters for a model.

*Backpropagation*: application of the calculus chain-rule for NNs.

*Learning Rate*: a hyperparameter that adjusts the training speed (too high will lead to divergence).

*Vanishing Gradients*: an issue for deeper NNs where gradients *saturate* (becomes close to zero) and training is effectively halted.

*Exploding Gradients*: an issue for deeper NNs where gradients accumulate and result in large updates causing gradient descent to diverge.

*Batch*: a subset of the input dataset used to update the NN parameters (as opposed to using the entire input dataset at once).

*Epoch*: each time a NN is updated using all inputs (whether all at once or using all batches).

*Momentum*: an SGD add-on that speeds up training when derivatives stay the same sign each update.

*AdaGrad*: a variant of SGD with an adaptive learning rate (see [Papers with Coe: AdaGrad](https://paperswithcode.com/method/adagrad)).

*AdaDelta*: a variant of SGD/AdaGrad (see [Papers With Code: AdaDelta](https://www.paperswithcode.com/method/adadelta)).

*RMSProp*: a variant of SGD with an adaptive learning rate (see [Papers With Code: RMSProp](https://paperswithcode.com/method/rmsprop)).

*Adam*: a variant of SGD with momentum and scaling (see [Papers With Code: Adam](https://paperswithcode.com/method/adam)).

*Automatic Differentiation* (AD): a technique to automatically evaluate the derivative of a function.

*Cross-Entropy Loss*: (negative log likelihood (NLL), logistic loss) a loss function commonly used for classification.

*Backpropagation Through Time*: (BPTT) a gradient-based optimization technique for **recurrent** neural networks.
