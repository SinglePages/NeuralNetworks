# Gradient Descent

Optimizing a neural network follows this process:

1. prepare dataset(s) (e.g., proxy, training, validation, evaluation, etc.),
2. set hyperparameters (e.g., learning rate, number of epochs, etc.),
3. create the model, and
4. train the model.

We've discussed each of these areas in general, but now we'll go into more details starting with the final step, training the model.

## Batch Gradient Descent

All examples thus far have used batch gradient descent (BGD). All gradient descent methods are iterative, meaning we continually make small changes to the parameters until we are satisfied or run out of time. BGD looks something like this:

~~~text
for each epoch
	1. compute gradient with respect to all examples
	2. average gradients across all examples
	3. update parameters using averaged gradients
~~~

In all variants of gradient descent, an epoch refers to the process by which we update the parameters with respect to all training examples. In batch gradient descent, we compute all gradients at once and average them across all examples, resulting in the parameters being updated a single time each epoch. This has the advantage of smoothing out the affect of any outliers and leveraging the parallel nature of modern CPUs and GPUs. On the other hand, it can be a waste of resources (mainly time) to only update the parameters once each epoch.

## Stochastic Gradient Descent

In stochastic Gradient Descent (SGD) we update parameter $N$ times per epoch---once per example. This means that we update parameters more frequently than in BGD.

The **stochastic** part of SGD refers to a random shuffling of the example each epoch. This tends to reduce loss "cycling" where some sequence of repeated example increases and then decreases loss.

~~~text
for each epoch
	randomly shuffle all examples
	for each example
		1. compute gradient with respect to single example
		2. update parameters using gradient
~~~

Although we update the parameters more frequently, not all updates are *good*. Outliers will make the model perform worse in the general case. Moreover, SGD does not take advantage of parallel computations.

## Mini-Batch Stochastic Gradient Descent

Mini-Batch SGD provides a middle ground. We chunk the input into some number of batches and take the average gradient over each batch.

~~~text
for each epoch
	randomly distribute examples into batches
	for each batch
		1. compute gradient with respect to all examples in batch
		2. average gradients across all examples in batch
		3. update parameters using averaged gradients
~~~

This enables us to get the best of both worlds:

- less susceptible to outliers and noise,
- a good number of updates per epoch, and
- good utilization of computing resources.

m4question([[What batch size turns Mini-Batch SGD into BGD? What batch size turns Mini-Batch SGD into SGD?]], [[$N$ and $1$, respectively.]])

m4question([[Will all batches be the same size?]], [[No. The last batch is frequently smaller than all other batches. It contains the leftovers.]])

The code-diff below shows how few changes are needed to convert our BGD example into Mini-Batch SGD.

<!-- Example converting 05-01-TwoLayerNeuralNetworkMNIST.py into MBSGD -->
