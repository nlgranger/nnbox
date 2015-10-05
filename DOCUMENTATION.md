# Main features

This library is built around the [AbstractNet](networks/AbstractNet.m) 
interface which imposes how neural networks should be build. In particular, 
neural network are expected to allow backpropagation based optimization. Neural 
networks can then be plugged together into different architectures such as: - 
Multilayer neural networks - Siamese NN - Parallel independant NN (the output 
of which can then get merged)

The following elementary models are implemented:
- Restricted Boltzmann machines (with sigmoïd and ReLU activations)
- CNN (with ReLU activation)
- Perceptrons (by using RBMs without pretraining)

For convenience, we provide an interface to standardize error cost functions in 
[ErrorCost](utils/ErrorCost.m) and a _train_ procedure which takes a neural 
network and trains it supervizedly against a given error cost function.

This library has been developped mostly for an experiment with Siamese 
networks, so a set of vector distances is also provided in distance/.

# Conventions

Input and output values of neural networks are usually arrays. By convention, 
one last additional dimension is added when multiple values are submitted at 
once. For example, a network which takes 2D input and returns 1D output 
can work as follows:

```matlab
im1  = imread('image1.jpg');
im2  = imread('image1.jpg');
data = cat(3, im1, im2);
res  = net.compute(data);
% res now contains 2 columns with the output of the network for im1 and im2
```

Which is equivalent to :

```matlab
im1  = imread('image1.jpg');
im2  = imread('image1.jpg');
res  = net.compute(im1);
res  = [res, net.compute(im1)]; % horizontal concatenation
```

Most of the time, the input and output values are vectors, but for convenience, 
this library also uses the notion of groups of inputs, which is implemented 
as a matlab cell array with arrays inside. If multiple values are passed at 
once, each array has an additional dimension corresponding to the different 
inputs in the same order.
  
# AbstractNet

In this library, a basic neural network is characterized by:
- a feed forward function: This function might be parametric or not, it 
  actually does not need to be a neural network because the internals not 
  standardized. All the models in this library assume the size of the input and 
  of the output are constant.
- a backpropagation function that computes the derivative of an error cost wrt 
  the input neurons (a well a the gradient of the parameters).
- a pretraining function which may optimize the network in any suitable way 
  given a set of input samples.
