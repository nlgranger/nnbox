# NNBox

NNBox is a Matlab &copy; toolbox for neural networks. Many other toolboxes are 
already available for matlab and may either offer more models, a higher levels 
of support, better optimization, or simply a bigger user community... This 
toolbox tries to focus on different objectives:
- Providing very clear and simple implementations of some neural networks 
  models and architectures.
- Providing a simple template to implement new models rapidly
- Providing a flexible interface where building blocks can be arranged 
  together easily.

In particular, this library provides support for Restricted Boltzmann Machines
(RBM), Convolutional Neural Networks (CNN), simple perceptrons models. It
allows to arrange these models in parallel, as stacked multiple layers, or even
in a Siamese network architecture.

This library does not focus on completeness though, because attempting to do so
rarely gives satisfying results. Instead it tries to provide simple and
flexible architectural fundations to help you implement your own model quickly.

For your information, here is a list of other existing libraries:

- [Matlab Neural Network toolbox](http://fr.mathworks.com/help/nnet/index.html)
- [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) 
  A popular deep learning toolbox
- [MEDAL](https://github.com/dustinstansbury/medal) Similarily provides 
  implementations for several sorts of Deep Learning models.
- [MatConvNet](http://www.vlfeat.org/matconvnet/) Provides awrapper to a C++ 
  implementation of convolutional neural networks. It is actually used here 
  for the CNN model.


## Requirements

As far as I can tell, any version of matlab above R2011a should work, R2014a 
is known to work. Octave is not supported because classes are not yet fully 
supported.


## Installation

Just add nnbox subfolders to your path:

```matlab
addpath('nnbox/utils:nnbox/networks:nnbox/costfun:nnbox/distances');
```

CNN implementation requires the [MatConvNet](http://www.vlfeat.org/matconvnet/) 
library as a backend, follow installation instructions and add the matlab 
bindings to the path.


## Examples

```matlab
X = [0  1  0  1;
     0  0  1  1];
Y = [0 .5 .5  1];
net = Perceptron(2, 1, struct('lRate', 0.5));
trainOpts = struct('nIter', 100, 'displayEvery', 10);
train(net, SquareCost(), X, Y, trainOpts);
```

- MNIST figure recognition using a Deep belief network : 
  [examples/MNIST_DNN.m](examples/MNIST_DNN.m)


## Documentation

Refer to [DOCUMENTATION.md](DOCUMENTATION.md)
