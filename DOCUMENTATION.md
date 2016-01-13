# NNBox Documentation

## Main features

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
[ErrorCost](costfun/ErrorCost.m) and a [train](utils/train.m) procedure which 
takes a neural network and trains it supervizedly against a given error cost 
function.

This library has been developped mostly for an experiment with Siamese 
networks, so a set of vector distances is also provided in distance/.

## Conventions

### Inputs, outputs and datasets

Input and output values of neural networks are usually arrays. By convention, 
one last additional dimension is added when multiple values are submitted at 
once. For example, a network which takes 2D input and returns 1D output 
can work as follows:

```matlab
im1  = imread('image1.jpg');
im2  = imread('image1.jpg');
data = cat(3, im1, im2); % concatenate samples along the third dimension
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

When building datasets for supervized training, input samples are associated to 
output labels. The labels should follow the same convention and the same order 
as the input values (i-th input is assumed to match the i-th labe).

Most of the time, the input and output values are vectors, but for convenience, 
this library also uses the notion of groups of inputs, which is implemented 
as a matlab cell array with arrays inside. If multiple values are passed at 
once, each array has an additional dimension corresponding to the different 
inputs in the same order.
  
### Object Oriented Programming, copies and handles in Matlab

This paragraph assumes that the reader is familiar with the basic concepts of 
OOP. This library uses Object oriented programming to organize the manipulation 
of neural networks into classes. In order to simplify the notations and get 
have a behaviour similar to other programming languages, the network models
are manipulated as handles. This means class methods alter the supporting 
instance directly instead of working on copies.

Example:

```matlab
% Matlab's way of using methods
obj1 = NormalClassConstructor();
obj2 = obj1.methodWhichAltersTheInstance();
obj1 = obj2; % override original instance

% Alternatively (both methods are always supported by Matlab)
obj1 = NormalClassConstructor();
obj2 = methodWhichAltersTheInstance(obj1); % functional approach to methods
obj1 = obj2;

% Using handles
obj = ClassHandleConstructor();
obj.methodWhichAltersTheInstance(); % modifies obj directly
```

One drawback of using handles is that copies of an instance handle still refer 
to the original instance (C++ users: handles are closer to pointers than 
references; Java users: as usual, an explicit cloning is required for 
duplication):

```matlab
classdef TestClass < handle % encapsulate class inside handles
    properties
        a = 0;
    end
    
    methods
        function increment(self)
            self.a = self.a +1;
        end
    end
end

obj1 = TestClass();
assert(obj1.a == 0); % initial value
obj2 = obj1;         % make a copy (of the handle actually)
obj2.increment();    % alter obj2
assert(obj2.a == 1)  % as expected
assert(obj1.a == 1)  % original object is also modified because both handle 
                     % point to the same instance
```

To make a copy, one must implement matlab.mixin.Copyable and use the 
copy() method to request a copy of an instance:

```matlab
obj2 = copy(obj1);         % make a 'deep' copy
```

Copyable has a default implementation inherited by the root class of all models 
([AbstractNet](networks/AbstractNet.m)). Unless you use nested networks, this 
should be sufficient. Otherwise, you need to explicitely tell matlab how to 
copy the nested networks.

Example:

```matlab
classdef TestNet < handle & AbstractNet % encapsulate class inside handles
    properties
        % Some properties ...
        subnet;
    end
    
    methods
        % Some methods ...
    end
    
    methods(Access = protected) 
        function copy = copyElement(self) % Copyable implementation
            copy = self; % automatic copy of the properties
            % override handle to the older subnet with an independant copy
            copy.subnet = copy(self.subnnet);
        end
    end
end
```
  
# AbstractNet

In this library, a basic neural network is characterized by:
- a feed forward function: This function might be parametric or not, it 
  actually does not need to be a neural network because the internals not 
  standardized. Please, note that all the models in this library assume the 
  size of the input and of the output are constant.
- a backpropagation function that computes the derivative of an error cost wrt 
  the input neurons (a well a the gradient of the parameters).
- a pretraining function which may optimize the network in any suitable way 
  given a set of input samples.
  
As an example we will implement a very simple perceptron neural network.
First, we need to implement AbstractNet which in turn requires to explicit the 
use of handles:

```matlab
classdef Perceptron < handle & AbstractNet
    properties
        W;     % connection weights
        b;     % bias
        lRate; % learning rate
    end
```

A simple constructor:

```matlab
    methods
        function obj = Perceptron(inSz, outSz, lRate)
            obj.W     = randn(inSz, outSz) / sqrt(inSz);
            obj.b     = zeros(outSz, 1);
            obj.lRate = lRate
        end
```

Let's finally implement each method of the AbstractNet interface:

```matlab
        function S = insize(self)
            S = size(self.W, 1);
        end
        
        function S = outsize(self)
            S = size(self.W, 2);
        end
        
        function [Y, A] = compute(self, X)
            if nargout == 2 % training
                % Save necessary values for gradient computation
                A.S = bsxfun(@plus, self.W' * X, self.b); % stimuli
                A.X = X;
                Y   = self.activation(A.S);
                A.Y = Y;
            else % normal
                Y = self.activation(bsxfun(@plus, self.W' * X, self.b));
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)            
            % Gradient computation
            delta  = outErr .* A.Y .* (1 - A.Y);
            G.dW   = A.x * delta';
            G.db   = sum(delta, 2);
            
            % Error backpropagation
            inErr = self.W * delta;
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.W = self.W - opts.lRate * G.dW;
            self.b = self.b - opts.lRate * G.db;
        end
    end % methods
end % classdef
```

# ErrorCost

[ErrorCost](costfun/ErrorCost) provides an interface for error cost functions.
Error cost function guide the training process.

It requires to implement three methods:
- compute which compute the overall error on a dataset, mostly for debugging or
  validation.
- computeEach which compute sample errors individually for debugging
- gradient which computes the derivative of the error w.r.t. the output 
  for each of the provided samples. 
  
# train

While you may want to write your own training procedure, 
[utils/train.m](utils/train.m) might be sufficient or at least provide a good 
starting point. 

Example 1:
```matlab
trainOpts = struct(...
    'nIter', 50, ...
    'batchSz', 500, ...
    'batchFn', @customBatchFn, ...
    'displayEvery', 6);
train(wholeNet, ExpCost(0.75), X, Y, trainOpts);
```

Example 2:
```matlab
X = [0  1  0  1;
     0  0  1  1];
Y = [0 .5 .5  1];
net = Perceptron(2, 1, struct('lRate', 0.5));
trainOpts = struct('nIter', 100, 'displayEvery', 10);
train(net, SquareCost(), X, Y, trainOpts);
```
