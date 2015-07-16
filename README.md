# NNBox

NNBox is a Matlab &copy; toolbox for neural networks. Many other toolboxes are 
already available for matlab and may either offer more models, a higher levels 
of support, better optimization, or simply a bigger user community... This 
toolboxes has been concieved with two main objectives:
- Providing very clear and simple implementations of some neural networks 
  models and architectures.
- Providing a flexible interface where building block can be arranged 
  together easily.

Below is a list of other existing libraries:

- [Matlab Neural Network toolbox](http://fr.mathworks.com/help/nnet/index.html)
- [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) 
  A popular deep learning toolbox
- [MEDAL](https://github.com/dustinstansbury/medal) Similarily provides 
  implementations for several sorts of Deep Learning models.
- [MatConvNet](http://www.vlfeat.org/matconvnet/) Provides awrapper to a c++ 
  implementation of convolutional neural networks. It is actually used here 
  for the CNN model.

## Requirements

As far as I can tell, any version of matlab above R2011a should work, R2014 is 
known to work. Octave is not supported because classes are not yet fully 
supported.

## Installation

Just add nnbox to your path:

```matlab
addpath('nnbox');
```

CNN require the [MatConvNet](http://www.vlfeat.org/matconvnet/) library, follow 
installation instruction and add the matlab interface to the path 

## Usage

```matlab
net = MultiLayerNet(struct('skipBelow', 1)); 
% TODO
```
    
The documentation will soon be available.

## License

The MIT License (MIT)

Copyright (c) 2015 Nicolas Granger <nicolas.granger @ telecom-sudparis.eu>

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
