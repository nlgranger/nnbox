% This file demonstrates the use of the NNBox on the MNIST figure database
% Using the model from Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A
% fast learning algorithm for deep belief nets. Neural computation, 18(7),
% 1527-1554.

% Load Database
[trainX, trainY, testX, testY] = getMNIST();
trainX = double(reshape(trainX, 28*28, 60000)) / 255;
trainY = ((0:9)' * ones(1, 60000)) == (ones(10, 1) * double(trainY'));
testX  = double(reshape(testX, 28*28, 10000)) / 255;
testY  = ((0:9)' * ones(1, 10000)) == (ones(10, 1) * double(testY'));

% Setup network

net  = MultiLayerNet();

pretrainOpts = struct( ...
    'nEpochs', 50, ...
    'momentum', 0.7, ...
    'lRate', 1e-3, ...
    'batchSz', 200, ...
    'dropout', 0.3, ...
    'displayEvery', 5);
trainOpts = struct( ...
    'lRate', 5e-4, ...
    'batchSz', 200);
rbm1 = RBM(28*28, 500, pretrainOpts, trainOpts);
net.add(rbm1);

pretrainOpts.nEpochs = 15;
trainOpts = struct( ...
    'lRate', 5e-4, ...
    'batchSz', 200);
rbm2 = RBM(500, 500, pretrainOpts, trainOpts);
net.add(rbm2);

% Pretrain bottom layers
fprintf('Pretraining first two layer\n');
net.pretrain(trainX);

% Finish network
rbm3 = RBM(500, 2000, pretrainOpts, trainOpts);
net.add(rbm3);

per  = Perceptron(2000, 10, trainOpts);
net.add(per);

% Train
fprintf('Fine-tuning\n');

trainOpts = struct(...
    'nIter', 50, ...
    'batchSz', 500, ...
    'displayEvery', 3);
train(net, CrossEntropyCost(), trainX, trainY, trainOpts);
