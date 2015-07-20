classdef AbstractCost
   
    methods (Abstract)
        C = compute(self, net, X, Y, varargin)
        % COMPUTE compute error cost on a dataset
        %   C = COMPUTE(self, net, X, Y) computes the error cost of network
        %   net on input samples X with associated labels Y.
        %   C = COMPUTE(self, net, X, Y, 'each') returns a vector with the
        %   error cost for each sample independantly.
        
        trained = train(self, net, X, Y, opts)
        % TRAIN execute a supervized (backpropagation) learning
        %   trained = train(self, net, X, Y, trainOpts) returns a trained
        %   copy of the network net using input samples X and corresponding 
        %   outputs Y. Training is done via backpropagation with respect to an 
        %   implementation specific cost function. 
        %   The opts structure is used to tune training and should at least 
        %   support the following options:
        %       nIter        -- # of training epochs
        %       batchSz      -- # of samples in each batches [optional]
        %       displayEvery -- display rate of training information 
        %                       (iteration number, error cost on the dataset, 
        %                       ...) [optional]
    end
    
end