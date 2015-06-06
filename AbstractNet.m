classdef AbstractNet < handle
    % Interface for general neural networks operations
    
    methods (Abstract)
        S = insize(self)
        % INSIZE Input size
        %   S = INSIZE(obj) returns the input configuration of the networks. 
        %   S is a vector with the sizes along each dimension. If several 
        %   groups of neurons are distinguished (in expert models for example), 
        %   S is a cell array with one size vector for each group.
        
        S = outsize(self)
        % OUTSIZE Output size
        %   S = OUTSIZE(obj) returns the output configuration of the networks. 
        %   S is a vector with the sizes along each dimension. If several 
        %   groups of neurons are distinguished (in expert models for example), 
        %   S is a cell array with one size vector for each group.
        
        Y = compute(self, X)
        % COMPUTE Compute network's output
        %   Y = COMPUTE(obj, X) returns the output of the network for a given input 
        %   X. The size of X (resp. Y) must match the input (resp. output) 
        %   sizes of the network. X may have one additional dimension if 
        %   several inputs are submitted at once.
        %
        %   [Y, A] = COMPUTE(obj, X) A contains forward pass values for further
        %   use in a backpropagation update (usually contains the input
        %   activity of the neurons).
        
        pretrain(self, X)
        % PRETRAIN Unsupervised training
        %   PRETRAIN(obj, X, opts) Perform unsupervized training.
        %
        %   X    -- training dataset (see compute(obj, X) on how to format X)
       
        [G, inErr] = backprop(self, A, outErr)
        % BACKPROP Compute gradient update and back-propagated error
        %   [G, inErr] = BACKPROP(A, outErr) compute parameters gradient G
        %   and backpropagated error inErr. G will be submitted to
        %   gradientupdate while inErr might be propagated to a lower
        %   neural network.
        %
        %   A      -- forward pass data as return by compute
        %   outErr -- network output error derivative w.r.t. neurons output
        %             (not deltas).
        %
        %   G      -- parameters gradient (can take any type)
        %   inErr  -- network output error derivative w.r.t. neurons
        %             input (not deltas).
        
        gradientupdate(G)
        % GRADIENTUPDATE(G) perform one gradient update using data
        %   returned by the BACKPROP method earlier.
        
        train(self, X, Y)
        % TRAIN Perform supervized training on network
        %   TRAIN(obj, X, Y, opts)
        %   X    -- input data as a numeric array with network input
        %           dimensions and one last dimension to enumerate samples
        %   Y    -- target output data, last dimension corresponds to
        %           samples.
        
    end % methods (Abstract)
    
end

