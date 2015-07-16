classdef AbstractNet < handle & matlab.mixin.Copyable
    % Interface for neural networks models.
    
    methods (Abstract)
        S = insize(self)
        % INSIZE Input size
        %   S = INSIZE(obj) returns the expected size of the input similarily 
        %   to size() function. If the network supports distinct groups of 
        %   inputs (eg: expert model), all size vectors are stored in a cell 
        %   array.
        
        S = outsize(self)
        % OUTSIZE Output size
        %   S = OUTSIZE(obj) returns the expected size of the output similarily 
        %   to size() function. If the network supports distinct groups of 
        %   outputs (eg: expert model), all size vectors are stored in a cell 
        %   array.
        
        Y = compute(self, X)
        % COMPUTE Compute network's output
        %   Y = COMPUTE(obj, X) returns the output of the network for a given 
        %   input X with the proper size as returned by insize(). X (and 
        %   consequently Y) may have one extra last dimension in order to pass 
        %   several inputs at once.
        %
        %   [Y, A] = COMPUTE(obj, X) returns forward pass values in A for 
        %   further use in a backpropagation update (implementation specific).
        
        pretrain(self, X)
        % PRETRAIN Unsupervised training
        %   PRETRAIN(obj, X, opts) Perform unsupervized training on input 
        %   data X.
       
        [G, inErr] = backprop(self, A, outErr)
        % BACKPROP Compute gradient update and back-propagated error
        %   [G, inErr] = BACKPROP(A, outErr) compute parameters gradient G
        %   and backpropagated error inErr. G will be submitted to
        %   gradientupdate while inErr is back-propagated.
        %
        %   A      -- forward pass data as return by compute
        %   outErr -- error cost derivative w.r.t. neurons output (not deltas).
        %
        %   G      -- parameters gradient (implementation specific)
        %   inErr  -- error cost derivative w.r.t. neurons input (not deltas).
        
        gradientupdate(G)
        % GRADIENTUPDATE(G) perform one gradient update using data returned 
        %    by the BACKPROP.
        
        train(self, X, Y)
        % TRAIN Perform supervized training on network
        %   TRAIN(obj, X, Y) Optimizes the network using input sample X and  
        %   the corresponding output labels Y.
        
    end % methods (Abstract)
    
end

