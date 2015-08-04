function trained = train(net, costFn, X, Y, opts)
% TRAIN trains a neural network
%	trained = train(net, costFn, X, Y, opts) returns a trained copy of net.
%   The supervized training is done using X and Y the input and associated
%   output samples under the cost function costFn. 
%   A cost function should take three arguments: 
%       - the network's output, 
%       - the expected output samples
%       - a flag 'gradient'
%   and return the derivative of the cost function with respect to its 
%   input variables.
%   opts is a structure containing training settings:
%       nIter        -- # of training epochs
%       batchSz      -- # of samples by mini-batch [optional]
%       displayEvery -- frequency of progress display [optional]
%       batchFn      -- returns mini-batches from the dataset
%   train(net, costFn, X, Y, opts) takes net as a handle (reference) and
%   trains in-place

assert(isa(net, 'AbstractNet'), ...
    'net should implement AbstractNet');
assert(isnumeric(Y) || islogical(Y), 'Only numeric output is supported');

if ~isfield(opts, 'batchFn') && isfield(opts, 'batchSz')
    opts.batchFn = @simpleBatchFn;
end

if nargin == 1
    trained = net.copy();
else
    trained = net;
end

for i = 1:opts.nIter
    if isfield(opts, 'batchFn') % train over mini-batches
        idx = [];
        moreBatches = true; % set to any value to get inside the loop
        while moreBatches
            [batchX, batchY, idx] = opts.batchFn(X, Y, opts.batchSz, idx);
            [O, A] = trained.compute(batchX);        % forward pass
            moreBatches = ~isempty(idx);
            clear batchX;                            % release memory
            outGrad = costFn(O, batchY, 'gradient'); % error derivative
            G = trained.backprop(A, outGrad);        % backpropagation
            trained.gradientupdate(G);               % update
        end
    else % train over the whole dataset
        [O, A] = trained.compute(X);        % forward pass
        outGrad = costFn(O, Y, 'gradient'); % L2 error derivative
        G = trained.backprop(A, outGrad);   % error derivative
        trained.gradientupdate(G);          % update
    end
    
    % Report
    if isfield(opts, 'displayEvery') && mod(i, opts.displayEvery) == 0
        [batchX, batchY] = opts.batchFn(X, Y, inf, []);
        O  = trained.compute(batchX);
        MC = costFn(O, batchY);
        fprintf('%03d , Error cost : %1.3e\n', i, MC);
    end
end
end
