function trained = train(net, costFn, X, Y, opts)
% TRAIN trains a neural network
%	trained = train(net, costFn, X, Y, opts) returns a trained copy of net.
%   Supervized training performs gradient updates to minimize the cost function 
%   costFn on the dataset. Sample inputs and associated outputs are given by X 
%   and Y respectively.
%
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
		if isfield(opts, 'batchFn')
			E           = [];
			S           = [];
			moreBatches = true;
			idx         = [];
			while moreBatches
				[batchX, batchY, idx] = ...
					opts.batchFn(X, Y, opts.batchSz, idx);
				nSamples    = size(batchY, ndims(batchY));
				O           = trained.compute(batchX);
				E           = [E costFn(O, batchY)];
				S           = [S nSamples];
				moreBatches = ~isempty(idx);
			end
			MC = sum(E .* S ./ sum(S));
		else
			O  = trained.compute(X);
			MC = costFn(O, Y);
		end
		fprintf('%03d , Error cost : %1.3e\n', i, MC);
    end
end
end
