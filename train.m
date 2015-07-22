function trained = train(net, costFn, X, Y, opts)
% TRAIN trains a neural network
%   trained = train(net, costFn, X, Y, opts) returns a trained copy of net.
%   The supervized training is done using X and Y the input and associated
%   output samples under the cost function costFn. 
%   A cost function should take three arguments: 
%       - the network's output, 
%       - the expected output samples
%       - a flag 'gradient'
%   and return the derivative of the cost function with respect to its input 
%   variables.
%   opts is a structure containing training settings:
%      nIter        -- # of training epochs
%      batchSz      -- # of samples by mini-batch [optional]
%      displayEvery -- frequency of progress display [optional]

assert(isa(net, 'AbstractNet'), ...
    'net should implement AbstractNet');
assert(isnumeric(Y) || islogical(Y), 'Only numeric output is supported');

if iscell(X)
    nSamples = size(X{1}, ndims(X{1}));
    XSz      = cell(numel(X), 1);
    Xcol     = cell(numel(X), 1);
    for i = 1:numel(X);
        XSz{i}  = size(X{i});
        Xcol{i} = reshape(X{i}, [], nSamples);
    end
else
    nSamples = size(X, ndims(X));
    XSz      = size(X);
    Xcol     = reshape(X, [], nSamples);
end

trained = net.copy();

Ycol = reshape(Y, [], nSamples);
YSz  = size(Y);


for i = 1:opts.nIter
    shuffle = randperm(nSamples);
    
    for start = 1:opts.batchSz:nSamples % batch loop
        idx = shuffle(start:min(start+opts.batchSz-1, nSamples));
        batchSz = numel(idx);
        
        if iscell(X)
            batchX = cell(length(X), 1);
            for g = 1:length(X)
                batchX{g} = reshape(Xcol{g}(:, idx), ...
                    [XSz{g}(1:end-1) batchSz]);
            end
        else
            batchX = reshape(Xcol(:, idx), [XSz(1:end-1) batchSz]);
        end
        batchY = reshape(Ycol(:,idx), [YSz(1:end-1), batchSz]);
        
        [O, A] = trained.compute(batchX); % forward pass
        outGrad = costFn(O, batchY, 'gradient'); % L2 error derivative
        G = trained.backprop(A, outGrad);
        trained.gradientupdate(G);
    end
    
    % Report
    if isfield(opts, 'displayEvery') && mod(i, opts.displayEvery) == 0
        MC = costFn(net.compute(X), Y);
        fprintf('%03d , Error cost : %1.3e\n', i, MC);
    end
end
end
