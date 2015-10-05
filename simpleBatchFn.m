function [batchX, batchY, idx] = simpleBatchFn(X, Y, batchSz, idx)
% Generates batches from a dataset
%   [batchX, batchY, I1] = SIMPLEBATCHFN(X, Y, N, []) Generates on batch of
%   N samples out of the array based dataset X and Y (assuming samples are
%   concatenated alongside the last dimension for each). I1 should be
%   passed to SIMPLEBATCHFN in order to generate the next batch. If less
%   than N samples are availables, the batch will contain all remaining
%   samples.
%
%   [batchX, batchY, I2] = simpleBatchFn(X, Y, N, I1) Generates a batch of
%   samples which avoids samples that led to I1 (I2 becomes I1 for the next
%   iteration).
%
%   Example:
%       N = 100;
%       [batchX, batchY, I] = opts.batchFn(X, Y, N, []);
%       while ~isempty(batchX)
%           % do somthing with batchX and batchY
%           [batchX, batchY, I] = opts.batchFn(X, Y, N, I);
%       end

    nS  = size(X, ndims(X));
    szX = size(X);
    szX = szX(1:end-1);
    szY = size(Y);
    szY = szY(1:end-1);
    X   = reshape(X, [], nS);
    Y   = reshape(Y, [], nS);
    
    if isempty(idx) % first mini-batch
        idx = randperm(nS);
    end
    batchX = reshape(X(:, idx(1:min(batchSz, end))), szX, []);
    batchY = reshape(Y(:, idx(1:min(batchSz, end))), szY, []);
    
    if numel(idx) > batchSz
        idx = idx(batchSz:end);
    else
        idx = [];
    end
end