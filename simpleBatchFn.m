function [batchX, batchY, idx] = simpleBatchFn(X, Y, batchSz, idx)
    nSamples = size(X, ndims(X));
    szX = size(X);
    szX = szX(1:end-1);
    szY = size(Y);
    szY = szY(1:end-1);
    X = reshape(X, [], nSamples);
    Y = reshape(Y, [], nSamples);
    
    if isempty(idx) % first mini-batch
        idx = randperm(nSamples);
    end
    batchX = reshape(X(:, idx(1:min(batchSz, end))), szX, []);
    batchY = reshape(Y(:, idx(1:min(batchSz, end))), szY, []);
    
    if numel(idx) > batchSz
        idx = idx(batchSz:end);
    else
        idx = [];
    end
end