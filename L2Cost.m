function C = L2Cost(O, Y, varargin)
assert(isnumeric(Y), 'Only numeric O and Y are supported');

nSamples = size(O, ndims(O));
if isempty(varargin)
    C = mean(sqrt(sum(reshape((O - Y) .^ 2, [], nSamples), 1)));
elseif strcmp(varargin{1}, 'each')
    C = sqrt(sum(reshape((O - Y) .^ 2, [], nSamples), 1));
elseif strcmp(varargin{1}, 'gradient');
    eachC = sqrt(sum(reshape((O - Y) .^ 2, [], nSamples), 1));
    C    = bsxfun(@rdivide, O - Y, eachC * nSamples);
else
    error('Unrecognized optional argument');
end
end