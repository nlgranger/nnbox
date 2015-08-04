function C = L2Cost(O, Y, varargin)
assert(isnumeric(Y), 'Only numeric O and Y are supported');

nSamples = size(O, ndims(O));
if isempty(varargin)
    C = mean(sum(reshape((O - Y) .^ 2, [], nSamples), 1));
elseif strcmp(varargin{1}, 'each')
    C = sum(reshape((O - Y) .^ 2, [], nSamples), 1);
elseif strcmp(varargin{1}, 'gradient');
    C     = (O - Y);
else
    error('Unrecognized optional argument');
end
end