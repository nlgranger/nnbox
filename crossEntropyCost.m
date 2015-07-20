function C = crossEntropyCost(O, Y, varargin)
assert(isnumeric(O), 'Only numeric O is supported');
assert(islogical(Y), 'Y must be a boolean array');

nSamples = size(O, ndims(O));
if isempty(varargin)
    C     = zeros(size(O));
    C(Y)  = - log(O(Y));
    C(~Y) = - log(1 - O(~Y));
    C     = mean(sum(C, 1));
elseif strcmp(varargin{1}, 'each')
    C     = zeros(size(O));
    C(Y)  = - log(O(Y));
    C(~Y) = - log(1 - O(~Y));
    C     = sum(C, 1);
elseif strcmp(varargin{1}, 'gradient');
    C     = zeros(size(O));
    C(Y)  = - 1 ./ O(Y) ./ nSamples;
    C(~Y) = 1 ./ (1 - O(~Y)) ./ nSamples;
else
    error('Unrecognized optional argument');
end
end
