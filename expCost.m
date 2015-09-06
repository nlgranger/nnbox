function C = expCost(O, Y, varargin)
thres = 0.7;
assert(isvector(O) && isvector(Y), 'One dimensional input expected');
assert(isnumeric(O), 'Only numeric O is supported');
assert(islogical(Y), 'Only binary Y is supported');

if isempty(varargin)
    C = mean(exp(- 4 * (O-thres) .* (Y-thres)));
elseif strcmp(varargin{1}, 'each')
    C = exp(- 4 * (O-thres) .* (Y-thres));
elseif strcmp(varargin{1}, 'gradient');
    C = -4 * (Y-thres) .* exp(- 4 * (O-thres) .* (Y-thres));
else
    error('Unrecognized optional argument');
end
end