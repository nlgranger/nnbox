function C = expCost(O, Y, varargin)
assert(isvector(O) && isvector(Y), 'One dimensional input expected');
assert(isnumeric(O), 'Only numeric O is supported');
assert(islogical(Y), 'Only binary Y is supported');

if isempty(varargin)
    C = mean(exp(- 4 * (O-.5) .* (Y-.5)));
elseif strcmp(varargin{1}, 'each')
    C = exp(- 4 * (O-.5) .* (Y-.5));
elseif strcmp(varargin{1}, 'gradient');
    C = -4 * (Y-.5) .* exp(- 4 * (O-.5) .* (Y-.5));
else
    error('Unrecognized optional argument');
end
end