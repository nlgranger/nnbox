function C = L2Cost(O, Y, varargin)
assert(isnumeric(Y), 'Only numeric output is supported');

if isempty(varargin)
    C = sum(reshape((O - Y) .^ 2, [], 1));
elseif strcmp(varargin{1}, 'each')
    C = reshape((O - Y) .^ 2, [], 1);
elseif strcmp(varargin{1}, 'gradient');
    C = O - Y;
else
    error('Unrecognized optional argument');
end
end