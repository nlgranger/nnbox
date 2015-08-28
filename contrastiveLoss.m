function C = contrastiveLoss(O, Y, varargin)
Q = 0.9;
assert(isnumeric(O), 'Only numeric O is supported');
assert(islogical(Y), 'Y must be a boolean array');
nSamples = numel(O);
if isempty(varargin)
    C   = 0.5 / nSamples * ( 4 * sum(O(~Y) .^2) ...
                           + 2 * Q *sum(exp(- 2.77 / Q * O(Y))));
elseif strcmp(varargin{1}, 'each')
    C   = 0.5 * ( 4 * O .* ~Y .^2 ...
                + 2 * Q * exp(- 2.77 / Q * O) .* Y);
elseif strcmp(varargin{1}, 'gradient')
    C   = 2 * O .* ~Y ...
        - 2 * 2.77 * exp(- 2.77 / Q * O) .* Y;
else
    error('Unrecognized optional argument');
end
end
