function C = contrastiveLoss(O, Y, varargin)
% A loss function inspired by : 
% Chopra, S., Hadsell, R., & LeCun, Y. (2005, June). Learning a similarity
% metric discriminatively, with application to face verification. In
% Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer
% Society Conference on (Vol. 1, pp. 539-546). IEEE.

% author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
% licence : MIT

Q = 1.5;
assert(isnumeric(O), 'Only numeric O is supported');
assert(islogical(Y), 'Y must be a boolean array');
nSamples = numel(O);
if isempty(varargin)
    C   = 0.5 / nSamples * ( sum(O(~Y) .^2) ...
                           + 2 * Q *sum(exp(- 2.77 / Q * O(Y))));
elseif strcmp(varargin{1}, 'each')
    C   = 0.5 * ( O .* ~Y .^2 ...
                + 2 * Q * exp(- 2.77 / Q * O) .* Y);
elseif strcmp(varargin{1}, 'gradient')
    C   = O .* ~Y ...
        - 2 * 2.77 * exp(- 2.77 / Q * O) .* Y;
else
    error('Unrecognized optional argument');
end
end
