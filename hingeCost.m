function C = hingeCost(O, Y, varargin)
% Draft, not supported or tested

% author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
% licence : MIT

assert(isnumeric(O) && isvector(O) && isnumeric(Y) && isvector(Y), ...
    'Only numeric one dimensional output is supported');

if isempty(varargin)
    C = mean(.25 * max(0, 1 - O .* Y) .^2);
elseif strcmp(varargin{1}, 'each')
    C = .25 * max(0, 1 - O .* Y) .^2;
elseif strcmp(varargin{1}, 'gradient');
    C = 1 - Y .* O;
    C = - 0.5 * Y .* C .* (C>0);
else
    error('Unrecognized optional argument');
end
end
