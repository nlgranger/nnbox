classdef SquareCost < ErrorCost
    % SQUARECOST Square error cost function
    % SQUARECOST implements ErrorCost for the square error cost function.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    methods
        
        function C = compute(~, O, Y)
            assert(isnumeric(Y), 'Only numeric O and Y are supported');
            
            nSamples = size(O, ndims(O));
            C        = mean(sum(reshape((O - Y) .^ 2, [], nSamples), 1));
        end
        
        function C = computeEach(~, O, Y)
            assert(isnumeric(Y), 'Only numeric O and Y are supported');
            C = sum(reshape((O - Y) .^ 2, [], nSamples), 1);
        end
        
        function C = gradient(~, O, Y)
            assert(isnumeric(Y), 'Only numeric O and Y are supported');
            C = (O - Y);
        end
    end
    
end