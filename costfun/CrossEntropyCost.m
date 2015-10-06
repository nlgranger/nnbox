classdef CrossEntropyCost < ErrorCost
    % CROSSENTROPYCOST Cross entropy error cost function
    %   CROSSENTROPYCOST implements ErrorCost for the cross entropy error
    %   cost function.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    methods
        
        function C = compute(~, O, Y)
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Y must be a boolean array');
            C     = zeros(size(O));
            C(Y)  = - log(O(Y));
            C(~Y) = - log(1 - O(~Y));
            C     = mean(sum(C, 1));
        end
        
        function C = computeEach(~, O, Y)
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Y must be a boolean array');
            C     = zeros(size(O));
            C(Y)  = - log(O(Y));
            C(~Y) = - log(1 - O(~Y));
            C     = sum(C, 1);
        end
        
        function C = gradient(~, O, Y)
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Y must be a boolean array');
            C     = zeros(size(O));
            C(Y)  = - 1 ./ O(Y);
            C(~Y) = 1 ./ (1 - O(~Y));
        end
        
    end
    
end