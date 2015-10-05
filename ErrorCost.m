% Error Cost function interface

% author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
% licence : Public Domain

classdef ErrorCost
    % Interface for error cost functions
    
    methods (Abstract)
        
        % COMPUTE mean error cost
        %   C = COMPUTE(O, Y) returns the total cost for a set of outputs O
        %   and the expected target values Y. Implementations may decide to
        %   use the total or mean value of the error according to common
        %   usage.
        C = compute(self, O, Y)
        
        % COMPUTEEACH sample-wise error
        %   C = COMPUTEEACH(O, Y) returns the error cost for each sample
        %   using outputs O and labels Y
        C = computeEach(self, O, Y)
        
        % GRADIENT Error cost derivative
        %   C = COMPUTEEACH(O, Y) returns the derivative of the error cost
        %   for each pair of sample output and label from O and Y
        %   respectively.
        G = gradient(self, O, Y)
        
    end
    
end