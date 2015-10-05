classdef ExpCost < ErrorCost
    % EXPCOST Exponential loss function
    %   EXPCOST implements ErrorCost for an experimental cost function
    %   between binary labels and continuous outputs.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        thres;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = ExpCost(thres)
            % obj = EXPCOST(T) returns an instance of CONTRASTIVELOSS with
            % T the target threshold between the two classes.
            obj.thres = thres;
        end
        
        % ErrorCost implementation ------------------------------------------ %
        
        function C = compute(self, O, Y)
            assert(isvector(O) && isvector(Y), 'One dimensional input expected');
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Only binary Y is supported');
            C = mean(exp(- 4 * (O - self.thres) .* (Y - self.thres)));
        end
        
        function C = computeEach(self, O, Y)
            assert(isvector(O) && isvector(Y), 'One dimensional input expected');
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Only binary Y is supported');
            C = exp(- 4 * (O - self.thres) .* (Y - self.thres));
        end
        
        function C = gradient(self, O, Y)
            assert(isvector(O) && isvector(Y), 'One dimensional input expected');
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Only binary Y is supported');
            C = -4 * (Y - self.thres) ...
                .* exp(- 4 * (O - self.thres) .* (Y - self.thres));
        end
        
    end
end