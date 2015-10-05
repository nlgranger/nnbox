classdef CosineCompare < handle & AbstractNet
    % An attempt to build a metric between representation vectors, seems 
    % unstable numerically with cancellation issues.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        inSize;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = CosineCompare(inSize)
            obj.inSize = inSize;
        end
        
        % AbstractNet implementation----------------------------------------- %
        
        function S = insize(self)
            S = {self.inSize; self.inSize};
        end
        
        function S = outsize(~)
            S = 1;
        end
        
        function [Y, A] = compute(~, X)
            Y = 1 - sum(X{1} .* X{2}) ./ sqrt(sum(X{1} .^ 2) .* sum(X{2} .^2));
            if nargout > 1
                A.X2 = X{2};
                A.X1 = X{1};
                A.Y  = Y;
            end
            
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, A, outErr)
            G        = [];
            inErr    = cell(2, 1);
            normX1   = sqrt(sum(A.X1 .^ 2));
            normX2   = sqrt(sum(A.X2 .^ 2));
            coeff    = outErr ./ (normX1 .* normX2);
            inErr{1} = - bsxfun(@times, ...
                A.X2 - bsxfun(@times, A.X1, A.Y ./ normX1), ...
                coeff);
            inErr{2} = - bsxfun(@times, ...
                A.X1 - bsxfun(@times, A.X2, A.Y ./ normX2), ...
                coeff);
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
