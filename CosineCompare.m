classdef CosineCompare < handle & AbstractNet
    properties
        inSize;
    end
    
    methods
        function obj = CosineCompare(inSize)
            obj.inSize = inSize;
        end
        
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
            G = [];
            inErr    = cell(2, 1);
            normX1   = sum(A.X1 .^ 2);
            normX2   = sum(A.X2 .^ 2);
            inErr{1} = bsxfun(@times, ...
                bsxfun(@times, A.X1, A.Y ./ normX1) ...
                - bsxfun(@rdivide, A.X2, sqrt(normX1 .* normX2)), ...
                outErr);
            inErr{2} = bsxfun(@times, ...
                bsxfun(@times, A.X2, A.Y ./ normX2) ...
                - bsxfun(@rdivide, A.X1, sqrt(normX1 .* normX2)), ...
                outErr);
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
