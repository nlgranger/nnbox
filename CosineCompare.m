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
            if nargout > 1
                A.prod  = X{1} .* X{2};
                A.normX1 = sqrt(sum(X{1} .^ 2));
                A.normX2 = sqrt(sum(X{2} .^ 2));
                Y = sum(A.prod) ./ (A.normX1 .* A.normX2);
            else
                Y = sum(X{1} .* X{2}) ./ sqrt(sum(X{1} .^ 2) .* sum(X{2} .^2));
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, A, outErr)
            G = [];
            inErr = cell(2, 1);
            common = bsxfun(@times, A.prod, outErr ./ (A.normX1 .* A.normX2));
            inErr{1} = bsxfun(@rdivide, common, A.normX1 .^2);
            inErr{2} = bsxfun(@rdivide, common, A.normX2 .^2);
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
        function [] = train(~, ~)
            error('There is nothing to train on a Cosine distance.')
        end
        
    end
end