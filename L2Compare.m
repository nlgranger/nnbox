classdef L2Compare < handle & AbstractNet
    properties
        inSize;
    end
    
    methods
        function obj = L2Compare(inSize)
            obj.inSize = inSize;
        end
        
        function S = insize(self)
            S = {self.inSize; self.inSize};
        end
        
        function S = outsize(~)
            S = 1;
        end
        
        function [Y, A] = compute(~, X)
            Y = .5 * sum((X{1} - X{2}).^2);
            if nargout > 1
                A.X = X;
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, A, outErr)
            G = [];
            inErr = { bsxfun(@times, A.X{1} - A.X{2}, outErr); ...
                      bsxfun(@times, A.X{2} - A.X{1}, outErr)};
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
        function [] = train(~, ~)
            error('There is nothing to train on a Cosine distance.')
        end
        
    end
end