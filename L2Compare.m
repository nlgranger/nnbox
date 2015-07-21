classdef L2Compare < handle & AbstractNet
    properties
        inSize;
    end
    
    methods
        function obj = L2Compare(inSize)
            assert(isnumeric(inSize), 'Only numerical input is supported');
            obj.inSize = inSize;
        end
        
        function S = insize(self)
            S = {self.inSize; self.inSize};
        end
        
        function S = outsize(~)
            S = 1;
        end
        
        function [Y, A] = compute(self, X)
            nSamples = size(X{1}, numel(self.inSize) + 1);
            X{1} = reshape(X{1}, [], nSamples);
            X{2} = reshape(X{2}, [], nSamples);
            Y    = sum((X{1} - X{2}) .^2, 1);
            if nargout > 1
                A = X{1} - X{2};
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            G = [];
            inErr = { bsxfun(@times, 2 * A, outErr), ...
                - bsxfun(@times, 2 * A, outErr)};
            inErr{1} = reshape(inErr{1}, [self.inSize numel(outErr)]);
            inErr{2} = reshape(inErr{2}, [self.inSize numel(outErr)]);
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
