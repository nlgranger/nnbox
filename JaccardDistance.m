classdef JaccardDistance < handle & AbstractNet
    properties
        inSize;
        offset;
    end
    
    methods
        function obj = JaccardDistance(inSize, regularisation)
            obj.inSize = inSize;
            obj.offset = regularisation;
        end
        
        function S = insize(self)
            S = {self.inSize; self.inSize};
        end
        
        function S = outsize(~)
            S = 1;
        end
        
        function [Y, A] = compute(self, X)
            if nargout > 1
                m    = sum(min(X{1}, X{2}), 1) + self.offset;
                M    = sum(max(X{1}, X{2}), 1) + self.offset;
                Y    = m ./ M;
                t1   = 1 ./ M;
                t2   = - M .^ -2;
                t3   = X{1} < X{2};
                A    = {- bsxfun(@times, t3, t1) - bsxfun(@times, ~t3, t2), ...
                        - bsxfun(@times, ~t3, t1) - bsxfun(@times, t3, t2) };
                Y = 1 - Y;
            else
                Y = 1 - sum(min(X{1}, X{2}) + self.offset, 1) ...
                     ./ sum(max(X{1}, X{2}) + self.offset, 1);
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, A, outErr)
            G = [];
            inErr = { bsxfun(@times, A{1}, outErr), ...
                      bsxfun(@times, A{2}, outErr) };
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
