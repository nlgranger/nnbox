classdef PatchNet < handle & AbstractNet
    
    properties
        inSz;    % 2D input size
        patchSz; % patch size
        overlap; % # of overlapping pixel between patches
        nPatches % number of patches vertically and horizontally
    end % properties
    
    methods
        
        % Constructor *********************************************************
        
        function obj = PatchNet(inSz, patchSz, overlap)
            obj.inSz   = inSz;
            obj.patchSz = patchSz;
            obj.overlap = overlap;
            obj.nPatches = floor((inSz - patchSz) ./ (patchSz - overlap)) + 1;
        end
        
        % AbstractNet implementation ******************************************
        
        function S = insize(self)
            S = self.inSz;
        end
        
        function S = outsize(self)
            S = cell(prod(self.nPatches), 1);
            S(:) = {self.patchSz};
        end
        
        function [Y, A] = compute(self, X)
            Y = cell(prod(self.nPatches), 1);
            for i = 1:self.nPatches(1)
                for j = 1:self.nPatches(2)
                    voff = (i-1) * (self.patchSz(1) - self.overlap(1)) + 1;
                    hoff = (j-1) * (self.patchSz(2) - self.overlap(2)) + 1;
                    Y{(j-1) * self.nPatches(1) + i} = ...
                        reshape(X(voff:voff + self.patchSz(1) - 1, ...
                                hoff:hoff + self.patchSz(2) - 1, :), ...
                                prod(self.patchSz), size(X, 3));
                end
            end
            A = [];
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, ~, ~)
            G = [];
            inErr = [];
            % TODO: implement this
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
        function [] = train(~, ~, ~)
            error('PatchNet has nothing to train.');
        end
        
    end % methods
    
end % classdef RBM