classdef SiameseNet < AbstractNet & handle
    %UNTITLED Compute outputs for the same network on different inputs
    
    properties
        nNets;        % Number of replications
        net;          % actual network instance
        pretrainOpts; % supervized training options
    end
    
    methods
        function obj = SiameseNet(net, n, varargin)
            assert(~iscell(net.outsize()) && ~iscell(net.insize()), ...
                'Grouped input or output not supported');
            if n == 1, warning('CompareNet is useless with n = 1'); end
            obj.nNets = n;
            obj.net = net.copy();
            obj.pretrainOpts.skip = false;
            
            if ~isempty(varargin) && strcmp(varargin{1}, 'skipPretrain')
                obj.pretrainOpts.skip = true;
            end
        end
        
        % AbstractNet implementation ******************************************
        
        function s = insize(self)
            s    = cell(self.nNets, 1);
            s(:) = {self.net.insize()};
        end
        
        function s = outsize(self)
            s    = cell(self.nNets, 1);
            s(:) = {self.net.outsize()};
        end
        
        function [Y, A] = compute(self, X)
            if nargout == 1
                Y = cell(self.nNets, 1);
                for i = 1:self.nNets
                    Y{i} = self.net.compute(X{i});
                end
            else
                Y = cell(self.nNets, 1);
                A = cell(self.nNets, 1);
                for i = 1:self.nNets
                    [Y{i}, A{i}] = self.net.compute(X{i});
                end
            end
        end
        
        function [] = pretrain(self, X)
            if ~self.pretrainOpts.skip
                X = cell2mat(reshape(X, 1, numel(X)));
                %TODO support multidimensional input
                self.net.pretrain(X);
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            G     = cell(self.nNets, 1);
            inErr = cell(self.nNets, 1);
            for c = 1:self.nNets
                [G{c}, inErr{c}] = self.net.backprop(A{c}, outErr{c});
            end
        end
        
        function gradientupdate(self, G)
            for c = 1:self.nNets
                self.net.gradientupdate(G{c});
            end
        end
        
        function [] = train(~, ~, ~)
            % TODO implement this
            error('Not implemented');
        end
    end
    
end

