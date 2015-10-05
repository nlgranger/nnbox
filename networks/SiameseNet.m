classdef SiameseNet < AbstractNet & handle
    % SIAMESENET implements the siamese network pattern of network
    % replication and weight sharing as an AbstractNet
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        nNets;        % Number of replications
        net;          % actual network instance
        pretrainOpts; % supervized training options
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = SiameseNet(net, n, varargin)
            % obj = SIAMESENET(net, N) returns an instance of SIAMESENET
            % with N copies of the neural network net
            %
            % obj = SIAMESENET(net, N, 'skipPretrain') makes the network
            % ignore pretraining requests
            
            assert(isa(net, 'AbstractNet'), 'net must implement AbstractNet');
            assert(~iscell(net.outsize()) && ~iscell(net.insize()), ...
                'Grouped input or output not supported');
            if n == 1, warning('SiameseNet is useless with n = 1'); end
            
            obj.nNets = n;
            obj.net = net.copy();
            obj.pretrainOpts.skip = false;
            
            if ~isempty(varargin) && strcmp(varargin{1}, 'skipPretrain')
                obj.pretrainOpts.skip = true;
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
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
    end
    
    methods(Access = protected)
        
        % Copyable implementation ------------------------------------------- %
        
        % Override copyElement method
        function copy = copyElement(self)
            if self.pretrainOpts.skip
                copy = SiameseNet(self.net.copy(), self.nNets, 'skipPretrain');
            else
                copy = SiameseNet(self.net.copy(), self.nNets);
            end
        end
        
    end
    
end

