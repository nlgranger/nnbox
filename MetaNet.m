classdef MetaNet < handle & AbstractNet
    %METANET NN Horizontal concatenation of NN
    %   MetaNet combines several networks in parallel to make one
    %   articicial AbstractNet instance. Inputs and outputs are separate
    %   groups of neurons corresponding to each subnetwork.
    
    properties
        nets         = {}; % subnetworks
    end
    
    methods
        
        % AbstractNet Implementation ******************************************
        
        function S = insize(self)
            S = cellfun(@(n) n.insize(), self.nets, 'UniformOutput', false);
        end
        
        function S = outsize(self)
            S = cellfun(@(n) n.outsize(), self.nets, 'UniformOutput', false);
        end
        
        function [Y, A] = compute(self, X)
            nbNets = length(self.nets);
            Y = cell(nbNets, 1);
            if nargout > 1
                A = cell(nbNets, 1);
            end
            if ~iscell(X)
                X = {X};
            end
            
            for o = 1:nbNets
                if nargout > 1
                    [Y{o}, A{o}] = self.nets{o}.compute(X{o});
                else
                    Y{o} = self.nets{o}.compute(X{o});
                end
            end
            
            if nbNets == 1
                Y = Y{1};
                if nargout > 1
                    A = A{1};
                end
            end
        end % compute(self, X)
        
        function [] = pretrain(self, X)
            if ~iscell(X)
                X = {X};
            end
            for o = 1:length(self.nets)
                self.nets{o}.pretrain(X{o});
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            G     = cell(length(self.nets), 1);
            inErr = cell(length(self.nets), 1);
            for n = 1:length(self.nets)
                [G{n}, inErr{n}] = self.nets{n}.backprop(A{n}, outErr{n});
            end
        end
        
        function [] = gradientupdate(self, G)
            for n = 1:length(self.nets)
                self.nets{n}.gradientupdate(G{n});
            end
        end
        
        function train(~, ~)
            % TODO: implement this
            error('Not implemented');
        end
        
        % Methods *************************************************************
        
        function [] = add(self, net)
            % addNetwork Add new network layer
            %   addNetwork(obj, net) append net below existing networks.
            assert(isa(net, 'AbstractNet'), 'net must implement AbstractNet');
            
            nbNets              = length(self.nets);
            self.nets{nbNets+1} = net;
        end % add(self, net)
        
    end % methods
end % MetaNet