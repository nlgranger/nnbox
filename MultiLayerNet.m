classdef MultiLayerNet < handle & AbstractNet
    % MultiLayerNet Stack of neural networks
    %   Stores stacked network with interconnected inputs and outputs
    
    properties
        nets = {};
        trainOpts;
    end
    
    methods
        % Constructor *********************************************************
        
        function obj = MultiLayerNet(trainOpts)
        % mln = MULTILAYERNET(trainOpts) returns an empty multilayer neural 
        % network. Layers can be pushed from the bottom using the add() method.
        % trainOpts is a structure of supervized training settings:
        %   skipBelow    -- training skips layers up to this value [optional]
        %   displayEvery -- error cost value display rate
        %   
            if ~isfield(trainOpts, 'skipBelow')
                trainOpts.skipBelow = 0;
            end
            obj.trainOpts = trainOpts;
        end
        
        % AbstractNet Implementation ******************************************
        
        function S = insize(self)
            S = self.nets{1}.insize();
        end
        
        function S = outsize(self)
            S = self.nets{end}.outsize();
        end
        
        function [Y, A] = compute(self, X)
            nbNets = length(self.nets);
            
            if nargout > 1
                computeA = true;
                A = cell(nbNets, 1);
            else
                computeA = false;
            end
            
            for o = 1:nbNets
                if computeA
                    [X, A{o}] = self.nets{o}.compute(X);
                else
                    X = self.nets{o}.compute(X);
                end
            end
            Y = X;
        end % compute(X)
        
        function [] = pretrain(self, X)
            for o = 1:length(self.nets)
                self.nets{o}.pretrain(X);
                X = self.nets{o}.compute(X);
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            opts = self.trainOpts;
            G = cell(length(self.nets), 1);
            for l = length(self.nets):-1:opts.skipBelow + 1 % backprop
                [G{l}, outErr] = self.nets{l}.backprop(A{l}, outErr);
            end
            inErr = outErr;
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            for l = length(self.nets):-1:opts.skipBelow + 1
                self.nets{l}.gradientupdate(G{l});
            end
        end
        
        function [] = train(self, X, Y)
            assert(ismatrix(Y), 'only one dimensional output supported');
            opts = self.trainOpts;
            if iscell(X)
                nSamples = size(X{1}, ndims(X{1}));
            else
                nSamples = size(X, ndims(X));
            end
            
            for i = 1:opts.nIter
                shuffle = randperm(nSamples);
                
                for start = 1:opts.batchSz:nSamples % batch loop
                    idx = shuffle(start:min(start+opts.batchSz-1, nSamples));
                    if iscell(X)
                        batchX = cell(length(X), 1);
                        for g = 1:length(X)
                            batchX{g} = X{g}(:,:, idx);
                        end
                    else
                        batchX = X(:, :, idx);
                    end
                    
                    [O, A] = self.compute(batchX); % forward pass
                    E = O - Y(idx); % L2 error derivative
                    for l = length(self.nets):-1:opts.skipBelow + 1 % backprop
                        [G, E] = self.nets{l}.backprop(A{l}, E);
                        self.nets{l}.gradientupdate(G);
                    end
                end
                
                % Report
                if isfield(opts, 'displayEvery') ...
                        && mod(i, opts.displayEvery) == 0
                    % Reconstruct input samples
                    O = self.compute(X); % forward pass
                    % Mean square reconstruction error
                    msre = mean(sqrt(mean((O - Y) .^2)), 2);
                    fprintf('%03d , msre = %f\n', i, msre);
                end
            end
        end % train(self, X, Y, opts)
        
        % Methods *************************************************************
        
        function [] = add(self, net)
            % ADD Add a new network on top of the existing ones
            %   [] = addNetwork(self, net) add net, an implementation of 
            %   AbstractNet on top of the networks currently in self. Note that
            %   the input size of net must match the current output size of 
            %   the multilayer network.
            assert(isa(net, 'AbstractNet'), 'net must implement AbstractNet');
            
            nbNets                    = length(self.nets) + 1;
            self.nets{nbNets}         = net.copy();
        end % add(self, net)
        
    end % methods
    
    methods(Access = protected)
        
        % Override copyElement method
        function copy = copyElement(self)
            copy = MultiLayerNet(self.trainOpts);
            copy.nets = cell(size(self.nets));
            for i = 1:numel(self.nets)
                copy.nets{i} = self.nets{i}.copy();
            end
        end
        
    end
    
end % MultiLayerNet

