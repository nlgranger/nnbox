classdef RELURBM < handle & AbstractNet
    % RELURBM Restricted Boltzmann Machine model
    %   RBM implements AbstractNet for Restricted Boltzmann Machines with
    %   rectified linear units on both visible and hidden units.
    %
    %   Pretraining regularization includes L2 and L1 weight decay, dropout
    %   and hidden units sparsity.
    %
    %   author: Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    
    properties
        nVis;                  % # of visible units (dimensions)
        nHid;                  % # of hidden units
        W;                     % connection weights
        b;                     % visible unit biases
        c;                     % hidden unit biases
        hasHidBias = true;
        
        pretrainOpts = struct();
        trainOpts    = struct();
    end % properties
    
    methods
        
        % Constructor *********************************************************
        
        function obj = RELURBM(nVis, nHid, pretrainOpts, trainOpts, varargin)
            obj.nVis         = nVis;
            obj.nHid         = nHid;
            
            if ~isfield(pretrainOpts, 'dropVis')
                pretrainOpts.dropVis = 0;
            end
            if ~isfield(pretrainOpts, 'dropout')
                pretrainOpts.dropout = 0;
            end
            if ~isfield(pretrainOpts, 'momentum')
                pretrainOpts.momentum = 0;
            end
            obj.pretrainOpts = pretrainOpts;
            
            if ~isfield(trainOpts, 'decayNorm')
                trainOpts.decayNorm = -1;
            end
            obj.trainOpts    = trainOpts;
            
            % Initializing weights
            obj.W = abs(randn(nVis, nHid)/(2*nVis));
            obj.b = ones(nVis, 1) * 1/(20*nVis);
            obj.c = ones(nHid, 1) * 1/(20*nVis);
            
            if ~isempty(varargin) && varargin{1} == false
                obj.hasHidBias = false;
                obj.c = 0;
            end
        end
        
        % AbstractNet implementation ******************************************
        
        function S = insize(self)
            S = self.nVis;
        end
        
        function S = outsize(self)
            S = self.nHid;
        end
        
        function [Y, A] = compute(self, X)
            Y = max(0, bsxfun(@plus, (X' * self.W)', self.c));
            if nargout > 1
                A.x = X;
                A.ds = Y>0;
            end
        end
        
        function [] = pretrain(self, X)
            nObs = size(X, 2);
            opts = self.pretrainOpts;
            dWold   = zeros(size(self.W));
            dbold   = zeros(size(self.b));
            dcold   = zeros(size(self.c));
            
            for e = 1:opts.nEpochs
                shuffle  = randperm(nObs);
                
                % Batch loop
                for batchBeg = 1:opts.batchSz:nObs
                    bind  = shuffle(batchBeg : min(nObs, ...
                        batchBeg + opts.batchSz -1));
                    
                    % Gibbs sampling
                    [dW, db, dc] = self.cd(X(:,bind));
                    
                    % Weight decay
                    if isfield(opts, 'wPenalty')
                        dW = dW + opts.wPenalty * sign(self.W);
                        db = db + opts.wPenalty * sign(self.b);
                        dc = dc + opts.wPenalty * sign(self.c);
                    end
                    
                    % Momentum
                    dW = dWold * opts.momentum + (1 - opts.momentum) * dW;
                    db = dbold * opts.momentum + (1 - opts.momentum) * db;
                    dc = dcold * opts.momentum + (1 - opts.momentum) * dc;
                    
                    % Apply gradient
                    self.W = self.W - opts.lRate * dW;
                    self.b = self.b - opts.lRate * db;
                    if self.hasHidBias
                        self.c = self.c - opts.lRate * dc;
                    end
                    
                    % Save gradient
                    dWold = dW;
                    dbold = db;
                    dcold = dc;
                end
                
                % Report
                if isfield(opts, 'displayEvery') ...
                        && mod(e, opts.displayEvery) == 0
                    % Reconstruct input samples
                    R = self.compute(X);
                    R = max(0, bsxfun(@plus, self.W * R, self.b));
                    % Mean square reconstruction error
                    msre = mean(sqrt(mean((R - X) .^2)), 2);
                    fprintf('%03d , msre = %f\n', e, msre);
                end
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            % backprop implementation of AbstractNet.backprop
            %   inErr = backprop(self, A, outErr, opts)
            %
            %   A      -- forward pass data as return by compute outErr --
            %   network output error derivative w.r.t. output
            %             neurons stimulation
            %
            %   inErr  -- network output error derivative w.r.t. neurons
            %             outputs (not activation)
            nSamples = size(outErr, 2);
            
            % Gradient computation
            delta  = outErr .* A.ds;
            G.dW   = A.x * delta' / nSamples;
            G.dc   = mean(delta, 2);
            
            % Error backpropagation
            inErr = self.W * delta;
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.W = self.W - opts.lRate * G.dW;
            self.c = self.c - opts.lRate * G.dc;
            
            % Weight decay
            if opts.decayNorm == 2
                self.W = self.W - opts.lRate * opts.decayRate * self.W;
                self.c = self.c - opts.lRate * opts.decayRate * self.c;
            elseif opts.decayNorm == 1
                self.W = self.W - opts.lRate * opts.decayRate * sign(self.W);
                self.c = self.c - opts.lRate * opts.decayRate * sign(self.c);
            end
        end
        
        function [] = train(self, X, Y)
            nSamples = size(X, 2);
            opts     = self.trainOpts;
            
            if ~isfield(opts, 'batchSz')
                opts.batchSz = nSamples; % no batch is one batch
            end
            
            for i = 1:opts.nIter
                shuffle = randperm(nSamples);
                for start = 1:opts.batchSz:nSamples
                    batch = shuffle(start:min(start + opts.batchSz - 1, nSamples));
                    [O, A] = self.compute(X(:,batch));
                    E = O - Y(:, batch);
                    self.gradientupdate(self.backprop(A, E));
                end
            end
        end
        
        % Methods *************************************************************
        
        function [dW, db, dc] = cd(self, X)
            % CD Contrastive divergence (Hinton's CD(k))
            %   [dW, db, dc, act] = cd(self, X) returns the gradients of
            %   the weihgts, visible and hidden biases using Hinton's
            %   approximated CD. The sum of the average hidden units
            %   activity is returned in act as well.
            opts = self.pretrainOpts;
            
            nObs = size(X, 2);
            
            % Forward pass
            vis0 = X;
            
            if opts.dropVis > 0 % Visible units dropout
                mask = rand(size(X)) < opts.dropVis;
                X = X .* mask / (1 - opts.dropVis);
            end
            
            act  = bsxfun(@plus, (X' * self.W)', self.c);
            hid0 = max(0, act);
            
            % Gibbs sampling
            hid  = max(0, act + randn(self.nHid, nObs) .* sqrt(1./(1+exp(-act))));
            act  = bsxfun(@plus, self.W * hid, self.b);
            if opts.dropout > 0
                mask = rand(size(act)) < opts.dropout;
                act = act .* mask / (1 - opts.dropout);
            end
            vis  = max(0, act);
            hid  = max(0, bsxfun(@plus, (vis' * self.W)', self.c));
            
            % Contrastive divergence
            dW = (vis0 * hid0' - vis * hid') / nObs;
            if self.hasHidBias
                dc = (sum(hid0, 2) - sum(hid, 2)) / nObs;
            else
                dc = zeros(self.nHid, 1);
            end
            db = (sum(vis0, 2) - sum(vis, 2)) / nObs;
        end % cd(self, X)
        
    end % methods
    
end % classdef RBM