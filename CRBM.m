classdef CRBM < handle & AbstractNet
    %CRBM Convolutional Restricted Boltzmann Machine model
    %   Implementation of AbstractNet for Convolutional Restricted Boltzmann
    %   Machines. The CRBM can be seen as a regular RBM with shared weight,
    %   but this property is due to the fact that a convolution is applied
    %   on the input to compute the stimulation of output neurons.
    %
    %   A pooling layer reduces the output dimension and removes
    %   sensibility to small translations on the input.
    %
    %   CRBM also features the possibility to use multiple filter which is
    %   equivalent to applying several CRBMS on the same input.
    %
    %   Note: This is a draft and it has not been tested nor reviewed.
    
    properties
        nFilters; % number of filters
        filterSz; % filters size
        filters;  % filter weights
        b;        % input units biases
        c;        % output units biases
        inSz;     % input size
        hidSz;    % hidden layer size for each filter
        outSz;    % output size for each filter
        poolSz;   % pooling area size
        
        pretrainOpts;
        trainOpts;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %

        function obj = CRBM(inSz, nFilters, filterSz, poolSz, pretrainOpts, ...
            trainOpts)
            % CRBM Construct a CRBM
            %   crbm = CRBM(inSz, nfilters, filterSz, poolSz)
            %
            %     nFilters -- number of filters
            %     filterSz -- filters size in a 2 element wide array
            %     inSz     -- input size as a 2 element wide array
            %     poolSz   -- pooling area size as a 2 element wide array
            %
            %   Note: initial weights are generated uniformly in [-.5, .5].
            obj.inSz     = inSz;
            obj.nFilters = nFilters;
            obj.filterSz = filterSz;
            obj.poolSz   = poolSz;
            
            if ~isfield(pretrainOpts, 'momentum')
                pretrainOpts.momentum = 0;
            end
            obj.pretrainOpts = pretrainOpts;
            obj.trainOpts    = trainOpts;
            
            obj.hidSz    = obj.inSz - obj.filterSz + 1;
            obj.outSz    = obj.hidSz ./ obj.poolSz;
            assert(~any(obj.outSz > floor(obj.outSz)), ...
                'Pool size must divide hidden layer''s size [%d %d]', ...
                obj.hidSz(1), obj.hidSz(2));
            
            obj.filters  = cell(nFilters, 1);
            obj.b        = 0; % common bias for input neurons
            obj.c        = zeros(obj.nFilters, 1); % common bias of filters
            for f = 1:nFilters
                obj.filters{f} = rand(obj.filterSz) - .5;
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = self.inSz;
        end
        
        function S = outSize(self)
            S = {self.outSz} * ones(self.nFilters, 1);
        end
        
        function [Y, A] = compute(self, X)
            if nargout > 1
                A.x = X;
                Y = self.vis2hid(X);
                [Y, A.p] = self.pool(Y);
                A.s = Y;
            else
                Y = self.pool(self.vis2hid(X));
            end
        end % compute(self, X)
        
        function [] = pretrain(self, X)
            nbSamples = size(X, 3);
            opts = self.pretrainOpts;
            
            for e = 1:opts.nEpoch
                
                % batches
                if ~isField(opts, 'batchSz')
                    opts.batchSz = nSamples;
                end
                shuffle = randperm(nbSamples);
                X       = X(:,:,shuffle);
                
                for i = 1:opts.batchSz:nbSamples
                    batch = X(:, :, i:min(i + opts.batchSz - 1, nSamples));
                    
                    % Gradient update
                    [dW, dc, db] = self.cd(batch);
                    for f = 1:self.nFilters
                        dW{f} = dWold{f} * opts.momentum ...
                            + dW{f} * (1 - opts.momentum);
                    end
                    db = dbold * opts.momentum + db * (1 - opts.momentum);
                    dc = dcold * opts.momentum + dc * (1 - opts.momentum);
                    
                    self.W = self.W - opts.lRate * dW;
                    self.c = self.c - opts.lRate * dc;
                    self.b = self.b - opts.lRate * db;
                    
                    % Weight decay
                    if isfield(opts, 'wPenalty') && e > opts.wDecayDelay
                        self.W = self.W - opts.lRate * opts.wDecayRate * self.W;
                        self.b = self.b - opts.lRate * opts.wDecayRate * self.b;
                        self.c = self.c - opts.lRate * opts.wDecayRate * self.c;
                    end
                    
                    % Gradient Backup
                    dWold = dW;
                    dbold = db;
                    dcold = dc;
                end
            end
        end % pretrain(self, X)
        
        function inErr = backprop(self, A, outErr, opts)
            % gradient
            for f = 1:self.nFilters
                delta     = outErr{f} .* A.s{f} .* (1 - A.s{f});
                dW        = conv2(rot90(A.x,2), delta);
                self.W{f} = self.W{f} - opts.lRate * dW;
                self.c(f) = self.c(f) - opts.lRate * sum(sum(outErr{f}));
            end
        end
        
        function [] = train(self, X, Y, opts)
            nSamples = size(X, 3);
            
            for i = 1:opts.nbIter
                shuffle = randperm(nSamples);
                
                for start = 1:opts.batchSz:nSamples % batch loop
                    batch = shuffle(start:min(start + opts.batchSz - 1, nSamples));
                    [O, A] = self.compute(X(:, batch)); % forward pass
                    E = O - Y; % L2 error derivative
                    for n = length(net):1 % backprop
                        E = self.nets{n}.backprop(A{n}, E, ...
                            self.backpropOpts{n});
                    end
                end
            end
        end % train(self, X, Y, opts)
        
        % Methods  ---------------------------------------------------------- %
        
        function H = vis2hid(self, X)
            nSamples = size(X, 3);
            mirFilters = cellfun(@(F) rot90(F,2), self.filters, ...
                'UniformOutput', false); % cache mirrored filters
            H = cell(1, self.nFilters);
            
            for f = 1:self.nFilters
                H{f} = zeros([self.hidSz nSamples]);
                for s = 1:nSamples
                    H{f}(:,:,s) = CRBM.activationFn( ...
                        conv2(mirFilters{f}, V, 'valid') ...
                        + self.c(f));
                end
            end
        end % vis2hid(self, X)
        
        function V = hid2vis(self, H)
            % TODO: fix borders
            nSamples = size(H{1}, 3);
            V = zeros([self.visSz, nSamples]);
            
            for f = 1:self.nFilters
                for s = 1:nSamples
                    V(:,:,s) = V(:,:,s) + conv2(self.Filters{f}, H{f}, 'full');
                end
            end
            V = crbm.activationFn(V + self.b);
        end % hid2vis(self, H)
        
        function [dW, dc, db, phid0] = cd(self, X)
            % CD Contrastive divergence
            %   [dW, dc, db] = cd(obj, X) returns the approximated
            %   contrastive divergence by using 1 Gibbs sampling iteration
            %   over training data X to approximate the model distribution
            %   (Hinton's CD1)
            
            nSamples = size(X, 3);
            dW = {zeros(size(self.filterSz))} * ones(self.nFilters, 1);
            dc = zeros(self.nFilters, 1);
            
            % Data expectation
            phid0 = self.vis2hid(X); % get hidden units
            for f = 1:self.nFilters
                mir   = rot90(phid0{f}, 2);
                for s = 1:nSamples
                    dW{f} = dW{f} - conv2(X(:,:,s), mir(:,:,s), 'valid');
                end
                dc(f) = - mean(sum(sum(phid0{f}, 1), 2)); % TODO: check this
            end
            db = - mean(sum(sum(X, 1), 2)); % TODO: check this too
            
            % Model expectation
            hid = cell(self.nFilters, 1);
            for f = 1:self.nFilters
                hid{f} = rand([seld.hidSz nSamples]) < phid0{f};
            end
            vis = self.hid2vis(hid);
            hid = self.vis2hid(vis);
            for f = 1:self.nFilters
                mir   = rot90(hid{f}, 2);
                for s = 1:nSamples
                    dW{f} = dW{f} + conv2(vis(:,:,s), mir(:,:,s), 'valid');
                end
                dc(f) = dc(f) + mean(sum(sum(hid{f}, 1), 2)); % TODO: check this
            end
            db = db + mean(sum(sum(X, 1), 2)); % TODO: check this too
            
            % Scale by the number of samples
            for f = 1:self.nFilters
                dW{f} = dW{f} / nSamples;
            end
        end % cd(self, X)
        
        function [P, I] = pool(self, H)
            if nargin > 1
                PI = blockproc(H, self.poolSz, @self.poolWithIdxFn);
                P = PI(:,:,1);
                I = PI(:,:,[2 3]);
            else
                P = blockproc(H, self.poolSz, @self.poolFn);
            end
        end
    end % methods
    
    methods(Static)
        
        % Sigmoïd activation function
        function H = activationFn(S)
            H = 1 ./ (1 + exp(-S));
        end
        
        % Max-pooling subsampling function
        function p = poolFn(B)
            p = max(max(B));
        end
        
        % Max-pooling subsampling function with inversion information
        function pi = poolWithIdxFn(B)
            [m, i1] = max(B.data);
            [m, i2] = max(m);
            pi = zeros(1,1,3);
            pi(1,1,1) = m;
            pi(1,1,2) = i1(i2) + B.location(1) - 1;
            pi(1,1,3) = i2 + B.location(2) - 1;
        end
    end % methods(Static)
    
end

