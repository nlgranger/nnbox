classdef L2Cost < AbstractCost
    methods
        function C = compute(self, net, X, Y, varargin)
            assert(isa(net, 'AbstractNet'), ...
                'net should implement AbstractNet');
            assert(isnumeric(Y), 'Only numeric output is supported');

            O = net.compute(X); % forward pass
            C = sum(reshape((O - Y) .^ 2, [], 1));
        end
        
        function trained = train(self, net, X, Y, opts)
            assert(isa(net, 'AbstractNet'), ...
                'net should implement AbstractNet');
            assert(isnumeric(Y), 'Only numeric output is supported');

            trained = net.copy();
            if iscell(X)
                nSamples = size(X{1}, ndims(X{1}));
            else
                nSamples = size(X, ndims(X));
            end
            Ycol = reshape(Y, [], nSamples);
            YSz  = size(Y);
            
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
                    batchY = reshape(Ycol(:,idx), ...
                        [YSz(1:end-1), opts.batchSz]);
                    
                    [O, A] = trained.compute(batchX); % forward pass
                    E = O - batchY; % L2 error derivative
                    G = trained.backprop(A, E);
                    trained.gradientupdate(G);
                end
                
                % Report
                if isfield(opts, 'displayEvery') ...
                        && mod(i, opts.displayEvery) == 0
                    MC = self.compute(net, X, Y) / nSamples;
                    fprintf('%03d , mean quadratic cost : %f\n', i, MC);
                end
            end
        end
    end
end