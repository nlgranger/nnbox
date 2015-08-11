classdef CNN < handle & AbstractNet
    
    properties
        nFilters;  % # of filters
        nChannels; % # of channels
        fSz;       % filters dimensions
        stride;    % Stride
        inSz;      % input image size
        poolSz;    % pool size
        
        trainOpts; % training options
        
        filters;   % filters weights
        b;         % biases
    end
    
    methods
        
        % Constructor *********************************************************
        
        function obj = CNN(inSz, filterSz, nFilters, trainOpts, varargin)
            if numel(inSz) == 2
                inSz(3) = 1;
            end
            
            obj.nFilters  = nFilters;
            obj.nChannels = inSz(3);
            obj.stride    = [];
            obj.fSz       = filterSz;
            obj.inSz      = inSz;
            obj.poolSz    = [];
            obj.trainOpts = trainOpts;
            wRange        = 1/sqrt(prod(filterSz) * inSz(3));
            obj.filters   = ...
                rand([filterSz inSz(3) nFilters], 'single') * wRange;
            obj.b         = ones(nFilters, 1, 'single') * wRange / 4;
            
            assert(mod(numel(varargin), 2) == 0, ...
                'options should be ''option'', values pairs');
            for o = 1:2:numel(varargin)
                if strcmp(varargin{o}, 'bias') && ~varargin{o+1} % no bias
                    obj.b = [];
                elseif strcmp(varargin{o}, 'stride') % stride
                    assert(numel(varargin{o+1} == 2), ...
                        'stride should have two values');
                    obj.stride = reshape(varargin{o+1}, 1, 2);
                elseif strcmp(varargin{o}, 'pool')
                    assert(numel(varargin{o+1}) == 2, ...
                        'pool size should have two values');
                    obj.poolSz = reshape(varargin{o+1}, 1, 2);
                end
            end
        end
        
        % AbstractNet implementation ******************************************
        
        function S = insize(self)
            S = [self.inSz self.nChannels];
        end
        
        function S = outsize(self)
            if isempty(self.stride)
                S = self.inSz(1:2) - self.fSz + 1;
            else
                S = floor((self.inSz(1:2) - self.fSz) ./ self.stride) + 1;
            end
            if ~isempty(self.poolSz)
                S = S ./ self.poolSz;
            end
            S = [S self.nFilters];
        end
        
        function [Y, A] = compute(self, X)
            assert(isa(X, 'single'), 'only single precision input supported');
            
            % Convolution
            X = reshape(X, size(X, 1), size(X,2), self.nChannels, []);
            if ~isempty(self.stride)
                Y = vl_nnconv(X, self.filters, self.b, 'Stride', self.stride);
            else
                Y = vl_nnconv(X, self.filters, self.b);
            end
            
            % Dropout
            if nargout > 1 && isfield(self.trainOpts, 'dropout')
                rate = self.trainOpts.dropout;
                A.mask = rand([self.inSz(1:2) - self.fSz(1:2) + 1, ...
                               self.nFilters]) > rate;
                Y = bsxfun(@times, Y, single(A.mask / (1-rate)));
            end
            
            % Max-pooling
            if ~isempty(self.poolSz)
                if nargin > 1
                    A.Y = Y;
                end
                Y = vl_nnpool(Y, self.poolSz, 'Stride', self.poolSz, ...
                	'Method', 'max');
            end
            
            % Rectification
            Y = max(0, Y);
            
            % Save values for backprop
            if nargin > 1
                A.X = X;
            end
        end
        
        function [] = pretrain(~, ~)
            % Not implemented
            warning('Pretraining not implemented for CNN');
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            % Unpool and rectification derivation
            if ~isempty(self.poolSz)
                outErr = vl_nnpool(A.Y, self.poolSz, outErr, ...
                    'Stride', self.poolSz) .* (A.Y > 0);
            end
            
            % Dropout
            if isfield(self.trainOpts, 'dropout')
                outErr = bsxfun(@times, outErr, A.mask);
            end
            
            % Backprop
            if ~isempty(self.stride)
                [inErr, G.dW, G.db] = vl_nnconv(A.X, self.filters, self.b, ...
                    outErr, 'Stride', self.stride);
            else
                [inErr, G.dW, G.db] = vl_nnconv(A.X, self.filters, self.b, ...
                    outErr);
            end
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            
            % Gradient update
            self.filters = self.filters + opts.lRate * G.dW;
            if ~isempty(self.b)
                self.b = self.b + opts.lRate * G.db;
            end
            
            % Weight decay
            if isfield(opts, 'decayNorm') && opts.decayNorm == 2
                self.W = self.W - opts.lRate * opts.decayRate * self.W;
                self.b = self.b - opts.lRate * opts.decayRate * self.b;
            elseif isfield(opts, 'decayNorm') && opts.decayNorm == 1
                self.W = self.W - opts.lRate * opts.decayRate * sign(self.W);
                self.b = self.b - opts.lRate * opts.decayRate * sign(self.b);
            end
        end
    end
    
end