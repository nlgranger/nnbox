classdef ReshapeNet < handle & AbstractNet
    properties
        inSz;
        outSz;
    end
    
    methods
        
        % Constructor *********************************************************
        
        function obj = ReshapeNet(in, out)
            if isa(in, 'AbstractNet')
                in = in.outsize();
            end
            if isa(out, 'AbstractNet')
                out = out.insize();
            end
            
            if ~iscell(in)
                in = {in};
            end
            if ~iscell(out)
                out = {out};
            end
            assert(sum(cellfun(@prod, in)) == sum(cellfun(@prod, out)), ...
                'Incompatible sizes.');
            
            obj.inSz  = in;
            obj.outSz = out;
        end % ReshapeNet(in, out)

        % AbstractNet Implementation ******************************************
        
        function S = insize(self)
            if length(self.inSz) == 1
                S = self.inSz{1};
            else
                S = self.inSz;
            end
        end
        
        function S = outsize(self)
            if length(self.outSz) == 1
                S = self.outSz{1};
            else
                S = self.outSz;
            end                
        end
        
        function [Y, A] = compute(self, X)
            if ~iscell(X);
                X = {X};
            else
                X = reshape(X, 1, numel(X)); % horizontal X
            end
            extra  = size(X{1}, numel(self.inSz{1}) + 1); % number of samples
            
            for g = 1:length(X) % one line for each sample
                X{g} = reshape(X{g}, numel(X{g})/extra, extra)';
            end
            X = cell2mat(X);
            Y = mat2cell(X, extra, cellfun(@prod, self.outSz))';
            for g = 1:length(self.outSz) % split output groups and reshape
                Y{g} = reshape(Y{g}', [self.outSz{g} extra]);
            end
            
            if length(Y) == 1
                Y = Y{1};
            end
            A = [];
        end % compute(self, X)
        
        function [] = pretrain(~, ~)
        end
       
        function [G, inErr] = backprop(self, ~, outErr)
            G = []; % ReshapeNet doesn't have parameters
            
            if ~iscell(outErr);
                O = {outErr};
            else
                O = reshape(outErr, 1, numel(outErr)); % horizontal outErr
            end
            extra  = size(O{1}, numel(self.outSz{1}) + 1); % number of samples
            
            for g = 1:length(O) % one line for each sample
                O{g} = reshape(O{g}, numel(O{g})/extra, extra)';
            end
            O = cell2mat(O);
            inErr = mat2cell(O, extra, cellfun(@prod, self.inSz))';
            for g = 1:length(self.inSz) % split output groups and reshape
                inErr{g} = reshape(inErr{g}', [self.inSz{g} extra]);
            end
            
            if length(inErr) == 1
                inErr = inErr{1};
            end
        end % backprop(self, ~, outErr, ~)
        
        function [] = gradientupdate(~, ~)
            % Nothing to update
        end
        
        function [] = train(~, ~, ~)
            error('ReshapeNet has nothing to train');
        end
        
    end % methods
end