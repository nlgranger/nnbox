classdef ReshapeNet < handle & AbstractNet
    % RESHAPENET implements AbstractNet for a dummy network which simply
    % reshape its input
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        inSz;
        outSz;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = ReshapeNet(in, out)
            % obj = RESHAPENET(in, out) returns an instance of RESHAPENET
            % which reformats an input of size in to an output of size out.
            %
            % obj = RESHAPENET(N1, N2) returns an instance of RESHAPENET
            % which reformats the output of AbstractNet implementation N1 to 
            % feed N2
            
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

        % AbstractNet implementation ---------------------------------------- %
        
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
            N = size(X{1}, numel(self.inSz{1}) + 1); % number of samples
            
            for g = 1:length(X) % one line for each sample
                X{g} = reshape(X{g}, [], N)';
            end
            X = cell2mat(X);
            Y = mat2cell(X, N, cellfun(@prod, self.outSz))';
            for g = 1:length(self.outSz) % split output groups and reshape
                Y{g} = reshape(Y{g}', [self.outSz{g} N]);
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
        
    end % methods
end