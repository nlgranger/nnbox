classdef ContrastiveLoss < ErrorCost
    % CONTRASTIVELOSS Contrastive loss function
    %   CONTRASTIVELOSS implements ErrorCost for the contrastive loss for
    %   binary labels and continuous outputs. This cost function is ispired
    %   by: Chopra, S., Hadsell, R., & LeCun, Y. (2005, June). Learning a
    %   similarity metric discriminatively, with application to face
    %   verification. In Computer Vision and Pattern Recognition, 2005.
    %   CVPR 2005. IEEE Computer Society Conference on (Vol. 1, pp.
    %   539-546). IEEE.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        Q;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = ContrastiveLoss(Q)
            % obj = CONTRASTIVELOSS(Q) returns an instance of
            % CONTRASTIVELOSS with Q a parameters which balances the cost
            % between the two classes.
            obj.Q = Q;
        end
        
        % ErrorCost implementation ------------------------------------------ %
        
        function C = compute(self, O, Y)
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Y must be a boolean array');
            nSamples = numel(O);
            C   = 0.5 / nSamples * ( sum(O(~Y) .^2) ...
                + 2 * self.Q *sum(exp(- 2.77 / self.Q * O(Y))));
        end
        
        function C = computeEach(self, O, Y)
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Y must be a boolean array');
            C   = 0.5 * ( O .* ~Y .^2 ...
                + 2 * self.Q * exp(- 2.77 / self.Q * O) .* Y);
        end
        
        function C = gradient(self, O, Y)
            assert(isnumeric(O), 'Only numeric O is supported');
            assert(islogical(Y), 'Y must be a boolean array');
            C   = O .* ~Y ...
                - 2 * 2.77 * exp(- 2.77 / self.Q * O) .* Y;
        end
        
    end
end