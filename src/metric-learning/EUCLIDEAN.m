% EUCLIDEAN.m 
% A trainableModel class for to wrap Euclidean distance as one Metric Learning
% There are no parameters that need to be trained here, but it is wrapped around a trainable model
% to ease comaparisons with trainableModels.
%
% Luis Gonzalo Sanchez Giraldo
% June 2023 

classdef EUCLIDEAN < trainableModel
    methods
        function obj = EUCLIDEAN(hyperparams)
            obj.hyperparams = hyperparams;
            if ~isfield(obj.hyperparams, 'd_y')
                obj.hyperparams(1).d_y = [];
            end
        end
        
        function train(obj, data, varargin)
            % initialize training parameters
            assert(length(varargin) <= 1, 'EULCIDEAN error. Invalid number of input arguments.');
            if isempty(varargin)
                param = struct([]);
            else
                param = varargin{1};
            end
            
            % RANDOM INITIALIZATION
            X = data{1};
            N = size(X, 1);        % Number of exemplars
            d_x = size(X,2);      % Input dimensionality
            
            if isempty(obj.hyperparams.d_y)
                obj.hyperparams.d_y = d_x;
            end
            
            % Make sure data is zero mean
            obj.trainable_params(1).mean = mean(X, 1);
            A = eye(d_x);
            A = A(:, 1: obj.hyperparams.d_y);
            M = A*A';
            obj.trainable_params(1).A = A;
            obj.trainable_params(1).M = M;
        end
        
        function pred = predict(obj, x)
            x = bsxfun(@minus, x, obj.trainable_params.mean);
            A = obj.trainable_params.A;
            pred = x*A;
        end
        
    end
end