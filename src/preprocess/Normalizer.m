% Normalizer.m 
% A trainableModel class for data normalization
% 
% Luis Gonzalo Sanchez Giraldo
% June 2023 

classdef Normalizer < trainableModel
    methods
        function obj = Normalizer(hyperparams)
            obj.hyperparams = hyperparams;
            if ~isfield(obj.hyperparams,'per_dim')
                obj.hyperparams(1).per_dim = true;
            end
            if ~isfield(obj.hyperparams,'lambda')
                obj.hyperparams(1).lambda = 0;
            end
        end
        
        function train(obj, data, varargin)
            % initialize training parameters
            assert(length(varargin) <= 1, 'Normalizer error. Invalid number of input arguments.');
            if isempty(varargin)
                param = struct([]);
            else
                param = varargin{1};
            end
            param = obj.initializeTrainParams(param);
            % RANDOM INITIALIZATION
            X = data;
            N = size(X, 1);
            d_x = size(X, 2);
            obj.trainable_params(1).mean = mean(X, 1);
            X = bsxfun(@minus, X, obj.trainable_params.mean);
            if param.unbiased == true
                obj.trainable_params(1).var = sum(X.^2, 1)/(N-1);
            else
                obj.trainable_params(1).var = sum(X.^2, 1)/N;
            end
            
            if obj.hyperparams.per_dim == true
                obj.trainable_params(1).scaler = 1 ./ sqrt(obj.trainable_params.var + obj.hyperparams.lambda);
            else
                obj.trainable_params(1).scaler = sqrt(d_x) / sqrt(sum(obj.trainable_params.var) + d_x * obj.hyperparams.lambda);
            end
            
        end
        
        function pred = predict(obj, x)
            x = bsxfun(@minus, x, obj.trainable_params.mean);
            pred = bsxfun(@times, x, obj.trainable_params.scaler);
        end
        
    end
    methods (Access = private)
        function param = initializeTrainParams(obj, param)
            if ~isfield(param, 'unbiased')
                param(1).unbiased = true;
            end
        end
    end
end