% PCANormalizer.m 
% A trainableModel class for to normalize data using PCA
% 
% Luis Gonzalo Sanchez Giraldo
% June 2023 

classdef PCANormalizer < trainableModel
    methods
        function obj = PCANormalizer(hyperparams)
            obj.hyperparams = hyperparams;
            if ~isfield(obj.hyperparams,'d_y')
                obj.hyperparams(1).d_y = [];
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
            % COMPUTE PC
            X = data;
            N = size(X, 1);
            d_x = size(X, 2);
            if isempty(obj.hyperparams.d_y)
                obj.hyperparams.d_y = min(d_x, N);
            end
            d_y = obj.hyperparams.d_y;
            obj.trainable_params(1).mean = mean(X, 1);
            X = bsxfun(@minus, X, obj.trainable_params.mean);
            [~, D, V] = svd(X);
            d = diag(D);
            if param.unbiased == true
                obj.trainable_params(1).var = (d(1:d_y).^2) / (N-1);
            else
                obj.trainable_params(1).var = (d(1:d_y).^2) / N;
            end
            
            obj.trainable_params(1).T = V(:, 1: d_y) * sqrt(d_y) / sqrt(sum(obj.trainable_params.var) + d_y * obj.hyperparams.lambda);
        end
        
        function pred = predict(obj, x)
            x = bsxfun(@minus, x, obj.trainable_params.mean);
            pred = x * obj.trainable_params.T;
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