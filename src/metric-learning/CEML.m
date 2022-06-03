classdef CEML < trainableModel
    methods
        function obj = CEML(hyperparams)
            obj.hyperparams = hyperparams;
            if ~isfield(obj.hyperparams, 'd_y')
                obj.hyperparams(1).d_y = [];
            end
            if ~isfield(obj.hyperparams, 'var') 
                obj.hyperparams(1).sigma = [];
            end
            if ~isfield(obj.hyperparams, 'alpha') 
                obj.hyperparams(1).alpha = 1.01;
            end
        end
        
        function train(obj, data, varargin)
            % initialize training parameters
            assert(length(varargin) <= 1, 'CEML error. Invalid number of input arguments.');
            if isempty(varargin)
                param = struct([]);
            else
                param = varargin{1};
            end
            param = obj.initializeTrainParams(param);
            
            % RANDOM INITIALIZATION
            X = data{1};
            N = size(X, 1);        % Number of exemplars
            d_x = size(X,2);      % Input dimensionality
            
            if isempty(obj.hyperparams.d_y)
                obj.hyperparams.d_y = d_x;
            end
            d_y = obj.hyperparams.d_y;        % Dimensionality of the Projected space
            
            if isempty(obj.hyperparams.sigma)
                obj.hyperparams.sigma = sqrt(d_y/2);
            end
            
            sigma = obj.hyperparams.sigma;
            alpha = obj.hyperparams.alpha;
            A = randn(d_x,d_y);   % Transformation Matrix
            A = sqrt(d_y)*A/sqrt(trace(A'*A));
            [L, ~] = labelMatrix(data{2});  % Label matrix (one-hot encoding)
            K_l = L* L'/N;                 % Gram matrix for labels
            
            % Make sure data is zero mean
            obj.trainable_params(1).mean = mean(X, 1);
            X = bsxfun(@minus, X, obj.trainable_params.mean);
            
            Y = X*A;                       % Initial transformation
            
            % Initialization of optimization parameters
            mu = param.mu_in;  % step size
            beta1 = param.beta1;
            beta2 = param.beta2;
            reg_par = 1;
            pepsilon = param.epsilon;
            Gm1 = zeros(size(A));
            Gm2 = zeros(size(A));
            F_old = log(size(L,2));
            
            for i= 1 : param.n_iter
                mu = mu - (param.mu_in - param.mu_fin)/param.n_iter;  % Step size schedulling
                K_y = real(guassianMatrix(Y,sigma))/N;  % Gram matrix of tranformed inputs
                K_ly = K_l.*K_y*N;                      % Gram matrix joint-space
                
                %%% Compute spectrum of K_y and K_ly
                [V_y,L_y] = eig(K_y);
                V_y = real(V_y);
                lambda_y = abs(diag(L_y));
                
                [V_ly,L_ly] = eig(K_ly);
                V_ly = real(V_ly);
                lambda_ly = abs(diag(L_ly));
                
                
                %%% Compute gradient terms
                Grad_y = V_y*diag((alpha/(1 - alpha))*(1/(sum(lambda_y.^alpha)))*(lambda_y.^(alpha - 1)))*V_y';
                Grad_ly = V_ly*diag((alpha/(1 - alpha))*(1/(sum(lambda_ly.^alpha)))*(lambda_ly.^(alpha - 1)))*V_ly';
                Grad = N*Grad_ly.*K_l - Grad_y;
                P = Grad.*K_y;
                
                %%% Updating the transformation matrix (Adam)
                G = X'*(P - diag(P*ones(N,1)))*(X*A) + reg_par*A;
                Gm1 = beta1 * Gm1 + (1 - beta1) * G;
                Gm2 = beta2 * Gm2 + (1 - beta2) * G.^2;
                mu_i = mu * sqrt(1 - beta2^i) / (1 - beta1^i);
                A = A - mu_i*(Gm1 ./ (sqrt(Gm2) + pepsilon));
                
                %%% Fixed point update of the lagrange multiplier
                reg_par = -trace((A'*X')*(P - diag(P*ones(N,1)))*(X*A))/(d_y);
                
                %%% Update Transformed inputs
                Y = X*A;
                
                %%% Display progress
                if (~mod(i,30))
                    F =  (1/(1 - alpha))*log( (sum(lambda_ly.^alpha))/(sum(lambda_y.^alpha)) );
                    fprintf('CondEntropyMetricLearning at Iteration %d. CondEntropy = %f, \nLagrange multiplier = %f, trace value = %f \n', i, F, reg_par, trace(A'*A));
                    if abs((F_old - F)/F) < param.tol
                        break;
                    end
                    F_old = F;
                end
                
            end
            
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
    methods (Access = private)
        function param = initializeTrainParams(obj, param)
            
            if ~isfield(param, 'mu_in')
                param(1).mu_in = 0.01;
            end
            if ~isfield(param, 'mu_fin')
                param(1).mu_fin = param.mu_in/10;
            end
            if ~isfield(param, 'tol')
                param(1).tol = 1e-3;
            end
            if ~isfield(param, 'n_iter')
                param(1).n_iter = 2000;
            end
            if ~isfield(param, 'beta1')
                param(1).beta1 = 0.9;
            end
            if ~isfield(param, 'beta2')
                param(1).beta2 = 0.99;
            end
            if ~isfield(param, 'epsilon')
                param(1).epsilon = 1e-8;
            end
            if (param.mu_in <  param.mu_fin)
                warning('CEML warning. Final stepsize is larger than initial stepsize.')
            end
        end
    end
end