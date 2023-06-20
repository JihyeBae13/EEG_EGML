% NCA.m 
% A trainableModel class for Neighborhood Component Analysis
% 
% Luis Gonzalo Sanchez Giraldo
% June 2023 

classdef NCA < trainableModel
    methods
        function obj = NCA(hyperparams)
            obj.hyperparams = hyperparams;
            if ~isfield(obj.hyperparams, 'd_y') 
                obj.hyperparams(1).d_y = [];
            end
            if ~isfield(obj.hyperparams, 'lambda') 
                obj.hyperparams(1).lambda = 0;
            end
        end
        
        function train(obj, data, varargin)
            % initialize training parameters
            assert(length(varargin) <= 1, 'NCA error. Invalid number of input arguments.');
            if isempty(varargin)
                param = struct([]);
            else
                param = varargin{1};
            end
            % RANDOM INITIALIZATION
            X = data{1};
            n = size(X, 1);        % Number of exemplars
            d_x = size(X,2);      % Input dimensionality
            labels = data{2};
            if isempty(obj.hyperparams(1).d_y)
                obj.hyperparams(1).d_y = size(X, 2);
            end
            
            % Make sure data is zero mean
            obj.trainable_params(1).mean = mean(X, 1);
            X = bsxfun(@minus, X, obj.trainable_params.mean);
            
            % Initialize some variables
            max_iter = 200;
            batch_size = min(5000, n);
            no_batches = ceil(n / batch_size);
            max_iter = ceil(max_iter / no_batches);
            [lablist, foo, labels] = unique(labels);
            A = randn(d_x, obj.hyperparams(1).d_y) * .01;
            converged = false;
            iter = 0;
            
            % Main iteration loop
            while iter < max_iter && ~converged
                
                % Loop over batches
                iter = iter + 1;
                disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) '...']);
                ind = randperm(n);
                for batch=1:batch_size:n
                    
                    % Run NCA minimization using three linesearches
                    cur_X    = double(X(ind(batch:min([batch + batch_size - 1 n])),:));
                    cur_labels = labels(ind(batch:min([batch + batch_size - 1 n])));
                    [A, f] = minimize(A(:), 'nca_lin_grad', 5, cur_X, cur_labels, obj.hyperparams(1).d_y, obj.hyperparams(1).lambda);
                    A = reshape(A, [d_x obj.hyperparams(1).d_y]);
                    if isempty(f) || f(end) - f(1) > -1e-4
                        disp('Converged!');
                        converged = true;
                    end
                end
            end
            
            % Compute embedding
                   
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