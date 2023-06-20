% SVM.m 
% A trainableModel class that wrap LIBSVM support vector machine
    % 

    % To use this class you must have LIBSVM package installed
    % https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% 
% Luis Gonzalo Sanchez Giraldo
% June 2023 


classdef SVM < trainableModel


    methods
        function obj = SVM(hyperparams)
           obj.hyperparams = obj.initializeHyperparams(hyperparams);
        end
        
        function train(obj, data, varargin)
            % Initialize training parameters
            assert(length(varargin) <= 1, 'SVM error. Invalid number of input arguments.');
            if isempty(varargin)
                param = struct([]);
            else
                param = varargin{1};
            end
            param = obj.initializeTrainParams(param); % not used currently

            % WRAPPER FOR LIBSVM
            % Make sure data is zero mean and double 
            X = double(data{1});
            labels = double(data{2});
            obj.trainable_params(1).mean = mean(X, 1);
            X = bsxfun(@minus, X, obj.trainable_params.mean);
            % gamma is normalized by the number of dimensions of the data
            % truegamma = gamma/n_dims
            if ~isempty(obj.hyperparams.gamma)
                obj.hyperparams.truegamma = obj.hyperparams.gamma / size(X, 2);
            else
                obj.hyperparams.truegamma = [];
            end
            libsvm_train_options = obj.makeLIBSVMOptionsString();
            
            obj.trainable_params(1).svm_model = svmtrain(labels, X, libsvm_train_options);
                       
        end
        
        function pred = predict(obj, x)
            x = bsxfun(@minus, x, obj.trainable_params.mean);
            libsvm_test_options = obj.makeLIBSVMOptionsString(true);
            [pred.class_label, ~, pred.scores] = svmpredict(ones(size(x,1),1), x, obj.trainable_params.svm_model, libsvm_test_options);
        end
        
    end

    methods (Access = private)
        function param = initializeTrainParams(obj, param)
            % do nothing
        end
        
        function hyperparam = initializeHyperparams(obj, hyperparam)
           svm_param_list = {'svm_type',...
                             'kernel_type',...
                             'degree',...
                             'gamma',...
                             'coef0',...
                             'cost',...
                             'nu',...
                             'epsilon',...
                             'shrinking',...
                             'probability_estimates',...
                             'weight', ...
                             'n_fold_cv_mode',...
                             'verbose'};
           for iPrm = 1:length(svm_param_list)
               if ~isfield(hyperparam, svm_param_list{iPrm})
                   hyperparam(1).(svm_param_list{iPrm}) = [];
               end
           end
        end
        
        function options_string = makeLIBSVMOptionsString(obj, istest)
            if ~exist('istest', 'var')
                istest = false;
            end
            svm_param_list = {'svm_type',               's %d';...
                              'kernel_type',            't %d';...
                              'degree',                 'd %d';...
                              'truegamma',              'g %5.5d';...
                              'coef0',                  'r %5.5d';...
                              'cost',                   'c %5.5d';...
                              'nu',                     'n %1.5d';...
                              'epsilon',                'p %1.6d';...
                              'shrinking',              'h %d';...
                              'probability_estimates',  'b %d';...
                              'weight',                 'w-%d %5.5d';...
                              'n_fold_cv_mode',         'v %d';...
                              'verbose',                'q %d'};
            options_string = '';
            if istest ~= true   
                for iPrm = 1 : size(svm_param_list,1)
                    if ~isempty(obj.hyperparams.(svm_param_list{iPrm, 1}))
                        if strcmp(svm_param_list{iPrm, 1}, 'weight')
                            for iCls = 1 : length(obj.hyperparams.weight)
                                options_string = strcat(options_string, ' -', sprintf(svm_param_list{iPrm, 2},iCls, obj.hyperparams.(svm_param_list{iPrm, 1})(iCls)));
                            end
                        else
                            options_string = strcat(options_string, ' -', sprintf(svm_param_list{iPrm, 2}, obj.hyperparams.(svm_param_list{iPrm, 1})));
                        end
                    end
                end
            else
                if ~isempty(obj.hyperparams.probability_estimates)
                    options_string = sprintf('-b %d', obj.hyperparams.probability_estimates);
                end
            end
        end
            
    end
end