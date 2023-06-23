clear all
close all
clc

%% IMPORTANT:
% Set the path to  the code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make sure you have set the right paths in this script
set_paths;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% here we fix the partitions to compare all methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
METHOD_USED = 'egml';

%% here we fix the partitions to compare all methods
results_root_path = fullfile(fullfile(root_path, '/results/cv_results/'), METHOD_USED);

feature_type = {'FTA_Features',...
                'FTA5_Features',...
                'RawEEG_Features'};

subject_id = {'B', 'C1', 'C2'};  


sim_params.feature_type = feature_type{2};
sim_params.pre_onset = true;
sim_params.n_folds = 10; % number of folds to get test error estimates
sim_params.n_runs = 1;
sim_params.n_subfolds = 10; % number of folds to do model selection (this is a nested cv fold within each training-test fold)

% For this experiment, we are generating data windows of 850 milliseconds
if sim_params.pre_onset
    sim_params.wd_str_t = -0.85; % in seconds
    sim_params.wd_end_t = 0;
else %post-onset
    sim_params.wd_str_t = 0;
    sim_params.wd_end_t = 0.85;
end

%% 
data_path = fullfile(data_root_path, sim_params.feature_type);
results_path = fullfile(results_root_path, sim_params.feature_type);
if ~isfolder(results_path)
    mkdir(results_path);
end

%% parameters of cross-validation
% window_name = 'z000p085';
window_name = 'm085z000';


%% cv pipeline parameters
%%%%%%%%%%%%%%%% If using metric learning
if any(strcmp(METHOD_USED, {'nca', 'ceml', 'egml'}))
    if strcmp(sim_params.feature_type, 'FTA5_Features')
        preprocess_params = {'d_y',  {30, []}};
    else
        preprocess_params = {'d_y',  {300, []}};
    end
    metric_params = {'d_y', {3, 10, 100}};
    classifier_params = {'svm_type',    {1};...
                         'nu',          {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8};...
                         'kernel_type', {0}}; 

%%%%%%%%%%%%%%%% Eulcidean does a more thorough search for input dimension%
elseif strcmp(METHOD_USED, 'euclidean')
    if strcmp(sim_params.feature_type, 'FTA5_Features')
        preprocess_params = {'d_y',  {10, 20, 30, 40, 50, []}};
    else
        preprocess_params = {'d_y',  {100, 200, 300, 400, 500, []}};
    end
    metric_params = {'d_y', {[]}};
    classifier_params = {'svm_type',    {1};...
                         'nu',          {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8};...
                         'kernel_type', {0}};
end

preprocess_model_class = @PCANormalizer;
switch(METHOD_USED)
    case 'nca'
    metric_model_class = @NCA;
    case 'ceml'
        metric_model_class = @CEML;
    case 'egml'
        metric_model_class = @EGML;
    case 'euclidean'
        metric_model_class = @EUCLIDEAN;
    otherwise
        error('Method not implemented')
end
classifier_model_class = @SVM;

preproces_param_grid = paramGrid([], preprocess_params, 1);
metric_param_grid = paramGrid([], metric_params, 1);
classifier_param_grid = paramGrid([], classifier_params, 1);
cv_loss = @(true_l, pred) 1-mean(true_l == pred.class_label);

for iSbj = 1:length(subject_id)
    fprintf('Reading folds for subject %s, window %s\n', subject_id{iSbj}, window_name);
        
    % read data dir for subject
    data_subject_path = fullfile(data_path, sprintf('Subject_%s',subject_id{iSbj}));
    assert(isfolder(data_subject_path), 'data path does not exist');
    
    data_window_path = fullfile(data_subject_path, window_name);
    assert(isfolder(data_window_path), 'data window path does not exist');

    
    % read results dir for subject
    results_subject_path = fullfile(results_path, sprintf('Subject_%s',subject_id{iSbj}));
    assert(isfolder(results_subject_path), 'results data path does not exist');
    
    results_window_path = fullfile(results_subject_path, window_name);
    assert(isfolder(results_window_path), 'results window path does not exist');

    for iRun = 1:sim_params.n_runs
        fprintf('\tRun %d\n', iRun)
        % data dir where all folds are stored
        data_run_path = fullfile(data_window_path, sprintf('run_%d', iRun));
        assert(isfolder(data_run_path), 'data run path does not exist')
        
        results_run_path = fullfile(results_window_path, sprintf('run_%d', iRun));
        assert(isfolder(results_run_path), 'results run path does not exist')
        
        
        
        for iFld = 1:sim_params.n_folds
            foldname = sprintf('fold_%d', iFld);
            data_fold_file = strcat(fullfile(data_run_path, foldname), '.mat');
            assert(isfile(data_fold_file), 'data fold file does not exist');
   
            results_fold_path = fullfile(results_run_path, foldname);
            assert(isfolder(results_fold_path), 'results fold path does not exist');
    
            fprintf('\t\tReading fold %d\n', iFld);
            fold_results = load(fullfile(results_fold_path, 'cv_loss.mat'));
            
            val_losses{iFld} = fold_results.val_losses;
            val_losses_fold{iFld} = fold_results.val_losses_fold;
            %%%%%%% Train Best Model if fold %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [~, best_preprocessor, best_metric, best_classifier] = getBestValIdx(val_losses{iFld});
            fprintf('Training best model for fold %d\n', iFld);
            data_fold = load(data_fold_file);
            X = data_fold.X_train;
            X = reshape(X, [], size(X, 3));
            data_train{1} = X';
            data_train{2} = data_fold.l_train;
            X = data_fold.X_test;
            X = reshape(X, [], size(X, 3));
            data_test{1} = X';
            data_test{2} = data_fold.l_test;
            
            % retrain best preprocessor
            preprocessor = preprocess_model_class(preproces_param_grid(best_preprocessor));
            preprocessor.train(data_train{1});
            data_prep_train = preprocessor.predict(data_train{1});
            prep_data_fold_train = {data_prep_train, data_train{2}};
            
            data_prep_test = preprocessor.predict(data_test{1});
            
            % retrain best metric
            metric = metric_model_class(metric_param_grid(best_metric));
            metric.train(prep_data_fold_train);
            data_metric_train = metric.predict(data_prep_train);
            metric_data_fold_train = {data_metric_train, data_train{2}};
            data_metric_test = metric.predict(data_prep_test);
            
            % retrain best classifier
            classifier = classifier_model_class(classifier_param_grid(best_classifier));
            classifier.train(metric_data_fold_train);
            pred_fold_test = classifier.predict(data_metric_test);

            % the cv looss can use either class labels prediction or
            % classifier scores (for instance with ROC areas)
            true_label = data_test{2};

            best_model(iFld).preprocessor = preprocessor;
            best_model(iFld).metric = metric;
            best_model(iFld).classifier = classifier;
            test_losses_fold(iFld) = cv_loss(true_label, pred_fold_test); 
        end
        fprintf('\tsaving all folds accuracies\n')
        save(fullfile(results_run_path, 'all_val__test_losses.mat'), 'val_losses', 'val_losses_fold', 'test_losses_fold');
        save(fullfile(results_run_path, 'all_best_model.mat'), 'best_model');
        
    end
end


function [val, varargout] = getBestValIdx(val_losses)
ndims_val = ndims(val_losses);
assert(nargout-1 == ndims_val, 'Mismatch number of output arguments');
[val, best_val_idx] = min(val_losses(:));
for iDim = 1:ndims_val
    varargout{iDim} = mod(best_val_idx-1,size(val_losses, iDim))+1;
    best_val_idx = ceil(best_val_idx / size(val_losses, iDim));
end
end