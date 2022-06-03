clear all
close all
clc

addpath(genpath('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/src'));
addpath('/home/lgsanchez/work/Code/libraries/libsvm/matlab/')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
METHOD_USED = 'egml';

%% here we fix the partitions to compare all methods
data_root_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/data/';
results_root_path = fullfile('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/', strcat('nonlinear_',METHOD_USED));
% results_root_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/ceml';

feature_type = {'FTA_Features',...
                'FTA5_Features',...
                'RawEEG_Features'};

subject_id = {'B', 'C1', 'C2'};  


sim_params.feature_type = feature_type{3};
sim_params.pre_onset = false;
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
window_name = 'z000p085';
% window_name = 'm085z000';


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
		         'kernel_type', {2};...
                         'gamma',       {0.1, 0.5, 1, 2, 10}};

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
	                 'kernel_type', {2};...
                         'gamma',       {0.1, 0.5, 1, 2, 10}};

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
        
        load(fullfile(results_run_path, 'all_best_model.mat'), 'best_model');
        
        for iFld = 1:sim_params.n_folds
            foldname = sprintf('fold_%d', iFld);
            data_fold_file = strcat(fullfile(data_run_path, foldname), '.mat');
            assert(isfile(data_fold_file), 'data fold file does not exist');
   
            results_fold_path = fullfile(results_run_path, foldname);
            assert(isfolder(results_fold_path), 'results fold path does not exist');
    
            %%%%%%% Train Best Model if fold %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%% Test Best Model if fold %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            fprintf('Testing best model for fold %d\n', iFld);
            data_fold = load(data_fold_file);
            X = data_fold.X_test;
            X = reshape(X, [], size(X, 3));
            data_test{1} = X';
            data_test{2} = data_fold.l_test;
            
            % test best preprocessor
            data_prep_test = best_model(iFld).preprocessor.predict(data_test{1});
            
            % test best metric
            data_metric_test = best_model(iFld).metric.predict(data_prep_test);
            
            % test best classifier
            pred_fold_test = best_model(iFld).classifier.predict(data_metric_test);

            % the cv looss can use either class labels prediction or
            % classifier scores (for instance with ROC areas)
            true_label = data_test{2};
            confusion_matrix{iFld} = confusionmat(true_label, pred_fold_test.class_label);
            accuracy(iFld) = 100*sum(diag(confusion_matrix{iFld}))/sum(confusion_matrix{iFld}(:));
            class_sensistivity(:,iFld) = 100*diag(confusion_matrix{iFld}) ./ sum(confusion_matrix{iFld}, 2);
            class_predictive_values(:, iFld) = 100*diag(confusion_matrix{iFld}) ./ sum(confusion_matrix{iFld}, 1)';

        end
        fprintf('\tsaving all folds confusion matrices\n')
        save(fullfile(results_run_path, 'all_test_confusion_matrices.mat'), 'confusion_matrix', 'accuracy', 'class_sensistivity', 'class_predictive_values');
        fprintf('%s, %s, %s : %.2f(%.2f), %.2f(%.2f), %.2f(%.2f), %.2f(%.2f), %.2f(%.2f) \n', sim_params.feature_type,...
                                                  subject_id{iSbj}, ...
                                                  METHOD_USED, ...
                                                  mean(accuracy), std(accuracy), ...
                                                  mean(class_sensistivity(1,:)), std(class_sensistivity(1,:)),...
                                                  mean(class_predictive_values(1,:)), std(class_predictive_values(1,:)),...
                                                  mean(class_sensistivity(2,:)), std(class_sensistivity(2,:)),...
                                                  mean(class_predictive_values(2,:)), std(class_predictive_values(2,:)) );

    end
end

function sign_prefix = getSignPrefix(x)
if x > 0
    sign_prefix = 'p';
elseif x < 0
    sign_prefix = 'm';
elseif x == 0
    sign_prefix = 'z';
end
end
