clear all
close all
clc

%% IMPORTANT:
% Set the path to  the code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make sure you have set the right paths in this script
set_paths;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% here we fix the partitions to compare all methods

results_root_path = fullfile(root_path, 'results/cv_results/euclidean');
pretrained_results_root_path = fullfile(root_path, 'results/cv_results/euclidean');

feature_type = {'FTA_Features',...
                'FTA5_Features',...
                'RawEEG_Features'};

subject_id = {'B', 'C1', 'C2'};  


sim_params.feature_type = feature_type{2};
sim_params.pre_onset = true;
sim_params.n_runs = 1;
% number of folds to get test error estimates
sim_params.n_folds = 10; 
% number of folds to do model selection (this is a nested cv fold within each 
% training-test fold)
sim_params.n_subfolds = 10;

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
pretrained_results_path = fullfile(pretrained_results_root_path, sim_params.feature_type);

if ~isfolder(results_path)
    mkdir(results_path);
end

%% parameters of cross-validation
window_name = 'm085z000';
% window_name = 'z000p085';


%% cv pipeline parameters
if strcmp(sim_params.feature_type, 'FTA5_Features')
    preprocess_params = {'d_y',  {10, 20, 30, 40, 50, []}};
else
    preprocess_params = {'d_y',  {100, 200, 300, 400, 500, []}};
end
metric_params = {'d_y', {[]}};
classifier_params = {'svm_type',    {1};...
		     'nu',          {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8};...
		     'kernel_type', {0}};

cv_loss = @(true_l, pred) 1-mean(true_l == pred.class_label);
preprocessor = @PCANormalizer;
metric = @EUCLIDEAN;
classifier = @SVM;


%% cross-validation
if PRETRAINED_
    run_cross_validation_pretrained;
else
    run_cross_validation;
end