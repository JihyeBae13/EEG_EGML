clear all
close all
clc

addpath(genpath('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/src'));
addpath('/home/lgsanchez/work/Code/libraries/libsvm/matlab/')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% here we fix the partitions to compare all methods

data_root_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/data/';
pretrained_results_root_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/euclidean';
results_root_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/nonlinear_euclidean';

feature_type = {'FTA_Features',...
                'FTA5_Features',...
                'RawEEG_Features'};

subject_id = {'B', 'C1', 'C2'};  


sim_params.feature_type = feature_type{3};
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
pretrained_results_path = fullfile(pretrained_results_root_path, sim_params.feature_type);
results_path = fullfile(results_root_path, sim_params.feature_type);
if ~isfolder(results_path)
    mkdir(results_path);
end

%% parameters of cross-validation
window_name = 'm085z000';
% window_name = 'z000p085';


%% cv pipeline parameters
classifier_params = {'svm_type',    {1};...
        		     'nu',          {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8};...
		             'kernel_type', {2};...
                     'gamma',       {0.1, 0.5, 1, 2, 10}};

cv_loss = @(true_l, pred) 1-mean(true_l == pred.class_label);
classifier = @SVM;

run_cross_validation_pretrained;


% for iSbj = 1:length(subject_id)
%     fprintf('Cross validating folds for subject %s, window %s\n', subject_id{iSbj}, window_name);
%     % read  data dir for subject
%     subject_path = fullfile(data_path, sprintf('Subject_%s',subject_id{iSbj}));
%     assert(isfolder(subject_path), 'data path does not exist');
%         
%     % pretrained results dir for subject
%     pretrained_results_subject_path = fullfile(pretrained_results_path, sprintf('Subject_%s',subject_id{iSbj}));
%     % create  results dir for subject
%     results_subject_path = fullfile(results_path, sprintf('Subject_%s',subject_id{iSbj}));
%     if ~isfolder(results_subject_path)
%         mkdir(results_subject_path);
%     end
%     
%     window_path = fullfile(subject_path, window_name);
%     assert(isfolder(window_path), 'window path does not exist');
%     
%     pretrained_results_window_path = fullfile(pretrained_results_subject_path, window_name);
%     
%     results_window_path = fullfile(results_subject_path, window_name);
%     if ~isfolder(results_window_path)
%         mkdir(results_window_path);
%     end
% 
%     
%     for iRun = 1:sim_params.n_runs
%         fprintf('\tRun %d\n', iRun)
%         run_path = fullfile(window_path, sprintf('run_%d', iRun));
%         assert(isfolder(run_path), 'run path does not exist')
%         
%         pretrained_results_run_path = fullfile(pretrained_results_window_path, sprintf('run_%d', iRun));
% 
%         results_run_path = fullfile(results_window_path, sprintf('run_%d', iRun));
%         if ~isfolder(results_run_path)
%             mkdir(results_run_path)
%         end
%         
%         for iFld = 1:sim_params.n_folds
%             foldname = sprintf('fold_%d', iFld);
%             data_fold = load(fullfile(run_path, foldname));
%             pretrained_results_fold_path = fullfile(pretrained_results_run_path, foldname);
%             results_fold_path = fullfile(results_run_path, foldname);
%             if ~isfolder(results_fold_path)
%                 mkdir(results_fold_path)
%             end
%             fprintf('\t\tFold %d\n', iFld)
%             
%             X = data_fold.X_train;
%             X = reshape(X, [], size(X, 3));
%             
%             data{1} = X';
%             data{2} = data_fold.l_train;
%             
%             cv_object = data_fold.cv_subobj;
%             
%             pretraiend_model_path = pretrained_results_fold_path;
%             store_model_path = results_fold_path;
%             %%% save fold data
%             [val_losses, val_losses_fold] = crossvalOnlyClassifierGrid(data,...
%                                                                        cv_object,... 
%                                                                        pretrained_model_path,...
%                                                                        classifier,...
%                                                                        classifier_params,...
%                                                                        cv_loss, ...
%                                                                        store_model_path);
%             % save losses
%             save(fullfile(store_model_path, 'cv_loss.mat'), 'val_losses', 'val_losses_fold');
% 
%         end
%         
%     end
% end


function sign_prefix = getSignPrefix(x)
if x > 0
    sign_prefix = 'p';
elseif x < 0
    sign_prefix = 'm';
elseif x == 0
    sign_prefix = 'z';
end
end



% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% feature_path = 'FTA_Features';
% % feature_path = 'RawEEG_Features';
% 
% % window_path = 'm06m02'; % time interval relative to onset [-0.6, -0.2]
% window_path = 'z000p085'; % time interval relative to onset [0, 0.85];
% 
% %%% compute expected window size TODO do it properly with regular expressions 
% if strcmp(window_path,'z000p085')
%     exp_window = 0.85;
% elseif strcmp(window_path,'m06m02')
%     exp_window = 0.4;
% else
%     error('I do not know this window size');
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% experiment_path = fullfile(data_path, feature_path, window_path);
% 
% subject_dirs = dir(experiment_path);
% subject_id = {'B', 'C1', 'C2'};  
% % optimizer = 'adam';
% optimizer = '';
% met_learn_method = {'NCA', 'CEML', 'Euclidean'};
% per_dim = true;
% 
% params_table = {'Check me',...
%                 'Subject Dir',...
%                 'method dir', ...
%                 'sim_par n_ch',...
%                 'samp freq',...
%                 'sim par d_y',...
%                 'alpha',...
%                 'sigma',...
%                 'ex_t',...
%                 'sim expected ex_t',...
%                 'sim par pcs',...
%                 'fold m',...
%                 'fold pcs',...
%                 'fold d_y',...
%                 'fold mean m',...
%                 'fold_n_ch'};
% 
% for iSbj = 1:length(subject_id)
%     % process folders that match the pattern
%     subject_exp = strcat('(Subject)',subject_id{iSbj}, '.+(win).+', window_path, '.+', optimizer, '$');
%     
%    
%     for iSbjDr = 1:length(subject_dirs)
%         % check pattern
%         exp_idx = regexp(subject_dirs(iSbjDr).name, subject_exp);
%         if isempty(exp_idx)
%             continue
%         end
%         % process files in matched folder
%         current_subject_dir = subject_dirs(iSbjDr);
%         fprintf('Processing %s\n',current_subject_dir.name);
%         % print parameters
%         sim_params = checkSimParams(fullfile(current_subject_dir.folder, current_subject_dir.name, 'simParameters.mat'));
%         sim_params.exp_ex_t = exp_window;
%         sim_params.is_fta = strcmp(feature_path, 'FTA_Features');
%         for iML = 1:length(met_learn_method)
%             method_dirs = dir(fullfile(current_subject_dir.folder, current_subject_dir.name, met_learn_method{iML}));
%             for iMth = 1:length(method_dirs)
%                 run_dir = method_dirs(iMth);
%                 if any(strcmp(method_dirs(iMth).name, {'.', '..'})) || ~isfolder(fullfile(method_dirs(iMth).folder, method_dirs(iMth).name))
%                     continue
%                 end
%                 fprintf('Processing %s\n',method_dirs(iMth).name);
%                 subject_runs_dir = dir(fullfile(method_dirs(iMth).folder, method_dirs(iMth).name));
%                 oh_my = false;
%                 for iFl = 1:length(subject_runs_dir)
%                     [~, ~, ext] = fileparts(subject_runs_dir(iFl).name);
%                     if any(strcmp(subject_runs_dir(iFl).name, {'.', '..'})) || isfolder(fullfile(subject_runs_dir(iFl).folder, subject_runs_dir(iFl).name)) || ~strcmp(ext, '.mat')
%                         continue
%                     end
%                     fprintf('Reading %s\n', subject_runs_dir(iFl).name)
%                     [fold_params, fold_oh_my] = checkFoldParams(fullfile(subject_runs_dir(iFl).folder, subject_runs_dir(iFl).name), sim_params);
%                     oh_my = oh_my || fold_oh_my;
%                 end
%                 new_row = {oh_my,...
%                     current_subject_dir.name, ...
%                     strcat(method_dirs(iMth).name,'_', met_learn_method{iML}),...
%                     sim_params.n_ch, ...
%                     sim_params.samp_freq, ...
%                     sim_params.d_y, ...
%                     sim_params.alpha, ...
%                     sim_params.sigma, ...
%                     sim_params.ex_t, ...
%                     sim_params.exp_ex_t,...
%                     sim_params.pcs, ...
%                     fold_params.m, ...
%                     fold_params.T_pcs, ...
%                     fold_params.A_dy, ...
%                     fold_params.B_m,...
%                     fold_params.n_ch};
%                 
%                 params_table = cat(1, params_table, new_row);
%             end
%         end
%     end
% end
% writecell(params_table, strcat(feature_path, '_', window_path));
% 
% 
% function sim_params = checkSimParams(sim_params_file, verbose)
% if ~exist('verbose', 'var')
%     verbose = false;
% end
% sim_params_struct = load(sim_params_file);    
% sim_params.ch_names = sim_params_struct.ds.chnames;
% sim_params.n_ch = length(sim_params.ch_names);
% sim_params.samp_freq = sim_params_struct.ds.sampFreq;
% sim_params.d_y = sim_params_struct.d_y;
% sim_params.alpha = sim_params_struct.alpha;
% sim_params.sigma = sim_params_struct.sigma;
% sim_params.ex_t = sim_params_struct.ex_t;
% sim_params.pcs = sim_params_struct.pcs;
% if verbose
%     disp(sim_params);
% end
% end
% 
% function [fold_params, oh_my] = checkFoldParams(fold_params_file, sim_params)
% if ~exist('verbose', 'var')
%     verbose = false;
% end
% 
% oh_my = false;
% fold_params_struct = load(fold_params_file);    
% 
% fold_params.m = fold_params_struct.m;
% fold_params.n = fold_params_struct.n;
% 
% % check T (pca transform)
% if ~isempty(fold_params_struct.T)
%     [T_d, T_pcs] = size(fold_params_struct.T);
% else
%     T_d = [];
%     T_pcs = [];
% end
% fold_params.T_pcs = T_pcs;
% oh_my = oh_my || (fold_params.m ~= T_d);
% 
% % check A
% if ~isempty(fold_params_struct.A)
%     [A_d, A_p] = size(fold_params_struct.A);
% else % case of Euclidean
%     A_d = [];
%     A_p = [];
% end
% fold_params.A_dy = A_p;
% if ~isempty(fold_params.A_dy)
%     oh_my = oh_my || (fold_params.A_dy ~= sim_params.d_y);
%     oh_my = oh_my || (fold_params.T_pcs ~= A_d);
% else
%     oh_my = true;
% end
% 
% % check B
% if ~isempty(fold_params_struct.B)
%     B_d = length(fold_params_struct.B);
% else % case of Euclidean
%     B_d = [];
% end
% fold_params.B_m = B_d;
% oh_my = oh_my || (fold_params.B_m ~= fold_params.m);
% 
% % check number of channels
% if sim_params.is_fta
%     if mod(fold_params.m, 5) == 0
%         fold_params.n_ch = fold_params.m / 5;
%     elseif mod(fold_params.m, 9) == 0
%         fold_params.n_ch = fold_params.m / 9;
%     else
%         error('I can not guess the number of frequency components');
%     end
% else
%     fold_params.n_ch = fold_params.m / (sim_params.ex_t * sim_params.samp_freq);
% end
% 
% oh_my = oh_my || (fold_params.n_ch ~= sim_params.n_ch);
% 
% % check if window size matches the expected window size
% oh_my = oh_my || (sim_params.ex_t ~= sim_params.exp_ex_t);
% 
% if verbose
%     disp(fold_params);
%     if oh_my
%         disp('Something is fishy!')
%     end
% end
% end
