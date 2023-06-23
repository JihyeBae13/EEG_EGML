clear all
close all
clc


%% IMPORTANT:
% Set the path to  the code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make sure you have set the right paths in this script
set_paths;

addpath(genpath('/home/lgsanchez/work/Code/research/bci-eeg/plot_topography'))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods = {"euclidean",...
           "ceml",...
           "nca", ...
           "egml",...
           "nonlinear_euclidean",...
           "nonlinear_ceml",...
           "nonlinear_nca", ...
           "nonlinear_egml"};

%% here we fix the partitions to compare all methods

% results_root_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/ceml';

feature_type = {"FTA_Features",...
                "FTA5_Features",...
                "RawEEG_Features"};

subject_id = {"B", "C1", "C2"};  


% sim_params.feature_type = feature_type{2};
% sim_params.pre_onset = true;
sim_params.n_folds = 10; % number of folds to get test error estimates
sim_params.n_runs = 1;
% sim_params.n_subfolds = 10; % number of folds to do model selection (this is a nested cv fold within each training-test fold)
% 
% % For this experiment, we are generating data windows of 850 milliseconds
% if sim_params.pre_onset
%     sim_params.wd_str_t = -0.85; % in seconds
%     sim_params.wd_end_t = 0;
% else %post-onset
%     sim_params.wd_str_t = 0;
%     sim_params.wd_end_t = 0.85;
% end

%% parameters of cross-validation

window_names = {"m085z000", "z000p085"};

channel_names = {'Fp1',...
                 'Fp2',...
                 'F3' ,...
                 'F4' ,...
                 'C3' ,...
                 'C4' ,...
                 'P3' ,...
                 'P4' ,...
                 'O1' ,...
                 'O2' ,...
                 'A1' ,...
                 'A2' ,...
                 'F7' ,...
                 'F8' ,...
                 'T3' ,...
                 'T4' ,...
                 'T5' ,...
                 'T6' ,...
                 'Fz' ,...
                 'Cz' ,...
                 'Pz' ,...
                 'X5' };

%% 
% columns of table
METHOD = methods{4};
FEATURE = feature_type{1};
SUBJECT = subject_id{3};
WINDOW = window_names{2};
RUN = 1;
results_root_path = fullfile('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/', METHOD);
results_path = fullfile(results_root_path, FEATURE);
results_subject_path = fullfile(results_path, sprintf('Subject_%s',SUBJECT));
results_window_path = fullfile(results_subject_path, WINDOW);
results_run_path = fullfile(results_window_path, sprintf('run_%d', RUN));
model_string = sprintf('%s %s, %s, %s , % s \n', METHOD, FEATURE, sprintf('Subject_%s',SUBJECT), WINDOW, sprintf('run_%d', RUN));
models_data = load(fullfile(results_run_path, "all_best_model.mat"));
%%  compute channel importance 
if strcmp(FEATURE, 'FTA5_Features')
    window_sz = 9;
elseif strcmp(FEATURE, 'FTA_Features')
    window_sz = 0.85*200 - 1;
else
    window_sz = 0.85*200;
end
n_dim = 3;
channel_importance = channelImportance(models_data.best_model, window_sz, true, n_dim);

%%%%%%%%%%%%%%%%%%%%%%%%% Plot Importance per dimension
%%%%%%%%%%%%%%%%%%%%%%%%%
fig_path = sprintf('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/figures/scalp_maps/%s_%s', FEATURE, METHOD);
if ~isdir(fig_path)
    mkdir(fig_path);
end
for iDim = 1:n_dim
    figure_title = sprintf('%s_dim_%d', model_string, iDim);
    fig_filename = sprintf('Subject_%s_%s_dim_%d', SUBJECT, WINDOW, iDim);
    figure()
    tl = tiledlayout(2,5);
    title(tl, figure_title, 'Interpreter',  'none')
    for iFld = 1:length(channel_importance)
        nexttile(iFld)
        plot_topography(channel_names, channel_importance{iFld}(:,iDim));
    end
    set(gcf, 'Position', get(0, 'Screensize'));
    saveas(gcf,fullfile(fig_path, fig_filename), 'fig');
    saveas(gcf,fullfile(fig_path, fig_filename), 'png');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% Plot Combined Importance
%%%%%%%%%%%%%%%%%%%%%%%%%
% h = figure('visible','off');
% h = figure()
% h.Position = [0 0 1920 1080];
figure()
for iFld = 1:length(channel_importance)
    combined_channel_importance_folds(:,iFld) = mean(channel_importance{iFld}, 2);
end
combined_channel_importance = mean(combined_channel_importance_folds, 2);
plot_topography(channel_names, combined_channel_importance);
fig_filename = sprintf('Subject_%s_%s_combined', SUBJECT, WINDOW);
saveas(gcf,fullfile(fig_path, fig_filename), 'fig');
saveas(gcf,fullfile(fig_path, fig_filename), 'png');


