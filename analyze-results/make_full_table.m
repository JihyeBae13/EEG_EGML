clear all
% close all
clc

addpath(genpath('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/src'));
addpath('/home/lgsanchez/work/Code/libraries/libsvm/matlab/')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methods = {"euclidean",...
           "ceml",...
           "nca", ...
           "egml"};

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

%% 
% columns of table
MethodName = [];
FeatureName = [];
SubjectID = [];
WindowName = [];
RunID = [];
TestAccuracyMean = [];
TestAccuracyStd = [];


full_test_array_mean = zeros(length(feature_type),...
                             length(window_names),...
                             length(subject_id),...
                             length(methods),...
                             sim_params.n_runs);
full_test_array_std = zeros(length(feature_type),...
                             length(window_names),...
                             length(subject_id),...
                             length(methods),...
                             sim_params.n_runs);


for iMtd = 1:length(methods)
    METHOD_USED = methods{iMtd};
    %% cv pipeline parameters
    %%%%%%%%%%%%%%%% If using metric learning
    if any(strcmp(METHOD_USED, {'nca', 'ceml', 'egml'}))
        preprocess_params = {'d_y',  {300, []}};
        metric_params = {'d_y', {3, 10, 100}};
        classifier_params = {'svm_type',    {1};...
            'nu',          {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8};...
            'kernel_type', {0}};
        
        %%%%%%%%%%%%%%%% Eulcidean does a more thorough search for input dimension%
    elseif strcmp(METHOD_USED, 'euclidean')
        preprocess_params = {'d_y',  {100, 200, 300, 400, 500, []}};
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
    
    
    
    results_root_path = fullfile('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/', methods{iMtd});
    for iFt = 1:length(feature_type)
        results_path = fullfile(results_root_path, feature_type{iFt});
        assert(isfolder(results_path), 'results feature data path does not exist')
        for iSbj = 1:length(subject_id)
            % read results dir for subject
            results_subject_path = fullfile(results_path, sprintf('Subject_%s',subject_id{iSbj}));
            assert(isfolder(results_subject_path), 'results data path does not exist');
            
            for iWnd = 1:length(window_names)
                
                results_window_path = fullfile(results_subject_path, window_names{iWnd});
                assert(isfolder(results_window_path), 'results window path does not exist');
                
                for iRun = 1:sim_params.n_runs
                    
                    results_run_path = fullfile(results_window_path, sprintf('run_%d', iRun));
                    assert(isfolder(results_run_path), 'results run path does not exist')
                    
                    
                    fprintf('%s %s, %s, %s , % s \n', METHOD_USED, feature_type{iFt}, sprintf('Subject_%s',subject_id{iSbj}), window_names{iWnd}, sprintf('run_%d', iRun));
                    results_data = load(fullfile(results_run_path, 'all_val__test_losses.mat'));
                    fprintf('mean test accuracy %f, std %f\n', 1 -mean(results_data.test_losses_fold), std(results_data.test_losses_fold));
                    disp(results_data.test_losses_fold)
                    
                    % update columns of table
                    MethodName = cat(1, MethodName, METHOD_USED);
                    FeatureName = cat(1, FeatureName, feature_type{iFt});
                    SubjectID = cat(1, SubjectID, subject_id{iSbj});
                    WindowName = cat(1, WindowName, window_names{iWnd});
                    RunID = cat(1, RunID, sprintf('run_%d', iRun));
                    TestAccuracyMean = cat(1, TestAccuracyMean, 1 -mean(results_data.test_losses_fold));
                    TestAccuracyStd = cat(1, TestAccuracyStd, std(results_data.test_losses_fold));
                    
                    full_test_array_mean(iFt, iWnd, iSbj, iMtd, iRun) = 1 -mean(results_data.test_losses_fold);
                    full_test_array_std(iFt, iWnd, iSbj, iMtd, iRun) = std(results_data.test_losses_fold);
                    
                    
                end
            end
        end
    end
end

%% create table 

full_test_table = table(MethodName, FeatureName, SubjectID, WindowName, RunID, TestAccuracyMean, TestAccuracyStd);
writetable(full_test_table, '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/all_test_linear.xlsx','Sheet',1);

for iFt = 1 : length(feature_type)
    for iWd = 1 : length(window_names)
        test_array_mean = squeeze(full_test_array_mean(iFt, iWd, :, :, :));
        test_array_std = squeeze(full_test_array_std(iFt, iWd, :, :, :));
        makeNiceBar(test_array_mean, test_array_std, subject_id, methods);
        title(sprintf('%s %s', feature_type{iFt}, window_names{iWd}), 'Interpreter','none');
    end
end

function fh = makeNiceBar(subject_method_mean, subject_method_std, subject_list, method_list)
    fh = figure();
    h = bar(subject_method_mean, .8);
    hold on
    % Get group centers
    xCnt = get(h(1),'XData') + cell2mat(get(h,'XOffset')); % XOffset is undocumented!
    % Set individual ticks
    std_errors = reshape(subject_method_std', [],1);
   
    er = errorbar(xCnt(:), reshape(subject_method_mean', [],1), std_errors, std_errors);
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';  
    % Get group centers
    % Set individual ticks
    set(gca, 'XTickLabel', subject_list)
    legend(method_list, 'Location','southeast')
    grid on
end
% 
% figure()
% v = randi(20,12,3); 
% h = bar(v,.8); 
% % Get group centers
% xCnt = get(h(1),'XData') + cell2mat(get(h,'XOffset')); % XOffset is undocumented!
% % Create Tick Labels
% xLab = repmat({'p1','p2','p3'},1,numel(xCnt)/3); 
% % Set individual ticks
% set(gca, 'XTick', sort(xCnt(:)), 'XTickLabel', xLab)