%svm_grid_search_test - a script to test the svm_grid_search function
% pathToData = 'F:\MetricLearning_Intention\FTA_Features\SubjectB_win_m06m02_ch21_20210618T152208_FTA';
% dirInfo = dir(pathToData);
% dirInfo(ismember( {dirInfo.name}, {'.', '..'})) = [];
% dirInfo = dirInfo([dirInfo.isdir]);
% 
% cd(pathToData)
clear;

gamma = 2.^linspace(-15, 3, 10);
cost = 2.^linspace(-5, 15, 11);

% merge the runs and folds
mergeRunsAndFolds % script

%% Euclidean
euc_results = svm_grid_search(gamma, cost, 'merged_Euclidean.mat');
%% CEML
CEML_results = svm_grid_search(gamma, cost, 'merged_CEML.mat');
%% NCA
NCA_results = svm_grid_search(gamma, cost, 'merged_NCA.mat');

%% Report
simparams = load('simParameters.mat');
disp("------------------------------------------------------------")
disp("|                      REPORT                              |")
disp("------------------------------------------------------------")
disp("Parameters:")
fprintf("(d_y, sigma): (%d, %d)\n", simparams.d_y, simparams.sigma)
fprintf("pcs: %d\n", simparams.pcs)
disp("------------------------------------------------------------")
disp("Euclidean:")
fprintf("Acc + std: %d +/- %d\n", mean([euc_results.test_accs{:}]), std([euc_results.test_accs{:}]));
disp("------------------------------------------------------------")
disp("CEML:")
fprintf("Acc + std: %d +/- %d\n", mean([CEML_results.test_accs{:}]), std([CEML_results.test_accs{:}]));
disp("------------------------------------------------------------")
disp("NCA:")
fprintf("Acc + std: %d +/- %d\n", mean([NCA_results.test_accs{:}]), std([NCA_results.test_accs{:}]));

save('nonlinear_svm_results.mat', 'euc_results', 'CEML_results', 'NCA_results');

beep