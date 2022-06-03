%svm_grid_search_wrapper - a function modified from a script to test the svm_grid_search function
function [tableRow] = svm_grid_search_wrapper(dataFolder)
originalDir = pwd;
cd(dataFolder)

gamma = 2.^linspace(-15, 3, 10);
cost = 2.^linspace(-5, 20, 11);

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
cd(originalDir)

[~, fileName, ~] = fileparts(dataFolder);
[type, subj, window] = stripFolderName(fileName);
tableRow = {type, ...
    subj,...
    window,...
    simparams.d_y, simparams.sigma, ...
    mean([euc_results.validationAcc{:}]), std([euc_results.validationAcc{:}]),...% TODO : euc validation
    mean([euc_results.test_accs{:}]), std([euc_results.test_accs{:}]), ...
    mean([CEML_results.validationAcc{:}]), std([CEML_results.validationAcc{:}]),...% TODO : CEML validation
    mean([CEML_results.test_accs{:}]), std([CEML_results.test_accs{:}]), ...
    mean([NCA_results.validationAcc{:}]), std([NCA_results.validationAcc{:}]), ...% TODO : NCA validation
    mean([NCA_results.test_accs{:}]), std([NCA_results.test_accs{:}]), ...
    dataFolder
    };
end

%% Helper function
function [type, subj, window] = stripFolderName(folderName)
% folder name is "subject{}_win_{}_ch21_TIME_{type}
folderParts = split(folderName, '_');

type = folderParts{end};
subj = folderParts{1}(8:end);
window = folderParts{3};
end
