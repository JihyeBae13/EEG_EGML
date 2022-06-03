close all
clear all
clc
% add source code path
addpath(genpath('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/src/'));


%%% Test crossvalidation function on a single fold
data_fold_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/data/RawEEG_Features/Subject_C1/z000p085/run_1/';
fold_file = 'fold_1.mat';
data_fold = load(fullfile(data_fold_path, fold_file));

%% parameters of cross-validation
pca_params = {'d_y',  [50, 80, 120]};
ceml_params = {'d_y', [3 10 80]};
svm_params = {'svm_type',    [1];...
              'nu',          [0.05, 0.1, 0.2, 0.3, 0.5, 0.8];...
              'kernel_type', [0]};

X = data_fold.X_train;
X = reshape(X, [], size(X, 3));

data{1} = X';
data{2} = data_fold.l_train;

cv_object = data_fold.cv_subobj;
cv_loss = @(true_l, pred) 1-mean(true_l == pred.class_label);

store_model_path = '/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/results/cv_results/RawEEG_Features/Subject_B/m085z000/run_1/';

[val_losses, val_losses_fold] = crossvalMetricClassifierGrid(data,...
                                          cv_object,...
                                          @PCANormalizer,...
                                          pca_params,...
                                          @CEML,...
                                          ceml_params, ...
                                          @SVM,...
                                          svm_params,...
                                          cv_loss, ...
                                          store_model_path);