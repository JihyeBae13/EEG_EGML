% script_test_trainbel_model_classes.m 
% This is a demo script that illustrates the usage of some of the trainable 
% classes for implemented in here. 
% 
% Make sure you have added the appropirate paths before running the script.
% 
% Luis Gonzalo Sanchez Giraldo
% June 2023 

close all
clear all
clc

%% IMPORTANT!  need to edit these lines to add the appropriate paths %%%%%%%%%%%
%%%%%% Directory where the repository has been downloaded
addpath(genpath('<your-path-to>/metric-learning-premovement/src/'));
%%%%%% Directory that contains your LIBSVM installation
addpath('<your-path-to>/libsvm/matlab');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% load some test data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load iris_dataset.mat;
% need to convert to the data form for model object
% data for model object is a cell array with first element the inputs and the 
% second element the labels or targets
data{1} = irisInputs';
data{2} = irisTargets'*[1;2;3];


%% create and test a Normalizer object %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Example 1: normalize each dim individually
normalizer_hyperparams.per_dim = true;
normalizer_perdim = Normalizer(normalizer_hyperparams);
normalizer_perdim.train(data{1});
data_norm = normalizer_perdim.predict(data{1});
figure(), plot(var(data_norm));

% Example 2: normalize based on total variance
normalizer_hyperparams.per_dim = false; 
normalizer_totalvar = Normalizer(normalizer_hyperparams);
normalizer_totalvar.train(data{1});
data_norm = normalizer_totalvar.predict(data{1});
figure(), plot(var(data_norm));


%% create and test a PCANormalizer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pca_hyperparams.d_y = 2;
pca_normalizer = PCANormalizer(pca_hyperparams);
pca_normalizer.train(data{1});
data_norm = pca_normalizer.predict(data{1});
figure(), plot(var(data_norm));
figure(), gscatter(data_norm(:,1), data_norm(:,2), data{2});


%% create and train CEML %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_params = struct('mu_in', 0.01,...
                      'mu_fin', 0.01,...
                      'n_iter', 1000,...
                      'tol', 1e-10,...
                      'beta1', 0.9,...
                      'beta2', 0.99,...
                      'epsilon', 1e-8);
ceml_hyperparams.d_y = 2;
ceml = CEML(ceml_hyperparams);
ceml.train(data, train_params);
Y = ceml.predict(data{1});
figure(), gscatter(Y(:,1), Y(:,2), data{2});


%% create and train NCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nca_hyperparams.d_y = 2;
nca = NCA(nca_hyperparams);
nca.train(data, train_params);
Y = nca.predict(data{1});
figure(), gscatter(Y(:,1), Y(:,2), data{2});


%% create and train SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
svm_hyperparams = struct('svm_type', 0,...
                    'kernel_type', 3, ...
                    'gamma', 0.4,...
                    'cost', 1.5,...
                    'probability_estimates', 1,...
                    'weight', [0.5, 0.5, 0.5],...
                    'verbose', 1);
svm = SVM(svm_hyperparams);
svm.train(data);
pred = svm.predict(data{1});
figure(), plot(pred.class_label)