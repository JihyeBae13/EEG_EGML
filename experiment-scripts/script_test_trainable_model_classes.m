close all
clear all
clc

addpath(genpath('/home/lgsanchez/work/Code/research/bci-eeg/metric-learning-premovement/src/'));

%% load some test data
load iris_dataset.mat;
% need to convert to the data form for model object
data{1} = irisInputs';
data{2} = irisTargets'*[1;2;3];

%% create and test Normalizer
 % normalize each dim individually
normalizer_hyperparams.per_dim = true;
normalizer_perdim = Normalizer(normalizer_hyperparams);
normalizer_perdim.train(data{1});
data_norm = normalizer_perdim.predict(data{1});
figure(), plot(var(data_norm));

normalizer_hyperparams.per_dim = false; % normalize based on total variance
normalizer_totalvar = Normalizer(normalizer_hyperparams);
normalizer_totalvar.train(data{1});
data_norm = normalizer_totalvar.predict(data{1});
figure(), plot(var(data_norm));

%% create and test PCANormalizer
pca_hyperparams.d_y = 2;
pca_normalizer = PCANormalizer(pca_hyperparams);
pca_normalizer.train(data{1});
data_norm = pca_normalizer.predict(data{1});
figure(), plot(var(data_norm));
figure(), gscatter(data_norm(:,1), data_norm(:,2), data{2});
%% create and train CEML
train_params = struct('mu_in', 0.01,...
                      'mu_fin', 0.01,...
                      'n_iter', 1000,...
                      'tol', 1e-10,...
                      'beta1', 0.9,...
                      'beta2', 0.99,...
                      'epsilon', 1e-8);
ceml_hyperparams.d_y = 2;
% ceml = CEML(2, 1, 1.01);
ceml = CEML(ceml_hyperparams);

ceml.train(data, train_params);
Y = ceml.predict(data{1});
figure(), gscatter(Y(:,1), Y(:,2), data{2});


%% create and train NCA
nca_hyperparams.d_y = 2;
nca = NCA(nca_hyperparams);
nca.train(data, train_params);
Y = nca.predict(data{1});
figure(), gscatter(Y(:,1), Y(:,2), data{2});

%% create and train SVM
addpath('/home/lgsanchez/work/Code/libraries/libsvm/matlab');
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