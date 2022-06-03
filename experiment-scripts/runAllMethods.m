%% RUN ALL THE METHODS
% Modified by William Plucknett, 2/11/2021 to save more frequently.
% Modified by William Plucknett, 2/9/2021 to save confusion matrices
%   and to save the transformation matrices
num_runs = 1;   % FIXME: reset to 10
num_folds = 4;
knn_neighbor_size = 4;

%NOTE: cm refers to confusion matrix, m refers to metric matrix
%DATA IS SAVED as a struct matrix. 

for i  = 1:num_runs...
    fprintf('Beginning run %d\n', i);
    for j = 1:length(datasets)
        this_set = datasets(j);
        X = this_set.data;
        labels = this_set.labels;
        name = this_set.name;
        
        %name = sprintf('%s_run%d', this_set.name, i); % for greater debugging  power
        
        % for future reference
        acc(j,i).name = name;
        CM(j,i).name = name;
        M(j,i).name = name;
        A(j,i).name = name;
        Y(j,i).name = name;
        ts(j,i).name = name;
        
        %%% Conditional Entropy Metric Learning
        d_y = 120;
        sigma = 12;
        alpha = 1.01;
        [acc(j,i).CEML, ~, CM(j,i).CEML, ~, A(j,i).CEML, Y(j,i).CEML, ts(j,i).CEML] = CrossValidateKNN(labels, X, @(labels, X) CondEntropyMetricLearning(X, labels, d_y, sigma, alpha), num_folds, knn_neighbor_size);
        fprintf('CEML kNN cross-validated accuracy = %f\n', acc(j,i).CEML);
        
        %%% Information Theoretic Metric Learning (Davis 2007)
%         [acc(j,i).ITML, ~, CM(j,i).ITML, ~, A(j,i).ITML, ~, ts(j,i).ITML] = CrossValidateKNN(labels, X, @(labels,X) MetricLearningAutotuneKnn(@ItmlAlg, labels, X), num_folds, knn_neighbor_size);
%         fprintf('ITML kNN cross-validated accuracy = %f\n', acc(j,i).ITML);
        
        %% Neigbourghood Component Analys (Goldberger 2004)
%         [acc(j,i).NCA, ~, CM(j,i).NCA, ~, A(j,i).NCA, ~, ts(j,i).NCA] = CrossValidateKNN(labels, X, @(labels, X) ncaWrap(X, labels, 3), num_folds, knn_neighbor_size);
%         fprintf('NCA kNN cross-validated accuracy = %f\n', acc(j,i).NCA);
        
        %%% Maximally Collapsing Metric Learning (Globerson 2005)
%         [acc(j,i).MCML, ~, CM(j,i).MCML, ~] = CrossValidateKNN(labels, X, @(labels, X) mcmlWrap(X, labels, 3), num_folds, knn_neighbor_size);
%         fprintf('MCML kNN cross-validated accuracy = %f\n', acc(j,i).MCML);
        
        
        %%% Large Margin Nearest Neighbor (Weinberger 2005)
%         [acc(j,i).LMNN, ~, CM(j,i).LMNN, ~, A(j,i).LMNN, Y(j,i).LMNN, ts(j,i).LMNN] = CrossValidateKNN(labels, X, @(labels, X) lmnnWrap(X, labels, d_y), num_folds, knn_neighbor_size);
%         fprintf('LMNN kNN cross-validated accuracy = %f\n', acc(j,i).LMNN);
        
        
        %%% Inverse Covariance (Whitening)
%         [acc(j,i).invCov, ~, CM(j,i).invCov, ~] = CrossValidateKNN(labels, X, @(labels, X) invCovWrap(X, labels), num_folds, knn_neighbor_size);
%         fprintf('InvCov kNN cross-validated accuracy = %f\n', acc(j,i).invCov);
        
        
        %%% Eucledian Distance
%         [acc(j,i).Euclidean, ~, CM(j,i).Euclidean, ~, A(j,i).Euclidean, ~, ts(j,i).Euclidean] = CrossValidateKNN(labels, X, @(labels, X) euclideanWrap(X, labels), num_folds, knn_neighbor_size);
%         fprintf('Euclidean kNN cross-validated accuracy = %f\n', acc(j,i).Euclidean);
        
        %%%save
        %TODO: change file names
        save results/CEML3CLASS_results_acc acc
        save results/CEML3CLASS_results_cm  CM
%         save results/3CLASS_LMNN_results_m   M
        save results/CEML3CLASS_results_a   A
%        save results/3CLASS_results_y   Y
        save results/CEML3CLASS_results_testSet ts
    
    end % for dataset
end % for run
