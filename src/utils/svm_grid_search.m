function [results] = svm_grid_search(gamma_vals, cost_vals, data)
% svm_evaluate - script to evaluate the performance of svms with various
% parameters on the dataset
%
% Usage: results = svm_grid_search(gamma_vals, cost_vals, data)
% where gamma_vals and cost_vals are vectors
% Data is a link to the dataset
n_gamma = length(gamma_vals);
n_cost = length(cost_vals);
n_opt = n_gamma * n_cost;

[gamma_grid, cost_grid] = meshgrid(gamma_vals, cost_vals);
opt_acc = zeros(length(gamma_vals), length(cost_vals)); % acc per option set
cm_v = cell(length(gamma_vals), length(cost_vals)); % the confusion matrix for each option set

%dataset = "D:\thewi\Documents\Research\Project1-Metric Learning and Premovement Intention\Results Collection\Baseline\merged snapshots\Bm06m02\CEML_results.mat";
dataset = data;
load(dataset)
n_runs = size(snapshots, 1);
n_folds = size(snapshots, 2); % per run

%%% Some outputs
% store results %
results.gamma = gamma_vals;
results.cost = cost_vals'; % transpose to emphasize that it varies on row of the meshgrids
results.test_accs = cell(n_runs, n_folds);
results.grid_accs = cell(n_runs, n_folds);
results.validationAcc = cell(n_runs, n_folds);
results.validation_CM = cell(n_runs, n_folds);
results.bestOpts = cell(n_runs, n_folds);
results.CM = cell(n_runs, n_folds);
% for diagnostics %
D_IDENTS = 0; % number of times the identity matrix substituted an empty transformation matrix

for run_idx = 1:1:n_runs
    fprintf("Run number %d\n", run_idx);
    for fold_idx = 1:n_folds
        %%%
        fprintf("Fold number %d\n", fold_idx);
        ss = snapshots{run_idx, fold_idx}; % this snapshot
        [Xt, Xv, XT, yt, yv, yT] = setSplit(ss); % split into training, validation, test sets
        A = ss.A; % transformation matrix (what happens if this is empty?)
        if isempty(A) % this happens if the transformation matrix is empty (e.g. euclidean)
            fprintf("Transformation matrix is empty. Substituting in the identity matrix.")
            A = eye(size(Xt, 2));
            D_IDENTS = D_IDENTS + 1;
        end % end if A is empty
        % transform the features
        Xt = Xt * A;
        Xv = Xv * A;
        XT = XT * A;
        parfor opt_idx = 1:n_opt %n_opts % cycle through all of the options %PARFOR
            fprintf("Option set %d\n", opt_idx);
            svmOptions = ['-t 2 ', '-g ', num2str(gamma_grid(opt_idx), '%0.10f'), ' -c ', num2str(cost_grid(opt_idx))];
            %training
            disp(sprintf("Training: param %s", svmOptions))
            svm_model = svmtrain(yt, Xt, svmOptions);
%             svm_model.Parameters
            %validating
            disp("Validating")
            [pred, acc, dv] = svmpredict(yv, Xv, svm_model); % evaluate performance on validation set
            opt_acc(opt_idx) = acc(1); % accuracy for this set of options
            cm_v{opt_idx} = confusionmat(yv, pred);
        end % end parfor options
        results.grid_accs{run_idx, fold_idx} = opt_acc;
        results.validation_CM{run_idx, fold_idx} = cm_v;
        %testing
        [results.validationAcc{run_idx, fold_idx}, lin_idx] = max(opt_acc, [], 'all', 'linear');
        results.bestOpts{run_idx, fold_idx} = ['-t 2 ', '-g ', num2str(gamma_grid(lin_idx)), ' -c ', num2str(cost_grid(lin_idx))];
        model = svmtrain(yt, Xt, results.bestOpts{run_idx, fold_idx});
        disp("Testing")
        [pred, acc, dv] = svmpredict(yT, XT, model);
        results.test_accs{run_idx, fold_idx} = acc(1);
        results.CM{run_idx, fold_idx} = confusionmat(yT, pred);
    end % end fold
    
end % end run
if D_IDENTS > 0
    disp("The transformation matrix A was empty for at least one instance.")
    fprintf("\tIf this is not the Euclidean algorithm, ensure that the\n")
    fprintf("\ttransformation matrices were saved properly.\n");
end
end % end function

%% functions
function [t, v, T, yt, yv, yT] = setSplit(ss)
%setSplit - split the dataset into the training, validation, and test sets
%   The test set of the input ss is split to produce the validation set.
% Usage: [train, valid, test] = setSplit(snapshot)
    t = ss.Xt; % training set
    yt = ss.yt;% training labels
    
    rp = randperm(size(ss.XT, 1)); % random permutation to randomize the test set prior to split
    vT = ss.XT(rp,:); % set combining the validation and test sets
    yvT = ss.yT(rp, :); % combined validation and test labels
    
    n_validation = floor(size(vT, 1) / 2); % number of examples to be in the validation set
    v = vT(1:n_validation, :);      % validation set
    yv = yvT(1:n_validation, :);    % validation labels
    T = vT(n_validation+1:end, :);  % test set
    yT = yvT(n_validation+1:end, :);% test labels
end