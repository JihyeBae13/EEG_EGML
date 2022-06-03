function[results] = svm_linear(data)
dataset = data;
load(dataset)
n_runs = size(snapshots, 1);
n_folds = size(snapshots, 2); % per run

%%% Some outputs
results.test_accs = cell(n_runs, n_folds);
results.validationAcc = cell(n_runs, n_folds);
results.validation_CM = cell(n_runs, n_folds);
results.CM = cell(n_runs, n_folds);
% for diagnostics %
D_IDENTS = 0; % number of times the identity matrix subbed an empty transformation matrix

for run_idx = 1:1:n_runs
    fprintf("Run number %d\n", run_idx);
    for fold_idx = 1:n_folds
        %%%
        fprintf("Fold number %d\n", fold_idx);
        ss = snapshots{run_idx, fold_idx}; % this snapshot
        [Xt, Xv, XT, yt, yv, yT] = setSplit(ss); % split into train, valid, test sets
        A = ss.A; % transformation matrix (what happens if this is empty?)
        if isempty(A) % this happens if the transformation matrix is empty (e.g. euclidean)
            fprintf("Transformation matrix is empty. Substituting in the identity matrix.")
            A = eye(size(Xt, 2));
            D_IDENTS = D_IDENTS + 1;
        end % end if A is empty
        % transform the data
        Xt = Xt * A;
        Xv = Xv * A;
        XT = XT * A;
        
        disp("Training");
%         svm_model = svmtrain(yt, Xt, '-t 0'); % linear % libsvm
        template_SVM = templateSVM('Standardize',true);
        svm = fitcecoc(Xt,yt,'Learners',template_SVM,'ObservationsIn','rows');
        %validating
        disp("Validating")
%         [pred, acc, ~] = svmpredict(yv, Xv, svm_model); % evaluate performance % libsvm
        pred = predict(svm, Xv, 'ObservationsIn', 'rows'); % fitcecoc
        acc = sum(pred == yv) / length(yv); % fitcecoc
        results.validationAcc{run_idx, fold_idx} = acc(1);
        results.validation_CM{run_idx, fold_idx} = confusionmat(yv, pred);
        fprintf("Validation acc: %d\n", acc);
        %testing
        disp("Testing")
%         [pred, acc, ~] = svmpredict(yT, XT, svm_model);
        pred = predict(svm, XT, 'ObservationsIn', 'rows'); % fitcecoc
        acc = sum(pred == yT) / length(yT); % fitcecoc
        results.test_accs{run_idx, fold_idx} = acc(1);
        results.CM{run_idx, fold_idx} = confusionmat(yT, pred);
        fprintf("Testing acc: %d\n", acc);
    end % end fold
end % end run
if D_IDENTS > 0
        disp("The transformation matrix A was empty for at least one instance.")
    fprintf("\tIf this is not the Euclidean algorithm, ensure that the\n")
    fprintf("\ttransformation matrices were saved properly.\n");
end % end D_IDENTS
end % end function
 %% functions
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