% run_cross_validation.m 
% This is a script that is called from another script that trains or evaluates
% a given metric.
% 
% Luis Gonzalo Sanchez Giraldo

for iSbj = 1:length(subject_id)
    fprintf('Cross validating folds for subject %s, window %s\n', subject_id{iSbj}, window_name);
    % read  data dir for subject
    subject_path = fullfile(data_path, sprintf('Subject_%s',subject_id{iSbj}));
    assert(isfolder(subject_path), 'data path does not exist');
        
    % create  results dir for subject
    results_subject_path = fullfile(results_path, sprintf('Subject_%s',subject_id{iSbj}));
    if ~isfolder(results_subject_path)
        mkdir(results_subject_path);
    end
    
    window_path = fullfile(subject_path, window_name);
    assert(isfolder(window_path), 'window path does not exist');
    
    results_window_path = fullfile(results_subject_path, window_name);
    if ~isfolder(results_window_path)
        mkdir(results_window_path);
    end

    
    for iRun = 1:sim_params.n_runs
        fprintf('\tRun %d\n', iRun)
        run_path = fullfile(window_path, sprintf('run_%d', iRun));
        assert(isfolder(run_path), 'run path does not exist')
        results_run_path = fullfile(results_window_path, sprintf('run_%d', iRun));
        if ~isfolder(results_run_path)
            mkdir(results_run_path)
        end
        
        for iFld = 1:sim_params.n_folds
            foldname = sprintf('fold_%d', iFld);
            data_fold = load(fullfile(run_path, foldname));
            results_fold_path = fullfile(results_run_path, foldname);
            if ~isfolder(results_fold_path)
                mkdir(results_fold_path)
            end
            fprintf('\t\tFold %d\n', iFld)
            
            X = data_fold.X_train;
            X = reshape(X, [], size(X, 3));
            
            data{1} = X';
            data{2} = data_fold.l_train;
            
            cv_object = data_fold.cv_subobj;
            store_model_path = results_fold_path;
            %%% save fold data
            [val_losses, val_losses_fold] = crossvalMetricClassifierGrid(data,...
                                                                         cv_object,... 
                                                                         preprocessor,...
                                                                         preprocess_params,... 
                                                                         metric,...
                                                                         metric_params, ...
                                                                         classifier,...
                                                                         classifier_params,...
                                                                         cv_loss, ...
                                                                         store_model_path);
            % save losses
            save(fullfile(store_model_path, 'cv_loss.mat'), 'val_losses', 'val_losses_fold');

        end
        
    end
end