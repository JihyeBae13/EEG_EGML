function [val_losses, val_losses_fold] = crossvalMetricClassifierGrid(data,...
                                                     cv_object,...
                                                     preprocess_model_class,...
                                                     preprocess_params,...
                                                     metric_model_class,...
                                                     metric_params, ...
                                                     classifier_model_class,...
                                                     classifier_params,...
                                                     cv_loss, ...
                                                     store_model_path)
% This functions compute crossvalidation for the given loss
% param_grid is a n_param x 2 cell array, where the first column contains
% the model parameter names to be computed
% 
preproces_param_grid = paramGrid([], preprocess_params, 1);
metric_param_grid = paramGrid([], metric_params, 1);
classifier_param_grid = paramGrid([], classifier_params, 1);

n_preprocess = length(preproces_param_grid);
n_metric = length(metric_param_grid);
n_classifier = length(classifier_param_grid);
n_folds = cv_object.NumTestSets;

val_losses_fold = zeros(n_preprocess, n_metric, n_classifier, n_folds);

for iFld = 1:cv_object.NumTestSets %%%%%%%%%%%%%%%%%%%%%%% loop over folds
    fprintf('Processing fold %d\n', iFld);
    data_fold = {data{1}(cv_object.training(iFld), :), data{2}(cv_object.training(iFld))};
    for iPre = 1:n_preprocess %%%%%%%%%%%%%%%%%%%% loop over preprocessors
        preprocessor = preprocess_model_class(preproces_param_grid(iPre));
        preprocessor.train(data_fold{1});
        data_prep = preprocessor.predict(data{1});
        prep_data_fold = {data_prep(cv_object.training(iFld), :), data_fold{2}};
        for iMtr = 1 : n_metric %%%%%%%%%%%%%%%% loop over metric learning
            metric = metric_model_class(metric_param_grid(iMtr));
            metric.train(prep_data_fold);
            data_metric = metric.predict(data_prep);
            metric_data_fold = {data_metric(cv_object.training(iFld), :), data_fold{2}};
            
            for iClf = 1 : n_classifier %%%%%%%%%%%% loop over classifiers
                classifier = classifier_model_class(classifier_param_grid(iClf));
                classifier.train(metric_data_fold);
                pred_fold = classifier.predict(data_metric(cv_object.test(iFld), :));
                
                % the cv ooss can use either class labels prediction or
                % classifier scores (for instance with ROC areas)
                true_label = data{2}(cv_object.test(iFld));
                val_losses_fold(iPre, iMtr, iClf, iFld) = cv_loss(true_label, pred_fold);
                
                classifier_models{iPre, iMtr, iClf} = classifier;
                
            end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loop over classifiers
            metric_models{iPre, iMtr}.metric = metric;
                
        end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loop over metric learning
        preprocessor_models{iPre} = preprocessor;
    end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loop over preprocessors
    %%%% Store models at each fold %%%%%
    model_filename = sprintf('models_subfold_%d', iFld);
    save(fullfile(store_model_path, model_filename), 'preprocessor_models',...
                                                     'metric_models',...
                                                     'classifier_models',...
                                                     'preproces_param_grid',...
                                                     'metric_param_grid',...
                                                     'classifier_param_grid');
    fprintf('Saved fold %d\n', iFld);
end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loop over folds

val_losses = mean(val_losses_fold, 4);
end 


