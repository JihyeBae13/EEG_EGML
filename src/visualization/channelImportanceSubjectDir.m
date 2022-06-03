function channel_importance = channelImportanceSubjectDir(subject_dir, sim_params, per_dim, n_dims);
if ~exist('per_dim', 'var')
    per_dim = 0;
end
fold = 1;
for iFl = 1:length(subject_dir)
    [~, ~, ext] = fileparts(subject_dir(iFl).name); 
    if any(strcmp(subject_dir(iFl).name, {'.', '..'})) || isfolder(fullfile(subject_dir(iFl).folder, subject_dir(iFl).name)) || ~strcmp(ext, '.mat')
        continue
    end
    fprintf('Reading %s\n', subject_dir(iFl).name)
    fold_data = load(fullfile(subject_dir(iFl).folder, subject_dir(iFl).name));
    % compute importance per channel per feature as the norm of the time
    % window
    if sim_params.is_fta
        window_sz = size(fold_data.T,1)/length(sim_params.chnames);
    else
        window_sz = (sim_params.samp_freq * sim_params.ex_t);
    end
    
    if per_dim == 0
        channel_importance(:,fold) = channelImportanceAllDims(fold_data.T, fold_data.A, window_sz);
    else
        channel_importance{fold} = channelImportancePerDim(fold_data.T, fold_data.A, window_sz, n_dims);
    end
    fold = fold + 1;
end
end


function [channel_importance] = channelImportanceAllDims(T, A, window_sz)
TA = T*A;
TA = reshape(TA, [window_sz, size(TA,1)/window_sz, size(TA,2)]);
channel_norm = sqrt(squeeze(sum(sum(TA.^2,1), 3)));
channel_importance = channel_norm / sum(channel_norm);
end


function [channel_importance] = channelImportancePerDim(T, A, window_sz, n_dims)
n_dims = min(n_dims, size(A, 2));
[A, ~, ~] = svds(A, n_dims);
TA = T*A;
TA = reshape(TA, [window_sz, size(TA,1)/window_sz, size(TA,2)]);
channel_norm = sqrt(squeeze(sum(TA.^2,1)));
channel_importance = bsxfun(@times, channel_norm, 1./ sum(channel_norm,1));
end

