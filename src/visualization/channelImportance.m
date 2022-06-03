function channel_importance = channelImportance(models, window_sz, per_dim, n_dims)
if ~exist('per_dim', 'var')
    per_dim = 0;
end
for fold = 1:length(models)
    % compute importance per channel per feature as the norm of the time
    % window
    T = models(fold).preprocessor.trainable_params.T;
    A = models(fold).metric.trainable_params.A;
    if per_dim == 0
        channel_importance(:,fold) = channelImportanceAllDims(T, A, window_sz);
    else
        channel_importance{fold} = channelImportancePerDim(T, A, window_sz, n_dims);
    end
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

