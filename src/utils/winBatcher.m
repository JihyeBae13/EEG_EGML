function [batches, labels, starts] = winBatcher(ds, n_batches, n_examples, ex_time, repeats)
%DEPRECIATED - separated into startIdxSelector, exampleWindowExtractor, and batcher
%functions
%winBatcher - produce batches using the windows given
% Usage: [batches, labels, starts (the start indices)] = winBatcher(dataset, numberOfBatches,
% numberOfExamplesPerBatch, lengthOfExample (in seconds), whether repeats are allowed)
%
% Enter a 0 in the last argument to prevent the same window from appearing in multiple batches.
%
%TODO: varargin to determine ratios - for now assume equal ratios
%TODO: varargin for pmf function
%Note: Always returns the specified number of examples in each batch.
%      Truncates off the last examples if the number of examples is not
%      divisible by the number of classes.
% William Plucknett, 2021

%%
%%% extract some necessary information %%%
ex_len = ex_time * ds.sampFreq;  % length of an example in samples per channel
n_channels = length(ds.chnames); % number of channels
n_features = n_channels * ex_len;% number of features per datapoint
NO_REPEAT_FLAG = (repeats == 0);
%%
%%% preallocate the batch datastructure %%%
batches = zeros(n_examples, n_features, n_batches); % example x feature x batch
labels = zeros(n_examples, n_batches);
%%
%%% get the indices that indicate class changes %%%
class_borders = extractClassTransitions(ds.labels);% holds both the left and right side boundaries of each class
                                        %%% note: the last entry cannot
                                        %%% actually be accessed ( equal to end+1 )
n_classes = length(class_borders) - 1; % last entry is not the start of a class
%%
%%% within each class, randomly select windows to sample from %%%
%n_per_class = ceil(n_examples / 3); % how many of each class should be in a batch
batch_win_idxs = winSelector(class_borders, n_batches, n_examples, n_classes); % the index into the class_windows n_examples x n_batches

%%
%%% extract examples from each previously selected window %%%
starts = [];    % hold all the starting indicies
for batch_num = 1:n_batches
    for ex_num = 1:n_examples
        ds_idx = batch_win_idxs(ex_num, batch_num);
        win = ds.class_windows(ds_idx, :);
        start_idx = randi([win(1), win(2) - ex_len + 1]); % TODO: replace randi with a generic random number generator
        while (sum(starts == start_idx) > 0) && NO_REPEAT_FLAG % prevent the same start_idx from being selected multiple times
            start_idx = randi([win(1), win(2) - ex_len + 1]);
        end
        starts = [starts; start_idx];
        end_idx = start_idx + ex_len - 1;
        % push into output variables %
        batches(ex_num, :, batch_num) = reshape(ds.data(start_idx:end_idx,:), 1, []);
        labels(ex_num, batch_num) = ds.labels(ds_idx);
    end
end

end %winBatcher

%% Helper functions
function batchedWindows = winSelector(class_borders, n_batches, n_examples, n_classes)
    batchedWindows = [];
    n_per_class = ceil(n_examples / n_classes);
    for batch_num = 1:n_batches % for each batch
        new_batch = idxSelector(class_borders, n_classes, n_per_class);
        batchedWindows = [batchedWindows, new_batch(1:n_examples)]; % the window indices for this batch are concatenated to the old ones
    end
end
function indices = idxSelector(class_borders, n_classes, n_per_class)
    indices = [];         % hold the new batch
    for class = 1:n_classes % for each class
        indices = [indices; ...
            randi([class_borders(class), class_borders(class+1)-1], ...
                  [n_per_class, 1]) ...
        ]; 
    end
end
function trans = extractClassTransitions(labels)
    class_start_idxs = [1; find(diff(labels)) + 1]; % starts of each index, plus 1 bc diff removes the first index
    trans = [class_start_idxs; size(labels, 1) + 1];
end