function [Y] = unbatcher(X)
%unbatcher - unbatches a batched set of examples
%  Usage: Y = unbatcher(X)
%  where X is three dimensional: n_examplespbatch x n_features x n_batches
%  and Y is two dimensional: n_examples x n_features
%       where n_examples = n_examplespbatch * n_batches
%  This function concatenates the batches vertically along the examples
%  dimension.
%
% William Plucknett, 2021

n_examplespbatch = size(X, 1);
n_features = size(X, 2);
n_batches = size(X, 3);

Y = zeros(n_examplespbatch * n_batches, ...
    n_features);
for b_idx = 1:n_batches
    %%%start and end indices of this batch in the Y matrix%%%
    Y_start = (b_idx - 1) * n_examplespbatch + 1;
    Y_end = Y_start + n_examplespbatch - 1;
    %%%import to y%%%
    Y(Y_start:Y_end,:) = X(:,:,b_idx);
end

end
