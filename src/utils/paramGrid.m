function [param_grid, counter] = paramGrid(param_grid, params, counter)
%%%%% PARAMGRID the miracle function

for iVal = 1:length(params{1, 2})
    param_grid(counter).(params{1, 1}) = params{1, 2}{iVal};
    if size(params,1) > 1
        [param_grid, counter] = paramGrid(param_grid, params(2:end,:), counter);
    end
    param_grid(counter + 1) = param_grid(counter);
    counter = counter+1;
end
% remove extra path after reaching a leaf of the tree
counter = counter-1;
param_grid(end) = [];
end
