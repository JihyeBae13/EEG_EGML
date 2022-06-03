%svm_linear_automator
%   automates the linear svms
%
%
% William Plucknett, 2021

parentFolder = 'F:\MetricLearning_Intention\FTA_Features\m06m02';
dirInfo = dir(parentFolder);
dirInfo(ismember( {dirInfo.name}, {'.', '..'})) = [];
subfolders = dirInfo([dirInfo.isdir]);

outputTable = initializeOutputTable();

for folder_idx = 1:length(subfolders)
    this_folder = subfolders(folder_idx).name;
    dataPath = fullfile(parentFolder, this_folder);
    
    [newRow] = svm_linear_wrapper(dataPath); 
    
    outputTable(end+1,:) = newRow;
    
    writetable(outputTable, fullfile(parentFolder,'updated_fitcecoc_linear_automated_table.xlsx'))
end
%% Functions
function [outTble] = initializeOutputTable()
    colTypes = {'string', ... % Type
        'string', ... % Subject
        'string', ... % Window
        'int64', ...  % d_y
        'double', ... % sigma
        'double', ... % Euclidean verification accuracy
        'double', ... % Euclidean verification std
        'double', ... % Euclidean test accuracy
        'double', ... % Euclidean test std
        'double', ... % CEML verification accuracy
        'double', ... % CEML verification std
        'double', ... % CEML test accuracy
        'double', ... % CEML test std
        'double', ... % NCA verification accuracy
        'double', ... % NCA verification std
        'double', ... % NCA test accuracy
        'double', ... % NCA test std
        'string' % Folder
        };
    varNames = {'Type', ...
        'Subject', ...
        'Window', ...
        'd_y', ...
        'sigma', ...
        'Euclidean_verification_accuracy', ...
        'Euclidean_verification_std', ...
        'Euclidean_test_accuracy', ...
        'Euclidean_test_std', ...
        'CEML_verification_accuracy', ...
        'CEML_verification_std', ...
        'CEML_test_accuracy', ...
        'CEML_test_std', ...
        'NCA_verification_accuracy', ...
        'NCA_verification_std', ...
        'NCA_test_accuracy', ...
        'NCA_test_std', ...
        'Folder'
        };
    outTble = table('Size', [0,length(colTypes)], ...
        'VariableTypes', colTypes, ...
        'VariableNames', varNames ...
    );
end