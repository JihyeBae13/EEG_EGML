% script_generate_folds.m
% Generates train and validation folds from EEG data from 
% To run this script you must have Downloaded the dataset from
%   Kaya, M., Binli, M., Ozbay, E. et al. A large electroencephalographic motor 
%   imagery dataset for electroencephalographic brain computer interfaces. 
%   Sci Data 5, 180211 (2018). https://doi.org/10.1038/sdata.2018.211
%
% To access the data https://doi.org/10.6084/m9.figshare.c.3917698.v1
% For experiments, you must download the following 3 files:
%   - FREEFORMSubjectB1511112StLRHand.mat
%   - FREEFORMSubjectC1512082StLRHand.mat
%   - REEFORMSubjectC1512102StLRHand.mat
% 
% Create a dfirectory called 'originalKayaEEG' and strore the three files above
% in this directory. 
% 
% Also, you must set the paths where the code is located and the paths where the
% data folds will be stored
%
% After running this code as it is, the following directories are created inside
% the data directory:
%   - FTA_Features
%   - FTA5_Features
%   - RawEEG_Features
%
% Luis Gonzalo Sanchez Giraldo

clear all
close all
clc

%% IMPORTANT:
% Set the path to  the code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make sure you have set the right paths in this script
set_paths;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% here we fix the partitions to compare all methods %%%%%%%%%%%%%%%%%%%
data_root_path = fullfile(root_path,'/data/');


original_data_path = fullfile(data_root_path, 'originalKayaEEG');
feature_type = {'FTA_Features',...
                'FTA5_Features',...
                'RawEEG_Features'};

subject_file = {'FREEFORMSubjectB1511112StLRHand.mat',...  % subject B
                'FREEFORMSubjectC1512082StLRHand.mat',...  % subject C1
                'FREEFORMSubjectC1512102StLRHand.mat'};    % subject C2

subject_id = {'B', 'C1', 'C2'};  


sim_params.feature_type = feature_type{2};
sim_params.pre_onset = false;
sim_params.n_folds = 10;
sim_params.n_subfolds = 10;
sim_params.n_runs = 1;
sim_params.fta_max_freq = 5; % in Hz
 

% For this experiment, we are generating data windows of 850 milliseconds
if sim_params.pre_onset
    sim_params.wd_str_t = -0.85; % in seconds
    sim_params.wd_end_t = 0;
else %post-onset
    sim_params.wd_str_t = 0;
    sim_params.wd_end_t = 0.85;
end

%% 
data_path = fullfile(root_path, sim_params.feature_type);
if ~isfolder(data_path)
    mkdir(data_path);
end

window_name = sprintf('%s%03d%s%03d', getSignPrefix(sim_params.wd_str_t),...
                                      uint32(abs(sim_params.wd_str_t * 100)),...
                                      getSignPrefix(sim_params.wd_end_t),...
                                      uint32(abs(sim_params.wd_end_t *100)));

for iSbj = 1:length(subject_id)
    fprintf('Generating folds for subject %s, window %s\n', subject_id{iSbj},...
                                                            window_name);
    % creates dir for subject if not existing
    subject_path = fullfile(data_path, sprintf('Subject_%s',subject_id{iSbj}));
    if ~isfolder(subject_path)
        mkdir(subject_path);
    end
    
    original_data = load(fullfile(original_data_path, subject_file{iSbj}), 'o');
    [X, labels] = extractTrialWindows(original_data.o, sim_params);
    % compute Fourier features if needed
    if strcmp(sim_params.feature_type, 'FTA_Features')
        X = computeFTA(X);
    elseif strcmp(sim_params.feature_type, 'FTA5_Features')
        wd_size = sim_params.wd_end_t - sim_params.wd_str_t;
        n = ceil(sim_params.fta_max_freq * wd_size);
        X = computeFTA(X, n);
    end
    
    window_path = fullfile(subject_path, window_name);
    if ~isfolder(window_path)
        mkdir(window_path);
    end
    
    % create folds containing the windows for training and for validation
    for iRun = 1:sim_params.n_runs
        fprintf('\tRun %d\n', iRun)
        run_path = fullfile(window_path, sprintf('run_%d', iRun));
        if ~isfolder(run_path)
            mkdir(run_path)
        end
        cv_obj = cvpartition(size(X,3), 'KFold', sim_params.n_folds);
        for iFld = 1:sim_params.n_folds
            fprintf('\t\tFold %d\n', iFld)
            X_train = X(:, :, cv_obj.training(iFld)); 
            X_test = X(:, :, ~cv_obj.training(iFld));
            l_train = labels(cv_obj.training(iFld));
            l_test = labels(~cv_obj.training(iFld));
            %%% save fold data
            foldname = sprintf('fold_%d', iFld);
            cv_subobj = cvpartition(size(X_train,3), 'KFold',... 
                                                    sim_params.n_subfolds);
            save(fullfile(run_path, foldname),  'X_train',...
                                                'X_test',... 
                                                'l_train',...
                                                'l_test',...
                                                'cv_subobj');
        end
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sign_prefix = getSignPrefix(x)
% sign_prefix:  prefix to name data based on time positions
%   p : plus
%   m : minus
%   z : zero
    if x > 0
        sign_prefix = 'p';
    elseif x < 0
        sign_prefix = 'm';
    elseif x == 0
        sign_prefix = 'z';
    end
end