% set_paths.m
% This simple script set paths to run the experiments scripts. 
% You can alternatively add these paths permanently to MATLAB, but we have not
% choose to do so as users may want to modify functionalities and directory
% locations for comparisons with methods of their own.
% 
% Luis Gonzalo Sanchez Giraldo 2023 

%% Set the path where the metric-learning-ptemovement is located
root_path = "< your-path-to >/metric-learning-premovement/";
addpath(genpath(fullfile(root_path, '/src')));
%% Set the path to your installation of LIBSVM
addpath('< your-path-to >/libsvm/matlab/')
% Set path where the data will be located. This same data folder, where you 
% placed the directory "originalKayaEEG" that contains the .mat files with the 
% EEG data
data_root_path = '<your-path-to>/metric-learning-premovement/data/';

%% scalp map visualization
% For visualization, we use the plot_topography library
% Víctor Martínez-Cagigal (2023). Topographic EEG/MEG plot 
% (https://www.mathworks.com/matlabcentral/fileexchange/72729-topographic-eeg-meg-plot), 
% MATLAB Central File Exchange.
addpath(genpath('<your-path-to>/plot_topography'))