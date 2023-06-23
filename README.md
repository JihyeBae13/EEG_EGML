# Metric Learning in EEG based pre-movement intention

This repository contains the code to replicate the results in the paper:

Plucknett W, Sanchez Giraldo LG and Bae J (2022) Metric Learning in Freewill EEG
Pre-Movement and Movement Intention Classification for Brain Machine Interfaces.
Frontiers in Human Neuroscience 16:902183. doi: 10.3389/fnhum.2022.902183

## Data
You must have Downloaded the dataset from\
   Kaya, M., Binli, M., Ozbay, E. et al. A large electroencephalographic motor 
   imagery dataset for electroencephalographic brain computer interfaces. 
   Sci Data 5, 180211 (2018). https://doi.org/10.1038/sdata.2018.211

To access the data https://doi.org/10.6084/m9.figshare.c.3917698.v1 \
 For experiments, you must download the following 3 files:
   - FREEFORMSubjectB1511112StLRHand.mat
   - FREEFORMSubjectC1512082StLRHand.mat
   - REEFORMSubjectC1512102StLRHand.mat

## Additional libraries required
- You must have LIBSVM package installed\
 https://www.csie.ntu.edu.tw/~cjlin/libsvm/

- To visualize the topograpic maps\
 Víctor Martínez-Cagigal (2023). Topographic EEG/MEG plot 
 (https://www.mathworks.com/matlabcentral/fileexchange/72729-topographic-eeg-meg-plot), 
 MATLAB Central File Exchange. 

## Running the code
First, you should be familiar with the directory structure of the code
### Directory structure 
repository   
|____src\
|____experiment-scripts\
|____analyse-results\
|____data (this can be placed elsewhere)\
|____results (this can be placed elsewhere)\

"repository" is the name of the repo which by default is "metric-learning-premovement"\
"src" contains the models and functions used in the experiments and analysis\
"experiments-scripts" contains the code to run the experiments reported in the paper\
"analyse-results" contains the code to get figures and tables reported in the paper

## Install the code
- In MATLAB set as your working directory the location of this repository to run 
"install.m" which adds the location of the repository permanently.

- Alternatively, you can run:\
`addpath('\<path-to-the-repo\>');`\
In this second case, you must add the path everytime you start a new MATLAB session. 

## Setting paths
Once you have downloaded the data and additional libraries, you must set the paths
where these are located. Do this by modifying the matlab file "set_paths.m" located 
in the src directory. 
