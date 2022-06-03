%FTA - pull the fourier transform amplitudes as features
% Usage: features = FTA(X)
%   X is the trial data, organized as trial x sample x channel
%   The output is organized by trial x feature
function [featureVectors] = FTA(trialData)
%Apply Fourier Transform to each channel
fftTrialData = fft(trialData, [], 2);
%f= (0:length(fftTrialData(:,1,1))-1)*200/length(fftTrialData(:,1,1));
%plot(f,abs(fftTrialData(:,1,1)));             

%just keep first fc data points
fc = 3;             % number of frequency components
numFeatures = 2*fc-1; % number of features

filteredTrialData = fftTrialData(:,1:fc,:);
filteredTrialData(:,2:end,:) = filteredTrialData(:,2:end,:) * 2; % for single-sided

%Separate real and imaginary parts
realAndImagData(:,1,:) = filteredTrialData(:,1,:);
for idx = 2:fc
realAndImagData(:,2*idx-2,:) = real(filteredTrialData(:,idx,:));
realAndImagData(:,2*idx-1,:) = imag(filteredTrialData(:,idx,:));
end

%Combine into feature vector
numTrials = size(trialData, 1);
numChannels = size(trialData, 3);
featureVectors = zeros(numTrials, numFeatures*numChannels);
for i = 1:numChannels
    featureVectors(:,1+(numFeatures*(i-1)):numFeatures*i) = realAndImagData(:,:,i);
end
end