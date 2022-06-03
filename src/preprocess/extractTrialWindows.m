function [wd_data, labels] = extractTrialWindows(data_struct, params)
% only find onset for L and R events (markers 1 and 2)

% extract wd_data in the form WxCxN
% where W is the length of the window in samples,
% C the number of channels, and N is the number of trials

%calculate window size
wd_str = round(params.wd_str_t * data_struct.sampFreq);
wd_end = round(params.wd_end_t * data_struct.sampFreq)-1;
W = wd_end - wd_str + 1;
C = length(data_struct.chnames);
% get channels
wd_data = zeros(W,C,0);
labels  = [];
for class = 1:2
    onset_times = extractOnsetTimes(data_struct.marker, class);
    for i = 1:length(onset_times)
        trial_data = data_struct.data(onset_times(i)+wd_str:onset_times(i)+wd_end, :);
        wd_data = cat(3, wd_data, reshape(trial_data, [W,C,1]));
    end
    labels = cat(1, labels, class*ones(length(onset_times),1));
end 


end


function onset_times = extractOnsetTimes(markers, event_class)
event_wd = markers == event_class;
trigg = diff(event_wd);
event_str = find(trigg == 1) + 1;
event_end = find(trigg == -1); % not used but there
% select onset time as start of event window
onset_times = event_str;
end