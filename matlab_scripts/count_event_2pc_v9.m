function [num_events]=count_event_2pc(delta_f_over_f,samplerate)

% Extract signal data (assuming 405 nm and 465 nm channels)
% time_405nm = data(3).Data(2).Data;
% signal_405nm = data(3).Data(1).Data;
% time_465nm = data(4).Data(2).Data;
% signal_465nm = data(4).Data(1).Data;

% Use the original time and data
% processed_time = time_405nm;
% processed_signal_405nm = signal_405nm;
% processed_signal_465nm = signal_465nm;
%% Fit Signals
% Fit isosbestic 405 nm signals to 465 nm signals using linear least squares fit
% fitted_405nm = polyfit(processed_signal_405nm, processed_signal_465nm, 1);
% fitted_signal_405nm = polyval(fitted_405nm, processed_signal_405nm);
%% Calculate %ΔF/F
% Calculate %ΔF/F using fitted signal 405nm
% delta_f_over_f = ((processed_signal_465nm - fitted_signal_405nm) ./ fitted_signal_405nm) * 100;
%% Adjust %ΔF/F so that 0 corresponds to a chosen percentile of the signal
% percentile_value = input(‘Enter the percentile for adjustment (e.g., 2, 5): ‘);
% chosen_percentile = prctile(delta_f_over_f, percentile_value);
% adjusted_delta_f_over_f = delta_f_over_f - chosen_percentile;
%% Convert dF/F values to events
% Define the threshold for event detection
% bottom = prctile(delta_f_over_f, 2);
% std_df_f = std(delta_f_over_f);
% threshold = max(bottom + 1.5 * std_df_f, bottom+0.03);
% % Detect events
% event_duration_threshold = 1; % in seconds
% event_samples_threshold = event_duration_threshold * samplerate;
% event_mask = delta_f_over_f > threshold;
% event_starts = strfind([0 event_mask'], [0 1]); % Find rising edges
% event_ends = strfind([event_mask' 0], [1 0]); % Find falling edges
% % Filter events by duration
% event_durations = event_ends - event_starts;
% valid_events = event_durations >= event_samples_threshold;
% event_starts = event_starts(valid_events);
% event_ends = event_ends(valid_events);
% % Count events
% num_events = length(event_starts);

%%
% Parameters
event_duration_threshold = 1; % seconds
event_samples_threshold = event_duration_threshold * samplerate;

% Use findpeaks to detect transients
[min_prominence, min_height] = deal(2 * std(delta_f_over_f), 0.03);
baseline = prctile(delta_f_over_f, 2);
threshold = max(baseline + min_prominence, baseline + min_height);

[peak_vals, peak_locs, peak_widths] = findpeaks(delta_f_over_f, ...
    'MinPeakHeight', threshold, ...
    'MinPeakWidth', event_samples_threshold);

% Get event start and end indices using peak widths
half_width_samples = round(peak_widths / 2);
event_starts = max(1, round(peak_locs - half_width_samples));
event_ends   = min(length(delta_f_over_f), round(peak_locs + half_width_samples));

% Count events
num_events = length(peak_locs);

%% Plot Raw Data and %ΔF/F
% figure;
% subplot(3,1,1);
% plot(processed_time, processed_signal_405nm, ‘b’, processed_time, processed_signal_465nm, ‘r’);
% xlabel(‘Time (s)’);
% ylabel(‘Raw Signal’);
% title(‘Raw Data’);
% legend(‘405 nm’, ‘465 nm’);
% grid on;
% subplot(3,1,2);
% plot(processed_time, baseline_corrected_signal, ‘b’);
% xlabel(‘Time (s)‘);
% ylabel(‘%ΔF/F’);
% title(‘%ΔF/F vs Time’);
% grid on;
% Plot detected events
% subplot(3,1,3);

%plot example figure
% processed_time=1:length(delta_f_over_f);
% plot(processed_time, delta_f_over_f, 'b');
% hold on;
% for i = 1:num_events
%     event_time_range = processed_time(event_starts(i):event_ends(i));
%     event_signal = max(delta_f_over_f(event_time_range));
%     event_time=event_time_range(1)+processed_time(delta_f_over_f(event_time_range)==event_signal);
% 
%     plot(event_time, event_signal, 'ro', 'MarkerSize', 5,'LineWidth', 2); 
% end
% yline(threshold, 'r--', 'Threshold');  % Draws a red dashed line at y = 5 with label 'Threshold'
% xlabel('Time (s)');
% ylabel('%ΔF/F');
% title('Detected Events');
end