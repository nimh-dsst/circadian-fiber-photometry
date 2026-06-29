function [df,beta_prev,new_uv,calcium_filt] = FP_IRLS_regularized_v9(calcium, isosbestic, fs, chunk, lambda,IRLS_constant,beta_prev)
% IRLS ΔF/F with temporal regularization of chunk-wise fits
% Inputs:
%   calcium, isosbestic – signals
%   fs – sampling rate
%   chunk – number of samples per chunk
%   lambda – regularization strength (0–1)

if nargin < 4
    chunk=2.5*fs*60; 
end
if nargin < 5
    lambda1 = 0.6; %high regularization on intercpt
end
if nargin < 6
    IRLS_constant = 4.685;  
end

%% Ensure column vectors
calcium = calcium(:);
isosbestic = isosbestic(:);
n = min(length(calcium), length(isosbestic));
calcium = calcium(1:n);
isosbestic = isosbestic(1:n);

% shift to remove effect of incept
ca_base = mean(calcium);
iso_base = mean(isosbestic);
calcium_corr = calcium - ca_base + 1;
isosbestic_corr = isosbestic - iso_base + 1;
% calcium_corr=calcium;
% isosbestic_corr=isosbestic;

% Low-pass filter
[b_filt, a_filt] = butter(1, 0.8 / (fs/2), 'low');
calcium_filt = filtfilt(b_filt, a_filt, calcium_corr);
isosbestic_filt = filtfilt(b_filt, a_filt, isosbestic_corr);
% calcium_filt=calcium_corr;
% isosbestic_filt=isosbestic_corr;



% Initialize
new_uv = zeros(size(calcium));
df = zeros(size(calcium));
[beta_prev, ~] = robustfit(isosbestic_filt, calcium_filt,'bisquare', IRLS_constant, 'on');
% beta_prev = [];

% Chunk-wise robust fit + regularization
for j = 1:chunk:n
    idx_end = min(j+chunk-1, n);
    idx = j:idx_end;

    x = isosbestic_filt(idx);
    y = calcium_filt(idx);

    if std(x) == 0
        new_uv(idx) = mean(y);
        continue
    end

    % Robust fit
    [beta, ~] = robustfit(x, y,'bisquare', IRLS_constant, 'on');  % returns [b; a]

    % Regularize with previous beta
    if ~isempty(beta_prev)
        beta = (1 - lambda) .* beta + lambda .* beta_prev;
    end
    beta_prev = beta;

    % Fitted baseline
    fit_uv = beta(1) + beta(2) * x;

    % ΔF/F using raw UV
    % raw_iso = isosbestic(idx);
    %dff_chunk = y - fit_uv;
    % dff_chunk = max(dff_chunk, 0);  % clamp negatives
    % dff_chunk = filtfilt(b_filt, a_filt, dff_chunk);  % smooth

    % Store results
    new_uv(idx) = fit_uv;
    %df(idx) = dff_chunk;
end

df=calcium_filt-new_uv;
df=filtfilt(b_filt, a_filt, df);

% Output
% results.dff = dff;
% results.new_uv = new_uv;
% results.auc = trapz(dff) / fs;
% results.std = std(dff);
% results.signal_filtered = calcium_filt;
% results.uv_filtered = isosbestic_filt;
% results.signal_raw = calcium;
% results.uv_raw = isosbestic;
% results.fs = fs;
% results.lambda = lambda;
end

% %%
% 
% fs = 60;
% figure
% subplot(4,1,1);
% plot(calcium, 'b'); hold on;
% plot(isosbestic, 'm');
% title('Raw Signal and UV'); ylabel('Fluorescence');
% % legend('Calcium', 'UV'); grid on;
% 
% subplot(4,1,2);
% plot(calcium_filt, 'b'); hold on;
% plot(isosbestic_filt, 'm');
% title('Lowpass Filtered Signal and UV'); ylabel('Filtered F');
% % legend('Filtered Calcium', 'Filtered UV'); grid on;
% 
% subplot(4,1,3);
% calcium_filt = filtfilt(b_filt, a_filt, calcium_corr);
% plot(calcium_filt, 'b'); hold on;
% plot(new_uv, 'r');
% title('Dynamic Fit (new\_uv) vs Filtered Signal'); ylabel('Fluorescence');
% % legend('Filtered Calcium', 'new\_uv'); grid on;
% 
% subplot(4,1,4);
% plot(df-prctile(df,15), 'g');
% title('\DeltaF/F'); xlabel('Time (min)'); ylabel('\DeltaF/F');
% ylim([0, 0.02]); grid on;
% 
% sgtitle('Fiber Photometry Analysis Pipeline (Full Trace)', 'FontWeight', 'bold');
% %%
%  calcium=data_dff(:,1,18);
%  isosbestic=data_org(:,2,18);
