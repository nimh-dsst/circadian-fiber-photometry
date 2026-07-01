# MATLAB Scripts vs. Publication Analysis Comparison

## Source Context

This document compares the legacy MATLAB scripts in `matlab_scripts/` and the
repo summary in `MATLAB_SUMMARY.md` against analysis methods described in the
publication **"Long-term optical monitoring of genetically encoded fluorescent
indicators"**.

Publication link: <https://doi.org/10.1093/pnasnexus/pgaf372>

The table maps analyses described from the publication to implementations found
in the MATLAB scripts.

## Comparison Table

| Publication analysis | Found in MATLAB scripts? | Evidence / notes |
|---|---:|---|
| Use a globally fitted isosbestic trace to calculate delta F/F. Global fitting preserves the underlying circadian oscillation. | Yes | The main script flattens all sessions, fits one `polyfit` 405-to-465 model, applies it per interval, then computes `data_dff` in `Doric_circadian_v9_hattarlab.m`. |
| The relationship between the isosbestic and calcium signal may drift over time. A sliding average window, such as 24 hours, is recommended to correct this effect. | Partial | The script applies 24-hour `movmean`/`movstd` detrending to interval tonic and average metrics. It does not appear to use a 24-hour sliding fit to correct the raw isosbestic-calcium relationship. |
| Use the 2nd percentile of each threshold to disentangle phasic from tonic responses. | Partial / mismatch | Event detection uses the 2nd percentile as baseline in `count_event_2pc_v9.m`. The main tonic/phasic metrics use the 10th percentile, despite comments mentioning 2%. |
| Ordinary least squares fitting of the isosbestic signal is standard for short-term recordings. | Partial | Ordinary least-squares style `polyfit` is used in the main global fit and frequency helper, but there is no distinct short-recording-specific OLS workflow. |
| Iteratively Reweighted Least Squares reduces the influence of outliers and increases fitting stability. | Yes | `FP_IRLS_regularized_v9.m` uses `robustfit(..., 'bisquare', ...)` in chunk-wise fits for dynamic artifact correction. |
| Technique in development: use overlapping windows of about 1 minute and run separate linear regression on each window, then apply heuristics to adapt the fit. | No | The closest match is non-overlapping 2.5-minute chunk-wise IRLS with regularization. No overlapping 1-minute adaptive-window regression workflow was found. |
| Technique in development: Bayesian regression framework. | No | No Bayesian regression implementation or reference was found in the MATLAB scripts. |
| Recommend an independent baseline reference signal other than the isosbestic signal for baseline normalization. | No | The scripts use 405 nm isosbestic and 465 nm calcium channels. No separate independent baseline reference channel normalization was found. |
| Peak identification after artifact correction and baseline normalization, with user-tunable parameters because peaks vary by cell type and brain region. | Partial | Active event counting runs on `data_dff_dynamiccorrected` and uses `findpeaks` with hard-coded threshold and width rules. Parameters are not exposed except by editing code. |
| Integrated fluorescence is a useful metric in addition to peak identification. A per-session threshold can separate phasic signal from baseline and noise. | Partial | `level_phasic` is computed as `sum(data_dff_cut0)` per interval/session. The scripts do not implement the publication's flexible threshold options. A separate light-pulse AUC exists via `trapz`, but that is a different windowed light-response analysis. |
| Spectral analysis of the signal by converting it into a spectrogram using wavelet transformation or fast Fourier transforms. | Yes, wavelet only | `count_frequence.m` computes dF/F, runs `cwt`, and returns median absolute coefficients by frequency. No FFT implementation was found. |

## Bottom Line

The MATLAB scripts implement the main global-fit tonic workflow, IRLS dynamic
correction, event counting, summed phasic signal, and wavelet power map. They do
not contain the Bayesian regression method, independent baseline reference
normalization, or overlapping adaptive-window regression method described as
developing or recommended approaches in the publication.
