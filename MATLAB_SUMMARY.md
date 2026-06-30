# MATLAB Script Summary

This document summarizes the legacy MATLAB workflow in
`matlab_scripts/Doric_circadian_v9_hattarlab.m` and related helper scripts. It is
intended as a design reference for deciding what belongs in the Python package
and how the package should be organized.

## Main Mental Model

`Doric_circadian_v9_hattarlab.m` is a cell-by-cell MATLAB analysis notebook, not
a clean single-entry pipeline. Users are expected to run selected cells depending
on the data source and analysis goal.

The core workflow is:

```text
Doric files
-> sessionized 405/465 traces
-> initial dF/F
-> dynamic correction
-> events, summary metrics, and frequency map
-> plots and exports
```

The central MATLAB data object is `data_org`, shaped as:

```text
samples x columns x intervals
```

The columns are organized as:

```text
1 = time/sample index
2 = channel 1 405 nm
3 = channel 1 465 nm
4 = channel 2 405 nm
5 = channel 2 465 nm
...
```

## 1. Global Parameters

The first section defines experiment and analysis settings.

- `fs`: expected sample frequency of the already-imported data. The script does
  not downsample to this value during import.
- `t_int`: interval length in hours, usually `1`.
- `t_session`: recording session length in seconds, currently `610`.
- `t_start`: wall-clock start time used in ClockLab export metadata.
- `date_start`: start date used in ClockLab export metadata.
- `zt_start`: starting zeitgeber time label used in ClockLab export metadata.
- `n_Chn`: number of recorded fiber channels.
- `fitting_cutoff`: percentile exclusion used during initial 405-to-465 fitting.
- `weight_fit`: legacy fit behavior that zeroes part of the pooled fit arrays.
- `sortnot`: whether imported recording sessions should be sorted by time.
- `droplast`: whether to drop the final imported session.

The script also builds:

```matlab
x = 1:(t_session-10)*fs;
```

This is the expected sample index after dropping the first and last 5 seconds
from each session.

`lowpass_not`, `cutoff_frequency`, and `lp_filter` are defined near the top of
the script, but they are not used by the main processing path.

## 2. Data Import Cells

The import section contains several alternative loading paths. These are meant
to be run selectively, not all at once.

The supported import patterns in the script are:

- Standard Doric reader using `readdoric_QT_v4`.
- Additional standard Doric files appended with `cat(3, data1, data2)`.
- Special reader using `readdoric_QT_v4_special` for files missing Analog-out
  channels.
- Digital-IO reader using `readdoric_QT_v4_digital`.
- A corrupt-file rescue path that reads 405 and 465 files separately, recombines
  channels, and appends them to existing `data_org`.

For the Python package, this should be a separate I/O or sessionization layer,
not part of the core signal-analysis API. The current Python package already
keeps Doric-specific I/O out of the main analysis functions, which is a useful
boundary.

## 3. Initial 405-to-465 Fit

The main analysis starts by fitting the 405 nm isosbestic signal to the 465 nm
calcium signal.

For each channel, the MATLAB script:

1. Flattens all intervals into one long 405 vector and one long 465 vector.
2. Builds a percentile mask from the 465 signal using `fitting_cutoff`.
3. Optionally applies the legacy `weight_fit` behavior.
4. Fits one global linear model:

   ```text
   465 ~= slope * 405 + intercept
   ```

5. Applies that same fit back to each interval.
6. Smooths the fitted 405 curve with a 5-second moving mean.
7. Computes initial fractional dF/F:

   ```matlab
   data_dff = (y_465 - fit_405_curve) ./ median(fit_405_curve);
   ```

This belongs in core Python as a standalone fitting function, such as
`fit_405_to_465`.

## 4. Dynamic Artifact Correction

Each interval's initial `data_dff` is passed to `FP_IRLS_regularized_v9`.

That helper:

1. Converts traces to column vectors.
2. Mean-shifts both traces around 1.
3. Low-pass filters both traces at 0.8 Hz.
4. Splits the trace into 2.5-minute chunks.
5. Uses robust IRLS fitting between the isosbestic trace and the dF/F trace.
6. Regularizes each chunk's fit toward the previous chunk.
7. Returns the residual as `data_dff_dynamiccorrected`.

The dynamic correction is core signal processing and should remain part of the
main Python analysis workflow.

## 5. Adjusted And Positive dF/F Traces

After dynamic correction, the script creates additional derived traces:

- `data_dff_dynamiccorrected`: residual from IRLS correction.
- `data_dff_2padj`: dynamic-corrected dF/F shifted by subtracting its 10th
  percentile.
- `data_dff_cut0`: `data_dff_2padj` with negative values clamped to zero.

These traces should remain separate in Python result objects because they have
different meanings:

- initial dF/F
- dynamic-corrected dF/F
- percentile-adjusted dF/F
- positive-only phasic dF/F

## 6. Frequency / Power Map

The script calls `count_frequence` once per channel and interval.

That helper:

1. Fits 405 nm to 465 nm again inside the helper.
2. Computes percent dF/F.
3. Runs a continuous wavelet transform.
4. Returns median absolute wavelet coefficients by frequency.

Important design point: the frequency map is currently parallel to the main
dF/F pipeline. It is not derived from `data_dff_dynamiccorrected`. In Python,
this should probably be an optional `frequency.py` feature where the caller can
choose which trace is used for the power map.

## 7. Event Counting

The active event-counting call uses:

```matlab
count_event_2pc_v9(data_dff_dynamiccorrected, fs)
```

It does not use the positive-clamped `data_dff_cut0` trace.

The event helper:

1. Sets baseline to the 2nd percentile of the trace.
2. Computes a threshold:

   ```text
   baseline + max(2 * std(signal), 0.03)
   ```

3. Uses `findpeaks`.
4. Requires a peak width of at least 1 second.
5. Returns the number of detected peaks.

This should stay as a clean standalone event-detection function, such as
`count_events`, with optional return details for threshold, peak indices, peak
widths, and event bounds.

## 8. Interval-Level Metrics

The script computes one summary value per channel per interval.

The main metrics are:

- `level_phasic`: sum of positive adjusted dF/F.
- `level_cal_raw`: median raw 465 nm signal.
- `level_405_raw`: median raw 405 nm signal.
- `level_tonic`: 10th percentile of initial `data_dff`.
- `level_average`: median initial `data_dff`.
- `level_tonic_dtr`: tonic minus 24-hour moving mean.
- `level_tonic_z`: detrended tonic divided by 24-hour moving standard
  deviation.
- `level_average_dtr`: average minus 24-hour moving mean.

These should be exposed as structured result fields. They could live in a
`metrics.py` module or remain part of a high-level `analyze_sessions` result.

## 9. Main Summary Plot

The main plotting section creates an 8-row figure per channel:

1. Raw 405 nm median by interval.
2. Raw 465 nm median by interval.
3. Tonic baseline.
4. Detrended tonic baseline.
5. Z-scored tonic baseline.
6. Phasic transient level.
7. Event count.
8. Frequency power map.

This should not be part of the core analysis API. It is better as optional
plotting code that consumes result objects, for example in `plotting.py`.

## 10. Save And ClockLab Export

The script saves a MATLAB `.mat` file named `data_all_proc_v9` containing most
intermediate and summary arrays.

It also writes ClockLab-style `.awd` text files for:

- phasic level
- tonic level
- detrended tonic level
- event counts
- average dF/F
- detrended average dF/F

Each `.awd` file is a one-column file with a 7-row metadata header followed by
one value per interval.

The header includes:

- signal name
- start date
- start time
- interval metadata encoded as `t_int * 60 * 4`
- ZT label
- free-text placeholder
- sex or animal metadata placeholder

For Python, this should be a dedicated `exports.py` module with explicit
metadata fields instead of hard-coded placeholder strings.

## 11. Raw Trace And Correction Inspection

The later QC plotting cells:

- plot whole-recording raw 405 and 465 traces
- stack dark-phase and light-phase adjusted dF/F traces
- overlay filtered calcium and fitted dynamic-correction traces

These are useful inspection views, but they are not part of the reproducible
analysis pipeline. They should be optional plotting or quality-control helpers.

## 12. Light-Pulse Response Analysis

The light-pulse section handles CT6, CT14, and CT22 response windows.

The script:

1. Selects sessions relative to `int_beforeCT6LP`.
2. Downsamples dF/F to 1 Hz using `downsample(..., fs)`.
3. Extracts three windows per CT condition.
4. Baseline-shifts each window using the first 15 seconds.
5. Z-scores each window using the first 15 seconds.
6. Computes AUC over these windows:

   ```text
   0-15 seconds
   15-30 seconds
   30-90 seconds
   ```

This is a distinct analysis feature. It should be separate from the core
circadian baseline workflow, for example in `light_pulse.py`.

## Recommended Python Organization

A clean Python package organization would be:

```text
io.py / streams.py        Doric/session ingestion, cropping, sorting, concatenation
analysis.py               high-level analyze_sessions workflow
fitting.py                405-to-465 fit and dF/F
correction.py             IRLS dynamic correction
events.py                 transient/event detection
metrics.py                tonic/phasic/average/detrending summaries
frequency.py              wavelet power map / count_frequence equivalent
light_pulse.py            CT6/CT14/CT22 window extraction and AUC
exports.py                ClockLab .awd, CSV/Parquet/NPZ exports
plotting.py               summary and QC plots
results.py                typed result containers
```

Current Python package coverage already includes:

- initial 405-to-465 fit
- IRLS dynamic correction
- event counting
- high-level session analysis
- stream sessionization
- light-pulse window extraction

The main MATLAB-equivalent pieces still worth considering are:

- frequency or wavelet power maps
- ClockLab `.awd` export
- Doric-specific import, if desired
- summary and QC plotting helpers

## Known MATLAB Issues To Keep In Mind

These issues should not necessarily be reproduced in Python unless legacy
compatibility requires it.

- Some helper function declarations do not match their filenames.
- Some import helper functions referenced by the main script are not present in
  this repository.
- Several ClockLab export blocks write the ZT metadata into `table_AUC` while
  exporting `table_cal` or `table_mean`.
- `weight_fit` is a legacy behavior and may not represent conventional weighted
  least squares.
- `count_frequence` recomputes dF/F independently instead of using the main
  pipeline's corrected trace.
- Later exploratory cells include hard-coded session numbers and one likely typo
  where `rate_downsample(...)` appears to mean `downsample(...)`.
