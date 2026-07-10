# Feature Checklist

This is the single checklist for tracking repository features. Add new requests
to the **Feature request inbox**, then move them into the appropriate section
once their scope is clear.

## How to use this checklist

- `[x]` means the feature exists in the Python package and has tests where
  practical.
- `[ ]` means the feature is missing, partial, or not yet verified.
- Check an item only after its acceptance criteria are met.
- User-facing changes should include tests and documentation in the same pull
  request.
- Add an issue or pull-request link to an item when one exists.

Last reviewed: **2026-07-10**. At that review, `uv run pytest` passed all 58
tests and `uv run ruff check .` passed.

## Feature request inbox

Copy this line for each new request:

```markdown
- [ ] **Short feature name** — Intended outcome. Acceptance: observable behavior. Issue: #___
```

## Package foundation and public API

- [x] Use a `src/`-layout, installable Python package.
- [x] Keep core analysis code independent of Streamlit and other UI frameworks.
- [x] Keep core dependencies limited to NumPy, SciPy, and h5py.
- [x] Preserve the original MATLAB files separately under `matlab_scripts/`.
- [x] Credit Qijun Tang for the original MATLAB work in the README.
- [x] Expose a small root API for loading, simulation, analysis, and result
  objects.
- [x] Support modular imports from `io`, `simulation`, `analyses`, `tonic`, and
  `phasic` namespaces.
- [ ] Add a built-wheel installation and import smoke test.
- [ ] Define and document a release/versioning process.

## Doric HDF5 input and normalized data

- [x] Load Doric-style FPConsole HDF5 files through `load_doric`.
- [x] Discover series and matching 405/465 lock-in channels.
- [x] Normalize photometry arrays to `(samples, channels, series)`.
- [x] Load timestamps, sampling rate, analog inputs, analog outputs, digital IO,
  and file metadata.
- [x] Return a typed `DoricDataset` independent of HDF5 internals.
- [x] Raise clear errors for missing or malformed required groups.
- [x] Feed a loaded dataset directly into the analysis registry.
- [ ] Test the loader against representative files produced by real Doric
  hardware/software versions, not only simulator output.
- [ ] Support unequal-length series through an explicit trim, pad, or ragged-data
  policy.
- [ ] Add a general-purpose Doric writer if writing normalized, non-simulated
  datasets becomes a requirement.

## Stream adapters and session handling

- [x] Detect timestamp gaps and timestamp resets in paired streams.
- [x] Validate that isosbestic and experimental stream gaps agree.
- [x] Split concatenated stream dictionaries into session matrices.
- [x] Support fixed-duration session splitting and optional edge cropping.
- [x] Estimate the session interval in hours and report irregular timing.
- [x] Analyze paired stream dictionaries through a UI-independent adapter.
- [ ] Define a policy for sessions with unequal lengths or missing channel data.
- [ ] Support sorting and concatenating sessions from multiple input files when
  timestamps arrive out of order.

## Doric HDF5 simulator

- [x] Generate deterministic Doric-style HDF5 files with configurable sampling
  rate, duration, series count, gaps, channels, and random seed.
- [x] Write Doric-like configuration metadata and lock-in, analog, and digital
  groups.
- [x] Generate configurable tonic signal components.
- [x] Generate scheduled calcium events with known timing.
- [x] Generate seeded random calcium events.
- [x] Add independently configurable Gaussian calcium and isosbestic noise.
- [x] Generate fixed and random TTL behavior events, including pulse-count
  behavior codes.
- [x] Return ground-truth event and TTL metadata in `SyntheticDoricSummary`.
- [x] Reject invalid configurations and protect existing files unless overwrite
  is explicitly enabled.
- [ ] **Session-start spike simulation** — Add a configurable spike to both the
  calcium and isosbestic signals at the beginning of every session. Default the
  spike duration to 1 second, make its magnitude configurable, express timing in
  seconds, and include its known sample bounds and parameters in the simulator
  ground truth.
- [ ] **Scheduled phasic box artifacts** — Accept explicit start times and
  durations in seconds and add the same signed box-shaped change to the calcium
  and isosbestic signals. Default the magnitude to 10% of each signal's mean and
  support both positive spikes and negative drops.
- [ ] **Random phasic box artifacts** — Place seeded random box artifacts using
  configurable counts or rates, time windows, durations, channels, and series.
  Reuse the signed-amplitude behavior of scheduled box artifacts and return
  their ground-truth locations and parameters.
- [ ] **Single-exponential photobleaching** — Replace the current linear
  across-session and within-session bleaching factors with configurable
  single-exponential decay for both calcium and isosbestic signals. Preserve the
  selected amplitude and time constant in simulator metadata and test generated
  traces against the defining equation.
- [ ] **Double-exponential photobleaching** — Support configurable
  double-exponential decay for both calcium and isosbestic signals, with
  independently configurable component amplitudes and time constants. Preserve
  all parameters in simulator metadata and test generated traces against the
  defining equation.
- [ ] **Selectable photobleaching model** — Let callers explicitly select no
  photobleaching, single-exponential photobleaching, or double-exponential
  photobleaching without changing other signal-generation behavior. Document
  the model equations, units, defaults, and tradeoffs without presenting either
  exponential model as universally preferred.
- [ ] **Optional flat interval shape** — Add a non-default sample-and-hold mode
  modeled after
  [`generateFiberPhotometryTraces.m` lines 49-62](https://github.com/qjtang12/Long-term_optical_monitoring_of_genetically-encoded_fluorescent_indicators/blob/73c5ba40b10c13b722b201ecf43aec5f74cdd227/generateFiberPhotometryTraces.m#L49-L62).
  For each interval, evaluate the underlying rise/fall trajectory at the
  interval midpoint and hold that value across the entire interval, producing
  plateaus and step transitions. Keep the existing continuous within-interval
  rise/fall behavior as the default.
- [ ] Add additional noise/artifact models beyond the items above, such as
  non-Gaussian noise, if required.
- [ ] Add a public batch-generation API for producing multiple files/configuration
  sweeps.
- [ ] Document which Doric software versions/layouts the simulator targets and
  validate GUI compatibility if that becomes a goal.

Photobleaching reference note: the linked
[`generateFiberPhotometryTraces.m`](https://github.com/qjtang12/Long-term_optical_monitoring_of_genetically-encoded_fluorescent_indicators/blob/73c5ba40b10c13b722b201ecf43aec5f74cdd227/generateFiberPhotometryTraces.m)
provides the single-exponential reference `exp(-t / decayTauSamples)`. A
double-exponential model must use an explicitly documented equation and default
parameters from an identified reference; do not infer or claim MATLAB
equivalence for that model.

## Analysis discovery and result contracts

- [x] Register tonic and phasic analyses in one `ANALYSES` registry.
- [x] List available analyses through `list_analyses`.
- [x] Run a selected analysis through `run_analysis`.
- [x] Describe each analysis with a name, description, expected inputs,
  configuration fields, and outputs.
- [x] Return UI-independent `AnalysisResult` objects with tables, arrays, figures,
  metadata, parameters, warnings, and provenance containers.
- [x] Reject unknown analysis names with a useful error.
- [ ] Replace prose-only configuration metadata with a machine-readable schema
  containing types, defaults, constraints, units, and choices for downstream
  form generation.
- [ ] Reject unknown configuration keys and validate all registry parameters
  consistently.
- [ ] Include package and algorithm version information in result provenance.
- [ ] Define stable table/array names and a compatibility policy for result
  schema changes.

## Tonic analysis

- [x] Globally fit the 405 nm signal to the 465 nm signal across sessions.
- [x] Support percentile-based fitting exclusion.
- [x] Preserve the optional legacy MATLAB zero-block `weight_fit` behavior.
- [x] Support conventional nonnegative weighted least squares separately from
  the legacy behavior.
- [x] Smooth the fitted isosbestic signal and calculate dF/F.
- [x] Compute tonic percentile, average, and raw-channel median levels.
- [x] Compute moving-window tonic and average detrending.
- [x] Compute moving-window tonic z-scores.
- [x] Support single-channel and multi-channel session arrays.
- [ ] Add a sliding-window raw-signal regression method for correcting a drifting
  isosbestic/calcium relationship, distinct from level detrending.
- [ ] Decide whether the default tonic percentile should remain MATLAB-compatible
  at 10% or expose a publication-oriented 2% preset.

## Phasic analysis

- [x] Perform chunk-wise, regularized IRLS dynamic artifact correction.
- [x] Produce dynamic-corrected, percentile-adjusted, and positive-only phasic
  traces as separate outputs.
- [x] Detect transient events using the MATLAB threshold and width rules.
- [x] Return event count, threshold, baseline, peak locations, heights, widths,
  and event bounds from the standalone detector.
- [x] Allow duration, minimum height, and standard-deviation factor tuning in the
  standalone event detector.
- [x] Compute phasic level and integrated fluorescence.
- [x] Extract CT6, CT14, and CT22 light-pulse windows with shifted, z-scored, and
  AUC outputs.
- [ ] **Trim session-start artifacts** — Add a reusable function under the
  `phasic` package that removes the first configurable number of seconds from
  every interval (for example, 5 seconds) using the sampling rate. It must trim
  calcium and isosbestic signals consistently, support all documented
  session-array shapes, reject durations that consume an interval, and have
  sample-exact tests.
- [ ] Expose session-start trimming through the registered phasic analysis and
  high-level pipeline without making the package depend on a UI.
- [ ] Expose event-detection and IRLS tuning parameters through the registered
  phasic analysis configuration.
- [ ] Add the wavelet power-map analysis represented by MATLAB
  `count_frequence.m`.
- [ ] Add optional FFT/spectrogram analysis if required beyond MATLAB parity.
- [ ] Add an overlapping adaptive-window regression method if the publication's
  developing technique is adopted.
- [ ] Add Bayesian regression if it becomes a supported analysis method.
- [ ] Support normalization against an independent baseline reference channel if
  such recordings become available.

## Exports, figures, and downstream consumption

- [x] Keep analysis outputs as plain Python objects consumable by scripts,
  notebooks, tests, CLIs, or a separate Streamlit app.
- [x] Reserve structured result containers for tables and optional figures.
- [ ] Export result tables/arrays to stable general-purpose formats such as CSV,
  NPZ, JSON, or Parquet as appropriate.
- [ ] Export ClockLab-compatible `.awd` files with explicit metadata fields.
- [ ] Add reusable summary plotting helpers for raw, tonic, phasic, event, and
  spectral results.
- [ ] Add reusable quality-control plots for raw traces and dynamic correction.
- [ ] Decide whether plotting support belongs in an optional dependency extra.

## Marimo examples

- [ ] Add an executable marimo simulation notebook that demonstrates every
  simulator feature, including session-start spikes, scheduled and random signed
  box artifacts, single- and double-exponential photobleaching, photobleaching
  model selection, continuous rise/fall traces, and flat sample-and-hold
  intervals.
- [ ] Add an executable marimo analysis notebook that loads or generates data,
  trims session-start artifacts, and demonstrates every tonic and phasic analysis
  available through the public API.
- [ ] Demonstrate every new user-facing simulation or analysis feature in a
  marimo notebook as part of that feature's definition of done.
- [ ] Add a lightweight automated check that marimo example files import or run
  successfully without committing their generated datasets.

## Testing and numerical validation

- [x] Test tonic, phasic, stream, simulator, Doric loader, error, and public API
  behavior with pytest.
- [x] Use deterministic simulator-generated datasets in tests.
- [x] Test generated Doric files by loading and analyzing them end to end.
- [x] Test multi-channel and multi-session array shapes.
- [x] Test legacy fitting behavior and conventional fit weights separately.
- [x] Keep the current codebase passing Ruff checks.
- [ ] Store representative MATLAB outputs as stable test fixtures.
- [ ] Add MATLAB-to-Python numerical comparison tests with documented tolerances.
- [ ] Add regression tests for real Doric fixtures small enough to redistribute.
- [ ] Add performance tests for long circadian recordings.
- [ ] Add package build and installed-package public API tests.

## Documentation

- [x] Explain that the MATLAB sources are preserved and Python is a modular
  reimplementation.
- [x] Document installation and basic `load_doric`/`run_analysis` usage.
- [x] Document tonic and phasic package responsibilities and public imports.
- [x] Document simulator configuration and ground-truth outputs.
- [x] Document stream-dictionary sessionization.
- [x] State the known missing wavelet power-map feature.
- [ ] Add a concise API reference covering every stable public function and
  result type.
- [ ] Add a validation document that distinguishes tested MATLAB equivalence from
  conceptual translation.
- [ ] Add export, plotting, and spectral-analysis examples when those features
  exist.

## Automation and maintenance

- [ ] Add continuous integration for `uv run ruff check .`.
- [ ] Add continuous integration for `uv run pytest`.
- [ ] Add continuous integration for package build and installed-package import.
- [ ] Add release automation only after the versioning process is defined.
- [ ] Add a changelog for user-visible API and numerical behavior changes.

## Completion rule for new features

A feature can be checked off when all applicable items below are true:

- The behavior is available through a reusable, UI-independent Python API.
- Inputs, outputs, units, defaults, and failure modes are clear.
- Meaningful success and error paths have pytest coverage.
- Public behavior is documented.
- Every user-facing simulation or analysis feature is demonstrated in a marimo
  notebook.
- `uv run pytest` and `uv run ruff check .` pass.
- MATLAB equivalence is claimed only when a fixture-based comparison test proves
  it within a documented tolerance.
