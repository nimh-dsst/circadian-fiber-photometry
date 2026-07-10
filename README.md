# circadian-fiber-photometry

A Python library of analysis code for circadian fiber photometry experiments

This package converts the analysis portions of the legacy MATLAB scripts in
`matlab_scripts/` to Python. It can load Doric-style HDF5 files, generate
synthetic Doric HDF5 files with plausible photometry traces for tests and
simulations, and run tonic or phasic analyses through plain Python APIs.

Development status and planned work are tracked in the
[feature checklist](FEATURE_CHECKLIST.md).

## Attribution

The original MATLAB scripts were authored by Qijun Tang. The analysis code in
this repository is intended to support long-term circadian fiber photometry
workflows, with methodological background and best-practice considerations
discussed in [Tang et al., "Long-term optical monitoring of genetically encoded
fluorescent indicators," PNAS Nexus,
2025](https://doi.org/10.1093/pnasnexus/pgaf372). Codex, using HHS ChatGPT 5.5
Thinking Extra High at 1.5 speed, was used to convert the MATLAB analysis
scripts into a reusable Python package. The conversion was authored by Josh
Lawrimore, PhD.

## Install from GitHub with uv

Pin a tag or commit from your application:

```bash
uv add "circadian-fiber-photometry @ git+https://github.com/nimh-dsst/circadian-fiber-photometry.git@<tag-or-commit>"
```

## Package layout

The package uses a `src/` layout and separates I/O, simulation, and analysis
logic:

- `circadian_fiber_photometry.io`: Doric HDF5 loading.
- `circadian_fiber_photometry.simulation`: synthetic Doric HDF5 generation.
- `circadian_fiber_photometry.analyses`: discoverable tonic and phasic analysis
  registry.
- `circadian_fiber_photometry.tonic`: global 405-to-465 fitting, tonic
  percentile levels, raw median levels, and 24-hour moving-window detrending.
- `circadian_fiber_photometry.phasic`: IRLS dynamic correction, event counting,
  positive percentile-adjusted phasic traces, integrated fluorescence, and
  light-pulse windows.

The root package re-exports common functions for compatibility. New code should
prefer `io`, `simulation`, and `analyses` for package-level workflows.

## Usage

```python
from circadian_fiber_photometry import load_doric, run_analysis

dataset = load_doric("example.doric")

result = run_analysis(
    dataset,
    analysis="tonic",
    config={
        "interval_hours": 1,
        "weight_fit": False,
    },
)

print(result.arrays["level_tonic"])
```

Session-aware inputs can be either `(samples, sessions)` for one channel or
`(samples, channels, sessions)` for multi-channel recordings. Returned trace
arrays are normalized to `(samples, channels, sessions)`.

Pass `fit_weights` to use conventional weighted least squares when fitting the
405 nm signal to the 465 nm signal. Weights must be nonnegative and match or
broadcast to the input signal shape; zero-weight samples are ignored. The
`weight_fit` argument preserves the legacy MATLAB behavior by overwriting the
first session-length block of the pooled fitting arrays with zeros before
regression.

The package exposes:

- I/O: `load_doric`
- Registry: `list_analyses`, `run_analysis`
- Tonic: `fit_405_to_465`, `compute_tonic_level`,
  `detrend_levels_by_moving_window`, `zscore_levels_by_moving_window`
- Phasic: `irls_dynamic_correction`, `count_events`, `compute_phasic_trace`,
  `compute_phasic_level`, `integrated_fluorescence`,
  `extract_light_pulse_windows`
- Pipelines and adapters: `analyze_sessions`, `sessionize_stream_pair`,
  `analyze_stream_pair`
- Simulation: `generate_synthetic_doric`, `add_tonic_component`,
  `add_scheduled_calcium_events`, `add_random_calcium_events`,
  `add_gaussian_noise`, `configure_session_start_spike`,
  `add_scheduled_box_artifacts`, `add_random_box_artifacts`

Modular imports are available when you want to build custom pipelines:

```python
from circadian_fiber_photometry.io import load_doric
from circadian_fiber_photometry.analyses import list_analyses, run_analysis
from circadian_fiber_photometry.phasic import count_events, irls_dynamic_correction
from circadian_fiber_photometry.tonic import fit_405_to_465, compute_tonic_level
```

## Synthetic Doric files

Use `generate_synthetic_doric` to create deterministic `.doric` HDF5 files that
mirror the Doric FPConsole hierarchy used by the legacy MATLAB readers:

```python
from circadian_fiber_photometry.simulation import (
    SyntheticDoricConfig,
    SyntheticSignalConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    add_gaussian_noise,
    add_random_box_artifacts,
    add_random_calcium_events,
    add_scheduled_box_artifacts,
    add_scheduled_calcium_events,
    add_tonic_component,
    configure_session_start_spike,
    generate_synthetic_doric,
)

signal = SyntheticSignalConfig()
signal = add_tonic_component(
    signal,
    amplitude=0.012,
    frequency_hz=1 / 86400,
    name="daily calcium rhythm",
)
signal = add_scheduled_calcium_events(
    signal,
    [30.0, 120.0, 240.0],
    amplitude=0.04,
    name="known events",
)
signal = add_random_calcium_events(
    signal,
    rate_per_minute=2.0,
    start_window_seconds=(10.0, 590.0),
    name="background events",
)
signal = add_gaussian_noise(signal, calcium_std=0.001, isosbestic_std=0.0005)
signal = configure_session_start_spike(signal)
signal = add_scheduled_box_artifacts(
    signal,
    [45.0, 180.0],
    durations_seconds=[1.0, 2.0],
    magnitude_fraction=-0.10,
    name="known signal drops",
)
signal = add_random_box_artifacts(
    signal,
    count_per_series=2,
    start_window_seconds=(10.0, 590.0),
    duration_range_seconds=(0.5, 1.5),
    magnitude_fraction=0.10,
    name="random shared artifacts",
)

summary = generate_synthetic_doric(
    "synthetic.doric",
    SyntheticDoricConfig(
        series_count=8,
        session_duration_seconds=610,
        inter_series_gap_seconds=1190,
        fs=60,
        channel_count=2,
        seed=123,
        signal=signal,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("lick", channel=1, pulse_count=1),
            SyntheticTTLBehaviorCodeConfig("entry", channel=1, pulse_count=2),
            SyntheticTTLBehaviorCodeConfig("reward", channel=1, pulse_count=3),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig("lick", start_seconds=(5.0,)),
            SyntheticTTLBehaviorEventConfig("entry", start_seconds=(15.0,)),
            SyntheticTTLBehaviorEventConfig("reward", start_seconds=(25.0,)),
        ),
    ),
)

print(summary.event_sample_indices)
print(summary.artifact_occurrences)
print(summary.ttl_pulse_sample_indices)
print(summary.ttl_behavior_events)
```

The generator writes Doric-like configuration metadata and per-series lock-in,
analog input, analog output, and digital IO groups. It does not attempt to
guarantee Doric Neuroscience Studio GUI import compatibility. Digital IO TTL
pulse timing is relative to the start of each series. Behavior codes are encoded
by pulse count; by default each pulse is 50 ms high with a 50 ms low gap, so
within-sequence rising edges are 100 ms apart.

Generated files can be loaded directly:

```python
from circadian_fiber_photometry import load_doric, run_analysis

dataset = load_doric("synthetic.doric")
result = run_analysis(dataset, analysis="phasic", config={"interval_hours": 0.5})
```

Tonic components are additive calcium-channel sinusoids with amplitude and
frequency controls. Scheduled calcium events use seconds relative to each
series start; random calcium events are seeded from `SyntheticDoricConfig.seed`.

### Signal artifacts

Artifact configuration lives under
`circadian_fiber_photometry.simulation.artifact` and is re-exported from the
`simulation` namespace. Artifacts are disabled unless added to a
`SyntheticSignalConfig`. Enabling them changes only the 405 and 465 lock-in
traces; simulated analog input remains unchanged.

For each target signal, an artifact's constant box offset is

```text
offset = magnitude_fraction * mean(pre-artifact signal)
```

The fraction is signed and unitless. A value of `0.10` adds 10% of that
signal's pre-artifact mean, while `-0.10` produces a 10% drop. Because each
signal uses its own mean, 405 and 465 offsets have the same sign but can have
different voltage magnitudes.

`configure_session_start_spike` adds a box at sample zero to every channel in
every series. Its defaults are 1 second and `magnitude_fraction=1.0`, which adds
100% of each signal's mean. Scheduled boxes use times relative to each series;
one duration can be broadcast to every start or a matching duration can be
provided per start. Channel and series selectors use Doric's one-based
numbering.

Random boxes require exactly one of `count_per_series` or `rate_per_minute`.
Rate mode draws a Poisson count for each selected series. Durations are sampled
uniformly from `duration_range_seconds`, and selected channels share the same
random occurrence times. Box artifacts cannot overlap other scheduled or
random boxes on the same series/channel, although they may overlap the
session-start spike. Random placement raises `ValueError` when the requested
non-overlapping schedule cannot fit.

`SyntheticDoricSummary.artifact_occurrences` reports the realized ground truth
for every affected series/channel: half-open sample bounds, within-series and
absolute seconds, requested and sample-realized duration, signed fraction, and
the exact 405/465 voltage offsets. Random scheduling uses a dedicated stream
derived from `SyntheticDoricConfig.seed`, so it is repeatable without changing
the simulator's other random traces.

Calcium events default to a 9 s^-1 rise rate and 1 s^-1 fall rate, matching the
jGCaMP7-style kinetics used by the simulator.

## Stream dictionaries

If your app already loads subject streams into dictionaries like:

```python
{
    "fs": 10.0,
    "data": signal_values,
    "ts": timestamps,
    "channel": [1],
}
```

use `sessionize_stream_pair` to convert concatenated iso/experimental streams
into the `(samples, sessions)` arrays expected by `analyze_sessions`:

```python
from circadian_fiber_photometry import analyze_sessions, sessionize_stream_pair

sessionized = sessionize_stream_pair(
    iso_stream,
    exp_stream,
    crop_seconds=5.0,  # optional: matches the old MATLAB first/last 5 s crop
)

result = analyze_sessions(
    sessionized.isosbestic_405,
    sessionized.calcium_465,
    fs=sessionized.fs,
    interval_hours=1,
)
```

By default, sessions are split at timestamp resets or gaps greater than one
second. If timestamps are continuous with no gaps, pass
`session_duration_seconds`.

The legacy MATLAB wavelet power-map output from `count_frequence.m` has not
been ported into the Python package yet.
