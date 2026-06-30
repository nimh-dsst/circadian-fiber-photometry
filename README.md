# circadian-fiber-photometry
A Python library of analysis code for circadian fiber photometry experiments

This package converts the analysis portions of the legacy MATLAB scripts in
`matlab_scripts/` to Python. It intentionally does not read `.doric` files; pass
NumPy arrays from your existing reader into the public API. It can also generate
synthetic Doric HDF5 files with plausible photometry traces for tests and
simulations.

## Attribution

The original MATLAB scripts were authored by Qijun Tang. Codex, using HHS
ChatGPT 5.5 Thinking Extra High at 1.5 speed, was used to convert the MATLAB
analysis scripts into a reusable Python package. The conversion was authored by
Josh Lawrimore, PhD.

## Install from GitHub with uv

Pin a tag or commit from your application:

```bash
uv add "circadian-fiber-photometry @ git+https://github.com/nimh-dsst/circadian-fiber-photometry.git@<tag-or-commit>"
```

## Usage

```python
from circadian_fiber_photometry import analyze_sessions

result = analyze_sessions(
    isosbestic_405,
    calcium_465,
    fs=60,
    interval_hours=1,
)

print(result.level_tonic)
print(result.event_counts)
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

- `fit_405_to_465`
- `irls_dynamic_correction`
- `count_events`
- `analyze_sessions`
- `extract_light_pulse_windows`
- `sessionize_stream_pair`
- `analyze_stream_pair`
- `generate_synthetic_doric`

## Synthetic Doric files

Use `generate_synthetic_doric` to create deterministic `.doric` HDF5 files that
mirror the Doric FPConsole hierarchy used by the legacy MATLAB readers:

```python
from circadian_fiber_photometry import (
    SyntheticDoricConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    generate_synthetic_doric,
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
print(summary.ttl_pulse_sample_indices)
print(summary.ttl_behavior_events)
```

The generator writes Doric-like configuration metadata and per-series lock-in,
analog input, analog output, and digital IO groups. It does not attempt to
guarantee Doric Neuroscience Studio GUI import compatibility. Digital IO TTL
pulse timing is relative to the start of each series. Behavior codes are encoded
by pulse count; by default each pulse is 50 ms high with a 50 ms low gap, so
within-sequence rising edges are 100 ms apart.

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

The legacy MATLAB power-map output is not implemented because the referenced
`count_frequence` helper is not present in this repository.
