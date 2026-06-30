# circadian-fiber-photometry
A Python library of analysis code for circadian fiber photometry experiments

This package converts the analysis portions of the legacy MATLAB scripts in
`matlab_scripts/` to Python. It intentionally does not read `.doric` files; pass
NumPy arrays from your existing reader into the public API.

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
