"""Phasic event detection utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .._utils import as_float_array, matlab_std
from ..results import EventDetectionResult


def count_events(
    delta_f_over_f: np.ndarray,
    fs: float,
    duration_threshold_seconds: float = 1.0,
    min_height: float = 0.03,
    prominence_std_factor: float = 2.0,
) -> EventDetectionResult:
    """Count transient events using the MATLAB ``findpeaks`` rule."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    if duration_threshold_seconds <= 0:
        raise ValueError("duration_threshold_seconds must be positive")
    signal = as_float_array(delta_f_over_f, "delta_f_over_f").reshape(-1)

    event_samples_threshold = duration_threshold_seconds * fs
    baseline = float(np.percentile(signal, 2))
    min_prominence = prominence_std_factor * matlab_std(signal)
    threshold = baseline + max(min_prominence, min_height)

    peak_indices, properties = find_peaks(
        signal,
        height=threshold,
        width=event_samples_threshold,
    )
    peak_widths = properties.get("widths", np.array([], dtype=float))
    half_width_samples = np.round(peak_widths / 2).astype(int)
    event_starts = np.maximum(0, peak_indices - half_width_samples)
    event_ends = np.minimum(signal.size - 1, peak_indices + half_width_samples)

    return EventDetectionResult(
        count=int(peak_indices.size),
        threshold=float(threshold),
        baseline=baseline,
        peak_indices=peak_indices.astype(int),
        peak_heights=properties.get("peak_heights", np.array([], dtype=float)),
        peak_widths=peak_widths,
        event_starts=event_starts.astype(int),
        event_ends=event_ends.astype(int),
    )
