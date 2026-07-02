"""Tonic-scale summary metrics for sessionized photometry traces."""

from __future__ import annotations

import numpy as np

from .._utils import moving_mean, moving_std, normalize_session_array


def compute_tonic_level(
    delta_f_over_f: np.ndarray,
    percentile: float = 10,
) -> np.ndarray:
    """Return per-channel, per-session tonic dF/F percentile levels."""

    _validate_percentile(percentile)
    data, _ = normalize_session_array(
        delta_f_over_f,
        "delta_f_over_f",
        allow_trace=True,
    )
    return np.percentile(data, percentile, axis=0)


def compute_average_level(delta_f_over_f: np.ndarray) -> np.ndarray:
    """Return per-channel, per-session median dF/F levels."""

    data, _ = normalize_session_array(
        delta_f_over_f,
        "delta_f_over_f",
        allow_trace=True,
    )
    return np.median(data, axis=0)


def compute_raw_median_level(signal: np.ndarray) -> np.ndarray:
    """Return per-channel, per-session raw median signal levels."""

    data, _ = normalize_session_array(signal, "signal", allow_trace=True)
    return np.median(data, axis=0)


def detrend_levels_by_moving_window(
    levels: np.ndarray,
    interval_hours: float,
    window_hours: float = 24,
) -> np.ndarray:
    """Subtract a centered moving-window mean from session-level metrics."""

    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    values = _level_array(levels)
    window = window_hours / interval_hours
    return values - moving_mean(values, window, axis=-1)


def zscore_levels_by_moving_window(
    levels: np.ndarray,
    interval_hours: float,
    window_hours: float = 24,
    *,
    ddof: int = 1,
) -> np.ndarray:
    """Z-score levels using moving-window mean subtraction and std scaling."""

    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")

    values = _level_array(levels)
    window = window_hours / interval_hours
    detrended = values - moving_mean(values, window, axis=-1)
    moving_scale = moving_std(values, window, axis=-1, ddof=ddof)
    zscored = np.full_like(detrended, np.nan)
    np.divide(detrended, moving_scale, out=zscored, where=moving_scale != 0)
    return zscored


def _level_array(levels: np.ndarray) -> np.ndarray:
    """Return finite-capable floating levels shaped by trailing session axis."""

    values = np.asarray(levels, dtype=float)
    if values.size == 0:
        raise ValueError("levels must not be empty")
    return values


def _validate_percentile(percentile: float) -> None:
    """Validate a percentile in NumPy/MATLAB range."""

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
