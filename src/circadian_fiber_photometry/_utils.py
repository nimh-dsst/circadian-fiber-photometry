"""Internal array and MATLAB-compatibility helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

ArrayKind = Literal["trace", "single_channel_sessions", "channels_sessions"]


def as_float_array(values: np.ndarray, name: str) -> np.ndarray:
    """Convert input to a finite-capable floating NumPy array."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def normalize_session_array(
    values: np.ndarray,
    name: str,
    *,
    allow_trace: bool = False,
) -> tuple[np.ndarray, ArrayKind]:
    """Normalize trace/session arrays to ``(samples, channels, sessions)``."""

    array = as_float_array(values, name)
    if array.ndim == 1 and allow_trace:
        return array[:, np.newaxis, np.newaxis], "trace"
    if array.ndim == 2:
        return array[:, np.newaxis, :], "single_channel_sessions"
    if array.ndim == 3:
        return array, "channels_sessions"
    expected = "1D, 2D, or 3D" if allow_trace else "2D or 3D"
    raise ValueError(f"{name} must be {expected}; got shape {array.shape}")


def require_matching_shapes(left: np.ndarray, right: np.ndarray) -> None:
    """Raise when normalized session arrays differ in shape."""

    if left.shape != right.shape:
        raise ValueError(
            "isosbestic_405 and calcium_465 must have matching shapes; "
            f"got {left.shape} and {right.shape}"
        )


def matlab_window_length(window: float | int) -> int:
    """Convert a MATLAB-style scalar moving window length to an integer."""

    length = int(round(float(window)))
    if length < 1:
        raise ValueError("window length must be at least 1")
    return length


def window_bounds(size: int, window: float | int) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive-exclusive centered moving-window bounds.

    MATLAB centers even-length windows over the current and previous samples.
    For a scalar window length ``k``, that corresponds to ``k / 2`` samples
    behind the current element and ``k / 2 - 1`` samples ahead.
    """

    length = matlab_window_length(window)
    before = length // 2
    after = length - before - 1
    indices = np.arange(size)
    starts = np.maximum(0, indices - before)
    stops = np.minimum(size, indices + after + 1)
    return starts, stops


def moving_mean(values: np.ndarray, window: float | int, axis: int = 0) -> np.ndarray:
    """MATLAB-like ``movmean`` with centered windows and shrinking endpoints."""

    array = np.asarray(values, dtype=float)
    moved = np.moveaxis(array, axis, 0)
    original_shape = moved.shape
    flat = moved.reshape(original_shape[0], -1)
    starts, stops = window_bounds(original_shape[0], window)
    cumulative = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    sums = cumulative[stops] - cumulative[starts]
    counts = (stops - starts)[:, np.newaxis]
    result = sums / counts
    result = result.reshape(original_shape)
    return np.moveaxis(result, 0, axis)


def moving_std(
    values: np.ndarray,
    window: float | int,
    axis: int = 0,
    *,
    ddof: int = 1,
) -> np.ndarray:
    """MATLAB-like ``movstd`` with shrinking endpoints."""

    array = np.asarray(values, dtype=float)
    moved = np.moveaxis(array, axis, 0)
    original_shape = moved.shape
    flat = moved.reshape(original_shape[0], -1)
    starts, stops = window_bounds(original_shape[0], window)
    cumulative = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    cumulative_sq = np.vstack(
        [np.zeros((1, flat.shape[1])), np.cumsum(flat * flat, axis=0)]
    )
    sums = cumulative[stops] - cumulative[starts]
    sums_sq = cumulative_sq[stops] - cumulative_sq[starts]
    counts = (stops - starts)[:, np.newaxis]
    means = sums / counts
    numerators = sums_sq - counts * means * means
    denominators = counts - ddof
    variances = np.zeros_like(numerators)
    np.divide(
        numerators,
        denominators,
        out=variances,
        where=denominators > 0,
    )
    variances = np.maximum(variances, 0)
    result = np.sqrt(variances).reshape(original_shape)
    return np.moveaxis(result, 0, axis)


def matlab_std(values: np.ndarray) -> float:
    """Sample standard deviation matching MATLAB's default ``std``."""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1))


def baseline_shift_and_zscore(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract and z-score by the first 15 samples of each window column."""

    baseline = windows[:15, :]
    baseline_mean = np.mean(baseline, axis=0)
    shifted = windows - baseline_mean
    baseline_std = np.std(baseline, axis=0, ddof=1)
    zscored = np.full_like(shifted, np.nan)
    np.divide(shifted, baseline_std, out=zscored, where=baseline_std != 0)
    return shifted, zscored
