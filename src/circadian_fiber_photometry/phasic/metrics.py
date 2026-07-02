"""Phasic signal isolation and integrated-fluorescence metrics."""

from __future__ import annotations

import numpy as np

from .._utils import (
    as_float_array,
    normalize_session_array,
    restore_session_array,
)


def percentile_adjust(
    delta_f_over_f: np.ndarray,
    percentile: float = 10,
) -> np.ndarray:
    """Subtract each trace's sample-axis percentile from dF/F."""

    _validate_percentile(percentile)
    data, kind = normalize_session_array(
        delta_f_over_f,
        "delta_f_over_f",
        allow_trace=True,
    )
    adjusted = data - np.percentile(data, percentile, axis=0, keepdims=True)
    return restore_session_array(adjusted, kind)


def positive_only(delta_f_over_f: np.ndarray) -> np.ndarray:
    """Clamp negative values to zero."""

    values = as_float_array(delta_f_over_f, "delta_f_over_f").copy()
    values[values < 0] = 0
    return values


def compute_phasic_trace(
    delta_f_over_f: np.ndarray,
    percentile: float = 10,
) -> np.ndarray:
    """Return percentile-adjusted, positive-only phasic dF/F."""

    return positive_only(percentile_adjust(delta_f_over_f, percentile=percentile))


def compute_phasic_level(
    delta_f_over_f: np.ndarray,
    percentile: float = 10,
) -> np.ndarray:
    """Sum positive percentile-adjusted dF/F per channel and session."""

    phasic = compute_phasic_trace(delta_f_over_f, percentile=percentile)
    data, _ = normalize_session_array(phasic, "delta_f_over_f", allow_trace=True)
    return np.sum(data, axis=0)


def integrated_fluorescence(
    delta_f_over_f: np.ndarray,
    *,
    fs: float | None = None,
    axis: int = 0,
) -> np.ndarray:
    """Integrate fluorescence over an axis using samples or seconds.

    When ``fs`` is omitted, this returns a MATLAB-style sample sum. When ``fs``
    is provided, integration uses seconds as the x-axis spacing.
    """

    values = as_float_array(delta_f_over_f, "delta_f_over_f")
    if fs is None:
        return np.sum(values, axis=axis)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, dx=1 / fs, axis=axis)
    return np.trapz(values, dx=1 / fs, axis=axis)


def _validate_percentile(percentile: float) -> None:
    """Validate a percentile in NumPy/MATLAB range."""

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
