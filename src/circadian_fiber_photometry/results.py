"""Typed result containers for fiber photometry analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DffFitResult:
    """Result from fitting 405 nm isosbestic signal onto 465 nm signal.

    Session-aware arrays are normalized to ``(samples, channels, sessions)``.
    Coefficients are ``(channels, 2)`` in ``[slope, intercept]`` order.
    """

    dff: np.ndarray
    fitted_405: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class IRLSResult:
    """Result from chunk-wise robust dynamic artifact correction."""

    corrected: np.ndarray
    beta: np.ndarray
    fitted_isosbestic: np.ndarray
    calcium_filtered: np.ndarray
    isosbestic_filtered: np.ndarray


@dataclass(frozen=True)
class EventDetectionResult:
    """Detected transient events in a dF/F trace.

    Indices are zero-based Python indices.
    """

    count: int
    threshold: float
    baseline: float
    peak_indices: np.ndarray
    peak_heights: np.ndarray
    peak_widths: np.ndarray
    event_starts: np.ndarray
    event_ends: np.ndarray


@dataclass(frozen=True)
class CircadianAnalysisResult:
    """End-to-end session analysis output.

    Trace arrays are ``(samples, channels, sessions)``. Level arrays are
    ``(channels, sessions)``.
    """

    dff: np.ndarray
    dff_dynamic_corrected: np.ndarray
    dff_adjusted: np.ndarray
    dff_phasic: np.ndarray
    fitted_405: np.ndarray
    fitted_dynamic: np.ndarray
    calcium_filtered: np.ndarray
    fit_coefficients: np.ndarray
    event_counts: np.ndarray
    level_phasic: np.ndarray
    level_tonic: np.ndarray
    level_tonic_detrended: np.ndarray
    level_tonic_z: np.ndarray
    level_average: np.ndarray
    level_average_detrended: np.ndarray
    level_405_raw: np.ndarray
    level_465_raw: np.ndarray


@dataclass(frozen=True)
class LightPulseWindowResult:
    """Light-pulse response windows extracted from dF/F traces.

    Dictionaries use ``"ct6"``, ``"ct14"``, and ``"ct22"`` keys. Values are
    arrays with shape ``(90, channels * 3)`` for shifted/z-scored windows and
    ``(3, channels * 3)`` for AUC summaries.
    """

    shifted: dict[str, np.ndarray]
    zscored: dict[str, np.ndarray]
    auc: dict[str, np.ndarray]


@dataclass(frozen=True)
class SessionizedStreamPair:
    """Loaded stream dictionaries split into session matrices.

    ``isosbestic_405`` and ``calcium_465`` are shaped ``(samples, sessions)``.
    Session start/stop times are taken from the isosbestic timestamp array before
    any optional crop is applied.
    """

    isosbestic_405: np.ndarray
    calcium_465: np.ndarray
    fs: float
    session_start_times: np.ndarray
    session_stop_times: np.ndarray
    samples_per_session: int


@dataclass(frozen=True)
class TimestampGapReport:
    """Timestamp gap report for paired iso/experimental streams.

    ``gap_indices`` are zero-based sample indices where a new session starts.
    ``session_stop_indices`` are exclusive stop indices.
    """

    gap_indices: np.ndarray
    gap_durations: np.ndarray
    session_start_indices: np.ndarray
    session_stop_indices: np.ndarray
    session_start_times: np.ndarray
    session_stop_times: np.ndarray
    session_lengths: np.ndarray
    gap_threshold_seconds: float
    is_consistent: bool


@dataclass(frozen=True)
class IntervalHoursEstimate:
    """Suggested circadian analysis interval from stream session starts."""

    suggested_interval_hours: float
    raw_median_interval_hours: float
    interval_hours_min: float
    interval_hours_max: float
    interval_hours_mean: float
    interval_hours_std: float
    session_count: int
    interval_count: int
    is_regular: bool
