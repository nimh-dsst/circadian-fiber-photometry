"""High-level analysis pipelines composed from tonic and phasic functions."""

from __future__ import annotations

import numpy as np

from ._utils import normalize_session_array, require_matching_shapes
from .phasic import (
    compute_phasic_trace,
    count_events,
    irls_dynamic_correction,
    percentile_adjust,
)
from .results import CircadianAnalysisResult
from .tonic import (
    compute_average_level,
    compute_raw_median_level,
    compute_tonic_level,
    detrend_levels_by_moving_window,
    fit_405_to_465,
    zscore_levels_by_moving_window,
)


def analyze_sessions(
    isosbestic_405: np.ndarray,
    calcium_465: np.ndarray,
    fs: float = 60,
    interval_hours: float = 1,
    fitting_cutoff: float = 0,
    weight_fit: bool = True,
    fit_weights: np.ndarray | None = None,
) -> CircadianAnalysisResult:
    """Run the converted circadian fiber photometry session analysis."""

    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    iso, _ = normalize_session_array(isosbestic_405, "isosbestic_405")
    calcium, _ = normalize_session_array(calcium_465, "calcium_465")
    require_matching_shapes(iso, calcium)

    fit = fit_405_to_465(
        iso,
        calcium,
        fs,
        fitting_cutoff=fitting_cutoff,
        weight_fit=weight_fit,
        fit_weights=fit_weights,
    )

    samples, channels, sessions = fit.dff.shape
    dff_dynamic = np.empty((samples, channels, sessions), dtype=float)
    fitted_dynamic = np.empty_like(dff_dynamic)
    calcium_filtered = np.empty_like(dff_dynamic)
    event_counts = np.empty((channels, sessions), dtype=int)

    for channel in range(channels):
        for session in range(sessions):
            dynamic = irls_dynamic_correction(
                fit.dff[:, channel, session],
                iso[:, channel, session],
                fs,
                chunk_seconds=150,
                lambda_=0.6,
                irls_constant=4.685,
            )
            dff_dynamic[:, channel, session] = dynamic.corrected
            fitted_dynamic[:, channel, session] = dynamic.fitted_isosbestic
            calcium_filtered[:, channel, session] = dynamic.calcium_filtered
            event_counts[channel, session] = count_events(
                dynamic.corrected,
                fs,
            ).count

    dff_adjusted = percentile_adjust(dff_dynamic, percentile=10)
    dff_phasic = compute_phasic_trace(dff_dynamic, percentile=10)

    level_tonic = compute_tonic_level(fit.dff, percentile=10)
    level_average = compute_average_level(fit.dff)

    return CircadianAnalysisResult(
        dff=fit.dff,
        dff_dynamic_corrected=dff_dynamic,
        dff_adjusted=dff_adjusted,
        dff_phasic=dff_phasic,
        fitted_405=fit.fitted_405,
        fitted_dynamic=fitted_dynamic,
        calcium_filtered=calcium_filtered,
        fit_coefficients=fit.coefficients,
        event_counts=event_counts,
        level_phasic=np.sum(dff_phasic, axis=0),
        level_tonic=level_tonic,
        level_tonic_detrended=detrend_levels_by_moving_window(
            level_tonic,
            interval_hours=interval_hours,
            window_hours=24,
        ),
        level_tonic_z=zscore_levels_by_moving_window(
            level_tonic,
            interval_hours=interval_hours,
            window_hours=24,
            ddof=1,
        ),
        level_average=level_average,
        level_average_detrended=detrend_levels_by_moving_window(
            level_average,
            interval_hours=interval_hours,
            window_hours=24,
        ),
        level_405_raw=compute_raw_median_level(iso),
        level_465_raw=compute_raw_median_level(calcium),
    )
