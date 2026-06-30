"""Converted MATLAB analysis routines for circadian fiber photometry."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from scipy.signal import butter, filtfilt, find_peaks

from ._utils import (
    as_float_array,
    baseline_shift_and_zscore,
    matlab_std,
    moving_mean,
    moving_std,
    normalize_session_array,
    require_matching_shapes,
)
from .results import (
    CircadianAnalysisResult,
    DffFitResult,
    EventDetectionResult,
    IRLSResult,
    LightPulseWindowResult,
)


def fit_405_to_465(
    isosbestic_405: np.ndarray,
    calcium_465: np.ndarray,
    fs: float,
    fitting_cutoff: float = 0,
    weight_fit: bool = True,
    smooth_seconds: float = 5,
    fit_weights: np.ndarray | None = None,
) -> DffFitResult:
    """Fit 405 nm isosbestic traces onto 465 nm traces and compute dF/F.

    Inputs may be 1D traces, ``(samples, sessions)``, or
    ``(samples, channels, sessions)`` arrays. Outputs are always normalized to
    ``(samples, channels, sessions)``.

    ``fit_weights`` supplies conventional weighted least-squares weights for
    fitting 405 nm to 465 nm. It may match the normalized signal shape or
    broadcast to it; zero-weight samples are ignored. ``weight_fit`` preserves
    the legacy MATLAB behavior by overwriting the first session-length block of
    the pooled fitting arrays with zeros before regression.
    """

    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 <= fitting_cutoff < 50:
        raise ValueError("fitting_cutoff must be in [0, 50)")
    if smooth_seconds <= 0:
        raise ValueError("smooth_seconds must be positive")

    iso, _ = normalize_session_array(
        isosbestic_405, "isosbestic_405", allow_trace=True
    )
    calcium, _ = normalize_session_array(calcium_465, "calcium_465", allow_trace=True)
    require_matching_shapes(iso, calcium)
    weights = _normalize_fit_weights(fit_weights, iso.shape)

    samples, channels, sessions = iso.shape
    fitted_405 = np.empty_like(calcium)
    dff = np.empty_like(calcium)
    coefficients = np.empty((channels, 2), dtype=float)
    smooth_window = max(1, int(round(smooth_seconds * fs)))

    for channel in range(channels):
        flat_405 = np.ravel(iso[:, channel, :], order="F").copy()
        flat_465 = np.ravel(calcium[:, channel, :], order="F").copy()
        lower = np.percentile(flat_465, fitting_cutoff)
        upper = np.percentile(flat_465, 100 - fitting_cutoff)
        fit_mask = (
            (flat_465 > lower)
            & (flat_465 < upper)
            & np.isfinite(flat_405)
            & np.isfinite(flat_465)
        )
        flat_weights = None
        if weights is not None:
            flat_weights = np.ravel(weights[:, channel, :], order="F")
            fit_mask &= np.isfinite(flat_weights) & (flat_weights > 0)

        if weight_fit and sessions > 1:
            zero_count = min(samples, flat_405.size)
            flat_405[:zero_count] = 0
            flat_465[:zero_count] = 0

        if np.count_nonzero(fit_mask) < 2:
            raise ValueError(
                "not enough finite, non-extreme points to fit 405 to 465 signal"
            )

        fit_weight_values = None if flat_weights is None else flat_weights[fit_mask]
        slope, intercept = _weighted_linear_fit(
            flat_405[fit_mask],
            flat_465[fit_mask],
            fit_weight_values,
        )
        coefficients[channel, :] = [slope, intercept]

        for session in range(sessions):
            fit_temp = slope * iso[:, channel, session] + intercept
            fitted = moving_mean(fit_temp, smooth_window, axis=0)
            fitted_405[:, channel, session] = fitted
            denominator = np.median(fitted)
            dff[:, channel, session] = (calcium[:, channel, session] - fitted) / (
                denominator
            )

    return DffFitResult(dff=dff, fitted_405=fitted_405, coefficients=coefficients)


def _normalize_fit_weights(
    fit_weights: np.ndarray | None,
    target_shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Normalize nonnegative WLS weights to the signal shape."""

    if fit_weights is None:
        return None

    weights, _ = normalize_session_array(fit_weights, "fit_weights", allow_trace=True)
    try:
        weights = np.broadcast_to(weights, target_shape)
    except ValueError as exc:
        raise ValueError(
            "fit_weights must match or broadcast to the normalized signal shape "
            f"{target_shape}; got {np.asarray(fit_weights).shape}"
        ) from exc

    if not np.all(np.isfinite(weights)):
        raise ValueError("fit_weights must be finite")
    if np.any(weights < 0):
        raise ValueError("fit_weights must be nonnegative")
    if not np.any(weights > 0):
        raise ValueError("fit_weights must contain at least one positive value")
    return weights


def _weighted_linear_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit y = slope * x + intercept by ordinary or weighted least squares."""

    design = np.column_stack([x_values, np.ones_like(x_values)])
    response = y_values

    if weights is not None:
        sqrt_weights = np.sqrt(weights)
        design = design * sqrt_weights[:, np.newaxis]
        response = response * sqrt_weights

    coefficients, _, rank, _ = lstsq(design, response, check_finite=False)
    if rank < 2:
        raise ValueError("405 signal must vary enough to fit slope and intercept")
    slope, intercept = coefficients
    return float(slope), float(intercept)


def irls_dynamic_correction(
    calcium: np.ndarray,
    isosbestic: np.ndarray,
    fs: float,
    chunk_seconds: float = 150,
    lambda_: float = 0.6,
    irls_constant: float = 4.685,
) -> IRLSResult:
    """Apply chunk-wise robust dynamic artifact correction.

    This mirrors ``FP_IRLS_regularized_v9.m``: traces are mean-shifted, low-pass
    filtered at 0.8 Hz, robustly fit in chunks, and the residual is filtered.
    """

    if fs <= 1.6:
        raise ValueError("fs must be greater than 1.6 Hz for the 0.8 Hz low-pass")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be in [0, 1]")
    if irls_constant <= 0:
        raise ValueError("irls_constant must be positive")

    calcium_array = as_float_array(calcium, "calcium").reshape(-1)
    isosbestic_array = as_float_array(isosbestic, "isosbestic").reshape(-1)
    n = min(calcium_array.size, isosbestic_array.size)
    if n < 2:
        raise ValueError("calcium and isosbestic must have at least two samples")
    calcium_array = calcium_array[:n]
    isosbestic_array = isosbestic_array[:n]

    calcium_corr = calcium_array - np.mean(calcium_array) + 1
    isosbestic_corr = isosbestic_array - np.mean(isosbestic_array) + 1

    b_filter, a_filter = butter(1, 0.8 / (fs / 2), btype="low")
    calcium_filtered = _safe_filtfilt(b_filter, a_filter, calcium_corr)
    isosbestic_filtered = _safe_filtfilt(b_filter, a_filter, isosbestic_corr)

    fitted_isosbestic = np.zeros_like(calcium_filtered)
    beta_previous = _robust_linear_fit(
        isosbestic_filtered,
        calcium_filtered,
        tune=irls_constant,
    )
    chunk = max(1, int(round(chunk_seconds * fs)))

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x_chunk = isosbestic_filtered[start:stop]
        y_chunk = calcium_filtered[start:stop]

        if np.std(x_chunk) == 0:
            fitted_isosbestic[start:stop] = np.mean(y_chunk)
            continue

        beta = _robust_linear_fit(x_chunk, y_chunk, tune=irls_constant)
        beta = (1 - lambda_) * beta + lambda_ * beta_previous
        beta_previous = beta
        fitted_isosbestic[start:stop] = beta[0] + beta[1] * x_chunk

    corrected = calcium_filtered - fitted_isosbestic
    corrected = _safe_filtfilt(b_filter, a_filter, corrected)

    return IRLSResult(
        corrected=corrected,
        beta=beta_previous,
        fitted_isosbestic=fitted_isosbestic,
        calcium_filtered=calcium_filtered,
        isosbestic_filtered=isosbestic_filtered,
    )


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
    dff_dynamic = np.empty_like(fit.dff)
    dff_adjusted = np.empty_like(fit.dff)
    fitted_dynamic = np.empty_like(fit.dff)
    calcium_filtered = np.empty_like(fit.dff)
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
            dff_adjusted[:, channel, session] = dynamic.corrected - np.percentile(
                dynamic.corrected,
                10,
            )
            event_counts[channel, session] = count_events(
                dynamic.corrected,
                fs,
            ).count

    dff_phasic = dff_adjusted.copy()
    dff_phasic[dff_phasic < 0] = 0

    level_phasic = np.sum(dff_phasic, axis=0)
    level_465_raw = np.median(calcium, axis=0)
    level_405_raw = np.median(iso, axis=0)
    level_tonic = np.percentile(fit.dff, 10, axis=0)
    level_average = np.median(fit.dff, axis=0)

    daily_window = 24 / interval_hours
    level_tonic_detrended = level_tonic - moving_mean(
        level_tonic,
        daily_window,
        axis=1,
    )
    tonic_moving_std = moving_std(level_tonic, daily_window, axis=1, ddof=1)
    level_tonic_z = np.full_like(level_tonic_detrended, np.nan)
    np.divide(
        level_tonic_detrended,
        tonic_moving_std,
        out=level_tonic_z,
        where=tonic_moving_std != 0,
    )
    level_average_detrended = level_average - moving_mean(
        level_average,
        daily_window,
        axis=1,
    )

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
        level_phasic=level_phasic,
        level_tonic=level_tonic,
        level_tonic_detrended=level_tonic_detrended,
        level_tonic_z=level_tonic_z,
        level_average=level_average,
        level_average_detrended=level_average_detrended,
        level_405_raw=level_405_raw,
        level_465_raw=level_465_raw,
    )


def extract_light_pulse_windows(
    delta_f_over_f: np.ndarray,
    fs: float,
    int_before_ct6_lp: int = 63,
    interval_offset: int = 8,
) -> LightPulseWindowResult:
    """Extract CT6/CT14/CT22 light-pulse windows from dF/F session traces."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    step = int(round(fs))
    if step < 1:
        raise ValueError("fs must round to a positive integer")
    data, _ = normalize_session_array(delta_f_over_f, "delta_f_over_f")
    _, channels, sessions = data.shape
    session_indices = {
        "ct6": int_before_ct6_lp,
        "ct14": int_before_ct6_lp + interval_offset,
        "ct22": int_before_ct6_lp + 2 * interval_offset,
    }
    max_session = max(session_indices.values())
    if max_session >= sessions:
        raise ValueError(
            "delta_f_over_f does not contain enough sessions for CT6/CT14/CT22 "
            f"indices {session_indices}"
        )

    shifted: dict[str, np.ndarray] = {}
    zscored: dict[str, np.ndarray] = {}
    auc: dict[str, np.ndarray] = {}
    slices = [slice(100, 190), slice(220, 310), slice(340, 430)]

    for label, session_index in session_indices.items():
        columns = []
        for channel in range(channels):
            downsampled = data[::step, channel, session_index]
            if downsampled.size < 430:
                raise ValueError(
                    f"{label} session has {downsampled.size} downsampled samples; "
                    "at least 430 are required"
                )
            columns.extend(downsampled[window_slice] for window_slice in slices)
        windows = np.column_stack(columns)
        shifted_windows, zscored_windows = baseline_shift_and_zscore(windows)
        shifted[label] = shifted_windows
        zscored[label] = zscored_windows
        auc[label] = np.vstack(
            [
                _trapz(shifted_windows[0:15, :]),
                _trapz(shifted_windows[15:30, :]),
                _trapz(shifted_windows[30:90, :]),
            ]
        )

    return LightPulseWindowResult(shifted=shifted, zscored=zscored, auc=auc)


def _safe_filtfilt(
    b_filter: np.ndarray,
    a_filter: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Use zero-phase filtering when the trace is long enough."""

    padlen = 3 * max(len(a_filter), len(b_filter))
    if values.size <= padlen:
        return values.copy()
    return filtfilt(b_filter, a_filter, values)


def _robust_linear_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    tune: float,
    max_iter: int = 50,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Tukey bisquare IRLS linear fit returning ``[intercept, slope]``."""

    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        raise ValueError("robust linear fit requires at least two finite samples")

    design = np.column_stack([np.ones_like(x), x])
    beta = _least_squares(design, y)

    for _ in range(max_iter):
        residual = y - design @ beta
        scale = _robust_scale(residual)
        if scale == 0:
            break
        u = residual / (tune * scale)
        weights = np.zeros_like(u)
        inlier = np.abs(u) < 1
        weights[inlier] = (1 - u[inlier] ** 2) ** 2
        if np.count_nonzero(weights) < 2:
            break
        next_beta = _least_squares(design, y, weights)
        beta_delta = np.linalg.norm(next_beta - beta)
        beta_norm = max(np.linalg.norm(beta), np.finfo(float).eps)
        beta = next_beta
        if beta_delta / beta_norm < tolerance:
            break

    return beta


def _least_squares(
    design: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve unweighted or weighted least squares."""

    if weights is None:
        return np.linalg.lstsq(design, y_values, rcond=None)[0]

    sqrt_weights = np.sqrt(np.clip(weights, 0, None))
    weighted_design = design * sqrt_weights[:, np.newaxis]
    weighted_y = y_values * sqrt_weights
    return np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]


def _robust_scale(residual: np.ndarray) -> float:
    """Median-absolute-deviation scale estimate used by the IRLS loop."""

    centered = residual - np.median(residual)
    mad = np.median(np.abs(centered))
    if mad > 0:
        return float(mad / 0.6745)
    fallback = np.std(residual, ddof=1) if residual.size > 1 else 0.0
    return float(fallback)


def _trapz(values: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for NumPy's trapezoid integration."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, axis=0)
    return np.trapz(values, axis=0)
