"""Tonic-scale 405-to-465 fitting and initial dF/F calculation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq

from .._utils import moving_mean, normalize_session_array, require_matching_shapes
from ..results import DffFitResult


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
