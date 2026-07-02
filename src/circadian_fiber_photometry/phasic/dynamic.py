"""Phasic-scale dynamic artifact correction."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .._utils import as_float_array
from ..results import IRLSResult


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
