"""Light-pulse phasic response windows."""

from __future__ import annotations

import numpy as np

from .._utils import baseline_shift_and_zscore, normalize_session_array
from ..results import LightPulseWindowResult


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


def _trapz(values: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for NumPy's trapezoid integration."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, axis=0)
    return np.trapz(values, axis=0)
