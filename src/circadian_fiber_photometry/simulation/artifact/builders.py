"""Immutable configuration helpers for artifact simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .models import (
    SyntheticRandomBoxArtifactConfig,
    SyntheticScheduledBoxArtifactConfig,
    SyntheticSessionStartSpikeConfig,
)

if TYPE_CHECKING:
    from ..doric import SyntheticSignalConfig


def configure_session_start_spike(
    signal: SyntheticSignalConfig,
    *,
    duration_seconds: float = 1.0,
    magnitude_fraction: float = 1.0,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` configured with one spike at every series start."""

    spike = SyntheticSessionStartSpikeConfig(
        duration_seconds=duration_seconds,
        magnitude_fraction=magnitude_fraction,
        name=name,
    )
    return replace(signal, session_start_spike=spike)


def add_scheduled_box_artifacts(
    signal: SyntheticSignalConfig,
    start_times_seconds: Sequence[float] | np.ndarray,
    *,
    durations_seconds: float | Sequence[float] | np.ndarray = 1.0,
    magnitude_fraction: float = 0.10,
    channels: Sequence[int] | np.ndarray | None = None,
    series_numbers: Sequence[int] | np.ndarray | None = None,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` with explicit signed box artifacts appended."""

    normalized_durations: float | tuple[float, ...]
    if np.isscalar(durations_seconds):
        normalized_durations = float(durations_seconds)
    else:
        normalized_durations = _normalize_float_tuple(durations_seconds)
    artifact = SyntheticScheduledBoxArtifactConfig(
        start_times_seconds=_normalize_float_tuple(start_times_seconds),
        durations_seconds=normalized_durations,
        magnitude_fraction=magnitude_fraction,
        channels=_normalize_optional_int_tuple(channels),
        series_numbers=_normalize_optional_int_tuple(series_numbers),
        name=name,
    )
    return replace(
        signal,
        scheduled_box_artifacts=signal.scheduled_box_artifacts + (artifact,),
    )


def add_random_box_artifacts(
    signal: SyntheticSignalConfig,
    *,
    count_per_series: int | None = None,
    rate_per_minute: float | None = None,
    start_window_seconds: tuple[float, float] | None = None,
    duration_range_seconds: tuple[float, float] = (1.0, 1.0),
    magnitude_fraction: float = 0.10,
    channels: Sequence[int] | np.ndarray | None = None,
    series_numbers: Sequence[int] | np.ndarray | None = None,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` with a seeded random box-artifact source appended."""

    artifact = SyntheticRandomBoxArtifactConfig(
        count_per_series=count_per_series,
        rate_per_minute=rate_per_minute,
        start_window_seconds=(
            None
            if start_window_seconds is None
            else tuple(float(value) for value in start_window_seconds)
        ),
        duration_range_seconds=tuple(
            float(value) for value in duration_range_seconds
        ),
        magnitude_fraction=magnitude_fraction,
        channels=_normalize_optional_int_tuple(channels),
        series_numbers=_normalize_optional_int_tuple(series_numbers),
        name=name,
    )
    return replace(
        signal,
        random_box_artifacts=signal.random_box_artifacts + (artifact,),
    )


def _normalize_optional_int_tuple(
    values: Sequence[int] | np.ndarray | None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(
        _coerce_selector_int(value) for value in np.asarray(values).reshape(-1)
    )


def _normalize_float_tuple(values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float).reshape(-1))


def _coerce_selector_int(value: Any) -> int:
    if isinstance(value, (np.bool_, bool)):
        raise ValueError("selector values must be integers")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    raise ValueError("selector values must be integers")
