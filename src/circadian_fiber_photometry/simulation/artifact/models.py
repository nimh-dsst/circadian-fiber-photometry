"""Public configuration and ground-truth models for simulated artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SyntheticSessionStartSpikeConfig:
    """Box-shaped artifact at the beginning of every simulated session."""

    duration_seconds: float = 1.0
    magnitude_fraction: float = 1.0
    name: str | None = None


@dataclass(frozen=True)
class SyntheticScheduledBoxArtifactConfig:
    """Explicit box artifacts with starts relative to each selected series."""

    start_times_seconds: tuple[float, ...]
    durations_seconds: float | tuple[float, ...] = 1.0
    magnitude_fraction: float = 0.10
    channels: tuple[int, ...] | None = None
    series_numbers: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SyntheticRandomBoxArtifactConfig:
    """Seeded random box-artifact source for selected series and channels."""

    count_per_series: int | None = None
    rate_per_minute: float | None = None
    start_window_seconds: tuple[float, float] | None = None
    duration_range_seconds: tuple[float, float] = (1.0, 1.0)
    magnitude_fraction: float = 0.10
    channels: tuple[int, ...] | None = None
    series_numbers: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SyntheticPhotometryDisconnectionConfig:
    """Equipment-shutdown artifact that clamps simulated lock-in signals."""

    start_seconds: float | None = None
    time_reference: Literal["series", "experiment"] = "experiment"
    duration_seconds: float | None = None
    isosbestic_floor: float = 0.0
    calcium_floor: float = 0.0
    channels: tuple[int, ...] | None = None
    series_numbers: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SyntheticArtifactOccurrence:
    """Ground truth for one artifact applied to one channel in one series.

    ``stop_sample`` is exclusive. Relative times describe the realized sample
    bounds within the series; absolute times include the series start time.
    Additive-only fraction and offset fields are ``None`` for a photometry
    disconnection, which instead reports the configured signal floors.
    """

    artifact_type: Literal[
        "session_start_spike",
        "scheduled_box",
        "random_box",
        "photometry_disconnection",
    ]
    name: str | None
    source_index: int
    series_number: int
    channel_number: int
    start_sample: int
    stop_sample: int
    requested_start_seconds: float
    requested_duration_seconds: float | None
    start_seconds_within_series: float
    stop_seconds_within_series: float
    absolute_start_seconds: float
    absolute_stop_seconds: float
    realized_duration_seconds: float
    magnitude_fraction: float | None
    isosbestic_offset: float | None
    calcium_offset: float | None
    time_reference: Literal["series", "experiment"] = "series"
    isosbestic_floor: float | None = None
    calcium_floor: float | None = None
