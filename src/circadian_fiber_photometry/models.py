"""Public data models for package APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .results import (
    CircadianAnalysisResult,
    DffFitResult,
    EventDetectionResult,
    IntervalHoursEstimate,
    IRLSResult,
    LightPulseWindowResult,
    SessionizedStreamPair,
    TimestampGapReport,
)


@dataclass(frozen=True)
class AnalysisResult:
    """Structured, UI-independent output from one registered analysis."""

    name: str
    tables: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisSpec:
    """Discoverable metadata for an analysis implementation."""

    name: str
    description: str
    expected_inputs: tuple[str, ...]
    config_fields: dict[str, Any]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class DoricDataset:
    """Normalized Doric FPConsole dataset loaded from HDF5.

    Photometry arrays are shaped ``(samples, channels, series)``. Auxiliary
    analog and digital streams are dictionaries keyed by Doric channel name with
    values shaped ``(samples, series)``.
    """

    path: Path
    isosbestic_405: np.ndarray
    calcium_465: np.ndarray
    timestamps: np.ndarray
    fs: float
    series_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    analog_in: dict[str, np.ndarray] = field(default_factory=dict)
    analog_out: dict[str, np.ndarray] = field(default_factory=dict)
    digital_io: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def samples_per_series(self) -> int:
        """Number of samples in each loaded series."""

        return int(self.isosbestic_405.shape[0])

    @property
    def channel_count(self) -> int:
        """Number of photometry input channels."""

        return int(self.isosbestic_405.shape[1])

    @property
    def series_count(self) -> int:
        """Number of loaded series."""

        return int(self.isosbestic_405.shape[2])

    @property
    def session_start_times(self) -> np.ndarray:
        """Absolute start time for each series."""

        return self.timestamps[0, :].copy()


__all__ = [
    "AnalysisResult",
    "AnalysisSpec",
    "CircadianAnalysisResult",
    "DffFitResult",
    "DoricDataset",
    "EventDetectionResult",
    "IRLSResult",
    "IntervalHoursEstimate",
    "LightPulseWindowResult",
    "SessionizedStreamPair",
    "TimestampGapReport",
]
