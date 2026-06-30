"""Circadian fiber photometry analysis utilities.

This package intentionally excludes Doric IO. Pass NumPy arrays from your
application's reader into the analysis functions exported here. It can generate
synthetic Doric HDF5 files for tests and simulations.
"""

from .analysis import (
    analyze_sessions,
    count_events,
    extract_light_pulse_windows,
    fit_405_to_465,
    irls_dynamic_correction,
)
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
from .streams import (
    analyze_stream_pair,
    detect_stream_gaps,
    estimate_interval_hours,
    sessionize_stream_pair,
)
from .synthetic_doric import (
    SyntheticDoricConfig,
    SyntheticDoricSummary,
    SyntheticSignalConfig,
    generate_synthetic_doric,
)

__all__ = [
    "CircadianAnalysisResult",
    "DffFitResult",
    "EventDetectionResult",
    "IRLSResult",
    "IntervalHoursEstimate",
    "LightPulseWindowResult",
    "SessionizedStreamPair",
    "SyntheticDoricConfig",
    "SyntheticDoricSummary",
    "SyntheticSignalConfig",
    "TimestampGapReport",
    "analyze_sessions",
    "analyze_stream_pair",
    "count_events",
    "detect_stream_gaps",
    "estimate_interval_hours",
    "extract_light_pulse_windows",
    "fit_405_to_465",
    "generate_synthetic_doric",
    "irls_dynamic_correction",
    "sessionize_stream_pair",
]
