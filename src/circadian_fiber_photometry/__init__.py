"""Circadian fiber photometry analysis utilities.

Tonic functions live in ``circadian_fiber_photometry.tonic``. Phasic functions
live in ``circadian_fiber_photometry.phasic``. Synthetic Doric generation lives
in ``circadian_fiber_photometry.simulator``. Common functions are also exported
at the package root for compatibility.
"""

from .phasic import (
    compute_phasic_level,
    compute_phasic_trace,
    count_events,
    extract_light_pulse_windows,
    integrated_fluorescence,
    irls_dynamic_correction,
    percentile_adjust,
    positive_only,
)
from .pipeline import analyze_sessions
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
from .simulator import (
    SyntheticDoricConfig,
    SyntheticDoricSummary,
    SyntheticSignalConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    SyntheticTTLBehaviorEventSummary,
    SyntheticTTLRandomBehaviorEventConfig,
    generate_synthetic_doric,
)
from .streams import (
    analyze_stream_pair,
    detect_stream_gaps,
    estimate_interval_hours,
    sessionize_stream_pair,
)
from .tonic import (
    compute_average_level,
    compute_raw_median_level,
    compute_tonic_level,
    detrend_levels_by_moving_window,
    fit_405_to_465,
    zscore_levels_by_moving_window,
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
    "SyntheticTTLBehaviorCodeConfig",
    "SyntheticTTLBehaviorEventConfig",
    "SyntheticTTLBehaviorEventSummary",
    "SyntheticTTLRandomBehaviorEventConfig",
    "TimestampGapReport",
    "analyze_sessions",
    "analyze_stream_pair",
    "compute_average_level",
    "compute_phasic_level",
    "compute_phasic_trace",
    "compute_raw_median_level",
    "compute_tonic_level",
    "count_events",
    "detect_stream_gaps",
    "detrend_levels_by_moving_window",
    "estimate_interval_hours",
    "extract_light_pulse_windows",
    "fit_405_to_465",
    "generate_synthetic_doric",
    "integrated_fluorescence",
    "irls_dynamic_correction",
    "percentile_adjust",
    "positive_only",
    "sessionize_stream_pair",
    "zscore_levels_by_moving_window",
]
