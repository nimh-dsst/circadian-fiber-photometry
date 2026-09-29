"""Synthetic behavior-table generation.

The behavior package produces raw CSV tables that can be uploaded to a
downstream application such as ChronoXIV.  It does not write normalized
ChronoXIV session outputs or depend on Streamlit.
"""

from .simulation import (
    EVENT_ROLE_INTERVAL,
    EVENT_ROLE_POINT,
    EVENT_TYPE_STRUCTURED,
    EVENT_TYPE_UNSTRUCTURED,
    SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
    SOURCE_FORMAT_SINGLE_BINARY,
    SyntheticBehaviorCoMetricConfig,
    SyntheticBehaviorConfig,
    SyntheticBehaviorDataset,
    SyntheticBehaviorEventConfig,
    SyntheticBehaviorEventMapping,
    SyntheticBehaviorOccurrence,
    SyntheticBehaviorSummary,
    build_behavior_event_configs_from_ttl,
    generate_synthetic_behavior_csv,
    simulate_behavior,
    write_behavior_csv,
)

__all__ = [
    "EVENT_ROLE_INTERVAL",
    "EVENT_ROLE_POINT",
    "EVENT_TYPE_STRUCTURED",
    "EVENT_TYPE_UNSTRUCTURED",
    "SOURCE_FORMAT_PAIRED_ONSET_OFFSET",
    "SOURCE_FORMAT_SINGLE_BINARY",
    "SyntheticBehaviorCoMetricConfig",
    "SyntheticBehaviorConfig",
    "SyntheticBehaviorDataset",
    "SyntheticBehaviorEventConfig",
    "SyntheticBehaviorEventMapping",
    "SyntheticBehaviorOccurrence",
    "SyntheticBehaviorSummary",
    "build_behavior_event_configs_from_ttl",
    "generate_synthetic_behavior_csv",
    "simulate_behavior",
    "write_behavior_csv",
]
