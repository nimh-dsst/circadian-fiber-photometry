"""Configurable artifacts for synthetic photometry signals."""

from .builders import (
    add_random_box_artifacts,
    add_scheduled_box_artifacts,
    configure_session_start_spike,
)
from .models import (
    SyntheticArtifactOccurrence,
    SyntheticRandomBoxArtifactConfig,
    SyntheticScheduledBoxArtifactConfig,
    SyntheticSessionStartSpikeConfig,
)

__all__ = [
    "SyntheticArtifactOccurrence",
    "SyntheticRandomBoxArtifactConfig",
    "SyntheticScheduledBoxArtifactConfig",
    "SyntheticSessionStartSpikeConfig",
    "add_random_box_artifacts",
    "add_scheduled_box_artifacts",
    "configure_session_start_spike",
]
