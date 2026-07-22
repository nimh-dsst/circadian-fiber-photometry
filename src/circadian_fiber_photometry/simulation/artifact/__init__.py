"""Configurable artifacts for synthetic photometry signals."""

from .builders import (
    add_random_box_artifacts,
    add_scheduled_box_artifacts,
    configure_photometry_disconnection,
    configure_session_start_spike,
)
from .models import (
    SyntheticArtifactOccurrence,
    SyntheticPhotometryDisconnectionConfig,
    SyntheticRandomBoxArtifactConfig,
    SyntheticScheduledBoxArtifactConfig,
    SyntheticSessionStartSpikeConfig,
)

__all__ = [
    "SyntheticArtifactOccurrence",
    "SyntheticPhotometryDisconnectionConfig",
    "SyntheticRandomBoxArtifactConfig",
    "SyntheticScheduledBoxArtifactConfig",
    "SyntheticSessionStartSpikeConfig",
    "add_random_box_artifacts",
    "add_scheduled_box_artifacts",
    "configure_photometry_disconnection",
    "configure_session_start_spike",
]
