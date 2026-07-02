"""Synthetic data generation utilities."""

from .doric import (
    SyntheticDoricConfig,
    SyntheticDoricSummary,
    SyntheticGaussianNoiseConfig,
    SyntheticRandomCalciumEventConfig,
    SyntheticScheduledCalciumEventConfig,
    SyntheticSignalConfig,
    SyntheticTonicComponentConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    SyntheticTTLBehaviorEventSummary,
    SyntheticTTLRandomBehaviorEventConfig,
    add_gaussian_noise,
    add_random_calcium_events,
    add_scheduled_calcium_events,
    add_tonic_component,
    generate_synthetic_doric,
)

__all__ = [
    "SyntheticDoricConfig",
    "SyntheticDoricSummary",
    "SyntheticGaussianNoiseConfig",
    "SyntheticRandomCalciumEventConfig",
    "SyntheticScheduledCalciumEventConfig",
    "SyntheticSignalConfig",
    "SyntheticTTLBehaviorCodeConfig",
    "SyntheticTTLBehaviorEventConfig",
    "SyntheticTTLBehaviorEventSummary",
    "SyntheticTTLRandomBehaviorEventConfig",
    "SyntheticTonicComponentConfig",
    "add_gaussian_noise",
    "add_random_calcium_events",
    "add_scheduled_calcium_events",
    "add_tonic_component",
    "generate_synthetic_doric",
]
