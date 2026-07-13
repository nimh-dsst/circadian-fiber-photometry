"""Photobleaching models for synthetic photometry traces."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

PhotobleachingModel = Literal[
    "none",
    "single_exponential",
    "double_exponential",
]

_TIME_BASIS = "cumulative_exposure_seconds"
_DOUBLE_FAST_TAU_SECONDS = -1.0 / math.log(0.98)
_DOUBLE_SLOW_TAU_SECONDS = -1.0 / math.log(0.998)


@dataclass(frozen=True)
class SyntheticPhotobleachingComponentConfig:
    """One fractional-loss component in an exponential decay model.

    ``amplitude_fraction`` is the component's long-run fractional signal loss.
    ``time_constant_seconds`` is the exponential time constant. A value of
    ``None`` selects the documented default for the component and model.
    """

    amplitude_fraction: float
    time_constant_seconds: float | None = None


_ComponentTuple = tuple[SyntheticPhotobleachingComponentConfig, ...] | None


@dataclass(frozen=True)
class SyntheticPhotobleachingConfig:
    """Select and configure photobleaching for both photometry wavelengths.

    Component tuples may be omitted to use model-specific defaults. A single
    exponential requires one component per signal and a double exponential
    requires two. The ``none`` model accepts no components.
    """

    model: PhotobleachingModel = "single_exponential"
    isosbestic_components: _ComponentTuple = None
    calcium_components: _ComponentTuple = None


@dataclass(frozen=True)
class SyntheticPhotobleachingMetadata:
    """Resolved photobleaching ground truth for a synthetic Doric file."""

    model: PhotobleachingModel
    time_basis: str
    isosbestic_components: tuple[SyntheticPhotobleachingComponentConfig, ...]
    calcium_components: tuple[SyntheticPhotobleachingComponentConfig, ...]


def resolve_photobleaching(
    config: SyntheticPhotobleachingConfig,
    *,
    active_duration_seconds: float,
    legacy_bleaching_fraction: float | None,
) -> SyntheticPhotobleachingMetadata:
    """Validate and resolve defaults for one simulation configuration."""

    if not math.isfinite(active_duration_seconds) or active_duration_seconds <= 0:
        raise ValueError("active_duration_seconds must be positive")

    if legacy_bleaching_fraction is not None:
        config = _resolve_legacy_config(config, legacy_bleaching_fraction)

    if config.model not in {
        "none",
        "single_exponential",
        "double_exponential",
    }:
        raise ValueError(
            "photobleaching model must be 'none', 'single_exponential', or "
            "'double_exponential'"
        )

    defaults = _default_components(config.model, active_duration_seconds)
    isosbestic = _resolve_components(
        config.isosbestic_components,
        defaults,
        config.model,
        "isosbestic",
    )
    calcium = _resolve_components(
        config.calcium_components,
        defaults,
        config.model,
        "calcium",
    )
    return SyntheticPhotobleachingMetadata(
        model=config.model,
        time_basis=_TIME_BASIS,
        isosbestic_components=isosbestic,
        calcium_components=calcium,
    )


def photobleaching_factor(
    metadata: SyntheticPhotobleachingMetadata,
    exposure_time_seconds: np.ndarray,
    *,
    signal: Literal["isosbestic", "calcium"],
) -> np.ndarray:
    """Evaluate a resolved photobleaching factor at exposure times."""

    exposure_time = np.asarray(exposure_time_seconds, dtype=float)
    if np.any(~np.isfinite(exposure_time)) or np.any(exposure_time < 0):
        raise ValueError("exposure_time_seconds must be finite and nonnegative")

    if signal == "isosbestic":
        components = metadata.isosbestic_components
    elif signal == "calcium":
        components = metadata.calcium_components
    else:
        raise ValueError("signal must be 'isosbestic' or 'calcium'")
    factor = np.ones_like(exposure_time, dtype=float)
    for component in components:
        amplitude = component.amplitude_fraction
        time_constant = component.time_constant_seconds
        if time_constant is None:  # Resolved metadata never contains ``None``.
            raise ValueError(
                "photobleaching metadata contains an unresolved time constant"
            )
        factor += amplitude * (np.exp(-exposure_time / time_constant) - 1.0)
    return factor


def _resolve_legacy_config(
    config: SyntheticPhotobleachingConfig,
    legacy_bleaching_fraction: float,
) -> SyntheticPhotobleachingConfig:
    if (
        not math.isfinite(legacy_bleaching_fraction)
        or legacy_bleaching_fraction < 0
        or legacy_bleaching_fraction >= 1
    ):
        raise ValueError("bleaching_fraction must be in [0, 1)")
    if config != SyntheticPhotobleachingConfig():
        raise ValueError(
            "bleaching_fraction cannot be combined with an explicit photobleaching "
            "configuration"
        )

    warnings.warn(
        "bleaching_fraction is deprecated; configure photobleaching with "
        "SyntheticPhotobleachingConfig instead",
        DeprecationWarning,
        stacklevel=5,
    )
    if legacy_bleaching_fraction == 0:
        return SyntheticPhotobleachingConfig(model="none")

    component = SyntheticPhotobleachingComponentConfig(
        amplitude_fraction=float(legacy_bleaching_fraction)
    )
    return SyntheticPhotobleachingConfig(
        model="single_exponential",
        isosbestic_components=(component,),
        calcium_components=(component,),
    )


def _default_components(
    model: PhotobleachingModel,
    active_duration_seconds: float,
) -> tuple[SyntheticPhotobleachingComponentConfig, ...]:
    if model == "none":
        return ()
    if model == "single_exponential":
        return (
            SyntheticPhotobleachingComponentConfig(
                amplitude_fraction=1.0,
                time_constant_seconds=2.0 * active_duration_seconds,
            ),
        )
    return (
        SyntheticPhotobleachingComponentConfig(
            amplitude_fraction=0.20,
            time_constant_seconds=_DOUBLE_FAST_TAU_SECONDS,
        ),
        SyntheticPhotobleachingComponentConfig(
            amplitude_fraction=0.20,
            time_constant_seconds=_DOUBLE_SLOW_TAU_SECONDS,
        ),
    )


def _resolve_components(
    configured: _ComponentTuple,
    defaults: tuple[SyntheticPhotobleachingComponentConfig, ...],
    model: PhotobleachingModel,
    signal_name: str,
) -> tuple[SyntheticPhotobleachingComponentConfig, ...]:
    components = defaults if configured is None else tuple(configured)
    expected_count = {"none": 0, "single_exponential": 1, "double_exponential": 2}[
        model
    ]
    if len(components) != expected_count:
        raise ValueError(
            f"{signal_name}_components must contain {expected_count} component(s) "
            f"for the {model} photobleaching model"
        )

    resolved: list[SyntheticPhotobleachingComponentConfig] = []
    for index, component in enumerate(components):
        amplitude = component.amplitude_fraction
        if not math.isfinite(amplitude) or not 0 <= amplitude <= 1:
            raise ValueError(
                f"{signal_name} photobleaching amplitudes must be finite and in [0, 1]"
            )
        time_constant = component.time_constant_seconds
        if time_constant is None:
            time_constant = defaults[index].time_constant_seconds
        if (
            time_constant is None
            or not math.isfinite(time_constant)
            or time_constant <= 0
        ):
            raise ValueError(
                f"{signal_name} photobleaching time constants must be finite and "
                "positive"
            )
        resolved.append(
            SyntheticPhotobleachingComponentConfig(
                amplitude_fraction=float(amplitude),
                time_constant_seconds=float(time_constant),
            )
        )

    if sum(component.amplitude_fraction for component in resolved) > 1:
        raise ValueError(
            f"{signal_name} photobleaching amplitudes must sum to at most 1"
        )
    return tuple(resolved)
