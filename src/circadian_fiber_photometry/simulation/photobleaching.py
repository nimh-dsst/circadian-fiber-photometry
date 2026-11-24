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
_DEFAULT_TURNOVER_HALF_LIFE_HOURS = 48.0
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
    requires two. The ``none`` model accepts no components. Protein turnover
    uses a shared half-life for both wavelengths; set
    ``turnover_half_life_hours`` to ``None`` to disable replacement.
    """

    model: PhotobleachingModel = "single_exponential"
    isosbestic_components: _ComponentTuple = None
    calcium_components: _ComponentTuple = None
    turnover_half_life_hours: float | None = _DEFAULT_TURNOVER_HALF_LIFE_HOURS


@dataclass(frozen=True)
class SyntheticPhotobleachingMetadata:
    """Resolved photobleaching ground truth for a synthetic Doric file."""

    model: PhotobleachingModel
    time_basis: str
    turnover_half_life_hours: float | None
    turnover_rate_per_second: float
    isosbestic_components: tuple[SyntheticPhotobleachingComponentConfig, ...]
    calcium_components: tuple[SyntheticPhotobleachingComponentConfig, ...]


@dataclass(frozen=True)
class _PhotobleachingFactors:
    """Resolved factors with shape ``(series, samples)`` for both signals."""

    isosbestic: np.ndarray
    calcium: np.ndarray


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

    turnover_half_life_hours, turnover_rate_per_second = _resolve_turnover(
        config.turnover_half_life_hours
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
        turnover_half_life_hours=turnover_half_life_hours,
        turnover_rate_per_second=turnover_rate_per_second,
        isosbestic_components=isosbestic,
        calcium_components=calcium,
    )


def photobleaching_factor(
    metadata: SyntheticPhotobleachingMetadata,
    exposure_time_seconds: np.ndarray,
    *,
    signal: Literal["isosbestic", "calcium"],
) -> np.ndarray:
    """Evaluate a resolved factor during uninterrupted illumination."""

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
    turnover_rate = metadata.turnover_rate_per_second
    for component in components:
        amplitude = component.amplitude_fraction
        time_constant = component.time_constant_seconds
        if time_constant is None:  # Resolved metadata never contains ``None``.
            raise ValueError(
                "photobleaching metadata contains an unresolved time constant"
            )
        bleaching_rate = 1.0 / time_constant
        combined_rate = bleaching_rate + turnover_rate
        steady_state = amplitude * turnover_rate / combined_rate
        component_state = steady_state + (amplitude - steady_state) * np.exp(
            -combined_rate * exposure_time
        )
        factor += component_state - amplitude
    return factor


def build_photobleaching_factors(
    metadata: SyntheticPhotobleachingMetadata,
    *,
    session_start_times_seconds: np.ndarray,
    samples_per_series: int,
    sampling_rate_hz: float,
) -> _PhotobleachingFactors:
    """Build factors while carrying renewable component pools across sessions."""

    starts = np.asarray(session_start_times_seconds, dtype=float)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("session_start_times_seconds must be a nonempty 1D array")
    if np.any(~np.isfinite(starts)) or np.any(np.diff(starts) <= 0):
        raise ValueError(
            "session_start_times_seconds must be finite and strictly increasing"
        )
    if not isinstance(samples_per_series, int) or samples_per_series <= 0:
        raise ValueError("samples_per_series must be a positive integer")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")

    keyword_arguments = {
        "turnover_rate_per_second": metadata.turnover_rate_per_second,
        "session_start_times_seconds": starts,
        "samples_per_series": samples_per_series,
        "sampling_rate_hz": sampling_rate_hz,
    }
    return _PhotobleachingFactors(
        isosbestic=_factor_schedule(
            metadata.isosbestic_components,
            **keyword_arguments,
        ),
        calcium=_factor_schedule(
            metadata.calcium_components,
            **keyword_arguments,
        ),
    )


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
        return SyntheticPhotobleachingConfig(
            model="none",
            turnover_half_life_hours=None,
        )

    component = SyntheticPhotobleachingComponentConfig(
        amplitude_fraction=float(legacy_bleaching_fraction)
    )
    return SyntheticPhotobleachingConfig(
        model="single_exponential",
        isosbestic_components=(component,),
        calcium_components=(component,),
        turnover_half_life_hours=None,
    )


def _resolve_turnover(
    turnover_half_life_hours: float | None,
) -> tuple[float | None, float]:
    if turnover_half_life_hours is None:
        return None, 0.0
    if (
        not math.isfinite(turnover_half_life_hours)
        or turnover_half_life_hours <= 0
    ):
        raise ValueError("turnover_half_life_hours must be finite and positive")
    resolved_half_life = float(turnover_half_life_hours)
    rate_per_second = math.log(2.0) / (resolved_half_life * 3600.0)
    return resolved_half_life, rate_per_second


def _factor_schedule(
    components: tuple[SyntheticPhotobleachingComponentConfig, ...],
    *,
    turnover_rate_per_second: float,
    session_start_times_seconds: np.ndarray,
    samples_per_series: int,
    sampling_rate_hz: float,
) -> np.ndarray:
    series_count = session_start_times_seconds.size
    factors = np.ones((series_count, samples_per_series), dtype=float)
    if not components:
        return factors

    amplitudes = np.array(
        [component.amplitude_fraction for component in components],
        dtype=float,
    )
    bleaching_rates = np.array(
        [
            1.0 / _resolved_time_constant(component)
            for component in components
        ],
        dtype=float,
    )
    combined_rates = bleaching_rates + turnover_rate_per_second
    steady_states = amplitudes * turnover_rate_per_second / combined_rates
    component_states = amplitudes.copy()
    relative_time = np.arange(samples_per_series, dtype=float) / sampling_rate_hz
    active_duration = samples_per_series / sampling_rate_hz
    unbleachable_fraction = 1.0 - float(amplitudes.sum())

    for series_index in range(series_count):
        illuminated_states = steady_states[:, None] + (
            component_states - steady_states
        )[:, None] * np.exp(-combined_rates[:, None] * relative_time[None, :])
        factors[series_index] = unbleachable_fraction + illuminated_states.sum(axis=0)

        component_states = steady_states + (
            component_states - steady_states
        ) * np.exp(-combined_rates * active_duration)
        if series_index + 1 == series_count:
            continue

        start_delta = (
            session_start_times_seconds[series_index + 1]
            - session_start_times_seconds[series_index]
        )
        dark_duration = start_delta - active_duration
        rounding_tolerance = 0.5 / sampling_rate_hz + 1e-12
        if dark_duration < -rounding_tolerance:
            raise ValueError("session start times overlap active recording intervals")
        dark_duration = max(0.0, dark_duration)
        if turnover_rate_per_second > 0 and dark_duration > 0:
            component_states = amplitudes + (
                component_states - amplitudes
            ) * np.exp(-turnover_rate_per_second * dark_duration)

    return factors


def _resolved_time_constant(
    component: SyntheticPhotobleachingComponentConfig,
) -> float:
    time_constant = component.time_constant_seconds
    if time_constant is None:  # Resolved metadata never contains ``None``.
        raise ValueError("photobleaching metadata contains an unresolved time constant")
    return time_constant


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
