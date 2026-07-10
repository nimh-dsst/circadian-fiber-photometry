"""Validation, scheduling, and application of simulated artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .models import (
    SyntheticArtifactOccurrence,
    SyntheticRandomBoxArtifactConfig,
    SyntheticScheduledBoxArtifactConfig,
    SyntheticSessionStartSpikeConfig,
)

_ArtifactType = Literal["session_start_spike", "scheduled_box", "random_box"]


@dataclass(frozen=True)
class _PlannedArtifact:
    artifact_type: _ArtifactType
    name: str | None
    source_index: int
    start_sample: int
    stop_sample: int
    requested_start_seconds: float
    requested_duration_seconds: float
    magnitude_fraction: float


@dataclass(frozen=True)
class ArtifactSchedule:
    """Realized artifact intervals keyed by one-based series and channel."""

    intervals_by_key: dict[tuple[int, int], tuple[_PlannedArtifact, ...]]


def build_artifact_schedule(
    *,
    session_start_spike: SyntheticSessionStartSpikeConfig | None,
    scheduled_box_artifacts: tuple[SyntheticScheduledBoxArtifactConfig, ...],
    random_box_artifacts: tuple[SyntheticRandomBoxArtifactConfig, ...],
    series_count: int,
    channel_count: int,
    samples_per_series: int,
    session_duration_seconds: float,
    fs: float,
    seed: int,
) -> ArtifactSchedule:
    """Validate artifact settings and realize all deterministic/random bounds."""

    intervals_by_key: dict[tuple[int, int], list[_PlannedArtifact]] = {
        (series_number, channel_number): []
        for series_number in range(1, series_count + 1)
        for channel_number in range(1, channel_count + 1)
    }
    box_bounds_by_key: dict[tuple[int, int], list[tuple[int, int]]] = {
        key: [] for key in intervals_by_key
    }

    if session_start_spike is not None:
        _add_session_start_spikes(
            session_start_spike,
            samples_per_series,
            fs,
            intervals_by_key,
        )

    for source_index, artifact_config in enumerate(scheduled_box_artifacts):
        _add_scheduled_boxes(
            artifact_config,
            source_index,
            series_count,
            channel_count,
            samples_per_series,
            fs,
            intervals_by_key,
            box_bounds_by_key,
        )

    artifact_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x415254])
    )
    for source_index, artifact_config in enumerate(random_box_artifacts):
        _add_random_boxes(
            artifact_rng,
            artifact_config,
            source_index,
            series_count,
            channel_count,
            samples_per_series,
            session_duration_seconds,
            fs,
            intervals_by_key,
            box_bounds_by_key,
        )

    return ArtifactSchedule(
        intervals_by_key={
            key: tuple(
                sorted(
                    intervals,
                    key=lambda item: (
                        item.start_sample,
                        item.stop_sample,
                        item.artifact_type,
                        item.source_index,
                    ),
                )
            )
            for key, intervals in intervals_by_key.items()
        }
    )


def apply_artifact_schedule(
    schedule: ArtifactSchedule,
    *,
    series_number: int,
    channel_number: int,
    series_start_seconds: float,
    fs: float,
    isosbestic: np.ndarray,
    calcium: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[SyntheticArtifactOccurrence, ...]]:
    """Apply scheduled offsets and return copied traces plus ground truth."""

    intervals = schedule.intervals_by_key[(series_number, channel_number)]
    if not intervals:
        return isosbestic, calcium, ()

    isosbestic_mean = float(np.mean(isosbestic))
    calcium_mean = float(np.mean(calcium))
    artifact_isosbestic = isosbestic.copy()
    artifact_calcium = calcium.copy()
    occurrences: list[SyntheticArtifactOccurrence] = []

    for interval in intervals:
        isosbestic_offset = interval.magnitude_fraction * isosbestic_mean
        calcium_offset = interval.magnitude_fraction * calcium_mean
        artifact_isosbestic[interval.start_sample : interval.stop_sample] += (
            isosbestic_offset
        )
        artifact_calcium[interval.start_sample : interval.stop_sample] += (
            calcium_offset
        )

        start_seconds = interval.start_sample / fs
        stop_seconds = interval.stop_sample / fs
        occurrences.append(
            SyntheticArtifactOccurrence(
                artifact_type=interval.artifact_type,
                name=interval.name,
                source_index=interval.source_index,
                series_number=series_number,
                channel_number=channel_number,
                start_sample=interval.start_sample,
                stop_sample=interval.stop_sample,
                requested_start_seconds=interval.requested_start_seconds,
                requested_duration_seconds=interval.requested_duration_seconds,
                start_seconds_within_series=start_seconds,
                stop_seconds_within_series=stop_seconds,
                absolute_start_seconds=series_start_seconds + start_seconds,
                absolute_stop_seconds=series_start_seconds + stop_seconds,
                realized_duration_seconds=stop_seconds - start_seconds,
                magnitude_fraction=interval.magnitude_fraction,
                isosbestic_offset=isosbestic_offset,
                calcium_offset=calcium_offset,
            )
        )

    return artifact_isosbestic, artifact_calcium, tuple(occurrences)


def _add_session_start_spikes(
    config: SyntheticSessionStartSpikeConfig,
    samples_per_series: int,
    fs: float,
    intervals_by_key: dict[tuple[int, int], list[_PlannedArtifact]],
) -> None:
    duration_samples = _positive_seconds_to_samples(
        config.duration_seconds,
        fs,
        "session-start spike duration_seconds",
    )
    _require_finite(config.magnitude_fraction, "session-start spike magnitude_fraction")
    if duration_samples > samples_per_series:
        raise ValueError("session-start spike duration extends beyond the session")

    interval = _PlannedArtifact(
        artifact_type="session_start_spike",
        name=config.name,
        source_index=0,
        start_sample=0,
        stop_sample=duration_samples,
        requested_start_seconds=0.0,
        requested_duration_seconds=float(config.duration_seconds),
        magnitude_fraction=float(config.magnitude_fraction),
    )
    for intervals in intervals_by_key.values():
        intervals.append(interval)


def _add_scheduled_boxes(
    config: SyntheticScheduledBoxArtifactConfig,
    source_index: int,
    series_count: int,
    channel_count: int,
    samples_per_series: int,
    fs: float,
    intervals_by_key: dict[tuple[int, int], list[_PlannedArtifact]],
    box_bounds_by_key: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    starts = tuple(float(value) for value in config.start_times_seconds)
    if not starts:
        raise ValueError("start_times_seconds must contain at least one time")
    durations = _expand_scheduled_durations(config.durations_seconds, len(starts))
    channels = _resolve_selectors(config.channels, channel_count, "channels")
    series_numbers = _resolve_selectors(
        config.series_numbers,
        series_count,
        "series_numbers",
    )
    _require_finite(config.magnitude_fraction, "box artifact magnitude_fraction")

    planned: list[_PlannedArtifact] = []
    for start_seconds, duration_seconds in zip(starts, durations, strict=True):
        start_sample = _nonnegative_seconds_to_samples(
            start_seconds,
            fs,
            "start_times_seconds",
        )
        duration_samples = _positive_seconds_to_samples(
            duration_seconds,
            fs,
            "durations_seconds",
        )
        stop_sample = start_sample + duration_samples
        if start_sample >= samples_per_series or stop_sample > samples_per_series:
            raise ValueError("scheduled box artifact extends beyond the session")
        planned.append(
            _PlannedArtifact(
                artifact_type="scheduled_box",
                name=config.name,
                source_index=source_index,
                start_sample=start_sample,
                stop_sample=stop_sample,
                requested_start_seconds=start_seconds,
                requested_duration_seconds=duration_seconds,
                magnitude_fraction=float(config.magnitude_fraction),
            )
        )

    for series_number in series_numbers:
        for channel_number in channels:
            key = (series_number, channel_number)
            for interval in planned:
                bounds = (interval.start_sample, interval.stop_sample)
                if _overlaps_any(bounds, box_bounds_by_key[key]):
                    raise ValueError(
                        "box artifacts overlap for series "
                        f"{series_number} channel {channel_number}"
                    )
                intervals_by_key[key].append(interval)
                box_bounds_by_key[key].append(bounds)


def _add_random_boxes(
    rng: np.random.Generator,
    config: SyntheticRandomBoxArtifactConfig,
    source_index: int,
    series_count: int,
    channel_count: int,
    samples_per_series: int,
    session_duration_seconds: float,
    fs: float,
    intervals_by_key: dict[tuple[int, int], list[_PlannedArtifact]],
    box_bounds_by_key: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    channels = _resolve_selectors(config.channels, channel_count, "channels")
    series_numbers = _resolve_selectors(
        config.series_numbers,
        series_count,
        "series_numbers",
    )
    event_count, rate = _validate_random_density(config)
    _require_finite(config.magnitude_fraction, "box artifact magnitude_fraction")
    window_start, window_stop = _random_window_samples(
        config.start_window_seconds,
        samples_per_series,
        session_duration_seconds,
        fs,
    )
    _, maximum_duration = _validate_duration_range(
        config.duration_range_seconds,
        fs,
    )
    if maximum_duration > window_stop - window_start:
        raise ValueError("duration_range_seconds does not fit in the random window")

    for series_number in series_numbers:
        realized_count = event_count
        if rate is not None:
            window_minutes = (window_stop - window_start) / fs / 60
            realized_count = int(rng.poisson(rate * window_minutes))
        assert realized_count is not None

        for _ in range(realized_count):
            duration_seconds = _draw_duration_seconds(
                rng,
                config.duration_range_seconds,
            )
            duration_samples = _positive_seconds_to_samples(
                duration_seconds,
                fs,
                "duration_range_seconds",
            )
            possible_starts = [
                start_sample
                for start_sample in range(
                    window_start,
                    window_stop - duration_samples + 1,
                )
                if all(
                    not _overlaps_any(
                        (start_sample, start_sample + duration_samples),
                        box_bounds_by_key[(series_number, channel_number)],
                    )
                    for channel_number in channels
                )
            ]
            if not possible_starts:
                raise ValueError(
                    "could not place requested non-overlapping random box artifacts"
                )
            start_sample = int(rng.choice(possible_starts))
            stop_sample = start_sample + duration_samples
            interval = _PlannedArtifact(
                artifact_type="random_box",
                name=config.name,
                source_index=source_index,
                start_sample=start_sample,
                stop_sample=stop_sample,
                requested_start_seconds=start_sample / fs,
                requested_duration_seconds=duration_seconds,
                magnitude_fraction=float(config.magnitude_fraction),
            )
            for channel_number in channels:
                key = (series_number, channel_number)
                intervals_by_key[key].append(interval)
                box_bounds_by_key[key].append((start_sample, stop_sample))


def _expand_scheduled_durations(
    durations: float | tuple[float, ...],
    count: int,
) -> tuple[float, ...]:
    if isinstance(durations, Real) and not isinstance(durations, bool):
        return (float(durations),) * count
    values = tuple(float(value) for value in durations)
    if len(values) != count:
        raise ValueError(
            "durations_seconds must be a scalar or match start_times_seconds"
        )
    return values


def _validate_random_density(
    config: SyntheticRandomBoxArtifactConfig,
) -> tuple[int | None, float | None]:
    provided = (config.count_per_series is not None) + (
        config.rate_per_minute is not None
    )
    if provided != 1:
        raise ValueError(
            "exactly one of count_per_series or rate_per_minute must be provided"
        )
    if config.count_per_series is not None:
        if (
            not isinstance(config.count_per_series, Integral)
            or isinstance(config.count_per_series, bool)
            or config.count_per_series < 0
        ):
            raise ValueError("count_per_series must be a nonnegative integer")
        return int(config.count_per_series), None

    assert config.rate_per_minute is not None
    _require_nonnegative(config.rate_per_minute, "rate_per_minute")
    return None, float(config.rate_per_minute)


def _random_window_samples(
    window: tuple[float, float] | None,
    samples_per_series: int,
    session_duration_seconds: float,
    fs: float,
) -> tuple[int, int]:
    if window is None:
        return 0, samples_per_series
    if len(window) != 2:
        raise ValueError("start_window_seconds must contain start and stop times")
    start_seconds, stop_seconds = (float(value) for value in window)
    _require_nonnegative(start_seconds, "start_window_seconds")
    _require_nonnegative(stop_seconds, "start_window_seconds")
    if stop_seconds <= start_seconds:
        raise ValueError("start_window_seconds stop must be greater than start")
    if stop_seconds > session_duration_seconds:
        raise ValueError("start_window_seconds must fall within the session")
    start_sample = _nonnegative_seconds_to_samples(
        start_seconds,
        fs,
        "start_window_seconds",
    )
    stop_sample = _nonnegative_seconds_to_samples(
        stop_seconds,
        fs,
        "start_window_seconds",
    )
    if stop_sample <= start_sample or stop_sample > samples_per_series:
        raise ValueError("start_window_seconds must contain at least one sample")
    return start_sample, stop_sample


def _validate_duration_range(
    duration_range: tuple[float, float],
    fs: float,
) -> tuple[int, int]:
    if len(duration_range) != 2:
        raise ValueError("duration_range_seconds must contain minimum and maximum")
    minimum_seconds, maximum_seconds = (float(value) for value in duration_range)
    if maximum_seconds < minimum_seconds:
        raise ValueError("duration_range_seconds maximum must be at least minimum")
    minimum_samples = _positive_seconds_to_samples(
        minimum_seconds,
        fs,
        "duration_range_seconds",
    )
    maximum_samples = _positive_seconds_to_samples(
        maximum_seconds,
        fs,
        "duration_range_seconds",
    )
    return minimum_samples, maximum_samples


def _draw_duration_seconds(
    rng: np.random.Generator,
    duration_range: tuple[float, float],
) -> float:
    minimum_seconds, maximum_seconds = duration_range
    if minimum_seconds == maximum_seconds:
        return float(minimum_seconds)
    return float(rng.uniform(minimum_seconds, maximum_seconds))


def _resolve_selectors(
    values: Sequence[int] | None,
    maximum: int,
    name: str,
) -> tuple[int, ...]:
    if values is None:
        return tuple(range(1, maximum + 1))
    resolved: list[int] = []
    for value in values:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"{name} values must be integers")
        resolved.append(int(value))
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    if any(value < 1 or value > maximum for value in resolved):
        upper_name = "channel_count" if name == "channels" else "series_count"
        raise ValueError(f"{name} must contain values between 1 and {upper_name}")
    return tuple(resolved)


def _overlaps_any(
    candidate: tuple[int, int],
    existing: list[tuple[int, int]],
) -> bool:
    start, stop = candidate
    return any(
        start < other_stop and other_start < stop
        for other_start, other_stop in existing
    )


def _positive_seconds_to_samples(value: float, fs: float, name: str) -> int:
    _require_positive(value, name)
    samples = int(round(float(value) * fs))
    if samples < 1:
        raise ValueError(f"{name} must round to at least 1 sample")
    return samples


def _nonnegative_seconds_to_samples(value: float, fs: float, name: str) -> int:
    _require_nonnegative(value, name)
    return int(round(float(value) * fs))


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    _require_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(value: float, name: str) -> None:
    _require_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
