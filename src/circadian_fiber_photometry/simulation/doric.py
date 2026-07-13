"""Synthetic Doric HDF5 file generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .artifact.engine import (
    ArtifactSchedule,
    apply_artifact_schedule,
    build_artifact_schedule,
)
from .artifact.models import (
    SyntheticArtifactOccurrence,
    SyntheticRandomBoxArtifactConfig,
    SyntheticScheduledBoxArtifactConfig,
    SyntheticSessionStartSpikeConfig,
)
from .photobleaching import (
    SyntheticPhotobleachingConfig,
    SyntheticPhotobleachingMetadata,
    _PhotobleachingFactors,
    build_photobleaching_factors,
    resolve_photobleaching,
)


@dataclass(frozen=True)
class SyntheticTonicComponentConfig:
    """Additive low-frequency calcium component for synthetic traces."""

    amplitude: float
    frequency_hz: float
    phase_radians: float = 0.0
    offset: float = 0.0
    channels: tuple[int, ...] | None = None
    series_numbers: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SyntheticScheduledCalciumEventConfig:
    """Explicit calcium-event starts in seconds relative to each series."""

    event_times_seconds: tuple[float, ...]
    amplitude: float = 0.030
    rise_rate_per_second: float = 9.0
    fall_rate_per_second: float = 1.0
    channels: tuple[int, ...] | None = None
    series_numbers: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SyntheticRandomCalciumEventConfig:
    """Seeded random calcium-event source for synthetic traces."""

    rate_per_minute: float = 1.0
    amplitude: float = 0.030
    rise_rate_per_second: float = 9.0
    fall_rate_per_second: float = 1.0
    start_window_seconds: tuple[float, float] | None = None
    channels: tuple[int, ...] | None = None
    series_numbers: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SyntheticGaussianNoiseConfig:
    """Additional independent Gaussian noise for synthetic signal streams."""

    isosbestic_std: float = 0.0
    calcium_std: float = 0.0
    analog_in_std: float = 0.0
    name: str | None = None


@dataclass(frozen=True)
class SyntheticSignalConfig:
    """Signal-shape parameters for generated photometry traces."""

    isosbestic_baseline: float = 0.08
    calcium_baseline: float = 0.18
    channel_baseline_step: float = 0.006
    bleaching_fraction: float | None = None
    artifact_amplitude: float = 0.004
    circadian_amplitude: float = 0.012
    noise_std: float = 0.0015
    analog_noise_std: float = 0.05
    transient_rate_per_minute: float = 1.0
    transient_amplitude: float = 0.030
    transient_rise_seconds: float = 0.4
    transient_decay_seconds: float = 3.0
    tonic_components: tuple[SyntheticTonicComponentConfig, ...] = ()
    scheduled_calcium_events: tuple[SyntheticScheduledCalciumEventConfig, ...] = ()
    random_calcium_events: tuple[SyntheticRandomCalciumEventConfig, ...] = ()
    gaussian_noise: tuple[SyntheticGaussianNoiseConfig, ...] = ()
    session_start_spike: SyntheticSessionStartSpikeConfig | None = None
    scheduled_box_artifacts: tuple[SyntheticScheduledBoxArtifactConfig, ...] = ()
    random_box_artifacts: tuple[SyntheticRandomBoxArtifactConfig, ...] = ()
    photobleaching: SyntheticPhotobleachingConfig = field(
        default_factory=SyntheticPhotobleachingConfig
    )


def add_tonic_component(
    signal: SyntheticSignalConfig,
    *,
    amplitude: float,
    frequency_hz: float,
    phase_radians: float = 0.0,
    offset: float = 0.0,
    channels: Sequence[int] | np.ndarray | None = None,
    series_numbers: Sequence[int] | np.ndarray | None = None,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` with an additive tonic calcium component appended."""

    component = SyntheticTonicComponentConfig(
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase_radians=phase_radians,
        offset=offset,
        channels=_normalize_optional_int_tuple(channels),
        series_numbers=_normalize_optional_int_tuple(series_numbers),
        name=name,
    )
    return replace(
        signal,
        tonic_components=signal.tonic_components + (component,),
    )


def add_scheduled_calcium_events(
    signal: SyntheticSignalConfig,
    event_times_seconds: Sequence[float] | np.ndarray,
    *,
    amplitude: float = 0.030,
    rise_rate_per_second: float = 9.0,
    fall_rate_per_second: float = 1.0,
    channels: Sequence[int] | np.ndarray | None = None,
    series_numbers: Sequence[int] | np.ndarray | None = None,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` with explicit calcium-event starts appended."""

    event_source = SyntheticScheduledCalciumEventConfig(
        event_times_seconds=_normalize_float_tuple(event_times_seconds),
        amplitude=amplitude,
        rise_rate_per_second=rise_rate_per_second,
        fall_rate_per_second=fall_rate_per_second,
        channels=_normalize_optional_int_tuple(channels),
        series_numbers=_normalize_optional_int_tuple(series_numbers),
        name=name,
    )
    return replace(
        signal,
        scheduled_calcium_events=signal.scheduled_calcium_events + (event_source,),
    )


def add_random_calcium_events(
    signal: SyntheticSignalConfig,
    *,
    rate_per_minute: float = 1.0,
    amplitude: float = 0.030,
    rise_rate_per_second: float = 9.0,
    fall_rate_per_second: float = 1.0,
    start_window_seconds: tuple[float, float] | None = None,
    channels: Sequence[int] | np.ndarray | None = None,
    series_numbers: Sequence[int] | np.ndarray | None = None,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` with a seeded random calcium-event source appended."""

    event_source = SyntheticRandomCalciumEventConfig(
        rate_per_minute=rate_per_minute,
        amplitude=amplitude,
        rise_rate_per_second=rise_rate_per_second,
        fall_rate_per_second=fall_rate_per_second,
        start_window_seconds=start_window_seconds,
        channels=_normalize_optional_int_tuple(channels),
        series_numbers=_normalize_optional_int_tuple(series_numbers),
        name=name,
    )
    return replace(
        signal,
        random_calcium_events=signal.random_calcium_events + (event_source,),
    )


def add_gaussian_noise(
    signal: SyntheticSignalConfig,
    *,
    isosbestic_std: float = 0.0,
    calcium_std: float = 0.0,
    analog_in_std: float = 0.0,
    name: str | None = None,
) -> SyntheticSignalConfig:
    """Return ``signal`` with an additional Gaussian noise source appended."""

    noise = SyntheticGaussianNoiseConfig(
        isosbestic_std=isosbestic_std,
        calcium_std=calcium_std,
        analog_in_std=analog_in_std,
        name=name,
    )
    return replace(signal, gaussian_noise=signal.gaussian_noise + (noise,))


@dataclass(frozen=True)
class SyntheticTTLBehaviorCodeConfig:
    """Behavior event code encoded by a DigitalIO pulse count."""

    name: str
    channel: int
    pulse_count: int
    enabled: bool = True


@dataclass(frozen=True)
class SyntheticTTLBehaviorEventConfig:
    """Explicit behavior-code sequence starts for synthetic DigitalIO."""

    code_name: str
    start_seconds: tuple[float, ...]
    series_numbers: tuple[int, ...] | None = None
    enabled: bool = True


@dataclass(frozen=True)
class SyntheticTTLRandomBehaviorEventConfig:
    """Seeded random behavior-code sequence generation."""

    code_name: str
    event_count_per_series: int
    start_window_seconds: tuple[float, float] | None = None
    series_numbers: tuple[int, ...] | None = None
    enabled: bool = True


@dataclass(frozen=True)
class SyntheticTTLBehaviorEventSummary:
    """Ground-truth summary for one behavior-code TTL sequence."""

    code_name: str
    channel: int
    pulse_count: int
    series_number: int
    sequence_start_sample: int
    sequence_start_seconds: float
    pulse_sample_indices: np.ndarray
    pulse_times_seconds: np.ndarray


@dataclass(frozen=True)
class SyntheticDoricConfig:
    """Configuration for generating a synthetic Doric HDF5 file."""

    series_count: int
    session_duration_seconds: float
    inter_series_gap_seconds: float
    fs: float
    channel_count: int
    seed: int
    configured_series_count: int | None = None
    decimation_factor: int = 200
    software_version: str = "6.2.4.0"
    created: str | None = None
    filename_metadata: str | None = None
    signal: SyntheticSignalConfig = field(default_factory=SyntheticSignalConfig)
    ttl_behavior_codes: tuple[SyntheticTTLBehaviorCodeConfig, ...] = ()
    ttl_behavior_events: tuple[SyntheticTTLBehaviorEventConfig, ...] = ()
    ttl_random_behavior_events: tuple[SyntheticTTLRandomBehaviorEventConfig, ...] = ()
    ttl_pulse_width_seconds: float = 0.050
    ttl_pulse_off_interval_seconds: float = 0.050


@dataclass(frozen=True)
class SyntheticDoricSummary:
    """Ground-truth summary returned after writing a synthetic Doric file.

    Event dictionary keys are one-based ``(series_number, channel_number)``
    pairs, matching the numbering in Doric path names.
    """

    path: Path
    series_count: int
    configured_series_count: int
    channel_count: int
    fs: float
    samples_per_series: int
    session_start_times: np.ndarray
    event_sample_indices: dict[tuple[int, int], np.ndarray]
    event_times_seconds: dict[tuple[int, int], np.ndarray]
    ttl_pulse_sample_indices: dict[tuple[int, int], np.ndarray]
    ttl_pulse_times_seconds: dict[tuple[int, int], np.ndarray]
    ttl_behavior_events: tuple[SyntheticTTLBehaviorEventSummary, ...]
    photobleaching: SyntheticPhotobleachingMetadata
    artifact_occurrences: tuple[SyntheticArtifactOccurrence, ...] = ()


def generate_synthetic_doric(
    path: str | Path,
    config: SyntheticDoricConfig,
    *,
    overwrite: bool = False,
) -> SyntheticDoricSummary:
    """Write a synthetic Doric-style HDF5 file.

    The generated layout mirrors Doric ``FPConsole`` files closely enough for
    HDF5-based readers and MATLAB-style tests. It does not attempt to guarantee
    import compatibility with Doric Neuroscience Studio.
    """

    validated = _ValidatedConfig.from_config(config)
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it"
        )
    if output_path.parent and not output_path.parent.exists():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )

    rng = np.random.default_rng(config.seed)
    session_start_times = np.array(
        [
            index
            * (config.session_duration_seconds + config.inter_series_gap_seconds)
            for index in range(config.series_count)
        ],
        dtype=float,
    )
    photobleaching_factors = build_photobleaching_factors(
        validated.photobleaching,
        session_start_times_seconds=session_start_times,
        samples_per_series=validated.samples_per_series,
        sampling_rate_hz=config.fs,
    )
    ttl_schedule = _build_ttl_schedule(config, validated, session_start_times)
    artifact_schedule = build_artifact_schedule(
        session_start_spike=config.signal.session_start_spike,
        scheduled_box_artifacts=config.signal.scheduled_box_artifacts,
        random_box_artifacts=config.signal.random_box_artifacts,
        series_count=config.series_count,
        channel_count=config.channel_count,
        samples_per_series=validated.samples_per_series,
        session_duration_seconds=config.session_duration_seconds,
        fs=config.fs,
        seed=config.seed,
    )
    event_sample_indices: dict[tuple[int, int], np.ndarray] = {}
    event_times_seconds: dict[tuple[int, int], np.ndarray] = {}
    artifact_occurrences: list[SyntheticArtifactOccurrence] = []

    with h5py.File(output_path, "w", track_order=True) as h5_file:
        _write_root_attrs(h5_file, config)
        _write_configuration_groups(h5_file, output_path, config, validated)

        signals = _require_group(h5_file, "DataAcquisition")
        fpconsole = _create_group(signals, "FPConsole")
        signals_group = _create_group(fpconsole, "Signals")

        for series_index, series_start in enumerate(session_start_times):
            series_name = f"Series{series_index + 1:04d}"
            series_group = _create_group(signals_group, series_name)
            time = series_start + np.arange(validated.samples_per_series) / config.fs
            series_signals: dict[str, np.ndarray] = {}

            for channel_index in range(config.channel_count):
                generated = _generate_channel_signals(
                    rng,
                    config,
                    validated,
                    time,
                    series_index,
                    channel_index,
                    artifact_schedule,
                    photobleaching_factors,
                )
                key = (series_index + 1, channel_index + 1)
                event_sample_indices[key] = generated.event_indices
                event_times_seconds[key] = (
                    series_start + generated.event_indices / config.fs
                )
                artifact_occurrences.extend(generated.artifact_occurrences)

                ain_name = f"AIN{channel_index + 1:02d}"
                for output_number, values in (
                    (1, generated.isosbestic_405),
                    (2, generated.calcium_465),
                ):
                    lockin_name = f"{ain_name}xAOUT{output_number:02d}-LockIn"
                    lockin_group = _create_group(series_group, lockin_name)
                    _create_signal_dataset(lockin_group, "Time", time)
                    _create_signal_dataset(
                        lockin_group,
                        "Values",
                        values,
                        _signal_attrs(lockin_name, -10.0, 10.0, "Voltage (V)"),
                    )
                series_signals[ain_name] = generated.analog_in

            analog_in = _create_group(series_group, "AnalogIn")
            _create_signal_dataset(analog_in, "Time", time)
            for channel_index in range(config.channel_count):
                ain_name = f"AIN{channel_index + 1:02d}"
                _create_signal_dataset(
                    analog_in,
                    ain_name,
                    series_signals[ain_name],
                    _signal_attrs(ain_name, -10.0, 10.0, "Voltage (V)"),
                )

            analog_out = _create_group(series_group, "AnalogOut")
            _create_signal_dataset(analog_out, "Time", time)
            for output_number in (1, 2):
                aout_name = f"AOUT{output_number:02d}"
                _create_signal_dataset(
                    analog_out,
                    aout_name,
                    np.ones(validated.samples_per_series, dtype=float),
                    _signal_attrs(aout_name, -5.5, 5.5, "Voltage (V)"),
                )

            digital_io = _create_group(series_group, "DigitalIO")
            _create_signal_dataset(digital_io, "Time", time)
            for dio_number in (1, 2):
                dio_name = f"DIO{dio_number:02d}"
                _create_signal_dataset(
                    digital_io,
                    dio_name,
                    _ttl_values_for_key(
                        ttl_schedule,
                        series_index + 1,
                        dio_number,
                        validated.samples_per_series,
                    ),
                    _signal_attrs(dio_name, -0.1, 1.1, "ON/OFF"),
                )

    return SyntheticDoricSummary(
        path=output_path,
        series_count=config.series_count,
        configured_series_count=validated.configured_series_count,
        channel_count=config.channel_count,
        fs=float(config.fs),
        samples_per_series=validated.samples_per_series,
        session_start_times=session_start_times,
        event_sample_indices=event_sample_indices,
        event_times_seconds=event_times_seconds,
        ttl_pulse_sample_indices=ttl_schedule.pulse_sample_indices,
        ttl_pulse_times_seconds=ttl_schedule.pulse_times_seconds,
        ttl_behavior_events=ttl_schedule.behavior_events,
        photobleaching=validated.photobleaching,
        artifact_occurrences=tuple(
            sorted(
                artifact_occurrences,
                key=lambda occurrence: (
                    occurrence.series_number,
                    occurrence.channel_number,
                    occurrence.start_sample,
                    occurrence.artifact_type,
                    occurrence.source_index,
                ),
            )
        ),
    )


@dataclass(frozen=True)
class _GeneratedChannel:
    isosbestic_405: np.ndarray
    calcium_465: np.ndarray
    analog_in: np.ndarray
    event_indices: np.ndarray
    artifact_occurrences: tuple[SyntheticArtifactOccurrence, ...]


@dataclass(frozen=True)
class _TTLSchedule:
    intervals_by_key: dict[tuple[int, int], tuple[tuple[int, int], ...]]
    pulse_sample_indices: dict[tuple[int, int], np.ndarray]
    pulse_times_seconds: dict[tuple[int, int], np.ndarray]
    behavior_events: tuple[SyntheticTTLBehaviorEventSummary, ...]


@dataclass(frozen=True)
class _ValidatedConfig:
    configured_series_count: int
    samples_per_series: int
    crop_samples: int
    photobleaching: SyntheticPhotobleachingMetadata

    @classmethod
    def from_config(cls, config: SyntheticDoricConfig) -> _ValidatedConfig:
        _require_positive_int(config.series_count, "series_count")
        _require_positive_int(config.channel_count, "channel_count")
        _require_positive_number(config.session_duration_seconds, "session_duration")
        _require_nonnegative_number(
            config.inter_series_gap_seconds,
            "inter_series_gap_seconds",
        )
        _require_positive_number(config.fs, "fs")
        _require_positive_int(config.decimation_factor, "decimation_factor")
        _require_positive_number(
            config.ttl_pulse_width_seconds,
            "ttl_pulse_width_seconds",
        )
        _require_nonnegative_number(
            config.ttl_pulse_off_interval_seconds,
            "ttl_pulse_off_interval_seconds",
        )

        configured_series_count = (
            config.series_count
            if config.configured_series_count is None
            else config.configured_series_count
        )
        _require_positive_int(configured_series_count, "configured_series_count")
        if configured_series_count < config.series_count:
            raise ValueError(
                "configured_series_count must be greater than or equal to series_count"
            )

        samples_per_series = int(round(config.fs * config.session_duration_seconds))
        if samples_per_series < 2:
            raise ValueError(
                "session_duration_seconds and fs produce fewer than 2 samples"
            )
        crop_samples = int(round(5 * config.fs))
        if samples_per_series - 2 * crop_samples < 2:
            raise ValueError(
                "session_duration_seconds is too short for MATLAB's first/last "
                "5 second crop"
            )

        _validate_signal_config(config.signal, config)
        photobleaching = resolve_photobleaching(
            config.signal.photobleaching,
            active_duration_seconds=(
                config.series_count * samples_per_series / config.fs
            ),
            legacy_bleaching_fraction=config.signal.bleaching_fraction,
        )
        return cls(
            configured_series_count=configured_series_count,
            samples_per_series=samples_per_series,
            crop_samples=crop_samples,
            photobleaching=photobleaching,
        )


def _build_ttl_schedule(
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
    session_start_times: np.ndarray,
) -> _TTLSchedule:
    intervals_by_key: dict[tuple[int, int], list[tuple[int, int]]] = {
        (series_number, dio_channel): []
        for series_number in range(1, config.series_count + 1)
        for dio_channel in (1, 2)
    }
    behavior_events: list[SyntheticTTLBehaviorEventSummary] = []
    all_codes = _validate_ttl_behavior_codes(config)
    width_samples = _positive_seconds_to_samples(
        config.ttl_pulse_width_seconds,
        config.fs,
        "ttl_pulse_width_seconds",
    )
    off_samples = _nonnegative_seconds_to_samples(
        config.ttl_pulse_off_interval_seconds,
        config.fs,
        "ttl_pulse_off_interval_seconds",
    )

    for event_config in config.ttl_behavior_events:
        if not event_config.enabled:
            continue
        code = _ttl_code_for_schedule(event_config.code_name, all_codes)
        if code is None:
            continue
        _validate_ttl_behavior_event(event_config, config)

        for series_number in _resolve_ttl_series_numbers(
            event_config.series_numbers,
            config.series_count,
        ):
            for sequence_start_seconds in event_config.start_seconds:
                sequence_start_sample = _nonnegative_seconds_to_samples(
                    sequence_start_seconds,
                    config.fs,
                    "start_seconds",
                )
                _add_ttl_behavior_sequence(
                    code,
                    series_number,
                    sequence_start_sample,
                    config,
                    validated,
                    session_start_times,
                    width_samples,
                    off_samples,
                    intervals_by_key,
                    behavior_events,
                )

    ttl_rng = np.random.default_rng(
        np.random.SeedSequence([int(config.seed), 0x54544C])
    )
    for random_config in config.ttl_random_behavior_events:
        if not random_config.enabled:
            continue
        code = _ttl_code_for_schedule(random_config.code_name, all_codes)
        if code is None:
            continue
        _validate_ttl_random_behavior_event(random_config, config)

        sequence_duration = _ttl_sequence_duration_samples(
            code.pulse_count,
            width_samples,
            off_samples,
        )
        window_start_sample, window_stop_sample = _random_window_samples(
            random_config.start_window_seconds,
            config,
            validated,
        )

        for series_number in _resolve_ttl_series_numbers(
            random_config.series_numbers,
            config.series_count,
        ):
            _add_random_ttl_behavior_sequences(
                ttl_rng,
                random_config,
                code,
                series_number,
                window_start_sample,
                window_stop_sample,
                sequence_duration,
                config,
                validated,
                session_start_times,
                width_samples,
                off_samples,
                intervals_by_key,
                behavior_events,
            )

    return _freeze_ttl_schedule(
        config,
        session_start_times,
        intervals_by_key,
        behavior_events,
    )


def _add_ttl_behavior_sequence(
    code: SyntheticTTLBehaviorCodeConfig,
    series_number: int,
    sequence_start_sample: int,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
    session_start_times: np.ndarray,
    width_samples: int,
    off_samples: int,
    intervals_by_key: dict[tuple[int, int], list[tuple[int, int]]],
    behavior_events: list[SyntheticTTLBehaviorEventSummary],
) -> None:
    sequence_duration = _ttl_sequence_duration_samples(
        code.pulse_count,
        width_samples,
        off_samples,
    )
    if (
        sequence_start_sample < 0
        or sequence_start_sample + sequence_duration > validated.samples_per_series
    ):
        raise ValueError(
            "TTL behavior sequence starts outside the session or extends beyond it"
        )

    pulse_starts = np.array(
        [
            sequence_start_sample
            + pulse_index * (width_samples + off_samples)
            for pulse_index in range(code.pulse_count)
        ],
        dtype=int,
    )
    intervals = tuple(
        (int(start_sample), int(start_sample + width_samples))
        for start_sample in pulse_starts
    )
    key = (series_number, code.channel)
    if _ttl_sequence_overlaps(intervals, intervals_by_key[key]):
        raise ValueError(
            f"TTL behavior sequences overlap for series {series_number} "
            f"DIO{code.channel:02d}"
        )

    intervals_by_key[key].extend(intervals)
    sequence_start_seconds = (
        session_start_times[series_number - 1] + sequence_start_sample / config.fs
    )
    pulse_times_seconds = session_start_times[series_number - 1] + (
        pulse_starts / config.fs
    )
    behavior_events.append(
        SyntheticTTLBehaviorEventSummary(
            code_name=code.name,
            channel=code.channel,
            pulse_count=code.pulse_count,
            series_number=series_number,
            sequence_start_sample=sequence_start_sample,
            sequence_start_seconds=float(sequence_start_seconds),
            pulse_sample_indices=pulse_starts,
            pulse_times_seconds=pulse_times_seconds,
        )
    )


def _add_random_ttl_behavior_sequences(
    rng: np.random.Generator,
    random_config: SyntheticTTLRandomBehaviorEventConfig,
    code: SyntheticTTLBehaviorCodeConfig,
    series_number: int,
    window_start_sample: int,
    window_stop_sample: int,
    sequence_duration: int,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
    session_start_times: np.ndarray,
    width_samples: int,
    off_samples: int,
    intervals_by_key: dict[tuple[int, int], list[tuple[int, int]]],
    behavior_events: list[SyntheticTTLBehaviorEventSummary],
) -> None:
    if random_config.event_count_per_series == 0:
        return

    latest_start = window_stop_sample - sequence_duration
    if latest_start < window_start_sample:
        raise ValueError("TTL random behavior event window is too short")

    accepted = 0
    candidates = rng.permutation(np.arange(window_start_sample, latest_start + 1))
    for candidate in candidates:
        intervals = _ttl_sequence_intervals(
            int(candidate),
            code.pulse_count,
            width_samples,
            off_samples,
        )
        if _ttl_sequence_overlaps(
            intervals,
            intervals_by_key[(series_number, code.channel)],
        ):
            continue
        _add_ttl_behavior_sequence(
            code,
            series_number,
            int(candidate),
            config,
            validated,
            session_start_times,
            width_samples,
            off_samples,
            intervals_by_key,
            behavior_events,
        )
        accepted += 1
        if accepted >= random_config.event_count_per_series:
            return

    raise ValueError(
        "could not place requested non-overlapping TTL random behavior events"
    )


def _freeze_ttl_schedule(
    config: SyntheticDoricConfig,
    session_start_times: np.ndarray,
    intervals_by_key: dict[tuple[int, int], list[tuple[int, int]]],
    behavior_events: list[SyntheticTTLBehaviorEventSummary],
) -> _TTLSchedule:
    frozen_intervals: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    pulse_sample_indices: dict[tuple[int, int], np.ndarray] = {}
    pulse_times_seconds: dict[tuple[int, int], np.ndarray] = {}

    for key, intervals in intervals_by_key.items():
        series_number, dio_channel = key
        sorted_intervals = tuple(sorted(intervals))
        previous_stop = -1
        for start_sample, stop_sample in sorted_intervals:
            if start_sample < previous_stop:
                raise ValueError(
                    "TTL behavior sequences overlap for "
                    f"series {series_number} DIO{dio_channel:02d}"
                )
            previous_stop = stop_sample

        starts = np.array(
            [start_sample for start_sample, _ in sorted_intervals],
            dtype=int,
        )
        frozen_intervals[key] = sorted_intervals
        pulse_sample_indices[key] = starts
        pulse_times_seconds[key] = session_start_times[series_number - 1] + (
            starts / config.fs
        )

    sorted_behavior_events = tuple(
        sorted(
            behavior_events,
            key=lambda event: (
                event.series_number,
                event.channel,
                event.sequence_start_sample,
                event.code_name,
            ),
        )
    )
    return _TTLSchedule(
        intervals_by_key=frozen_intervals,
        pulse_sample_indices=pulse_sample_indices,
        pulse_times_seconds=pulse_times_seconds,
        behavior_events=sorted_behavior_events,
    )


def _ttl_values_for_key(
    schedule: _TTLSchedule,
    series_number: int,
    dio_channel: int,
    samples: int,
) -> np.ndarray:
    values = np.zeros(samples, dtype=float)
    for start_sample, stop_sample in schedule.intervals_by_key[
        (series_number, dio_channel)
    ]:
        values[start_sample:stop_sample] = 1.0
    return values


def _generate_channel_signals(
    rng: np.random.Generator,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
    absolute_time: np.ndarray,
    series_index: int,
    channel_index: int,
    artifact_schedule: ArtifactSchedule,
    photobleaching_factors: _PhotobleachingFactors,
) -> _GeneratedChannel:
    signal = config.signal
    samples = validated.samples_per_series
    relative_time = np.arange(samples, dtype=float) / config.fs
    isosbestic_bleaching = photobleaching_factors.isosbestic[series_index]
    calcium_bleaching = photobleaching_factors.calcium[series_index]
    channel_phase = 0.7 * channel_index
    session_phase = 2 * np.pi * absolute_time[0] / 86400 + channel_phase

    base_405 = signal.isosbestic_baseline + (
        signal.channel_baseline_step * channel_index
    )
    base_465 = signal.calcium_baseline + (
        signal.channel_baseline_step * channel_index * 1.6
    )
    artifact = signal.artifact_amplitude * np.sin(
        2 * np.pi * 0.07 * relative_time + session_phase
    )
    drift = signal.artifact_amplitude * 0.5 * np.sin(
        2 * np.pi * 0.011 * relative_time + channel_phase
    )
    noise_405 = rng.normal(0.0, signal.noise_std, samples)
    bleached_base_405 = base_405 * isosbestic_bleaching
    isosbestic = bleached_base_405 + artifact + drift
    isosbestic = isosbestic + noise_405 + _gaussian_noise_trace(
        rng,
        signal.gaussian_noise,
        "isosbestic_std",
        samples,
    )

    circadian = signal.circadian_amplitude * np.sin(session_phase)
    tonic_signal = _tonic_component_trace(
        signal,
        absolute_time,
        series_index,
        channel_index,
    )
    legacy_event_indices = _draw_event_indices(rng, config, validated)
    legacy_transient_signal = _transient_trace(
        legacy_event_indices,
        samples,
        config.fs,
        signal.transient_amplitude * (1 + 0.15 * channel_index),
        signal.transient_rise_seconds,
        signal.transient_decay_seconds,
    )
    configured_event_indices, configured_transient_signal = (
        _configured_calcium_event_signal(
            rng,
            config,
            validated,
            series_index,
            channel_index,
        )
    )
    event_indices = _merge_event_indices(
        legacy_event_indices,
        configured_event_indices,
    )
    transient_signal = legacy_transient_signal + configured_transient_signal
    noise_465 = rng.normal(0.0, signal.noise_std, samples) + _gaussian_noise_trace(
        rng,
        signal.gaussian_noise,
        "calcium_std",
        samples,
    )
    calcium = (
        (base_465 + circadian + tonic_signal) * calcium_bleaching
        + 1.25 * (isosbestic - bleached_base_405)
        + transient_signal
        + noise_465
    )

    analog_in = (
        0.75
        + 2.5 * isosbestic
        + 1.5 * transient_signal
        + rng.normal(0.0, signal.analog_noise_std, samples)
        + _gaussian_noise_trace(
            rng,
            signal.gaussian_noise,
            "analog_in_std",
            samples,
        )
    )

    isosbestic, calcium, artifact_occurrences = apply_artifact_schedule(
        artifact_schedule,
        series_number=series_index + 1,
        channel_number=channel_index + 1,
        series_start_seconds=float(absolute_time[0]),
        fs=config.fs,
        isosbestic=isosbestic,
        calcium=calcium,
    )

    return _GeneratedChannel(
        isosbestic_405=isosbestic.astype(np.float64),
        calcium_465=calcium.astype(np.float64),
        analog_in=analog_in.astype(np.float64),
        event_indices=event_indices.astype(int),
        artifact_occurrences=artifact_occurrences,
    )


def _gaussian_noise_trace(
    rng: np.random.Generator,
    noise_configs: tuple[SyntheticGaussianNoiseConfig, ...],
    std_attr: str,
    samples: int,
) -> np.ndarray:
    trace = np.zeros(samples, dtype=float)
    for noise_config in noise_configs:
        std = getattr(noise_config, std_attr)
        if std > 0:
            trace += rng.normal(0.0, std, samples)
    return trace


def _tonic_component_trace(
    signal: SyntheticSignalConfig,
    absolute_time: np.ndarray,
    series_index: int,
    channel_index: int,
) -> np.ndarray:
    trace = np.zeros_like(absolute_time, dtype=float)
    for component in signal.tonic_components:
        if not _component_applies(
            component.channels,
            component.series_numbers,
            series_index,
            channel_index,
        ):
            continue
        trace += component.offset + component.amplitude * np.sin(
            2 * np.pi * component.frequency_hz * absolute_time
            + component.phase_radians
        )
    return trace


def _configured_calcium_event_signal(
    rng: np.random.Generator,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
    series_index: int,
    channel_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = validated.samples_per_series
    trace = np.zeros(samples, dtype=float)
    event_indices_by_source: list[np.ndarray] = []

    for event_config in config.signal.scheduled_calcium_events:
        if not _component_applies(
            event_config.channels,
            event_config.series_numbers,
            series_index,
            channel_index,
        ):
            continue
        event_indices = _event_times_to_indices(
            event_config.event_times_seconds,
            config,
        )
        event_indices_by_source.append(event_indices)
        trace += _calcium_event_trace(
            event_indices,
            samples,
            config.fs,
            event_config.amplitude,
            event_config.rise_rate_per_second,
            event_config.fall_rate_per_second,
        )

    for event_config in config.signal.random_calcium_events:
        if not _component_applies(
            event_config.channels,
            event_config.series_numbers,
            series_index,
            channel_index,
        ):
            continue
        event_indices = _draw_random_calcium_event_indices(
            rng,
            event_config,
            config,
            validated,
        )
        event_indices_by_source.append(event_indices)
        trace += _calcium_event_trace(
            event_indices,
            samples,
            config.fs,
            event_config.amplitude,
            event_config.rise_rate_per_second,
            event_config.fall_rate_per_second,
        )

    return _merge_event_indices(*event_indices_by_source), trace


def _event_times_to_indices(
    event_times_seconds: tuple[float, ...],
    config: SyntheticDoricConfig,
) -> np.ndarray:
    if len(event_times_seconds) == 0:
        return np.array([], dtype=int)
    return np.array(
        [
            _nonnegative_seconds_to_samples(
                event_time_seconds,
                config.fs,
                "event_times_seconds",
            )
            for event_time_seconds in event_times_seconds
        ],
        dtype=int,
    )


def _draw_random_calcium_event_indices(
    rng: np.random.Generator,
    event_config: SyntheticRandomCalciumEventConfig,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
) -> np.ndarray:
    if event_config.rate_per_minute == 0:
        return np.array([], dtype=int)

    window_start_sample, window_stop_sample = _calcium_random_window_samples(
        event_config.start_window_seconds,
        config,
        validated,
    )
    available = window_stop_sample - window_start_sample
    if available <= 0:
        return np.array([], dtype=int)

    expected_events = event_config.rate_per_minute * (available / config.fs) / 60
    event_count = max(1, int(rng.poisson(expected_events)))
    event_count = min(event_count, available)
    return np.sort(
        rng.choice(
            np.arange(window_start_sample, window_stop_sample),
            size=event_count,
            replace=False,
        )
    )


def _calcium_random_window_samples(
    start_window_seconds: tuple[float, float] | None,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
) -> tuple[int, int]:
    if start_window_seconds is None:
        return (
            max(validated.crop_samples, 0),
            validated.samples_per_series - max(validated.crop_samples, 0),
        )

    start_seconds, stop_seconds = start_window_seconds
    start_sample = _nonnegative_seconds_to_samples(
        start_seconds,
        config.fs,
        "start_window_seconds",
    )
    stop_sample = _nonnegative_seconds_to_samples(
        stop_seconds,
        config.fs,
        "start_window_seconds",
    )
    return start_sample, min(stop_sample, validated.samples_per_series)


def _calcium_event_trace(
    event_indices: np.ndarray,
    samples: int,
    fs: float,
    amplitude: float,
    rise_rate_per_second: float,
    fall_rate_per_second: float,
) -> np.ndarray:
    trace = np.zeros(samples, dtype=float)
    if event_indices.size == 0:
        return trace

    for event_index in event_indices:
        tail_time = np.arange(samples - event_index, dtype=float) / fs
        kernel = (1 - np.exp(-rise_rate_per_second * tail_time)) * np.exp(
            -fall_rate_per_second * tail_time
        )
        peak = np.max(kernel)
        if peak > 0:
            kernel = kernel / peak
        trace[event_index:] += amplitude * kernel
    return trace


def _merge_event_indices(*event_indices_by_source: np.ndarray) -> np.ndarray:
    nonempty = [
        np.asarray(indices, dtype=int)
        for indices in event_indices_by_source
        if np.asarray(indices).size > 0
    ]
    if not nonempty:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(nonempty)).astype(int)


def _component_applies(
    channels: tuple[int, ...] | None,
    series_numbers: tuple[int, ...] | None,
    series_index: int,
    channel_index: int,
) -> bool:
    channel_number = channel_index + 1
    series_number = series_index + 1
    return (
        (channels is None or channel_number in channels)
        and (series_numbers is None or series_number in series_numbers)
    )


def _draw_event_indices(
    rng: np.random.Generator,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
) -> np.ndarray:
    if config.signal.transient_rate_per_minute == 0:
        return np.array([], dtype=int)

    first = max(validated.crop_samples, 0)
    stop = validated.samples_per_series - max(validated.crop_samples, 0)
    if stop <= first:
        return np.array([], dtype=int)

    expected_events = (
        config.signal.transient_rate_per_minute
        * config.session_duration_seconds
        / 60
    )
    event_count = max(1, int(rng.poisson(expected_events)))
    available = stop - first
    event_count = min(event_count, available)
    return np.sort(rng.choice(np.arange(first, stop), size=event_count, replace=False))


def _transient_trace(
    event_indices: np.ndarray,
    samples: int,
    fs: float,
    amplitude: float,
    rise_seconds: float,
    decay_seconds: float,
) -> np.ndarray:
    trace = np.zeros(samples, dtype=float)
    if event_indices.size == 0:
        return trace

    for event_index in event_indices:
        tail_time = np.arange(samples - event_index, dtype=float) / fs
        kernel = (1 - np.exp(-tail_time / rise_seconds)) * np.exp(
            -tail_time / decay_seconds
        )
        peak = np.max(kernel)
        if peak > 0:
            kernel = kernel / peak
        trace[event_index:] += amplitude * kernel
    return trace


def _write_root_attrs(h5_file: h5py.File, config: SyntheticDoricConfig) -> None:
    h5_file.attrs["Created"] = config.created or "Mon Jan 01 00:00:00 2024"
    h5_file.attrs["SoftwareName"] = "Doric Neuroscience Studio"
    h5_file.attrs["SoftwareVersion"] = config.software_version


def _write_configuration_groups(
    h5_file: h5py.File,
    path: Path,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
) -> None:
    configurations = _require_group(h5_file, "Configurations")
    fpconsole = _create_group(configurations, "FPConsole")
    fpconsole.attrs.update(
        {
            "ChannelVersion": "2.1.10",
            "CommunicatorType": 1,
            "Compatibility": 3,
            "DeviceName": "Acquisition Console",
            "DriverType": 3,
            "MotherboardVersion": "4.0.0",
            "NumberOfChannels": config.channel_count + 2,
            "PID": 62846,
            "ReleaseNumber": 5,
            "Serial": "SYNTHETIC-DORIC",
            "Status": 3,
            "UID": "SYNTHETIC-DORIC",
            "VID": 1240,
        }
    )

    for channel_number in range(1, config.channel_count + 1):
        _write_analog_input_config(fpconsole, channel_number, config)
    for output_number in (1, 2):
        _write_analog_output_config(fpconsole, output_number)
    for dio_number in (1, 2):
        _write_digital_io_config(fpconsole, dio_number, config)

    _create_group(fpconsole, "GlobalSettings").attrs.update(
        {
            "AutoscrollSize": 30.0,
            "GlobalTriggerMode": 1,
            "GlobalTriggerSource": 0,
            "SamplingFrequency": int(round(config.fs * config.decimation_factor)),
            "WindowsState": "",
            "isOptimalZoom": 0,
        }
    )
    _create_group(fpconsole, "SavingSettings").attrs.update(
        {
            "DecimationEnabled": 1,
            "DecimationFactor": config.decimation_factor,
            "FileExtension": 0,
            "FileIndex": 1,
            "Filename": config.filename_metadata or path.stem,
            "Filepath": str(path.parent),
        }
    )
    _create_group(fpconsole, "TimeseriesSettings").attrs.update(
        {
            "IntervalBetweenSeries(ms)": int(
                round(config.inter_series_gap_seconds * 1000)
            ),
            "NumberOfSeries": validated.configured_series_count,
            "TimeActive(ms)": int(round(config.session_duration_seconds * 1000)),
            "TotalDuration(ms)": int(
                round(
                    (
                        (validated.configured_series_count - 1)
                        * (
                            config.session_duration_seconds
                            + config.inter_series_gap_seconds
                        )
                        + config.session_duration_seconds
                    )
                    * 1000
                )
            ),
            "UsingTimeSeries": 1,
        }
    )


def _write_analog_input_config(
    fpconsole: h5py.Group,
    channel_number: int,
    config: SyntheticDoricConfig,
) -> None:
    ain_name = f"AIN{channel_number:02d}"
    ain_group = _create_group(fpconsole, ain_name)
    graphsettings = _create_group(ain_group, "Graphsettings")
    _create_group(graphsettings, ain_name).attrs.update(
        _graph_attrs(
            ain_name,
            "#3d8ec9",
            channel_number - 1,
            -10.0,
            10.0,
            "Voltage (V)",
        )
    )
    _create_group(graphsettings, f"{ain_name}xAOUT01-LockIn").attrs.update(
        _graph_attrs(
            f"{ain_name}xAOUT01-LockIn",
            "#9b59b6",
            1,
            -10.0,
            10.0,
            "Voltage (V)",
        )
    )
    _create_group(graphsettings, f"{ain_name}xAOUT02-LockIn").attrs.update(
        _graph_attrs(
            f"{ain_name}xAOUT02-LockIn",
            "#e74c3c",
            2,
            -10.0,
            10.0,
            "Voltage (V)",
        )
    )

    settings = _create_group(ain_group, "Settings")
    settings.attrs.update(
        {
            "AcquisitionMode": 2,
            "ChannelIndex": channel_number - 1,
            "CustomFile": "",
            "CustomSaturation": 5.0,
            "CutoffFrequency": 12.0,
            "FilterName": 0,
            "FilterOrder": 4,
            "FilterSampleRate": int(round(config.fs * config.decimation_factor)),
            "FilterType": 3,
            "GlobalSampleRate": int(round(config.fs * config.decimation_factor)),
            "HighCutoffFrequency": 1000.0,
            "LowCutoffFrequency": 244.0,
            "ModuleIdentifier": 3,
            "ModuleType": 0,
            "Name": ain_name,
            "PassBandRipple": 0.01,
            "RiseFallTime": 15,
            "SaturationMode": 0,
            "SignalMode": 0,
            "StopBandRipple": 30.0,
            "TriggerMode": 0,
            "TriggerSource": 0,
            "Username": ain_name,
            "isFilterEnabled": 0,
        }
    )
    for output_number, frequency in ((1, 208.616), (2, 572.205)):
        lockin_name = f"{ain_name}xAOUT{output_number:02d}-LockIn"
        _create_group(settings, f"LockInAOUT{output_number}").attrs.update(
            {
                "BandpassRange": 0.0,
                "CarrierChannel": output_number - 1,
                "MasterChannel": channel_number - 1,
                "MaximumCurrent": 1000.0,
                "Name": lockin_name,
                "OutputLevel": 100.0,
                "OwnerChannel": 0,
                "ReferenceFrequency": frequency,
                "Username": lockin_name,
                "VoltageMax": 2.5,
                "VoltageMin": 0.2,
                "isCustomMode": 0,
                "isILFMCMode": 0,
                "isOwnerLocked": 1,
            }
        )


def _write_analog_output_config(fpconsole: h5py.Group, output_number: int) -> None:
    aout_name = f"AOUT{output_number:02d}"
    frequency = 208.616 if output_number == 1 else 572.205
    aout_group = _create_group(fpconsole, aout_name)
    graphsettings = _create_group(aout_group, "Graphsettings")
    _create_group(graphsettings, aout_name).attrs.update(
        _graph_attrs(aout_name, "#3d8ec9", output_number - 1, -5.5, 5.5, "Voltage (V)")
    )
    modulations = _create_group(aout_group, "Modulations")
    _create_group(modulations, "Modulation1").attrs.update(
        {
            "DelayBetweenSequence": 0,
            "DutyCycle": 50.0,
            "FallingTime": 0,
            "Frequency": frequency,
            "Inverted": 0,
            "ModulationType": 7,
            "NumberOfPulsesPerSequence": 0,
            "NumberOfSequence": 1,
            "NumberOfSteps": 2,
            "Period": 100.0,
            "Phase": 0,
            "RisingTime": 0,
            "Smoothed": 0,
            "StartingDelay": 0,
            "StepsVoltage": np.array([0.0, 0.0], dtype=float),
            "TimeON": 50.0,
            "TotalDuration": 0,
            "UsingFrequency": 1,
            "UsingTimeON": 0,
            "VoltageMax": 2.5,
            "VoltageMin": 0.2,
        }
    )
    _create_group(aout_group, "Settings").attrs.update(
        {
            "ChannelIndex": output_number - 1,
            "CustomFile": "",
            "ModuleIdentifier": 2,
            "ModuleType": 1,
            "Name": aout_name,
            "SignalMode": 11,
            "TriggerMode": 1,
            "TriggerSource": 0,
            "Username": aout_name,
        }
    )


def _first_ttl_code_for_channel(
    config: SyntheticDoricConfig,
    dio_number: int,
) -> SyntheticTTLBehaviorCodeConfig | None:
    for code in config.ttl_behavior_codes:
        if code.enabled and code.channel == dio_number:
            return code
    return None


def _digital_io_modulation_attrs(
    code: SyntheticTTLBehaviorCodeConfig | None,
    config: SyntheticDoricConfig,
) -> dict[str, int | float | np.ndarray]:
    time_on = 50.0
    period = 100.0
    frequency = 10.0
    duty_cycle = 50.0
    if code is not None:
        time_on = config.ttl_pulse_width_seconds * 1000
        period = (
            config.ttl_pulse_width_seconds + config.ttl_pulse_off_interval_seconds
        ) * 1000
        frequency = 1000 / period
        duty_cycle = 100 * time_on / period

    return {
        "DelayBetweenSequence": 0,
        "DutyCycle": duty_cycle,
        "FallingTime": 0,
        "Frequency": frequency,
        "Inverted": 0,
        "ModulationType": 3,
        "NumberOfPulsesPerSequence": 0,
        "NumberOfSequence": 1,
        "NumberOfSteps": 2,
        "Period": period,
        "Phase": 0,
        "RisingTime": 0,
        "Smoothed": 0,
        "StartingDelay": 0,
        "StepsVoltage": np.array([0.0, 0.0], dtype=float),
        "TimeON": time_on,
        "TotalDuration": 0,
        "UsingFrequency": 1,
        "UsingTimeON": 1,
        "VoltageMax": 4.75,
        "VoltageMin": 0.0,
    }


def _write_digital_io_config(
    fpconsole: h5py.Group,
    dio_number: int,
    config: SyntheticDoricConfig,
) -> None:
    dio_name = f"DIO{dio_number:02d}"
    dio_group = _create_group(fpconsole, dio_name)
    graphsettings = _create_group(dio_group, "Graphsettings")
    _create_group(graphsettings, dio_name).attrs.update(
        _graph_attrs(dio_name, "#3d8ec9", dio_number - 1, -0.1, 1.1, "ON/OFF")
    )
    modulations = _create_group(dio_group, "Modulations")
    _create_group(modulations, "Modulation1").attrs.update(
        _digital_io_modulation_attrs(
            _first_ttl_code_for_channel(config, dio_number),
            config,
        )
    )
    _create_group(dio_group, "Settings").attrs.update(
        {
            "ChannelIndex": dio_number - 1,
            "CustomFile": "",
            "ModuleIdentifier": 0,
            "ModuleType": 2,
            "Name": dio_name,
            "SignalMode": 3,
            "TriggerMode": 0,
            "TriggerSource": 0,
            "Username": dio_name,
        }
    )


def _create_group(parent: h5py.Group, name: str) -> h5py.Group:
    return parent.create_group(name, track_order=True)


def _require_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        return parent[name]  # type: ignore[return-value]
    return _create_group(parent, name)


def _create_signal_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    attrs: dict[str, Any] | None = None,
) -> h5py.Dataset:
    dataset = group.create_dataset(
        name,
        data=np.asarray(data, dtype=np.float64),
        dtype="float64",
    )
    if attrs is not None:
        dataset.attrs.update(attrs)
    return dataset


def _signal_attrs(
    name: str,
    range_min: float,
    range_max: float,
    unit: str,
) -> dict[str, str | float]:
    return {
        "Name": name,
        "RangeMax": range_max,
        "RangeMin": range_min,
        "Unit": unit,
        "Username": name,
    }


def _graph_attrs(
    name: str,
    color: str,
    index: int,
    range_min: float,
    range_max: float,
    unit: str,
) -> dict[str, str | int | float]:
    return {
        "Color": color,
        "Index": index,
        "Name": name,
        "PenSize": 1,
        "PenStyle": 1,
        "PointsStyle": 0,
        "RangeMax": range_max,
        "RangeMin": range_min,
        "Unit": unit,
        "Username": name,
    }


def _normalize_optional_int_tuple(
    values: Sequence[int] | np.ndarray | None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(
        _coerce_selector_int(value) for value in np.asarray(values).reshape(-1)
    )


def _normalize_float_tuple(values: Sequence[float] | np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float).reshape(-1))


def _coerce_selector_int(value: Any) -> int:
    if isinstance(value, (np.bool_, bool)):
        raise ValueError("selector values must be integers")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    raise ValueError("selector values must be integers")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_number(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_number(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_finite_number(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive_seconds_to_samples(value: float, fs: float, name: str) -> int:
    _require_positive_number(value, name)
    samples = int(round(value * fs))
    if samples < 1:
        raise ValueError(f"{name} must round to at least 1 sample")
    return samples


def _nonnegative_seconds_to_samples(value: float, fs: float, name: str) -> int:
    _require_nonnegative_number(value, name)
    return int(round(value * fs))


def _validate_signal_config(
    signal: SyntheticSignalConfig,
    config: SyntheticDoricConfig,
) -> None:
    _require_positive_number(signal.isosbestic_baseline, "isosbestic_baseline")
    _require_positive_number(signal.calcium_baseline, "calcium_baseline")
    _require_nonnegative_number(signal.channel_baseline_step, "channel_baseline_step")
    _require_nonnegative_number(signal.artifact_amplitude, "artifact_amplitude")
    _require_nonnegative_number(signal.circadian_amplitude, "circadian_amplitude")
    _require_nonnegative_number(signal.noise_std, "noise_std")
    _require_nonnegative_number(signal.analog_noise_std, "analog_noise_std")
    _require_nonnegative_number(
        signal.transient_rate_per_minute,
        "transient_rate_per_minute",
    )
    _require_nonnegative_number(signal.transient_amplitude, "transient_amplitude")
    _require_positive_number(signal.transient_rise_seconds, "transient_rise_seconds")
    _require_positive_number(signal.transient_decay_seconds, "transient_decay_seconds")
    for component in signal.tonic_components:
        _validate_tonic_component(component, config)
    for event_config in signal.scheduled_calcium_events:
        _validate_scheduled_calcium_event(event_config, config)
    for event_config in signal.random_calcium_events:
        _validate_random_calcium_event(event_config, config)
    for noise_config in signal.gaussian_noise:
        _validate_gaussian_noise(noise_config)


def _validate_tonic_component(
    component: SyntheticTonicComponentConfig,
    config: SyntheticDoricConfig,
) -> None:
    _require_nonnegative_number(component.amplitude, "tonic amplitude")
    _require_nonnegative_number(component.frequency_hz, "frequency_hz")
    _require_finite_number(component.phase_radians, "phase_radians")
    _require_finite_number(component.offset, "offset")
    _resolve_channel_numbers(component.channels, config.channel_count)
    _resolve_ttl_series_numbers(component.series_numbers, config.series_count)


def _validate_scheduled_calcium_event(
    event_config: SyntheticScheduledCalciumEventConfig,
    config: SyntheticDoricConfig,
) -> None:
    if len(event_config.event_times_seconds) == 0:
        raise ValueError("event_times_seconds must contain at least one time")
    _validate_calcium_event_shape(
        event_config.amplitude,
        event_config.rise_rate_per_second,
        event_config.fall_rate_per_second,
    )
    for event_time_seconds in event_config.event_times_seconds:
        _require_nonnegative_number(event_time_seconds, "event_times_seconds")
        event_sample = _nonnegative_seconds_to_samples(
            event_time_seconds,
            config.fs,
            "event_times_seconds",
        )
        if event_sample >= int(round(config.fs * config.session_duration_seconds)):
            raise ValueError("event_times_seconds must fall within the session")
    _resolve_channel_numbers(event_config.channels, config.channel_count)
    _resolve_ttl_series_numbers(event_config.series_numbers, config.series_count)


def _validate_random_calcium_event(
    event_config: SyntheticRandomCalciumEventConfig,
    config: SyntheticDoricConfig,
) -> None:
    _require_nonnegative_number(event_config.rate_per_minute, "rate_per_minute")
    _validate_calcium_event_shape(
        event_config.amplitude,
        event_config.rise_rate_per_second,
        event_config.fall_rate_per_second,
    )
    if event_config.start_window_seconds is not None:
        if len(event_config.start_window_seconds) != 2:
            raise ValueError("start_window_seconds must contain start and stop times")
        start_seconds, stop_seconds = event_config.start_window_seconds
        _require_nonnegative_number(start_seconds, "start_window_seconds")
        _require_nonnegative_number(stop_seconds, "start_window_seconds")
        if stop_seconds <= start_seconds:
            raise ValueError("start_window_seconds stop must be greater than start")
        if start_seconds >= config.session_duration_seconds:
            raise ValueError("start_window_seconds must overlap the session")
    _resolve_channel_numbers(event_config.channels, config.channel_count)
    _resolve_ttl_series_numbers(event_config.series_numbers, config.series_count)


def _validate_calcium_event_shape(
    amplitude: float,
    rise_rate_per_second: float,
    fall_rate_per_second: float,
) -> None:
    _require_nonnegative_number(amplitude, "calcium event amplitude")
    _require_positive_number(rise_rate_per_second, "rise_rate_per_second")
    _require_positive_number(fall_rate_per_second, "fall_rate_per_second")


def _validate_gaussian_noise(noise_config: SyntheticGaussianNoiseConfig) -> None:
    _require_nonnegative_number(noise_config.isosbestic_std, "isosbestic_std")
    _require_nonnegative_number(noise_config.calcium_std, "calcium_std")
    _require_nonnegative_number(noise_config.analog_in_std, "analog_in_std")


def _validate_ttl_behavior_codes(
    config: SyntheticDoricConfig,
) -> dict[str, SyntheticTTLBehaviorCodeConfig]:
    codes: dict[str, SyntheticTTLBehaviorCodeConfig] = {}
    for code in config.ttl_behavior_codes:
        if not code.name:
            raise ValueError("TTL behavior code name must not be empty")
        if code.name in codes:
            raise ValueError(f"duplicate TTL behavior code name: {code.name}")
        _validate_dio_channel(code.channel)
        _require_positive_int(code.pulse_count, "pulse_count")
        codes[code.name] = code
    return codes


def _ttl_code_for_schedule(
    code_name: str,
    codes: dict[str, SyntheticTTLBehaviorCodeConfig],
) -> SyntheticTTLBehaviorCodeConfig | None:
    try:
        code = codes[code_name]
    except KeyError as exc:
        raise ValueError(f"unknown TTL behavior code: {code_name}") from exc
    if not code.enabled:
        return None
    return code


def _validate_ttl_behavior_event(
    event_config: SyntheticTTLBehaviorEventConfig,
    config: SyntheticDoricConfig,
) -> None:
    if len(event_config.start_seconds) == 0:
        raise ValueError("start_seconds must contain at least one time")
    for start_seconds in event_config.start_seconds:
        _require_nonnegative_number(start_seconds, "start_seconds")
    _resolve_ttl_series_numbers(event_config.series_numbers, config.series_count)


def _validate_ttl_random_behavior_event(
    random_config: SyntheticTTLRandomBehaviorEventConfig,
    config: SyntheticDoricConfig,
) -> None:
    _require_nonnegative_int(
        random_config.event_count_per_series,
        "event_count_per_series",
    )
    _resolve_ttl_series_numbers(random_config.series_numbers, config.series_count)
    if random_config.start_window_seconds is not None:
        if len(random_config.start_window_seconds) != 2:
            raise ValueError("start_window_seconds must contain two values")
        start_seconds, stop_seconds = random_config.start_window_seconds
        _require_nonnegative_number(start_seconds, "start_window_seconds")
        _require_nonnegative_number(stop_seconds, "start_window_seconds")
        if stop_seconds < start_seconds:
            raise ValueError(
                "start_window_seconds stop must be greater than or equal to start"
            )


def _ttl_sequence_duration_samples(
    pulse_count: int,
    width_samples: int,
    off_samples: int,
) -> int:
    return pulse_count * width_samples + (pulse_count - 1) * off_samples


def _ttl_sequence_intervals(
    sequence_start_sample: int,
    pulse_count: int,
    width_samples: int,
    off_samples: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            sequence_start_sample + pulse_index * (width_samples + off_samples),
            sequence_start_sample
            + pulse_index * (width_samples + off_samples)
            + width_samples,
        )
        for pulse_index in range(pulse_count)
    )


def _ttl_sequence_overlaps(
    candidate_intervals: tuple[tuple[int, int], ...],
    existing_intervals: list[tuple[int, int]],
) -> bool:
    return any(
        candidate_start < existing_stop and existing_start < candidate_stop
        for candidate_start, candidate_stop in candidate_intervals
        for existing_start, existing_stop in existing_intervals
    )


def _random_window_samples(
    start_window_seconds: tuple[float, float] | None,
    config: SyntheticDoricConfig,
    validated: _ValidatedConfig,
) -> tuple[int, int]:
    if start_window_seconds is None:
        return 0, validated.samples_per_series

    start_seconds, stop_seconds = start_window_seconds
    start_sample = _nonnegative_seconds_to_samples(
        start_seconds,
        config.fs,
        "start_window_seconds",
    )
    stop_sample = _nonnegative_seconds_to_samples(
        stop_seconds,
        config.fs,
        "start_window_seconds",
    )
    return start_sample, min(stop_sample, validated.samples_per_series)


def _validate_dio_channel(channel: int) -> None:
    if not isinstance(channel, int) or channel not in (1, 2):
        raise ValueError("TTL channel must be 1 or 2")


def _resolve_channel_numbers(
    channels: tuple[int, ...] | None,
    channel_count: int,
) -> tuple[int, ...]:
    if channels is None:
        return tuple(range(1, channel_count + 1))

    channel_numbers = tuple(_coerce_selector_int(channel) for channel in channels)
    if len(channel_numbers) == 0:
        raise ValueError("channels must not be empty")
    if len(set(channel_numbers)) != len(channel_numbers):
        raise ValueError("channels must not contain duplicates")
    for channel_number in channel_numbers:
        if (
            not isinstance(channel_number, int)
            or isinstance(channel_number, bool)
            or not 1 <= channel_number <= channel_count
        ):
            raise ValueError("channels must contain values between 1 and channel_count")
    return channel_numbers


def _resolve_ttl_series_numbers(
    series_numbers: tuple[int, ...] | None,
    series_count: int,
) -> tuple[int, ...]:
    if series_numbers is None:
        return tuple(range(1, series_count + 1))

    resolved_series_numbers = tuple(
        _coerce_selector_int(series_number) for series_number in series_numbers
    )
    if len(resolved_series_numbers) == 0:
        raise ValueError("series_numbers must not be empty")
    if len(set(resolved_series_numbers)) != len(resolved_series_numbers):
        raise ValueError("series_numbers must not contain duplicates")
    for series_number in resolved_series_numbers:
        if (
            not isinstance(series_number, int)
            or isinstance(series_number, bool)
            or not 1 <= series_number <= series_count
        ):
            raise ValueError(
                "series_numbers must contain values between 1 and series_count"
            )
    return resolved_series_numbers
