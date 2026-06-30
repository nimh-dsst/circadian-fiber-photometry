"""Synthetic Doric HDF5 file generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class SyntheticSignalConfig:
    """Signal-shape parameters for generated photometry traces."""

    isosbestic_baseline: float = 0.08
    calcium_baseline: float = 0.18
    channel_baseline_step: float = 0.006
    bleaching_fraction: float = 0.20
    artifact_amplitude: float = 0.004
    circadian_amplitude: float = 0.012
    noise_std: float = 0.0015
    analog_noise_std: float = 0.05
    transient_rate_per_minute: float = 1.0
    transient_amplitude: float = 0.030
    transient_rise_seconds: float = 0.4
    transient_decay_seconds: float = 3.0


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
    ttl_schedule = _build_ttl_schedule(config, validated, session_start_times)
    event_sample_indices: dict[tuple[int, int], np.ndarray] = {}
    event_times_seconds: dict[tuple[int, int], np.ndarray] = {}

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
                )
                key = (series_index + 1, channel_index + 1)
                event_sample_indices[key] = generated.event_indices
                event_times_seconds[key] = (
                    series_start + generated.event_indices / config.fs
                )

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
    )


@dataclass(frozen=True)
class _GeneratedChannel:
    isosbestic_405: np.ndarray
    calcium_465: np.ndarray
    analog_in: np.ndarray
    event_indices: np.ndarray


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

        _validate_signal_config(config.signal)
        return cls(
            configured_series_count=configured_series_count,
            samples_per_series=samples_per_series,
            crop_samples=crop_samples,
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
) -> _GeneratedChannel:
    signal = config.signal
    samples = validated.samples_per_series
    relative_time = np.arange(samples, dtype=float) / config.fs
    total_series = max(config.series_count - 1, 1)
    experiment_progress = series_index / total_series
    channel_phase = 0.7 * channel_index
    session_phase = 2 * np.pi * absolute_time[0] / 86400 + channel_phase

    base_405 = signal.isosbestic_baseline + (
        signal.channel_baseline_step * channel_index
    )
    base_465 = signal.calcium_baseline + (
        signal.channel_baseline_step * channel_index * 1.6
    )
    bleaching = 1 - signal.bleaching_fraction * experiment_progress
    within_session_bleach = 1 - 0.015 * (relative_time / relative_time[-1])
    artifact = signal.artifact_amplitude * np.sin(
        2 * np.pi * 0.07 * relative_time + session_phase
    )
    drift = signal.artifact_amplitude * 0.5 * np.sin(
        2 * np.pi * 0.011 * relative_time + channel_phase
    )
    noise_405 = rng.normal(0.0, signal.noise_std, samples)
    isosbestic = (base_405 * bleaching * within_session_bleach) + artifact + drift
    isosbestic = isosbestic + noise_405

    circadian = signal.circadian_amplitude * np.sin(session_phase)
    event_indices = _draw_event_indices(rng, config, validated)
    transient_signal = _transient_trace(
        event_indices,
        samples,
        config.fs,
        signal.transient_amplitude * (1 + 0.15 * channel_index),
        signal.transient_rise_seconds,
        signal.transient_decay_seconds,
    )
    noise_465 = rng.normal(0.0, signal.noise_std, samples)
    calcium = (
        base_465
        + 1.25 * (isosbestic - base_405)
        + circadian
        + transient_signal
        + noise_465
    )

    analog_in = (
        0.75
        + 2.5 * isosbestic
        + 1.5 * transient_signal
        + rng.normal(0.0, signal.analog_noise_std, samples)
    )

    return _GeneratedChannel(
        isosbestic_405=isosbestic.astype(np.float64),
        calcium_465=calcium.astype(np.float64),
        analog_in=analog_in.astype(np.float64),
        event_indices=event_indices.astype(int),
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


def _positive_seconds_to_samples(value: float, fs: float, name: str) -> int:
    _require_positive_number(value, name)
    samples = int(round(value * fs))
    if samples < 1:
        raise ValueError(f"{name} must round to at least 1 sample")
    return samples


def _nonnegative_seconds_to_samples(value: float, fs: float, name: str) -> int:
    _require_nonnegative_number(value, name)
    return int(round(value * fs))


def _validate_signal_config(signal: SyntheticSignalConfig) -> None:
    _require_positive_number(signal.isosbestic_baseline, "isosbestic_baseline")
    _require_positive_number(signal.calcium_baseline, "calcium_baseline")
    _require_nonnegative_number(signal.channel_baseline_step, "channel_baseline_step")
    if not 0 <= signal.bleaching_fraction < 1:
        raise ValueError("bleaching_fraction must be in [0, 1)")
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


def _resolve_ttl_series_numbers(
    series_numbers: tuple[int, ...] | None,
    series_count: int,
) -> tuple[int, ...]:
    if series_numbers is None:
        return tuple(range(1, series_count + 1))

    if len(series_numbers) == 0:
        raise ValueError("series_numbers must not be empty")
    if len(set(series_numbers)) != len(series_numbers):
        raise ValueError("series_numbers must not contain duplicates")
    for series_number in series_numbers:
        if not isinstance(series_number, int) or not 1 <= series_number <= series_count:
            raise ValueError(
                "series_numbers must contain values between 1 and series_count"
            )
    return series_numbers
