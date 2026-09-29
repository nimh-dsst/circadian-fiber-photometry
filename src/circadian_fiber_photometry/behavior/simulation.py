"""Generate raw, ChronoXIV-compatible synthetic behavior CSV tables."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from circadian_fiber_photometry.simulation.doric import (
        SyntheticTTLBehaviorEventSummary,
    )

EVENT_TYPE_STRUCTURED = "Structured"
EVENT_TYPE_UNSTRUCTURED = "Unstructured"
EVENT_TYPE_OPTIONS = (EVENT_TYPE_STRUCTURED, EVENT_TYPE_UNSTRUCTURED)

EVENT_ROLE_POINT = "point"
EVENT_ROLE_INTERVAL = "interval"
EVENT_ROLE_OPTIONS = (EVENT_ROLE_POINT, EVENT_ROLE_INTERVAL)

SOURCE_FORMAT_SINGLE_BINARY = "single_binary"
SOURCE_FORMAT_PAIRED_ONSET_OFFSET = "paired_onset_offset"
SOURCE_FORMAT_OPTIONS = (
    SOURCE_FORMAT_SINGLE_BINARY,
    SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
)
CHRONOXIV_TIME_COLUMN_OUTPUT = "Time (s)"


@dataclass(frozen=True)
class SyntheticBehaviorEventConfig:
    """Describe one event and its raw behavior-table encoding.

    ``event_type``, ``event_role``, and ``output_column`` are mapping hints for
    ChronoXIV.  A single-binary event writes ``source_column``.  A paired event
    writes one-row markers to ``onset_column`` and ``offset_column``; ChronoXIV
    can combine those markers into ``output_column``.
    """

    output_column: str
    event_type: str = EVENT_TYPE_STRUCTURED
    event_role: str = EVENT_ROLE_POINT
    source_format: str = SOURCE_FORMAT_SINGLE_BINARY
    point_times_seconds: tuple[float, ...] = ()
    intervals_seconds: tuple[tuple[float, float], ...] = ()
    source_column: str | None = None
    onset_column: str | None = None
    offset_column: str | None = None
    interpretation: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class SyntheticBehaviorCoMetricConfig:
    """Numeric trace written beside event columns on the behavior time grid."""

    column_name: str
    values: Sequence[float | None] | NDArray[np.floating]
    description: str = ""


@dataclass(frozen=True)
class SyntheticBehaviorConfig:
    """Configuration for one raw synthetic behavior table."""

    duration_seconds: float
    sampling_rate_hz: float
    events: tuple[SyntheticBehaviorEventConfig, ...] = ()
    co_metrics: tuple[SyntheticBehaviorCoMetricConfig, ...] = ()
    time_column: str = "Time (s)"


@dataclass(frozen=True)
class SyntheticBehaviorEventMapping:
    """ChronoXIV mapping hint for generated source columns."""

    output_column: str
    event_type: str
    event_role: str
    source_format: str
    interpretation: str
    source_column: str | None = None
    onset_column: str | None = None
    offset_column: str | None = None


@dataclass(frozen=True)
class SyntheticBehaviorOccurrence:
    """Requested and sample-aligned ground truth for one event occurrence."""

    output_column: str
    requested_start_seconds: float
    requested_stop_seconds: float | None
    start_sample: int
    stop_sample: int | None
    realized_start_seconds: float
    realized_stop_seconds: float | None


@dataclass(frozen=True)
class SyntheticBehaviorDataset:
    """In-memory behavior table plus mapping and ground-truth information."""

    time_column: str
    time_seconds: NDArray[np.float64]
    columns: Mapping[str, NDArray[np.float64] | NDArray[np.int_]]
    sampling_rate_hz: float
    requested_duration_seconds: float
    event_mappings: tuple[SyntheticBehaviorEventMapping, ...]
    event_occurrences: tuple[SyntheticBehaviorOccurrence, ...]
    co_metric_descriptions: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        """Number of time rows in the generated table."""

        return int(self.time_seconds.size)

    @property
    def column_names(self) -> tuple[str, ...]:
        """CSV columns in write order, including the time column."""

        return (self.time_column, *self.columns.keys())


@dataclass(frozen=True)
class SyntheticBehaviorSummary:
    """Provenance and ground truth returned after writing a behavior CSV."""

    path: Path
    row_count: int
    sampling_rate_hz: float
    requested_duration_seconds: float
    realized_duration_seconds: float
    time_column: str
    column_names: tuple[str, ...]
    event_mappings: tuple[SyntheticBehaviorEventMapping, ...]
    event_occurrences: tuple[SyntheticBehaviorOccurrence, ...]
    co_metric_descriptions: Mapping[str, str]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ValidatedEvent:
    config: SyntheticBehaviorEventConfig
    mapping: SyntheticBehaviorEventMapping
    source_columns: tuple[str, ...]


def simulate_behavior(config: SyntheticBehaviorConfig) -> SyntheticBehaviorDataset:
    """Materialize a configured behavior table without writing it.

    Time rows are a regular, zero-based grid. Point times and interval edges
    are aligned to their nearest behavior sample, matching the synthetic Doric
    simulator's seconds-to-samples convention.
    """

    duration = _positive_real(config.duration_seconds, "duration_seconds")
    sampling_rate = _positive_real(config.sampling_rate_hz, "sampling_rate_hz")
    sample_count = int(round(duration * sampling_rate))
    if sample_count < 1:
        raise ValueError(
            "duration_seconds and sampling_rate_hz must produce at least one sample"
        )

    time_column = _column_name(config.time_column, "time_column")
    validated_events = _validate_events(config.events, time_column)
    co_metrics = _validate_co_metrics(
        config.co_metrics,
        sample_count=sample_count,
        reserved_columns={
            time_column,
            *(event.mapping.output_column for event in validated_events),
            *(column for event in validated_events for column in event.source_columns),
        },
    )

    if not validated_events and not co_metrics:
        raise ValueError("behavior config must define at least one event or co-metric")

    time_seconds = np.arange(sample_count, dtype=np.float64) / sampling_rate
    columns: dict[str, NDArray[np.float64] | NDArray[np.int_]] = {
        co_metric.column_name: values for co_metric, values in co_metrics
    }
    occurrences: list[SyntheticBehaviorOccurrence] = []
    warnings: list[str] = []

    for validated_event in validated_events:
        event_columns, event_occurrences, event_warnings = _materialize_event(
            validated_event,
            sample_count=sample_count,
            duration_seconds=duration,
            sampling_rate_hz=sampling_rate,
        )
        columns.update(event_columns)
        occurrences.extend(event_occurrences)
        warnings.extend(event_warnings)

    return SyntheticBehaviorDataset(
        time_column=time_column,
        time_seconds=time_seconds,
        columns=columns,
        sampling_rate_hz=sampling_rate,
        requested_duration_seconds=duration,
        event_mappings=tuple(event.mapping for event in validated_events),
        event_occurrences=tuple(occurrences),
        co_metric_descriptions={
            co_metric.column_name: str(co_metric.description)
            for co_metric, _ in co_metrics
        },
        warnings=tuple(warnings),
    )


def write_behavior_csv(
    path: str | Path,
    dataset: SyntheticBehaviorDataset,
    *,
    overwrite: bool = False,
) -> SyntheticBehaviorSummary:
    """Write an in-memory synthetic behavior table as a raw CSV file."""

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it"
        )
    if output_path.parent and not output_path.parent.exists():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )
    _validate_dataset_for_write(dataset)

    with output_path.open("w", encoding="utf-8", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerow(dataset.column_names)
        column_values = tuple(dataset.columns.values())
        for row_index, time_value in enumerate(dataset.time_seconds):
            writer.writerow(
                [
                    _format_float(float(time_value)),
                    *(
                        _format_column_value(values[row_index])
                        for values in column_values
                    ),
                ]
            )

    return SyntheticBehaviorSummary(
        path=output_path,
        row_count=dataset.row_count,
        sampling_rate_hz=dataset.sampling_rate_hz,
        requested_duration_seconds=dataset.requested_duration_seconds,
        realized_duration_seconds=dataset.row_count / dataset.sampling_rate_hz,
        time_column=dataset.time_column,
        column_names=dataset.column_names,
        event_mappings=dataset.event_mappings,
        event_occurrences=dataset.event_occurrences,
        co_metric_descriptions=dict(dataset.co_metric_descriptions),
        warnings=dataset.warnings,
    )


def generate_synthetic_behavior_csv(
    path: str | Path,
    config: SyntheticBehaviorConfig,
    *,
    overwrite: bool = False,
) -> SyntheticBehaviorSummary:
    """Materialize and write one raw synthetic behavior CSV."""

    return write_behavior_csv(
        path,
        simulate_behavior(config),
        overwrite=overwrite,
    )


def build_behavior_event_configs_from_ttl(
    events: Sequence[SyntheticTTLBehaviorEventSummary],
    *,
    fs: float,
    series_number: int,
    code_names: Sequence[str] | None = None,
    output_columns: Mapping[str, str] | None = None,
    event_type: str = EVENT_TYPE_STRUCTURED,
) -> tuple[SyntheticBehaviorEventConfig, ...]:
    """Convert Doric TTL ground truth into single-binary point-event configs.

    TTL summary times may be absolute across a multi-series recording.  This
    adapter deliberately uses ``sequence_start_sample / fs`` so each behavior
    CSV remains relative to the beginning of its selected series.
    """

    sampling_rate = _positive_real(fs, "fs")
    selected_series = _positive_integer(series_number, "series_number")
    if event_type not in EVENT_TYPE_OPTIONS:
        raise ValueError("event_type must be one of: " + ", ".join(EVENT_TYPE_OPTIONS))

    selected_codes = None
    code_order: list[str] = []
    if code_names is not None:
        code_order = [_column_name(name, "code name") for name in code_names]
        if len(set(code_order)) != len(code_order):
            raise ValueError("code_names must be unique")
        selected_codes = set(code_order)

    grouped: dict[str, list[float]] = {}
    for event in events:
        if int(event.series_number) != selected_series:
            continue
        code_name = _column_name(event.code_name, "TTL behavior code name")
        if selected_codes is not None and code_name not in selected_codes:
            continue
        if code_name not in grouped:
            grouped[code_name] = []
            if code_names is None:
                code_order.append(code_name)
        sequence_start_sample = _nonnegative_integer(
            event.sequence_start_sample,
            "TTL sequence_start_sample",
        )
        grouped[code_name].append(sequence_start_sample / sampling_rate)

    output_names = output_columns or {}
    unknown_output_names = set(output_names).difference(code_order)
    if unknown_output_names:
        raise ValueError(
            "output_columns contains unknown TTL codes: "
            + ", ".join(sorted(unknown_output_names))
        )

    return tuple(
        SyntheticBehaviorEventConfig(
            output_column=_column_name(
                output_names.get(code_name, code_name),
                f"output column for {code_name!r}",
            ),
            event_type=event_type,
            event_role=EVENT_ROLE_POINT,
            source_format=SOURCE_FORMAT_SINGLE_BINARY,
            point_times_seconds=tuple(sorted(grouped.get(code_name, ()))),
            interpretation="TTL behavior event",
        )
        for code_name in code_order
        if code_name in grouped
    )


def _validate_events(
    events: Sequence[SyntheticBehaviorEventConfig],
    time_column: str,
) -> tuple[_ValidatedEvent, ...]:
    validated: list[_ValidatedEvent] = []
    output_columns: set[str] = set()
    source_columns: set[str] = {time_column}

    for index, config in enumerate(events, start=1):
        if not isinstance(config, SyntheticBehaviorEventConfig):
            raise TypeError(
                f"events item {index} must be a SyntheticBehaviorEventConfig"
            )
        if not config.enabled:
            continue

        output_column = _column_name(config.output_column, "output_column")
        if output_column in {time_column, CHRONOXIV_TIME_COLUMN_OUTPUT}:
            raise ValueError(
                f"event output_column cannot reuse a time column: {output_column}"
            )
        if output_column in output_columns:
            raise ValueError(f"duplicate event output_column: {output_column}")
        output_columns.add(output_column)

        if config.event_type not in EVENT_TYPE_OPTIONS:
            raise ValueError(
                f"event {output_column!r} event_type must be one of: "
                + ", ".join(EVENT_TYPE_OPTIONS)
            )
        if config.event_role not in EVENT_ROLE_OPTIONS:
            raise ValueError(
                f"event {output_column!r} event_role must be one of: "
                + ", ".join(EVENT_ROLE_OPTIONS)
            )
        if config.source_format not in SOURCE_FORMAT_OPTIONS:
            raise ValueError(
                f"event {output_column!r} source_format must be one of: "
                + ", ".join(SOURCE_FORMAT_OPTIONS)
            )
        if config.event_role == EVENT_ROLE_POINT and config.intervals_seconds:
            raise ValueError(
                f"point event {output_column!r} cannot define intervals_seconds"
            )
        if config.event_role == EVENT_ROLE_INTERVAL and config.point_times_seconds:
            raise ValueError(
                f"interval event {output_column!r} cannot define point_times_seconds"
            )

        interpretation = config.interpretation
        if interpretation is None:
            interpretation = (
                "Subject-driven event"
                if config.event_type == EVENT_TYPE_UNSTRUCTURED
                else "Timed event"
            )
        else:
            interpretation = str(interpretation).strip()
            if not interpretation:
                raise ValueError(
                    f"event {output_column!r} interpretation must not be blank"
                )

        if config.source_format == SOURCE_FORMAT_SINGLE_BINARY:
            if config.onset_column is not None or config.offset_column is not None:
                raise ValueError(
                    f"single-binary event {output_column!r} cannot define onset or "
                    "offset columns"
                )
            source_column = _column_name(
                config.source_column or output_column,
                f"source column for {output_column!r}",
            )
            event_source_columns = (source_column,)
            mapping = SyntheticBehaviorEventMapping(
                output_column=output_column,
                event_type=config.event_type,
                event_role=config.event_role,
                source_format=config.source_format,
                interpretation=interpretation,
                source_column=source_column,
            )
        else:
            if config.event_role != EVENT_ROLE_INTERVAL:
                raise ValueError(
                    f"paired event {output_column!r} must have event_role 'interval'"
                )
            if config.source_column is not None:
                raise ValueError(
                    f"paired event {output_column!r} cannot define source_column"
                )
            onset_column = _column_name(
                config.onset_column or f"{output_column} onset",
                f"onset column for {output_column!r}",
            )
            offset_column = _column_name(
                config.offset_column or f"{output_column} offset",
                f"offset column for {output_column!r}",
            )
            if onset_column == offset_column:
                raise ValueError(
                    f"paired event {output_column!r} onset and offset columns must "
                    "differ"
                )
            event_source_columns = (onset_column, offset_column)
            mapping = SyntheticBehaviorEventMapping(
                output_column=output_column,
                event_type=config.event_type,
                event_role=config.event_role,
                source_format=config.source_format,
                interpretation=interpretation,
                onset_column=onset_column,
                offset_column=offset_column,
            )

        for column in event_source_columns:
            if column in source_columns:
                raise ValueError(f"duplicate behavior source column: {column}")
            source_columns.add(column)
        validated.append(
            _ValidatedEvent(
                config=config,
                mapping=mapping,
                source_columns=event_source_columns,
            )
        )

    return tuple(validated)


def _validate_co_metrics(
    co_metrics: Sequence[SyntheticBehaviorCoMetricConfig],
    *,
    sample_count: int,
    reserved_columns: set[str],
) -> tuple[tuple[SyntheticBehaviorCoMetricConfig, NDArray[np.float64]], ...]:
    validated: list[tuple[SyntheticBehaviorCoMetricConfig, NDArray[np.float64]]] = []
    used_columns = set(reserved_columns)
    for index, config in enumerate(co_metrics, start=1):
        if not isinstance(config, SyntheticBehaviorCoMetricConfig):
            raise TypeError(
                f"co_metrics item {index} must be a SyntheticBehaviorCoMetricConfig"
            )
        column_name = _column_name(config.column_name, "co-metric column_name")
        if column_name in used_columns:
            raise ValueError(f"duplicate behavior column: {column_name}")
        used_columns.add(column_name)
        try:
            values = np.asarray(config.values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"co-metric {column_name!r} must contain numeric values or nulls"
            ) from error
        if values.ndim != 1 or values.size != sample_count:
            raise ValueError(
                f"co-metric {column_name!r} must contain exactly {sample_count} values"
            )
        if np.isinf(values).any():
            raise ValueError(f"co-metric {column_name!r} cannot contain infinity")
        validated.append((config, values.copy()))
    return tuple(validated)


def _materialize_event(
    event: _ValidatedEvent,
    *,
    sample_count: int,
    duration_seconds: float,
    sampling_rate_hz: float,
) -> tuple[
    dict[str, NDArray[np.int_]],
    list[SyntheticBehaviorOccurrence],
    list[str],
]:
    config = event.config
    output_column = event.mapping.output_column
    values = {
        column: np.zeros(sample_count, dtype=int) for column in event.source_columns
    }
    occurrences: list[SyntheticBehaviorOccurrence] = []
    warnings: list[str] = []

    if config.event_role == EVENT_ROLE_POINT:
        source_column = event.source_columns[0]
        used_samples: set[int] = set()
        for requested_time in config.point_times_seconds:
            time = _event_time(requested_time, output_column, "point time")
            sample = int(round(time * sampling_rate_hz))
            if time >= duration_seconds or sample >= sample_count:
                raise ValueError(
                    f"point event {output_column!r} at {time:g}s is outside "
                    "the behavior session"
                )
            if sample in used_samples:
                warnings.append(
                    f"point event {output_column!r} has multiple occurrences "
                    f"aligned to sample {sample}; the binary source column stores "
                    "one marker"
                )
            used_samples.add(sample)
            values[source_column][sample] = 1
            occurrences.append(
                SyntheticBehaviorOccurrence(
                    output_column=output_column,
                    requested_start_seconds=time,
                    requested_stop_seconds=None,
                    start_sample=sample,
                    stop_sample=None,
                    realized_start_seconds=sample / sampling_rate_hz,
                    realized_stop_seconds=None,
                )
            )
        return values, occurrences, warnings

    aligned_intervals: list[tuple[int, int]] = []
    for interval_index, interval in enumerate(config.intervals_seconds, start=1):
        if isinstance(interval, (str, bytes)):
            interval_values: tuple[object, ...] = ()
        else:
            try:
                interval_values = tuple(interval)
            except TypeError:
                interval_values = ()
        if len(interval_values) != 2:
            raise ValueError(
                f"event {output_column!r} interval {interval_index} must "
                "contain start and stop seconds"
            )
        requested_start = _event_time(
            interval_values[0],
            output_column,
            f"interval {interval_index} start",
        )
        requested_stop = _event_time(
            interval_values[1],
            output_column,
            f"interval {interval_index} stop",
        )
        if requested_start >= requested_stop:
            raise ValueError(
                f"event {output_column!r} interval {interval_index} start "
                "must be before stop"
            )
        if requested_start >= duration_seconds or requested_stop > duration_seconds:
            raise ValueError(
                f"event {output_column!r} interval {interval_index} is "
                "outside the behavior session"
            )

        start_sample = int(round(requested_start * sampling_rate_hz))
        stop_sample = int(round(requested_stop * sampling_rate_hz))
        if start_sample >= stop_sample:
            raise ValueError(
                f"event {output_column!r} interval {interval_index} must "
                "span at least one behavior sample after alignment"
            )
        if start_sample >= sample_count or stop_sample > sample_count:
            raise ValueError(
                f"event {output_column!r} interval {interval_index} is "
                "outside the realized behavior time grid"
            )
        if (
            config.source_format == SOURCE_FORMAT_PAIRED_ONSET_OFFSET
            and stop_sample >= sample_count
        ):
            raise ValueError(
                f"paired event {output_column!r} interval {interval_index} "
                "needs an offset marker before the final behavior row"
            )
        aligned_intervals.append((start_sample, stop_sample))
        occurrences.append(
            SyntheticBehaviorOccurrence(
                output_column=output_column,
                requested_start_seconds=requested_start,
                requested_stop_seconds=requested_stop,
                start_sample=start_sample,
                stop_sample=stop_sample,
                realized_start_seconds=start_sample / sampling_rate_hz,
                realized_stop_seconds=stop_sample / sampling_rate_hz,
            )
        )

    if config.source_format == SOURCE_FORMAT_SINGLE_BINARY:
        source_column = event.source_columns[0]
        for start_sample, stop_sample in aligned_intervals:
            values[source_column][start_sample:stop_sample] = 1
        return values, occurrences, warnings

    _validate_paired_intervals(output_column, aligned_intervals)
    onset_column, offset_column = event.source_columns
    for start_sample, stop_sample in aligned_intervals:
        values[onset_column][start_sample] = 1
        values[offset_column][stop_sample] = 1
    return values, occurrences, warnings


def _validate_paired_intervals(
    output_column: str,
    intervals: Sequence[tuple[int, int]],
) -> None:
    previous_stop = -1
    for start_sample, stop_sample in sorted(intervals):
        if start_sample <= previous_stop:
            raise ValueError(
                f"paired event {output_column!r} intervals overlap or share an "
                "onset/offset sample"
            )
        previous_stop = stop_sample


def _validate_dataset_for_write(dataset: SyntheticBehaviorDataset) -> None:
    if not isinstance(dataset, SyntheticBehaviorDataset):
        raise TypeError("dataset must be a SyntheticBehaviorDataset")
    if dataset.time_seconds.ndim != 1 or dataset.time_seconds.size == 0:
        raise ValueError("behavior time grid must be a nonempty vector")
    if not np.isfinite(dataset.time_seconds).all():
        raise ValueError("behavior time grid must contain only finite values")
    if dataset.time_seconds.size > 1 and np.any(np.diff(dataset.time_seconds) < 0):
        raise ValueError("behavior time grid must be non-decreasing")
    for name, values in dataset.columns.items():
        _column_name(name, "behavior column")
        if np.asarray(values).shape != dataset.time_seconds.shape:
            raise ValueError(f"behavior column {name!r} does not match the time grid")


def _format_column_value(value: object) -> str | int:
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    numeric = float(value)
    if np.isnan(numeric):
        return ""
    return _format_float(numeric)


def _format_float(value: float) -> str:
    return format(value, ".15g")


def _column_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    normalized = value.strip()
    if any(character in normalized for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{label} must not contain line control characters")
    return normalized


def _event_time(value: object, output_column: str, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"event {output_column!r} {label} must be a finite number")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0:
        raise ValueError(
            f"event {output_column!r} {label} must be finite and nonnegative"
        )
    return normalized


def _positive_real(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number greater than zero")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{label} must be a finite number greater than zero")
    return normalized


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return normalized


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return normalized


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
