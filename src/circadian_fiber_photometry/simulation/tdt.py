"""Adapters from normalized simulated datasets to ChronoXIV TDT extracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from ..io.tdt import (
    DEFAULT_POINTS_PER_ROW,
    TDTExtractEpoc,
    TDTExtractError,
    TDTExtractMetadata,
    TDTExtractStream,
    TDTExtractSummary,
    write_tdt_extract,
)
from ..models import DoricDataset
from .doric import SyntheticTTLBehaviorEventSummary


@dataclass(frozen=True)
class SyntheticTDTSubject:
    """Map one Doric photometry channel to one simulated TDT subject."""

    subject_id: str
    channel_number: int
    isosbestic_store_name: str = "x405"
    experimental_store_name: str = "x465"
    ttl_source: str | None = None
    ttl_store_name: str | None = None
    epocs_by_series: Mapping[int, TDTExtractEpoc] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticTDTExtractionRecord:
    """Provenance for one subject/series extraction."""

    summary: TDTExtractSummary
    subject_id: str
    channel_number: int
    series_number: int
    series_name: str
    session_datetime: datetime
    tank_dir: Path
    ttl_source: str | None
    epoc: TDTExtractEpoc | None


@dataclass(frozen=True)
class SyntheticTDTBatchSummary:
    """Summary of a multi-series simulated TDT export."""

    output_root: Path
    cohort: str
    records: tuple[SyntheticTDTExtractionRecord, ...]

    @property
    def extracted_count(self) -> int:
        """Number of subject-session extraction directories written."""

        return len(self.records)


def export_synthetic_tdt_extracts(
    output_root: str | Path,
    dataset: DoricDataset,
    *,
    cohort: str,
    subjects: Sequence[SyntheticTDTSubject],
    base_datetime: datetime,
    series_numbers: Sequence[int] | None = None,
    tank_dir_root: str | Path | None = None,
    tank_name_prefix: str = "synthetic",
    points_per_row: int = DEFAULT_POINTS_PER_ROW,
    overwrite: bool = False,
) -> SyntheticTDTBatchSummary:
    """Export selected Doric series/channels as ChronoXIV TDT extracts.

    ``base_datetime`` is the wall-clock datetime of the dataset's first series.
    Later session datetimes retain the offsets in
    :attr:`DoricDataset.session_start_times`. Subject channel and series numbers
    are one-based, matching Doric path numbering.
    """

    root = Path(output_root).expanduser()
    cohort_name = _safe_component(cohort, "cohort")
    prefix = _safe_component(tank_name_prefix, "tank_name_prefix")
    if not isinstance(base_datetime, datetime):
        raise TDTExtractError("base_datetime must be a datetime instance")
    _validate_dataset(dataset)
    selected_series = _normalize_series_numbers(
        series_numbers,
        dataset.series_count,
    )
    subject_configs = _normalize_subjects(subjects, dataset, selected_series)
    points = _positive_integer(points_per_row, "points_per_row")
    if dataset.samples_per_series % points != 0:
        raise TDTExtractError(
            f"dataset sample count {dataset.samples_per_series} is not divisible "
            f"by points_per_row={points}"
        )

    analysis_dir = root / "Photometry" / cohort_name / "Analysis"
    logical_tank_root = (
        Path(tank_dir_root).expanduser()
        if tank_dir_root is not None
        else root / "Photometry" / cohort_name / "TDT Binaries"
    )
    subject_ids = tuple(subject.subject_id for subject in subject_configs)
    series_datetimes = _series_datetimes(dataset, base_datetime, selected_series)

    targets = [
        analysis_dir
        / f"{subject.subject_id}_{series_datetimes[series_number]:%Y%m%d-%H%M%S}"
        for series_number in selected_series
        for subject in subject_configs
    ]
    if len(set(targets)) != len(targets):
        raise TDTExtractError(
            "series datetimes collide after YYYYMMDD-HHMMSS formatting"
        )
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                f"{existing[0]} already exists; pass overwrite=True to replace it"
            )

    records: list[SyntheticTDTExtractionRecord] = []
    for series_number in selected_series:
        series_index = series_number - 1
        session_datetime = series_datetimes[series_number]
        logical_tank_dir = logical_tank_root / (
            f"{prefix}-{session_datetime:%Y%m%d-%H%M%S}"
        )
        block_name = logical_tank_dir.name

        for subject_index, subject in enumerate(subject_configs):
            channel_index = subject.channel_number - 1
            ttl = None
            if subject.ttl_source is not None:
                ttl = TDTExtractStream(
                    store_name=str(subject.ttl_store_name),
                    data=dataset.digital_io[subject.ttl_source][:, series_index],
                    fs=dataset.fs,
                )
            epoc = subject.epocs_by_series.get(series_number)
            metadata = TDTExtractMetadata(
                subject_id=subject.subject_id,
                subject_ids=subject_ids,
                order="First" if subject_index == 0 else "Second",
                datetime=session_datetime,
                tank_dir=logical_tank_dir,
                block_name=block_name,
            )
            target = (
                analysis_dir
                / f"{subject.subject_id}_{session_datetime:%Y%m%d-%H%M%S}"
            )
            summary = write_tdt_extract(
                target,
                isosbestic=TDTExtractStream(
                    store_name=subject.isosbestic_store_name,
                    data=dataset.isosbestic_405[:, channel_index, series_index],
                    fs=dataset.fs,
                ),
                experimental=TDTExtractStream(
                    store_name=subject.experimental_store_name,
                    data=dataset.calcium_465[:, channel_index, series_index],
                    fs=dataset.fs,
                ),
                ttl=ttl,
                epoc=epoc,
                metadata=metadata,
                points_per_row=points,
                overwrite=overwrite,
            )
            records.append(
                SyntheticTDTExtractionRecord(
                    summary=summary,
                    subject_id=subject.subject_id,
                    channel_number=subject.channel_number,
                    series_number=series_number,
                    series_name=dataset.series_names[series_index],
                    session_datetime=session_datetime,
                    tank_dir=logical_tank_dir,
                    ttl_source=subject.ttl_source,
                    epoc=epoc,
                )
            )

    return SyntheticTDTBatchSummary(
        output_root=root,
        cohort=cohort_name,
        records=tuple(records),
    )


def build_tdt_epocs_from_behavior_events(
    events: Sequence[SyntheticTTLBehaviorEventSummary],
    *,
    name: str,
    fs: float,
    pulse_width_seconds: float,
    code_names: Sequence[str] | None = None,
) -> dict[int, TDTExtractEpoc]:
    """Convert simulator TTL behavior ground truth to per-series epocs.

    Each behavior sequence becomes one epoc. ``index`` contains its pulse count,
    onset is the sequence's first pulse, and offset is the end of its final
    pulse. Returned mapping keys are one-based series numbers.
    """

    epoc_name = _safe_text(name, "name")
    sampling_rate = _positive_real(fs, "fs")
    pulse_width = _positive_real(pulse_width_seconds, "pulse_width_seconds")
    selected_codes = None if code_names is None else set(code_names)
    if selected_codes is not None and any(
        not isinstance(code_name, str) or not code_name
        for code_name in selected_codes
    ):
        raise TDTExtractError("code_names must contain nonempty strings")

    grouped: dict[int, list[tuple[float, float, float]]] = {}
    for event in events:
        if selected_codes is not None and event.code_name not in selected_codes:
            continue
        pulse_samples = np.asarray(event.pulse_sample_indices)
        if pulse_samples.ndim != 1 or pulse_samples.size == 0:
            raise TDTExtractError("behavior events must contain pulse samples")
        onset = float(event.sequence_start_sample) / sampling_rate
        offset = float(np.max(pulse_samples)) / sampling_rate + pulse_width
        grouped.setdefault(int(event.series_number), []).append(
            (onset, offset, float(event.pulse_count))
        )

    epocs: dict[int, TDTExtractEpoc] = {}
    for series_number, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        epocs[series_number] = TDTExtractEpoc(
            name=epoc_name,
            index=np.asarray([row[2] for row in rows], dtype=np.float64),
            onset_seconds=np.asarray([row[0] for row in rows], dtype=np.float64),
            offset_seconds=np.asarray([row[1] for row in rows], dtype=np.float64),
        )
    return epocs


def _validate_dataset(dataset: DoricDataset) -> None:
    if not isinstance(dataset, DoricDataset):
        raise TypeError("dataset must be a DoricDataset")
    iso = np.asarray(dataset.isosbestic_405)
    calcium = np.asarray(dataset.calcium_465)
    if iso.ndim != 3 or calcium.ndim != 3:
        raise TDTExtractError("Doric photometry arrays must be three-dimensional")
    if iso.shape != calcium.shape:
        raise TDTExtractError("Doric isosbestic and calcium shapes must match")
    if iso.shape[0] == 0 or iso.shape[1] == 0 or iso.shape[2] == 0:
        raise TDTExtractError("Doric photometry arrays must not be empty")
    if len(dataset.series_names) != iso.shape[2]:
        raise TDTExtractError("series_names do not match the photometry arrays")
    if len(dataset.channel_names) != iso.shape[1]:
        raise TDTExtractError("channel_names do not match the photometry arrays")
    _positive_real(dataset.fs, "dataset.fs")

    timestamps = np.asarray(dataset.timestamps)
    if timestamps.shape != (iso.shape[0], iso.shape[2]):
        raise TDTExtractError("timestamps do not match the photometry arrays")
    starts = np.asarray(dataset.session_start_times, dtype=float)
    if not np.all(np.isfinite(starts)):
        raise TDTExtractError("session start times must be finite")
    if starts.size > 1 and np.any(np.diff(starts) <= 0):
        raise TDTExtractError("session start times must increase")

    for name, values in dataset.digital_io.items():
        if np.asarray(values).shape != (iso.shape[0], iso.shape[2]):
            raise TDTExtractError(
                f"digital_io stream {name!r} does not match the dataset shape"
            )


def _normalize_series_numbers(
    values: Sequence[int] | None,
    series_count: int,
) -> tuple[int, ...]:
    if values is None:
        return tuple(range(1, series_count + 1))
    normalized = tuple(
        _positive_integer(value, "series number") for value in values
    )
    if not normalized:
        raise TDTExtractError("series_numbers must not be empty")
    if len(set(normalized)) != len(normalized):
        raise TDTExtractError("series_numbers must be unique")
    if any(value > series_count for value in normalized):
        raise TDTExtractError(
            f"series_numbers must be between 1 and {series_count}"
        )
    return normalized


def _normalize_subjects(
    subjects: Sequence[SyntheticTDTSubject],
    dataset: DoricDataset,
    selected_series: tuple[int, ...],
) -> tuple[SyntheticTDTSubject, ...]:
    normalized = tuple(subjects)
    if not normalized or len(normalized) > 2:
        raise TDTExtractError("subjects must contain one or two mappings")
    subject_ids = [
        _safe_component(subject.subject_id, "subject_id")
        for subject in normalized
    ]
    if len(set(subject_ids)) != len(subject_ids):
        raise TDTExtractError("subject IDs must be unique")

    channels: list[int] = []
    for subject in normalized:
        channel = _positive_integer(subject.channel_number, "channel_number")
        if channel > dataset.channel_count:
            raise TDTExtractError(
                f"channel_number must be between 1 and {dataset.channel_count}"
            )
        channels.append(channel)
        _validate_store_name(subject.isosbestic_store_name, "isosbestic_store_name")
        _validate_store_name(
            subject.experimental_store_name,
            "experimental_store_name",
        )
        if (subject.ttl_source is None) != (subject.ttl_store_name is None):
            raise TDTExtractError(
                "ttl_source and ttl_store_name must either both be set or both be None"
            )
        if subject.ttl_source is not None:
            if subject.ttl_source not in dataset.digital_io:
                raise TDTExtractError(
                    f"unknown digital_io TTL source: {subject.ttl_source!r}"
                )
            _validate_store_name(str(subject.ttl_store_name), "ttl_store_name")
        if not isinstance(subject.epocs_by_series, Mapping):
            raise TDTExtractError("epocs_by_series must be a mapping")
        invalid_epoc_series = set(subject.epocs_by_series) - set(
            range(1, dataset.series_count + 1)
        )
        if invalid_epoc_series:
            raise TDTExtractError(
                f"epoc series number is outside the dataset: "
                f"{min(invalid_epoc_series)}"
            )
        for epoc in subject.epocs_by_series.values():
            if not isinstance(epoc, TDTExtractEpoc):
                raise TDTExtractError(
                    "epocs_by_series values must be TDTExtractEpoc instances"
                )
    if len(set(channels)) != len(channels):
        raise TDTExtractError("subject channel_number values must be unique")
    return normalized


def _series_datetimes(
    dataset: DoricDataset,
    base_datetime: datetime,
    selected_series: tuple[int, ...],
) -> dict[int, datetime]:
    starts = dataset.session_start_times
    first_start = float(starts[0])
    return {
        series_number: base_datetime
        + timedelta(seconds=float(starts[series_number - 1]) - first_start)
        for series_number in selected_series
    }


def _validate_store_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TDTExtractError(f"{label} must be a string")
    try:
        encoded = value.encode("cp437")
    except UnicodeEncodeError as error:
        raise TDTExtractError(f"{label} must be CP437-encodable") from error
    if len(encoded) != 4 or b"\x00" in encoded or not value.isprintable():
        raise TDTExtractError(
            f"{label} must contain exactly four printable CP437 bytes"
        )
    return value


def _safe_component(value: object, label: str) -> str:
    text = _safe_text(value, label)
    if text in {".", ".."} or Path(text).name != text:
        raise TDTExtractError(f"{label} must be one safe path component")
    return text


def _safe_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TDTExtractError(f"{label} must be a nonempty string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise TDTExtractError(f"{label} must not contain line control characters")
    return value


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise TDTExtractError(f"{label} must be a positive integer")
    return int(value)


def _positive_real(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TDTExtractError(f"{label} must be a finite number greater than zero")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise TDTExtractError(f"{label} must be a finite number greater than zero")
    return normalized


__all__ = [
    "SyntheticTDTBatchSummary",
    "SyntheticTDTExtractionRecord",
    "SyntheticTDTSubject",
    "build_tdt_epocs_from_behavior_events",
    "export_synthetic_tdt_extracts",
]
