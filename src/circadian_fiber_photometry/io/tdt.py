"""Write simulated streams in ChronoXIV's post-extraction TDT format.

This module does not create proprietary TDT tank files. It writes the
``streams.json``, ``tdt.StructType`` pickle, packed CSV, and optional epoc files
that ChronoXIV creates after reading a tank with ``tdt.read_block``.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np

DEFAULT_POINTS_PER_ROW = 128
PICKLE_PROTOCOL = 4
_STREAM_FIELD_NAMES = (
    "name",
    "code",
    "size",
    "type",
    "type_str",
    "ucf",
    "fs",
    "dform",
    "start_time",
    "data",
    "channel",
)


class TDTExtractError(ValueError):
    """Raised when a requested TDT extraction is malformed."""


class TDTDependencyError(RuntimeError):
    """Raised when faithful TDT export is requested without ``tdt`` installed."""


@dataclass(frozen=True)
class TDTExtractStream:
    """One uniformly sampled stream to serialize as ``tdt.StructType``.

    ``store_name`` is the unsanitized four-byte TDT store code, such as
    ``"470d"``. The serialized object and ``streams.json`` use
    ``tdt.fix_var_name(store_name)``, such as ``"_470d"``.
    """

    store_name: str
    data: np.ndarray
    fs: float
    channel: int = 1
    start_time: float = 0.0


@dataclass(frozen=True)
class TDTExtractEpoc:
    """Epoc events written to ``epoc.csv``."""

    name: str
    index: np.ndarray
    onset_seconds: np.ndarray
    offset_seconds: np.ndarray


@dataclass(frozen=True)
class TDTExtractMetadata:
    """Subject and logical source-tank metadata for one extraction."""

    subject_id: str
    subject_ids: tuple[str, ...]
    order: str
    datetime: datetime
    tank_dir: Path
    block_name: str


@dataclass(frozen=True)
class TDTExtractSummary:
    """Summary of one complete subject-session extraction."""

    path: Path
    subject_id: str
    datetime: str
    samples: int
    fs: float
    files: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NormalizedStream:
    store_name: str
    name: str
    data: np.ndarray
    fs: float
    channel: int
    start_time: float


@dataclass(frozen=True)
class _NormalizedEpoc:
    name: str
    index: np.ndarray
    onset_seconds: np.ndarray
    offset_seconds: np.ndarray


@dataclass(frozen=True)
class _NormalizedMetadata:
    subject_id: str
    subject_ids: tuple[str, ...]
    order: str
    datetime_text: str
    tank_dir: Path
    block_name: str


def write_tdt_extract(
    path: str | Path,
    *,
    isosbestic: TDTExtractStream,
    experimental: TDTExtractStream,
    metadata: TDTExtractMetadata,
    ttl: TDTExtractStream | None = None,
    epoc: TDTExtractEpoc | None = None,
    points_per_row: int = DEFAULT_POINTS_PER_ROW,
    remainder_policy: Literal["error"] = "error",
    overwrite: bool = False,
) -> TDTExtractSummary:
    """Write one ChronoXIV-compatible subject-session TDT extraction.

    The output is a post-extraction interchange directory, not a proprietary
    TDT tank. Stream sample counts must be divisible by ``points_per_row``;
    version 1 intentionally supports only ``remainder_policy="error"``.

    The optional :mod:`tdt` dependency is required to create faithful
    ``tdt.StructType`` pickles. Install this package with the ``tdt-export``
    extra before calling the writer.
    """

    tdt = _require_tdt()
    normalized_metadata = _normalize_metadata(metadata)
    normalized_isosbestic = _normalize_stream(isosbestic, "isosbestic", tdt)
    normalized_experimental = _normalize_stream(
        experimental,
        "experimental",
        tdt,
    )
    normalized_ttl = None if ttl is None else _normalize_stream(ttl, "TTL", tdt)
    _validate_stream_set(
        normalized_isosbestic,
        normalized_experimental,
        normalized_ttl,
    )
    points = _normalize_points_per_row(points_per_row, remainder_policy)
    if normalized_isosbestic.data.size % points != 0:
        raise TDTExtractError(
            f"stream sample count {normalized_isosbestic.data.size} is not "
            f"divisible by points_per_row={points}; remainder_policy='error'"
        )
    normalized_epoc = (
        None
        if epoc is None
        else _normalize_epoc(
            epoc,
            duration_seconds=normalized_isosbestic.data.size
            / normalized_isosbestic.fs,
        )
    )

    target = Path(path).expanduser()
    expected_name = (
        f"{normalized_metadata.subject_id}_{normalized_metadata.datetime_text}"
    )
    if target.name != expected_name:
        raise TDTExtractError(
            f"target directory must be named {expected_name!r}, got {target.name!r}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists; pass overwrite=True to replace it"
        )
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise TDTExtractError(
            "an existing extraction target must be a real directory"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.writing-",
            dir=target.parent,
        )
    )
    written_names: list[str] = []
    commit_warnings: tuple[str, ...] = ()
    try:
        manifest = _build_manifest(
            normalized_metadata,
            normalized_isosbestic,
            normalized_experimental,
            normalized_ttl,
            normalized_epoc,
        )
        (temporary / "streams.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        written_names.append("streams.json")

        _write_stream_files(
            temporary,
            "iso_stream",
            normalized_isosbestic,
            normalized_metadata.block_name,
            points,
            tdt,
        )
        written_names.extend(("iso_stream.pkl", "iso_stream.csv"))
        _write_stream_files(
            temporary,
            "exp_stream",
            normalized_experimental,
            normalized_metadata.block_name,
            points,
            tdt,
        )
        written_names.extend(("exp_stream.pkl", "exp_stream.csv"))

        if normalized_ttl is not None:
            _write_stream_files(
                temporary,
                "ttl_stream",
                normalized_ttl,
                normalized_metadata.block_name,
                points,
                tdt,
            )
            written_names.extend(("ttl_stream.pkl", "ttl_stream.csv"))
        if normalized_epoc is not None:
            _write_epoc_csv(temporary / "epoc.csv", normalized_epoc)
            written_names.append("epoc.csv")

        commit_warnings = _commit_directory(
            temporary,
            target,
            overwrite=overwrite,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return TDTExtractSummary(
        path=target,
        subject_id=normalized_metadata.subject_id,
        datetime=normalized_metadata.datetime_text,
        samples=int(normalized_isosbestic.data.size),
        fs=normalized_isosbestic.fs,
        files=tuple(target / name for name in written_names),
        warnings=commit_warnings,
    )


def _require_tdt() -> ModuleType:
    try:
        import tdt
    except ImportError as error:
        raise TDTDependencyError(
            "TDT extract export requires the optional 'tdt' package. "
            "Install the project with its 'tdt-export' extra."
        ) from error
    return tdt


def _normalize_metadata(metadata: TDTExtractMetadata) -> _NormalizedMetadata:
    subject_id = _validate_path_component(metadata.subject_id, "subject_id")
    subject_ids = tuple(
        _validate_path_component(value, "subject_ids entry")
        for value in metadata.subject_ids
    )
    if not subject_ids:
        raise TDTExtractError("subject_ids must contain at least one subject")
    if len(subject_ids) > 2:
        raise TDTExtractError("ChronoXIV TDT extracts support at most two subjects")
    if len(set(subject_ids)) != len(subject_ids):
        raise TDTExtractError("subject_ids must be unique")
    if subject_ids.count(subject_id) != 1:
        raise TDTExtractError("subject_id must occur exactly once in subject_ids")

    if metadata.order not in {"First", "Second"}:
        raise TDTExtractError("order must be 'First' or 'Second'")
    expected_index = 0 if metadata.order == "First" else 1
    if expected_index >= len(subject_ids) or subject_ids[expected_index] != subject_id:
        raise TDTExtractError(
            f"order {metadata.order!r} is inconsistent with subject_ids"
        )
    if not isinstance(metadata.datetime, datetime):
        raise TDTExtractError("datetime must be a datetime instance")

    block_name = _validate_csv_text(metadata.block_name, "block_name")
    try:
        tank_dir = Path(metadata.tank_dir).expanduser()
    except TypeError as error:
        raise TDTExtractError("tank_dir must be path-like") from error
    if not str(tank_dir):
        raise TDTExtractError("tank_dir must not be empty")

    return _NormalizedMetadata(
        subject_id=subject_id,
        subject_ids=subject_ids,
        order=metadata.order,
        datetime_text=metadata.datetime.strftime("%Y%m%d-%H%M%S"),
        tank_dir=tank_dir,
        block_name=block_name,
    )


def _normalize_stream(
    stream: TDTExtractStream,
    label: str,
    tdt: ModuleType,
) -> _NormalizedStream:
    if not isinstance(stream.store_name, str):
        raise TDTExtractError(f"{label} store_name must be a string")
    try:
        encoded_name = stream.store_name.encode("cp437")
    except UnicodeEncodeError as error:
        raise TDTExtractError(
            f"{label} store_name must be CP437-encodable"
        ) from error
    if len(encoded_name) != 4:
        raise TDTExtractError(
            f"{label} store_name must encode to exactly four CP437 bytes"
        )
    if b"\x00" in encoded_name or not stream.store_name.isprintable():
        raise TDTExtractError(
            f"{label} store_name must contain four printable, non-NUL characters"
        )

    fs = _positive_real(stream.fs, f"{label} fs")
    start_time = _nonnegative_real(stream.start_time, f"{label} start_time")
    if not isinstance(stream.channel, Integral) or isinstance(stream.channel, bool):
        raise TDTExtractError(f"{label} channel must be a positive integer")
    channel = int(stream.channel)
    if channel <= 0:
        raise TDTExtractError(f"{label} channel must be a positive integer")

    array = np.asarray(stream.data)
    if array.ndim != 1:
        raise TDTExtractError(f"{label} data must be one-dimensional")
    if array.size == 0:
        raise TDTExtractError(f"{label} data must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TDTExtractError(f"{label} data must be real numeric values")
    if not np.all(np.isfinite(array)):
        raise TDTExtractError(f"{label} data contains NaN or infinite values")
    data = np.array(array, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(data)):
        raise TDTExtractError(
            f"{label} data is outside the finite float32 range"
        )

    name = str(tdt.fix_var_name(stream.store_name))
    if not name:
        raise TDTExtractError(f"{label} store_name has no valid serialized name")
    return _NormalizedStream(
        store_name=stream.store_name,
        name=name,
        data=data,
        fs=fs,
        channel=channel,
        start_time=start_time,
    )


def _normalize_epoc(
    epoc: TDTExtractEpoc,
    *,
    duration_seconds: float,
) -> _NormalizedEpoc:
    name = _validate_csv_text(epoc.name, "epoc name")
    index = _numeric_1d_array(epoc.index, "epoc index")
    onset = _numeric_1d_array(epoc.onset_seconds, "epoc onset_seconds")
    offset = _numeric_1d_array(epoc.offset_seconds, "epoc offset_seconds")
    if not (index.size == onset.size == offset.size):
        raise TDTExtractError("epoc index, onset, and offset arrays must match")
    if np.any(onset < 0):
        raise TDTExtractError("epoc onsets must be nonnegative")
    if np.any(offset < onset):
        raise TDTExtractError("epoc offsets must be greater than or equal to onsets")
    if np.any(offset > duration_seconds):
        raise TDTExtractError("epoc offsets must not exceed the stream duration")
    if onset.size > 1 and np.any(np.diff(onset) < 0):
        raise TDTExtractError("epoc events must be ordered by onset")
    return _NormalizedEpoc(
        name=name,
        index=index,
        onset_seconds=onset,
        offset_seconds=offset,
    )


def _numeric_1d_array(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise TDTExtractError(f"{label} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TDTExtractError(f"{label} must contain real numeric values")
    normalized = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(normalized)):
        raise TDTExtractError(f"{label} contains NaN or infinite values")
    return normalized


def _validate_stream_set(
    isosbestic: _NormalizedStream,
    experimental: _NormalizedStream,
    ttl: _NormalizedStream | None,
) -> None:
    streams = [isosbestic, experimental]
    if ttl is not None:
        streams.append(ttl)
    names = [stream.name for stream in streams]
    if len(set(names)) != len(names):
        raise TDTExtractError("serialized stream names must be unique")

    samples = isosbestic.data.size
    fs = isosbestic.fs
    for stream in streams[1:]:
        if stream.data.size != samples:
            raise TDTExtractError("all selected streams must have equal lengths")
        if not np.isclose(stream.fs, fs, rtol=1e-12, atol=0.0):
            raise TDTExtractError(
                "all selected streams must have the same sampling rate"
            )


def _normalize_points_per_row(
    points_per_row: int,
    remainder_policy: str,
) -> int:
    if (
        not isinstance(points_per_row, Integral)
        or isinstance(points_per_row, bool)
        or points_per_row <= 0
    ):
        raise TDTExtractError("points_per_row must be a positive integer")
    if remainder_policy != "error":
        raise TDTExtractError(
            "version 1 supports only remainder_policy='error'"
        )
    return int(points_per_row)


def _build_manifest(
    metadata: _NormalizedMetadata,
    isosbestic: _NormalizedStream,
    experimental: _NormalizedStream,
    ttl: _NormalizedStream | None,
    epoc: _NormalizedEpoc | None,
) -> dict[str, object]:
    return {
        "order": metadata.order,
        "ttl_stream": "None" if ttl is None else ttl.name,
        "iso_stream": isosbestic.name,
        "exp_stream": experimental.name,
        "epoc": "None" if epoc is None else epoc.name,
        "tank_dir": str(metadata.tank_dir),
        "num_subjects_in_tank": len(metadata.subject_ids),
        "subject_ids": list(metadata.subject_ids),
        "subject_id": metadata.subject_id,
        "datetime": metadata.datetime_text,
    }


def _write_stream_files(
    directory: Path,
    file_stem: str,
    stream: _NormalizedStream,
    block_name: str,
    points_per_row: int,
    tdt: ModuleType,
) -> None:
    struct = _build_struct_type(stream, points_per_row, tdt)
    with (directory / f"{file_stem}.pkl").open("wb") as file_out:
        pickle.dump(struct, file_out, protocol=PICKLE_PROTOCOL)
    _write_packed_csv(
        directory / f"{file_stem}.csv",
        stream,
        block_name,
        points_per_row,
    )


def _build_struct_type(
    stream: _NormalizedStream,
    points_per_row: int,
    tdt: ModuleType,
) -> object:
    struct = tdt.StructType(
        name=stream.name,
        code=np.uint32(
            int.from_bytes(stream.store_name.encode("cp437"), "little")
        ),
        size=np.uint32(points_per_row + 10),
        type=np.uint32(tdt.EVTYPE_STREAM),
        type_str="streams",
        ucf=np.bool_(False),
        fs=np.float64(stream.fs),
        dform=np.uint32(tdt.DFORM_FLOAT),
        start_time=np.float64(stream.start_time),
        data=stream.data,
        channel=[stream.channel],
    )
    if tuple(struct.keys()) != _STREAM_FIELD_NAMES:
        raise TDTExtractError(
            "installed tdt.StructType did not preserve the expected field contract"
        )
    return struct


def _write_packed_csv(
    path: Path,
    stream: _NormalizedStream,
    block_name: str,
    points_per_row: int,
) -> None:
    data_rows = stream.data.reshape(-1, points_per_row)
    header = [
        "BLOCK",
        "EVENT",
        "TIME",
        "CHAN",
        "Sampling_Freq",
        "NumOfPoints",
        *(f"D{index}" for index in range(points_per_row)),
    ]
    with path.open("w", encoding="utf-8", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerow(header)
        for row_index, values in enumerate(data_rows):
            start_sample = row_index * points_per_row
            writer.writerow(
                [
                    block_name,
                    stream.name,
                    start_sample / stream.fs,
                    stream.channel,
                    stream.fs,
                    points_per_row,
                    *values.tolist(),
                ]
            )


def _write_epoc_csv(path: Path, epoc: _NormalizedEpoc) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerow(["index", "onset", "offset"])
        writer.writerows(
            zip(
                epoc.index.tolist(),
                epoc.onset_seconds.tolist(),
                epoc.offset_seconds.tolist(),
                strict=True,
            )
        )


def _commit_directory(
    temporary: Path,
    target: Path,
    *,
    overwrite: bool,
) -> tuple[str, ...]:
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists; pass overwrite=True to replace it"
            )
        if not target.is_dir() or target.is_symlink():
            raise TDTExtractError(
                "an existing extraction target must be a real directory"
            )
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        target.rename(backup)

    try:
        os.replace(temporary, target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            try:
                backup.rename(target)
            except Exception as restore_error:
                raise OSError(
                    f"failed to install {target} and restore its backup"
                ) from restore_error
        raise

    if backup is None:
        return ()
    try:
        shutil.rmtree(backup)
    except OSError as error:
        return (f"could not remove overwritten extraction backup {backup}: {error}",)
    return ()


def _positive_real(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TDTExtractError(f"{label} must be a finite number greater than zero")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise TDTExtractError(f"{label} must be a finite number greater than zero")
    return normalized


def _nonnegative_real(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TDTExtractError(f"{label} must be a finite nonnegative number")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0:
        raise TDTExtractError(f"{label} must be a finite nonnegative number")
    return normalized


def _validate_path_component(value: object, label: str) -> str:
    text = _validate_csv_text(value, label)
    if text in {".", ".."} or Path(text).name != text:
        raise TDTExtractError(f"{label} must be one safe path component")
    return text


def _validate_csv_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TDTExtractError(f"{label} must be a nonempty string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise TDTExtractError(f"{label} must not contain control line characters")
    return value


__all__ = [
    "DEFAULT_POINTS_PER_ROW",
    "PICKLE_PROTOCOL",
    "TDTDependencyError",
    "TDTExtractEpoc",
    "TDTExtractError",
    "TDTExtractMetadata",
    "TDTExtractStream",
    "TDTExtractSummary",
    "write_tdt_extract",
]
