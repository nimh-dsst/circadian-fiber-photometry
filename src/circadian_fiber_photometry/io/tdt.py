"""Serialize simulated data in the shape returned by :func:`tdt.read_block`.

This module does not create proprietary TDT tank files. It creates one trusted
pickle per simulated recording. The pickle contains the top-level
``tdt.StructType`` object returned by ``tdt.read_block``, rather than files
extracted from that object by a downstream application.
"""

from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np

DEFAULT_POINTS_PER_ROW = 128
PICKLE_PROTOCOL = 4
TDT_BLOCK_PICKLE_FILENAME = "tdt_block.pkl"
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
_BLOCK_FIELD_NAMES = ("epocs", "snips", "streams", "scalars", "info")


class TDTExtractError(ValueError):
    """Raised when requested simulated TDT data is malformed."""


class TDTDependencyError(RuntimeError):
    """Raised when faithful TDT serialization needs an unavailable dependency."""


@dataclass(frozen=True)
class TDTExtractStream:
    """One uniformly sampled stream to include in a simulated TDT block.

    ``store_name`` is the stream's name in the block. ``code_name`` is its
    four-byte TDT event code. It may be omitted when ``store_name`` itself is
    exactly four CP437 bytes; those legacy names are normalized with
    ``tdt.fix_var_name``. Supplying ``code_name`` preserves ``store_name``
    exactly, allowing descriptive simulated names such as ``"405_left"``.
    """

    store_name: str
    data: np.ndarray
    fs: float
    channel: int = 1
    start_time: float = 0.0
    code_name: str | None = None


@dataclass(frozen=True)
class TDTExtractEpoc:
    """One TDT epoc store to include in a simulated TDT block."""

    name: str
    index: np.ndarray
    onset_seconds: np.ndarray
    offset_seconds: np.ndarray


@dataclass(frozen=True)
class TDTExtractMetadata:
    """Recording and logical source-block metadata for one simulated block."""

    subject_id: str
    subject_ids: tuple[str, ...]
    order: str
    datetime: datetime
    tank_dir: Path
    block_name: str


@dataclass(frozen=True)
class TDTExtractSummary:
    """Summary of one serialized simulated TDT block."""

    path: Path
    subject_id: str
    datetime: str
    samples: int
    fs: float
    files: tuple[Path, ...]
    warnings: tuple[str, ...] = ()

    @property
    def block_file(self) -> Path:
        """The sole pickle written inside :attr:`path`."""

        return self.files[0]


# Clearer names for new callers. The Extract-prefixed names remain public so
# existing simulation configuration code does not need an unrelated migration.
TDTBlockError = TDTExtractError
TDTBlockEpoc = TDTExtractEpoc
TDTBlockMetadata = TDTExtractMetadata
TDTBlockStream = TDTExtractStream
TDTBlockSummary = TDTExtractSummary


@dataclass(frozen=True)
class _NormalizedStream:
    store_name: str
    name: str
    code_name: str
    data: np.ndarray
    fs: float
    channel: int
    start_time: float


@dataclass(frozen=True)
class _NormalizedEpoc:
    name: str
    variable_name: str
    index: np.ndarray
    onset_seconds: np.ndarray
    offset_seconds: np.ndarray


@dataclass(frozen=True)
class _NormalizedMetadata:
    subject_id: str
    subject_ids: tuple[str, ...]
    order: str
    session_datetime: datetime
    datetime_text: str
    tank_dir: Path
    block_name: str


def write_tdt_block(
    path: str | Path,
    *,
    metadata: TDTExtractMetadata,
    streams: Sequence[TDTExtractStream] | None = None,
    isosbestic: TDTExtractStream | None = None,
    experimental: TDTExtractStream | None = None,
    ttl: TDTExtractStream | None = None,
    epoc: TDTExtractEpoc | None = None,
    overwrite: bool = False,
) -> TDTExtractSummary:
    """Write one pickle matching the object returned by ``tdt.read_block``.

    Pass ``streams`` to write any number of named streams. The legacy
    ``isosbestic``/``experimental``/``ttl`` arguments remain available for a
    conventional single-site block. ``path`` is the recording directory. Its
    only file is :data:`TDT_BLOCK_PICKLE_FILENAME`. The optional :mod:`tdt`
    dependency is required both to create and to unpickle this object; install
    the project with its ``tdt-export`` extra.
    """

    tdt = _require_tdt()
    normalized_metadata = _normalize_metadata(metadata)
    selected_streams = _select_streams(
        streams=streams,
        isosbestic=isosbestic,
        experimental=experimental,
        ttl=ttl,
    )
    normalized_streams = tuple(
        _normalize_stream(stream, f"stream {index}", tdt)
        for index, stream in enumerate(selected_streams, start=1)
    )
    _validate_stream_set(normalized_streams)
    reference_stream = normalized_streams[0]
    normalized_epoc = (
        None
        if epoc is None
        else _normalize_epoc(
            epoc,
            duration_seconds=reference_stream.data.size / reference_stream.fs,
            tdt=tdt,
        )
    )

    target = Path(path).expanduser()
    expected_suffix = (
        f"{normalized_metadata.subject_id}_{normalized_metadata.datetime_text}"
    )
    legacy_site_name = target.name.endswith(f"__{expected_suffix}")
    structured_block_name = target.name.startswith(
        f"{normalized_metadata.subject_id}__"
    ) and target.name.endswith(normalized_metadata.datetime_text)
    if (
        target.name != expected_suffix
        and not legacy_site_name
        and not structured_block_name
    ):
        raise TDTExtractError(
            "target directory must contain the subject and end with its "
            f"YYYYMMDD-HHMMSS datetime; got {target.name!r}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists; pass overwrite=True to replace it"
        )
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise TDTExtractError("an existing TDT block target must be a real directory")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.writing-", dir=target.parent)
    )
    pickle_path = temporary / TDT_BLOCK_PICKLE_FILENAME
    commit_warnings: tuple[str, ...] = ()
    try:
        block = _build_block(
            normalized_metadata,
            normalized_streams,
            normalized_epoc,
            tdt,
        )
        with pickle_path.open("wb") as file_out:
            pickle.dump(block, file_out, protocol=PICKLE_PROTOCOL)
        commit_warnings = _commit_directory(temporary, target, overwrite=overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return TDTExtractSummary(
        path=target,
        subject_id=normalized_metadata.subject_id,
        datetime=normalized_metadata.datetime_text,
        samples=int(reference_stream.data.size),
        fs=reference_stream.fs,
        files=(target / TDT_BLOCK_PICKLE_FILENAME,),
        warnings=commit_warnings,
    )


def write_tdt_extract(
    path: str | Path,
    *,
    isosbestic: TDTExtractStream,
    experimental: TDTExtractStream,
    metadata: TDTExtractMetadata,
    ttl: TDTExtractStream | None = None,
    epoc: TDTExtractEpoc | None = None,
    points_per_row: int = DEFAULT_POINTS_PER_ROW,
    include_packed_csv: bool = False,
    remainder_policy: Literal["error"] = "error",
    overwrite: bool = False,
) -> TDTExtractSummary:
    """Compatibility wrapper for :func:`write_tdt_block`.

    The old function incorrectly wrote post-extraction ChronoXIV artifacts.
    It now has the corrected one-pickle semantics. Obsolete packed-CSV
    arguments are accepted only when they do not request those extra files.
    """

    _normalize_points_per_row(points_per_row, remainder_policy)
    if include_packed_csv:
        raise TDTExtractError(
            "include_packed_csv is no longer supported; simulated TDT folders "
            "contain only tdt_block.pkl"
        )
    return write_tdt_block(
        path,
        isosbestic=isosbestic,
        experimental=experimental,
        metadata=metadata,
        ttl=ttl,
        epoc=epoc,
        overwrite=overwrite,
    )


def _require_tdt() -> ModuleType:
    try:
        import tdt
    except ImportError as error:
        raise TDTDependencyError(
            "TDT block export requires the optional 'tdt' package. "
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
        raise TDTExtractError("simulated TDT blocks support at most two subjects")
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

    block_name = _validate_text(metadata.block_name, "block_name")
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
        session_datetime=metadata.datetime,
        datetime_text=metadata.datetime.strftime("%Y%m%d-%H%M%S"),
        tank_dir=tank_dir,
        block_name=block_name,
    )


def _normalize_stream(
    stream: TDTExtractStream,
    label: str,
    tdt: ModuleType,
) -> _NormalizedStream:
    if not isinstance(stream, TDTExtractStream):
        raise TDTExtractError(f"{label} must be a TDTBlockStream")
    store_name = _validate_stream_name(stream.store_name, f"{label} store_name")
    if stream.code_name is None:
        try:
            code_name = _validate_store_code(store_name, f"{label} store_name")
        except TDTExtractError as error:
            raise TDTExtractError(
                f"{label} code_name is required when store_name is not "
                "exactly four CP437 bytes"
            ) from error
    else:
        code_name = _validate_store_code(stream.code_name, f"{label} code_name")
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
        raise TDTExtractError(f"{label} data is outside the finite float32 range")

    name = (
        store_name
        if stream.code_name is not None
        else str(tdt.fix_var_name(store_name))
    )
    if not name:
        raise TDTExtractError(f"{label} store_name has no valid serialized name")
    return _NormalizedStream(
        store_name=store_name,
        name=name,
        code_name=code_name,
        data=data,
        fs=fs,
        channel=channel,
        start_time=start_time,
    )


def _normalize_epoc(
    epoc: TDTExtractEpoc,
    *,
    duration_seconds: float,
    tdt: ModuleType,
) -> _NormalizedEpoc:
    name = _validate_store_code(epoc.name, "epoc name")
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
        variable_name=str(tdt.fix_var_name(name)),
        index=index,
        onset_seconds=onset,
        offset_seconds=offset,
    )


def _validate_stream_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TDTExtractError(f"{label} must be a string")
    try:
        encoded_name = value.encode("cp437")
    except UnicodeEncodeError as error:
        raise TDTExtractError(f"{label} must be CP437-encodable") from error
    if not encoded_name:
        raise TDTExtractError(f"{label} must not be empty")
    if b"\x00" in encoded_name or not value.isprintable():
        raise TDTExtractError(f"{label} must contain printable, non-NUL characters")
    return value


def _validate_store_code(value: object, label: str) -> str:
    code_name = _validate_stream_name(value, label)
    if len(code_name.encode("cp437")) != 4:
        raise TDTExtractError(f"{label} must encode to exactly four CP437 bytes")
    return code_name


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


def _select_streams(
    *,
    streams: Sequence[TDTExtractStream] | None,
    isosbestic: TDTExtractStream | None,
    experimental: TDTExtractStream | None,
    ttl: TDTExtractStream | None,
) -> tuple[TDTExtractStream, ...]:
    legacy_streams_supplied = any(
        stream is not None for stream in (isosbestic, experimental, ttl)
    )
    if streams is not None:
        if legacy_streams_supplied:
            raise TDTExtractError(
                "streams cannot be combined with isosbestic, experimental, or ttl"
            )
        if isinstance(streams, (str, bytes)):
            raise TDTExtractError("streams must be a sequence of TDTBlockStream values")
        selected = tuple(streams)
        if not selected:
            raise TDTExtractError("streams must contain at least one stream")
        return selected

    if isosbestic is None or experimental is None:
        raise TDTExtractError(
            "provide streams or both isosbestic and experimental streams"
        )
    selected = [isosbestic, experimental]
    if ttl is not None:
        selected.append(ttl)
    return tuple(selected)


def _validate_stream_set(streams: Sequence[_NormalizedStream]) -> None:
    names = [stream.name for stream in streams]
    if len(set(names)) != len(names):
        raise TDTExtractError("serialized stream names must be unique")

    samples = streams[0].data.size
    fs = streams[0].fs
    for stream in streams[1:]:
        if stream.data.size != samples:
            raise TDTExtractError("all selected streams must have equal lengths")
        if not np.isclose(stream.fs, fs, rtol=1e-12, atol=0.0):
            raise TDTExtractError(
                "all selected streams must have the same sampling rate"
            )


def _normalize_points_per_row(points_per_row: int, remainder_policy: str) -> int:
    if (
        not isinstance(points_per_row, Integral)
        or isinstance(points_per_row, bool)
        or points_per_row <= 0
    ):
        raise TDTExtractError("points_per_row must be a positive integer")
    if remainder_policy != "error":
        raise TDTExtractError("only remainder_policy='error' is supported")
    return int(points_per_row)


def _build_block(
    metadata: _NormalizedMetadata,
    streams: Sequence[_NormalizedStream],
    epoc: _NormalizedEpoc | None,
    tdt: ModuleType,
) -> object:
    block = tdt.StructType()
    block.epocs = tdt.StructType()
    block.snips = tdt.StructType()
    block.streams = tdt.StructType()
    block.scalars = tdt.StructType()
    block.info = _build_info(metadata, streams[0], tdt)

    for stream in streams:
        block.streams[stream.name] = _build_stream_type(stream, tdt)
    if epoc is not None:
        block.epocs[epoc.variable_name] = _build_epoc_type(epoc, tdt)

    if tuple(block.keys()) != _BLOCK_FIELD_NAMES:
        raise TDTExtractError(
            "installed tdt.StructType did not preserve the read_block contract"
        )
    return block


def _build_info(
    metadata: _NormalizedMetadata,
    stream: _NormalizedStream,
    tdt: ModuleType,
) -> object:
    duration_seconds = stream.data.size / stream.fs
    stop_date = metadata.session_datetime + timedelta(seconds=duration_seconds)
    tankpath = (
        metadata.tank_dir.parent
        if metadata.tank_dir.name == metadata.block_name
        else metadata.tank_dir
    )
    return tdt.StructType(
        tankpath=str(tankpath),
        blockname=metadata.block_name,
        start_date=metadata.session_datetime,
        utc_start_time=metadata.session_datetime.strftime("%H:%M:%S"),
        stop_date=stop_date,
        utc_stop_time=stop_date.strftime("%H:%M:%S"),
        duration=timedelta(seconds=duration_seconds),
        stream_channel=0,
        snip_channel=0,
    )


def _build_stream_type(stream: _NormalizedStream, tdt: ModuleType) -> object:
    struct = tdt.StructType(
        name=stream.name,
        code=np.uint32(int.from_bytes(stream.code_name.encode("cp437"), "little")),
        size=np.uint32(DEFAULT_POINTS_PER_ROW + 10),
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
            "installed tdt.StructType did not preserve the stream contract"
        )
    return struct


def _build_epoc_type(epoc: _NormalizedEpoc, tdt: ModuleType) -> object:
    return tdt.StructType(
        name=epoc.name,
        onset=epoc.onset_seconds,
        offset=epoc.offset_seconds,
        type="onset",
        type_str="epocs",
        data=epoc.index,
        dform=4,
        size=10,
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
                "an existing TDT block target must be a real directory"
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
        return (f"could not remove overwritten TDT block backup {backup}: {error}",)
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
    text = _validate_text(value, label)
    if text in {".", ".."} or Path(text).name != text:
        raise TDTExtractError(f"{label} must be one safe path component")
    return text


def _validate_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TDTExtractError(f"{label} must be a nonempty string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise TDTExtractError(f"{label} must not contain control line characters")
    return value


__all__ = [
    "DEFAULT_POINTS_PER_ROW",
    "PICKLE_PROTOCOL",
    "TDT_BLOCK_PICKLE_FILENAME",
    "TDTBlockEpoc",
    "TDTBlockError",
    "TDTBlockMetadata",
    "TDTBlockStream",
    "TDTBlockSummary",
    "TDTDependencyError",
    "TDTExtractEpoc",
    "TDTExtractError",
    "TDTExtractMetadata",
    "TDTExtractStream",
    "TDTExtractSummary",
    "write_tdt_block",
    "write_tdt_extract",
]
