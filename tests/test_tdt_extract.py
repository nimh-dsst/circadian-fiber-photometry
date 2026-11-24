from __future__ import annotations

import builtins
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tdt
from tdt.TDTbin2py import code_to_name

import circadian_fiber_photometry.io.tdt as tdt_io
from circadian_fiber_photometry.io import (
    TDT_BLOCK_PICKLE_FILENAME,
    TDTDependencyError,
    TDTExtractEpoc,
    TDTExtractError,
    TDTExtractMetadata,
    TDTExtractStream,
    write_tdt_block,
    write_tdt_extract,
)

STREAM_FIELDS = [
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
]


def _metadata(
    *,
    subject_id: str = "MouseA",
    subject_ids: tuple[str, ...] = ("MouseA",),
    order: str = "First",
) -> TDTExtractMetadata:
    return TDTExtractMetadata(
        subject_id=subject_id,
        subject_ids=subject_ids,
        order=order,
        datetime=datetime(2025, 1, 2, 3, 4, 5),
        tank_dir=Path("/synthetic/TDT Binaries/tank-20250102-030405"),
        block_name="tank-20250102-030405",
    )


def _stream(
    store_name: str,
    *,
    data: np.ndarray | None = None,
    fs: float = 20.0,
    channel: int = 1,
) -> TDTExtractStream:
    values = np.linspace(0.0, 1.0, 256, dtype=np.float64) if data is None else data
    return TDTExtractStream(
        store_name=store_name,
        data=values,
        fs=fs,
        channel=channel,
    )


def _target(tmp_path: Path, subject_id: str = "MouseA") -> Path:
    return tmp_path / f"vta__{subject_id}_20250102-030405"


def _load_block(path: Path):
    with (path / TDT_BLOCK_PICKLE_FILENAME).open("rb") as file_in:
        return pickle.load(file_in)


def test_write_tdt_block_matches_read_block_structtype_contract(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)

    summary = write_tdt_block(
        target,
        isosbestic=_stream("405A"),
        experimental=_stream("470d", data=np.arange(256, dtype=np.float64)),
        metadata=_metadata(),
    )

    assert summary.path == target
    assert summary.subject_id == "MouseA"
    assert summary.datetime == "20250102-030405"
    assert summary.samples == 256
    assert summary.fs == pytest.approx(20.0)
    assert summary.warnings == ()
    assert summary.files == (target / TDT_BLOCK_PICKLE_FILENAME,)
    assert summary.block_file == target / TDT_BLOCK_PICKLE_FILENAME
    assert {path.name for path in target.iterdir()} == {TDT_BLOCK_PICKLE_FILENAME}

    pickle_bytes = summary.block_file.read_bytes()
    block = pickle.loads(pickle_bytes)
    assert isinstance(block, tdt.StructType)
    assert list(block.keys()) == ["epocs", "snips", "streams", "scalars", "info"]
    assert isinstance(block.epocs, tdt.StructType)
    assert isinstance(block.snips, tdt.StructType)
    assert isinstance(block.streams, tdt.StructType)
    assert isinstance(block.scalars, tdt.StructType)
    assert isinstance(block.info, tdt.StructType)
    assert list(block.epocs.keys()) == []
    assert list(block.streams.keys()) == ["_405A", "_470d"]

    stream = block.streams._470d
    assert isinstance(stream, tdt.StructType)
    assert len(stream) == 0
    assert dict(stream) == {}
    assert list(stream.keys()) == STREAM_FIELDS
    assert stream["data"] is stream.data
    assert stream.get("data") is None
    assert stream.name == "_470d"
    assert isinstance(stream.code, np.uint32)
    assert code_to_name(stream.code) == "470d"
    assert stream.size == 138
    assert stream.type == tdt.EVTYPE_STREAM
    assert stream.type_str == "streams"
    assert not stream.ucf
    assert stream.fs == pytest.approx(20.0)
    assert stream.dform == tdt.DFORM_FLOAT
    assert stream.start_time == pytest.approx(0.0)
    assert stream.data.dtype == np.float32
    np.testing.assert_array_equal(stream.data, np.arange(256, dtype=np.float32))
    assert stream.channel == [1]

    assert list(block.info.keys()) == [
        "tankpath",
        "blockname",
        "start_date",
        "utc_start_time",
        "stop_date",
        "utc_stop_time",
        "duration",
        "stream_channel",
        "snip_channel",
    ]
    assert block.info.tankpath == "/synthetic/TDT Binaries"
    assert block.info.blockname == "tank-20250102-030405"
    assert block.info.start_date == datetime(2025, 1, 2, 3, 4, 5)
    assert block.info.stop_date == datetime(2025, 1, 2, 3, 4, 17, 800000)
    assert block.info.duration == timedelta(seconds=12.8)
    assert block.info.stream_channel == 0
    assert pickle_bytes[:2] == b"\x80\x04"
    assert b"tdt" in pickle_bytes[:64]
    assert b"StructType" in pickle_bytes[:64]


def test_write_tdt_block_nests_ttl_and_epoc(tmp_path: Path) -> None:
    ttl_values = np.zeros(256, dtype=float)
    ttl_values[20:25] = 1
    epoc = TDTExtractEpoc(
        name="PtAB",
        index=np.array([1, 2]),
        onset_seconds=np.array([1.0, 5.0]),
        offset_seconds=np.array([1.25, 5.5]),
    )

    summary = write_tdt_block(
        _target(tmp_path),
        isosbestic=_stream("Iso1"),
        experimental=_stream("Ca1_"),
        ttl=_stream("Wav1", data=ttl_values),
        epoc=epoc,
        metadata=_metadata(),
    )

    assert len(summary.files) == 1
    block = _load_block(summary.path)
    assert list(block.streams.keys()) == ["Iso1", "Ca1_", "Wav1"]
    np.testing.assert_array_equal(
        block.streams.Wav1.data,
        ttl_values.astype(np.float32),
    )
    assert list(block.epocs.keys()) == ["PtAB"]
    assert list(block.epocs.PtAB.keys()) == [
        "name",
        "onset",
        "offset",
        "type",
        "type_str",
        "data",
        "dform",
        "size",
    ]
    assert block.epocs.PtAB.name == "PtAB"
    assert block.epocs.PtAB.type == "onset"
    assert block.epocs.PtAB.type_str == "epocs"
    np.testing.assert_array_equal(block.epocs.PtAB.data, [1.0, 2.0])
    np.testing.assert_array_equal(block.epocs.PtAB.onset, [1.0, 5.0])
    np.testing.assert_array_equal(block.epocs.PtAB.offset, [1.25, 5.5])


def test_write_tdt_block_supports_shared_ttl_and_two_named_sites(
    tmp_path: Path,
) -> None:
    target = tmp_path / "MouseA_20250102-030405"
    names_and_codes = (
        ("TTL", "TTL1"),
        ("405_left", "4L05"),
        ("405_right", "4R05"),
        ("470_left", "4L70"),
        ("470_right", "4R70"),
    )

    summary = write_tdt_block(
        target,
        streams=tuple(
            TDTExtractStream(
                name,
                np.arange(256, dtype=float),
                fs=20.0,
                code_name=code,
            )
            for name, code in names_and_codes
        ),
        metadata=_metadata(),
    )

    block = _load_block(summary.path)
    assert list(block.streams.keys()) == [name for name, _code in names_and_codes]
    assert [block.streams[name].name for name, _code in names_and_codes] == [
        name for name, _code in names_and_codes
    ]
    assert [
        code_to_name(block.streams[name].code) for name, _code in names_and_codes
    ] == [code for _name, code in names_and_codes]


def test_write_tdt_block_rejects_mixed_stream_apis(tmp_path: Path) -> None:
    with pytest.raises(TDTExtractError, match="cannot be combined"):
        write_tdt_block(
            _target(tmp_path),
            streams=(_stream("Iso1"),),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )


def test_compatibility_writer_has_corrected_one_pickle_semantics(
    tmp_path: Path,
) -> None:
    summary = write_tdt_extract(
        _target(tmp_path),
        isosbestic=_stream("Iso1", data=np.zeros(255)),
        experimental=_stream("Ca1_", data=np.zeros(255)),
        metadata=_metadata(),
    )
    assert summary.files == (summary.path / TDT_BLOCK_PICKLE_FILENAME,)

    with pytest.raises(TDTExtractError, match="no longer supported"):
        write_tdt_extract(
            tmp_path / "MouseA_20250102-030405",
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
            include_packed_csv=True,
        )


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (_stream("abc"), "code_name is required"),
        (_stream("ab😀d"), "CP437"),
        (_stream("abcd", data=np.array([])), "must not be empty"),
        (_stream("abcd", data=np.zeros((2, 128))), "one-dimensional"),
        (_stream("abcd", data=np.array([1.0, np.nan])), "NaN"),
        (_stream("abcd", data=np.array([1.0 + 2.0j])), "real numeric"),
        (_stream("abcd", fs=0.0), "greater than zero"),
        (_stream("abcd", channel=0), "positive integer"),
    ],
)
def test_write_tdt_block_rejects_invalid_streams(
    tmp_path: Path,
    stream: TDTExtractStream,
    message: str,
) -> None:
    with pytest.raises(TDTExtractError, match=message):
        write_tdt_block(
            _target(tmp_path),
            isosbestic=stream,
            experimental=_stream("wxyz"),
            metadata=_metadata(),
        )
    assert not _target(tmp_path).exists()


def test_write_tdt_block_rejects_contract_mismatches_and_invalid_epoc(
    tmp_path: Path,
) -> None:
    with pytest.raises(TDTExtractError, match="equal lengths"):
        write_tdt_block(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_", data=np.zeros(128)),
            metadata=_metadata(),
        )
    with pytest.raises(TDTExtractError, match="same sampling rate"):
        write_tdt_block(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_", fs=10.0),
            metadata=_metadata(),
        )
    with pytest.raises(TDTExtractError, match="names must be unique"):
        write_tdt_block(
            _target(tmp_path),
            isosbestic=_stream("a-b!"),
            experimental=_stream("a_b_"),
            metadata=_metadata(),
        )

    invalid_epoc = TDTExtractEpoc(
        name="PtAB",
        index=np.array([1.0]),
        onset_seconds=np.array([2.0]),
        offset_seconds=np.array([1.0]),
    )
    with pytest.raises(TDTExtractError, match="greater than or equal"):
        write_tdt_block(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            epoc=invalid_epoc,
            metadata=_metadata(),
        )


def test_write_tdt_block_validates_subject_order_and_target_name(
    tmp_path: Path,
) -> None:
    metadata = _metadata(
        subject_id="MouseB",
        subject_ids=("MouseA", "MouseB"),
        order="First",
    )
    with pytest.raises(TDTExtractError, match="inconsistent"):
        write_tdt_block(
            _target(tmp_path, "MouseB"),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=metadata,
        )

    with pytest.raises(TDTExtractError, match="target directory must contain"):
        write_tdt_block(
            tmp_path / "wrong-name",
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )


def test_write_tdt_block_protects_and_explicitly_overwrites_target(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    write_tdt_block(
        target,
        isosbestic=_stream("Iso1"),
        experimental=_stream("Ca1_"),
        metadata=_metadata(),
    )
    marker = target / "old-marker.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError):
        write_tdt_block(
            target,
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )
    assert marker.read_text() == "preserve"

    summary = write_tdt_block(
        target,
        isosbestic=_stream("Iso1"),
        experimental=_stream("Ca1_"),
        metadata=_metadata(),
        overwrite=True,
    )
    assert not marker.exists()
    assert summary.block_file.exists()
    assert not list(tmp_path.glob(".*.backup-*"))


def test_failed_write_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)

    def fail_block(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated block failure")

    monkeypatch.setattr(tdt_io, "_build_block", fail_block)
    with pytest.raises(OSError, match="simulated block failure"):
        write_tdt_block(
            target,
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".*.writing-*"))


def test_writer_reports_missing_optional_tdt_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_tdt(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "tdt":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_tdt)
    with pytest.raises(TDTDependencyError, match="tdt-export"):
        write_tdt_block(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )
