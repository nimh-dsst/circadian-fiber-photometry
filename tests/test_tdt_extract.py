from __future__ import annotations

import builtins
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import tdt
from tdt.TDTbin2py import code_to_name

import circadian_fiber_photometry.io.tdt as tdt_io
from circadian_fiber_photometry.io import (
    TDTDependencyError,
    TDTExtractEpoc,
    TDTExtractError,
    TDTExtractMetadata,
    TDTExtractStream,
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
    values = (
        np.linspace(0.0, 1.0, 256, dtype=np.float64)
        if data is None
        else data
    )
    return TDTExtractStream(
        store_name=store_name,
        data=values,
        fs=fs,
        channel=channel,
    )


def _target(tmp_path: Path, subject_id: str = "MouseA") -> Path:
    return tmp_path / f"{subject_id}_20250102-030405"


def test_write_tdt_extract_matches_structtype_and_packed_csv_contract(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)

    summary = write_tdt_extract(
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
    assert {path.name for path in summary.files} == {
        "streams.json",
        "iso_stream.pkl",
        "iso_stream.csv",
        "exp_stream.pkl",
        "exp_stream.csv",
    }
    assert {path.name for path in target.iterdir()} == {
        path.name for path in summary.files
    }

    manifest = json.loads((target / "streams.json").read_text())
    assert manifest == {
        "order": "First",
        "ttl_stream": "None",
        "iso_stream": "_405A",
        "exp_stream": "_470d",
        "epoc": "None",
        "tank_dir": "/synthetic/TDT Binaries/tank-20250102-030405",
        "num_subjects_in_tank": 1,
        "subject_ids": ["MouseA"],
        "subject_id": "MouseA",
        "datetime": "20250102-030405",
    }

    pickle_bytes = (target / "exp_stream.pkl").read_bytes()
    stream = pickle.loads(pickle_bytes)
    assert isinstance(stream, tdt.StructType)
    assert len(stream) == 0
    assert dict(stream) == {}
    assert list(stream.keys()) == STREAM_FIELDS
    assert stream["data"] is stream.data
    assert stream.get("data") is None
    assert stream.name == "_470d"
    assert isinstance(stream.code, np.uint32)
    assert code_to_name(stream.code) == "470d"
    assert isinstance(stream.size, np.uint32)
    assert stream.size == 138
    assert isinstance(stream.type, np.uint32)
    assert stream.type == tdt.EVTYPE_STREAM
    assert stream.type_str == "streams"
    assert isinstance(stream.ucf, np.bool_)
    assert not stream.ucf
    assert isinstance(stream.fs, np.float64)
    assert stream.fs == pytest.approx(20.0)
    assert isinstance(stream.dform, np.uint32)
    assert stream.dform == tdt.DFORM_FLOAT
    assert isinstance(stream.start_time, np.float64)
    assert stream.start_time == pytest.approx(0.0)
    assert stream.data.dtype == np.float32
    np.testing.assert_array_equal(stream.data, np.arange(256, dtype=np.float32))
    assert stream.channel == [1]
    assert pickle_bytes[:2] == b"\x80\x04"
    assert b"tdt" in pickle_bytes[:64]
    assert b"StructType" in pickle_bytes[:64]

    with (target / "exp_stream.csv").open(newline="") as file_in:
        rows = list(csv.reader(file_in))
    expected_header = [
        "BLOCK",
        "EVENT",
        "TIME",
        "CHAN",
        "Sampling_Freq",
        "NumOfPoints",
        *(f"D{index}" for index in range(128)),
    ]
    assert rows[0] == expected_header
    assert len(rows) == 3
    assert rows[1][:6] == [
        "tank-20250102-030405",
        "_470d",
        "0.0",
        "1",
        "20.0",
        "128",
    ]
    assert float(rows[2][2]) == pytest.approx(128 / 20.0)
    reconstructed = np.asarray(
        [[float(value) for value in row[6:]] for row in rows[1:]],
        dtype=np.float32,
    ).reshape(-1)
    np.testing.assert_array_equal(reconstructed, stream.data)


def test_write_tdt_extract_supports_ttl_and_epoc(tmp_path: Path) -> None:
    target = _target(tmp_path)
    ttl_values = np.zeros(256, dtype=float)
    ttl_values[20:25] = 1
    epoc = TDTExtractEpoc(
        name="PtAB",
        index=np.array([1, 2]),
        onset_seconds=np.array([1.0, 5.0]),
        offset_seconds=np.array([1.25, 5.5]),
    )

    summary = write_tdt_extract(
        target,
        isosbestic=_stream("Iso1"),
        experimental=_stream("Ca1_"),
        ttl=_stream("Wav1", data=ttl_values),
        epoc=epoc,
        metadata=_metadata(),
    )

    assert {path.name for path in summary.files} == {
        "streams.json",
        "iso_stream.pkl",
        "iso_stream.csv",
        "exp_stream.pkl",
        "exp_stream.csv",
        "ttl_stream.pkl",
        "ttl_stream.csv",
        "epoc.csv",
    }
    manifest = json.loads((target / "streams.json").read_text())
    assert manifest["ttl_stream"] == "Wav1"
    assert manifest["epoc"] == "PtAB"
    with (target / "ttl_stream.pkl").open("rb") as file_in:
        ttl_stream = pickle.load(file_in)
    np.testing.assert_array_equal(ttl_stream.data, ttl_values.astype(np.float32))
    with (target / "epoc.csv").open(newline="") as file_in:
        assert list(csv.reader(file_in)) == [
            ["index", "onset", "offset"],
            ["1.0", "1.0", "1.25"],
            ["2.0", "5.0", "5.5"],
        ]


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (_stream("abc"), "exactly four"),
        (_stream("ab😀d"), "CP437"),
        (_stream("abcd", data=np.array([])), "must not be empty"),
        (_stream("abcd", data=np.zeros((2, 128))), "one-dimensional"),
        (_stream("abcd", data=np.array([1.0, np.nan])), "NaN"),
        (_stream("abcd", data=np.array([1.0 + 2.0j])), "real numeric"),
        (_stream("abcd", fs=0.0), "greater than zero"),
        (_stream("abcd", channel=0), "positive integer"),
    ],
)
def test_write_tdt_extract_rejects_invalid_streams(
    tmp_path: Path,
    stream: TDTExtractStream,
    message: str,
) -> None:
    with pytest.raises(TDTExtractError, match=message):
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=stream,
            experimental=_stream("wxyz"),
            metadata=_metadata(),
        )
    assert not _target(tmp_path).exists()


def test_write_tdt_extract_rejects_stream_contract_mismatches(
    tmp_path: Path,
) -> None:
    with pytest.raises(TDTExtractError, match="equal lengths"):
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_", data=np.zeros(128)),
            metadata=_metadata(),
        )
    with pytest.raises(TDTExtractError, match="same sampling rate"):
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_", fs=10.0),
            metadata=_metadata(),
        )
    with pytest.raises(TDTExtractError, match="names must be unique"):
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=_stream("a-b!"),
            experimental=_stream("a_b_"),
            metadata=_metadata(),
        )


def test_write_tdt_extract_rejects_remainder_and_invalid_epoc(
    tmp_path: Path,
) -> None:
    with pytest.raises(TDTExtractError, match="not divisible"):
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=_stream("Iso1", data=np.zeros(255)),
            experimental=_stream("Ca1_", data=np.zeros(255)),
            metadata=_metadata(),
        )
    invalid_epoc = TDTExtractEpoc(
        name="PtAB",
        index=np.array([1.0]),
        onset_seconds=np.array([2.0]),
        offset_seconds=np.array([1.0]),
    )
    with pytest.raises(TDTExtractError, match="greater than or equal"):
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            epoc=invalid_epoc,
            metadata=_metadata(),
        )


def test_write_tdt_extract_validates_subject_order_and_target_name(
    tmp_path: Path,
) -> None:
    metadata = _metadata(
        subject_id="MouseB",
        subject_ids=("MouseA", "MouseB"),
        order="First",
    )
    with pytest.raises(TDTExtractError, match="inconsistent"):
        write_tdt_extract(
            _target(tmp_path, "MouseB"),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=metadata,
        )

    with pytest.raises(TDTExtractError, match="target directory must be named"):
        write_tdt_extract(
            tmp_path / "wrong-name",
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )


def test_write_tdt_extract_protects_and_explicitly_overwrites_target(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    write_tdt_extract(
        target,
        isosbestic=_stream("Iso1"),
        experimental=_stream("Ca1_"),
        metadata=_metadata(),
    )
    marker = target / "old-marker.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError):
        write_tdt_extract(
            target,
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )
    assert marker.read_text() == "preserve"

    summary = write_tdt_extract(
        target,
        isosbestic=_stream("Iso1"),
        experimental=_stream("Ca1_"),
        metadata=_metadata(),
        overwrite=True,
    )
    assert not marker.exists()
    assert (summary.path / "streams.json").exists()
    assert not list(tmp_path.glob(".*.backup-*"))


def test_failed_write_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)

    def fail_csv(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated CSV failure")

    monkeypatch.setattr(tdt_io, "_write_packed_csv", fail_csv)
    with pytest.raises(OSError, match="simulated CSV failure"):
        write_tdt_extract(
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
        write_tdt_extract(
            _target(tmp_path),
            isosbestic=_stream("Iso1"),
            experimental=_stream("Ca1_"),
            metadata=_metadata(),
        )
