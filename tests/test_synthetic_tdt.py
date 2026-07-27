from __future__ import annotations

import csv
import json
import pickle
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import tdt

from circadian_fiber_photometry import load_doric
from circadian_fiber_photometry.io import TDTExtractError
from circadian_fiber_photometry.simulation import (
    SyntheticDoricConfig,
    SyntheticTDTSubject,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    build_tdt_epocs_from_behavior_events,
    export_synthetic_tdt_extracts,
    generate_synthetic_doric,
)


def _generated_dataset(tmp_path: Path):
    path = tmp_path / "source.doric"
    config = SyntheticDoricConfig(
        series_count=2,
        session_duration_seconds=12.8,
        inter_series_gap_seconds=1787.2,
        fs=20.0,
        channel_count=2,
        seed=123,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("reward", channel=1, pulse_count=3),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig(
                "reward",
                start_seconds=(5.0,),
                series_numbers=(1, 2),
            ),
        ),
    )
    summary = generate_synthetic_doric(path, config)
    return config, summary, load_doric(path)


def test_export_synthetic_tdt_extracts_maps_series_channels_ttl_and_epocs(
    tmp_path: Path,
) -> None:
    config, source_summary, dataset = _generated_dataset(tmp_path)
    epocs = build_tdt_epocs_from_behavior_events(
        source_summary.ttl_behavior_events,
        name="PtAB",
        fs=config.fs,
        pulse_width_seconds=config.ttl_pulse_width_seconds,
    )
    output_root = tmp_path / "experiment"
    subjects = (
        SyntheticTDTSubject(
            subject_id="MouseA",
            channel_number=1,
            isosbestic_store_name="405A",
            experimental_store_name="470d",
            ttl_source="DIO01",
            ttl_store_name="Wav1",
            epocs_by_series=epocs,
        ),
        SyntheticTDTSubject(
            subject_id="MouseB",
            channel_number=2,
            isosbestic_store_name="45bA",
            experimental_store_name="47bd",
            ttl_source="DIO02",
            ttl_store_name="Wav2",
        ),
    )

    batch = export_synthetic_tdt_extracts(
        output_root,
        dataset,
        cohort="Cohort 1",
        subjects=subjects,
        base_datetime=datetime(2025, 5, 20, 16, 26, 27),
        tank_name_prefix="sim-tank",
    )

    assert batch.output_root == output_root
    assert batch.cohort == "Cohort 1"
    assert batch.extracted_count == 4
    assert [
        (record.subject_id, record.series_number, record.session_datetime)
        for record in batch.records
    ] == [
        ("MouseA", 1, datetime(2025, 5, 20, 16, 26, 27)),
        ("MouseB", 1, datetime(2025, 5, 20, 16, 26, 27)),
        ("MouseA", 2, datetime(2025, 5, 20, 16, 56, 27)),
        ("MouseB", 2, datetime(2025, 5, 20, 16, 56, 27)),
    ]

    first_path = (
        output_root
        / "Photometry"
        / "Cohort 1"
        / "Analysis"
        / "MouseA_20250520-162627"
    )
    second_subject_path = first_path.parent / "MouseB_20250520-162627"
    later_path = first_path.parent / "MouseA_20250520-165627"
    assert first_path.exists()
    assert second_subject_path.exists()
    assert later_path.exists()

    first_manifest = json.loads((first_path / "streams.json").read_text())
    second_manifest = json.loads(
        (second_subject_path / "streams.json").read_text()
    )
    assert first_manifest == {
        "order": "First",
        "ttl_stream": "Wav1",
        "iso_stream": "_405A",
        "exp_stream": "_470d",
        "epoc": "PtAB",
        "tank_dir": str(
            output_root
            / "Photometry"
            / "Cohort 1"
            / "TDT Binaries"
            / "sim-tank-20250520-162627"
        ),
        "num_subjects_in_tank": 2,
        "subject_ids": ["MouseA", "MouseB"],
        "subject_id": "MouseA",
        "datetime": "20250520-162627",
    }
    assert second_manifest["order"] == "Second"
    assert second_manifest["subject_ids"] == ["MouseA", "MouseB"]
    assert second_manifest["ttl_stream"] == "Wav2"
    assert second_manifest["epoc"] == "None"
    assert second_manifest["tank_dir"] == first_manifest["tank_dir"]

    with (first_path / "exp_stream.pkl").open("rb") as file_in:
        experimental = pickle.load(file_in)
    assert isinstance(experimental, tdt.StructType)
    np.testing.assert_allclose(
        experimental.data,
        dataset.calcium_465[:, 0, 0].astype(np.float32),
    )
    with (first_path / "ttl_stream.pkl").open("rb") as file_in:
        ttl_stream = pickle.load(file_in)
    np.testing.assert_array_equal(
        ttl_stream.data,
        dataset.digital_io["DIO01"][:, 0].astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(ttl_stream.data == 1.0),
        np.concatenate(
            [
                np.arange(sample, sample + 1)
                for sample in source_summary.ttl_pulse_sample_indices[(1, 1)]
            ]
        ),
    )
    with (first_path / "epoc.csv").open(newline="") as file_in:
        epoc_rows = list(csv.DictReader(file_in))
    assert epoc_rows == [{"index": "3.0", "onset": "5.0", "offset": "5.25"}]
    assert batch.records[0].epoc is epocs[1]
    assert batch.records[0].ttl_source == "DIO01"


def test_build_tdt_epocs_filters_behavior_codes(tmp_path: Path) -> None:
    config, source_summary, _dataset = _generated_dataset(tmp_path)

    all_epocs = build_tdt_epocs_from_behavior_events(
        source_summary.ttl_behavior_events,
        name="PtAB",
        fs=config.fs,
        pulse_width_seconds=config.ttl_pulse_width_seconds,
    )
    omitted = build_tdt_epocs_from_behavior_events(
        source_summary.ttl_behavior_events,
        name="PtAB",
        fs=config.fs,
        pulse_width_seconds=config.ttl_pulse_width_seconds,
        code_names=("not-present",),
    )

    assert set(all_epocs) == {1, 2}
    assert omitted == {}
    np.testing.assert_array_equal(all_epocs[1].index, [3.0])
    np.testing.assert_allclose(all_epocs[1].onset_seconds, [5.0])
    np.testing.assert_allclose(all_epocs[1].offset_seconds, [5.25])


def test_export_synthetic_tdt_extracts_supports_series_selection(
    tmp_path: Path,
) -> None:
    _config, _source_summary, dataset = _generated_dataset(tmp_path)

    batch = export_synthetic_tdt_extracts(
        tmp_path / "experiment",
        dataset,
        cohort="Cohort 1",
        subjects=(SyntheticTDTSubject("MouseA", channel_number=1),),
        base_datetime=datetime(2025, 1, 1),
        series_numbers=(2,),
    )

    assert batch.extracted_count == 1
    assert batch.records[0].series_number == 2
    assert batch.records[0].session_datetime == datetime(2025, 1, 1, 0, 30)
    assert batch.records[0].summary.path.name == "MouseA_20250101-003000"


def test_export_synthetic_tdt_extracts_validates_subject_mappings(
    tmp_path: Path,
) -> None:
    _config, _source_summary, dataset = _generated_dataset(tmp_path)
    common = {
        "output_root": tmp_path / "experiment",
        "dataset": dataset,
        "cohort": "Cohort 1",
        "base_datetime": datetime(2025, 1, 1),
    }

    with pytest.raises(TDTExtractError, match="one or two"):
        export_synthetic_tdt_extracts(subjects=(), **common)
    with pytest.raises(TDTExtractError, match="must be unique"):
        export_synthetic_tdt_extracts(
            subjects=(
                SyntheticTDTSubject("MouseA", 1),
                SyntheticTDTSubject("MouseB", 1),
            ),
            **common,
        )
    with pytest.raises(TDTExtractError, match="both be set"):
        export_synthetic_tdt_extracts(
            subjects=(
                SyntheticTDTSubject(
                    "MouseA",
                    1,
                    ttl_source="DIO01",
                ),
            ),
            **common,
        )
    with pytest.raises(TDTExtractError, match="not divisible"):
        export_synthetic_tdt_extracts(
            subjects=(SyntheticTDTSubject("MouseA", 1),),
            points_per_row=100,
            **common,
        )


def test_export_synthetic_tdt_extracts_preflights_existing_targets(
    tmp_path: Path,
) -> None:
    _config, _source_summary, dataset = _generated_dataset(tmp_path)
    output_root = tmp_path / "experiment"
    target = (
        output_root
        / "Photometry"
        / "Cohort 1"
        / "Analysis"
        / "MouseA_20250101-000000"
    )
    target.mkdir(parents=True)
    marker = target / "marker.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError):
        export_synthetic_tdt_extracts(
            output_root,
            dataset,
            cohort="Cohort 1",
            subjects=(SyntheticTDTSubject("MouseA", 1),),
            base_datetime=datetime(2025, 1, 1),
        )

    assert marker.read_text() == "preserve"


def test_tdt_extract_simulation_marimo_example_imports() -> None:
    pytest.importorskip("marimo")
    example_path = (
        Path(__file__).parents[1] / "examples" / "tdt_extract_simulation.py"
    )

    namespace = runpy.run_path(example_path)

    assert namespace["app"].__class__.__module__.startswith("marimo")
