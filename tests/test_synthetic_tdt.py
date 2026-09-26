from __future__ import annotations

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
    export_synthetic_tdt_blocks,
    generate_synthetic_doric,
)


def _generated_dataset(tmp_path: Path):
    path = tmp_path / "source.doric"
    config = SyntheticDoricConfig(
        series_count=2,
        session_duration_seconds=12.75,
        inter_series_gap_seconds=1787.25,
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


def _load_block(path: Path):
    with (path / "tdt_block.pkl").open("rb") as file_in:
        return pickle.load(file_in)


def test_export_synthetic_tdt_blocks_maps_series_channels_ttl_and_epocs(
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
            site="vta-left",
        ),
        SyntheticTDTSubject(
            subject_id="MouseB",
            channel_number=2,
            isosbestic_store_name="45bA",
            experimental_store_name="47bd",
            ttl_source="DIO02",
            ttl_store_name="Wav2",
            site="vta-right",
        ),
    )

    batch = export_synthetic_tdt_blocks(
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
        (record.subject_id, record.site, record.series_number)
        for record in batch.records
    ] == [
        ("MouseA", "vta-left", 1),
        ("MouseB", "vta-right", 1),
        ("MouseA", "vta-left", 2),
        ("MouseB", "vta-right", 2),
    ]

    inbox = output_root / "Photometry" / "Cohort 1" / "Simulated TDT"
    first_path = inbox / "MouseA__vta-left__U-20250520-162627"
    second_subject_path = inbox / "MouseB__vta-right__U-20250520-162627"
    later_path = inbox / "MouseA__vta-left__U-20250520-165627"
    assert first_path.exists()
    assert second_subject_path.exists()
    assert later_path.exists()
    assert {path.name for path in first_path.iterdir()} == {"tdt_block.pkl"}

    first = _load_block(first_path)
    second = _load_block(second_subject_path)
    assert isinstance(first, tdt.StructType)
    assert list(first.keys()) == ["epocs", "snips", "streams", "scalars", "info"]
    assert list(first.streams.keys()) == ["_405A", "_470d", "Wav1"]
    assert list(second.streams.keys()) == ["_45bA", "_47bd", "Wav2"]
    np.testing.assert_allclose(
        first.streams._470d.data,
        dataset.calcium_465[:, 0, 0].astype(np.float32),
    )
    np.testing.assert_array_equal(
        first.streams.Wav1.data,
        dataset.digital_io["DIO01"][:, 0].astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(first.streams.Wav1.data == 1.0),
        np.concatenate(
            [
                np.arange(sample, sample + 1)
                for sample in source_summary.ttl_pulse_sample_indices[(1, 1)]
            ]
        ),
    )
    assert list(first.epocs.keys()) == ["PtAB"]
    np.testing.assert_array_equal(first.epocs.PtAB.data, [3.0])
    np.testing.assert_allclose(first.epocs.PtAB.onset, [5.0])
    np.testing.assert_allclose(first.epocs.PtAB.offset, [5.25])
    assert list(second.epocs.keys()) == []
    assert first.info.blockname == "sim-tank-20250520-162627"
    assert first.info.start_date == datetime(2025, 5, 20, 16, 26, 27)


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


def test_export_synthetic_tdt_blocks_supports_series_selection(
    tmp_path: Path,
) -> None:
    _config, _source_summary, dataset = _generated_dataset(tmp_path)

    batch = export_synthetic_tdt_blocks(
        tmp_path / "experiment",
        dataset,
        cohort="Cohort 1",
        subjects=(SyntheticTDTSubject("MouseA", channel_number=1, site="vta"),),
        base_datetime=datetime(2025, 1, 1),
        series_numbers=(2,),
    )

    assert batch.extracted_count == 1
    assert batch.records[0].series_number == 2
    assert batch.records[0].session_datetime == datetime(2025, 1, 1, 0, 30)
    assert batch.records[0].summary.path.name == "MouseA__vta__U-20250101-003000"


def test_export_synthetic_tdt_blocks_validates_subject_mappings(
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
        export_synthetic_tdt_blocks(subjects=(), **common)
    with pytest.raises(TDTExtractError, match="must be unique"):
        export_synthetic_tdt_blocks(
            subjects=(
                SyntheticTDTSubject("MouseA", 1),
                SyntheticTDTSubject("MouseB", 1),
            ),
            **common,
        )
    with pytest.raises(TDTExtractError, match="both be set"):
        export_synthetic_tdt_blocks(
            subjects=(SyntheticTDTSubject("MouseA", 1, ttl_source="DIO01"),),
            **common,
        )
    with pytest.raises(TDTExtractError, match="positive integer"):
        export_synthetic_tdt_blocks(
            subjects=(SyntheticTDTSubject("MouseA", 1),),
            points_per_row=0,
            **common,
        )


def test_export_synthetic_tdt_blocks_preflights_existing_targets(
    tmp_path: Path,
) -> None:
    _config, _source_summary, dataset = _generated_dataset(tmp_path)
    output_root = tmp_path / "experiment"
    target = (
        output_root
        / "Photometry"
        / "Cohort 1"
        / "Simulated TDT"
        / "MouseA__vta__U-20250101-000000"
    )
    target.mkdir(parents=True)
    marker = target / "marker.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError):
        export_synthetic_tdt_blocks(
            output_root,
            dataset,
            cohort="Cohort 1",
            subjects=(SyntheticTDTSubject("MouseA", 1, site="vta"),),
            base_datetime=datetime(2025, 1, 1),
        )

    assert marker.read_text() == "preserve"


def test_tdt_block_simulation_marimo_example_imports() -> None:
    pytest.importorskip("marimo")
    example_path = Path(__file__).parents[1] / "examples" / "tdt_extract_simulation.py"

    namespace = runpy.run_path(example_path)

    assert namespace["app"].__class__.__module__.startswith("marimo")


def test_chronoxiv_generator_uses_multi_site_folder_regex() -> None:
    generator_path = (
        Path(__file__).parents[1] / "examples" / "generate_chronoxiv_acc_dataset.py"
    )
    namespace = runpy.run_path(generator_path)

    name = namespace["_target_name"](
        "Subject01",
        datetime(2026, 1, 6, 9, 0, 0),
    )
    assert name == "Subject01__acc-left+acc-right__M-20260106-090000"
    match = namespace["TDT_BLOCK_FOLDER_RE"].fullmatch(name)
    assert match is not None
    assert match.groupdict() == {
        "subjects": "Subject01",
        "sites": "acc-left+acc-right",
        "sexes": "M",
        "datetime": "20260106-090000",
    }
    assert (
        namespace["_target_name"](
            "Subject06",
            datetime(2026, 1, 6, 9, 0, 0),
        )
        == "Subject06__acc-left+acc-right__F-20260106-090000"
    )
