from __future__ import annotations

import h5py
import numpy as np
import pytest

from circadian_fiber_photometry import (
    analyze_sessions,
    list_analyses,
    load_doric,
    run_analysis,
)
from circadian_fiber_photometry.io import DoricFileError
from circadian_fiber_photometry.simulation import (
    SyntheticDoricConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    generate_synthetic_doric,
)


def _config(**overrides: object) -> SyntheticDoricConfig:
    values = {
        "series_count": 3,
        "session_duration_seconds": 20.0,
        "inter_series_gap_seconds": 1780.0,
        "fs": 20.0,
        "channel_count": 2,
        "seed": 456,
    }
    values.update(overrides)
    return SyntheticDoricConfig(**values)


def test_load_doric_imports_generated_synthetic_file(tmp_path) -> None:
    path = tmp_path / "synthetic.doric"
    config = _config(
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("reward", channel=1, pulse_count=3),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig(
                "reward",
                start_seconds=(5.0,),
                series_numbers=(1,),
            ),
        ),
    )
    summary = generate_synthetic_doric(path, config)

    dataset = load_doric(path)

    assert dataset.path == path
    assert dataset.fs == pytest.approx(config.fs)
    assert dataset.series_names == ("Series0001", "Series0002", "Series0003")
    assert dataset.channel_names == ("AIN01", "AIN02")
    assert dataset.isosbestic_405.shape == (
        summary.samples_per_series,
        config.channel_count,
        config.series_count,
    )
    assert dataset.calcium_465.shape == dataset.isosbestic_405.shape
    assert dataset.timestamps.shape == (
        summary.samples_per_series,
        config.series_count,
    )
    np.testing.assert_allclose(
        dataset.session_start_times,
        summary.session_start_times,
    )
    assert dataset.metadata["attrs"]["SoftwareName"] == "Doric Neuroscience Studio"
    assert dataset.digital_io["DIO01"].shape == (
        summary.samples_per_series,
        config.series_count,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(dataset.digital_io["DIO01"][:, 0] == 1.0)[:3],
        summary.ttl_pulse_sample_indices[(1, 1)],
    )

    with h5py.File(path, "r") as h5_file:
        expected = h5_file[
            "DataAcquisition/FPConsole/Signals/Series0002/"
            "AIN02xAOUT02-LockIn/Values"
        ][:]
    np.testing.assert_allclose(dataset.calcium_465[:, 1, 1], expected)


def test_loaded_doric_dataset_can_feed_analysis(tmp_path) -> None:
    path = tmp_path / "analysis.doric"
    config = _config(series_count=4, seed=99)
    generate_synthetic_doric(path, config)

    dataset = load_doric(path)
    direct = analyze_sessions(
        dataset.isosbestic_405,
        dataset.calcium_465,
        fs=dataset.fs,
        interval_hours=0.5,
        weight_fit=False,
    )
    registered = run_analysis(
        dataset,
        analysis="tonic",
        config={"interval_hours": 0.5, "weight_fit": False},
    )

    assert direct.dff.shape == dataset.isosbestic_405.shape
    assert registered.name == "tonic"
    assert registered.arrays["dff"].shape == dataset.isosbestic_405.shape
    assert registered.arrays["level_tonic"].shape == (
        config.channel_count,
        config.series_count,
    )


def test_analysis_registry_lists_expected_categories() -> None:
    assert {spec.name for spec in list_analyses()} == {"tonic", "phasic"}


def test_load_doric_rejects_missing_signal_root(tmp_path) -> None:
    path = tmp_path / "invalid.doric"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_group("DataAcquisition")

    with pytest.raises(DoricFileError, match="missing required Doric group"):
        load_doric(path)
