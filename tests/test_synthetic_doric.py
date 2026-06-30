from __future__ import annotations

import h5py
import numpy as np
import pytest

from circadian_fiber_photometry import (
    SyntheticDoricConfig,
    analyze_sessions,
    generate_synthetic_doric,
)


def _config(**overrides: object) -> SyntheticDoricConfig:
    values = {
        "series_count": 3,
        "session_duration_seconds": 20.0,
        "inter_series_gap_seconds": 1780.0,
        "fs": 20.0,
        "channel_count": 2,
        "seed": 123,
    }
    values.update(overrides)
    return SyntheticDoricConfig(**values)


def _read_values(path, dataset_name: str) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        return h5_file[dataset_name][:]


def test_generate_synthetic_doric_writes_expected_layout(tmp_path) -> None:
    path = tmp_path / "synthetic.doric"
    config = _config(configured_series_count=5, filename_metadata="synthetic_test")

    summary = generate_synthetic_doric(path, config)

    assert summary.path == path
    assert summary.series_count == 3
    assert summary.configured_series_count == 5
    assert summary.samples_per_series == 400
    assert summary.session_start_times == pytest.approx([0.0, 1800.0, 3600.0])
    assert (1, 1) in summary.event_sample_indices

    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["SoftwareName"] == "Doric Neuroscience Studio"
        assert h5_file.attrs["SoftwareVersion"] == "6.2.4.0"
        assert "Configurations" in h5_file
        assert "DataAcquisition" in h5_file

        timeseries = h5_file["Configurations/FPConsole/TimeseriesSettings"]
        assert timeseries.attrs["NumberOfSeries"] == 5
        assert timeseries.attrs["TimeActive(ms)"] == 20000

        signals = h5_file["DataAcquisition/FPConsole/Signals"]
        assert list(signals.keys()) == ["Series0001", "Series0002", "Series0003"]

        series = signals["Series0001"]
        assert list(series.keys()) == [
            "AIN01xAOUT01-LockIn",
            "AIN01xAOUT02-LockIn",
            "AIN02xAOUT01-LockIn",
            "AIN02xAOUT02-LockIn",
            "AnalogIn",
            "AnalogOut",
            "DigitalIO",
        ]

        values = series["AIN01xAOUT01-LockIn/Values"]
        assert values.shape == (400,)
        assert values.dtype == np.dtype("float64")
        assert values.compression is None
        assert values.attrs["Name"] == "AIN01xAOUT01-LockIn"
        assert values.attrs["Unit"] == "Voltage (V)"
        assert values.attrs["RangeMin"] == pytest.approx(-10.0)
        assert values.attrs["RangeMax"] == pytest.approx(10.0)
        assert values.attrs["Username"] == "AIN01xAOUT01-LockIn"

        time = series["AIN01xAOUT01-LockIn/Time"][:]
        assert time[0] == pytest.approx(0.0)
        assert time[-1] == pytest.approx(399 / config.fs)

        analog_in = series["AnalogIn"]
        assert list(analog_in.keys()) == ["Time", "AIN01", "AIN02"]
        assert analog_in["AIN02"].shape == (400,)

        analog_out = series["AnalogOut"]
        assert list(analog_out.keys()) == ["Time", "AOUT01", "AOUT02"]
        assert analog_out["AOUT01"][:] == pytest.approx(np.ones(400))

        digital_io = series["DigitalIO"]
        assert list(digital_io.keys()) == ["Time", "DIO01", "DIO02"]
        assert digital_io["DIO02"][:] == pytest.approx(np.zeros(400))


def test_generate_synthetic_doric_is_seed_reproducible(tmp_path) -> None:
    config = _config(seed=99)
    first_path = tmp_path / "first.doric"
    second_path = tmp_path / "second.doric"
    different_path = tmp_path / "different.doric"

    generate_synthetic_doric(first_path, config)
    generate_synthetic_doric(second_path, config)
    generate_synthetic_doric(different_path, _config(seed=100))

    dataset_name = (
        "DataAcquisition/FPConsole/Signals/Series0001/"
        "AIN01xAOUT02-LockIn/Values"
    )
    first = _read_values(first_path, dataset_name)
    second = _read_values(second_path, dataset_name)
    different = _read_values(different_path, dataset_name)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different)


def test_generated_doric_lockin_traces_work_with_analysis(tmp_path) -> None:
    config = _config(series_count=4, seed=55)
    path = tmp_path / "analysis.doric"
    summary = generate_synthetic_doric(path, config)

    iso = np.empty(
        (summary.samples_per_series, config.channel_count, config.series_count)
    )
    calcium = np.empty_like(iso)

    with h5py.File(path, "r") as h5_file:
        for series_number in range(1, config.series_count + 1):
            series_path = (
                f"DataAcquisition/FPConsole/Signals/Series{series_number:04d}"
            )
            for channel_number in range(1, config.channel_count + 1):
                iso[:, channel_number - 1, series_number - 1] = h5_file[
                    f"{series_path}/AIN{channel_number:02d}xAOUT01-LockIn/Values"
                ][:]
                calcium[:, channel_number - 1, series_number - 1] = h5_file[
                    f"{series_path}/AIN{channel_number:02d}xAOUT02-LockIn/Values"
                ][:]

    result = analyze_sessions(
        iso,
        calcium,
        fs=config.fs,
        interval_hours=0.5,
        weight_fit=False,
    )

    assert result.dff.shape == iso.shape
    assert result.event_counts.shape == (config.channel_count, config.series_count)
    assert np.all(np.isfinite(result.dff))
    assert np.all(np.isfinite(result.dff_dynamic_corrected))
    assert np.all(np.isfinite(result.level_phasic))


def test_generate_synthetic_doric_supports_incomplete_file_metadata(tmp_path) -> None:
    path = tmp_path / "incomplete.doric"
    config = _config(series_count=2, configured_series_count=7)

    summary = generate_synthetic_doric(path, config)

    assert summary.series_count == 2
    assert summary.configured_series_count == 7
    with h5py.File(path, "r") as h5_file:
        signals = h5_file["DataAcquisition/FPConsole/Signals"]
        assert list(signals.keys()) == ["Series0001", "Series0002"]
        timeseries = h5_file["Configurations/FPConsole/TimeseriesSettings"]
        assert timeseries.attrs["NumberOfSeries"] == 7


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fs": 0.0}, "fs must be positive"),
        ({"series_count": 0}, "series_count must be a positive integer"),
        ({"channel_count": 0}, "channel_count must be a positive integer"),
        (
            {"session_duration_seconds": 9.0},
            "first/last 5 second crop",
        ),
        (
            {"configured_series_count": 1, "series_count": 2},
            "configured_series_count",
        ),
    ],
)
def test_generate_synthetic_doric_rejects_invalid_config(
    tmp_path,
    overrides,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_synthetic_doric(tmp_path / "invalid.doric", _config(**overrides))


def test_generate_synthetic_doric_rejects_existing_path_unless_overwrite(
    tmp_path,
) -> None:
    path = tmp_path / "existing.doric"
    generate_synthetic_doric(path, _config())

    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_synthetic_doric(path, _config())

    generate_synthetic_doric(path, _config(seed=321), overwrite=True)
    assert path.exists()
