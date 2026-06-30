from __future__ import annotations

import h5py
import numpy as np
import pytest

from circadian_fiber_photometry import (
    SyntheticDoricConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    SyntheticTTLRandomBehaviorEventConfig,
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


def _rising_edges(values: np.ndarray) -> np.ndarray:
    edges = np.flatnonzero((values[1:] == 1) & (values[:-1] == 0)) + 1
    if values[0] == 1:
        edges = np.r_[0, edges]
    return edges.astype(int)


def _pulse_widths(values: np.ndarray) -> np.ndarray:
    rising = _rising_edges(values)
    falling = np.flatnonzero((values[1:] == 0) & (values[:-1] == 1)) + 1
    if values[-1] == 1:
        falling = np.r_[falling, values.size]
    return (falling[: rising.size] - rising[: falling.size]).astype(int)


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


def test_generate_synthetic_doric_writes_three_pulse_behavior_code(tmp_path) -> None:
    path = tmp_path / "ttl_behavior.doric"
    config = _config(
        fs=60.0,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("reward", channel=1, pulse_count=3),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig(
                code_name="reward",
                start_seconds=(5.0,),
                series_numbers=(1,),
            ),
        ),
    )

    summary = generate_synthetic_doric(path, config)

    with h5py.File(path, "r") as h5_file:
        series = h5_file["DataAcquisition/FPConsole/Signals/Series0001"]
        dio1 = series["DigitalIO/DIO01"][:]
        dio2 = series["DigitalIO/DIO02"][:]
        modulation = h5_file[
            "Configurations/FPConsole/DIO01/Modulations/Modulation1"
        ]

        assert np.array_equal(np.unique(dio1), np.array([0.0, 1.0]))
        assert np.array_equal(np.unique(dio2), np.array([0.0]))
        np.testing.assert_array_equal(_rising_edges(dio1), [300, 306, 312])
        np.testing.assert_array_equal(_pulse_widths(dio1), [3, 3, 3])
        assert np.all(np.diff(_rising_edges(dio1)) == 6)
        assert modulation.attrs["TimeON"] == pytest.approx(50.0)
        assert modulation.attrs["Period"] == pytest.approx(100.0)
        assert modulation.attrs["Frequency"] == pytest.approx(10.0)

    np.testing.assert_array_equal(
        summary.ttl_pulse_sample_indices[(1, 1)],
        [300, 306, 312],
    )
    assert summary.ttl_pulse_times_seconds[(1, 1)] == pytest.approx(
        [5.0, 5.1, 5.2]
    )
    assert len(summary.ttl_behavior_events) == 1
    behavior = summary.ttl_behavior_events[0]
    assert behavior.code_name == "reward"
    assert behavior.channel == 1
    assert behavior.pulse_count == 3
    assert behavior.series_number == 1
    assert behavior.sequence_start_sample == 300
    assert behavior.sequence_start_seconds == pytest.approx(5.0)
    np.testing.assert_array_equal(behavior.pulse_sample_indices, [300, 306, 312])


def test_generate_synthetic_doric_summarizes_multiple_behavior_codes(tmp_path) -> None:
    path = tmp_path / "multiple_behaviors.doric"
    config = _config(
        fs=60.0,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("lick", channel=1, pulse_count=1),
            SyntheticTTLBehaviorCodeConfig("entry", channel=1, pulse_count=2),
            SyntheticTTLBehaviorCodeConfig("reward", channel=1, pulse_count=3),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig("lick", start_seconds=(1.0,)),
            SyntheticTTLBehaviorEventConfig("entry", start_seconds=(3.0,)),
            SyntheticTTLBehaviorEventConfig("reward", start_seconds=(5.0,)),
        ),
    )

    summary = generate_synthetic_doric(path, config)

    with h5py.File(path, "r") as h5_file:
        dio1 = h5_file[
            "DataAcquisition/FPConsole/Signals/Series0001/DigitalIO/DIO01"
        ][:]

    np.testing.assert_array_equal(_rising_edges(dio1), [60, 180, 186, 300, 306, 312])
    assert [event.code_name for event in summary.ttl_behavior_events[:3]] == [
        "lick",
        "entry",
        "reward",
    ]
    assert [event.pulse_count for event in summary.ttl_behavior_events[:3]] == [
        1,
        2,
        3,
    ]


def test_generate_synthetic_doric_skips_disabled_ttl_behavior_configs(
    tmp_path,
) -> None:
    path = tmp_path / "disabled_behaviors.doric"
    config = _config(
        fs=60.0,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig(
                "disabled_code",
                channel=1,
                pulse_count=1,
                enabled=False,
            ),
            SyntheticTTLBehaviorCodeConfig("enabled_code", channel=2, pulse_count=1),
            SyntheticTTLBehaviorCodeConfig("random_code", channel=1, pulse_count=1),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig(
                "disabled_code",
                start_seconds=(1.0,),
            ),
            SyntheticTTLBehaviorEventConfig(
                "enabled_code",
                start_seconds=(2.0,),
                enabled=False,
            ),
        ),
        ttl_random_behavior_events=(
            SyntheticTTLRandomBehaviorEventConfig(
                "random_code",
                event_count_per_series=3,
                enabled=False,
            ),
        ),
    )

    summary = generate_synthetic_doric(path, config)

    with h5py.File(path, "r") as h5_file:
        signals = h5_file["DataAcquisition/FPConsole/Signals"]
        for series_name in signals:
            assert signals[f"{series_name}/DigitalIO/DIO01"][:] == pytest.approx(
                np.zeros(summary.samples_per_series)
            )
            assert signals[f"{series_name}/DigitalIO/DIO02"][:] == pytest.approx(
                np.zeros(summary.samples_per_series)
            )
    assert summary.ttl_behavior_events == ()


def test_generate_synthetic_doric_places_behavior_events_on_selected_series(
    tmp_path,
) -> None:
    path = tmp_path / "selected_series_behavior.doric"
    config = _config(
        fs=60.0,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("entry", channel=2, pulse_count=2),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig(
                code_name="entry",
                start_seconds=(1.0,),
                series_numbers=(2,),
            ),
        ),
    )

    summary = generate_synthetic_doric(path, config)

    with h5py.File(path, "r") as h5_file:
        signals = h5_file["DataAcquisition/FPConsole/Signals"]
        series1_dio2 = signals["Series0001/DigitalIO/DIO02"][:]
        series2_dio1 = signals["Series0002/DigitalIO/DIO01"][:]
        series2_dio2 = signals["Series0002/DigitalIO/DIO02"][:]
        series3_dio2 = signals["Series0003/DigitalIO/DIO02"][:]

        assert series1_dio2[:] == pytest.approx(np.zeros(series1_dio2.size))
        assert series2_dio1[:] == pytest.approx(np.zeros(series2_dio1.size))
        assert series3_dio2[:] == pytest.approx(np.zeros(series3_dio2.size))
        np.testing.assert_array_equal(_rising_edges(series2_dio2), [60, 66])
        np.testing.assert_array_equal(_pulse_widths(series2_dio2), [3, 3])

    assert summary.ttl_pulse_sample_indices[(1, 2)].size == 0
    np.testing.assert_array_equal(summary.ttl_pulse_sample_indices[(2, 2)], [60, 66])
    assert summary.ttl_pulse_times_seconds[(2, 2)] == pytest.approx([1801.0, 1801.1])
    assert len(summary.ttl_behavior_events) == 1
    assert summary.ttl_behavior_events[0].code_name == "entry"


def test_generate_synthetic_doric_random_behavior_events_are_seed_reproducible(
    tmp_path,
) -> None:
    base_config = _config(
        fs=60.0,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("entry", channel=1, pulse_count=2),
        ),
        ttl_random_behavior_events=(
            SyntheticTTLRandomBehaviorEventConfig(
                "entry",
                event_count_per_series=4,
                start_window_seconds=(1.0, 10.0),
                series_numbers=(1,),
            ),
        ),
    )
    first_path = tmp_path / "random_first.doric"
    second_path = tmp_path / "random_second.doric"
    different_path = tmp_path / "random_different.doric"

    first_summary = generate_synthetic_doric(first_path, base_config)
    second_summary = generate_synthetic_doric(second_path, base_config)
    different_summary = generate_synthetic_doric(
        different_path,
        _config(
            fs=60.0,
            seed=321,
            ttl_behavior_codes=base_config.ttl_behavior_codes,
            ttl_random_behavior_events=base_config.ttl_random_behavior_events,
        ),
    )

    first_starts = np.array(
        [event.sequence_start_sample for event in first_summary.ttl_behavior_events]
    )
    second_starts = np.array(
        [event.sequence_start_sample for event in second_summary.ttl_behavior_events]
    )
    different_starts = np.array(
        [event.sequence_start_sample for event in different_summary.ttl_behavior_events]
    )
    np.testing.assert_array_equal(first_starts, second_starts)
    assert not np.array_equal(first_starts, different_starts)
    assert np.all(np.diff(np.sort(first_starts)) >= 9)

    with h5py.File(first_path, "r") as h5_file:
        dio1 = h5_file[
            "DataAcquisition/FPConsole/Signals/Series0001/DigitalIO/DIO01"
        ][:]
    assert set(np.unique(dio1)) <= {0.0, 1.0}
    assert len(first_summary.ttl_behavior_events) == 4


def test_generate_synthetic_doric_combines_behavior_sources_in_summary(
    tmp_path,
) -> None:
    path = tmp_path / "combined_behaviors.doric"
    config = _config(
        fs=60.0,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("lick", channel=1, pulse_count=1),
            SyntheticTTLBehaviorCodeConfig("entry", channel=1, pulse_count=2),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig("lick", start_seconds=(1.0,)),
        ),
        ttl_random_behavior_events=(
            SyntheticTTLRandomBehaviorEventConfig(
                "entry",
                event_count_per_series=1,
                start_window_seconds=(3.0, 4.0),
                series_numbers=(1,),
            ),
        ),
    )

    summary = generate_synthetic_doric(path, config)

    assert {event.code_name for event in summary.ttl_behavior_events} == {
        "lick",
        "entry",
    }
    with h5py.File(path, "r") as h5_file:
        dio1 = h5_file[
            "DataAcquisition/FPConsole/Signals/Series0001/DigitalIO/DIO01"
        ][:]
    np.testing.assert_array_equal(
        _rising_edges(dio1),
        summary.ttl_pulse_sample_indices[(1, 1)],
    )


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


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("event", channel=3, pulse_count=1),
                )
            },
            "TTL channel",
        ),
        (
            {
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("event", channel=1, pulse_count=0),
                )
            },
            "pulse_count",
        ),
        (
            {
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("event", channel=1, pulse_count=1),
                    SyntheticTTLBehaviorCodeConfig("event", channel=2, pulse_count=2),
                )
            },
            "duplicate",
        ),
        (
            {
                "ttl_behavior_events": (
                    SyntheticTTLBehaviorEventConfig("missing", start_seconds=(1.0,)),
                )
            },
            "unknown",
        ),
        (
            {
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("event", channel=1, pulse_count=1),
                ),
                "ttl_behavior_events": (
                    SyntheticTTLBehaviorEventConfig(
                        "event",
                        start_seconds=(1.0,),
                        series_numbers=(4,),
                    ),
                ),
            },
            "series_numbers",
        ),
        (
            {
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("event", channel=1, pulse_count=1),
                ),
                "ttl_behavior_events": (
                    SyntheticTTLBehaviorEventConfig("event", start_seconds=(-1.0,)),
                ),
            },
            "start_seconds must be nonnegative",
        ),
        (
            {
                "fs": 60.0,
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("event", channel=1, pulse_count=3),
                ),
                "ttl_behavior_events": (
                    SyntheticTTLBehaviorEventConfig("event", start_seconds=(19.9,)),
                ),
            },
            "extends beyond",
        ),
        (
            {
                "fs": 60.0,
                "ttl_behavior_codes": (
                    SyntheticTTLBehaviorCodeConfig("first", channel=1, pulse_count=3),
                    SyntheticTTLBehaviorCodeConfig("second", channel=1, pulse_count=1),
                ),
                "ttl_behavior_events": (
                    SyntheticTTLBehaviorEventConfig("first", start_seconds=(1.0,)),
                    SyntheticTTLBehaviorEventConfig("second", start_seconds=(1.02,)),
                ),
            },
            "overlap",
        ),
    ],
)
def test_generate_synthetic_doric_rejects_invalid_ttl_behavior_config(
    tmp_path,
    overrides,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_synthetic_doric(
            tmp_path / "invalid_ttl_behavior.doric",
            _config(**overrides),
        )


def test_generate_synthetic_doric_rejects_existing_path_unless_overwrite(
    tmp_path,
) -> None:
    path = tmp_path / "existing.doric"
    generate_synthetic_doric(path, _config())

    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_synthetic_doric(path, _config())

    generate_synthetic_doric(path, _config(seed=321), overwrite=True)
    assert path.exists()
