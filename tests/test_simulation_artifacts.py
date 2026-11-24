from __future__ import annotations

import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest

from circadian_fiber_photometry.simulation import (
    SyntheticArtifactOccurrence,
    SyntheticDoricConfig,
    SyntheticPhotobleachingConfig,
    SyntheticPhotometryDisconnectionConfig,
    SyntheticRandomBoxArtifactConfig,
    SyntheticScheduledBoxArtifactConfig,
    SyntheticSessionStartSpikeConfig,
    SyntheticSignalConfig,
    add_random_box_artifacts,
    add_scheduled_box_artifacts,
    configure_photometry_disconnection,
    configure_session_start_spike,
    generate_synthetic_doric,
)
from circadian_fiber_photometry.simulation.artifact import (
    SyntheticArtifactOccurrence as CanonicalArtifactOccurrence,
)
from circadian_fiber_photometry.simulation.artifact import (
    SyntheticPhotometryDisconnectionConfig as CanonicalDisconnectionConfig,
)
from circadian_fiber_photometry.simulation.artifact import (
    add_random_box_artifacts as canonical_add_random_box_artifacts,
)
from circadian_fiber_photometry.simulation.artifact import (
    add_scheduled_box_artifacts as canonical_add_scheduled_box_artifacts,
)
from circadian_fiber_photometry.simulation.artifact import (
    configure_photometry_disconnection as canonical_configure_disconnection,
)
from circadian_fiber_photometry.simulation.artifact import (
    configure_session_start_spike as canonical_configure_session_start_spike,
)


def _flat_signal() -> SyntheticSignalConfig:
    return SyntheticSignalConfig(
        channel_baseline_step=0.0,
        photobleaching=SyntheticPhotobleachingConfig(model="none"),
        artifact_amplitude=0.0,
        circadian_amplitude=0.0,
        noise_std=0.0,
        analog_noise_std=0.0,
        transient_rate_per_minute=0.0,
        transient_amplitude=0.0,
    )


def _config(signal: SyntheticSignalConfig, **overrides: object) -> SyntheticDoricConfig:
    values = {
        "series_count": 1,
        "session_duration_seconds": 20.0,
        "inter_series_gap_seconds": 10.0,
        "fs": 20.0,
        "channel_count": 1,
        "seed": 7,
        "signal": signal,
    }
    values.update(overrides)
    return SyntheticDoricConfig(**values)


def _read_signal(path, series: int, channel: int, output: int) -> np.ndarray:
    dataset = (
        f"DataAcquisition/FPConsole/Signals/Series{series:04d}/"
        f"AIN{channel:02d}xAOUT{output:02d}-LockIn/Values"
    )
    with h5py.File(path, "r") as h5_file:
        return h5_file[dataset][:]


def _read_analog(path, series: int, channel: int) -> np.ndarray:
    dataset = (
        f"DataAcquisition/FPConsole/Signals/Series{series:04d}/"
        f"AnalogIn/AIN{channel:02d}"
    )
    with h5py.File(path, "r") as h5_file:
        return h5_file[dataset][:]


def test_artifact_types_are_available_from_both_simulation_namespaces() -> None:
    assert SyntheticArtifactOccurrence is CanonicalArtifactOccurrence
    assert SyntheticPhotometryDisconnectionConfig is CanonicalDisconnectionConfig
    assert add_random_box_artifacts is canonical_add_random_box_artifacts
    assert add_scheduled_box_artifacts is canonical_add_scheduled_box_artifacts
    assert configure_session_start_spike is canonical_configure_session_start_spike
    assert configure_photometry_disconnection is canonical_configure_disconnection
    assert SyntheticSessionStartSpikeConfig().duration_seconds == 1.0
    assert SyntheticSessionStartSpikeConfig().magnitude_fraction == 1.0
    assert SyntheticScheduledBoxArtifactConfig((1.0,)).magnitude_fraction == 0.10
    random_config = SyntheticRandomBoxArtifactConfig(count_per_series=1)
    assert random_config.duration_range_seconds == (1.0, 1.0)
    disconnection = SyntheticPhotometryDisconnectionConfig()
    assert disconnection.time_reference == "experiment"
    assert disconnection.isosbestic_floor == 0.0
    assert disconnection.calcium_floor == 0.0


def test_artifact_marimo_example_imports() -> None:
    pytest.importorskip("marimo")
    example_path = Path(__file__).parents[1] / "examples" / "artifact_simulation.py"

    namespace = runpy.run_path(example_path)

    assert namespace["app"].__class__.__module__.startswith("marimo")


def test_session_start_spike_applies_exact_default_offset_everywhere(tmp_path) -> None:
    clean_path = tmp_path / "clean.doric"
    artifact_path = tmp_path / "startup.doric"
    clean_signal = _flat_signal()
    artifact_signal = configure_session_start_spike(clean_signal, name="startup")
    common = {
        "series_count": 2,
        "channel_count": 2,
        "fs": 20.0,
    }

    clean_summary = generate_synthetic_doric(
        clean_path,
        _config(clean_signal, **common),
    )
    artifact_summary = generate_synthetic_doric(
        artifact_path,
        _config(artifact_signal, **common),
    )

    assert clean_summary.artifact_occurrences == ()
    assert len(artifact_summary.artifact_occurrences) == 4
    for series in (1, 2):
        for channel in (1, 2):
            clean_405 = _read_signal(clean_path, series, channel, 1)
            clean_465 = _read_signal(clean_path, series, channel, 2)
            artifact_405 = _read_signal(artifact_path, series, channel, 1)
            artifact_465 = _read_signal(artifact_path, series, channel, 2)
            expected_405 = np.zeros_like(clean_405)
            expected_465 = np.zeros_like(clean_465)
            expected_405[:20] = np.mean(clean_405)
            expected_465[:20] = np.mean(clean_465)
            np.testing.assert_allclose(artifact_405 - clean_405, expected_405)
            np.testing.assert_allclose(artifact_465 - clean_465, expected_465)
            np.testing.assert_allclose(
                _read_analog(artifact_path, series, channel),
                _read_analog(clean_path, series, channel),
            )

    occurrence = artifact_summary.artifact_occurrences[-1]
    assert occurrence.artifact_type == "session_start_spike"
    assert occurrence.name == "startup"
    assert occurrence.series_number == 2
    assert occurrence.channel_number == 2
    assert (occurrence.start_sample, occurrence.stop_sample) == (0, 20)
    assert occurrence.requested_duration_seconds == 1.0
    assert occurrence.realized_duration_seconds == pytest.approx(1.0)
    assert occurrence.absolute_start_seconds == pytest.approx(30.0)
    assert occurrence.absolute_stop_seconds == pytest.approx(31.0)


def test_scheduled_boxes_support_signed_offsets_durations_and_selectors(
    tmp_path,
) -> None:
    clean_signal = _flat_signal()
    artifact_signal = add_scheduled_box_artifacts(
        clean_signal,
        [2.0, 5.0],
        durations_seconds=[1.0, 0.5],
        magnitude_fraction=0.10,
        channels=(1,),
        series_numbers=(1,),
        name="positive",
    )
    artifact_signal = add_scheduled_box_artifacts(
        artifact_signal,
        [8.0],
        durations_seconds=1.0,
        magnitude_fraction=-0.20,
        channels=(1,),
        series_numbers=(1,),
        name="drop",
    )
    common = {"series_count": 2, "channel_count": 2, "fs": 20.0}
    clean_path = tmp_path / "scheduled_clean.doric"
    artifact_path = tmp_path / "scheduled.doric"

    generate_synthetic_doric(clean_path, _config(clean_signal, **common))
    summary = generate_synthetic_doric(
        artifact_path,
        _config(artifact_signal, **common),
    )

    clean_405 = _read_signal(clean_path, 1, 1, 1)
    clean_465 = _read_signal(clean_path, 1, 1, 2)
    difference_405 = _read_signal(artifact_path, 1, 1, 1) - clean_405
    difference_465 = _read_signal(artifact_path, 1, 1, 2) - clean_465
    expected_fraction = np.zeros(clean_405.size)
    expected_fraction[40:60] = 0.10
    expected_fraction[100:110] = 0.10
    expected_fraction[160:180] = -0.20
    np.testing.assert_allclose(difference_405, expected_fraction * np.mean(clean_405))
    np.testing.assert_allclose(difference_465, expected_fraction * np.mean(clean_465))
    np.testing.assert_allclose(
        _read_signal(artifact_path, 1, 2, 1),
        _read_signal(clean_path, 1, 2, 1),
    )
    np.testing.assert_allclose(
        _read_signal(artifact_path, 2, 1, 1),
        _read_signal(clean_path, 2, 1, 1),
    )

    assert [item.name for item in summary.artifact_occurrences] == [
        "positive",
        "positive",
        "drop",
    ]
    assert [
        (item.start_sample, item.stop_sample)
        for item in summary.artifact_occurrences
    ] == [(40, 60), (100, 110), (160, 180)]


def test_box_artifacts_can_overlap_session_start_spike(tmp_path) -> None:
    signal = configure_session_start_spike(_flat_signal())
    signal = add_scheduled_box_artifacts(
        signal,
        [0.5],
        durations_seconds=1.0,
    )

    summary = generate_synthetic_doric(
        tmp_path / "allowed_overlap.doric",
        _config(signal),
    )

    assert [item.artifact_type for item in summary.artifact_occurrences] == [
        "session_start_spike",
        "scheduled_box",
    ]


def test_scheduled_boxes_use_sample_rounding_and_allow_touching_bounds(
    tmp_path,
) -> None:
    signal = add_scheduled_box_artifacts(
        _flat_signal(),
        [1.04, 1.15],
        durations_seconds=0.08,
    )

    summary = generate_synthetic_doric(
        tmp_path / "rounded_touching.doric",
        _config(signal, fs=20.0),
    )

    assert [
        (item.start_sample, item.stop_sample)
        for item in summary.artifact_occurrences
    ] == [(21, 23), (23, 25)]
    assert [
        item.realized_duration_seconds for item in summary.artifact_occurrences
    ] == pytest.approx([0.1, 0.1])


def test_random_boxes_are_reproducible_nonoverlapping_and_shared_across_channels(
    tmp_path,
) -> None:
    signal = add_scheduled_box_artifacts(
        _flat_signal(),
        [5.0],
        durations_seconds=1.0,
        channels=(1, 2),
    )
    signal = add_random_box_artifacts(
        signal,
        count_per_series=4,
        start_window_seconds=(2.0, 12.0),
        duration_range_seconds=(0.5, 1.0),
        channels=(1, 2),
        name="random",
    )
    config = _config(signal, channel_count=2, seed=22)

    first = generate_synthetic_doric(tmp_path / "random_first.doric", config)
    second = generate_synthetic_doric(tmp_path / "random_second.doric", config)
    different = generate_synthetic_doric(
        tmp_path / "random_different.doric",
        _config(signal, channel_count=2, seed=23),
    )

    def random_bounds(summary, channel):
        return [
            (item.start_sample, item.stop_sample)
            for item in summary.artifact_occurrences
            if item.artifact_type == "random_box" and item.channel_number == channel
        ]

    channel_one = random_bounds(first, 1)
    channel_two = random_bounds(first, 2)
    assert channel_one == channel_two
    assert channel_one == random_bounds(second, 1)
    assert channel_one != random_bounds(different, 1)
    assert len(channel_one) == 4
    assert all(40 <= start < stop <= 240 for start, stop in channel_one)
    assert all(10 <= stop - start <= 20 for start, stop in channel_one)
    all_bounds = sorted(channel_one + [(100, 120)])
    assert all(
        previous_stop <= next_start
        for (_, previous_stop), (next_start, _) in zip(
            all_bounds,
            all_bounds[1:],
            strict=False,
        )
    )


def test_random_box_rate_uses_poisson_count_for_selected_window(tmp_path) -> None:
    signal = add_random_box_artifacts(
        _flat_signal(),
        rate_per_minute=9.0,
        duration_range_seconds=(0.25, 0.25),
    )

    summary = generate_synthetic_doric(
        tmp_path / "rate.doric",
        _config(signal, fs=20.0, seed=0),
    )

    # Seed 0 and a 20-second window produce Poisson(lambda=3) == 3.
    assert len(summary.artifact_occurrences) == 3
    assert all(
        item.artifact_type == "random_box"
        and item.realized_duration_seconds == pytest.approx(0.25)
        for item in summary.artifact_occurrences
    )


def test_random_artifact_rng_does_not_perturb_existing_stochastic_signals(
    tmp_path,
) -> None:
    clean_signal = SyntheticSignalConfig()
    artifact_signal = add_random_box_artifacts(
        clean_signal,
        count_per_series=2,
        start_window_seconds=(2.0, 8.0),
        duration_range_seconds=(0.5, 0.75),
    )
    clean_path = tmp_path / "stochastic_clean.doric"
    artifact_path = tmp_path / "stochastic_artifact.doric"

    clean_summary = generate_synthetic_doric(
        clean_path,
        _config(clean_signal, seed=42),
    )
    artifact_summary = generate_synthetic_doric(
        artifact_path,
        _config(artifact_signal, seed=42),
    )

    np.testing.assert_array_equal(
        clean_summary.event_sample_indices[(1, 1)],
        artifact_summary.event_sample_indices[(1, 1)],
    )
    np.testing.assert_allclose(
        _read_analog(clean_path, 1, 1),
        _read_analog(artifact_path, 1, 1),
    )
    for output, offset_name in ((1, "isosbestic_offset"), (2, "calcium_offset")):
        clean = _read_signal(clean_path, 1, 1, output)
        restored = _read_signal(artifact_path, 1, 1, output).copy()
        for occurrence in artifact_summary.artifact_occurrences:
            restored[occurrence.start_sample : occurrence.stop_sample] -= getattr(
                occurrence,
                offset_name,
            )
        np.testing.assert_allclose(restored, clean)


def test_disconnection_defaults_to_final_second_of_final_series(tmp_path) -> None:
    clean_signal = _flat_signal()
    disconnected_signal = configure_photometry_disconnection(
        clean_signal,
        isosbestic_floor=0.005,
        calcium_floor=0.007,
        name="equipment off",
    )
    common = {"series_count": 3, "channel_count": 2, "fs": 20.0}
    clean_path = tmp_path / "disconnection_clean.doric"
    disconnected_path = tmp_path / "disconnection_default.doric"

    clean_summary = generate_synthetic_doric(
        clean_path,
        _config(clean_signal, **common),
    )
    disconnected_summary = generate_synthetic_doric(
        disconnected_path,
        _config(disconnected_signal, **common),
    )

    assert clean_summary.artifact_occurrences == ()
    assert len(disconnected_summary.artifact_occurrences) == 2
    for series in (1, 2, 3):
        for channel in (1, 2):
            np.testing.assert_allclose(
                _read_analog(disconnected_path, series, channel),
                _read_analog(clean_path, series, channel),
            )
            for output, floor in ((1, 0.005), (2, 0.007)):
                clean = _read_signal(clean_path, series, channel, output)
                disconnected = _read_signal(
                    disconnected_path,
                    series,
                    channel,
                    output,
                )
                if series < 3:
                    np.testing.assert_allclose(disconnected, clean)
                else:
                    np.testing.assert_allclose(disconnected[:380], clean[:380])
                    np.testing.assert_allclose(disconnected[380:], floor)

    occurrence = disconnected_summary.artifact_occurrences[0]
    assert occurrence.artifact_type == "photometry_disconnection"
    assert occurrence.name == "equipment off"
    assert occurrence.series_number == 3
    assert (occurrence.start_sample, occurrence.stop_sample) == (380, 400)
    assert occurrence.requested_start_seconds == pytest.approx(79.0)
    assert occurrence.requested_duration_seconds is None
    assert occurrence.time_reference == "experiment"
    assert occurrence.absolute_start_seconds == pytest.approx(79.0)
    assert occurrence.absolute_stop_seconds == pytest.approx(80.0)
    assert occurrence.realized_duration_seconds == pytest.approx(1.0)
    assert occurrence.magnitude_fraction is None
    assert occurrence.isosbestic_offset is None
    assert occurrence.calcium_offset is None
    assert occurrence.isosbestic_floor == pytest.approx(0.005)
    assert occurrence.calcium_floor == pytest.approx(0.007)


def test_series_relative_disconnection_supports_selectors_and_duration(
    tmp_path,
) -> None:
    clean_signal = _flat_signal()
    disconnected_signal = configure_photometry_disconnection(
        clean_signal,
        start_seconds=4.04,
        time_reference="series",
        duration_seconds=1.06,
        isosbestic_floor=-0.01,
        calcium_floor=0.02,
        channels=(2,),
        series_numbers=(1, 3),
    )
    common = {"series_count": 3, "channel_count": 2, "fs": 20.0}
    clean_path = tmp_path / "series_relative_clean.doric"
    disconnected_path = tmp_path / "series_relative.doric"
    generate_synthetic_doric(clean_path, _config(clean_signal, **common))
    summary = generate_synthetic_doric(
        disconnected_path,
        _config(disconnected_signal, **common),
    )

    assert [
        (item.series_number, item.channel_number, item.start_sample, item.stop_sample)
        for item in summary.artifact_occurrences
    ] == [(1, 2, 81, 102), (3, 2, 81, 102)]
    for series in (1, 3):
        for output, floor in ((1, -0.01), (2, 0.02)):
            clean = _read_signal(clean_path, series, 2, output)
            disconnected = _read_signal(disconnected_path, series, 2, output)
            np.testing.assert_allclose(disconnected[:81], clean[:81])
            np.testing.assert_allclose(disconnected[81:102], floor)
            np.testing.assert_allclose(disconnected[102:], clean[102:])
    np.testing.assert_allclose(
        _read_signal(disconnected_path, 2, 2, 1),
        _read_signal(clean_path, 2, 2, 1),
    )
    np.testing.assert_allclose(
        _read_signal(disconnected_path, 1, 1, 1),
        _read_signal(clean_path, 1, 1, 1),
    )
    assert all(
        item.time_reference == "series"
        and item.requested_start_seconds == pytest.approx(4.04)
        and item.requested_duration_seconds == pytest.approx(1.06)
        and item.realized_duration_seconds == pytest.approx(1.05)
        for item in summary.artifact_occurrences
    )


def test_experiment_relative_disconnection_maps_series_and_clips_duration(
    tmp_path,
) -> None:
    clean_signal = _flat_signal()
    disconnected_signal = configure_photometry_disconnection(
        clean_signal,
        start_seconds=34.04,
        duration_seconds=100.0,
        isosbestic_floor=0.003,
        calcium_floor=0.004,
    )
    common = {"series_count": 3, "channel_count": 1, "fs": 20.0}
    clean_path = tmp_path / "absolute_clean.doric"
    disconnected_path = tmp_path / "absolute_disconnection.doric"
    generate_synthetic_doric(clean_path, _config(clean_signal, **common))
    summary = generate_synthetic_doric(
        disconnected_path,
        _config(disconnected_signal, **common),
    )

    occurrence = summary.artifact_occurrences[0]
    assert occurrence.series_number == 2
    assert (occurrence.start_sample, occurrence.stop_sample) == (81, 400)
    assert occurrence.requested_start_seconds == pytest.approx(34.04)
    assert occurrence.requested_duration_seconds == pytest.approx(100.0)
    assert occurrence.absolute_start_seconds == pytest.approx(34.05)
    assert occurrence.absolute_stop_seconds == pytest.approx(50.0)
    assert occurrence.realized_duration_seconds == pytest.approx(15.95)
    for series in (1, 3):
        np.testing.assert_allclose(
            _read_signal(disconnected_path, series, 1, 1),
            _read_signal(clean_path, series, 1, 1),
        )


def test_disconnection_clamp_wins_over_overlapping_additive_artifacts(
    tmp_path,
) -> None:
    signal = configure_session_start_spike(_flat_signal(), duration_seconds=20.0)
    signal = add_scheduled_box_artifacts(
        signal,
        [19.0],
        durations_seconds=1.0,
        magnitude_fraction=0.5,
    )
    signal = configure_photometry_disconnection(
        signal,
        isosbestic_floor=0.001,
        calcium_floor=0.002,
    )

    summary = generate_synthetic_doric(
        tmp_path / "overlapping_disconnection.doric",
        _config(signal),
    )

    assert [item.artifact_type for item in summary.artifact_occurrences] == [
        "session_start_spike",
        "photometry_disconnection",
        "scheduled_box",
    ]
    np.testing.assert_allclose(
        _read_signal(tmp_path / "overlapping_disconnection.doric", 1, 1, 1)[380:],
        0.001,
    )
    np.testing.assert_allclose(
        _read_signal(tmp_path / "overlapping_disconnection.doric", 1, 1, 2)[380:],
        0.002,
    )


@pytest.mark.parametrize(
    ("signal", "match"),
    [
        (
            configure_photometry_disconnection(
                _flat_signal(),
                time_reference="invalid",  # type: ignore[arg-type]
            ),
            "time_reference",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                start_seconds=25.0,
            ),
            "recorded series",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                start_seconds=90.0,
            ),
            "recorded series",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                start_seconds=-1.0,
            ),
            "nonnegative",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                series_numbers=(1,),
            ),
            "series_numbers cannot",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                duration_seconds=0.0,
            ),
            "duration_seconds must be positive",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                duration_seconds=0.001,
            ),
            "round to at least 1 sample",
        ),
        (
            configure_photometry_disconnection(
                _flat_signal(),
                calcium_floor=np.nan,
            ),
            "calcium_floor must be finite",
        ),
    ],
)
def test_disconnection_rejects_invalid_configuration(
    tmp_path,
    signal,
    match,
) -> None:
    config = _config(signal, series_count=3)
    with pytest.raises(ValueError, match=match):
        generate_synthetic_doric(tmp_path / "invalid_disconnection.doric", config)


@pytest.mark.parametrize(
    ("signal", "match"),
    [
        (
            add_scheduled_box_artifacts(
                _flat_signal(),
                [1.0, 2.0],
                durations_seconds=[0.5],
            ),
            "match start_times_seconds",
        ),
        (
            add_scheduled_box_artifacts(
                add_scheduled_box_artifacts(
                    _flat_signal(),
                    [2.0],
                    durations_seconds=2.0,
                ),
                [3.0],
                durations_seconds=1.0,
            ),
            "overlap",
        ),
        (
            add_random_box_artifacts(_flat_signal()),
            "exactly one",
        ),
        (
            add_random_box_artifacts(
                _flat_signal(),
                count_per_series=1,
                rate_per_minute=1.0,
            ),
            "exactly one",
        ),
        (
            add_random_box_artifacts(
                _flat_signal(),
                count_per_series=2,
                start_window_seconds=(0.0, 10.0),
                duration_range_seconds=(6.0, 6.0),
            ),
            "could not place",
        ),
        (
            add_scheduled_box_artifacts(
                _flat_signal(),
                [1.0],
                magnitude_fraction=np.nan,
            ),
            "magnitude_fraction must be finite",
        ),
        (
            add_scheduled_box_artifacts(
                _flat_signal(),
                [19.9],
                durations_seconds=0.2,
            ),
            "extends beyond",
        ),
        (
            add_random_box_artifacts(
                _flat_signal(),
                count_per_series=-1,
            ),
            "nonnegative integer",
        ),
        (
            add_random_box_artifacts(
                _flat_signal(),
                count_per_series=1,
                start_window_seconds=(1.0, 21.0),
            ),
            "within the session",
        ),
    ],
)
def test_artifact_configuration_rejects_invalid_values(tmp_path, signal, match) -> None:
    with pytest.raises(ValueError, match=match):
        generate_synthetic_doric(tmp_path / "invalid.doric", _config(signal))
