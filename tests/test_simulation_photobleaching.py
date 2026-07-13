from __future__ import annotations

import math
import runpy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from circadian_fiber_photometry.simulation import (
    SyntheticDoricConfig,
    SyntheticPhotobleachingComponentConfig,
    SyntheticPhotobleachingConfig,
    SyntheticSignalConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLRandomBehaviorEventConfig,
    add_random_box_artifacts,
    add_scheduled_calcium_events,
    add_tonic_component,
    generate_synthetic_doric,
)


def _component(
    amplitude: float,
    time_constant_seconds: float | None = None,
) -> SyntheticPhotobleachingComponentConfig:
    return SyntheticPhotobleachingComponentConfig(
        amplitude_fraction=amplitude,
        time_constant_seconds=time_constant_seconds,
    )


def _signal(
    photobleaching: SyntheticPhotobleachingConfig | None = None,
    **overrides: object,
) -> SyntheticSignalConfig:
    values: dict[str, object] = {
        "isosbestic_baseline": 0.08,
        "calcium_baseline": 0.18,
        "channel_baseline_step": 0.0,
        "artifact_amplitude": 0.0,
        "circadian_amplitude": 0.0,
        "noise_std": 0.0,
        "analog_noise_std": 0.0,
        "transient_rate_per_minute": 0.0,
        "transient_amplitude": 0.0,
    }
    if photobleaching is not None:
        values["photobleaching"] = photobleaching
    values.update(overrides)
    return SyntheticSignalConfig(**values)


def _config(signal: SyntheticSignalConfig, **overrides: object) -> SyntheticDoricConfig:
    values: dict[str, object] = {
        "series_count": 2,
        "session_duration_seconds": 12.0,
        "inter_series_gap_seconds": 1000.0,
        "fs": 2.0,
        "channel_count": 1,
        "seed": 31,
        "signal": signal,
        "ttl_pulse_width_seconds": 0.5,
        "ttl_pulse_off_interval_seconds": 0.5,
    }
    values.update(overrides)
    return SyntheticDoricConfig(**values)


def _read_signal(
    path,
    output_number: int,
    *,
    channel_number: int = 1,
) -> np.ndarray:
    series_values = []
    with h5py.File(path, "r") as h5_file:
        signals = h5_file["DataAcquisition/FPConsole/Signals"]
        for series_name in signals:
            series_values.append(
                signals[
                    f"{series_name}/AIN{channel_number:02d}xAOUT"
                    f"{output_number:02d}-LockIn/Values"
                ][:]
            )
    return np.concatenate(series_values)


def _factor(
    exposure_time: np.ndarray,
    components: tuple[SyntheticPhotobleachingComponentConfig, ...],
) -> np.ndarray:
    factor = np.ones_like(exposure_time)
    for component in components:
        assert component.time_constant_seconds is not None
        factor += component.amplitude_fraction * (
            np.exp(-exposure_time / component.time_constant_seconds) - 1.0
        )
    return factor


def test_single_exponential_uses_cumulative_exposure_time_and_signal_parameters(
    tmp_path,
) -> None:
    path = tmp_path / "single.doric"
    photobleaching = SyntheticPhotobleachingConfig(
        model="single_exponential",
        isosbestic_components=(_component(0.25, 8.0),),
        calcium_components=(_component(0.60, 20.0),),
    )
    config = _config(
        _signal(photobleaching, channel_baseline_step=0.01),
        channel_count=2,
    )

    summary = generate_synthetic_doric(path, config)

    exposure_time = np.arange(48, dtype=float) / config.fs
    expected_iso_factor = 0.75 + 0.25 * np.exp(-exposure_time / 8.0)
    expected_calcium_factor = 0.40 + 0.60 * np.exp(-exposure_time / 20.0)
    np.testing.assert_allclose(_read_signal(path, 1), 0.08 * expected_iso_factor)
    np.testing.assert_allclose(_read_signal(path, 2), 0.18 * expected_calcium_factor)
    np.testing.assert_allclose(
        _read_signal(path, 1, channel_number=2),
        0.09 * expected_iso_factor,
    )
    np.testing.assert_allclose(
        _read_signal(path, 2, channel_number=2),
        0.196 * expected_calcium_factor,
    )
    assert summary.photobleaching.model == photobleaching.model
    assert summary.photobleaching.isosbestic_components == (
        _component(0.25, 8.0),
    )
    assert summary.photobleaching.calcium_components == (_component(0.60, 20.0),)
    assert summary.photobleaching.time_basis == "cumulative_exposure_seconds"
    assert exposure_time[24] == 12.0
    assert summary.session_start_times[1] == 1012.0


def test_default_single_exponential_resolves_matlab_style_time_constant(
    tmp_path,
) -> None:
    path = tmp_path / "default_single.doric"
    config = _config(_signal())

    summary = generate_synthetic_doric(path, config)

    metadata = summary.photobleaching
    assert metadata.model == "single_exponential"
    assert metadata.isosbestic_components == metadata.calcium_components
    assert metadata.isosbestic_components[0].amplitude_fraction == 1.0
    assert metadata.isosbestic_components[0].time_constant_seconds == 48.0
    exposure_time = np.arange(48, dtype=float) / config.fs
    np.testing.assert_allclose(
        _read_signal(path, 1),
        0.08 * np.exp(-exposure_time / 48.0),
    )


def test_double_exponential_matches_defining_equation_for_each_signal(
    tmp_path,
) -> None:
    path = tmp_path / "double.doric"
    photobleaching = SyntheticPhotobleachingConfig(
        model="double_exponential",
        isosbestic_components=(_component(0.10, 5.0), _component(0.20, 50.0)),
        calcium_components=(_component(0.30, 10.0), _component(0.40, 100.0)),
    )
    config = _config(_signal(photobleaching))

    summary = generate_synthetic_doric(path, config)

    exposure_time = np.arange(48, dtype=float) / config.fs
    metadata = summary.photobleaching
    np.testing.assert_allclose(
        _read_signal(path, 1),
        0.08 * _factor(exposure_time, metadata.isosbestic_components),
    )
    np.testing.assert_allclose(
        _read_signal(path, 2),
        0.18 * _factor(exposure_time, metadata.calcium_components),
    )
    assert metadata.model == "double_exponential"


def test_double_exponential_defaults_match_regression_sim_reference(tmp_path) -> None:
    summary = generate_synthetic_doric(
        tmp_path / "double_defaults.doric",
        _config(
            _signal(SyntheticPhotobleachingConfig(model="double_exponential"))
        ),
    )

    components = summary.photobleaching.isosbestic_components
    assert [component.amplitude_fraction for component in components] == [0.20, 0.20]
    time_constants = [component.time_constant_seconds for component in components]
    assert time_constants == pytest.approx(
        [-1.0 / math.log(0.98), -1.0 / math.log(0.998)]
    )
    assert components == summary.photobleaching.calcium_components


def test_none_model_leaves_slow_baselines_flat(tmp_path) -> None:
    path = tmp_path / "none.doric"
    summary = generate_synthetic_doric(
        path,
        _config(_signal(SyntheticPhotobleachingConfig(model="none"))),
    )

    np.testing.assert_allclose(_read_signal(path, 1), 0.08)
    np.testing.assert_allclose(_read_signal(path, 2), 0.18)
    assert summary.photobleaching.model == "none"
    assert summary.photobleaching.isosbestic_components == ()
    assert summary.photobleaching.calcium_components == ()


def test_photobleaching_attenuates_tonic_but_not_phasic_components(
    tmp_path,
) -> None:
    path = tmp_path / "tonic_and_phasic.doric"
    photobleaching = SyntheticPhotobleachingConfig(
        model="single_exponential",
        isosbestic_components=(_component(0.0, 10.0),),
        calcium_components=(_component(0.5, 10.0),),
    )
    signal = add_tonic_component(
        _signal(photobleaching),
        amplitude=0.0,
        frequency_hz=0.0,
        offset=0.02,
    )
    signal = add_scheduled_calcium_events(
        signal,
        [6.0],
        amplitude=0.05,
    )
    config = _config(signal, series_count=1)

    generate_synthetic_doric(path, config)

    exposure_time = np.arange(24, dtype=float) / config.fs
    expected = 0.20 * (0.5 + 0.5 * np.exp(-exposure_time / 10.0))
    event_sample = int(6.0 * config.fs)
    tail_time = np.arange(expected.size - event_sample, dtype=float) / config.fs
    transient = (1 - np.exp(-9.0 * tail_time)) * np.exp(-tail_time)
    transient = 0.05 * transient / np.max(transient)
    expected[event_sample:] += transient
    np.testing.assert_allclose(_read_signal(path, 2), expected)


def test_model_selection_does_not_change_seeded_schedules(tmp_path) -> None:
    base_signal = _signal(
        SyntheticPhotobleachingConfig(model="none"),
        transient_rate_per_minute=12.0,
        transient_amplitude=0.03,
    )
    base_signal = add_random_box_artifacts(
        base_signal,
        count_per_series=2,
        start_window_seconds=(1.0, 8.0),
        duration_range_seconds=(0.5, 1.0),
    )
    bleached_signal = replace(
        base_signal,
        photobleaching=SyntheticPhotobleachingConfig(),
    )
    ttl_codes = (SyntheticTTLBehaviorCodeConfig("event", channel=1, pulse_count=1),)
    ttl_random = (
        SyntheticTTLRandomBehaviorEventConfig(
            "event",
            event_count_per_series=2,
            start_window_seconds=(1.0, 8.0),
        ),
    )

    none_summary = generate_synthetic_doric(
        tmp_path / "none_schedules.doric",
        _config(
            base_signal,
            ttl_behavior_codes=ttl_codes,
            ttl_random_behavior_events=ttl_random,
        ),
    )
    single_summary = generate_synthetic_doric(
        tmp_path / "single_schedules.doric",
        _config(
            bleached_signal,
            ttl_behavior_codes=ttl_codes,
            ttl_random_behavior_events=ttl_random,
        ),
    )

    for key in none_summary.event_sample_indices:
        np.testing.assert_array_equal(
            none_summary.event_sample_indices[key],
            single_summary.event_sample_indices[key],
        )
    for key in none_summary.ttl_pulse_sample_indices:
        np.testing.assert_array_equal(
            none_summary.ttl_pulse_sample_indices[key],
            single_summary.ttl_pulse_sample_indices[key],
        )
    none_bounds = [
        (item.series_number, item.channel_number, item.start_sample, item.stop_sample)
        for item in none_summary.artifact_occurrences
    ]
    single_bounds = [
        (item.series_number, item.channel_number, item.start_sample, item.stop_sample)
        for item in single_summary.artifact_occurrences
    ]
    assert none_bounds == single_bounds


def test_model_selection_does_not_change_seeded_noise_draws(tmp_path) -> None:
    none_path = tmp_path / "none_noise.doric"
    single_path = tmp_path / "single_noise.doric"
    none_signal = _signal(
        SyntheticPhotobleachingConfig(model="none"),
        noise_std=0.01,
        analog_noise_std=0.02,
    )
    single_signal = replace(
        none_signal,
        photobleaching=SyntheticPhotobleachingConfig(),
    )

    generate_synthetic_doric(none_path, _config(none_signal))
    single_summary = generate_synthetic_doric(single_path, _config(single_signal))

    exposure_time = np.arange(48, dtype=float) / 2.0
    components = single_summary.photobleaching.isosbestic_components
    decay_factor = _factor(exposure_time, components)
    np.testing.assert_allclose(
        _read_signal(none_path, 1) - 0.08,
        _read_signal(single_path, 1) - 0.08 * decay_factor,
    )
    np.testing.assert_allclose(
        _read_signal(none_path, 2) - 0.18,
        _read_signal(single_path, 2) - 0.18 * decay_factor,
    )


@pytest.mark.parametrize("fraction", [0.0, 0.25])
def test_legacy_bleaching_fraction_warns_and_maps_to_new_models(
    tmp_path,
    fraction: float,
) -> None:
    signal = _signal(bleaching_fraction=fraction)

    with pytest.warns(DeprecationWarning, match="bleaching_fraction is deprecated"):
        summary = generate_synthetic_doric(
            tmp_path / f"legacy_{fraction}.doric",
            _config(signal),
        )

    if fraction == 0:
        assert summary.photobleaching.model == "none"
    else:
        assert summary.photobleaching.model == "single_exponential"
        assert summary.photobleaching.isosbestic_components[0].amplitude_fraction == (
            fraction
        )
        assert (
            summary.photobleaching.isosbestic_components[0].time_constant_seconds
            == 48.0
        )


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan")])
def test_invalid_legacy_bleaching_fraction_is_rejected(
    tmp_path,
    fraction: float,
) -> None:
    with pytest.raises(ValueError, match="bleaching_fraction must be in"):
        generate_synthetic_doric(
            tmp_path / "invalid_legacy.doric",
            _config(_signal(bleaching_fraction=fraction)),
        )


@pytest.mark.parametrize(
    ("photobleaching", "match"),
    [
        (SyntheticPhotobleachingConfig(model="invalid"), "photobleaching model"),
        (
            SyntheticPhotobleachingConfig(
                model="single_exponential",
                isosbestic_components=(),
            ),
            "isosbestic_components",
        ),
        (
            SyntheticPhotobleachingConfig(
                model="none",
                calcium_components=(_component(0.1, 1.0),),
            ),
            "calcium_components",
        ),
        (
            SyntheticPhotobleachingConfig(
                isosbestic_components=(_component(-0.1, 1.0),),
            ),
            "amplitudes",
        ),
        (
            SyntheticPhotobleachingConfig(
                isosbestic_components=(_component(float("nan"), 1.0),),
            ),
            "amplitudes",
        ),
        (
            SyntheticPhotobleachingConfig(
                isosbestic_components=(_component(0.1, 0.0),),
            ),
            "time constants",
        ),
        (
            SyntheticPhotobleachingConfig(
                isosbestic_components=(_component(0.1, float("inf")),),
            ),
            "time constants",
        ),
        (
            SyntheticPhotobleachingConfig(
                model="double_exponential",
                calcium_components=(
                    _component(0.6, 1.0),
                    _component(0.5, 2.0),
                ),
            ),
            "sum to at most 1",
        ),
    ],
)
def test_invalid_photobleaching_config_is_rejected(
    tmp_path,
    photobleaching: SyntheticPhotobleachingConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_synthetic_doric(
            tmp_path / "invalid.doric",
            _config(_signal(photobleaching)),
        )


def test_legacy_and_explicit_photobleaching_configs_conflict(tmp_path) -> None:
    signal = _signal(
        SyntheticPhotobleachingConfig(model="none"),
        bleaching_fraction=0.2,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        generate_synthetic_doric(tmp_path / "conflict.doric", _config(signal))


def test_photobleaching_marimo_example_imports() -> None:
    pytest.importorskip("marimo")
    example_path = (
        Path(__file__).parents[1] / "examples" / "photobleaching_simulation.py"
    )

    namespace = runpy.run_path(example_path)

    assert namespace["app"].__class__.__module__.startswith("marimo")
