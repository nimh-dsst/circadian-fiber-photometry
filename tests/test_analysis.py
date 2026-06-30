from __future__ import annotations

import numpy as np
import pytest

from circadian_fiber_photometry import (
    analyze_sessions,
    count_events,
    detect_stream_gaps,
    estimate_interval_hours,
    extract_light_pulse_windows,
    fit_405_to_465,
    irls_dynamic_correction,
    sessionize_stream_pair,
)


def test_count_events_returns_no_events_for_flat_signal() -> None:
    result = count_events(np.zeros(120), fs=10)

    assert result.count == 0
    assert result.threshold == pytest.approx(0.03)
    assert result.peak_indices.size == 0


def test_count_events_detects_one_wide_peak() -> None:
    signal = np.zeros(200)
    signal[50:75] = 1.0

    result = count_events(signal, fs=10)

    assert result.count == 1
    assert result.peak_indices[0] in {62, 63}
    assert result.peak_widths[0] >= 10


def test_count_events_rejects_narrow_peak() -> None:
    signal = np.zeros(200)
    signal[50] = 1.0

    result = count_events(signal, fs=10)

    assert result.count == 0


def test_count_events_threshold_matches_matlab_rule() -> None:
    signal = np.zeros(60)
    signal[20:32] = 1.0

    result = count_events(signal, fs=10)
    expected_threshold = np.percentile(signal, 2) + max(
        2 * np.std(signal, ddof=1),
        0.03,
    )

    assert result.threshold == pytest.approx(expected_threshold)
    assert result.count == 1


def test_fit_405_to_465_accepts_single_channel_sessions() -> None:
    samples = 120
    sessions = 3
    t = np.linspace(0, 1, samples)
    iso = np.column_stack([1.0 + t + session * 0.1 for session in range(sessions)])
    calcium = 2.0 * iso + 0.5

    result = fit_405_to_465(iso, calcium, fs=10, weight_fit=False, smooth_seconds=1)

    assert result.dff.shape == (samples, 1, sessions)
    assert result.fitted_405.shape == (samples, 1, sessions)
    assert result.coefficients.shape == (1, 2)
    assert result.coefficients[0] == pytest.approx([2.0, 0.5])


def test_fit_405_to_465_accepts_multi_channel_sessions() -> None:
    samples = 80
    channels = 2
    sessions = 2
    base = np.linspace(1, 2, samples)
    iso = np.empty((samples, channels, sessions))
    calcium = np.empty_like(iso)
    for channel in range(channels):
        for session in range(sessions):
            iso[:, channel, session] = base + channel + 0.1 * session
            calcium[:, channel, session] = (channel + 1.5) * iso[:, channel, session]

    result = fit_405_to_465(iso, calcium, fs=10, weight_fit=False)

    assert result.dff.shape == (samples, channels, sessions)
    assert result.coefficients.shape == (channels, 2)


def test_fit_405_to_465_weight_fit_matches_matlab_zero_block() -> None:
    samples = 5
    iso = np.column_stack(
        [
            np.linspace(1, 5, samples),
            np.linspace(2, 6, samples),
        ]
    )
    calcium = 2 * iso + 1

    result = fit_405_to_465(iso, calcium, fs=1, weight_fit=True, smooth_seconds=1)

    flat_405 = np.ravel(iso, order="F").copy()
    flat_465 = np.ravel(calcium, order="F").copy()
    fit_mask = (flat_465 > np.percentile(flat_465, 0)) & (
        flat_465 < np.percentile(flat_465, 100)
    )
    flat_405[:samples] = 0
    flat_465[:samples] = 0
    expected = np.polyfit(flat_405[fit_mask], flat_465[fit_mask], 1)

    assert result.coefficients[0] == pytest.approx(expected)
    assert result.coefficients[0] != pytest.approx([2.0, 1.0])


def test_fit_405_to_465_uses_conventional_fit_weights() -> None:
    iso = np.arange(6, dtype=float)
    calcium = np.array([0, 1, 2, 25, 4, 50], dtype=float)
    weights = np.ones_like(iso)
    weights[3] = 1e-6

    unweighted = fit_405_to_465(
        iso,
        calcium,
        fs=1,
        weight_fit=False,
        smooth_seconds=1,
    )
    weighted = fit_405_to_465(
        iso,
        calcium,
        fs=1,
        weight_fit=False,
        fit_weights=weights,
        smooth_seconds=1,
    )

    assert abs(weighted.coefficients[0, 0] - 1.0) < abs(
        unweighted.coefficients[0, 0] - 1.0
    )
    assert weighted.coefficients[0] == pytest.approx([1.0, 0.0], abs=1e-4)


def test_fit_405_to_465_rejects_negative_fit_weights() -> None:
    iso = np.arange(6, dtype=float)
    calcium = iso + 1
    weights = np.ones_like(iso)
    weights[2] = -1

    with pytest.raises(ValueError, match="fit_weights must be nonnegative"):
        fit_405_to_465(iso, calcium, fs=1, fit_weights=weights)


def test_irls_dynamic_correction_returns_finite_trace() -> None:
    fs = 20
    t = np.arange(400) / fs
    isosbestic = 1 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
    calcium = 0.2 + 0.8 * isosbestic
    calcium[120:150] += 0.05

    result = irls_dynamic_correction(calcium, isosbestic, fs=fs, chunk_seconds=5)

    assert result.corrected.shape == calcium.shape
    assert result.fitted_isosbestic.shape == calcium.shape
    assert np.all(np.isfinite(result.corrected))


def test_analyze_sessions_outputs_expected_shapes() -> None:
    fs = 20
    samples = 500
    channels = 2
    sessions = 4
    t = np.arange(samples) / fs
    iso = np.empty((samples, channels, sessions))
    calcium = np.empty_like(iso)

    for channel in range(channels):
        for session in range(sessions):
            baseline = 1 + 0.03 * np.sin(2 * np.pi * 0.05 * t + session)
            iso[:, channel, session] = baseline + 0.05 * channel
            calcium[:, channel, session] = 1.2 * iso[:, channel, session] + 0.2
            calcium[100:130, channel, session] += 0.02 * (channel + 1)

    result = analyze_sessions(
        iso,
        calcium,
        fs=fs,
        interval_hours=1,
        weight_fit=False,
    )

    assert result.dff.shape == (samples, channels, sessions)
    assert result.dff_dynamic_corrected.shape == (samples, channels, sessions)
    assert result.event_counts.shape == (channels, sessions)
    assert result.level_phasic.shape == (channels, sessions)
    assert result.level_tonic.shape == (channels, sessions)
    assert np.all(result.dff_phasic >= 0)


def test_extract_light_pulse_windows_matches_matlab_window_shapes() -> None:
    fs = 2
    samples = 860
    sessions = 80
    data = np.empty((samples, 1, sessions))
    ramp = np.arange(samples, dtype=float)
    for session in range(sessions):
        data[:, 0, session] = ramp + 1000 * session

    result = extract_light_pulse_windows(data, fs=fs)

    assert set(result.shifted) == {"ct6", "ct14", "ct22"}
    assert result.shifted["ct6"].shape == (90, 3)
    assert result.zscored["ct14"].shape == (90, 3)
    assert result.auc["ct22"].shape == (3, 3)
    assert np.mean(result.shifted["ct6"][:15, :], axis=0) == pytest.approx(
        np.zeros(3)
    )


def test_sessionize_stream_pair_splits_timestamp_gaps() -> None:
    fs = 10.0
    session_samples = 50
    session_1_ts = np.arange(session_samples) / fs
    session_2_ts = 3600 + np.arange(session_samples) / fs
    ts = np.concatenate([session_1_ts, session_2_ts])
    iso = np.arange(ts.size, dtype=float)
    exp = iso + 100
    iso_stream = {"fs": fs, "data": iso, "ts": ts, "channel": [1]}
    exp_stream = {"fs": fs, "data": exp, "ts": ts, "channel": [1]}

    result = sessionize_stream_pair(iso_stream, exp_stream)

    assert result.isosbestic_405.shape == (session_samples, 2)
    assert result.calcium_465.shape == (session_samples, 2)
    assert result.session_start_times == pytest.approx([0.0, 3600.0])
    assert result.isosbestic_405[:, 0] == pytest.approx(iso[:session_samples])
    assert result.isosbestic_405[:, 1] == pytest.approx(iso[session_samples:])


def test_detect_stream_gaps_reports_matching_gaps() -> None:
    fs = 10.0
    ts = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    data = np.arange(ts.size, dtype=float)
    iso_stream = {"fs": fs, "data": data, "ts": ts}
    exp_stream = {"fs": fs, "data": data + 1, "ts": ts}

    report = detect_stream_gaps(iso_stream, exp_stream)

    assert report.is_consistent is True
    assert report.gap_threshold_seconds == pytest.approx(1.0)
    assert report.gap_indices == pytest.approx([3])
    assert report.gap_durations == pytest.approx([9.8])
    assert report.session_start_indices == pytest.approx([0, 3])
    assert report.session_stop_indices == pytest.approx([3, 6])
    assert report.session_lengths == pytest.approx([3, 3])
    assert report.session_start_times == pytest.approx([0.0, 10.0])
    assert report.session_stop_times == pytest.approx([0.2, 10.2])


def test_detect_stream_gaps_treats_timestamp_resets_as_gaps() -> None:
    fs = 10.0
    ts = np.array([0.0, 0.1, 0.2, 0.0, 0.1])
    data = np.arange(ts.size, dtype=float)
    stream = {"fs": fs, "data": data, "ts": ts}

    report = detect_stream_gaps(stream, stream)

    assert report.gap_indices == pytest.approx([3])
    assert report.gap_durations == pytest.approx([-0.2])


def test_detect_stream_gaps_rejects_mismatched_gap_indices() -> None:
    fs = 10.0
    iso_ts = np.array([0.0, 0.1, 0.2, 10.0, 10.1])
    exp_ts = np.array([0.0, 0.1, 10.0, 10.1, 10.2])
    data = np.arange(iso_ts.size, dtype=float)
    iso_stream = {"fs": fs, "data": data, "ts": iso_ts}
    exp_stream = {"fs": fs, "data": data, "ts": exp_ts}

    with pytest.raises(ValueError, match="timestamp gaps do not match"):
        detect_stream_gaps(iso_stream, exp_stream)


def test_detect_stream_gaps_rejects_mismatched_timestamp_lengths() -> None:
    fs = 10.0
    iso_stream = {
        "fs": fs,
        "data": np.arange(4, dtype=float),
        "ts": np.arange(4, dtype=float) / fs,
    }
    exp_stream = {
        "fs": fs,
        "data": np.arange(5, dtype=float),
        "ts": np.arange(5, dtype=float) / fs,
    }

    with pytest.raises(ValueError, match="timestamp arrays must have the same length"):
        detect_stream_gaps(iso_stream, exp_stream)


def test_estimate_interval_hours_suggests_half_hour() -> None:
    fs = 10.0
    session_samples = 5
    starts = [0.0, 1800.01, 3600.02]
    ts = np.concatenate([start + np.arange(session_samples) / fs for start in starts])
    data = np.arange(ts.size, dtype=float)
    iso_stream = {"fs": fs, "data": data, "ts": ts}
    exp_stream = {"fs": fs, "data": data + 1, "ts": ts}

    estimate = estimate_interval_hours(iso_stream, exp_stream)

    assert estimate.suggested_interval_hours == pytest.approx(0.5)
    assert estimate.raw_median_interval_hours == pytest.approx(1800.01 / 3600)
    assert estimate.interval_hours_min == pytest.approx(1800.01 / 3600)
    assert estimate.interval_hours_max == pytest.approx(1800.01 / 3600)
    assert estimate.session_count == 3
    assert estimate.interval_count == 2
    assert estimate.is_regular is True


def test_estimate_interval_hours_suggests_one_hour() -> None:
    fs = 10.0
    session_samples = 5
    starts = [0.0, 3600.0, 7200.0]
    ts = np.concatenate([start + np.arange(session_samples) / fs for start in starts])
    data = np.arange(ts.size, dtype=float)
    stream = {"fs": fs, "data": data, "ts": ts}

    estimate = estimate_interval_hours(stream, stream)

    assert estimate.suggested_interval_hours == pytest.approx(1.0)
    assert estimate.raw_median_interval_hours == pytest.approx(1.0)
    assert estimate.is_regular is True


def test_estimate_interval_hours_marks_irregular_intervals() -> None:
    fs = 10.0
    session_samples = 5
    starts = [0.0, 1800.0, 3700.0]
    ts = np.concatenate([start + np.arange(session_samples) / fs for start in starts])
    data = np.arange(ts.size, dtype=float)
    stream = {"fs": fs, "data": data, "ts": ts}

    estimate = estimate_interval_hours(stream, stream)

    assert estimate.suggested_interval_hours == pytest.approx(0.5166666666666667)
    assert estimate.is_regular is False


def test_estimate_interval_hours_requires_at_least_two_sessions() -> None:
    fs = 10.0
    ts = np.arange(5, dtype=float) / fs
    data = np.arange(ts.size, dtype=float)
    stream = {"fs": fs, "data": data, "ts": ts}

    with pytest.raises(ValueError, match="at least two sessions"):
        estimate_interval_hours(stream, stream)


def test_estimate_interval_hours_rejects_mismatched_gaps() -> None:
    fs = 10.0
    iso_ts = np.array([0.0, 0.1, 0.2, 10.0, 10.1])
    exp_ts = np.array([0.0, 0.1, 10.0, 10.1, 10.2])
    data = np.arange(iso_ts.size, dtype=float)
    iso_stream = {"fs": fs, "data": data, "ts": iso_ts}
    exp_stream = {"fs": fs, "data": data, "ts": exp_ts}

    with pytest.raises(ValueError, match="timestamp gaps do not match"):
        estimate_interval_hours(iso_stream, exp_stream)


def test_sessionize_stream_pair_splits_fixed_duration_and_crops() -> None:
    fs = 10.0
    session_seconds = 5.0
    session_samples = int(fs * session_seconds)
    ts = np.arange(session_samples * 2) / fs
    iso = np.arange(ts.size, dtype=float)
    exp = iso + 1
    iso_stream = {"fs": fs, "data": iso, "ts": ts}
    exp_stream = {"fs": fs, "data": exp, "ts": ts}

    result = sessionize_stream_pair(
        iso_stream,
        exp_stream,
        session_duration_seconds=session_seconds,
        crop_seconds=0.5,
    )

    assert result.isosbestic_405.shape == (40, 2)
    assert result.isosbestic_405[0, 0] == pytest.approx(5)
    assert result.isosbestic_405[-1, 0] == pytest.approx(44)
    assert result.isosbestic_405[0, 1] == pytest.approx(55)
