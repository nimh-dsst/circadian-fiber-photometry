"""Adapters for already-loaded stream dictionaries.

The package does not read Doric or pickle files. These helpers accept the
in-memory dictionaries produced by an upstream loader and convert concatenated
1D streams into the session matrices used by the analysis functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .analysis import analyze_sessions
from .results import (
    CircadianAnalysisResult,
    IntervalHoursEstimate,
    SessionizedStreamPair,
    TimestampGapReport,
)


def detect_stream_gaps(
    iso_stream: Mapping[str, Any],
    exp_stream: Mapping[str, Any],
    *,
    fs: float | None = None,
    gap_threshold_seconds: float | None = None,
) -> TimestampGapReport:
    """Detect and validate matching timestamp gaps in paired streams.

    A gap is any timestamp reset/non-increasing step or any timestamp jump above
    ``gap_threshold_seconds``. Returned gap indices are sample indices where the
    next session starts.
    """

    _, iso_ts, iso_fs = _stream_arrays(iso_stream, "iso_stream")
    _, exp_ts, exp_fs = _stream_arrays(exp_stream, "exp_stream")
    stream_fs = _resolve_fs(fs, iso_fs, exp_fs)

    if iso_ts.size != exp_ts.size:
        raise ValueError(
            "iso_stream and exp_stream timestamp arrays must have the same length; "
            f"got {iso_ts.size} and {exp_ts.size}"
        )

    threshold = _resolve_gap_threshold(stream_fs, gap_threshold_seconds)
    iso_gap_indices, iso_gap_durations = _timestamp_gaps(iso_ts, threshold)
    exp_gap_indices, _ = _timestamp_gaps(exp_ts, threshold)

    if not np.array_equal(iso_gap_indices, exp_gap_indices):
        raise ValueError(_gap_mismatch_message(iso_gap_indices, exp_gap_indices))

    session_start_indices = np.concatenate([[0], iso_gap_indices])
    session_stop_indices = np.concatenate([iso_gap_indices, [iso_ts.size]])
    session_lengths = session_stop_indices - session_start_indices
    session_start_times = iso_ts[session_start_indices]
    session_stop_times = iso_ts[session_stop_indices - 1]

    return TimestampGapReport(
        gap_indices=iso_gap_indices,
        gap_durations=iso_gap_durations,
        session_start_indices=session_start_indices,
        session_stop_indices=session_stop_indices,
        session_start_times=session_start_times,
        session_stop_times=session_stop_times,
        session_lengths=session_lengths,
        gap_threshold_seconds=threshold,
        is_consistent=True,
    )


def estimate_interval_hours(
    iso_stream: Mapping[str, Any],
    exp_stream: Mapping[str, Any],
    *,
    fs: float | None = None,
    gap_threshold_seconds: float | None = None,
    regularity_tolerance_seconds: float = 1.0,
) -> IntervalHoursEstimate:
    """Estimate the user-facing ``interval_hours`` value from session starts."""

    if regularity_tolerance_seconds < 0:
        raise ValueError("regularity_tolerance_seconds must be non-negative")

    gap_report = detect_stream_gaps(
        iso_stream,
        exp_stream,
        fs=fs,
        gap_threshold_seconds=gap_threshold_seconds,
    )
    if gap_report.session_start_times.size < 2:
        raise ValueError(
            "at least two sessions are required to estimate interval_hours"
        )

    interval_seconds = np.diff(gap_report.session_start_times)
    interval_hours = interval_seconds / 3600
    raw_median = float(np.median(interval_hours))
    suggested = round(raw_median * 60) / 60
    median_seconds = float(np.median(interval_seconds))
    is_regular = bool(
        np.all(
            np.abs(interval_seconds - median_seconds) <= regularity_tolerance_seconds
        )
    )

    return IntervalHoursEstimate(
        suggested_interval_hours=float(suggested),
        raw_median_interval_hours=raw_median,
        interval_hours_min=float(np.min(interval_hours)),
        interval_hours_max=float(np.max(interval_hours)),
        interval_hours_mean=float(np.mean(interval_hours)),
        interval_hours_std=float(np.std(interval_hours, ddof=0)),
        session_count=int(gap_report.session_start_times.size),
        interval_count=int(interval_hours.size),
        is_regular=is_regular,
    )


def sessionize_stream_pair(
    iso_stream: Mapping[str, Any],
    exp_stream: Mapping[str, Any],
    *,
    fs: float | None = None,
    session_duration_seconds: float | None = None,
    gap_threshold_seconds: float | None = None,
    crop_seconds: float = 0.0,
    trim_to_shortest: bool = True,
) -> SessionizedStreamPair:
    """Convert paired stream dictionaries to ``(samples, sessions)`` arrays.

    Sessions are split by timestamp resets or large timestamp gaps. If there are
    no detectable timestamp boundaries, pass ``session_duration_seconds`` to
    split the concatenated arrays into fixed-size sessions.
    """

    iso_data, iso_ts, iso_fs = _stream_arrays(iso_stream, "iso_stream")
    exp_data, exp_ts, exp_fs = _stream_arrays(exp_stream, "exp_stream")
    stream_fs = _resolve_fs(fs, iso_fs, exp_fs)

    if iso_data.size != exp_data.size:
        raise ValueError(
            "iso_stream and exp_stream data arrays must have the same length; "
            f"got {iso_data.size} and {exp_data.size}"
        )
    if iso_ts.size != exp_ts.size:
        raise ValueError(
            "iso_stream and exp_stream timestamp arrays must have the same length; "
            f"got {iso_ts.size} and {exp_ts.size}"
        )
    if crop_seconds < 0:
        raise ValueError("crop_seconds must be non-negative")

    if session_duration_seconds is None:
        gap_report = detect_stream_gaps(
            iso_stream,
            exp_stream,
            fs=stream_fs,
            gap_threshold_seconds=gap_threshold_seconds,
        )
        segments = [
            slice(int(start), int(stop))
            for start, stop in zip(
                gap_report.session_start_indices,
                gap_report.session_stop_indices,
                strict=True,
            )
        ]
    else:
        segments = _session_slices(
            iso_ts,
            fs=stream_fs,
            session_duration_seconds=session_duration_seconds,
        )
    crop_samples = int(round(crop_seconds * stream_fs))

    iso_segments = []
    exp_segments = []
    start_times = []
    stop_times = []

    for segment in segments:
        start = segment.start or 0
        stop = segment.stop if segment.stop is not None else iso_data.size
        cropped_start = start + crop_samples
        cropped_stop = stop - crop_samples
        if cropped_stop <= cropped_start:
            continue
        iso_segments.append(iso_data[cropped_start:cropped_stop])
        exp_segments.append(exp_data[cropped_start:cropped_stop])
        start_times.append(iso_ts[start])
        stop_times.append(iso_ts[stop - 1])

    if not iso_segments:
        raise ValueError("no non-empty sessions remained after splitting and cropping")

    lengths = np.array([segment.size for segment in iso_segments])
    if not np.all(lengths == lengths[0]):
        if not trim_to_shortest:
            raise ValueError(
                "sessions have unequal lengths; pass trim_to_shortest=True or "
                "adjust session splitting/cropping"
            )
        target_len = int(np.min(lengths))
    else:
        target_len = int(lengths[0])

    if target_len < 2:
        raise ValueError("sessions must contain at least two samples")

    iso_matrix = np.column_stack([segment[:target_len] for segment in iso_segments])
    exp_matrix = np.column_stack([segment[:target_len] for segment in exp_segments])

    return SessionizedStreamPair(
        isosbestic_405=iso_matrix,
        calcium_465=exp_matrix,
        fs=stream_fs,
        session_start_times=np.asarray(start_times, dtype=float),
        session_stop_times=np.asarray(stop_times, dtype=float),
        samples_per_session=target_len,
    )


def analyze_stream_pair(
    iso_stream: Mapping[str, Any],
    exp_stream: Mapping[str, Any],
    *,
    interval_hours: float = 1,
    fitting_cutoff: float = 0,
    weight_fit: bool = True,
    fs: float | None = None,
    session_duration_seconds: float | None = None,
    gap_threshold_seconds: float | None = None,
    crop_seconds: float = 0.0,
    trim_to_shortest: bool = True,
) -> CircadianAnalysisResult:
    """Sessionize paired stream dictionaries and run ``analyze_sessions``."""

    sessionized = sessionize_stream_pair(
        iso_stream,
        exp_stream,
        fs=fs,
        session_duration_seconds=session_duration_seconds,
        gap_threshold_seconds=gap_threshold_seconds,
        crop_seconds=crop_seconds,
        trim_to_shortest=trim_to_shortest,
    )
    return analyze_sessions(
        sessionized.isosbestic_405,
        sessionized.calcium_465,
        fs=sessionized.fs,
        interval_hours=interval_hours,
        fitting_cutoff=fitting_cutoff,
        weight_fit=weight_fit,
    )


def _stream_arrays(
    stream: Mapping[str, Any],
    name: str,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Extract and validate data, timestamps, and fs from a stream mapping."""

    try:
        data = np.asarray(stream["data"], dtype=float).reshape(-1)
        ts = np.asarray(stream["ts"], dtype=float).reshape(-1)
    except KeyError as exc:
        raise ValueError(f"{name} is missing required key {exc.args[0]!r}") from exc

    if data.size == 0:
        raise ValueError(f"{name} data must not be empty")
    if data.size != ts.size:
        raise ValueError(
            f"{name} data and ts arrays must have the same length; "
            f"got {data.size} and {ts.size}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} data contains NaN or infinite values")
    if not np.all(np.isfinite(ts)):
        raise ValueError(f"{name} ts contains NaN or infinite values")

    stream_fs = stream.get("fs")
    return data, ts, None if stream_fs is None else float(stream_fs)


def _resolve_fs(
    explicit_fs: float | None,
    iso_fs: float | None,
    exp_fs: float | None,
) -> float:
    """Resolve sampling rate from explicit input or stream metadata."""

    candidates = [value for value in (explicit_fs, iso_fs, exp_fs) if value is not None]
    if not candidates:
        raise ValueError("sampling rate is required via fs or stream['fs']")
    resolved = float(candidates[0])
    if resolved <= 0:
        raise ValueError("fs must be positive")
    for candidate in candidates[1:]:
        if not np.isclose(candidate, resolved):
            raise ValueError(
                "stream sampling rates do not match; "
                f"got {resolved} and {candidate}"
            )
    return resolved


def _session_slices(
    timestamps: np.ndarray,
    *,
    fs: float,
    session_duration_seconds: float | None,
) -> list[slice]:
    """Find session slices from a fixed duration."""

    if session_duration_seconds is None:
        raise ValueError("session_duration_seconds is required")
    if session_duration_seconds <= 0:
        raise ValueError("session_duration_seconds must be positive")
    samples_per_session = int(round(session_duration_seconds * fs))
    if samples_per_session < 2:
        raise ValueError("session_duration_seconds produces fewer than 2 samples")
    session_count = timestamps.size // samples_per_session
    if session_count == 0:
        raise ValueError("not enough samples for one complete session")
    return [
        slice(index * samples_per_session, (index + 1) * samples_per_session)
        for index in range(session_count)
    ]


def _resolve_gap_threshold(
    fs: float,
    gap_threshold_seconds: float | None,
) -> float:
    """Resolve timestamp gap threshold from explicit input or sampling rate."""

    threshold = (
        max(1.0, 5.0 / fs)
        if gap_threshold_seconds is None
        else gap_threshold_seconds
    )
    if threshold <= 0:
        raise ValueError("gap_threshold_seconds must be positive")
    return float(threshold)


def _timestamp_gaps(
    timestamps: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gap start indices and timestamp jumps before each gap."""

    diffs = np.diff(timestamps)
    gap_mask = (diffs <= 0) | (diffs > threshold)
    gap_indices = np.flatnonzero(gap_mask) + 1
    gap_durations = diffs[gap_indices - 1]
    return gap_indices.astype(int), gap_durations.astype(float)


def _gap_mismatch_message(
    iso_gap_indices: np.ndarray,
    exp_gap_indices: np.ndarray,
) -> str:
    """Build an actionable message for inconsistent stream gaps."""

    shared_len = min(iso_gap_indices.size, exp_gap_indices.size)
    if shared_len:
        mismatch_positions = np.flatnonzero(
            iso_gap_indices[:shared_len] != exp_gap_indices[:shared_len]
        )
    else:
        mismatch_positions = np.array([], dtype=int)

    if mismatch_positions.size:
        positions = mismatch_positions[:5]
        first_mismatches = [
            (
                int(position),
                int(iso_gap_indices[position]),
                int(exp_gap_indices[position]),
            )
            for position in positions
        ]
    else:
        start = shared_len
        stop = min(start + 5, max(iso_gap_indices.size, exp_gap_indices.size))
        first_mismatches = [
            (
                int(position),
                _optional_gap_index(iso_gap_indices, position),
                _optional_gap_index(exp_gap_indices, position),
            )
            for position in range(start, stop)
        ]

    return (
        "iso_stream and exp_stream timestamp gaps do not match; "
        f"iso gaps={iso_gap_indices.size}, exp gaps={exp_gap_indices.size}, "
        "first mismatches as (gap_position, iso_index, exp_index)="
        f"{first_mismatches}"
    )


def _optional_gap_index(gap_indices: np.ndarray, position: int) -> int | None:
    """Return a gap index if present at position."""

    if position >= gap_indices.size:
        return None
    return int(gap_indices[position])
