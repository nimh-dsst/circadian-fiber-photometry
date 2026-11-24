from __future__ import annotations

import csv
import runpy
from pathlib import Path

import numpy as np
import pytest

from circadian_fiber_photometry.behavior import simulate_behavior

SCRIPT = (
    Path(__file__).parents[1] / "examples" / "generate_chronoxiv_mpfc_loom_dataset.py"
)
MODULE = runpy.run_path(SCRIPT)


def test_loom_dataset_has_eight_single_site_subjects_and_one_session() -> None:
    assert MODULE["SUBJECT_IDS"] == tuple(
        f"Subject{number:02d}" for number in range(1, 9)
    )
    assert MODULE["SITE_ID"] == "medial-prefrontal-cortex"
    assert MODULE["SESSION_COUNT"] == 1
    assert MODULE["SESSION_DURATION_SECONDS"] == 20 * 60
    assert len(MODULE["_session_datetimes"]()) == 1

    name = MODULE["_target_name"](
        "Subject01",
        MODULE["_session_datetimes"]()[0],
    )
    assert name == "Subject01__medial-prefrontal-cortex__M-20260202-090000"


def test_subject_protocol_has_random_nonoverlapping_post_baseline_looms() -> None:
    protocol = MODULE["_subject_protocol"](0)
    repeated = MODULE["_subject_protocol"](0)

    assert protocol == repeated
    assert len(protocol.sessions) == 1
    flattened = [item for session in protocol.sessions for item in session]
    assert len(flattened) == 10
    assert [item.exposure_number for item in flattened] == list(range(1, 11))
    probabilities = np.asarray([item.response_probability for item in flattened])
    assert np.all(np.diff(probabilities) < 0)
    assert probabilities[0] == pytest.approx(0.95)
    assert probabilities[-1] == pytest.approx(0.95 * 0.96**9)
    assert 0 < protocol.response_count < len(flattened)

    all_onsets: list[float] = []
    for session in protocol.sessions:
        assert len(session) == 10
        onsets = [item.onset_seconds for item in session]
        all_onsets.extend(onsets)
        assert all(item.onset_seconds >= 300 for item in session)
        assert all(item.offset_seconds <= 1200 for item in session)
        assert all(
            item.offset_seconds - item.onset_seconds == pytest.approx(30.0)
            for item in session
        )
        gaps = [
            later.onset_seconds - earlier.offset_seconds
            for earlier, later in zip(session[:-1], session[1:], strict=True)
        ]
        assert min(gaps) >= 15.0 - 1e-9
    assert len(set(all_onsets)) == 10


def test_behavior_config_materializes_ten_thirty_second_loom_bouts() -> None:
    protocol = MODULE["_subject_protocol"](0)
    dataset = simulate_behavior(MODULE["_behavior_config"](protocol.sessions[0]))

    assert dataset.row_count == 20 * 60 * 30
    assert dataset.column_names == ("Time (s)", "loom")
    loom = dataset.columns["loom"]
    assert int(np.sum(loom)) == 10 * 30 * 30
    assert int(np.count_nonzero(np.diff(loom, prepend=0) == 1)) == 10
    mapping = dataset.event_mappings[0]
    assert mapping.output_column == "loom"
    assert mapping.event_type == "Structured"
    assert mapping.event_role == "interval"


def test_doric_config_aligns_ttl_and_probabilistic_preloom_calcium() -> None:
    protocol = MODULE["_subject_protocol"](1)
    config = MODULE["_simulation_config"](1, protocol)

    assert config.channel_count == 1
    assert config.series_count == 1
    assert len(config.ttl_behavior_codes) == 1
    assert config.ttl_behavior_codes[0].name == "loom"
    assert len(config.ttl_behavior_events) == 1

    calcium_sources = {
        source.series_numbers[0]: source
        for source in config.signal.scheduled_calcium_events
    }
    for series_number, session in enumerate(protocol.sessions, start=1):
        ttl = config.ttl_behavior_events[series_number - 1]
        assert ttl.series_numbers == (series_number,)
        assert ttl.start_seconds == tuple(item.onset_seconds for item in session)

        expected_responses = tuple(
            item.onset_seconds - 1.0
            for item in session
            if item.calcium_response_occurred
        )
        if expected_responses:
            assert calcium_sources[series_number].event_times_seconds == (
                expected_responses
            )
            assert calcium_sources[series_number].amplitude == pytest.approx(0.035)
        else:
            assert series_number not in calcium_sources


def test_behavior_csv_and_ground_truth_use_separate_paths(tmp_path: Path) -> None:
    protocol = MODULE["_subject_protocol"](0)
    behavior_path = tmp_path / MODULE["_behavior_filename"](
        protocol.subject_id,
        MODULE["_session_datetimes"]()[0],
    )
    generate = MODULE["generate_synthetic_behavior_csv"]

    summary = generate(behavior_path, MODULE["_behavior_config"](protocol.sessions[0]))
    ground_truth_path = MODULE["_write_ground_truth"](
        tmp_path,
        (protocol,),
    )

    assert summary.path == behavior_path
    with behavior_path.open(encoding="utf-8", newline="") as file_in:
        assert next(csv.reader(file_in)) == ["Time (s)", "loom"]
    with ground_truth_path.open(encoding="utf-8", newline="") as file_in:
        rows = list(csv.DictReader(file_in))
    assert len(rows) == 10
    assert rows[0]["subject_id"] == "Subject01"
    assert rows[0]["behavior_csv"].startswith("Behavior/Cohort 1/Simulated CSV/")
