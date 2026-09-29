from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from circadian_fiber_photometry.behavior import (
    EVENT_ROLE_INTERVAL,
    EVENT_ROLE_POINT,
    EVENT_TYPE_STRUCTURED,
    EVENT_TYPE_UNSTRUCTURED,
    SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
    SyntheticBehaviorCoMetricConfig,
    SyntheticBehaviorConfig,
    SyntheticBehaviorEventConfig,
    build_behavior_event_configs_from_ttl,
    generate_synthetic_behavior_csv,
    simulate_behavior,
)
from circadian_fiber_photometry.simulation import SyntheticTTLBehaviorEventSummary


def _representative_config() -> SyntheticBehaviorConfig:
    return SyntheticBehaviorConfig(
        duration_seconds=4.0,
        sampling_rate_hz=2.0,
        co_metrics=(
            SyntheticBehaviorCoMetricConfig(
                "Speed (m/s)",
                values=(0.0, 0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7),
                description="Synthetic movement speed",
            ),
        ),
        events=(
            SyntheticBehaviorEventConfig(
                output_column="Cue",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_POINT,
                point_times_seconds=(0.5, 2.5),
            ),
            SyntheticBehaviorEventConfig(
                output_column="Freezing",
                event_type=EVENT_TYPE_UNSTRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                intervals_seconds=((1.0, 2.5),),
            ),
            SyntheticBehaviorEventConfig(
                output_column="CS+ active",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                source_format=SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
                onset_column="CS+ ON activated",
                offset_column="CS + OFF activated",
                intervals_seconds=((0.5, 1.5), (2.5, 3.5)),
            ),
        ),
    )


def test_simulate_behavior_materializes_chronoxiv_source_formats() -> None:
    dataset = simulate_behavior(_representative_config())

    assert dataset.row_count == 8
    assert dataset.column_names == (
        "Time (s)",
        "Speed (m/s)",
        "Cue",
        "Freezing",
        "CS+ ON activated",
        "CS + OFF activated",
    )
    np.testing.assert_allclose(dataset.time_seconds, np.arange(8) / 2.0)
    np.testing.assert_array_equal(dataset.columns["Cue"], [0, 1, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(
        dataset.columns["Freezing"],
        [0, 0, 1, 1, 1, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        dataset.columns["CS+ ON activated"],
        [0, 1, 0, 0, 0, 1, 0, 0],
    )
    np.testing.assert_array_equal(
        dataset.columns["CS + OFF activated"],
        [0, 0, 0, 1, 0, 0, 0, 1],
    )

    mappings = {mapping.output_column: mapping for mapping in dataset.event_mappings}
    assert mappings["Cue"].source_column == "Cue"
    assert mappings["Freezing"].interpretation == "Subject-driven event"
    assert mappings["CS+ active"].onset_column == "CS+ ON activated"
    assert mappings["CS+ active"].offset_column == "CS + OFF activated"
    assert len(dataset.event_occurrences) == 5


def test_generate_synthetic_behavior_csv_writes_numeric_and_blank_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mouse_behavior.csv"

    summary = generate_synthetic_behavior_csv(path, _representative_config())

    assert summary.path == path
    assert summary.row_count == 8
    assert summary.realized_duration_seconds == pytest.approx(4.0)
    assert summary.co_metric_descriptions == {"Speed (m/s)": "Synthetic movement speed"}
    with path.open(newline="", encoding="utf-8") as file_in:
        rows = list(csv.DictReader(file_in))
    assert list(rows[0]) == list(summary.column_names)
    assert rows[0]["Time (s)"] == "0"
    assert rows[1]["Cue"] == "1"
    assert rows[3]["Speed (m/s)"] == ""
    assert rows[7]["CS + OFF activated"] == "1"


def test_generate_synthetic_behavior_csv_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "behavior.csv"
    first = generate_synthetic_behavior_csv(path, _representative_config())

    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_synthetic_behavior_csv(path, _representative_config())

    second = generate_synthetic_behavior_csv(
        path,
        _representative_config(),
        overwrite=True,
    )
    assert second.path == first.path


def test_simulate_behavior_reports_point_collisions_after_alignment() -> None:
    dataset = simulate_behavior(
        SyntheticBehaviorConfig(
            duration_seconds=2.0,
            sampling_rate_hz=2.0,
            events=(
                SyntheticBehaviorEventConfig(
                    "Cue",
                    point_times_seconds=(0.10, 0.20),
                ),
            ),
        )
    )

    np.testing.assert_array_equal(dataset.columns["Cue"], [1, 0, 0, 0])
    assert len(dataset.event_occurrences) == 2
    assert len(dataset.warnings) == 1
    assert "aligned to sample 0" in dataset.warnings[0]


def test_simulate_behavior_rejects_paired_intervals_that_collide() -> None:
    config = SyntheticBehaviorConfig(
        duration_seconds=4.0,
        sampling_rate_hz=2.0,
        events=(
            SyntheticBehaviorEventConfig(
                "Shelter",
                event_type=EVENT_TYPE_UNSTRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                source_format=SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
                intervals_seconds=((0.5, 1.5), (1.5, 2.5)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="share an onset/offset sample"):
        simulate_behavior(config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            SyntheticBehaviorConfig(1.0, 2.0),
            "at least one event or co-metric",
        ),
        (
            SyntheticBehaviorConfig(
                1.0,
                2.0,
                events=(
                    SyntheticBehaviorEventConfig(
                        "Point",
                        intervals_seconds=((0.0, 0.5),),
                    ),
                ),
            ),
            "point event 'Point' cannot define intervals_seconds",
        ),
        (
            SyntheticBehaviorConfig(
                1.0,
                2.0,
                events=(
                    SyntheticBehaviorEventConfig(
                        "Paired point",
                        source_format=SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
                    ),
                ),
            ),
            "must have event_role 'interval'",
        ),
        (
            SyntheticBehaviorConfig(
                1.0,
                2.0,
                co_metrics=(SyntheticBehaviorCoMetricConfig("Speed", (0.0,)),),
            ),
            "exactly 2 values",
        ),
        (
            SyntheticBehaviorConfig(
                1.0,
                2.0,
                events=(
                    SyntheticBehaviorEventConfig(
                        "Time (s)",
                        event_role=EVENT_ROLE_INTERVAL,
                        source_format=SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
                    ),
                ),
            ),
            "cannot reuse a time column",
        ),
        (
            SyntheticBehaviorConfig(
                1.0,
                2.0,
                events=(
                    SyntheticBehaviorEventConfig(
                        "Interval",
                        event_role=EVENT_ROLE_INTERVAL,
                        source_format=SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
                    ),
                ),
                co_metrics=(SyntheticBehaviorCoMetricConfig("Interval", (0.0, 0.0)),),
            ),
            "duplicate behavior column: Interval",
        ),
    ],
)
def test_simulate_behavior_validates_configuration(
    config: SyntheticBehaviorConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_behavior(config)


def test_build_behavior_event_configs_from_ttl_uses_series_relative_samples() -> None:
    events = (
        SyntheticTTLBehaviorEventSummary(
            code_name="reward",
            channel=1,
            pulse_count=2,
            series_number=2,
            sequence_start_sample=50,
            sequence_start_seconds=105.0,
            pulse_sample_indices=np.array([50, 51]),
            pulse_times_seconds=np.array([105.0, 105.1]),
        ),
        SyntheticTTLBehaviorEventSummary(
            code_name="entry",
            channel=1,
            pulse_count=1,
            series_number=2,
            sequence_start_sample=20,
            sequence_start_seconds=102.0,
            pulse_sample_indices=np.array([20]),
            pulse_times_seconds=np.array([102.0]),
        ),
        SyntheticTTLBehaviorEventSummary(
            code_name="reward",
            channel=1,
            pulse_count=2,
            series_number=1,
            sequence_start_sample=10,
            sequence_start_seconds=1.0,
            pulse_sample_indices=np.array([10, 11]),
            pulse_times_seconds=np.array([1.0, 1.1]),
        ),
    )

    configs = build_behavior_event_configs_from_ttl(
        events,
        fs=10.0,
        series_number=2,
        code_names=("entry", "reward"),
        output_columns={"entry": "Port entry", "reward": "Reward"},
    )

    assert [config.output_column for config in configs] == ["Port entry", "Reward"]
    assert configs[0].point_times_seconds == (2.0,)
    assert configs[1].point_times_seconds == (5.0,)
    assert all(config.interpretation == "TTL behavior event" for config in configs)
