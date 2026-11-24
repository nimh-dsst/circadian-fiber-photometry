"""Generate a ChronoXIV-uploadable synthetic behavior CSV.

Run from the repository root with::

    uv run python examples/behavior_simulation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from circadian_fiber_photometry.behavior import (
    EVENT_ROLE_INTERVAL,
    EVENT_TYPE_STRUCTURED,
    EVENT_TYPE_UNSTRUCTURED,
    SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
    SyntheticBehaviorCoMetricConfig,
    SyntheticBehaviorConfig,
    SyntheticBehaviorEventConfig,
    generate_synthetic_behavior_csv,
)

DURATION_SECONDS = 660.0
SAMPLING_RATE_HZ = 30.0
SEED = 123


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "generated" / "simulated_behavior.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _movement_co_metrics() -> tuple[SyntheticBehaviorCoMetricConfig, ...]:
    """Build a small seeded random walk for demonstration purposes."""

    sample_count = int(round(DURATION_SECONDS * SAMPLING_RATE_HZ))
    rng = np.random.default_rng(SEED)
    steps = rng.normal(0.0, 1.2, size=(sample_count, 2))
    position = np.cumsum(steps, axis=0)
    position -= np.min(position, axis=0)
    position += np.array([150.0, 170.0])
    speed = np.linalg.norm(steps, axis=1) * SAMPLING_RATE_HZ / 1000.0
    return (
        SyntheticBehaviorCoMetricConfig("Centre position X", position[:, 0]),
        SyntheticBehaviorCoMetricConfig("Centre position Y", position[:, 1]),
        SyntheticBehaviorCoMetricConfig("Speed (m/s)", speed),
    )


def _config() -> SyntheticBehaviorConfig:
    cs_intervals = ((120.0, 150.0), (300.0, 330.0), (480.0, 510.0))
    return SyntheticBehaviorConfig(
        duration_seconds=DURATION_SECONDS,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        co_metrics=_movement_co_metrics(),
        events=(
            SyntheticBehaviorEventConfig(
                output_column="Freezing",
                event_type=EVENT_TYPE_UNSTRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                intervals_seconds=(
                    (18.25, 20.30),
                    (64.17, 67.90),
                    (149.72, 154.28),
                    (474.08, 481.85),
                ),
            ),
            SyntheticBehaviorEventConfig(
                output_column="House light active",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                intervals_seconds=((0.0, DURATION_SECONDS),),
            ),
            SyntheticBehaviorEventConfig(
                output_column="New speaker active",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                intervals_seconds=cs_intervals,
            ),
            SyntheticBehaviorEventConfig(
                output_column="New shocker active",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
            ),
            SyntheticBehaviorEventConfig(
                output_column="CS+ active",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                source_format=SOURCE_FORMAT_PAIRED_ONSET_OFFSET,
                onset_column="CS+ ON activated",
                offset_column="CS + OFF activated",
                intervals_seconds=cs_intervals,
            ),
        ),
    )


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = generate_synthetic_behavior_csv(
        args.output,
        _config(),
        overwrite=args.overwrite,
    )
    print(f"Generated {summary.row_count:,} behavior rows at {summary.path}")
    for mapping in summary.event_mappings:
        print(
            f"{mapping.output_column}: {mapping.event_type} "
            f"{mapping.event_role}, {mapping.source_format}"
        )


if __name__ == "__main__":
    main()
