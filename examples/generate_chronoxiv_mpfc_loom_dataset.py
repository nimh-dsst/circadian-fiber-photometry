"""Generate a single-site mPFC loom-conditioning dataset for ChronoXIV.

The output contains eight subjects, one medial prefrontal cortex site per
subject, and one 20-minute session per subject. Each session has a five-minute
baseline followed by a 15-minute conditioning period containing ten random,
non-overlapping 30-second ``loom`` bouts.

Every loom exposure has a Bernoulli calcium response. The response probability
starts at 0.95 and is multiplied by 0.96 after every exposure. Realized
responses begin one second before loom onset and have a fixed
calcium-transient amplitude. Loom onset is also written to the simulated TDT
TTL stream and EPOC store.

The generated tree keeps raw behavior uploads separate from simulated TDT
blocks::

    Photometry/Cohort 1/Simulated TDT/
      Subject01__medial-prefrontal-cortex__M-20260202-090000/
        tdt_block.pkl
    Behavior/Cohort 1/Simulated CSV/
      Subject01__medial-prefrontal-cortex__M-20260202-090000_behavior.csv
    Ground Truth/
      loom_response_ground_truth.csv

Run from the repository root with::

    uv run python examples/generate_chronoxiv_mpfc_loom_dataset.py

Use ``--overwrite`` to replace a previously generated dataset.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from circadian_fiber_photometry import load_doric
from circadian_fiber_photometry.behavior import (
    EVENT_ROLE_INTERVAL,
    EVENT_TYPE_STRUCTURED,
    SyntheticBehaviorConfig,
    SyntheticBehaviorEventConfig,
    generate_synthetic_behavior_csv,
)
from circadian_fiber_photometry.io import (
    TDTExtractMetadata,
    TDTExtractStream,
    write_tdt_block,
)
from circadian_fiber_photometry.simulation import (
    SyntheticDoricConfig,
    SyntheticPhotobleachingComponentConfig,
    SyntheticPhotobleachingConfig,
    SyntheticSignalConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    add_gaussian_noise,
    add_scheduled_calcium_events,
    build_tdt_epocs_from_behavior_events,
    configure_session_start_spike,
    generate_synthetic_doric,
)

SUBJECT_IDS = tuple(f"Subject{number:02d}" for number in range(1, 9))
SUBJECT_SEXES = {
    subject_id: "M" if index <= 4 else "F"
    for index, subject_id in enumerate(SUBJECT_IDS, start=1)
}
SITE_ID = "medial-prefrontal-cortex"
COHORT = "Cohort 1"
BASE_DATETIME = datetime(2026, 2, 2, 9, 0, 0)

SESSION_COUNT = 1
SESSION_DURATION_SECONDS = 20 * 60
INTER_SESSION_GAP_SECONDS = 0.0
BASELINE_DURATION_SECONDS = 5 * 60
CONDITIONING_DURATION_SECONDS = 15 * 60

LOOMS_PER_SESSION = 10
LOOM_DURATION_SECONDS = 30.0
MINIMUM_INTER_LOOM_GAP_SECONDS = 15.0
RESPONSE_INITIAL_PROBABILITY = 0.95
RESPONSE_PROBABILITY_DECAY = 0.96
PRE_LOOM_CALCIUM_LEAD_SECONDS = 1.0
CALCIUM_RESPONSE_AMPLITUDE = 0.035

PHOTOMETRY_SAMPLING_RATE_HZ = 1017.2526245117188
BEHAVIOR_SAMPLING_RATE_HZ = 30.0
ISOSBESTIC_BASELINE = 0.090
EXPERIMENTAL_BASELINE = 0.180
STARTUP_ARTIFACT_SECONDS = 2.0
TTL_PULSE_WIDTH_SECONDS = 0.125
SEED = 20260202

TDT_BLOCK_FOLDER_RE = re.compile(
    r"^(?P<subject>[A-Za-z0-9._-]+)"
    r"__(?P<site>[A-Za-z0-9-]+)"
    r"__(?P<sex>[MFUO])-"
    r"(?P<datetime>\d{8}-\d{6})$"
)
GROUND_TRUTH_COLUMNS = (
    "subject_id",
    "session_number",
    "session_datetime",
    "loom_number_in_session",
    "exposure_number",
    "loom_onset_seconds",
    "loom_offset_seconds",
    "response_probability",
    "calcium_response_occurred",
    "calcium_event_seconds",
    "tdt_block_directory",
    "behavior_csv",
)


@dataclass(frozen=True)
class LoomOccurrence:
    """One scheduled loom and its probabilistic calcium ground truth."""

    session_number: int
    loom_number_in_session: int
    exposure_number: int
    onset_seconds: float
    offset_seconds: float
    response_probability: float
    calcium_response_occurred: bool
    calcium_event_seconds: float | None


@dataclass(frozen=True)
class SubjectProtocol:
    """The conditioning session for one subject."""

    subject_id: str
    sessions: tuple[tuple[LoomOccurrence, ...], ...]

    @property
    def response_count(self) -> int:
        return sum(
            occurrence.calcium_response_occurred
            for session in self.sessions
            for occurrence in session
        )


def _parse_args() -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parent
        / "generated"
        / "chronoxiv_mpfc_loom_conditioning"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"dataset root (default: {default_output})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output root if it already exists",
    )
    return parser.parse_args()


def _photometry_seconds_to_samples(seconds: float) -> int:
    return int(round(seconds * PHOTOMETRY_SAMPLING_RATE_HZ))


def _behavior_seconds_to_samples(seconds: float) -> int:
    return int(round(seconds * BEHAVIOR_SAMPLING_RATE_HZ))


def _session_datetimes() -> tuple[datetime, ...]:
    return (BASE_DATETIME,)


def _target_name(subject_id: str, session_datetime: datetime) -> str:
    try:
        sex = SUBJECT_SEXES[subject_id]
    except KeyError as error:
        raise ValueError(f"no sex designation configured for {subject_id!r}") from error
    name = f"{subject_id}__{SITE_ID}__{sex}-{session_datetime:%Y%m%d-%H%M%S}"
    if TDT_BLOCK_FOLDER_RE.fullmatch(name) is None:
        raise RuntimeError(f"generated TDT block folder name is invalid: {name}")
    return name


def _behavior_filename(subject_id: str, session_datetime: datetime) -> str:
    return f"{_target_name(subject_id, session_datetime)}_behavior.csv"


def _subject_protocol(subject_index: int) -> SubjectProtocol:
    """Draw reproducible loom times and response outcomes for one subject."""

    if subject_index < 0 or subject_index >= len(SUBJECT_IDS):
        raise ValueError("subject_index is outside SUBJECT_IDS")
    subject_id = SUBJECT_IDS[subject_index]
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, subject_index, 0x4C4F4F4D])
    )
    sessions: list[tuple[LoomOccurrence, ...]] = []

    for session_number in range(1, SESSION_COUNT + 1):
        session_occurrences: list[LoomOccurrence] = []
        loom_onsets = _random_loom_onsets(rng)
        for loom_index, onset_seconds in enumerate(loom_onsets):
            exposure_index = (session_number - 1) * LOOMS_PER_SESSION + loom_index
            offset_seconds = onset_seconds + LOOM_DURATION_SECONDS
            response_probability = RESPONSE_INITIAL_PROBABILITY * (
                RESPONSE_PROBABILITY_DECAY**exposure_index
            )
            responded = bool(rng.random() < response_probability)
            calcium_event_seconds = (
                onset_seconds - PRE_LOOM_CALCIUM_LEAD_SECONDS if responded else None
            )
            session_occurrences.append(
                LoomOccurrence(
                    session_number=session_number,
                    loom_number_in_session=loom_index + 1,
                    exposure_number=exposure_index + 1,
                    onset_seconds=onset_seconds,
                    offset_seconds=offset_seconds,
                    response_probability=response_probability,
                    calcium_response_occurred=responded,
                    calcium_event_seconds=calcium_event_seconds,
                )
            )
        sessions.append(tuple(session_occurrences))

    protocol = SubjectProtocol(subject_id=subject_id, sessions=tuple(sessions))
    _validate_subject_protocol(protocol)
    return protocol


def _random_loom_onsets(rng: np.random.Generator) -> tuple[float, ...]:
    """Place random behavior-aligned looms with a fixed minimum gap."""

    first_start = _behavior_seconds_to_samples(BASELINE_DURATION_SECONDS)
    latest_start = _behavior_seconds_to_samples(
        SESSION_DURATION_SECONDS - LOOM_DURATION_SECONDS
    )
    duration_samples = _behavior_seconds_to_samples(LOOM_DURATION_SECONDS)
    gap_samples = _behavior_seconds_to_samples(MINIMUM_INTER_LOOM_GAP_SECONDS)
    accepted: list[int] = []
    candidates = rng.permutation(np.arange(first_start, latest_start + 1))
    for candidate in candidates:
        candidate_start = int(candidate)
        candidate_stop = candidate_start + duration_samples
        if any(
            candidate_start < existing + duration_samples + gap_samples
            and existing < candidate_stop + gap_samples
            for existing in accepted
        ):
            continue
        accepted.append(candidate_start)
        if len(accepted) == LOOMS_PER_SESSION:
            return tuple(
                sample / BEHAVIOR_SAMPLING_RATE_HZ for sample in sorted(accepted)
            )
    raise RuntimeError("could not place the requested random loom bouts")


def _validate_subject_protocol(protocol: SubjectProtocol) -> None:
    if len(protocol.sessions) != SESSION_COUNT:
        raise RuntimeError("subject protocol has an unexpected session count")
    flattened = [occurrence for session in protocol.sessions for occurrence in session]
    if len(flattened) != SESSION_COUNT * LOOMS_PER_SESSION:
        raise RuntimeError("subject protocol has an unexpected loom count")
    if [item.exposure_number for item in flattened] != list(
        range(1, len(flattened) + 1)
    ):
        raise RuntimeError("loom exposure numbers are not consecutive")
    probabilities = np.asarray(
        [item.response_probability for item in flattened],
        dtype=float,
    )
    if np.any(np.diff(probabilities) >= 0):
        raise RuntimeError("calcium response probabilities do not decrease")

    conditioning_stop = BASELINE_DURATION_SECONDS + CONDITIONING_DURATION_SECONDS
    for session_number, session in enumerate(protocol.sessions, start=1):
        if len(session) != LOOMS_PER_SESSION:
            raise RuntimeError(f"session {session_number} has an unexpected loom count")
        previous_offset = -np.inf
        for occurrence in session:
            if occurrence.onset_seconds < BASELINE_DURATION_SECONDS:
                raise RuntimeError("a loom begins during baseline")
            if occurrence.offset_seconds > conditioning_stop:
                raise RuntimeError("a loom extends beyond the session")
            if not np.isclose(
                occurrence.offset_seconds - occurrence.onset_seconds,
                LOOM_DURATION_SECONDS,
            ):
                raise RuntimeError("a loom does not last 30 seconds")
            if occurrence.onset_seconds - previous_offset < (
                MINIMUM_INTER_LOOM_GAP_SECONDS - 1e-9
            ):
                raise RuntimeError("loom bouts do not retain their minimum gap")
            if occurrence.calcium_response_occurred:
                expected_event_time = (
                    occurrence.onset_seconds - PRE_LOOM_CALCIUM_LEAD_SECONDS
                )
                if not np.isclose(
                    occurrence.calcium_event_seconds,
                    expected_event_time,
                ):
                    raise RuntimeError("a calcium response is not before loom onset")
            elif occurrence.calcium_event_seconds is not None:
                raise RuntimeError("a non-response has a calcium event time")
            previous_offset = occurrence.offset_seconds


def _signal_config(protocol: SubjectProtocol) -> SyntheticSignalConfig:
    """Build a single-site signal with only realized pre-loom responses."""

    signal = SyntheticSignalConfig(
        isosbestic_baseline=ISOSBESTIC_BASELINE,
        calcium_baseline=EXPERIMENTAL_BASELINE,
        channel_baseline_step=0.0,
        artifact_amplitude=0.0,
        circadian_amplitude=0.0,
        noise_std=0.0,
        analog_noise_std=0.0,
        transient_rate_per_minute=0.0,
        transient_amplitude=0.0,
        photobleaching=SyntheticPhotobleachingConfig(
            model="single_exponential",
            isosbestic_components=(
                SyntheticPhotobleachingComponentConfig(
                    amplitude_fraction=0.12,
                    time_constant_seconds=7200.0,
                ),
            ),
            calcium_components=(
                SyntheticPhotobleachingComponentConfig(
                    amplitude_fraction=0.18,
                    time_constant_seconds=6000.0,
                ),
            ),
            turnover_half_life_hours=48.0,
        ),
    )
    signal = add_gaussian_noise(
        signal,
        isosbestic_std=0.0010,
        calcium_std=0.0020,
        name="measurement noise",
    )
    signal = configure_session_start_spike(
        signal,
        duration_seconds=STARTUP_ARTIFACT_SECONDS,
        magnitude_fraction=0.50,
        name="startup artifact",
    )
    for series_number, session in enumerate(protocol.sessions, start=1):
        response_times = tuple(
            float(occurrence.calcium_event_seconds)
            for occurrence in session
            if occurrence.calcium_event_seconds is not None
        )
        if response_times:
            signal = add_scheduled_calcium_events(
                signal,
                response_times,
                amplitude=CALCIUM_RESPONSE_AMPLITUDE,
                rise_rate_per_second=9.0,
                fall_rate_per_second=1.0,
                channels=(1,),
                series_numbers=(series_number,),
                name="pre-loom calcium response",
            )
    return signal


def _simulation_config(
    subject_index: int,
    protocol: SubjectProtocol,
) -> SyntheticDoricConfig:
    ttl_events = tuple(
        SyntheticTTLBehaviorEventConfig(
            "loom",
            start_seconds=tuple(item.onset_seconds for item in session),
            series_numbers=(series_number,),
        )
        for series_number, session in enumerate(protocol.sessions, start=1)
    )
    return SyntheticDoricConfig(
        series_count=SESSION_COUNT,
        session_duration_seconds=SESSION_DURATION_SECONDS,
        inter_series_gap_seconds=INTER_SESSION_GAP_SECONDS,
        fs=PHOTOMETRY_SAMPLING_RATE_HZ,
        channel_count=1,
        seed=SEED + subject_index,
        signal=_signal_config(protocol),
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("loom", channel=1, pulse_count=1),
        ),
        ttl_behavior_events=ttl_events,
        ttl_pulse_width_seconds=TTL_PULSE_WIDTH_SECONDS,
        ttl_pulse_off_interval_seconds=TTL_PULSE_WIDTH_SECONDS,
    )


def _behavior_config(
    session: tuple[LoomOccurrence, ...],
) -> SyntheticBehaviorConfig:
    return SyntheticBehaviorConfig(
        duration_seconds=SESSION_DURATION_SECONDS,
        sampling_rate_hz=BEHAVIOR_SAMPLING_RATE_HZ,
        events=(
            SyntheticBehaviorEventConfig(
                output_column="loom",
                event_type=EVENT_TYPE_STRUCTURED,
                event_role=EVENT_ROLE_INTERVAL,
                intervals_seconds=tuple(
                    (item.onset_seconds, item.offset_seconds) for item in session
                ),
                interpretation="Thirty-second loom conditioning bout",
            ),
        ),
    )


def _preflight_output(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden:
        raise ValueError(f"refusing to use broad output root: {resolved}")
    if resolved.exists() and not overwrite:
        raise FileExistsError(
            f"{resolved} already exists; pass --overwrite to replace it"
        )
    if resolved.exists() and (not resolved.is_dir() or resolved.is_symlink()):
        raise ValueError("an existing output root must be a real directory")


def _validate_source_simulation(
    protocol: SubjectProtocol,
    config: SyntheticDoricConfig,
    simulation_summary,
) -> None:
    if simulation_summary.photobleaching.model != "single_exponential":
        raise RuntimeError("single-exponential photobleaching was not applied")
    if len(simulation_summary.ttl_behavior_events) != (
        SESSION_COUNT * LOOMS_PER_SESSION
    ):
        raise RuntimeError("simulated TTL loom ground truth is incomplete")

    for series_number, session in enumerate(protocol.sessions, start=1):
        expected_response_samples = np.asarray(
            [
                _photometry_seconds_to_samples(float(item.calcium_event_seconds))
                for item in session
                if item.calcium_event_seconds is not None
            ],
            dtype=int,
        )
        np.testing.assert_array_equal(
            simulation_summary.event_sample_indices[(series_number, 1)],
            expected_response_samples,
        )
        ttl_events = [
            event
            for event in simulation_summary.ttl_behavior_events
            if event.series_number == series_number
        ]
        expected_ttl_samples = np.asarray(
            [_photometry_seconds_to_samples(item.onset_seconds) for item in session],
            dtype=int,
        )
        np.testing.assert_array_equal(
            [event.sequence_start_sample for event in ttl_events],
            expected_ttl_samples,
        )
        if any(event.code_name != "loom" for event in ttl_events):
            raise RuntimeError("an unexpected TTL behavior code was generated")
    if config.channel_count != 1:
        raise RuntimeError("the mPFC source simulation is not single-site")


def _export_dataset(
    stage_root: Path,
    *,
    final_output_root: Path,
) -> tuple[int, int, tuple[SubjectProtocol, ...]]:
    tdt_inbox = stage_root / "Photometry" / COHORT / "Simulated TDT"
    behavior_inbox = stage_root / "Behavior" / COHORT / "Simulated CSV"
    work_root = stage_root / ".simulation-work"
    tdt_inbox.mkdir(parents=True)
    behavior_inbox.mkdir(parents=True)
    work_root.mkdir()
    session_datetimes = _session_datetimes()
    protocols: list[SubjectProtocol] = []
    block_count = 0
    behavior_count = 0

    for subject_index, subject_id in enumerate(SUBJECT_IDS):
        protocol = _subject_protocol(subject_index)
        protocols.append(protocol)
        config = _simulation_config(subject_index, protocol)
        source_path = work_root / f"{subject_id}.doric"
        simulation_summary = generate_synthetic_doric(source_path, config)
        _validate_source_simulation(protocol, config, simulation_summary)
        dataset = load_doric(source_path)
        epocs_by_series = build_tdt_epocs_from_behavior_events(
            simulation_summary.ttl_behavior_events,
            name="Loom",
            fs=config.fs,
            pulse_width_seconds=config.ttl_pulse_width_seconds,
        )

        for series_index, session_datetime in enumerate(session_datetimes):
            series_number = series_index + 1
            target_name = _target_name(subject_id, session_datetime)
            logical_tank = (
                final_output_root
                / "Photometry"
                / COHORT
                / "TDT Binaries"
                / f"synthetic-mpfc-loom-{session_datetime:%Y%m%d-%H%M%S}"
            )
            write_tdt_block(
                tdt_inbox / target_name,
                streams=(
                    TDTExtractStream(
                        store_name="TTL",
                        code_name="TTL1",
                        data=dataset.digital_io["DIO01"][:, series_index],
                        fs=PHOTOMETRY_SAMPLING_RATE_HZ,
                    ),
                    TDTExtractStream(
                        store_name="405_mpfc",
                        code_name="M405",
                        data=dataset.isosbestic_405[:, 0, series_index],
                        fs=PHOTOMETRY_SAMPLING_RATE_HZ,
                    ),
                    TDTExtractStream(
                        store_name="470_mpfc",
                        code_name="M470",
                        data=dataset.calcium_465[:, 0, series_index],
                        fs=PHOTOMETRY_SAMPLING_RATE_HZ,
                    ),
                ),
                epoc=epocs_by_series[series_number],
                metadata=TDTExtractMetadata(
                    subject_id=subject_id,
                    subject_ids=(subject_id,),
                    order="First",
                    datetime=session_datetime,
                    tank_dir=logical_tank,
                    block_name=logical_tank.name,
                ),
            )
            block_count += 1

            behavior_summary = generate_synthetic_behavior_csv(
                behavior_inbox / _behavior_filename(subject_id, session_datetime),
                _behavior_config(protocol.sessions[series_index]),
            )
            if behavior_summary.row_count != _behavior_seconds_to_samples(
                SESSION_DURATION_SECONDS
            ):
                raise RuntimeError("behavior CSV has an unexpected row count")
            if len(behavior_summary.event_occurrences) != LOOMS_PER_SESSION:
                raise RuntimeError("behavior CSV loom ground truth is incomplete")
            behavior_count += 1

        del dataset
        source_path.unlink()
        gc.collect()

    shutil.rmtree(work_root)
    frozen_protocols = tuple(protocols)
    _write_ground_truth(stage_root, frozen_protocols)
    return block_count, behavior_count, frozen_protocols


def _write_ground_truth(
    stage_root: Path,
    protocols: tuple[SubjectProtocol, ...],
) -> Path:
    ground_truth_dir = stage_root / "Ground Truth"
    ground_truth_dir.mkdir()
    path = ground_truth_dir / "loom_response_ground_truth.csv"
    session_datetimes = _session_datetimes()
    with path.open("w", newline="", encoding="utf-8") as file_out:
        writer = csv.DictWriter(file_out, fieldnames=GROUND_TRUTH_COLUMNS)
        writer.writeheader()
        for protocol in protocols:
            for session_index, session in enumerate(protocol.sessions):
                session_datetime = session_datetimes[session_index]
                target_name = _target_name(protocol.subject_id, session_datetime)
                behavior_relative_path = (
                    Path("Behavior")
                    / COHORT
                    / "Simulated CSV"
                    / _behavior_filename(protocol.subject_id, session_datetime)
                )
                for occurrence in session:
                    writer.writerow(
                        {
                            "subject_id": protocol.subject_id,
                            "session_number": occurrence.session_number,
                            "session_datetime": session_datetime.isoformat(),
                            "loom_number_in_session": (
                                occurrence.loom_number_in_session
                            ),
                            "exposure_number": occurrence.exposure_number,
                            "loom_onset_seconds": occurrence.onset_seconds,
                            "loom_offset_seconds": occurrence.offset_seconds,
                            "response_probability": (occurrence.response_probability),
                            "calcium_response_occurred": int(
                                occurrence.calcium_response_occurred
                            ),
                            "calcium_event_seconds": (
                                ""
                                if occurrence.calcium_event_seconds is None
                                else occurrence.calcium_event_seconds
                            ),
                            "tdt_block_directory": str(
                                Path("Photometry")
                                / COHORT
                                / "Simulated TDT"
                                / target_name
                            ),
                            "behavior_csv": str(behavior_relative_path),
                        }
                    )
    return path


def _load_block(path: Path):
    with path.open("rb") as file_in:
        return pickle.load(file_in)


def _validate_generated_dataset(stage_root: Path) -> None:
    tdt_inbox = stage_root / "Photometry" / COHORT / "Simulated TDT"
    behavior_inbox = stage_root / "Behavior" / COHORT / "Simulated CSV"
    session_datetimes = _session_datetimes()
    expected_target_names = {
        _target_name(subject_id, session_datetime)
        for subject_id in SUBJECT_IDS
        for session_datetime in session_datetimes
    }
    actual_target_names = {path.name for path in tdt_inbox.iterdir() if path.is_dir()}
    if actual_target_names != expected_target_names:
        raise RuntimeError("generated TDT block folders do not match the protocol")
    expected_behavior_names = {
        _behavior_filename(subject_id, session_datetime)
        for subject_id in SUBJECT_IDS
        for session_datetime in session_datetimes
    }
    actual_behavior_names = {
        path.name for path in behavior_inbox.iterdir() if path.is_file()
    }
    if actual_behavior_names != expected_behavior_names:
        raise RuntimeError("generated behavior CSV files do not match the protocol")

    photometry_samples = _photometry_seconds_to_samples(SESSION_DURATION_SECONDS)
    behavior_samples = _behavior_seconds_to_samples(SESSION_DURATION_SECONDS)
    for subject_index, subject_id in enumerate(SUBJECT_IDS):
        protocol = _subject_protocol(subject_index)
        for session_index, session_datetime in enumerate(session_datetimes):
            session = protocol.sessions[session_index]
            target_name = _target_name(subject_id, session_datetime)
            directory = tdt_inbox / target_name
            if {path.name for path in directory.iterdir()} != {"tdt_block.pkl"}:
                raise RuntimeError(f"unexpected files in {directory}")
            block = _load_block(directory / "tdt_block.pkl")
            if tuple(block.streams.keys()) != ("TTL", "405_mpfc", "470_mpfc"):
                raise RuntimeError(f"unexpected streams in {directory}")
            streams = tuple(block.streams.values())
            if any(len(stream.data) != photometry_samples for stream in streams):
                raise RuntimeError(f"unexpected stream length in {directory}")
            if any(not np.all(np.isfinite(stream.data)) for stream in streams):
                raise RuntimeError(f"non-finite stream data in {directory}")
            if any(stream.fs != PHOTOMETRY_SAMPLING_RATE_HZ for stream in streams):
                raise RuntimeError(f"unexpected sampling rate in {directory}")
            if block.info.start_date != session_datetime:
                raise RuntimeError(f"block datetime mismatch in {directory}")
            if tuple(block.epocs.keys()) != ("Loom",):
                raise RuntimeError(f"block EPOC mismatch in {directory}")

            expected_onsets = np.asarray(
                [
                    _photometry_seconds_to_samples(item.onset_seconds)
                    / PHOTOMETRY_SAMPLING_RATE_HZ
                    for item in session
                ]
            )
            np.testing.assert_allclose(block.epocs.Loom.onset, expected_onsets)
            ttl_values = np.asarray(block.streams.TTL.data) > 0.5
            ttl_rising_edges = np.flatnonzero(
                ttl_values & ~np.r_[False, ttl_values[:-1]]
            )
            np.testing.assert_array_equal(
                ttl_rising_edges,
                [
                    _photometry_seconds_to_samples(item.onset_seconds)
                    for item in session
                ],
            )

            behavior_path = behavior_inbox / _behavior_filename(
                subject_id,
                session_datetime,
            )
            with behavior_path.open(encoding="utf-8", newline="") as file_in:
                header = next(csv.reader(file_in))
            if header != ["Time (s)", "loom"]:
                raise RuntimeError(f"unexpected behavior columns in {behavior_path}")
            behavior_values = np.loadtxt(
                behavior_path,
                delimiter=",",
                skiprows=1,
            )
            if behavior_values.shape != (behavior_samples, 2):
                raise RuntimeError(f"unexpected behavior shape in {behavior_path}")
            np.testing.assert_allclose(
                behavior_values[:, 0],
                np.arange(behavior_samples) / BEHAVIOR_SAMPLING_RATE_HZ,
            )
            expected_loom = np.zeros(behavior_samples, dtype=int)
            for item in session:
                expected_loom[
                    _behavior_seconds_to_samples(
                        item.onset_seconds
                    ) : _behavior_seconds_to_samples(item.offset_seconds)
                ] = 1
            np.testing.assert_array_equal(behavior_values[:, 1], expected_loom)

    ground_truth_path = stage_root / "Ground Truth" / "loom_response_ground_truth.csv"
    with ground_truth_path.open(encoding="utf-8", newline="") as file_in:
        rows = list(csv.DictReader(file_in))
    if len(rows) != len(SUBJECT_IDS) * SESSION_COUNT * LOOMS_PER_SESSION:
        raise RuntimeError("loom response ground truth has an unexpected row count")
    if tuple(rows[0]) != GROUND_TRUTH_COLUMNS:
        raise RuntimeError("loom response ground truth has unexpected columns")


def generate_dataset(output_root: Path, *, overwrite: bool = False) -> Path:
    """Generate, validate, and atomically install the complete dataset."""

    output_root = output_root.expanduser().resolve()
    _preflight_output(output_root, overwrite)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.writing-",
            dir=output_root.parent,
        )
    )
    try:
        block_count, behavior_count, _ = _export_dataset(
            stage_root,
            final_output_root=output_root,
        )
        _validate_generated_dataset(stage_root)
        expected = len(SUBJECT_IDS) * SESSION_COUNT
        if block_count != expected or behavior_count != expected:
            raise RuntimeError(
                f"expected {expected} blocks and behavior CSVs, generated "
                f"{block_count} and {behavior_count}"
            )
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(stage_root, output_root)
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
    return output_root


def main() -> None:
    args = _parse_args()
    output_root = generate_dataset(args.output, overwrite=args.overwrite)
    session_total = len(SUBJECT_IDS) * SESSION_COUNT
    print(
        f"Generated {session_total} simulated TDT blocks and "
        f"{session_total} behavior CSVs at {output_root}"
    )
    print(
        "Layout: 8 subjects x 1 20-minute session; "
        "one medial prefrontal cortex site per subject"
    )
    print(
        f"Protocol: {BASELINE_DURATION_SECONDS / 60:g}-minute baseline followed "
        f"by {LOOMS_PER_SESSION} random {LOOM_DURATION_SECONDS:g}-second loom "
        "bouts during 15 minutes of conditioning"
    )
    final_probability = RESPONSE_INITIAL_PROBABILITY * (
        RESPONSE_PROBABILITY_DECAY ** (SESSION_COUNT * LOOMS_PER_SESSION - 1)
    )
    print(
        "Pre-loom calcium response probability: "
        f"{RESPONSE_INITIAL_PROBABILITY:.3f} initially, multiplied by "
        f"{RESPONSE_PROBABILITY_DECAY:.3f} per exposure, "
        f"{final_probability:.3f} at the final exposure"
    )
    response_counts = [
        _subject_protocol(subject_index).response_count
        for subject_index in range(len(SUBJECT_IDS))
    ]
    print(f"Realized calcium responses by subject: {response_counts}")
    print(
        "Ground truth: Ground Truth/loom_response_ground_truth.csv; "
        "calcium events begin one second before their corresponding loom"
    )


if __name__ == "__main__":
    main()
