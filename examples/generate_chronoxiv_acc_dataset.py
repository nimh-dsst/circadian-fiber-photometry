"""Generate a two-site-per-block synthetic dataset for ChronoXIV.

The output contains ten subjects, two anterior cingulate cortex sites per
subject, and four 20-minute sessions.  Session 1 is the baseline condition;
sessions 2-4 progressively increase the post-injection calcium baseline,
transient count, and transient amplitude.  The calcium baseline response begins
at the five-second TTL injection marker and approaches the configured session
level exponentially.  Both photometry wavelengths undergo mild single-
exponential photobleaching with overnight protein-turnover recovery.  Every
trace also includes a two-second startup artifact and one independently placed
random box artifact.

The generated folders follow ``CHRONOXIV_FORMAT_SPEC.md``::

    Photometry/Cohort 1/Simulated TDT/
      Subject01__acc-left+acc-right__M-20260105-090000/
        tdt_block.pkl

Each pickle contains the top-level ``tdt.StructType`` returned by
``tdt.read_block``. Its streams are ``TTL``, ``405_left``, ``405_right``,
``470_left``, and ``470_right``.

Run from the repository root with::

    uv run python examples/generate_chronoxiv_acc_dataset.py

Use ``--overwrite`` to replace a previously generated dataset.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from circadian_fiber_photometry import load_doric
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
    add_random_box_artifacts,
    add_scheduled_calcium_events,
    build_tdt_epocs_from_behavior_events,
    configure_session_start_spike,
    generate_synthetic_doric,
)

SUBJECT_IDS = tuple(f"Subject{number:02d}" for number in range(1, 11))
SITES = ("acc_left", "acc_right")
FOLDER_SITE_IDS = ("acc-left", "acc-right")
SUBJECT_SEXES = {
    subject_id: "M" if index <= 5 else "F"
    for index, subject_id in enumerate(SUBJECT_IDS, start=1)
}
TDT_BLOCK_FOLDER_RE = re.compile(
    r"^(?P<subjects>[A-Za-z0-9._-]+(?:\+[A-Za-z0-9._-]+)?)"
    r"__(?P<sites>[A-Za-z0-9-]+(?:\+[A-Za-z0-9-]+)?)"
    r"__(?P<sexes>[MFUO](?:\+[MFUO])?)-"
    r"(?P<datetime>\d{8}-\d{6})$"
)
COHORT = "Cohort 1"
BASE_DATETIME = datetime(2026, 1, 5, 9, 0, 0)

SESSION_DURATION_SECONDS = 20 * 60
SESSION_SPACING_SECONDS = 24 * 60 * 60
INTER_SESSION_GAP_SECONDS = SESSION_SPACING_SECONDS - SESSION_DURATION_SECONDS
SAMPLING_RATE_HZ = 1017.2526245117188
ISOSBESTIC_BASELINE = 0.090
EXPERIMENTAL_BASELINE = 0.180
STARTUP_ARTIFACT_SECONDS = 2.0
TTL_START_SECONDS = 5.0
TTL_PULSE_WIDTH_SECONDS = 0.125
BASELINE_RISE_TIME_SECONDS = 90.0
BOX_ARTIFACT_DURATION_SECONDS = (10.0, 20.0)
SEED = 20260105


@dataclass(frozen=True)
class SessionProfile:
    """Post-injection calcium progression applied to one simulation series."""

    name: str
    baseline_offset: float
    transient_count: int
    transient_amplitude: float


SESSION_PROFILES = (
    SessionProfile("baseline", 0.000, 0, 0.000),
    SessionProfile("slight", 0.010, 6, 0.014),
    SessionProfile("moderate", 0.022, 12, 0.022),
    SessionProfile("highest", 0.038, 24, 0.034),
)


def _parse_args() -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parent / "generated" / "chronoxiv_acc_progression"
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


def _seconds_to_samples(seconds: float) -> int:
    """Use the simulator's nearest-sample conversion convention."""

    return int(round(seconds * SAMPLING_RATE_HZ))


def _event_times(
    rng: np.random.Generator,
    *,
    count: int,
) -> tuple[float, ...]:
    """Draw distinct, sample-aligned transient times outside startup."""

    first_sample = _seconds_to_samples(15)
    final_sample = _seconds_to_samples(SESSION_DURATION_SECONDS - 15)
    samples = rng.choice(
        np.arange(first_sample, final_sample, dtype=int),
        size=count,
        replace=False,
    )
    return tuple(float(value) / SAMPLING_RATE_HZ for value in np.sort(samples))


def _baseline_change_trace(profile: SessionProfile) -> np.ndarray:
    """Return a calcium-only response beginning at the injection TTL."""

    relative_time = (
        np.arange(_seconds_to_samples(SESSION_DURATION_SECONDS), dtype=float)
        / SAMPLING_RATE_HZ
    )
    elapsed = np.maximum(relative_time - TTL_START_SECONDS, 0.0)
    response = profile.baseline_offset * (
        1.0 - np.exp(-elapsed / BASELINE_RISE_TIME_SECONDS)
    )
    response[relative_time < TTL_START_SECONDS] = 0.0
    return response


def _build_signal_config(subject_index: int) -> SyntheticSignalConfig:
    """Build deterministic per-session and per-channel signal components."""

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
                    amplitude_fraction=0.18,
                    time_constant_seconds=7200.0,
                ),
            ),
            calcium_components=(
                SyntheticPhotobleachingComponentConfig(
                    amplitude_fraction=0.25,
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

    event_rng = np.random.default_rng(SEED + subject_index + 1)
    channel_count = len(SITES)
    for series_number, profile in enumerate(SESSION_PROFILES, start=1):
        if profile.transient_count == 0:
            continue
        for channel_number in range(1, channel_count + 1):
            signal = add_scheduled_calcium_events(
                signal,
                _event_times(event_rng, count=profile.transient_count),
                amplitude=profile.transient_amplitude,
                channels=(channel_number,),
                series_numbers=(series_number,),
                name=f"{profile.name} session transients",
            )

    for channel_number in range(1, channel_count + 1):
        direction = -1.0 if channel_number % 2 else 1.0
        signal = add_random_box_artifacts(
            signal,
            count_per_series=1,
            start_window_seconds=(30.0, SESSION_DURATION_SECONDS - 30.0),
            duration_range_seconds=BOX_ARTIFACT_DURATION_SECONDS,
            magnitude_fraction=direction * 0.08,
            channels=(channel_number,),
            name="sporadic shared-signal artifact",
        )
    return signal


def _simulation_config(subject_index: int) -> SyntheticDoricConfig:
    """Return the complete four-session source simulation configuration."""

    return SyntheticDoricConfig(
        series_count=len(SESSION_PROFILES),
        session_duration_seconds=SESSION_DURATION_SECONDS,
        inter_series_gap_seconds=INTER_SESSION_GAP_SECONDS,
        fs=SAMPLING_RATE_HZ,
        channel_count=len(SITES),
        seed=SEED + subject_index,
        signal=_build_signal_config(subject_index),
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig(
                "injection",
                channel=1,
                pulse_count=1,
            ),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig(
                "injection",
                start_seconds=(TTL_START_SECONDS,),
            ),
        ),
        ttl_pulse_width_seconds=TTL_PULSE_WIDTH_SECONDS,
        ttl_pulse_off_interval_seconds=TTL_PULSE_WIDTH_SECONDS,
    )


def _target_name(subject_id: str, session_datetime: datetime) -> str:
    try:
        sex = SUBJECT_SEXES[subject_id]
    except KeyError as error:
        raise ValueError(f"no sex designation configured for {subject_id!r}") from error
    name = (
        f"{subject_id}__{'+'.join(FOLDER_SITE_IDS)}__{sex}-"
        f"{session_datetime:%Y%m%d-%H%M%S}"
    )
    if TDT_BLOCK_FOLDER_RE.fullmatch(name) is None:
        raise RuntimeError(f"generated TDT block folder name is invalid: {name}")
    return name


def _session_datetimes() -> tuple[datetime, ...]:
    return tuple(
        BASE_DATETIME + timedelta(seconds=index * SESSION_SPACING_SECONDS)
        for index in range(len(SESSION_PROFILES))
    )


def _preflight_output(output_root: Path, overwrite: bool) -> None:
    """Reject ambiguous or unexpectedly broad replacement targets."""

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


def _export_multi_site_blocks(
    stage_root: Path,
    *,
    final_output_root: Path,
) -> int:
    """Generate and export one two-site source at a time to bound memory use."""

    inbox = stage_root / "Photometry" / COHORT / "Simulated TDT"
    inbox.mkdir(parents=True)
    work_root = stage_root / ".extract-work"
    work_root.mkdir()
    session_datetimes = _session_datetimes()
    baseline_changes = tuple(
        _baseline_change_trace(profile) for profile in SESSION_PROFILES
    )
    exported = 0

    for subject_index, subject_id in enumerate(SUBJECT_IDS):
        config = _simulation_config(subject_index)
        source_path = work_root / f"{subject_id}.doric"
        simulation_summary = generate_synthetic_doric(source_path, config)
        dataset = load_doric(source_path)
        if simulation_summary.photobleaching.model != "single_exponential":
            raise RuntimeError("single-exponential photobleaching was not applied")
        for series_number, profile in enumerate(SESSION_PROFILES, start=1):
            for channel_number in range(1, config.channel_count + 1):
                realized_count = simulation_summary.event_sample_indices[
                    (series_number, channel_number)
                ].size
                if realized_count != profile.transient_count:
                    raise RuntimeError(
                        "simulated transient count does not match its session profile"
                    )
        expected_artifacts = config.series_count * config.channel_count * 2
        if len(simulation_summary.artifact_occurrences) != expected_artifacts:
            raise RuntimeError("startup or random artifact ground truth is incomplete")
        random_boxes = (
            occurrence
            for occurrence in simulation_summary.artifact_occurrences
            if occurrence.artifact_type == "random_box"
        )
        if any(
            occurrence.realized_duration_seconds < BOX_ARTIFACT_DURATION_SECONDS[0]
            for occurrence in random_boxes
        ):
            raise RuntimeError("a random box artifact is shorter than 10 seconds")
        epocs_by_series = build_tdt_epocs_from_behavior_events(
            simulation_summary.ttl_behavior_events,
            name="Inj1",
            fs=config.fs,
            pulse_width_seconds=config.ttl_pulse_width_seconds,
        )

        for series_index, session_datetime in enumerate(session_datetimes):
            series_number = series_index + 1
            logical_tank = (
                final_output_root
                / "Photometry"
                / COHORT
                / "TDT Binaries"
                / f"synthetic-acc-{session_datetime:%Y%m%d-%H%M%S}"
            )
            calcium_left = np.array(
                dataset.calcium_465[:, 0, series_index],
                copy=True,
            )
            calcium_right = np.array(
                dataset.calcium_465[:, 1, series_index],
                copy=True,
            )
            calcium_left += baseline_changes[series_index]
            calcium_right += baseline_changes[series_index]
            target = inbox / _target_name(subject_id, session_datetime)
            write_tdt_block(
                target,
                streams=(
                    TDTExtractStream(
                        store_name="TTL",
                        code_name="TTL1",
                        data=dataset.digital_io["DIO01"][:, series_index],
                        fs=SAMPLING_RATE_HZ,
                    ),
                    TDTExtractStream(
                        store_name="405_left",
                        code_name="4L05",
                        data=dataset.isosbestic_405[:, 0, series_index],
                        fs=SAMPLING_RATE_HZ,
                    ),
                    TDTExtractStream(
                        store_name="405_right",
                        code_name="4R05",
                        data=dataset.isosbestic_405[:, 1, series_index],
                        fs=SAMPLING_RATE_HZ,
                    ),
                    TDTExtractStream(
                        store_name="470_left",
                        code_name="4L70",
                        data=calcium_left,
                        fs=SAMPLING_RATE_HZ,
                    ),
                    TDTExtractStream(
                        store_name="470_right",
                        code_name="4R70",
                        data=calcium_right,
                        fs=SAMPLING_RATE_HZ,
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
            exported += 1

        del dataset
        source_path.unlink()
        gc.collect()

    shutil.rmtree(work_root)
    return exported


def _load_stream(path: Path):
    """Load a pickle created during this trusted simulator run."""

    with path.open("rb") as file_in:
        return pickle.load(file_in)


def _validate_generated_dataset(stage_root: Path) -> None:
    """Validate the generated tree against the local ChronoXIV contract."""

    inbox = stage_root / "Photometry" / COHORT / "Simulated TDT"
    session_datetimes = _session_datetimes()
    expected_names = {
        _target_name(subject_id, session_datetime)
        for subject_id in SUBJECT_IDS
        for session_datetime in session_datetimes
    }
    directories = {path.name: path for path in inbox.iterdir() if path.is_dir()}
    if set(directories) != expected_names:
        raise RuntimeError("generated subject-session block folders do not match")

    required = {"tdt_block.pkl"}
    samples_per_session = _seconds_to_samples(SESSION_DURATION_SECONDS)
    progression_means: dict[tuple[str, str], list[float]] = {
        (subject_id, site): [] for subject_id in SUBJECT_IDS for site in SITES
    }

    for session_index, session_datetime in enumerate(session_datetimes):
        for subject_id in SUBJECT_IDS:
            directory = directories[_target_name(subject_id, session_datetime)]
            actual_files = {path.name for path in directory.iterdir()}
            if actual_files != required:
                raise RuntimeError(
                    f"expected only tdt_block.pkl in {directory}, got "
                    f"{sorted(actual_files)}"
                )
            block = _load_stream(directory / "tdt_block.pkl")
            if tuple(block.keys()) != (
                "epocs",
                "snips",
                "streams",
                "scalars",
                "info",
            ):
                raise RuntimeError(f"invalid read_block structure in {directory}")
            expected_stream_names = (
                "TTL",
                "405_left",
                "405_right",
                "470_left",
                "470_right",
            )
            if tuple(block.streams.keys()) != expected_stream_names:
                raise RuntimeError(f"unexpected streams in {directory}")
            ttl = block.streams.TTL
            streams = tuple(block.streams[name] for name in expected_stream_names)
            if any(np.asarray(stream.data).ndim != 1 for stream in streams):
                raise RuntimeError(f"non-vector stream in {directory}")
            if any(len(stream.data) != samples_per_session for stream in streams):
                raise RuntimeError(f"unexpected stream length in {directory}")
            if any(not np.all(np.isfinite(stream.data)) for stream in streams):
                raise RuntimeError(f"non-finite stream data in {directory}")
            if any(stream.fs != SAMPLING_RATE_HZ for stream in streams):
                raise RuntimeError(f"unexpected sampling rate in {directory}")

            ttl_high = np.flatnonzero(np.asarray(ttl.data) > 0.5)
            expected_ttl_sample = _seconds_to_samples(TTL_START_SECONDS)
            startup_stop = _seconds_to_samples(STARTUP_ARTIFACT_SECONDS)
            if ttl_high.size == 0 or ttl_high[0] != expected_ttl_sample:
                raise RuntimeError(f"TTL onset is incorrect in {directory}")
            if ttl_high[0] < startup_stop:
                raise RuntimeError(f"TTL overlaps startup artifact in {directory}")

            if block.info.start_date != session_datetime:
                raise RuntimeError(f"block datetime mismatch in {directory}")
            if tuple(block.epocs.keys()) != ("Inj1",):
                raise RuntimeError(f"block epoc mismatch in {directory}")
            injection_epoc = block.epocs.Inj1
            realized_ttl_seconds = expected_ttl_sample / SAMPLING_RATE_HZ
            np.testing.assert_allclose(
                injection_epoc.onset,
                [realized_ttl_seconds],
            )
            for site in SITES:
                side = site.removeprefix("acc_")
                iso = block.streams[f"405_{side}"]
                exp = block.streams[f"470_{side}"]
                progression_means[(subject_id, site)].append(float(np.mean(exp.data)))

                early_start = _seconds_to_samples(10)
                early_stop = _seconds_to_samples(130)
                late_start = _seconds_to_samples(SESSION_DURATION_SECONDS - 120)
                early_iso = float(np.mean(iso.data[early_start:early_stop]))
                late_iso = float(np.mean(iso.data[late_start:]))
                if late_iso >= early_iso:
                    raise RuntimeError(
                        f"isosbestic photobleaching is not visible in {directory}"
                    )

                profile = SESSION_PROFILES[session_index]
                response = _baseline_change_trace(profile)
                injection_sample = _seconds_to_samples(TTL_START_SECONDS)
                if np.any(response[:injection_sample] != 0.0):
                    raise RuntimeError("baseline response begins before injection")
                if (
                    profile.baseline_offset > 0
                    and response[-1] <= 0.95 * profile.baseline_offset
                ):
                    raise RuntimeError("baseline response did not reach plateau")

    for key, means in progression_means.items():
        if not np.all(np.diff(means) > 0):
            raise RuntimeError(f"experimental means do not increase for {key}")

    baseline_directory = directories[_target_name(SUBJECT_IDS[0], session_datetimes[0])]
    baseline_block = _load_stream(baseline_directory / "tdt_block.pkl")
    baseline_iso = baseline_block.streams["405_left"]
    baseline_exp = baseline_block.streams["470_left"]
    stable_start = _seconds_to_samples(STARTUP_ARTIFACT_SECONDS)
    injection_start = _seconds_to_samples(TTL_START_SECONDS)
    initial_ratio = float(
        np.mean(baseline_iso.data[stable_start:injection_start])
        / np.mean(baseline_exp.data[stable_start:injection_start])
    )
    if not np.isclose(initial_ratio, 0.5, atol=0.03):
        raise RuntimeError(
            f"initial isosbestic/experimental baseline ratio is {initial_ratio:.3f}"
        )


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
        exported = _export_multi_site_blocks(
            stage_root,
            final_output_root=output_root,
        )
        _validate_generated_dataset(stage_root)
        expected = len(SUBJECT_IDS) * len(SESSION_PROFILES)
        if exported != expected:
            raise RuntimeError(f"expected {expected} blocks, generated {exported}")
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
    block_count = len(SUBJECT_IDS) * len(SESSION_PROFILES)
    print(f"Generated {block_count} simulated TDT blocks at {output_root}")
    print(
        "Layout: 10 subjects x 4 daily 20-minute sessions; "
        "both ACC hemisphere sites in each block"
    )
    print(
        f"Sampling rate: {SAMPLING_RATE_HZ!r} Hz; samples per stream="
        f"{_seconds_to_samples(SESSION_DURATION_SECONDS):,}"
    )
    print(
        f"TTL: first rising edge at {TTL_START_SECONDS:g} s; startup artifact "
        f"ends at {STARTUP_ARTIFACT_SECONDS:g} s"
    )
    print(
        "Photobleaching: single exponential; 405 amplitude/tau="
        "0.18/7200 s; 465 amplitude/tau=0.25/6000 s; turnover half-life=48 h"
    )
    print(
        f"Nominal baselines: 405={ISOSBESTIC_BASELINE:.3f} V; "
        f"465={EXPERIMENTAL_BASELINE:.3f} V (1:2 ratio)"
    )
    print(
        "Random box artifacts: one per trace, duration="
        f"{BOX_ARTIFACT_DURATION_SECONDS[0]:g}-"
        f"{BOX_ARTIFACT_DURATION_SECONDS[1]:g} s"
    )
    for number, profile in enumerate(SESSION_PROFILES, start=1):
        print(
            f"Session {number} ({profile.name}): post-injection baseline change="
            f"{profile.baseline_offset:.3f} V, transients="
            f"{profile.transient_count}, amplitude="
            f"{profile.transient_amplitude:.3f} V"
        )


if __name__ == "__main__":
    main()
