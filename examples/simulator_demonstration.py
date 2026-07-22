import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulator demonstration

    This notebook begins with a simple long-timescale simulation: a 12-hour
    sinusoidal tonic calcium signal observed across a 48-hour experiment. The
    recording schedule contains a 10-minute phasic recording every 30 minutes,
    leaving a 20-minute unrecorded gap between sessions.

    Here, *phasic recording* describes the short acquisition window rather than
    the presence of simulated phasic calcium events. In the first two
    demonstrations, noise, photobleaching, artifacts, and calcium events are
    disabled so the tonic component and the recording schedule remain easy to
    see. A third demonstration then adds event transients explicitly.

    The simulator evaluates the tonic sinusoid against absolute wall-clock
    time. Its phase therefore continues through every unrecorded gap even though
    those gap samples are not written to the Doric file.

    At 60 Hz, this example writes an approximately 317 MB file under
    `examples/generated/`. That directory is ignored by Git, and rerunning the
    notebook replaces the previously generated demonstration file.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go

    from circadian_fiber_photometry import load_doric
    from circadian_fiber_photometry.simulation import (
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        add_gaussian_noise,
        add_random_calcium_events,
        add_tonic_component,
        generate_synthetic_doric,
    )

    notebook_dir = Path(__file__).resolve().parent
    return (
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        add_gaussian_noise,
        add_random_calcium_events,
        add_tonic_component,
        generate_synthetic_doric,
        go,
        load_doric,
        notebook_dir,
        np,
    )


@app.cell
def _(np):
    total_schedule_hours = 48.0
    total_schedule_seconds = total_schedule_hours * 60 * 60
    session_duration_seconds = 10 * 60
    inter_series_gap_seconds = 20 * 60
    cycle_seconds = session_duration_seconds + inter_series_gap_seconds
    series_count = int(total_schedule_seconds / cycle_seconds)

    fs = 60.0
    channel_count = 1
    samples_per_session = int(round(fs * session_duration_seconds))

    tonic_amplitude = 1.0
    tonic_period_hours = 12.0
    tonic_frequency_hz = 1 / (tonic_period_hours * 60 * 60)

    active_recording_hours = (
        series_count * session_duration_seconds / 3600
    )
    gap_hours = series_count * inter_series_gap_seconds / 3600
    schedule_hours_including_final_gap = series_count * cycle_seconds / 3600
    final_recording_end_hours = (
        (series_count - 1) * cycle_seconds + session_duration_seconds
    ) / 3600

    assert series_count == 96
    assert samples_per_session == 36_000
    assert np.isclose(active_recording_hours, 16.0)
    assert np.isclose(gap_hours, 32.0)
    assert np.isclose(schedule_hours_including_final_gap, total_schedule_hours)
    assert np.isclose(final_recording_end_hours, 47 + 40 / 60)
    return (
        active_recording_hours,
        channel_count,
        cycle_seconds,
        final_recording_end_hours,
        fs,
        gap_hours,
        inter_series_gap_seconds,
        samples_per_session,
        schedule_hours_including_final_gap,
        series_count,
        session_duration_seconds,
        tonic_amplitude,
        tonic_frequency_hz,
        tonic_period_hours,
        total_schedule_hours,
        total_schedule_seconds,
    )


@app.cell
def _(
    SyntheticDoricConfig,
    SyntheticPhotobleachingConfig,
    SyntheticSignalConfig,
    add_tonic_component,
    channel_count,
    fs,
    generate_synthetic_doric,
    inter_series_gap_seconds,
    load_doric,
    notebook_dir,
    series_count,
    session_duration_seconds,
    tonic_amplitude,
    tonic_frequency_hz,
):
    base_signal = SyntheticSignalConfig(
        channel_baseline_step=0.0,
        photobleaching=SyntheticPhotobleachingConfig(model="none"),
        artifact_amplitude=0.0,
        circadian_amplitude=0.0,
        noise_std=0.0,
        analog_noise_std=0.0,
        transient_rate_per_minute=0.0,
        transient_amplitude=0.0,
    )
    signal_config = add_tonic_component(
        base_signal,
        amplitude=tonic_amplitude,
        frequency_hz=tonic_frequency_hz,
        phase_radians=0.0,
        name="12-hour tonic component",
    )
    simulation_config = SyntheticDoricConfig(
        series_count=series_count,
        session_duration_seconds=session_duration_seconds,
        inter_series_gap_seconds=inter_series_gap_seconds,
        fs=fs,
        channel_count=channel_count,
        seed=123,
        signal=signal_config,
    )

    output_path = notebook_dir / "generated" / "simulator_demonstration.doric"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_summary = generate_synthetic_doric(
        output_path,
        simulation_config,
        overwrite=True,
    )
    dataset = load_doric(output_path)
    return dataset, output_path, signal_config, simulation_summary


@app.cell
def _(
    active_recording_hours,
    cycle_seconds,
    dataset,
    final_recording_end_hours,
    fs,
    gap_hours,
    inter_series_gap_seconds,
    mo,
    np,
    output_path,
    samples_per_session,
    schedule_hours_including_final_gap,
    series_count,
    session_duration_seconds,
    signal_config,
    simulation_summary,
    tonic_amplitude,
    tonic_frequency_hz,
    tonic_period_hours,
):
    validation_sample_indices = np.array([0, samples_per_session // 2, -1])
    validation_timestamps = dataset.timestamps[validation_sample_indices, :]
    expected_calcium = signal_config.calcium_baseline + tonic_amplitude * np.sin(
        2 * np.pi * tonic_frequency_hz * validation_timestamps
    )

    assert simulation_summary.series_count == series_count
    assert simulation_summary.samples_per_series == samples_per_session
    assert dataset.calcium_465.shape == (samples_per_session, 1, series_count)
    assert dataset.isosbestic_405.shape == (samples_per_session, 1, series_count)
    np.testing.assert_allclose(dataset.fs, fs)
    np.testing.assert_allclose(simulation_summary.session_start_times[0], 0.0)
    np.testing.assert_allclose(
        np.diff(simulation_summary.session_start_times),
        cycle_seconds,
    )
    np.testing.assert_allclose(
        dataset.session_start_times,
        simulation_summary.session_start_times,
    )
    np.testing.assert_allclose(
        dataset.calcium_465[validation_sample_indices, 0, :],
        expected_calcium,
    )
    np.testing.assert_allclose(
        dataset.isosbestic_405[validation_sample_indices, 0, :],
        signal_config.isosbestic_baseline,
    )

    generated_file_megabytes = output_path.stat().st_size / (1024**2)
    session_duration_minutes = session_duration_seconds / 60
    gap_duration_minutes = inter_series_gap_seconds / 60
    duty_cycle_percent = 100 * session_duration_seconds / cycle_seconds

    mo.vstack(
        [
            mo.md("## Generated 48-hour schedule"),
            mo.hstack(
                [
                    mo.stat(str(series_count), label="Recording sessions"),
                    mo.stat(f"{dataset.fs:.0f} Hz", label="Sampling rate"),
                    mo.stat(
                        f"{samples_per_session:,}",
                        label="Samples per session",
                    ),
                    mo.stat(
                        f"{active_recording_hours:.0f} h",
                        label="Recorded time",
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                f"""
                - **Generated file:** `{output_path}`
                  ({generated_file_megabytes:.1f} MiB)
                - **Recording / gap:** {session_duration_minutes:.0f} /
                  {gap_duration_minutes:.0f} minutes
                - **Session-start spacing:** {cycle_seconds / 60:.0f} minutes
                - **Duty cycle:** {duty_cycle_percent:.1f}%
                - **Recorded / unrecorded time:** {active_recording_hours:.0f} /
                  {gap_hours:.0f} hours
                - **Schedule including the final gap:**
                  {schedule_hours_including_final_gap:.0f} hours
                - **Last recording ends at:** {final_recording_end_hours:.3f} hours
                  (47 hours 40 minutes)
                - **Tonic period / frequency:** {tonic_period_hours:.0f} hours /
                  {tonic_frequency_hz:.8g} Hz

                All schedule, shape, sampling-rate, and signal-equation checks
                passed.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tonic signal across 48 hours

    The gray curve is the configured continuous sinusoid, evaluated once per
    minute for a lightweight visual reference. Red markers are means from the
    actual 60 Hz Doric sessions. Session summaries keep the full experiment
    responsive without sending all 3.46 million recorded samples to the browser.
    """)
    return


@app.cell
def _(
    dataset,
    go,
    np,
    signal_config,
    tonic_amplitude,
    tonic_frequency_hz,
    total_schedule_seconds,
):
    reference_step_seconds = 60.0
    reference_time_seconds = np.arange(
        0.0,
        total_schedule_seconds + reference_step_seconds,
        reference_step_seconds,
    )
    reference_time_hours = reference_time_seconds / 3600
    reference_calcium = signal_config.calcium_baseline + tonic_amplitude * np.sin(
        2 * np.pi * tonic_frequency_hz * reference_time_seconds
    )

    session_midpoint_hours = dataset.timestamps.mean(axis=0) / 3600
    calcium_session_means = dataset.calcium_465[:, 0, :].mean(axis=0)

    overview_figure = go.Figure()
    overview_figure.add_scatter(
        x=reference_time_hours,
        y=reference_calcium,
        name="Continuous tonic reference",
        mode="lines",
        line={"color": "#5f6368", "width": 2},
        hovertemplate="%{x:.2f} h<br>%{y:.4f} V<extra>%{fullData.name}</extra>",
    )
    overview_figure.add_scatter(
        x=session_midpoint_hours,
        y=calcium_session_means,
        name="60 Hz recording means",
        mode="markers",
        marker={"color": "#d73027", "size": 6},
        hovertemplate="%{x:.2f} h<br>%{y:.4f} V<extra>%{fullData.name}</extra>",
    )
    overview_figure.update_layout(
        template="plotly_white",
        title="Tonic calcium signal across the 48-hour schedule",
        xaxis_title="Elapsed experiment time (hours)",
        yaxis_title="Calcium signal (V)",
        legend_title_text="Trace",
        hovermode="x unified",
    )
    overview_figure = overview_figure.update_xaxes(range=[0, 48])
    return overview_figure, reference_calcium, reference_time_hours


@app.cell
def _(mo, overview_figure):
    mo.ui.plotly(
        overview_figure,
        config={"scrollZoom": True, "responsive": True},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ten-minute recordings separated by 20-minute gaps

    The complete 48-hour schedule is shown below. Blue segments are the recorded
    405 nm isosbestic signal, red segments are the recorded 465 nm calcium
    signal, and shaded regions are absent from the Doric file. The gray tonic
    reference remains continuous through each gap, making the simulator's
    absolute-time behavior visible.

    The generated Doric data remain at 60 Hz. For responsive browser rendering,
    this overview displays one sample per second from each recorded interval;
    zooming and panning preserve the 10-minute recording and 20-minute gap
    structure across all 48 hours.
    """)
    return


@app.cell
def _(
    cycle_seconds,
    dataset,
    go,
    np,
    reference_calcium,
    reference_time_hours,
    session_duration_seconds,
    total_schedule_hours,
):
    display_sampling_rate_hz = 1.0
    display_stride = int(round(dataset.fs / display_sampling_rate_hz))
    assert np.isclose(dataset.fs / display_stride, display_sampling_rate_hz)

    display_session_time_hours = dataset.timestamps[::display_stride, :].T / 3600
    display_session_isosbestic = dataset.isosbestic_405[
        ::display_stride, 0, :
    ].T
    display_session_calcium = dataset.calcium_465[::display_stride, 0, :].T
    separator_column = np.full((dataset.series_count, 1), np.nan)
    recorded_time_hours = np.concatenate(
        [display_session_time_hours, separator_column],
        axis=1,
    ).reshape(-1)
    recorded_calcium = np.concatenate(
        [display_session_calcium, separator_column],
        axis=1,
    ).reshape(-1)
    recorded_isosbestic = np.concatenate(
        [display_session_isosbestic, separator_column],
        axis=1,
    ).reshape(-1)

    recording_figure = go.Figure()
    for display_series_index in range(dataset.series_count):
        recording_end_hours = (
            display_series_index * cycle_seconds + session_duration_seconds
        ) / 3600
        gap_end_hours = min(
            (display_series_index + 1) * cycle_seconds / 3600,
            total_schedule_hours,
        )
        recording_figure.add_vrect(
            x0=recording_end_hours,
            x1=gap_end_hours,
            fillcolor="rgba(100, 116, 139, 0.14)",
            layer="below",
            line_width=0,
        )

    recording_figure.add_scatter(
        x=reference_time_hours,
        y=reference_calcium,
        name="Continuous tonic reference",
        mode="lines",
        line={"color": "#5f6368", "width": 2, "dash": "dot"},
        hovertemplate="%{x:.4f} h<br>%{y:.4f} V<extra>%{fullData.name}</extra>",
    )
    recording_figure.add_scattergl(
        x=recorded_time_hours,
        y=recorded_isosbestic,
        name="405 nm isosbestic (recorded)",
        mode="lines",
        connectgaps=False,
        line={"color": "#4575b4", "width": 2},
        hovertemplate="%{x:.4f} h<br>%{y:.4f} V<extra>%{fullData.name}</extra>",
    )
    recording_figure.add_scattergl(
        x=recorded_time_hours,
        y=recorded_calcium,
        name="465 nm calcium (recorded)",
        mode="lines",
        connectgaps=False,
        line={"color": "#d73027", "width": 2},
        hovertemplate="%{x:.4f} h<br>%{y:.4f} V<extra>%{fullData.name}</extra>",
    )
    recording_figure.update_layout(
        template="plotly_white",
        title="Recorded 405 and 465 nm signals across the 48-hour schedule",
        xaxis_title="Elapsed experiment time (hours)",
        yaxis_title="Signal (V)",
        legend_title_text="Trace",
        hovermode="x unified",
    )
    recording_figure = recording_figure.update_xaxes(
        range=[0, total_schedule_hours],
        dtick=6,
    )
    return (recording_figure,)


@app.cell
def _(mo, recording_figure):
    mo.ui.plotly(
        recording_figure,
        config={"scrollZoom": True, "responsive": True},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random and scheduled phasic calcium events

    Phasic calcium events can be layered onto the tonic signal in two ways:

    - `add_random_calcium_events` draws event starts from a seeded random
      process. Set `rate_per_minute`, optionally constrain
      `start_window_seconds`, and use `SyntheticDoricConfig.seed` to reproduce
      the same realized event times.
    - `add_scheduled_calcium_events` accepts exact `event_times_seconds`
      relative to the start of each targeted recording. Optional `channels` and
      `series_numbers` selectors use Doric's one-based numbering.

    Both builders return a new signal configuration, so random and scheduled
    sources can be added sequentially when a simulation needs both. For
    example, known event starts can be specified directly:

    ```python
    from circadian_fiber_photometry.simulation import (
        add_scheduled_calcium_events,
    )

    signal_with_known_events = add_scheduled_calcium_events(
        signal_config,
        event_times_seconds=(60.0, 240.0, 480.0),
        amplitude=0.08,
        name="known phasic events",
    )
    ```

    The next example instead adds seeded random events to one 10-minute,
    60 Hz recording. Ground-truth event starts returned by the simulator are
    marked above the traces.
    """)
    return


@app.cell
def _(
    SyntheticDoricConfig,
    add_random_calcium_events,
    channel_count,
    fs,
    generate_synthetic_doric,
    inter_series_gap_seconds,
    load_doric,
    notebook_dir,
    np,
    session_duration_seconds,
    signal_config,
):
    random_event_rate_per_minute = 1.0
    random_event_amplitude = 0.08
    random_event_window_seconds = (30.0, 570.0)
    random_phasic_signal_config = add_random_calcium_events(
        signal_config,
        rate_per_minute=random_event_rate_per_minute,
        amplitude=random_event_amplitude,
        start_window_seconds=random_event_window_seconds,
        channels=(1,),
        series_numbers=(1,),
        name="seeded random phasic events",
    )
    random_phasic_config = SyntheticDoricConfig(
        series_count=1,
        session_duration_seconds=session_duration_seconds,
        inter_series_gap_seconds=inter_series_gap_seconds,
        fs=fs,
        channel_count=channel_count,
        seed=321,
        signal=random_phasic_signal_config,
    )

    random_phasic_output_path = (
        notebook_dir / "generated" / "simulator_demonstration_random_phasic.doric"
    )
    random_phasic_summary = generate_synthetic_doric(
        random_phasic_output_path,
        random_phasic_config,
        overwrite=True,
    )
    random_phasic_dataset = load_doric(random_phasic_output_path)
    random_phasic_event_indices = random_phasic_summary.event_sample_indices[(1, 1)]
    random_phasic_event_times_minutes = (
        random_phasic_summary.event_times_seconds[(1, 1)] / 60
    )

    assert random_phasic_event_indices.size > 0
    assert np.all(
        random_phasic_event_times_minutes >= random_event_window_seconds[0] / 60
    )
    assert np.all(
        random_phasic_event_times_minutes < random_event_window_seconds[1] / 60
    )
    return (
        random_event_amplitude,
        random_event_rate_per_minute,
        random_phasic_dataset,
        random_phasic_event_indices,
        random_phasic_event_times_minutes,
        random_phasic_output_path,
        random_phasic_signal_config,
    )


@app.cell
def _(
    mo,
    random_event_amplitude,
    random_event_rate_per_minute,
    random_phasic_event_times_minutes,
    random_phasic_output_path,
):
    realized_event_times = ", ".join(
        f"{event_time:.2f}" for event_time in random_phasic_event_times_minutes
    )
    mo.md(
        f"""
        This seeded run generated **{len(random_phasic_event_times_minutes)}**
        random events at **{realized_event_times} minutes**, using a rate of
        **{random_event_rate_per_minute:g} events/minute** and amplitude
        **{random_event_amplitude:g} V**.

        Generated file: `{random_phasic_output_path}`
        """
    )
    return


@app.cell
def _(
    go,
    np,
    random_phasic_dataset,
    random_phasic_event_indices,
    random_phasic_event_times_minutes,
    session_duration_seconds,
    signal_config,
    tonic_amplitude,
    tonic_frequency_hz,
):
    random_phasic_time_minutes = (
        random_phasic_dataset.timestamps[:, 0]
        - random_phasic_dataset.timestamps[0, 0]
    ) / 60
    random_phasic_isosbestic = random_phasic_dataset.isosbestic_405[:, 0, 0]
    random_phasic_calcium = random_phasic_dataset.calcium_465[:, 0, 0]
    random_phasic_tonic_reference = (
        signal_config.calcium_baseline
        + tonic_amplitude
        * np.sin(
            2
            * np.pi
            * tonic_frequency_hz
            * random_phasic_dataset.timestamps[:, 0]
        )
    )
    event_marker_level = np.full(
        random_phasic_event_indices.size,
        np.max(random_phasic_calcium),
    )

    random_phasic_figure = go.Figure()
    random_phasic_figure.add_scattergl(
        x=random_phasic_time_minutes,
        y=random_phasic_isosbestic,
        name="405 nm isosbestic",
        mode="lines",
        line={"color": "#4575b4", "width": 1.5},
    )
    random_phasic_figure.add_scattergl(
        x=random_phasic_time_minutes,
        y=random_phasic_tonic_reference,
        name="Tonic calcium without phasic events",
        mode="lines",
        line={"color": "#5f6368", "width": 1.5, "dash": "dot"},
    )
    random_phasic_figure.add_scattergl(
        x=random_phasic_time_minutes,
        y=random_phasic_calcium,
        name="465 nm calcium with phasic events",
        mode="lines",
        line={"color": "#d73027", "width": 1.5},
    )
    random_phasic_figure.add_scatter(
        x=random_phasic_event_times_minutes,
        y=event_marker_level,
        name="Ground-truth random event start",
        mode="markers",
        marker={"color": "#1f2937", "size": 8, "symbol": "triangle-down"},
        hovertemplate="%{x:.2f} min<extra>%{fullData.name}</extra>",
    )
    random_phasic_figure.update_layout(
        template="plotly_white",
        title="Seeded random phasic calcium events",
        xaxis_title="Time within recording (minutes)",
        yaxis_title="Signal (V)",
        legend_title_text="Trace",
        hovermode="x unified",
    )
    random_phasic_figure = random_phasic_figure.update_xaxes(
        range=[0, session_duration_seconds / 60]
    )
    return (random_phasic_figure,)


@app.cell
def _(mo, random_phasic_figure):
    mo.ui.plotly(
        random_phasic_figure,
        config={"scrollZoom": True, "responsive": True},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Additive Gaussian noise

    The final demonstration takes the same seeded random-phasic signal shown
    above and layers independent Gaussian noise onto the recorded channels with
    `add_gaussian_noise`. Each noise source draws per-sample values from a
    zero-mean normal distribution with a configurable standard deviation for the
    465 nm calcium and 405 nm isosbestic streams, so the tonic sinusoid and
    phasic transients remain the underlying signal while acquisition-style noise
    is superimposed on top.

    Because the simulator draws the isosbestic noise before the random-event
    starts are placed, the noisy run realizes its own event times. The dotted
    gray trace is the clean tonic reference, and the ground-truth event starts
    below come from this noisy run's summary.
    """)
    return


@app.cell
def _(
    SyntheticDoricConfig,
    add_gaussian_noise,
    channel_count,
    fs,
    generate_synthetic_doric,
    inter_series_gap_seconds,
    load_doric,
    notebook_dir,
    np,
    random_phasic_signal_config,
    session_duration_seconds,
):
    gaussian_noise_calcium_std = 0.01
    gaussian_noise_isosbestic_std = 0.004
    noisy_phasic_signal_config = add_gaussian_noise(
        random_phasic_signal_config,
        calcium_std=gaussian_noise_calcium_std,
        isosbestic_std=gaussian_noise_isosbestic_std,
        name="additive Gaussian noise",
    )
    noisy_phasic_config = SyntheticDoricConfig(
        series_count=1,
        session_duration_seconds=session_duration_seconds,
        inter_series_gap_seconds=inter_series_gap_seconds,
        fs=fs,
        channel_count=channel_count,
        seed=321,
        signal=noisy_phasic_signal_config,
    )

    noisy_phasic_output_path = (
        notebook_dir / "generated" / "simulator_demonstration_noisy_phasic.doric"
    )
    noisy_phasic_summary = generate_synthetic_doric(
        noisy_phasic_output_path,
        noisy_phasic_config,
        overwrite=True,
    )
    noisy_phasic_dataset = load_doric(noisy_phasic_output_path)
    noisy_phasic_event_times_minutes = (
        noisy_phasic_summary.event_times_seconds[(1, 1)] / 60
    )

    assert noisy_phasic_event_times_minutes.size > 0
    assert noisy_phasic_dataset.calcium_465.shape[2] == 1
    return (
        gaussian_noise_calcium_std,
        gaussian_noise_isosbestic_std,
        noisy_phasic_dataset,
        noisy_phasic_event_times_minutes,
        noisy_phasic_output_path,
    )


@app.cell
def _(
    gaussian_noise_calcium_std,
    gaussian_noise_isosbestic_std,
    mo,
    noisy_phasic_output_path,
):
    mo.md(
        f"""
        The recorded channels now carry additive Gaussian noise with a
        **{gaussian_noise_calcium_std:g} V** standard deviation on the 465 nm
        calcium stream and **{gaussian_noise_isosbestic_std:g} V** on the
        405 nm isosbestic stream.

        Generated file: `{noisy_phasic_output_path}`
        """
    )
    return


@app.cell
def _(
    go,
    np,
    noisy_phasic_dataset,
    noisy_phasic_event_times_minutes,
    session_duration_seconds,
    signal_config,
    tonic_amplitude,
    tonic_frequency_hz,
):
    noisy_phasic_time_minutes = (
        noisy_phasic_dataset.timestamps[:, 0]
        - noisy_phasic_dataset.timestamps[0, 0]
    ) / 60
    noisy_phasic_isosbestic = noisy_phasic_dataset.isosbestic_405[:, 0, 0]
    noisy_phasic_calcium = noisy_phasic_dataset.calcium_465[:, 0, 0]
    noisy_phasic_tonic_reference = (
        signal_config.calcium_baseline
        + tonic_amplitude
        * np.sin(
            2
            * np.pi
            * tonic_frequency_hz
            * noisy_phasic_dataset.timestamps[:, 0]
        )
    )
    noisy_event_marker_level = np.full(
        noisy_phasic_event_times_minutes.size,
        np.max(noisy_phasic_calcium),
    )

    noisy_phasic_figure = go.Figure()
    noisy_phasic_figure.add_scattergl(
        x=noisy_phasic_time_minutes,
        y=noisy_phasic_isosbestic,
        name="405 nm isosbestic with Gaussian noise",
        mode="lines",
        line={"color": "#4575b4", "width": 1.5},
    )
    noisy_phasic_figure.add_scattergl(
        x=noisy_phasic_time_minutes,
        y=noisy_phasic_tonic_reference,
        name="Tonic calcium without phasic events",
        mode="lines",
        line={"color": "#5f6368", "width": 1.5, "dash": "dot"},
    )
    noisy_phasic_figure.add_scattergl(
        x=noisy_phasic_time_minutes,
        y=noisy_phasic_calcium,
        name="465 nm calcium with phasic events and Gaussian noise",
        mode="lines",
        line={"color": "#d73027", "width": 1.5},
    )
    noisy_phasic_figure.add_scatter(
        x=noisy_phasic_event_times_minutes,
        y=noisy_event_marker_level,
        name="Ground-truth random event start",
        mode="markers",
        marker={"color": "#1f2937", "size": 8, "symbol": "triangle-down"},
        hovertemplate="%{x:.2f} min<extra>%{fullData.name}</extra>",
    )
    noisy_phasic_figure.update_layout(
        template="plotly_white",
        title="Seeded random phasic calcium events with additive Gaussian noise",
        xaxis_title="Time within recording (minutes)",
        yaxis_title="Signal (V)",
        legend_title_text="Trace",
        hovermode="x unified",
    )
    noisy_phasic_figure = noisy_phasic_figure.update_xaxes(
        range=[0, session_duration_seconds / 60]
    )
    return (noisy_phasic_figure,)


@app.cell
def _(mo, noisy_phasic_figure):
    mo.ui.plotly(
        noisy_phasic_figure,
        config={"scrollZoom": True, "responsive": True},
    )
    return


if __name__ == "__main__":
    app.run()
