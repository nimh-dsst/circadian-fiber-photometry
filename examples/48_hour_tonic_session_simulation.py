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
    # 48-hour sessioned Doric simulation

    This example creates a synthetic Doric file for a 48-hour schedule with
    10 minutes of active recording followed by a 20-minute gap. The generated
    file contains one Doric `SeriesNNNN` group for each active recording
    session. The final 20-minute dark interval is part of the 48-hour schedule,
    but it is not stored as sampled data.

    The calcium signal includes an additive tonic component with amplitude
    `1.0` and a 12-hour period. Digital IO channel 1 contains two behavior event
    types: `button` is encoded by one TTL pulse, and `treat taken` is encoded by
    two TTL pulses. Pulses are 50 ms wide, with a 50 ms off interval separating
    pulses in a multi-pulse sequence.

    Because this is a marimo notebook, cells run from their declared
    dependencies instead of execution order. The source is plain Python and
    contains no saved output state.
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
        SyntheticTTLBehaviorCodeConfig,
        SyntheticTTLBehaviorEventConfig,
        add_gaussian_noise,
        add_tonic_component,
        generate_synthetic_doric,
    )

    notebook_dir = Path(__file__).resolve().parent
    return (
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        SyntheticTTLBehaviorCodeConfig,
        SyntheticTTLBehaviorEventConfig,
        add_gaussian_noise,
        add_tonic_component,
        generate_synthetic_doric,
        go,
        load_doric,
        notebook_dir,
        np,
    )


@app.cell
def _(np):
    total_schedule_hours = 48
    session_duration_seconds = 10 * 60
    inter_series_gap_seconds = 20 * 60
    cycle_seconds = session_duration_seconds + inter_series_gap_seconds
    series_count = int(total_schedule_hours * 60 * 60 / cycle_seconds)

    fs = 60.0
    channel_count = 1
    tonic_period_hours = 12
    tonic_frequency_hz = 1 / (tonic_period_hours * 60 * 60)

    isosbestic_noise_std = 0.005
    calcium_noise_std = 0.020
    analog_in_noise_std = 0.050

    ttl_pulse_width_seconds = 0.050
    ttl_pulse_off_interval_seconds = 0.050
    ttl_pulse_width_samples = int(round(ttl_pulse_width_seconds * fs))
    ttl_pulse_off_samples = int(round(ttl_pulse_off_interval_seconds * fs))

    active_recording_hours = series_count * session_duration_seconds / 3600
    schedule_hours_including_final_gap = series_count * cycle_seconds / 3600

    assert series_count == 96
    assert np.isclose(active_recording_hours, 16.0)
    assert np.isclose(schedule_hours_including_final_gap, total_schedule_hours)
    assert ttl_pulse_width_samples == 3
    assert ttl_pulse_off_samples == 3
    return (
        active_recording_hours,
        analog_in_noise_std,
        calcium_noise_std,
        channel_count,
        cycle_seconds,
        fs,
        inter_series_gap_seconds,
        isosbestic_noise_std,
        schedule_hours_including_final_gap,
        series_count,
        session_duration_seconds,
        tonic_frequency_hz,
        tonic_period_hours,
        total_schedule_hours,
        ttl_pulse_off_interval_seconds,
        ttl_pulse_width_samples,
        ttl_pulse_width_seconds,
    )


@app.cell
def _(
    SyntheticDoricConfig,
    SyntheticPhotobleachingConfig,
    SyntheticSignalConfig,
    SyntheticTTLBehaviorCodeConfig,
    SyntheticTTLBehaviorEventConfig,
    add_gaussian_noise,
    add_tonic_component,
    analog_in_noise_std,
    calcium_noise_std,
    channel_count,
    fs,
    generate_synthetic_doric,
    inter_series_gap_seconds,
    isosbestic_noise_std,
    notebook_dir,
    series_count,
    session_duration_seconds,
    tonic_frequency_hz,
    ttl_pulse_off_interval_seconds,
    ttl_pulse_width_seconds,
):
    base_signal = SyntheticSignalConfig(
        photobleaching=SyntheticPhotobleachingConfig(model="none"),
        artifact_amplitude=0.0,
        circadian_amplitude=0.0,
        noise_std=0.0,
        analog_noise_std=0.0,
        transient_rate_per_minute=0.0,
        transient_amplitude=0.0,
    )
    tonic_signal = add_tonic_component(
        base_signal,
        amplitude=1.0,
        frequency_hz=tonic_frequency_hz,
        name="12-hour tonic component",
    )
    signal_config = add_gaussian_noise(
        tonic_signal,
        isosbestic_std=isosbestic_noise_std,
        calcium_std=calcium_noise_std,
        analog_in_std=analog_in_noise_std,
        name="measurement noise",
    )

    simulation_config = SyntheticDoricConfig(
        series_count=series_count,
        session_duration_seconds=session_duration_seconds,
        inter_series_gap_seconds=inter_series_gap_seconds,
        fs=fs,
        channel_count=channel_count,
        seed=123,
        signal=signal_config,
        ttl_behavior_codes=(
            SyntheticTTLBehaviorCodeConfig("button", channel=1, pulse_count=1),
            SyntheticTTLBehaviorCodeConfig("treat taken", channel=1, pulse_count=2),
        ),
        ttl_behavior_events=(
            SyntheticTTLBehaviorEventConfig("button", start_seconds=(60.0, 300.0)),
            SyntheticTTLBehaviorEventConfig(
                "treat taken", start_seconds=(180.0, 420.0)
            ),
        ),
        ttl_pulse_width_seconds=ttl_pulse_width_seconds,
        ttl_pulse_off_interval_seconds=ttl_pulse_off_interval_seconds,
    )

    output_path = notebook_dir / "generated" / "48_hour_tonic_session_simulation.doric"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_summary = generate_synthetic_doric(
        output_path,
        simulation_config,
        overwrite=True,
    )
    return output_path, signal_config, simulation_summary


@app.cell
def _(
    active_recording_hours,
    analog_in_noise_std,
    calcium_noise_std,
    cycle_seconds,
    fs,
    inter_series_gap_seconds,
    isosbestic_noise_std,
    load_doric,
    mo,
    np,
    output_path,
    schedule_hours_including_final_gap,
    series_count,
    session_duration_seconds,
    signal_config,
    simulation_summary,
    tonic_frequency_hz,
    tonic_period_hours,
    total_schedule_hours,
    ttl_pulse_off_interval_seconds,
    ttl_pulse_width_samples,
    ttl_pulse_width_seconds,
):
    dataset = load_doric(output_path)
    button_events = [
        event
        for event in simulation_summary.ttl_behavior_events
        if event.code_name == "button"
    ]
    treat_events = [
        event
        for event in simulation_summary.ttl_behavior_events
        if event.code_name == "treat taken"
    ]
    first_treat_event = treat_events[0]
    inter_pulse_gap_seconds = (
        first_treat_event.pulse_sample_indices[1]
        - (first_treat_event.pulse_sample_indices[0] + ttl_pulse_width_samples)
    ) / fs

    np.testing.assert_allclose(simulation_summary.session_start_times[0], 0.0)
    np.testing.assert_allclose(
        np.diff(simulation_summary.session_start_times), cycle_seconds
    )
    np.testing.assert_allclose(
        dataset.session_start_times, simulation_summary.session_start_times
    )
    np.testing.assert_allclose(schedule_hours_including_final_gap, total_schedule_hours)
    np.testing.assert_allclose(active_recording_hours, 16.0)
    np.testing.assert_allclose(dataset.fs, fs)
    np.testing.assert_allclose(inter_pulse_gap_seconds, ttl_pulse_off_interval_seconds)
    assert signal_config.gaussian_noise[0].calcium_std == calcium_noise_std
    assert signal_config.gaussian_noise[0].isosbestic_std == isosbestic_noise_std
    assert len(button_events) == series_count * 2
    assert len(treat_events) == series_count * 2
    assert button_events[0].pulse_sample_indices.size == 1
    assert first_treat_event.pulse_sample_indices.size == 2

    session_duration_minutes = session_duration_seconds / 60
    gap_duration_minutes = inter_series_gap_seconds / 60

    mo.vstack(
        [
            mo.md("## Generated dataset"),
            mo.hstack(
                [
                    mo.stat(str(simulation_summary.series_count), label="Sessions"),
                    mo.stat(f"{dataset.fs:.1f} Hz", label="Sampling frequency"),
                    mo.stat(
                        f"{active_recording_hours:.1f} h",
                        label="Active recording",
                    ),
                    mo.stat(
                        f"{schedule_hours_including_final_gap:.1f} h",
                        label="Full schedule",
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                f"""
                - **Generated file:** `{output_path}`
                - **Session duration / gap:**
                  {session_duration_minutes:.1f} / {gap_duration_minutes:.1f} minutes
                - **Session start spacing:**
                  {np.median(np.diff(simulation_summary.session_start_times)) / 60:.1f}
                  minutes
                - **Tonic period / frequency:** {tonic_period_hours:.1f} hours /
                  {tonic_frequency_hz:.8g} Hz
                - **Gaussian noise standard deviation:** 405 =
                  {isosbestic_noise_std:g}, 465 = {calcium_noise_std:g}, analog =
                  {analog_in_noise_std:g}
                - **TTL encoding:** button = 1 pulse; treat taken = 2 pulses;
                  pulse width = {ttl_pulse_width_seconds * 1000:.0f} ms; off interval =
                  {ttl_pulse_off_interval_seconds * 1000:.0f} ms
                - **Behavior events:** {len(button_events)} button events and
                  {len(treat_events)} treat-taken events

                All configuration and ground-truth checks passed.
                """
            ),
        ]
    )
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal overview

    The loaded Doric dataset stores traces as `(samples, channels, sessions)`.
    Session means keep the full 48-hour structure visible without sending
    millions of raw samples to the browser.
    """)
    return


@app.cell
def _(dataset, go, mo):
    session_start_hours = dataset.session_start_times / 3600
    isosbestic_session_means = dataset.isosbestic_405[:, 0, :].mean(axis=0)
    calcium_session_means = dataset.calcium_465[:, 0, :].mean(axis=0)

    overview_figure = go.Figure()
    overview_figure.add_scatter(
        x=session_start_hours,
        y=isosbestic_session_means,
        name="405 nm isosbestic",
        mode="lines+markers",
        line={"color": "#4575b4", "width": 1},
        marker={"size": 4},
    )
    overview_figure.add_scatter(
        x=session_start_hours,
        y=calcium_session_means,
        name="465 nm experimental",
        mode="lines+markers",
        line={"color": "#d73027", "width": 1},
        marker={"size": 4},
    )
    overview_figure.update_layout(
        template="plotly_white",
        title="Session means across the 48-hour schedule",
        xaxis_title="Elapsed experiment time (hours)",
        yaxis_title="Mean raw signal (V)",
        legend_title_text="Signal",
        hovermode="x unified",
    )

    mo.ui.plotly(
        overview_figure,
        config={"scrollZoom": True, "responsive": True},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full-resolution session view

    Pick any session to inspect all of its raw samples. Changing the slider
    reruns only the dependent plot cell; it does not regenerate the dataset.
    """)
    return


@app.cell
def _(dataset, mo):
    session_picker = mo.ui.slider(
        start=1,
        stop=dataset.series_count,
        step=1,
        value=1,
        show_value=True,
        include_input=True,
        label="Session number",
        full_width=True,
    )
    mo.vstack([session_picker])
    return (session_picker,)


@app.cell
def _(dataset, go, mo, session_picker):
    selected_session_index = int(session_picker.value) - 1
    selected_time_minutes = (
        dataset.timestamps[:, selected_session_index]
        - dataset.timestamps[0, selected_session_index]
    ) / 60

    session_figure = go.Figure()
    session_figure.add_scattergl(
        x=selected_time_minutes,
        y=dataset.isosbestic_405[:, 0, selected_session_index],
        name="405 nm isosbestic",
        mode="lines",
        line={"color": "#4575b4", "width": 1},
    )
    session_figure.add_scattergl(
        x=selected_time_minutes,
        y=dataset.calcium_465[:, 0, selected_session_index],
        name="465 nm experimental",
        mode="lines",
        line={"color": "#d73027", "width": 1},
    )
    session_start_hour = dataset.session_start_times[selected_session_index] / 3600
    session_figure.update_layout(
        template="plotly_white",
        title=(
            f"Raw signals for session {selected_session_index + 1} "
            f"(starts at {session_start_hour:.1f} h)"
        ),
        xaxis_title="Time within session (minutes)",
        yaxis_title="Raw signal (V)",
        legend_title_text="Signal",
        hovermode="x unified",
    )

    mo.ui.plotly(
        session_figure,
        config={"scrollZoom": True, "responsive": True},
    )
    return


if __name__ == "__main__":
    app.run()
