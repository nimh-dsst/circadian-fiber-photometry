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
    # Synthetic photometry artifacts

    This focused example adds every artifact model currently available to a
    clean Doric-style recording: a brief five-sample session-start spike,
    positive and negative scheduled boxes, fixed-count and rate-driven random
    boxes, and a photometry disconnection. Artifact timing is expressed in
    seconds relative to a series or to the full experiment, depending on the
    configured time reference.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import plotly.graph_objects as go

    from circadian_fiber_photometry import load_doric
    from circadian_fiber_photometry.simulation import (
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        add_random_box_artifacts,
        add_scheduled_box_artifacts,
        configure_photometry_disconnection,
        configure_session_start_spike,
        generate_synthetic_doric,
    )

    notebook_dir = Path(__file__).resolve().parent
    return (
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        add_random_box_artifacts,
        add_scheduled_box_artifacts,
        configure_photometry_disconnection,
        configure_session_start_spike,
        generate_synthetic_doric,
        go,
        load_doric,
        notebook_dir,
    )


@app.cell
def _(
    SyntheticDoricConfig,
    SyntheticPhotobleachingConfig,
    SyntheticSignalConfig,
    add_random_box_artifacts,
    add_scheduled_box_artifacts,
    configure_photometry_disconnection,
    configure_session_start_spike,
    generate_synthetic_doric,
    notebook_dir,
):
    sampling_rate_hz = 20.0
    base_signal = SyntheticSignalConfig(
        photobleaching=SyntheticPhotobleachingConfig(model="none"),
        artifact_amplitude=0.0,
        circadian_amplitude=0.0,
        noise_std=0.0,
        analog_noise_std=0.0,
        transient_rate_per_minute=0.0,
        transient_amplitude=0.0,
    )
    artifact_signal = configure_session_start_spike(
        base_signal,
        duration_seconds=5 / sampling_rate_hz,
        name="five-sample startup spike",
    )
    artifact_signal = add_scheduled_box_artifacts(
        artifact_signal,
        [3.0],
        durations_seconds=1.0,
        magnitude_fraction=0.10,
        name="scheduled positive box",
    )
    artifact_signal = add_scheduled_box_artifacts(
        artifact_signal,
        [6.0],
        durations_seconds=0.75,
        magnitude_fraction=-0.10,
        name="scheduled negative box",
    )
    artifact_signal = add_random_box_artifacts(
        artifact_signal,
        count_per_series=2,
        start_window_seconds=(8.0, 12.0),
        duration_range_seconds=(0.5, 0.75),
        magnitude_fraction=0.10,
        name="two random boxes",
    )
    artifact_signal = add_random_box_artifacts(
        artifact_signal,
        rate_per_minute=30.0,
        start_window_seconds=(13.0, 19.0),
        duration_range_seconds=(0.25, 0.5),
        magnitude_fraction=-0.10,
        name="rate-driven random drops",
    )
    artifact_signal = configure_photometry_disconnection(
        artifact_signal,
        isosbestic_floor=0.005,
        calcium_floor=0.007,
        name="equipment switched off",
    )

    simulation_config = SyntheticDoricConfig(
        series_count=1,
        session_duration_seconds=20.0,
        inter_series_gap_seconds=0.0,
        fs=sampling_rate_hz,
        channel_count=1,
        seed=123,
        signal=artifact_signal,
    )
    output_path = notebook_dir / "generated" / "artifact_simulation.doric"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_summary = generate_synthetic_doric(
        output_path,
        simulation_config,
        overwrite=True,
    )
    return artifact_summary, output_path, simulation_config


@app.cell
def _(artifact_summary, load_doric, mo, output_path):
    artifact_dataset = load_doric(output_path)
    artifact_rows = [
        {
            "type": item.artifact_type,
            "name": item.name,
            "samples": f"[{item.start_sample}, {item.stop_sample})",
            "start (s)": item.start_seconds_within_series,
            "duration (s)": item.realized_duration_seconds,
            "fraction": item.magnitude_fraction,
            "405 offset (V)": item.isosbestic_offset,
            "465 offset (V)": item.calcium_offset,
            "time reference": item.time_reference,
            "405 floor (V)": item.isosbestic_floor,
            "465 floor (V)": item.calcium_floor,
        }
        for item in artifact_summary.artifact_occurrences
    ]
    mo.vstack(
        [
            mo.md("## Realized ground truth"),
            mo.md(
                f"Generated `{output_path}` with "
                f"{len(artifact_rows)} artifact occurrences."
            ),
            mo.ui.table(artifact_rows, selection=None),
        ]
    )
    return (artifact_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Artifact trace

    Shaded regions come directly from the half-open sample bounds in the
    simulator ground truth. Positive artifacts are red and negative artifacts
    are blue; the equipment-disconnection interval is gray.
    """)
    return


@app.cell
def _(artifact_dataset, artifact_summary, go, mo, simulation_config):
    artifact_time = (
        artifact_dataset.timestamps[:, 0] - artifact_dataset.timestamps[0, 0]
    )
    artifact_figure = go.Figure()
    artifact_figure.add_scatter(
        x=artifact_time,
        y=artifact_dataset.isosbestic_405[:, 0, 0],
        name="405 nm isosbestic",
        mode="lines",
    )
    artifact_figure.add_scatter(
        x=artifact_time,
        y=artifact_dataset.calcium_465[:, 0, 0],
        name="465 nm experimental",
        mode="lines",
    )
    for artifact_item in artifact_summary.artifact_occurrences:
        if artifact_item.artifact_type == "photometry_disconnection":
            fill_color = "rgba(64, 64, 64, 0.22)"
        else:
            assert artifact_item.magnitude_fraction is not None
            fill_color = (
                "rgba(215, 48, 39, 0.18)"
                if artifact_item.magnitude_fraction > 0
                else "rgba(69, 117, 180, 0.18)"
            )
        artifact_figure.add_vrect(
            x0=artifact_item.start_sample / simulation_config.fs,
            x1=artifact_item.stop_sample / simulation_config.fs,
            fillcolor=fill_color,
            line_width=0,
        )
    artifact_figure.update_layout(
        template="plotly_white",
        title="Synthetic 405/465 traces with known artifacts",
        xaxis_title="Time within series (seconds)",
        yaxis_title="Signal (V)",
        hovermode="x unified",
    )
    mo.ui.plotly(artifact_figure, config={"responsive": True})
    return


if __name__ == "__main__":
    app.run()
