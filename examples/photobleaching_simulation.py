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
    # Photobleaching model comparison

    This example generates three short, deterministic Doric-style datasets
    with photobleaching disabled, single-exponential photobleaching, and
    double-exponential photobleaching. Generated files live in a temporary
    directory and are not committed to the repository.

    For components with fractional losses $A_i$ and time constants $\tau_i$,
    the simulator applies

    $$B(t) = 1 - \sum_i A_i + \sum_i A_i e^{-t / \tau_i}.$$

    Time is cumulative recorded exposure in seconds, so decay pauses during
    gaps between sessions.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np
    import plotly.graph_objects as go

    from circadian_fiber_photometry import load_doric
    from circadian_fiber_photometry.simulation import (
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        generate_synthetic_doric,
    )

    return (
        Path,
        SyntheticDoricConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        TemporaryDirectory,
        generate_synthetic_doric,
        go,
        load_doric,
        np,
    )


@app.cell
def _(SyntheticPhotobleachingConfig):
    model_configs = {
        "none": SyntheticPhotobleachingConfig(model="none"),
        "single exponential": SyntheticPhotobleachingConfig(
            model="single_exponential"
        ),
        "double exponential": SyntheticPhotobleachingConfig(
            model="double_exponential"
        ),
    }
    return (model_configs,)


@app.cell
def _(
    Path,
    SyntheticDoricConfig,
    SyntheticSignalConfig,
    TemporaryDirectory,
    generate_synthetic_doric,
    load_doric,
    model_configs,
    np,
):
    fs = 20.0
    series_count = 2
    session_duration_seconds = 12.0
    inter_series_gap_seconds = 600.0
    temporary_directory = TemporaryDirectory(prefix="photobleaching-example-")
    output_directory = Path(temporary_directory.name)
    simulation_summaries = {}
    traces = {}

    for generation_model_name, generation_photobleaching in model_configs.items():
        generation_signal = SyntheticSignalConfig(
            channel_baseline_step=0.0,
            photobleaching=generation_photobleaching,
            artifact_amplitude=0.0,
            circadian_amplitude=0.0,
            noise_std=0.0,
            analog_noise_std=0.0,
            transient_rate_per_minute=0.0,
            transient_amplitude=0.0,
        )
        generation_config = SyntheticDoricConfig(
            series_count=series_count,
            session_duration_seconds=session_duration_seconds,
            inter_series_gap_seconds=inter_series_gap_seconds,
            fs=fs,
            channel_count=1,
            seed=123,
            signal=generation_signal,
        )
        generation_path = output_directory / f"{generation_model_name}.doric"
        generation_summary = generate_synthetic_doric(
            generation_path,
            generation_config,
        )
        generation_dataset = load_doric(generation_path)
        simulation_summaries[generation_model_name] = generation_summary
        traces[generation_model_name] = {
            "isosbestic": generation_dataset.isosbestic_405[:, 0, :]
            .T.reshape(-1),
            "calcium": generation_dataset.calcium_465[:, 0, :].T.reshape(-1),
        }

    exposure_seconds = np.arange(
        series_count * simulation_summaries["none"].samples_per_series,
        dtype=float,
    ) / fs
    assert simulation_summaries["none"].photobleaching.model == "none"
    assert (
        simulation_summaries["single exponential"].photobleaching.model
        == "single_exponential"
    )
    assert (
        simulation_summaries["double exponential"].photobleaching.model
        == "double_exponential"
    )
    np.testing.assert_allclose(traces["none"]["isosbestic"], 0.08)
    np.testing.assert_allclose(traces["none"]["calcium"], 0.18)
    return exposure_seconds, simulation_summaries, temporary_directory, traces


@app.cell
def _(go, traces, exposure_seconds):
    comparison_figure = go.Figure()
    model_colors = {
        "none": "#4d4d4d",
        "single exponential": "#4575b4",
        "double exponential": "#d73027",
    }
    for trace_model_name, trace_values in traces.items():
        comparison_figure.add_scatter(
            x=exposure_seconds,
            y=trace_values["calcium"] / trace_values["calcium"][0],
            mode="lines",
            name=trace_model_name,
            line={"color": model_colors[trace_model_name], "width": 2},
        )
    comparison_figure.update_layout(
        template="plotly_white",
        title="Calcium baseline under selectable photobleaching models",
        xaxis_title="Cumulative exposure (seconds)",
        yaxis_title="Normalized calcium baseline",
        legend_title_text="Model",
    )
    return (comparison_figure,)


@app.cell
def _(comparison_figure, mo):
    mo.ui.plotly(comparison_figure, config={"responsive": True})
    return


@app.cell
def _(simulation_summaries):
    metadata_rows = []
    for metadata_model_name, metadata_summary in simulation_summaries.items():
        metadata = metadata_summary.photobleaching
        metadata_rows.append(
            {
                "model": metadata_model_name,
                "time basis": metadata.time_basis,
                "isosbestic amplitudes": [
                    item.amplitude_fraction
                    for item in metadata.isosbestic_components
                ],
                "isosbestic taus (s)": [
                    item.time_constant_seconds
                    for item in metadata.isosbestic_components
                ],
                "calcium amplitudes": [
                    item.amplitude_fraction for item in metadata.calcium_components
                ],
                "calcium taus (s)": [
                    item.time_constant_seconds for item in metadata.calcium_components
                ],
            }
        )
    return (metadata_rows,)


@app.cell
def _(metadata_rows, mo):
    mo.vstack(
        [
            mo.md("## Resolved simulator metadata"),
            mo.ui.table(metadata_rows, pagination=False),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
