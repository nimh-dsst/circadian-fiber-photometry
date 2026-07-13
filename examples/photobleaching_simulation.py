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

    This example generates short, deterministic Doric-style datasets comparing
    photobleaching without protein turnover, the default 48-hour turnover
    half-life, a faster custom half-life, and a double-exponential model.
    Generated files live in a temporary directory and are not committed to the
    repository.

    A bleach-susceptible pool $x_i$ with amplitude $A_i$ follows

    $$\frac{dx_i}{dt} = k_T(A_i-x_i)-k_{B,i}x_i$$

    during illumination. In a dark inter-session gap, $k_{B,i}=0$, so protein
    turnover replenishes the unbleached pool. The plotted wall-clock gap makes
    that recovery visible. The simulator uses $B=1-\sum_i A_i+\sum_i x_i$ as
    the multiplicative baseline factor.

    Warm yellow bands mark recorded illumination, and the slate band marks the
    unrecorded dark phase. Horizontal padding separates the rapid drops at the
    beginning and end of the timeline from the plot frame.
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
        SyntheticPhotobleachingComponentConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        generate_synthetic_doric,
    )

    return (
        Path,
        SyntheticDoricConfig,
        SyntheticPhotobleachingComponentConfig,
        SyntheticPhotobleachingConfig,
        SyntheticSignalConfig,
        TemporaryDirectory,
        generate_synthetic_doric,
        go,
        load_doric,
        np,
    )


@app.cell
def _(SyntheticPhotobleachingComponentConfig, SyntheticPhotobleachingConfig):
    shared_component = SyntheticPhotobleachingComponentConfig(
        amplitude_fraction=0.8,
        time_constant_seconds=18.0,
    )
    model_configs = {
        "no photobleaching": SyntheticPhotobleachingConfig(model="none"),
        "turnover disabled": SyntheticPhotobleachingConfig(
            isosbestic_components=(shared_component,),
            calcium_components=(shared_component,),
            turnover_half_life_hours=None,
        ),
        "48 h turnover": SyntheticPhotobleachingConfig(
            isosbestic_components=(shared_component,),
            calcium_components=(shared_component,),
        ),
        "4 h turnover": SyntheticPhotobleachingConfig(
            isosbestic_components=(shared_component,),
            calcium_components=(shared_component,),
            turnover_half_life_hours=4.0,
        ),
        "double, 48 h turnover": SyntheticPhotobleachingConfig(
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
    inter_series_gap_seconds = 12.0 * 3600.0
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

    reference_summary = simulation_summaries["no photobleaching"]
    samples_per_series = reference_summary.samples_per_series
    wall_clock_hours = np.concatenate(
        [
            generation_series_start
            + np.arange(samples_per_series, dtype=float) / fs
            for generation_series_start in reference_summary.session_start_times
        ]
    ) / 3600.0
    session_duration_hours = samples_per_series / fs / 3600.0
    light_intervals_hours = tuple(
        (
            session_start_seconds / 3600.0,
            session_start_seconds / 3600.0 + session_duration_hours,
        )
        for session_start_seconds in reference_summary.session_start_times
    )
    dark_intervals_hours = tuple(
        (
            light_intervals_hours[interval_index][1],
            light_intervals_hours[interval_index + 1][0],
        )
        for interval_index in range(len(light_intervals_hours) - 1)
    )
    timeline_start_hours = light_intervals_hours[0][0]
    timeline_end_hours = light_intervals_hours[-1][1]
    timeline_padding_hours = 0.05 * (
        timeline_end_hours - timeline_start_hours
    )
    timeline_range_hours = (
        timeline_start_hours - timeline_padding_hours,
        timeline_end_hours + timeline_padding_hours,
    )
    assert (
        simulation_summaries["no photobleaching"].photobleaching.model == "none"
    )
    assert (
        simulation_summaries["48 h turnover"].photobleaching.model
        == "single_exponential"
    )
    assert (
        simulation_summaries["double, 48 h turnover"].photobleaching.model
        == "double_exponential"
    )
    assert (
        simulation_summaries["turnover disabled"]
        .photobleaching.turnover_half_life_hours
        is None
    )
    assert (
        simulation_summaries["48 h turnover"]
        .photobleaching.turnover_half_life_hours
        == 48.0
    )
    np.testing.assert_allclose(traces["no photobleaching"]["isosbestic"], 0.08)
    np.testing.assert_allclose(traces["no photobleaching"]["calcium"], 0.18)
    return (
        dark_intervals_hours,
        light_intervals_hours,
        simulation_summaries,
        temporary_directory,
        timeline_range_hours,
        traces,
        wall_clock_hours,
    )


@app.cell
def _(
    dark_intervals_hours,
    go,
    light_intervals_hours,
    timeline_range_hours,
    traces,
    wall_clock_hours,
):
    comparison_figure = go.Figure()
    model_colors = {
        "no photobleaching": "#4d4d4d",
        "turnover disabled": "#d73027",
        "48 h turnover": "#4575b4",
        "4 h turnover": "#1a9850",
        "double, 48 h turnover": "#984ea3",
    }
    for light_start_hours, light_end_hours in light_intervals_hours:
        comparison_figure.add_vrect(
            x0=light_start_hours,
            x1=light_end_hours,
            fillcolor="rgba(255, 193, 7, 0.18)",
            layer="below",
            line_width=0,
        )
        comparison_figure.add_annotation(
            x=(light_start_hours + light_end_hours) / 2.0,
            y=1.0,
            xref="x",
            yref="paper",
            text="Light",
            showarrow=False,
            yanchor="bottom",
            font={"color": "#9a6700", "size": 12},
        )
    for dark_start_hours, dark_end_hours in dark_intervals_hours:
        comparison_figure.add_vrect(
            x0=dark_start_hours,
            x1=dark_end_hours,
            fillcolor="rgba(71, 85, 105, 0.12)",
            layer="below",
            line_width=0,
        )
        comparison_figure.add_annotation(
            x=(dark_start_hours + dark_end_hours) / 2.0,
            y=1.0,
            xref="x",
            yref="paper",
            text="Dark",
            showarrow=False,
            yanchor="bottom",
            font={"color": "#475569", "size": 12},
        )
    for trace_model_name, trace_values in traces.items():
        comparison_figure.add_scatter(
            x=wall_clock_hours,
            y=trace_values["calcium"] / trace_values["calcium"][0],
            mode="lines",
            name=trace_model_name,
            line={"color": model_colors[trace_model_name], "width": 2},
        )
    comparison_figure.update_layout(
        template="plotly_white",
        title="Photobleaching and protein turnover across a 12-hour dark gap",
        xaxis_title="Wall-clock time (hours)",
        yaxis_title="Normalized calcium baseline",
        legend_title_text="Model",
    )
    comparison_figure.update_xaxes(range=timeline_range_hours)
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
                "turnover half-life (h)": metadata.turnover_half_life_hours,
                "turnover rate (1/s)": metadata.turnover_rate_per_second,
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
