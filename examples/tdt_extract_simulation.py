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
    # Simulated TDT read-block objects

    This example generates a small Doric-style dataset and exports its
    normalized streams into a pickle with the structure returned by
    `tdt.read_block`. The output is not a proprietary TDT tank and cannot be
    opened as `.tsq` or `.tev` data.

    Install the package with its `tdt-export` extra before running the notebook.
    Generated files live in a temporary directory and are removed when the
    notebook process exits.
    """)
    return


@app.cell
def _():
    import pickle
    from datetime import datetime
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from circadian_fiber_photometry import load_doric
    from circadian_fiber_photometry.simulation import (
        SyntheticDoricConfig,
        SyntheticTDTSubject,
        export_synthetic_tdt_blocks,
        generate_synthetic_doric,
    )

    return (
        Path,
        SyntheticDoricConfig,
        SyntheticTDTSubject,
        TemporaryDirectory,
        datetime,
        export_synthetic_tdt_blocks,
        generate_synthetic_doric,
        load_doric,
        pickle,
    )


@app.cell
def _(
    Path,
    SyntheticDoricConfig,
    SyntheticTDTSubject,
    TemporaryDirectory,
    datetime,
    export_synthetic_tdt_blocks,
    generate_synthetic_doric,
    load_doric,
):
    temporary_directory = TemporaryDirectory(prefix="tdt-block-example-")
    output_root = Path(temporary_directory.name)
    doric_path = output_root / "source.doric"
    generation_summary = generate_synthetic_doric(
        doric_path,
        SyntheticDoricConfig(
            series_count=2,
            session_duration_seconds=12.8,
            inter_series_gap_seconds=1787.2,
            fs=20.0,
            channel_count=1,
            seed=123,
        ),
    )
    dataset = load_doric(doric_path)
    batch_summary = export_synthetic_tdt_blocks(
        output_root / "experiment",
        dataset,
        cohort="Cohort 1",
        subjects=(
            SyntheticTDTSubject(
                subject_id="MouseA",
                channel_number=1,
                isosbestic_store_name="405A",
                experimental_store_name="470d",
                site="vta",
            ),
        ),
        base_datetime=datetime(2025, 1, 1, 12, 0, 0),
    )
    return (
        batch_summary,
        dataset,
        generation_summary,
        output_root,
        temporary_directory,
    )


@app.cell
def _(batch_summary, dataset, generation_summary, pickle):
    first_record = batch_summary.records[0]
    with first_record.summary.block_file.open("rb") as file_in:
        block = pickle.load(file_in)
    experimental_stream = block.streams._470d

    assert batch_summary.extracted_count == dataset.series_count
    assert generation_summary.samples_per_series == 256
    assert experimental_stream.__class__.__module__ == "tdt"
    assert experimental_stream.__class__.__name__ == "StructType"
    assert list(experimental_stream.keys()) == [
        "name",
        "code",
        "size",
        "type",
        "type_str",
        "ucf",
        "fs",
        "dform",
        "start_time",
        "data",
        "channel",
    ]
    return block, experimental_stream, first_record


@app.cell
def _(batch_summary, block, experimental_stream, first_record, mo):
    mo.vstack(
        [
            mo.md("## Exported subject sessions"),
            mo.ui.table(
                [
                    {
                        "subject": record.subject_id,
                        "series": record.series_name,
                        "datetime": record.summary.datetime,
                        "directory": str(record.summary.path),
                    }
                    for record in batch_summary.records
                ],
                pagination=False,
            ),
            mo.md(
                f"""
                The pickle root contains
                `{", ".join(block.keys())}`. Its experimental stream is a
                `{experimental_stream.__class__.__module__}.`
                `{experimental_stream.__class__.__name__}` with
                `{experimental_stream.data.size}` float32 samples.

                Files: `{", ".join(path.name for path in first_record.summary.files)}`
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
