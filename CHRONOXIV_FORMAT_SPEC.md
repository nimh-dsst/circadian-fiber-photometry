# ChronoXIV Simulated TDT Format

ChronoXIV simulated TDT input represents the result of calling
`tdt.read_block`, not the files ChronoXIV creates after extracting selected
streams. Each recording folder therefore contains one trusted pickle.

## Directory layout

```text
Photometry/
  Cohort 1/
    Simulated TDT/
      Subject01__acc-left+acc-right__M-20250102-030405/
        tdt_block.pkl
```

Each recording folder matches:

```regex
^(?P<subjects>[A-Za-z0-9._-]+(?:\+[A-Za-z0-9._-]+)?)__(?P<sites>[A-Za-z0-9-]+(?:\+[A-Za-z0-9-]+)?)__(?P<sexes>[MFUO](?:\+[MFUO])?)-(?P<datetime>\d{8}-\d{6})$
```

A block can contain one or two sites (and TDT acquisitions may similarly
contain one or two subjects). The generated ACC dataset uses the folder site
IDs `acc-left+acc-right` because the pattern does not permit underscores; the
stream names retain the requested `acc_left` and `acc_right` identity through
their `_left` and `_right` suffixes. In the generated example, subjects 01–05
are designated `M` and subjects 06–10 are designated `F`.

## Pickled object

`tdt_block.pkl` is a protocol-4 pickle of the top-level `tdt.StructType` that
the TDT Python package returns from `tdt.read_block`. Its top-level fields are:

```python
block.epocs
block.snips
block.streams
block.scalars
block.info
```

These fields and their nested records are also `tdt.StructType` objects. For a
two-site recording with a shared TTL waveform and an injection epoc, the useful
data are accessed like this:

```python
import pickle

with open("tdt_block.pkl", "rb") as file_in:
    block = pickle.load(file_in)

ttl = block.streams.TTL
left_isosbestic = block.streams["405_left"]
right_isosbestic = block.streams["405_right"]
left_experimental = block.streams["470_left"]
right_experimental = block.streams["470_right"]
injections = block.epocs.Inj1
```

Each stream has the fields emitted by `tdt.read_block`: `name`, `code`,
`size`, `type`, `type_str`, `ucf`, `fs`, `dform`, `start_time`, `data`, and
`channel`. Epocs carry `name`, `onset`, `offset`, `type`, `type_str`, `data`,
`dform`, and `size`. `block.info` records the logical tank path, block name,
start and stop datetimes, duration, and channel-selection metadata.

The `tdt` package must be installed when loading the pickle. As with every
pickle, only load files created by a trusted source because unpickling can
execute code.

## Generation

Use `write_tdt_block` for one recording or `export_synthetic_tdt_blocks` for
series from a simulated Doric dataset. Neither function creates proprietary
TDT `.tsq`, `.tev`, `.tbk`, or `.sev` files.

```python
from circadian_fiber_photometry.io import (
    TDTBlockMetadata,
    TDTBlockStream,
    write_tdt_block,
)

summary = write_tdt_block(
    "Subject01__acc-left+acc-right__M-20250102-030405",
    streams=(
        TDTBlockStream("TTL", ttl_data, fs=20.0, code_name="TTL1"),
        TDTBlockStream("405_left", left_iso, fs=20.0, code_name="4L05"),
        TDTBlockStream("405_right", right_iso, fs=20.0, code_name="4R05"),
        TDTBlockStream("470_left", left_exp, fs=20.0, code_name="4L70"),
        TDTBlockStream("470_right", right_exp, fs=20.0, code_name="4R70"),
    ),
    metadata=TDTBlockMetadata(
        subject_id="Subject01",
        subject_ids=("Subject01",),
        order="First",
        datetime=session_datetime,
        tank_dir=logical_block_path,
        block_name=logical_block_path.name,
    ),
)

assert summary.files == (summary.path / "tdt_block.pkl",)
```
