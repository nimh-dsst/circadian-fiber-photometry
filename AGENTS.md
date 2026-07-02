# Repository Instructions

## Tooling

* Use `uv` as the Python package manager for dependency management and command execution.
* Use `pytest` as the testing environment.
* Prefer running all project commands through `uv`.

Examples:

```bash
uv run pytest
uv run ruff check .
uv run python -m build
```

Do not introduce alternative package managers such as `pipenv`, `poetry`, or Conda-specific workflows unless explicitly requested.

## Repository Purpose

This repository has four primary purposes:

1. Preserve the original MATLAB source material.
2. Convert MATLAB functions into a modular Python package.
3. Organize photometry analyses by signal time scale.
4. Provide a comprehensive Doric HDF5 file simulator.

The Python package should be developed as an independent scientific analysis engine that can be used from scripts, notebooks, tests, command-line tools, and external applications such as a Streamlit app. The Streamlit app must be treated as a downstream consumer, not as the owner of analysis logic.

## MATLAB Source Preservation

The `matlab_scripts/` folder houses unedited MATLAB scripts by Qijun Tang.

These files should remain faithful to the original source material.

Do not:

* Reformat MATLAB files.
* Translate MATLAB files in place.
* Rename MATLAB functions or variables inside the original files.
* Modify MATLAB scripts as part of Python package work.
* Move original MATLAB source into the importable Python package.

The MATLAB scripts are retained for attribution, reproducibility, and validation. Python implementations should reference the MATLAB scripts conceptually, but the Python code should live separately in the Python package.

## Python Package Design

The repository should provide a clean, tested, importable Python package suitable for PyPI publication.

Use a `src/` package layout:

```text
src/
  circadian_fiber_photometry/
    __init__.py
    io/
    simulation/
    preprocessing/
    analyses/
    models.py
    pipeline.py
```

The package should expose stable, user-facing APIs. Avoid designing functions around any particular UI framework.

Good package-level responsibilities:

* Doric HDF5 loading and writing.
* Data validation.
* Simulation of synthetic Doric-style datasets.
* Signal preprocessing.
* Tonic analyses.
* Phasic analyses.
* Analysis orchestration.
* Structured result objects.
* Exportable tables and figures.

Do not add Streamlit as a dependency of this package. Streamlit-specific code belongs in the separate Streamlit application repository.

## Streamlit Compatibility Without Streamlit Coupling

This package should be designed so that a Streamlit app can easily consume it, but it should not contain Streamlit app code.

The package should expose plain Python interfaces such as:

```python
from circadian_fiber_photometry import load_doric, run_analysis

dataset = load_doric("example.h5")

result = run_analysis(
    dataset,
    analysis="tonic",
    config={
        "signal_channel": "465",
        "control_channel": "405",
    },
)
```

The Streamlit app should be able to call this package, inspect available analyses, build parameter forms, run selected analyses, and display result tables or figures.

Do not import or use:

```python
import streamlit
```

inside this repository.

If a change is needed only for Streamlit presentation, prefer one of the following:

* Add structured metadata to an analysis.
* Improve the result object.
* Add a general-purpose plotting or export helper.
* Add an adapter in the Streamlit app repository instead of this repository.

## Analysis Organization

The Python package should reflect the two analysis categories present in the MATLAB scripts.

### Tonic Analyses

Tonic analyses are long time scale analyses focused on:

* Global isosbestic correction.
* Sliding window averaging.
* Percentile filtering.
* Separation of tonic and phasic signal contributions.

Tonic analysis code should be organized under an appropriate module, for example:

```text
src/circadian_fiber_photometry/analyses/tonic/
```

or:

```text
src/circadian_fiber_photometry/analyses/tonic.py
```

Choose the simpler structure unless the module becomes too large.

### Phasic Analyses

Phasic analyses are short time scale analyses focused on:

* Intra-session event identification.
* Ordinary least squares fitting of the isosbestic signal.
* Iteratively reweighted least squares fitting of the isosbestic signal.
* Peak identification.
* Integrated fluorescence.
* Power spectra analysis.

Phasic analysis code should be organized under an appropriate module, for example:

```text
src/circadian_fiber_photometry/analyses/phasic/
```

or:

```text
src/circadian_fiber_photometry/analyses/phasic.py
```

Choose the simpler structure unless the module becomes too large.

## Analysis Registry

Analyses should be discoverable through a registry or equivalent mechanism so downstream applications can offer modular analysis choices.

A downstream app should be able to ask the package:

* What analyses are available?
* What parameters does each analysis accept?
* What input data does each analysis require?
* What outputs does each analysis produce?

Prefer a structure like:

```python
ANALYSES = {
    "tonic": TonicAnalysis,
    "phasic": PhasicAnalysis,
}
```

Each analysis should clearly define:

* `name`
* `description`
* expected inputs
* configuration schema or documented configuration fields
* output structure
* `run()` behavior

Avoid scattering analysis selection logic throughout the codebase.

## Result Objects

Analysis functions should return structured result objects rather than UI-specific output.

Prefer dataclasses, Pydantic models, typed dictionaries, or simple documented classes.

A result should be able to contain:

* Analysis name.
* Input metadata.
* Parameters used.
* Output tables.
* Output arrays.
* Optional figure objects or figure-ready data.
* Warnings.
* Provenance information.
* Version information where practical.

Example shape:

```python
AnalysisResult(
    name="tonic",
    tables={...},
    arrays={...},
    figures={...},
    metadata={...},
    parameters={...},
    warnings=[...],
)
```

Do not return Streamlit elements, widgets, or app-specific state.

## Doric HDF5 Simulator

The repository should include a simulator for generating Doric-style HDF5 files so users can build synthetic datasets for testing tonic and phasic analysis functions.

Simulator code and related assets should live in a dedicated simulator or simulation area and remain separate from the tonic and phasic analysis code.

Preferred package location:

```text
src/circadian_fiber_photometry/simulation/
```

Additional non-package simulator assets may live in:

```text
simulator/
```

The simulator should support:

* Synthetic signal generation.
* Configurable sampling rates.
* Configurable recording duration.
* Tonic signal components.
* Phasic event components.
* Noise models.
* Known ground-truth metadata.
* Doric-style HDF5 file writing.
* Batch dataset generation where practical.

The simulator is not only for demos. It should also serve as a test oracle for validating the analysis code against known ground truth.

## Doric I/O

Doric HDF5 reading and writing should be isolated from analysis code.

Prefer a module such as:

```text
src/circadian_fiber_photometry/io/doric.py
```

The I/O layer should handle:

* HDF5 file opening.
* Dataset discovery.
* Channel extraction.
* Metadata extraction.
* Validation of expected fields.
* Graceful errors for unsupported or malformed files.

Analysis modules should operate on normalized Python data structures, not directly on raw HDF5 internals whenever possible.

## Testing Strategy

Every meaningful Python translation from MATLAB should have tests.

Use `pytest`.

At minimum, include tests for:

* Doric HDF5 loading.
* Doric HDF5 simulation.
* Tonic analysis behavior.
* Phasic analysis behavior.
* Numerical comparison against known MATLAB outputs where available.
* Error handling for invalid inputs.
* Public API imports.
* Package installation behavior where practical.

Run tests with:

```bash
uv run pytest
```

Use fixtures for reusable test data.

Store small test fixtures in:

```text
tests/fixtures/
```

Avoid committing large binary datasets unless necessary. Prefer simulator-generated fixtures when possible.

## MATLAB Validation

The MATLAB scripts should serve as the reference implementation during Python translation.

When practical:

1. Run the original MATLAB code on representative input data.
2. Save expected MATLAB outputs as CSV, JSON, Parquet, NumPy, or another stable format.
3. Run the Python implementation on the same input.
4. Compare outputs within documented numerical tolerances.

Tests should prefer explicit tolerances, for example:

```python
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
```

If exact equivalence is not possible due to algorithmic or library differences, document the reason in the test or module docstring.

## Dependency Policy

Keep the core package lightweight.

Appropriate core dependencies may include:

* `numpy`
* `scipy`
* `pandas`
* `h5py`

Optional functionality should use optional dependencies.

Examples:

* Plotting extras.
* Documentation extras.
* Development dependencies.
* Simulator extras, if simulator dependencies become heavy.

Do not add GUI or app-framework dependencies to the core package.

Avoid adding dependencies unless they provide clear value and cannot be reasonably replaced by existing package dependencies.

## Public API Stability

Be deliberate about what is exported from the package.

The package should provide a small, documented public API through:

```text
src/circadian_fiber_photometry/__init__.py
```

Prefer stable entry points such as:

```python
load_doric
simulate_doric_dataset
run_analysis
list_analyses
AnalysisResult
```

Internal helpers should remain inside submodules and should not be treated as public API unless intentionally documented.

## Coding Style

Write readable, idiomatic Python.

Prefer:

* Small functions with clear responsibilities.
* Type hints for public functions.
* Descriptive names.
* NumPy/SciPy/Pandas idioms where appropriate.
* Explicit units in variable names or docstrings.
* Clear exceptions for invalid inputs.
* Deterministic behavior in tests.

Avoid:

* Translating MATLAB line-for-line when it produces awkward Python.
* Hidden global state.
* UI-specific assumptions.
* Large monolithic analysis functions.
* Mutating input data unexpectedly.
* Silent numerical failures.

## Documentation

Documentation should help both package users and downstream app developers.

Include or maintain:

* README usage examples.
* Installation instructions.
* Attribution to the original MATLAB author.
* Explanation of tonic versus phasic analyses.
* Doric simulator examples.
* API examples.
* Validation notes.
* Limitations and assumptions.

The README should make clear that:

* The original MATLAB scripts are preserved in `matlab_scripts/`.
* The Python package is a modular reimplementation.
* The simulator can generate Doric-style HDF5 test datasets.
* The package can be used independently of the Streamlit app.

## Attribution

Preserve credit for the original MATLAB work.

The repository should include attribution in appropriate places, such as:

* README.
* `AUTHORS.md`, if present.
* `CITATION.cff`, if present.
* Module or documentation notes where helpful.

Do not remove attribution to Qijun Tang from original MATLAB materials.

## Development Workflow

During active development with a downstream Streamlit app, install this package into the Streamlit app environment in editable mode from the app repository:

```bash
uv pip install -e ../circadian-fiber-photometry
```

The Streamlit app should consume this package through its public API.

Do not copy analysis code from this repository into the Streamlit app repository.

Do not make this repository depend on the Streamlit app repository.

## Continuous Integration Expectations

CI should eventually verify:

* Formatting and linting.
* Unit tests.
* Simulator tests.
* MATLAB comparison tests where expected outputs exist.
* Package build.
* Import of the built package.
* Public API smoke tests.

Useful commands:

```bash
uv run pytest
uv run ruff check .
uv run python -m build
```

## File Organization Guidelines

Preferred high-level structure:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── matlab_scripts/
├── simulator/
├── src/
│   └── circadian_fiber_photometry/
│       ├── __init__.py
│       ├── io/
│       ├── simulation/
│       ├── preprocessing/
│       ├── analyses/
│       ├── models.py
│       └── pipeline.py
└── tests/
    ├── fixtures/
    ├── test_io.py
    ├── test_simulation.py
    ├── test_tonic.py
    ├── test_phasic.py
    └── test_public_api.py
```

This structure is a guideline, not a rigid requirement. Prefer simple organization at first, then split modules when they become difficult to maintain.

## Agent Behavior Guidelines

When working in this repository:

* Preserve the MATLAB scripts.
* Keep Python package code independent from Streamlit.
* Prefer public APIs that downstream tools can call.
* Add or update tests when changing analysis behavior.
* Use simulator-generated data where practical.
* Keep Doric I/O separate from analysis logic.
* Keep analysis logic separate from plotting and UI concerns.
* Avoid large, unrelated refactors.
* Update documentation when changing public behavior.
* Prefer small, reviewable commits or patches.
* Do not introduce unnecessary dependencies.
* Do not claim MATLAB/Python numerical equivalence unless tests demonstrate it.

When unsure whether a change belongs in this package or the Streamlit app, default to this rule:

* Scientific computation, data models, validation, simulation, and reusable plotting helpers belong here.
* Upload workflows, widgets, caching, page layout, user interaction, and app-specific display logic belong in the Streamlit app.
