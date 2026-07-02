# Repository Instructions

## Tooling

- Use `uv` as the Python package manager for dependency management and command execution.
- Use `pytest` as the testing environment.
- Prefer running project commands through `uv`, for example:

```bash
uv run pytest
```

## Repository Purpose

This repository has four primary purposes:

1. Preserve the original MATLAB source material.

   The `matlab_scripts/` folder houses unedited MATLAB scripts by Qijun Tang. These files should remain faithful to the original source material and should not be reformatted, translated in place, or modified as part of Python package work.

2. Convert MATLAB functions into a Python package.

   The Python package should provide clear, tested implementations of the functions and analyses represented in the MATLAB scripts. Translations should favor readable, idiomatic Python while preserving the scientific intent and numerical behavior of the original analyses.

3. Organize analyses by signal time scale.

   The Python package should reflect the two analysis categories present in the MATLAB scripts:

   - Tonic analyses: long time scale analyses focused on global isosbestic correction, sliding window averaging, and percentile filtering to separate tonic and phasic contributions to the signal.
   - Phasic analyses: short time scale analyses focused on identifying intra-session events, ordinary least squares fitting of the isosbestic signal, iteratively reweighted least squares fitting of the isosbestic signal, peak identification, integrated fluorescence, and power spectra analysis.

4. Provide a comprehensive Doric HDF5 file simulator.

   The repository should include a simulator for generating Doric files, which are HDF5 files, so users can build large synthetic datasets for testing the tonic and phasic analysis functions. Simulator code and related assets should live in a dedicated `simulator/` subdirectory and remain separate from the tonic and phasic analysis code.
