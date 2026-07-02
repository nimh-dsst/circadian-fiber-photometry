from __future__ import annotations

from circadian_fiber_photometry import (
    AnalysisResult,
    DoricDataset,
    list_analyses,
    load_doric,
    run_analysis,
)
from circadian_fiber_photometry.analyses import ANALYSES
from circadian_fiber_photometry.analyses.phasic import count_events
from circadian_fiber_photometry.analyses.tonic import compute_tonic_level
from circadian_fiber_photometry.simulation import generate_synthetic_doric


def test_public_reorganized_imports_are_available() -> None:
    assert AnalysisResult.__name__ == "AnalysisResult"
    assert DoricDataset.__name__ == "DoricDataset"
    assert callable(load_doric)
    assert callable(run_analysis)
    assert callable(count_events)
    assert callable(compute_tonic_level)
    assert callable(generate_synthetic_doric)
    assert set(ANALYSES) == {spec.name for spec in list_analyses()}
