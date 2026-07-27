from __future__ import annotations

from circadian_fiber_photometry import (
    AnalysisResult,
    DoricDataset,
    SyntheticTDTSubject,
    TDTExtractStream,
    export_synthetic_tdt_extracts,
    list_analyses,
    load_doric,
    run_analysis,
    write_tdt_extract,
)
from circadian_fiber_photometry.analyses import ANALYSES
from circadian_fiber_photometry.analyses.phasic import count_events
from circadian_fiber_photometry.analyses.tonic import compute_tonic_level
from circadian_fiber_photometry.simulation import (
    SyntheticGaussianNoiseConfig,
    SyntheticPhotobleachingComponentConfig,
    SyntheticPhotobleachingConfig,
    SyntheticPhotobleachingMetadata,
    SyntheticRandomCalciumEventConfig,
    SyntheticScheduledCalciumEventConfig,
    SyntheticTonicComponentConfig,
    add_gaussian_noise,
    add_random_calcium_events,
    add_scheduled_calcium_events,
    add_tonic_component,
    generate_synthetic_doric,
)


def test_public_reorganized_imports_are_available() -> None:
    assert AnalysisResult.__name__ == "AnalysisResult"
    assert DoricDataset.__name__ == "DoricDataset"
    assert callable(load_doric)
    assert callable(run_analysis)
    assert callable(count_events)
    assert callable(compute_tonic_level)
    assert callable(generate_synthetic_doric)
    assert TDTExtractStream.__name__ == "TDTExtractStream"
    assert SyntheticTDTSubject.__name__ == "SyntheticTDTSubject"
    assert callable(write_tdt_extract)
    assert callable(export_synthetic_tdt_extracts)
    assert SyntheticGaussianNoiseConfig.__name__ == "SyntheticGaussianNoiseConfig"
    assert (
        SyntheticPhotobleachingComponentConfig.__name__
        == "SyntheticPhotobleachingComponentConfig"
    )
    assert SyntheticPhotobleachingConfig.__name__ == "SyntheticPhotobleachingConfig"
    assert (
        SyntheticPhotobleachingMetadata.__name__
        == "SyntheticPhotobleachingMetadata"
    )
    assert (
        SyntheticRandomCalciumEventConfig.__name__
        == "SyntheticRandomCalciumEventConfig"
    )
    assert (
        SyntheticScheduledCalciumEventConfig.__name__
        == "SyntheticScheduledCalciumEventConfig"
    )
    assert SyntheticTonicComponentConfig.__name__ == "SyntheticTonicComponentConfig"
    assert callable(add_gaussian_noise)
    assert callable(add_random_calcium_events)
    assert callable(add_scheduled_calcium_events)
    assert callable(add_tonic_component)
    assert set(ANALYSES) == {spec.name for spec in list_analyses()}
