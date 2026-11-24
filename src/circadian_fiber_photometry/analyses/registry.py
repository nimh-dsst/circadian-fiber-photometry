"""Discoverable analysis registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..models import AnalysisResult, AnalysisSpec
from ..pipeline import analyze_sessions
from ..tonic import (
    compute_average_level,
    compute_raw_median_level,
    compute_tonic_level,
    detrend_levels_by_moving_window,
    fit_405_to_465,
    zscore_levels_by_moving_window,
)


class TonicAnalysis:
    """Long-timescale tonic photometry analysis."""

    name = "tonic"
    description = (
        "Global isosbestic correction with tonic percentile and moving-window "
        "session-level summaries."
    )
    expected_inputs = ("isosbestic_405", "calcium_465", "fs")
    config_fields = {
        "interval_hours": "Session spacing in hours for moving-window summaries.",
        "fitting_cutoff": "Percentile cutoff for excluding extreme fitting points.",
        "weight_fit": "Preserve the legacy MATLAB zero-block fit behavior.",
        "fit_weights": "Optional conventional weighted least-squares weights.",
        "percentile": "Percentile used for tonic dF/F level summaries.",
    }
    outputs = (
        "dff",
        "fitted_405",
        "fit_coefficients",
        "level_tonic",
        "level_tonic_detrended",
        "level_tonic_z",
        "level_average",
        "level_average_detrended",
        "level_405_raw",
        "level_465_raw",
    )

    @classmethod
    def spec(cls) -> AnalysisSpec:
        """Return discoverable metadata for this analysis."""

        return AnalysisSpec(
            name=cls.name,
            description=cls.description,
            expected_inputs=cls.expected_inputs,
            config_fields=dict(cls.config_fields),
            outputs=cls.outputs,
        )

    @classmethod
    def run(
        cls,
        dataset: Any,
        config: Mapping[str, Any] | None = None,
    ) -> AnalysisResult:
        """Run tonic analysis on a Doric dataset or array mapping."""

        parameters = _tonic_parameters(config)
        iso, calcium, fs = _signal_inputs(dataset, parameters)
        parameters["fs"] = fs
        fit = fit_405_to_465(
            iso,
            calcium,
            fs=fs,
            fitting_cutoff=parameters["fitting_cutoff"],
            weight_fit=parameters["weight_fit"],
            fit_weights=parameters["fit_weights"],
        )
        level_tonic = compute_tonic_level(
            fit.dff,
            percentile=parameters["percentile"],
        )
        level_average = compute_average_level(fit.dff)

        return AnalysisResult(
            name=cls.name,
            arrays={
                "dff": fit.dff,
                "fitted_405": fit.fitted_405,
                "fit_coefficients": fit.coefficients,
                "level_tonic": level_tonic,
                "level_tonic_detrended": detrend_levels_by_moving_window(
                    level_tonic,
                    interval_hours=parameters["interval_hours"],
                    window_hours=24,
                ),
                "level_tonic_z": zscore_levels_by_moving_window(
                    level_tonic,
                    interval_hours=parameters["interval_hours"],
                    window_hours=24,
                    ddof=1,
                ),
                "level_average": level_average,
                "level_average_detrended": detrend_levels_by_moving_window(
                    level_average,
                    interval_hours=parameters["interval_hours"],
                    window_hours=24,
                ),
                "level_405_raw": compute_raw_median_level(iso),
                "level_465_raw": compute_raw_median_level(calcium),
            },
            metadata=_analysis_metadata(dataset, iso),
            parameters=_serializable_parameters(parameters),
            provenance={"implementation": f"{cls.__module__}.{cls.__name__}"},
        )


class PhasicAnalysis:
    """Short-timescale phasic event and fluorescence analysis."""

    name = "phasic"
    description = (
        "Dynamic isosbestic correction with event counting, positive phasic "
        "trace extraction, and integrated phasic fluorescence."
    )
    expected_inputs = ("isosbestic_405", "calcium_465", "fs")
    config_fields = {
        "interval_hours": "Session spacing in hours for shared pipeline summaries.",
        "fitting_cutoff": "Percentile cutoff for excluding extreme fitting points.",
        "weight_fit": "Preserve the legacy MATLAB zero-block fit behavior.",
        "fit_weights": "Optional conventional weighted least-squares weights.",
    }
    outputs = (
        "dff_dynamic_corrected",
        "dff_adjusted",
        "dff_phasic",
        "fitted_dynamic",
        "calcium_filtered",
        "event_counts",
        "level_phasic",
    )

    @classmethod
    def spec(cls) -> AnalysisSpec:
        """Return discoverable metadata for this analysis."""

        return AnalysisSpec(
            name=cls.name,
            description=cls.description,
            expected_inputs=cls.expected_inputs,
            config_fields=dict(cls.config_fields),
            outputs=cls.outputs,
        )

    @classmethod
    def run(
        cls,
        dataset: Any,
        config: Mapping[str, Any] | None = None,
    ) -> AnalysisResult:
        """Run phasic analysis on a Doric dataset or array mapping."""

        parameters = _phasic_parameters(config)
        iso, calcium, fs = _signal_inputs(dataset, parameters)
        parameters["fs"] = fs
        result = analyze_sessions(
            iso,
            calcium,
            fs=fs,
            interval_hours=parameters["interval_hours"],
            fitting_cutoff=parameters["fitting_cutoff"],
            weight_fit=parameters["weight_fit"],
            fit_weights=parameters["fit_weights"],
        )

        return AnalysisResult(
            name=cls.name,
            arrays={
                "dff_dynamic_corrected": result.dff_dynamic_corrected,
                "dff_adjusted": result.dff_adjusted,
                "dff_phasic": result.dff_phasic,
                "fitted_dynamic": result.fitted_dynamic,
                "calcium_filtered": result.calcium_filtered,
                "event_counts": result.event_counts,
                "level_phasic": result.level_phasic,
            },
            metadata=_analysis_metadata(dataset, iso),
            parameters=_serializable_parameters(parameters),
            provenance={"implementation": f"{cls.__module__}.{cls.__name__}"},
        )


ANALYSES = {
    TonicAnalysis.name: TonicAnalysis,
    PhasicAnalysis.name: PhasicAnalysis,
}


def list_analyses() -> tuple[AnalysisSpec, ...]:
    """Return metadata for all registered analyses."""

    return tuple(analysis.spec() for analysis in ANALYSES.values())


def get_analysis(name: str) -> type[TonicAnalysis] | type[PhasicAnalysis]:
    """Return a registered analysis class by name."""

    try:
        return ANALYSES[name]
    except KeyError as exc:
        available = ", ".join(sorted(ANALYSES))
        raise ValueError(
            f"unknown analysis {name!r}; available analyses: {available}"
        ) from exc


def run_analysis(
    dataset: Any,
    *,
    analysis: str,
    config: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Run a registered analysis against a dataset or array mapping."""

    return get_analysis(analysis).run(dataset, config=config)


def _tonic_parameters(config: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(config or {})
    return {
        "fs": values.get("fs"),
        "interval_hours": float(values.get("interval_hours", 1.0)),
        "fitting_cutoff": float(values.get("fitting_cutoff", 0.0)),
        "weight_fit": bool(values.get("weight_fit", True)),
        "fit_weights": values.get("fit_weights"),
        "percentile": float(values.get("percentile", 10.0)),
    }


def _phasic_parameters(config: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(config or {})
    return {
        "fs": values.get("fs"),
        "interval_hours": float(values.get("interval_hours", 1.0)),
        "fitting_cutoff": float(values.get("fitting_cutoff", 0.0)),
        "weight_fit": bool(values.get("weight_fit", True)),
        "fit_weights": values.get("fit_weights"),
    }


def _signal_inputs(
    dataset: Any,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    iso = _field(dataset, "isosbestic_405")
    calcium = _field(dataset, "calcium_465")
    fs = parameters.get("fs")
    if fs is None:
        fs = _field(dataset, "fs")
    return np.asarray(iso, dtype=float), np.asarray(calcium, dtype=float), float(fs)


def _field(dataset: Any, name: str) -> Any:
    if isinstance(dataset, Mapping):
        try:
            return dataset[name]
        except KeyError as exc:
            raise ValueError(
                f"dataset mapping is missing required key {name!r}"
            ) from exc
    if not hasattr(dataset, name):
        raise ValueError(f"dataset is missing required attribute {name!r}")
    return getattr(dataset, name)


def _analysis_metadata(dataset: Any, iso: np.ndarray) -> dict[str, Any]:
    metadata = {
        "input_shape": tuple(int(value) for value in np.shape(iso)),
    }
    for name in ("path", "series_names", "channel_names"):
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            metadata[name] = str(value) if isinstance(value, (str, bytes)) else value
    return metadata


def _serializable_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    serializable = dict(parameters)
    if serializable.get("fit_weights") is not None:
        serializable["fit_weights"] = "<array>"
    return serializable


__all__ = [
    "ANALYSES",
    "PhasicAnalysis",
    "TonicAnalysis",
    "get_analysis",
    "list_analyses",
    "run_analysis",
]
