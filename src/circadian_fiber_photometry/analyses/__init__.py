"""Analysis discovery and compatibility imports."""

from .registry import (
    ANALYSES,
    PhasicAnalysis,
    TonicAnalysis,
    get_analysis,
    list_analyses,
    run_analysis,
)

__all__ = [
    "ANALYSES",
    "PhasicAnalysis",
    "TonicAnalysis",
    "get_analysis",
    "list_analyses",
    "run_analysis",
]
