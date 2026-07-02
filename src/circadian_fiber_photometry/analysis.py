"""Backward-compatible analysis imports.

New code can use ``circadian_fiber_photometry.analyses`` for discovery and
registered dispatch, or import tonic/phasic functions from their dedicated
modules.
"""

from .analyses import ANALYSES, get_analysis, list_analyses, run_analysis
from .phasic import count_events, extract_light_pulse_windows, irls_dynamic_correction
from .pipeline import analyze_sessions
from .tonic import fit_405_to_465

__all__ = [
    "ANALYSES",
    "analyze_sessions",
    "count_events",
    "extract_light_pulse_windows",
    "fit_405_to_465",
    "get_analysis",
    "irls_dynamic_correction",
    "list_analyses",
    "run_analysis",
]
