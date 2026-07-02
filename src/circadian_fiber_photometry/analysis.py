"""Backward-compatible analysis imports.

New code can import tonic functions from ``circadian_fiber_photometry.tonic``,
phasic functions from ``circadian_fiber_photometry.phasic``, and composed
workflows from ``circadian_fiber_photometry.pipeline``.
"""

from .phasic import count_events, extract_light_pulse_windows, irls_dynamic_correction
from .pipeline import analyze_sessions
from .tonic import fit_405_to_465

__all__ = [
    "analyze_sessions",
    "count_events",
    "extract_light_pulse_windows",
    "fit_405_to_465",
    "irls_dynamic_correction",
]
