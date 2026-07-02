"""Input/output adapters for photometry datasets."""

from .doric import DoricFileError, load_doric

__all__ = ["DoricFileError", "load_doric"]
