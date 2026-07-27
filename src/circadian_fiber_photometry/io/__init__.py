"""Input/output adapters for photometry datasets."""

from .doric import DoricFileError, load_doric
from .tdt import (
    DEFAULT_POINTS_PER_ROW,
    PICKLE_PROTOCOL,
    TDTDependencyError,
    TDTExtractEpoc,
    TDTExtractError,
    TDTExtractMetadata,
    TDTExtractStream,
    TDTExtractSummary,
    write_tdt_extract,
)

__all__ = [
    "DEFAULT_POINTS_PER_ROW",
    "DoricFileError",
    "PICKLE_PROTOCOL",
    "TDTDependencyError",
    "TDTExtractEpoc",
    "TDTExtractError",
    "TDTExtractMetadata",
    "TDTExtractStream",
    "TDTExtractSummary",
    "load_doric",
    "write_tdt_extract",
]
