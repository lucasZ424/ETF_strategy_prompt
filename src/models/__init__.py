"""Models package: chronological splitter + training utilities."""

from .splitter import SplitResult, chronological_split
from .trainer import get_feature_cols, safe_X

__all__ = [
    "SplitResult",
    "chronological_split",
    "get_feature_cols",
    "safe_X",
]
