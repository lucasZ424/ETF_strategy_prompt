"""Target construction package for trading pipeline."""

from .alpha_target import build_alpha_target
from .dashboard_target import build_dashboard_targets
from .regime_labels import build_regime_labels
from .triple_barrier import build_barrier_labels

__all__ = [
    "build_alpha_target",
    "build_barrier_labels",
    "build_dashboard_targets",
    "build_regime_labels",
]
