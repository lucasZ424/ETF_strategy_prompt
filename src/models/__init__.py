"""Models package: splitter, training, and model I/O."""

from .splitter import SplitResult, chronological_split
from .trainer import (
    DashboardTrainResult,
    TrainResult,
    get_feature_cols,
    load_feature_manifest,
    load_model,
    safe_X,
    save_dashboard_bundle,
    save_model_bundle,
    train_alpha_regressor,
    train_dashboard_regressor,
    train_gate_classifier,
    train_regime_classifier,
)

__all__ = [
    "DashboardTrainResult",
    "SplitResult",
    "TrainResult",
    "chronological_split",
    "get_feature_cols",
    "load_feature_manifest",
    "load_model",
    "safe_X",
    "save_dashboard_bundle",
    "save_model_bundle",
    "train_alpha_regressor",
    "train_dashboard_regressor",
    "train_gate_classifier",
    "train_regime_classifier",
]
