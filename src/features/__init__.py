"""Features package exports."""

from .builder import TradeDatasets, build_trade_features
from .gate_features import add_gate_features
from .price_features import (
    add_return_features,
    add_mean_reversion_features,
    add_trend_features,
    add_volatility_extras,
    add_volume_features,
    add_relative_strength_features,
    add_cross_sectional_features,
)
from .regime_features import build_regime_features
from .screening import (
    pearson_correlation_filter,
    xgboost_importance_filter,
    run_feature_selection,
)
from .tags import FEATURE_TAGS, features_for, tags_for
from .volatility import add_volatility_features

__all__ = [
    "TradeDatasets",
    "build_trade_features",
    "add_return_features",
    "add_mean_reversion_features",
    "add_trend_features",
    "add_volatility_extras",
    "add_volume_features",
    "add_relative_strength_features",
    "add_cross_sectional_features",
    "add_gate_features",
    "build_regime_features",
    "add_volatility_features",
    "pearson_correlation_filter",
    "xgboost_importance_filter",
    "run_feature_selection",
    "FEATURE_TAGS",
    "features_for",
    "tags_for",
]
