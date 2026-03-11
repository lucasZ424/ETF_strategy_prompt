"""XGBoost model training with Optuna hyperparameter search.

Model architecture is pending redesign — the previous ranker + gate
dual-model approach was removed because the engineered features
(vol-adjusted 5d forward returns) had poor correlation with the
converted y_rank (daily integer rank).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_META_COLS = {"date", "symbol"}


def get_feature_cols(df) -> List[str]:
    return sorted([c for c in df.columns if c not in _META_COLS])


def safe_X(df, feature_cols: List[str]) -> np.ndarray:
    raw = df[feature_cols].values.astype(float)
    return np.nan_to_num(np.where(np.isinf(raw), np.nan, raw), nan=0.0)
