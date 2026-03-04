"""GRA-GRG feature screening: Grey Relational Analysis with Grey Relational Grade.

Ranks features by their grey relational grade to the target, selects the top-k.

GRA is a non-parametric, model-free method that measures similarity between
sequences. It works well with small samples and does not require normality —
suitable for financial time-series feature selection.

Algorithm
=========
1. Normalize all features and target to [0, 1] via min-max scaling.
2. For each feature, compute the absolute difference sequence vs. the target.
3. Compute Grey Relational Coefficient (GRC) per time step:
       GRC_i = (Δ_min + ξ·Δ_max) / (Δ_i + ξ·Δ_max)
   where ξ ∈ (0, 1) is the distinguishing coefficient (default 0.5).
4. Grey Relational Grade (GRG) = mean of GRC across all time steps.
5. Rank features by GRG descending, select top k.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1]. Returns NaN if constant."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def grey_relational_grades(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    xi: float = 0.5,
) -> pd.Series:
    """Compute Grey Relational Grade for each feature against the target.

    Parameters
    ----------
    df : DataFrame with features and target (no NaN allowed).
    feature_cols : list of feature column names.
    target_col : name of the target column.
    xi : distinguishing coefficient, typically 0.5.

    Returns
    -------
    pd.Series indexed by feature name, values = GRG, sorted descending.
    """
    # Normalize target
    target_norm = _min_max_normalize(df[target_col])

    grades: dict[str, float] = {}
    for col in feature_cols:
        feat_norm = _min_max_normalize(df[col])

        # Absolute difference sequence
        delta = (target_norm - feat_norm).abs()

        delta_min = delta.min()
        delta_max = delta.max()

        # Grey Relational Coefficient
        grc = (delta_min + xi * delta_max) / (delta + xi * delta_max)

        # Grey Relational Grade = mean GRC
        grades[col] = float(grc.mean())

    result = pd.Series(grades).sort_values(ascending=False)
    return result


def select_top_k_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int,
    target_col: str = "target",
    xi: float = 0.5,
) -> tuple[List[str], pd.Series]:
    """Run GRA-GRG screening and return the top-k feature names.

    Parameters
    ----------
    df : DataFrame with features and target (NaN rows should be dropped first).
    feature_cols : candidate feature column names.
    top_k : number of features to select. Clamped to len(feature_cols).
    target_col : target column name.
    xi : distinguishing coefficient.

    Returns
    -------
    (selected_features, all_grades) where selected_features is a list of the
    top-k column names and all_grades is the full GRG ranking.
    """
    top_k = min(top_k, len(feature_cols))

    grades = grey_relational_grades(df, feature_cols, target_col, xi)

    selected = grades.head(top_k).index.tolist()
    logger.info(
        "GRA-GRG screening: selected %d / %d features. Top-5 GRG: %s",
        top_k,
        len(feature_cols),
        grades.head(5).to_dict(),
    )

    return selected, grades
