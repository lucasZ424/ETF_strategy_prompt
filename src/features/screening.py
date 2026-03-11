"""Two-stage feature selection: Pearson correlation filter + XGBoost importance.

Stage A — Pearson Correlation Filter
=====================================
- Compute pairwise Pearson correlation on non-protected features.
- If |corr| >= threshold, drop the less interpretable / less horizon-aligned
  feature from the pair (heuristic: keep whichever has higher abs correlation
  with the target).
- Protected features (cross-market, macro, cross-sectional) are never dropped
  by this filter.

Stage B — XGBoost Importance Ranking (optional)
================================================
- Train a quick XGBoost regressor on the correlation-filtered feature set.
- Rank features by gain importance.
- Keep the top-k features. Protected features are always retained regardless
  of their importance ranking.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_protected(col: str, protected_prefixes: List[str]) -> bool:
    """Check if a feature column is protected from correlation filtering."""
    for prefix in protected_prefixes:
        if col.startswith(prefix) or col == prefix:
            return True
    return False


def pearson_correlation_filter(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    threshold: float = 0.90,
    protected_prefixes: List[str] | None = None,
) -> tuple[List[str], pd.DataFrame]:
    """Stage A: Remove one feature from highly correlated pairs.

    For each pair with |corr| >= threshold, keep the feature that has higher
    absolute Pearson correlation with the target. Protected features are never
    dropped.

    Parameters
    ----------
    df : DataFrame with feature columns and target.
    feature_cols : candidate feature column names.
    target_col : target column name.
    threshold : correlation threshold.
    protected_prefixes : feature name prefixes that are immune to dropping.

    Returns
    -------
    (surviving_features, correlation_matrix)
    """
    protected_prefixes = protected_prefixes or []

    # Separate protected and filterable features
    filterable = [c for c in feature_cols if not _is_protected(c, protected_prefixes)]
    protected = [c for c in feature_cols if _is_protected(c, protected_prefixes)]

    if not filterable:
        logger.info("Pearson filter: no filterable features, returning all.")
        return feature_cols, pd.DataFrame()

    # Compute correlation matrix on filterable features
    corr_matrix = df[filterable].corr(method="pearson")

    # Compute target correlations for tie-breaking
    target_corr = df[filterable].corrwith(df[target_col]).abs()

    # Iteratively remove the weaker feature from each high-correlation pair
    to_drop: set[str] = set()
    n_features = len(filterable)

    for i in range(n_features):
        if filterable[i] in to_drop:
            continue
        for j in range(i + 1, n_features):
            if filterable[j] in to_drop:
                continue
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                # Drop the one with lower target correlation
                feat_i, feat_j = filterable[i], filterable[j]
                if target_corr.get(feat_i, 0) >= target_corr.get(feat_j, 0):
                    to_drop.add(feat_j)
                    logger.info(
                        "Pearson filter: dropping '%s' (corr=%.3f with '%s', "
                        "target_corr=%.3f < %.3f)",
                        feat_j, corr_matrix.iloc[i, j], feat_i,
                        target_corr.get(feat_j, 0), target_corr.get(feat_i, 0),
                    )
                else:
                    to_drop.add(feat_i)
                    logger.info(
                        "Pearson filter: dropping '%s' (corr=%.3f with '%s', "
                        "target_corr=%.3f < %.3f)",
                        feat_i, corr_matrix.iloc[i, j], feat_j,
                        target_corr.get(feat_i, 0), target_corr.get(feat_j, 0),
                    )

    surviving_filterable = [c for c in filterable if c not in to_drop]
    surviving = surviving_filterable + protected

    logger.info(
        "Pearson filter: %d → %d features (dropped %d, %d protected kept).",
        len(feature_cols), len(surviving), len(to_drop), len(protected),
    )

    return surviving, corr_matrix


def xgboost_importance_filter(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    top_k: int | None = None,
    protected_prefixes: List[str] | None = None,
    seed: int = 42,
) -> tuple[List[str], pd.Series]:
    """Stage B: Rank features by XGBoost gain importance, keep top-k.

    Protected features are always retained regardless of ranking.

    Parameters
    ----------
    df : DataFrame with feature columns and target (NaN-free).
    feature_cols : candidate feature column names.
    target_col : target column name.
    top_k : number of features to keep. None = keep all (just rank).
    protected_prefixes : features always retained.
    seed : random seed for XGBoost.

    Returns
    -------
    (selected_features, importance_series)
    """
    import xgboost as xgb

    protected_prefixes = protected_prefixes or []

    X = df[feature_cols].values.copy()
    y = df[target_col].values.copy()

    # Replace inf/NaN for safe training
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)

    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    logger.info(
        "XGBoost importance: top-5 = %s",
        importance.head(5).to_dict(),
    )

    if top_k is None:
        return feature_cols, importance

    # Always keep protected features
    protected_set = {
        c for c in feature_cols if _is_protected(c, protected_prefixes)
    }
    non_protected = [c for c in importance.index if c not in protected_set]

    # Select top-k from non-protected, then add protected
    budget = max(0, top_k - len(protected_set))
    selected_non_protected = non_protected[:budget]
    selected = list(protected_set) + selected_non_protected

    logger.info(
        "XGBoost importance filter: %d → %d features (%d protected, %d by importance).",
        len(feature_cols), len(selected), len(protected_set), len(selected_non_protected),
    )

    return selected, importance


def run_feature_selection(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    correlation_threshold: float = 0.90,
    importance_top_k: int | None = None,
    protected_prefixes: List[str] | None = None,
    seed: int = 42,
) -> tuple[List[str], dict]:
    """Run the full two-stage feature selection pipeline.

    Returns
    -------
    (final_features, metadata_dict) where metadata_dict contains:
      - pearson_dropped: list of features dropped by correlation filter
      - importance_ranking: dict of feature -> importance score
      - final_count: number of features selected
    """
    metadata: dict = {}

    # Stage A: Pearson correlation filter
    surviving, corr_matrix = pearson_correlation_filter(
        df, feature_cols, target_col,
        threshold=correlation_threshold,
        protected_prefixes=protected_prefixes,
    )
    metadata["pearson_dropped"] = [c for c in feature_cols if c not in surviving]
    metadata["pearson_surviving_count"] = len(surviving)

    # Stage B: XGBoost importance ranking
    selected, importance = xgboost_importance_filter(
        df, surviving, target_col,
        top_k=importance_top_k,
        protected_prefixes=protected_prefixes,
        seed=seed,
    )
    metadata["importance_ranking"] = importance.to_dict()
    metadata["final_count"] = len(selected)

    return selected, metadata
