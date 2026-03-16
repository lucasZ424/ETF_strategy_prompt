"""Rule-based regime labels for optimizer-policy selection.

Baseline regime states:
    2 = aggressive  (broad uptrend, low stress)
    1 = balanced    (mixed trend, normal stress)
    0 = defensive   (weak trend, elevated stress)

These are date-level labels computed from the regime feature dataset.
A learned regime classifier may replace these rules in a later phase.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime state constants
AGGRESSIVE = 2
BALANCED = 1
DEFENSIVE = 0


def build_regime_labels(
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build rule-based regime labels from date-level regime features.

    Uses a simple composite scoring approach:
    - Trend signal: universe_ret10_mean, universe_breadth_pos10, cross_market_trend_score
    - Stress signal: macro_stress_score, universe_vol20_mean

    Parameters
    ----------
    regime_df : DataFrame with [date, regime features] from build_regime_features().

    Returns
    -------
    DataFrame with columns [date, regime_label], sorted by date.
    """
    df = regime_df.copy().sort_values("date").reset_index(drop=True)

    # Z-score each signal component for comparable scaling
    trend_score = _zscore_sum(df, [
        "universe_ret10_mean",
        "universe_breadth_pos10",
        "cross_market_trend_score",
    ])

    stress_score = _zscore_sum(df, [
        "macro_stress_score",
        "universe_vol20_mean",
    ])

    # Composite: positive trend minus stress
    composite = trend_score - stress_score

    # Classify into 3 states using tercile thresholds
    q33 = np.nanpercentile(composite, 33.3)
    q67 = np.nanpercentile(composite, 66.7)

    labels = np.where(
        composite >= q67, AGGRESSIVE,
        np.where(composite <= q33, DEFENSIVE, BALANCED)
    )

    result = pd.DataFrame({
        "date": df["date"],
        "regime_label": labels.astype(int),
    })

    # Drop any rows where composite was NaN
    valid_mask = ~np.isnan(composite)
    result = result[valid_mask].reset_index(drop=True)

    dist = result["regime_label"].value_counts().to_dict()
    logger.info(
        "Regime labels: %d rows. Distribution: %s "
        "(0=defensive, 1=balanced, 2=aggressive). "
        "Composite thresholds: q33=%.4f, q67=%.4f",
        len(result), dist, q33, q67,
    )

    return result


def _zscore_sum(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Sum of z-scored columns. Missing columns contribute 0."""
    total = np.zeros(len(df))
    for col in cols:
        if col not in df.columns:
            logger.warning("Regime label: missing column '%s', skipping.", col)
            continue
        vals = df[col].values.astype(float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std > 0:
            total += (vals - mean) / std
        # else: contributes 0
    return total
