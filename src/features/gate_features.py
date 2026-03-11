"""Gate-specific tradability overlay features (block 9).

These features capture path-risk and execution-cost information that the
alpha regressor should not see but the gate classifier needs.

All features on row T use only data from T-1 and earlier (no lookahead).

Produces:
  downside_semivol_10 — semideviation of negative ret1_adj over 10 bars
  max_drawdown_10     — rolling max drawdown over 10 bars using adj_close
  gap_freq_10         — fraction of last 10 bars with large gap_open_prev_adj
  whipsaw_5           — sign changes in ret1_adj over last 5 bars
  cost_buffer_10      — abs(ret10_adj) minus estimated round-trip cost threshold
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default round-trip cost in log-return terms (fee_bps + slippage_bps) * 2 sides
# 5 bps fee + 3 bps slippage = 8 bps one-way, 16 bps round-trip = 0.0016
_DEFAULT_COST_THRESHOLD = 0.0016

# Gap threshold for gap_freq_10: abs(gap) above this counts as a large gap
_GAP_THRESHOLD = 0.005


def add_gate_features(
    df: pd.DataFrame,
    cost_threshold: float = _DEFAULT_COST_THRESHOLD,
    gap_threshold: float = _GAP_THRESHOLD,
) -> pd.DataFrame:
    """Add gate-specific tradability overlay features.

    Requires: ret1_adj, ret10_adj, gap_open_prev_adj, adj_close already present.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 1. Downside semivol 10: std of negative ret1_adj over 10 bars (lagged)
    if "ret1_adj" in df.columns:
        neg_ret = df["ret1_adj"].where(df["ret1_adj"] < 0, 0.0)
        df["downside_semivol_10"] = neg_ret.groupby(df["symbol"]).transform(
            lambda s: s.shift(1).rolling(10, min_periods=10).std()
        )

    # 2. Max drawdown 10: rolling 10-bar max drawdown on lagged adj_close
    if "adj_close" in df.columns:
        df["max_drawdown_10"] = df.groupby("symbol").apply(
            _rolling_max_drawdown, window=10
        ).reset_index(level=0, drop=True)

    # 3. Gap frequency 10: fraction of last 10 bars with large gap
    if "gap_open_prev_adj" in df.columns:
        large_gap = (df["gap_open_prev_adj"].abs() > gap_threshold).astype(float)
        df["gap_freq_10"] = large_gap.groupby(df["symbol"]).transform(
            lambda s: s.shift(1).rolling(10, min_periods=10).mean()
        )

    # 4. Whipsaw 5: sign changes in ret1_adj over last 5 bars
    if "ret1_adj" in df.columns:
        sign_ret = np.sign(df["ret1_adj"])
        sign_change = (sign_ret != sign_ret.groupby(df["symbol"]).shift(1)).astype(float)
        df["whipsaw_5"] = sign_change.groupby(df["symbol"]).transform(
            lambda s: s.shift(1).rolling(5, min_periods=5).sum()
        )

    # 5. Cost buffer 10: abs(ret10_adj) - cost threshold
    if "ret10_adj" in df.columns:
        df["cost_buffer_10"] = df["ret10_adj"].abs() - cost_threshold

    return df


def _rolling_max_drawdown(group: pd.DataFrame, window: int = 10) -> pd.Series:
    """Compute rolling max drawdown over window bars, lagged by 1."""
    adj = group["adj_close"].shift(1)

    def _mdd(vals: np.ndarray) -> float:
        if len(vals) < window or np.any(np.isnan(vals)):
            return np.nan
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak
        return float(np.min(dd))

    return adj.rolling(window, min_periods=window).apply(_mdd, raw=True)
