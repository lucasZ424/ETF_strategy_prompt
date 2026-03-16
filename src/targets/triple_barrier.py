"""Triple-barrier ternary gate labels.

For each date t and symbol:
  1. Compute upper/lower return barriers from volatility-scaled distance.
  2. Walk forward up to H bars. Label by which barrier is hit first:
       +1 = upper barrier hit first (long opportunity)
        0 = neither barrier hit within horizon (flat / ambiguous)
       -1 = lower barrier hit first (short / avoid)

Barrier distances and horizon are config-driven.
"""

from __future__ import annotations
from math import sqrt

import logging

import numpy as np
import pandas as pd

from src.config import BarrierConfig

logger = logging.getLogger(__name__)


def build_barrier_labels(
    backbone_df: pd.DataFrame,
    config: BarrierConfig,
) -> pd.DataFrame:
    """Build triple-barrier ternary labels for the gate model.

    Parameters
    ----------
    backbone_df : DataFrame with [date, symbol, adj_close, ret1_adj] at minimum.
    config : BarrierConfig with horizon, upper/lower multipliers, vol_lookback.
             target_scaling:
               - "daily": AFML-style target volatility
               - "sqrt_horizon": legacy sqrt(H) widening

    Returns
    -------
    DataFrame with columns [date, symbol, barrier_label], sorted by [date, symbol].
    Rows where the label cannot be computed (tail or warmup) are dropped.
    """
    required = {"date", "symbol", "adj_close", "ret1_adj"}
    missing = required - set(backbone_df.columns)
    if missing:
        raise ValueError(f"Missing columns for barrier labels: {missing}")

    results = []

    for symbol, grp in backbone_df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        labels = _label_one_symbol(
            adj_close=grp["adj_close"].values,
            ret1=grp["ret1_adj"].values,
            horizon=config.horizon,
            upper_mult=config.upper_multiplier,
            lower_mult=config.lower_multiplier,
            vol_lookback=config.vol_lookback,
            target_scaling=config.target_scaling,
        )
        sym_df = grp[["date", "symbol"]].copy()
        sym_df["barrier_label"] = labels
        results.append(sym_df)

    out = pd.concat(results, ignore_index=True)
    pre_drop = len(out)
    out = out.dropna(subset=["barrier_label"]).reset_index(drop=True)
    out["barrier_label"] = out["barrier_label"].astype(int)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    dist = out["barrier_label"].value_counts().to_dict()
    logger.info(
        "Barrier labels (H=%d, up=%.1f, dn=%.1f, vol_lb=%d, target_scaling=%s): "
        "%d rows, dropped %d. Distribution: %s",
        config.horizon, config.upper_multiplier, config.lower_multiplier,
        config.vol_lookback, config.target_scaling, len(out), pre_drop - len(out), dist,
    )

    return out


def _scale_target_vol(vol_t: float, horizon: int, target_scaling: str) -> float:
    """Scale daily volatility into barrier target according to config."""
    if target_scaling == "daily":
        return vol_t
    if target_scaling == "sqrt_horizon":
        return vol_t * sqrt(horizon)
    raise ValueError(
        f"Unknown barrier.target_scaling={target_scaling!r}. "
        "Use one of {'daily', 'sqrt_horizon'}."
    )


def _label_one_symbol(
    adj_close: np.ndarray,
    ret1: np.ndarray,
    horizon: int,
    upper_mult: float,
    lower_mult: float,
    vol_lookback: int,
    target_scaling: str,
) -> np.ndarray:
    """Compute barrier labels for one symbol's time series.

    Returns array of float (NaN where label cannot be computed).
    """
    n = len(adj_close)
    labels = np.full(n, np.nan)

    # Rolling volatility (std of ret1_adj over vol_lookback)
    vol = pd.Series(ret1).rolling(vol_lookback, min_periods=vol_lookback).std().values

    for t in range(n):
        # Need vol estimate and enough forward bars
        if np.isnan(vol[t]) or t + horizon >= n:
            continue

        price_t = adj_close[t]
        sigma = vol[t]

        if sigma <= 0 or np.isnan(price_t):
            continue

        target_vol = _scale_target_vol(float(sigma), horizon, target_scaling)
        if target_vol <= 0:
            continue
        upper_ret_barrier = upper_mult * target_vol
        lower_ret_barrier = -lower_mult * target_vol

        label = 0  # default: neither hit
        for h in range(1, horizon + 1):
            fwd_price = adj_close[t + h]
            if np.isnan(fwd_price) or fwd_price <= 0:
                continue
            cum_log_ret = np.log(fwd_price / price_t)
            if cum_log_ret >= upper_ret_barrier:
                label = 1
                break
            if cum_log_ret <= lower_ret_barrier:
                label = -1
                break

        labels[t] = label

    return labels
