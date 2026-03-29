"""Build timezone-aligned cross-market and global-risk features for China ETFs.

Timezone alignment
==================
China bar for date D → finalised at 15:00 CST (UTC+8) on D.
US bar for date D    → finalised at 16:00 ET = 05:00 CST on D+1 (EST)
                                              or 04:00 CST on D+1 (EDT).

Therefore US bar dated D is available before China opens on D+1.
For China date T, the latest safe US data is the US bar whose date is
strictly less than T.  ``merge_asof(allow_exact_matches=False)`` achieves
this and handles holiday mismatches automatically.

Cross-market ETF features
=========================
1d returns: spy_ret_lag1, qqq_ret_lag1, ieur_ret_lag1
10d returns: spy_ret10d_lag1, ieur_ret10d_lag1  (horizon-aligned)

Global risk features (VIX, TNX, DXY)
======================================
VIX: level change (vix_chg_lag1).
TNX: level change (us10y_chg_lag1).
DXY: log return (dxy_ret_lag1).
All aligned with the same strict-< rule as ETF returns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_etf_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot ETF adj_close prices and compute 1d + 10d log returns."""
    pivoted = df.pivot_table(index="date", columns="symbol", values="adj_close")

    # 1-day log returns
    log_rets_1d = np.log(pivoted / pivoted.shift(1))
    log_rets_1d.columns = [f"{col.lower()}_ret_lag1" for col in log_rets_1d.columns]

    # 10-day log returns (horizon-aligned with trade target)
    log_rets_10d = np.log(pivoted / pivoted.shift(10))
    log_rets_10d.columns = [f"{col.lower()}_ret10d_lag1" for col in log_rets_10d.columns]

    combined = pd.concat([log_rets_1d, log_rets_10d], axis=1)
    return combined.reset_index()


def _load_and_clean_series(csv_path: Path) -> pd.DataFrame:
    """Load a single-series yfinance CSV (VIX, TNX, DXY), return [date, close]."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next(c for c in df.columns if c.lower() in ("date",))
    close_col = next(c for c in df.columns if c.lower() == "close")
    df = df.rename(columns={date_col: "date", close_col: "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = df["close"].ffill()
    return df[["date", "close"]]


def _compute_global_risk_features(
    raw_dir: Path,
    macro_loader: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build VIX change, TNX change, DXY return — all on US calendar.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``cross_market/{VIX,TNX,DXY}.csv``.
        Ignored when *macro_loader* is provided.
    macro_loader : callable, optional
        If provided, called as ``macro_loader("VIX")`` etc. to obtain
        ``[date, close]`` DataFrames from the DB instead of reading CSVs.

    Returns wide DataFrame [date, vix_chg_lag1, us10y_chg_lag1, dxy_ret_lag1].
    """
    cross_dir = raw_dir / "cross_market" if raw_dir else None
    frames: dict[str, pd.DataFrame] = {}

    for name, csv_name in [("vix", "VIX"), ("tnx", "TNX"), ("dxy", "DXY")]:
        if macro_loader is not None:
            try:
                frames[name] = macro_loader(csv_name)
            except Exception:
                logger.warning("macro_loader failed for %s, skipping", csv_name)
        elif cross_dir is not None:
            path = cross_dir / f"{csv_name}.csv"
            if not path.exists():
                logger.warning("%s.csv not found, skipping global risk features for %s", csv_name, name)
                continue
            frames[name] = _load_and_clean_series(path)
        else:
            logger.warning("No raw_dir or macro_loader for %s, skipping", csv_name)

    if not frames:
        return pd.DataFrame(columns=["date"])

    # Merge all series on date
    merged = frames[list(frames.keys())[0]].rename(columns={"close": list(frames.keys())[0]})
    for name, df in list(frames.items())[1:]:
        merged = merged.merge(df.rename(columns={"close": name}), on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Compute changes
    if "vix" in merged.columns:
        merged["vix_chg_lag1"] = merged["vix"].diff()
    if "tnx" in merged.columns:
        merged["us10y_chg_lag1"] = merged["tnx"].diff()
    if "dxy" in merged.columns:
        merged["dxy_ret_lag1"] = np.log(merged["dxy"] / merged["dxy"].shift(1))

    # Drop raw level columns
    merged = merged.drop(columns=[c for c in ("vix", "tnx", "dxy") if c in merged.columns])
    return merged


def _merge_asof_strict(china_dates: pd.Series, right: pd.DataFrame) -> pd.DataFrame:
    """Align any US-dated DataFrame to China dates with strict < (no lookahead)."""
    china_df = pd.DataFrame({"china_date": sorted(pd.to_datetime(china_dates).unique())})
    right = right.copy()
    right["date"] = pd.to_datetime(right["date"])

    aligned = pd.merge_asof(
        china_df,
        right,
        left_on="china_date",
        right_on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    return aligned.drop(columns=["date"]).rename(columns={"china_date": "date"})


def align_cross_market_to_china(
    china_dates: pd.Series,
    cross_market_df: pd.DataFrame,
    cross_symbols: List[str],
    raw_dir: Path | None = None,
    macro_loader: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Align cross-market ETF returns AND global risk features to China trading dates.

    ETF features (1d):  spy_ret_lag1, qqq_ret_lag1, ieur_ret_lag1
    ETF features (10d): spy_ret10d_lag1, ieur_ret10d_lag1
    Global risk:        vix_chg_lag1, us10y_chg_lag1, dxy_ret_lag1

    Parameters
    ----------
    macro_loader : callable, optional
        If provided, used instead of CSV reads for VIX/TNX/DXY.
        Signature: ``macro_loader("VIX") -> DataFrame[date, close]``.
    """

    # ETF returns (1d + 10d)
    etf_rets = _compute_etf_returns(cross_market_df)
    etf_rets = etf_rets.sort_values("date").reset_index(drop=True)
    aligned = _merge_asof_strict(china_dates, etf_rets)

    # Global risk features
    if raw_dir is not None or macro_loader is not None:
        global_risk = _compute_global_risk_features(raw_dir, macro_loader=macro_loader)
        if len(global_risk.columns) > 1:
            global_risk = global_risk.sort_values("date").reset_index(drop=True)
            aligned_risk = _merge_asof_strict(china_dates, global_risk)
            aligned = aligned.merge(aligned_risk, on="date", how="left")

    for col in aligned.columns:
        if col != "date":
            n_nan = int(aligned[col].isna().sum())
            if n_nan > 0:
                logger.info("Cross-market %s: %d NaN out of %d rows", col, n_nan, len(aligned))
    return aligned
