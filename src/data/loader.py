"""Load raw yfinance CSVs into standardised DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def load_single_csv(path: Path, symbol: str) -> pd.DataFrame:
    """Load one yfinance CSV, normalise columns, attach *symbol*."""

    df = pd.read_csv(path)
    df = df.rename(columns=_COLUMN_MAP)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "Loaded %s: %d rows, %s to %s",
        symbol,
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def load_china_etfs(
    raw_dir: Path,
    core_symbols: List[str],
    optional_symbols: List[str],
) -> pd.DataFrame:
    """Load and concatenate all China ETF CSVs into a pooled DataFrame."""

    china_dir = raw_dir / "china_etfs"
    frames: list[pd.DataFrame] = []
    for sym in core_symbols + optional_symbols:
        csv_path = china_dir / f"{sym}.csv"
        if not csv_path.exists():
            logger.warning("CSV not found for %s at %s, skipping", sym, csv_path)
            continue
        frames.append(load_single_csv(csv_path, sym))

    if not frames:
        raise FileNotFoundError(f"No China ETF CSVs found in {china_dir}")

    pooled = pd.concat(frames, ignore_index=True)
    logger.info(
        "Pooled China ETFs: %d rows, %d symbols",
        len(pooled),
        pooled["symbol"].nunique(),
    )
    return pooled


def load_cross_market_etfs(
    raw_dir: Path,
    symbols: List[str],
) -> pd.DataFrame:
    """Load cross-market ETF CSVs (e.g. SPY, QQQ, IEUR)."""

    cross_dir = raw_dir / "cross_market"
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        csv_path = cross_dir / f"{sym}.csv"
        if not csv_path.exists():
            logger.warning("CSV not found for %s at %s, skipping", sym, csv_path)
            continue
        frames.append(load_single_csv(csv_path, sym))

    if not frames:
        raise FileNotFoundError(f"No cross-market CSVs found in {cross_dir}")

    return pd.concat(frames, ignore_index=True)
