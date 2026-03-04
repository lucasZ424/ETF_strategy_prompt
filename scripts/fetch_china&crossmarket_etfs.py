"""Fetch representative China ETFs and cross-market context ETFs for open-universe model bootstrap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


# Smallest practical core set with long history and style diversity.
# - 510050.SS: SSE 50 (mega-cap)
# - 510300.SS: CSI 300 (broad large-cap)
# - 510500.SS: CSI 500 (mid/small-cap)
# - 159915.SZ: ChiNext (growth/innovation)
CHINA_ETFS = ["510050.SS", "510300.SS", "510500.SS", "159915.SZ","588000.SS"]

# Cross-market ETFs for feature engineering (lagged, timezone-aligned to China decision time).
CROSS_MARKET_ETFS = ["SPY", "QQQ", "IEUR"]

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _summarize(frame: pd.DataFrame, ticker: str) -> dict:
    if frame.empty:
        return {
            "ticker": ticker,
            "rows": 0,
            "start": None,
            "end": None,
            "max_na_ratio_ohlcv": None,
            "median_volume": None,
        }
    needed = ["Open", "High", "Low", "Close","Adj Close", "Volume"]
    max_na_ratio = float(frame[needed].isna().mean().max())
    return {
        "ticker": ticker,
        "rows": int(frame.shape[0]),
        "start": str(pd.to_datetime(frame["Date"]).min().date()),
        "end": str(pd.to_datetime(frame["Date"]).max().date()),
        "max_na_ratio_ohlcv": max_na_ratio,
        "median_volume": float(frame["Volume"].median()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    # --- China ETFs ---
    china_dir = root / "data" / "raw" / "china_etfs"
    china_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in CHINA_ETFS:
        frame = fetch_one(ticker, START_DATE, END_DATE)
        frame.to_csv(china_dir / f"{ticker}.csv", index=False)
        rows.append(_summarize(frame, ticker, ticker in CHINA_ETFS))

    summary = pd.DataFrame(rows).sort_values("ticker", ascending=[False, True])
    summary.to_csv(china_dir / "selection_summary.csv", index=False)
    print("=== China ETFs ===")
    print(summary.to_string(index=False))

    # --- Cross-market ETFs ---
    cross_dir = root / "data" / "raw" / "cross_market"
    cross_dir.mkdir(parents=True, exist_ok=True)

    cross_rows = []
    for ticker in CROSS_MARKET_ETFS:
        frame = fetch_one(ticker, START_DATE, END_DATE)
        frame.to_csv(cross_dir / f"{ticker}.csv", index=False)
        cross_rows.append(_summarize(frame, ticker, is_core=False))

    cross_summary = pd.DataFrame(cross_rows).sort_values("ticker")
    cross_summary.to_csv(cross_dir / "selection_summary.csv", index=False)
    print("\n=== Cross-market ETFs ===")
    print(cross_summary.to_string(index=False))


if __name__ == "__main__":
    main()
