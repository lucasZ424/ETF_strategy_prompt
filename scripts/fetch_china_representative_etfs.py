"""Fetch a minimal representative set of China ETFs for open-universe model bootstrap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


# Smallest practical core set with long history and style diversity.
# - 510050.SS: SSE 50 (mega-cap)
# - 510300.SS: CSI 300 (broad large-cap)
# - 510500.SS: CSI 500 (mid/small-cap)
# - 159915.SZ: ChiNext (growth/innovation)
CORE_CHINA_ETFS = ["510050.SS", "510300.SS", "510500.SS", "159915.SZ"]

# Optional extension ETF with shorter listing history.
OPTIONAL_CHINA_ETFS = ["588000.SS"]

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "raw" / "china_etfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in CORE_CHINA_ETFS + OPTIONAL_CHINA_ETFS:
        frame = fetch_one(ticker, START_DATE, END_DATE)
        csv_path = out_dir / f"{ticker}.csv"
        frame.to_csv(csv_path, index=False)

        if frame.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "is_core": ticker in CORE_CHINA_ETFS,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "max_na_ratio_ohlcv": None,
                    "median_volume": None,
                }
            )
            continue

        needed = ["Open", "High", "Low", "Close", "Volume"]
        max_na_ratio = float(frame[needed].isna().mean().max())
        rows.append(
            {
                "ticker": ticker,
                "is_core": ticker in CORE_CHINA_ETFS,
                "rows": int(frame.shape[0]),
                "start": str(pd.to_datetime(frame["Date"]).min().date()),
                "end": str(pd.to_datetime(frame["Date"]).max().date()),
                "max_na_ratio_ohlcv": max_na_ratio,
                "median_volume": float(frame["Volume"].median()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["is_core", "ticker"], ascending=[False, True])
    summary.to_csv(out_dir / "selection_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
