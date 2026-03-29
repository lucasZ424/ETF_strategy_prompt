"""Incremental fetch: pull new data from yfinance and upsert into the database.

Usage::

    python scripts/db_fetch.py [--config configs/china_open_universe_minimal.template.toml]
                               [--end-date 2026-03-16]
                               [--symbols 510050.SS SPY VIX]

When no ``--symbols`` are given, all active instruments in ``instrument_master``
are fetched.  The script queries ``get_latest_date()`` per symbol and only
fetches rows after that date, making it safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
from src.config import load_config
from src.data.db import get_engine, get_session_factory, resolve_db_url
from src.data.repository import BarRepository, InstrumentRepository

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily bars via yfinance and return a standardised DataFrame.

    Output schema: ``[date, open, high, low, close, adj_close, volume]``.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw.columns = [str(c).strip() for c in raw.columns]

    col_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    raw = raw.rename(columns=col_map)
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


def fetch_and_upsert(
    symbol: str,
    yf_ticker: str,
    repo: BarRepository,
    end_date: str,
) -> int:
    """Fetch new data for *symbol* and upsert into DB.

    Returns rows inserted (approximate).
    """
    latest = repo.get_latest_date(symbol)
    if latest is not None:
        start_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = "2015-01-01"

    if start_date >= end_date:
        logger.info("  %s: already up to date (latest=%s)", symbol, latest)
        return 0

    logger.info("  %s (yf=%s): fetching %s to %s", symbol, yf_ticker, start_date, end_date)
    df = _yf_download(yf_ticker, start_date, end_date)

    if df.empty:
        logger.info("  %s: no new data returned from yfinance", symbol)
        repo.log_fetch(
            symbol=symbol,
            start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
            end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
            rows_fetched=0,
            rows_inserted=0,
            rows_updated=0,
            status="success",
        )
        return 0

    inserted, updated = repo.upsert_bars(df, symbol, "yfinance")
    repo.log_fetch(
        symbol=symbol,
        start_date=df["date"].min().date(),
        end_date=df["date"].max().date(),
        rows_fetched=len(df),
        rows_inserted=inserted,
        rows_updated=updated,
        status="success",
    )
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental yfinance fetch to DB.")
    parser.add_argument(
        "--config",
        default="configs/china_open_universe_minimal.template.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for fetch (default: today). Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Specific symbols to fetch. If omitted, all active instruments are fetched.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    db_url = resolve_db_url(config.database.url, config.database.url_env)
    engine = get_engine(db_url)
    sf = get_session_factory(engine)

    bar_repo = BarRepository(sf)
    inst_repo = InstrumentRepository(sf)

    # Determine which symbols to fetch.
    if args.symbols:
        symbols_to_fetch = args.symbols
    else:
        instruments = inst_repo.list_all(active_only=True)
        symbols_to_fetch = [i.symbol for i in instruments]

    # Build symbol -> yfinance_ticker mapping from instrument_master.
    ticker_map: dict[str, str] = {}
    for sym in symbols_to_fetch:
        inst = inst_repo.get_by_symbol(sym)
        if inst and inst.yfinance_ticker:
            ticker_map[sym] = inst.yfinance_ticker
        else:
            # Fallback: symbol is also the yfinance ticker.
            ticker_map[sym] = sym

    logger.info("=== Incremental DB fetch: %d symbols, end_date=%s ===",
                len(symbols_to_fetch), end_date)

    total_inserted = 0
    for sym in symbols_to_fetch:
        try:
            n = fetch_and_upsert(sym, ticker_map[sym], bar_repo, end_date)
            total_inserted += n
        except Exception as exc:
            logger.error("  %s: fetch failed — %s", sym, exc)
            bar_repo.log_fetch(
                symbol=sym,
                start_date=date(2000, 1, 1),
                end_date=date(2000, 1, 1),
                rows_fetched=0,
                rows_inserted=0,
                rows_updated=0,
                status="error",
                error_message=str(exc),
            )

    logger.info("=== Fetch complete: %d new rows across %d symbols ===",
                total_inserted, len(symbols_to_fetch))


if __name__ == "__main__":
    main()
