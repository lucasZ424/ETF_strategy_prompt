"""Discover all A-share ETFs and onboard those with ≥3 months of history.

Usage::

    # Discover and onboard all available A-share ETFs
    python -m scripts.db_discover_etfs

    # Dry run — list ETFs without writing to DB
    python -m scripts.db_discover_etfs --dry-run

    # Only discover, save ticker list to CSV without fetching bar data
    python -m scripts.db_discover_etfs --discover-only

Sources:
    - Sina Finance ETF listing via akshare (covers SSE + SZSE)
    - yfinance for OHLCV history

New ETFs are registered with:
    - is_core_training = False
    - history_policy_years = 1
    - asset_type = "china_etf"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import load_config
from src.data.db import get_engine, get_session_factory, init_db, resolve_db_url
from src.data.repository import BarRepository, InstrumentRepository

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Minimum trading days for 3 months of history (~63 trading days)
MIN_ROWS_3M = 60


def discover_all_etfs() -> pd.DataFrame:
    """Fetch the full A-share ETF universe from Sina via akshare.

    Returns DataFrame with columns: [code, name, symbol]
    where symbol is yfinance-compatible (e.g. 510050.SS).
    """
    import akshare as ak

    logger.info("Fetching A-share ETF listing from Sina Finance...")
    df = ak.fund_etf_category_sina(symbol="ETF基金")

    cols = df.columns.tolist()
    code_col = cols[0]  # e.g. "sh510050" or "sz159919"
    name_col = cols[1]  # Chinese name

    records = []
    for _, row in df.iterrows():
        raw_code = str(row[code_col])
        name = str(row[name_col])

        # Convert "sh510050" -> "510050.SS", "sz159919" -> "159919.SZ"
        if raw_code.startswith("sh"):
            symbol = raw_code[2:] + ".SS"
        elif raw_code.startswith("sz"):
            symbol = raw_code[2:] + ".SZ"
        else:
            continue

        records.append({"code": raw_code, "name": name, "symbol": symbol})

    result = pd.DataFrame(records)
    logger.info("Discovered %d A-share ETFs (%d SS, %d SZ)",
                len(result),
                result["symbol"].str.endswith(".SS").sum(),
                result["symbol"].str.endswith(".SZ").sum())
    return result


def check_yf_history(symbol: str, min_rows: int = MIN_ROWS_3M) -> tuple[bool, int, str | None, str | None]:
    """Check if yfinance has ≥min_rows of daily data for *symbol*.

    Returns (has_enough, row_count, first_date, last_date).
    """
    try:
        raw = yf.download(symbol, period="1y", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()

        n = len(raw)
        if n == 0:
            return (False, 0, None, None)

        first = str(pd.to_datetime(raw["Date"]).min().date())
        last = str(pd.to_datetime(raw["Date"]).max().date())
        return (n >= min_rows, n, first, last)
    except Exception:
        return (False, 0, None, None)


def fetch_and_upsert_bars(
    symbol: str,
    bar_repo: BarRepository,
    end_date: str,
) -> int:
    """Fetch 1-year history from yfinance and upsert into DB.

    Returns rows inserted.
    """
    latest = bar_repo.get_latest_date(symbol)
    if latest is not None:
        start_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    if start_date >= end_date:
        return 0

    raw = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if raw.empty:
        return 0

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw.columns = [str(c).strip() for c in raw.columns]

    col_map = {
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    }
    raw = raw.rename(columns=col_map)
    raw["date"] = pd.to_datetime(raw["date"])

    inserted, _ = bar_repo.upsert_bars(raw, symbol, "yfinance")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover all A-share ETFs and onboard those with ≥3 months of history."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List qualifying ETFs without writing to DB.",
    )
    parser.add_argument(
        "--discover-only", action="store_true",
        help="Save full ticker list to CSV without checking history or fetching.",
    )
    parser.add_argument(
        "--config",
        default="configs/china_open_universe_minimal.template.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Number of symbols to check/fetch between pauses (default: 20).",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date for bar fetch (default: today).",
    )
    args = parser.parse_args()

    # --- Step 1: Discover all ETFs ---
    all_etfs = discover_all_etfs()
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_etfs.to_csv(output_dir / "a_share_etf_universe.csv", index=False, encoding="utf-8-sig")
    logger.info("Full ETF list saved to data/a_share_etf_universe.csv")

    if args.discover_only:
        logger.info("--discover-only: skipping history check and DB operations.")
        return

    # --- Step 2: Setup DB ---
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    db_url = resolve_db_url(config.database.url, config.database.url_env)
    engine = get_engine(db_url)
    init_db(engine)
    sf = get_session_factory(engine)
    inst_repo = InstrumentRepository(sf)
    bar_repo = BarRepository(sf)
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    # Find which symbols are already in DB
    existing = {inst.symbol for inst in inst_repo.list_all(active_only=False)}
    new_symbols = all_etfs[~all_etfs["symbol"].isin(existing)]
    logger.info("%d ETFs already in DB, %d new to check", len(existing), len(new_symbols))

    # --- Step 3: Check history and onboard ---
    qualified = []
    skipped = []
    total = len(new_symbols)

    for i, (_, row) in enumerate(new_symbols.iterrows()):
        sym = row["symbol"]
        name = row["name"]

        if (i + 1) % args.batch_size == 0:
            logger.info("Progress: %d/%d checked, %d qualified so far",
                        i + 1, total, len(qualified))
            time.sleep(1)  # Brief pause to avoid rate limiting

        has_enough, n_rows, first, last = check_yf_history(sym)

        if has_enough:
            qualified.append({
                "symbol": sym, "name": name, "rows": n_rows,
                "first": first, "last": last,
            })
        else:
            skipped.append({"symbol": sym, "name": name, "rows": n_rows})

    logger.info("=== Discovery complete: %d qualified, %d skipped (<%d rows) ===",
                len(qualified), len(skipped), MIN_ROWS_3M)

    # Save results
    qual_df = pd.DataFrame(qualified)
    skip_df = pd.DataFrame(skipped)
    qual_df.to_csv(output_dir / "etf_qualified.csv", index=False, encoding="utf-8-sig")
    skip_df.to_csv(output_dir / "etf_skipped.csv", index=False, encoding="utf-8-sig")

    if args.dry_run:
        logger.info("--dry-run: %d ETFs would be onboarded. See data/etf_qualified.csv", len(qualified))
        return

    # --- Step 4: Register and fetch ---
    logger.info("=== Onboarding %d ETFs into database ===", len(qualified))
    total_inserted = 0
    errors = 0

    for i, rec in enumerate(qualified):
        sym = rec["symbol"]
        name = rec["name"]

        if (i + 1) % args.batch_size == 0:
            logger.info("Onboarding progress: %d/%d, %d rows inserted so far",
                        i + 1, len(qualified), total_inserted)
            time.sleep(1)

        try:
            # Register instrument
            inst_repo.register_instrument(
                symbol=sym,
                market="SS" if sym.endswith(".SS") else "SZ",
                asset_type="china_etf",
                name=name,
                yfinance_ticker=sym,
                is_core_training=False,
                history_policy_years=1,
            )

            # Fetch and upsert bars
            n = fetch_and_upsert_bars(sym, bar_repo, end_date)
            total_inserted += n

            bar_repo.log_fetch(
                symbol=sym,
                start_date=(datetime.now() - timedelta(days=400)).date(),
                end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
                rows_fetched=n,
                rows_inserted=n,
                rows_updated=0,
                status="success",
            )
        except Exception as exc:
            logger.error("Failed to onboard %s: %s", sym, exc)
            errors += 1

    logger.info(
        "=== Onboarding complete: %d ETFs, %d rows inserted, %d errors ===",
        len(qualified), total_inserted, errors,
    )


if __name__ == "__main__":
    main()
