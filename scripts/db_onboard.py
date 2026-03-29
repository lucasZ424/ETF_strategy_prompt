"""On-demand symbol onboarding: register a new ETF and fetch its history.

Usage::

    # Onboard one or more ETFs with 1-year default history
    python -m scripts.db_onboard 510050.SS 159941.SZ

    # Onboard with custom history window
    python -m scripts.db_onboard 159941.SZ --years 3

    # Onboard all dashboard ETFs from config
    python -m scripts.db_onboard --dashboard

Idempotent — safe to run multiple times (upserts instrument + bars).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.config import load_config
from src.data.db import get_engine, get_session_factory, init_db, resolve_db_url
from src.data.repository import InstrumentRepository
from scripts.db_fetch import fetch_and_upsert
from src.data.repository import BarRepository

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Market suffix -> market code
_MARKET_MAP = {
    ".SS": "SS",
    ".SZ": "SZ",
}


def _infer_market(symbol: str) -> str:
    for suffix, code in _MARKET_MAP.items():
        if symbol.endswith(suffix):
            return code
    return "US"


def onboard_symbol(
    symbol: str,
    inst_repo: InstrumentRepository,
    bar_repo: BarRepository,
    *,
    history_years: int = 1,
    end_date: str | None = None,
) -> int:
    """Register symbol in instrument_master and fetch history.

    Returns rows inserted.
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    # Register in instrument_master (idempotent)
    inst_repo.register_instrument(
        symbol=symbol,
        market=_infer_market(symbol),
        asset_type="china_etf",
        yfinance_ticker=symbol,
        is_core_training=False,
        is_cross_market=False,
        is_macro_proxy=False,
        history_policy_years=history_years,
        is_active=True,
    )

    # Fetch history
    n = fetch_and_upsert(symbol, symbol, bar_repo, end_date)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Onboard new ETF symbols into the database."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="ETF symbols to onboard (e.g. 159941.SZ 510050.SS)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Years of history to fetch (default: 1).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for fetch (default: today). Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--config",
        default="configs/china_open_universe_minimal.template.toml",
        help="Path to TOML config file.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    db_url = resolve_db_url(config.database.url, config.database.url_env)
    engine = get_engine(db_url)
    init_db(engine)

    sf = get_session_factory(engine)
    inst_repo = InstrumentRepository(sf)
    bar_repo = BarRepository(sf)

    # Determine which symbols to onboard
    symbols: list[str] = list(args.symbols) if args.symbols else []

    if not symbols:
        parser.error("Provide one or more symbols to onboard. "
                     "For bulk discovery, use: python -m scripts.db_discover_etfs")

    # Deduplicate while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    logger.info(
        "=== Onboarding %d symbols (history=%d years) ===",
        len(unique_symbols), args.years,
    )

    total = 0
    for sym in unique_symbols:
        try:
            n = onboard_symbol(
                sym, inst_repo, bar_repo,
                history_years=args.years,
                end_date=args.end_date,
            )
            total += n
        except Exception as exc:
            logger.error("Failed to onboard %s: %s", sym, exc)

    logger.info("=== Onboarding complete: %d new rows across %d symbols ===", total, len(unique_symbols))


if __name__ == "__main__":
    main()
