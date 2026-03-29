"""Seed the instrument_master table with the known ETF universe.

Usage::

    python scripts/db_seed.py [--config configs/china_open_universe_minimal.template.toml]

Idempotent — safe to run multiple times.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from src.config import load_config
from src.data.db import get_engine, get_session_factory, init_db, resolve_db_url
from src.data.repository import InstrumentRepository

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapping from canonical symbol -> yfinance ticker (only where they differ).
_YFINANCE_TICKER_MAP = {
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
}

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed instrument_master table.")
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
    init_db(engine)  # ensure tables exist

    repo = InstrumentRepository(get_session_factory(engine))

    # --- Core China ETFs ---
    for sym in config.universe_core:
        repo.register_instrument(
            symbol=sym,
            market=_infer_market(sym),
            asset_type="china_etf",
            yfinance_ticker=sym,
            is_core_training=True,
            history_policy_years=10,
        )

    # --- Optional China ETFs ---
    for sym in config.universe_optional:
        repo.register_instrument(
            symbol=sym,
            market=_infer_market(sym),
            asset_type="china_etf",
            yfinance_ticker=sym,
            is_core_training=True,
            history_policy_years=10,
        )

    # --- Unseen ETFs ---
    for sym in config.unseen_etfs:
        repo.register_instrument(
            symbol=sym,
            market=_infer_market(sym),
            asset_type="china_etf",
            yfinance_ticker=sym,
            is_core_training=False,
            history_policy_years=1,
        )

    # Dashboard ETFs are discovered dynamically via db_discover_etfs.py
    # — no hardcoded list needed here.

    # --- Cross-market ETFs ---
    for sym in config.cross_market:
        repo.register_instrument(
            symbol=sym,
            market="US",
            asset_type="cross_market_etf",
            yfinance_ticker=sym,
            is_cross_market=True,
            history_policy_years=10,
        )

    # --- Macro proxies ---
    for sym in config.global_risk:
        repo.register_instrument(
            symbol=sym,
            market="MACRO",
            asset_type="macro_proxy",
            yfinance_ticker=_YFINANCE_TICKER_MAP.get(sym, sym),
            is_macro_proxy=True,
            history_policy_years=10,
        )

    # --- Summary ---
    all_instruments = repo.list_all()
    logger.info("instrument_master seeded: %d instruments total", len(all_instruments))
    for inst in all_instruments:
        logger.info(
            "  %s  market=%s  type=%s  core=%s  cross=%s  macro=%s  yf=%s",
            inst.symbol,
            inst.market,
            inst.asset_type,
            inst.is_core_training,
            inst.is_cross_market,
            inst.is_macro_proxy,
            inst.yfinance_ticker,
        )


if __name__ == "__main__":
    main()
