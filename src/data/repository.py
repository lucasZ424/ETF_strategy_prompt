"""Database repository layer for the ETF strategy project.

Provides ``BarRepository``, ``InstrumentRepository``, and
``PredictionRepository`` — the primary data access objects that abstract
away SQL details and return pandas DataFrames matching the contracts of
the existing file-based loaders.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from src.data.models_db import (
    DailyBar,
    DataFetchLog,
    InstrumentMaster,
    PredictionSnapshot,
)

logger = logging.getLogger(__name__)

# Column order returned by all bar-loading methods — matches loader.py output.
_BAR_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]


# ---------------------------------------------------------------------------
# BarRepository
# ---------------------------------------------------------------------------
class BarRepository:
    """Read/write daily bars from/to the database."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    # ---- read operations --------------------------------------------------

    def load_bars(
        self,
        symbols: List[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Load daily bars for *symbols*, returning the canonical DataFrame.

        Output schema: ``[date, open, high, low, close, adj_close, volume, symbol]``
        with ``date`` as ``datetime64[ns]``.
        """
        with self._sf() as session:
            stmt = (
                select(
                    DailyBar.trade_date,
                    DailyBar.open,
                    DailyBar.high,
                    DailyBar.low,
                    DailyBar.close,
                    DailyBar.adj_close,
                    DailyBar.volume,
                    DailyBar.symbol,
                )
                .where(DailyBar.symbol.in_(symbols))
            )
            if start_date is not None:
                stmt = stmt.where(DailyBar.trade_date >= start_date)
            if end_date is not None:
                stmt = stmt.where(DailyBar.trade_date <= end_date)
            stmt = stmt.order_by(DailyBar.symbol, DailyBar.trade_date)

            rows = session.execute(stmt).all()

        if not rows:
            return pd.DataFrame(columns=_BAR_COLUMNS)

        df = pd.DataFrame(rows, columns=_BAR_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_china_etf_bars(
        self,
        core_symbols: List[str],
        optional_symbols: List[str] | None = None,
    ) -> pd.DataFrame:
        """DB equivalent of ``loader.load_china_etfs()``.

        Returns pooled DataFrame with the same schema.
        """
        symbols = list(core_symbols) + (optional_symbols or [])
        df = self.load_bars(symbols)
        if df.empty:
            raise RuntimeError(f"No China ETF bars found in DB for {symbols}")
        logger.info(
            "Pooled China ETFs from DB: %d rows, %d symbols",
            len(df), df["symbol"].nunique(),
        )
        return df

    def load_cross_market_bars(
        self,
        symbols: List[str],
    ) -> pd.DataFrame:
        """DB equivalent of ``loader.load_cross_market_etfs()``."""
        df = self.load_bars(symbols)
        if df.empty:
            raise RuntimeError(f"No cross-market bars found in DB for {symbols}")
        return df

    def load_macro_series(self, symbol: str) -> pd.DataFrame:
        """DB equivalent of ``cross_market._load_and_clean_series()``.

        Returns ``[date, close]`` with forward-filled NaN and
        ``date`` as ``datetime64[ns]`` — matching the CSV-based loader.
        """
        with self._sf() as session:
            stmt = (
                select(DailyBar.trade_date, DailyBar.close)
                .where(DailyBar.symbol == symbol)
                .order_by(DailyBar.trade_date)
            )
            rows = session.execute(stmt).all()

        if not rows:
            raise RuntimeError(f"No macro series data found in DB for {symbol}")

        df = pd.DataFrame(rows, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = df["close"].ffill()
        return df

    def load_unseen_bars(
        self,
        symbols: List[str] | None = None,
    ) -> pd.DataFrame:
        """DB equivalent of ``loader.load_unseen_etfs()``.

        If *symbols* is None, load all non-core, non-cross-market china_etf
        instruments that are active.
        """
        if symbols is None:
            with self._sf() as session:
                stmt = (
                    select(InstrumentMaster.symbol)
                    .where(InstrumentMaster.asset_type == "china_etf")
                    .where(InstrumentMaster.is_core_training.is_(False))
                    .where(InstrumentMaster.is_active.is_(True))
                )
                rows = session.execute(stmt).all()
            symbols = [r[0] for r in rows]

        if not symbols:
            raise RuntimeError("No unseen ETF symbols to load.")

        df = self.load_bars(symbols)
        if df.empty:
            raise RuntimeError(f"No unseen ETF bars found in DB for {symbols}")
        logger.info(
            "Loaded unseen ETFs from DB: %d rows, %d symbols (%s)",
            len(df), df["symbol"].nunique(),
            sorted(df["symbol"].unique().tolist()),
        )
        return df

    def get_latest_date(self, symbol: str) -> date | None:
        """Return the most recent trade_date for *symbol*, or None."""
        with self._sf() as session:
            stmt = (
                select(DailyBar.trade_date)
                .where(DailyBar.symbol == symbol)
                .order_by(DailyBar.trade_date.desc())
                .limit(1)
            )
            row = session.execute(stmt).first()
        return row[0] if row else None

    def get_date_coverage(self, symbol: str) -> tuple[date, date] | None:
        """Return ``(min_date, max_date)`` for *symbol*, or None."""
        with self._sf() as session:
            stmt = select(
                DailyBar.trade_date,
            ).where(DailyBar.symbol == symbol)
            from sqlalchemy import func as sa_func
            stmt = select(
                sa_func.min(DailyBar.trade_date),
                sa_func.max(DailyBar.trade_date),
            ).where(DailyBar.symbol == symbol)
            row = session.execute(stmt).first()
        if row and row[0] is not None:
            return (row[0], row[1])
        return None

    # ---- write operations -------------------------------------------------

    def upsert_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_source: str = "yfinance",
    ) -> tuple[int, int]:
        """Upsert daily bars for *symbol*.

        Uses PostgreSQL ``INSERT ... ON CONFLICT DO UPDATE`` for
        idempotent loads.

        Parameters
        ----------
        df : DataFrame
            Must contain columns ``[date, open, high, low, close, adj_close, volume]``.
            The ``symbol`` column is overridden by the *symbol* parameter.
        symbol : str
            Canonical symbol string.
        data_source : str
            Source tag stored in every row (default ``"yfinance"``).

        Returns
        -------
        (rows_inserted, rows_updated) — approximate counts.
        """
        if df.empty:
            return (0, 0)

        instrument_id = self._resolve_instrument_id(symbol)

        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "symbol": symbol,
                    "trade_date": pd.Timestamp(row["date"]).date(),
                    "instrument_id": instrument_id,
                    "open": _safe_float(row.get("open")),
                    "high": _safe_float(row.get("high")),
                    "low": _safe_float(row.get("low")),
                    "close": float(row["close"]),
                    "adj_close": _safe_float(row.get("adj_close")),
                    "volume": _safe_float(row.get("volume")),
                    "data_source": data_source,
                }
            )

        with self._sf() as session:
            # Count existing rows for this symbol to estimate inserts vs updates.
            existing_dates = set()
            stmt_existing = (
                select(DailyBar.trade_date)
                .where(DailyBar.symbol == symbol)
            )
            for row in session.execute(stmt_existing):
                existing_dates.add(row[0])

            stmt = pg_insert(DailyBar).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint=DailyBar.__table__.primary_key,
                set_={
                    "instrument_id": stmt.excluded.instrument_id,
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "adj_close": stmt.excluded.adj_close,
                    "volume": stmt.excluded.volume,
                    "data_source": stmt.excluded.data_source,
                    "loaded_at": text("NOW()"),
                },
            )
            session.execute(stmt)
            session.commit()

        new_dates = {r["trade_date"] for r in records}
        inserted = len(new_dates - existing_dates)
        updated = len(new_dates & existing_dates)
        logger.info(
            "Upserted %s: %d inserted, %d updated",
            symbol, inserted, updated,
        )
        return (inserted, updated)

    def log_fetch(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        rows_fetched: int,
        rows_inserted: int,
        rows_updated: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Write an entry to ``data_fetch_log``."""
        with self._sf() as session:
            entry = DataFetchLog(
                symbol=symbol,
                fetch_start_date=start_date,
                fetch_end_date=end_date,
                rows_fetched=rows_fetched,
                rows_inserted=rows_inserted,
                rows_updated=rows_updated,
                status=status,
                error_message=error_message,
            )
            session.add(entry)
            session.commit()

    # ---- helpers ----------------------------------------------------------

    def _resolve_instrument_id(self, symbol: str) -> int:
        """Get the instrument_id for *symbol*, raising if not found."""
        with self._sf() as session:
            stmt = select(InstrumentMaster.instrument_id).where(
                InstrumentMaster.symbol == symbol
            )
            row = session.execute(stmt).first()
        if row is None:
            raise ValueError(
                f"Symbol '{symbol}' not found in instrument_master. "
                "Run db_seed.py first."
            )
        return row[0]


# ---------------------------------------------------------------------------
# InstrumentRepository
# ---------------------------------------------------------------------------
class InstrumentRepository:
    """CRUD operations on the instrument_master table."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def get_by_symbol(self, symbol: str) -> InstrumentMaster | None:
        with self._sf() as session:
            stmt = select(InstrumentMaster).where(InstrumentMaster.symbol == symbol)
            return session.execute(stmt).scalar_one_or_none()

    def get_core_training_symbols(self) -> List[str]:
        with self._sf() as session:
            stmt = (
                select(InstrumentMaster.symbol)
                .where(InstrumentMaster.is_core_training.is_(True))
                .where(InstrumentMaster.is_active.is_(True))
            )
            return [r[0] for r in session.execute(stmt)]

    def get_cross_market_symbols(self) -> List[str]:
        with self._sf() as session:
            stmt = (
                select(InstrumentMaster.symbol)
                .where(InstrumentMaster.is_cross_market.is_(True))
                .where(InstrumentMaster.is_active.is_(True))
            )
            return [r[0] for r in session.execute(stmt)]

    def get_macro_proxy_symbols(self) -> List[str]:
        with self._sf() as session:
            stmt = (
                select(InstrumentMaster.symbol)
                .where(InstrumentMaster.is_macro_proxy.is_(True))
                .where(InstrumentMaster.is_active.is_(True))
            )
            return [r[0] for r in session.execute(stmt)]

    def get_instrument_id(self, symbol: str) -> int | None:
        with self._sf() as session:
            stmt = select(InstrumentMaster.instrument_id).where(
                InstrumentMaster.symbol == symbol
            )
            row = session.execute(stmt).first()
        return row[0] if row else None

    def register_instrument(
        self,
        symbol: str,
        market: str,
        asset_type: str,
        *,
        name: str | None = None,
        yfinance_ticker: str | None = None,
        is_core_training: bool = False,
        is_cross_market: bool = False,
        is_macro_proxy: bool = False,
        history_policy_years: int = 10,
        is_active: bool = True,
    ) -> InstrumentMaster:
        """Insert or update an instrument in the master table."""
        with self._sf() as session:
            existing = session.execute(
                select(InstrumentMaster).where(InstrumentMaster.symbol == symbol)
            ).scalar_one_or_none()

            if existing is not None:
                existing.market = market
                existing.asset_type = asset_type
                existing.name = name or existing.name
                existing.yfinance_ticker = yfinance_ticker or existing.yfinance_ticker
                existing.is_core_training = is_core_training
                existing.is_cross_market = is_cross_market
                existing.is_macro_proxy = is_macro_proxy
                existing.history_policy_years = history_policy_years
                existing.is_active = is_active
                existing.updated_at = datetime.utcnow()
                session.commit()
                logger.info("Updated instrument %s", symbol)
                return existing

            inst = InstrumentMaster(
                symbol=symbol,
                market=market,
                asset_type=asset_type,
                name=name,
                yfinance_ticker=yfinance_ticker,
                is_core_training=is_core_training,
                is_cross_market=is_cross_market,
                is_macro_proxy=is_macro_proxy,
                history_policy_years=history_policy_years,
                is_active=is_active,
            )
            session.add(inst)
            session.commit()
            session.refresh(inst)
            logger.info("Registered new instrument %s (id=%d)", symbol, inst.instrument_id)
            return inst

    def list_all(self, active_only: bool = True) -> List[InstrumentMaster]:
        with self._sf() as session:
            stmt = select(InstrumentMaster)
            if active_only:
                stmt = stmt.where(InstrumentMaster.is_active.is_(True))
            return list(session.execute(stmt).scalars())


# ---------------------------------------------------------------------------
# PredictionRepository (Phase 7 — Grafana serving)
# ---------------------------------------------------------------------------
class PredictionRepository:
    """Write and read dashboard prediction snapshots."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def write_snapshot(
        self,
        predictions: pd.DataFrame,
        model_version: str,
        asof_date: date,
        data_freshness_date: date | None = None,
    ) -> int:
        """Persist dashboard predictions for Grafana.

        Parameters
        ----------
        predictions : DataFrame
            Must contain: ``symbol``, ``current_close``,
            ``y_close_1d_pred``, ``y_close_3d_pred``, ``y_close_5d_pred``.
        model_version : str
            Version tag of the model that produced these predictions.
        asof_date : date
            The "today" date of the prediction.
        data_freshness_date : date, optional
            Latest bar date used for feature computation.

        Returns
        -------
        Number of rows written.
        """
        freshness = data_freshness_date or asof_date
        records = []
        for _, row in predictions.iterrows():
            records.append(
                {
                    "model_name": "dashboard",
                    "model_version": model_version,
                    "asof_date": asof_date,
                    "symbol": row["symbol"],
                    "current_close": float(row["current_close"]),
                    "y_close_1d_pred": _safe_float(row.get("y_close_1d_pred")),
                    "y_close_3d_pred": _safe_float(row.get("y_close_3d_pred")),
                    "y_close_5d_pred": _safe_float(row.get("y_close_5d_pred")),
                    "data_freshness_date": freshness,
                }
            )

        with self._sf() as session:
            stmt = pg_insert(PredictionSnapshot).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_pred_version_date_symbol",
                set_={
                    "current_close": stmt.excluded.current_close,
                    "y_close_1d_pred": stmt.excluded.y_close_1d_pred,
                    "y_close_3d_pred": stmt.excluded.y_close_3d_pred,
                    "y_close_5d_pred": stmt.excluded.y_close_5d_pred,
                    "prediction_time": text("NOW()"),
                },
            )
            session.execute(stmt)
            session.commit()

        logger.info("Wrote %d prediction snapshots for %s", len(records), asof_date)
        return len(records)

    def get_latest_forecasts(self) -> pd.DataFrame:
        """Return the most recent prediction per symbol.

        Useful for Grafana ``vw_latest_dashboard_forecast`` equivalent.
        """
        sql = text("""
            SELECT DISTINCT ON (symbol)
                symbol,
                asof_date,
                current_close,
                y_close_1d_pred,
                y_close_3d_pred,
                y_close_5d_pred,
                model_version,
                prediction_time,
                data_freshness_date
            FROM prediction_snapshots
            ORDER BY symbol, asof_date DESC, prediction_time DESC
        """)
        with self._sf() as session:
            rows = session.execute(sql).all()

        columns = [
            "symbol", "asof_date", "current_close",
            "y_close_1d_pred", "y_close_3d_pred", "y_close_5d_pred",
            "model_version", "prediction_time", "data_freshness_date",
        ]
        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val) -> float | None:
    """Convert *val* to float, returning None for NaN / None."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
