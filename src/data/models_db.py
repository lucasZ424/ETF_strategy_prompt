"""SQLAlchemy 2.0 ORM models for the ETF strategy database.

Tables
------
- instrument_master : symbol registry with universe / policy flags
- daily_bars        : unified OHLCV for all instruments
- data_fetch_log    : audit trail for every fetch operation
- prediction_snapshots : dashboard predictions for Grafana serving
- model_runs        : training run metadata (Phase 6+)
- dashboard_metrics_history : per-horizon evaluation metrics (Phase 7+)
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Table 1: instrument_master
# ---------------------------------------------------------------------------
class InstrumentMaster(Base):
    __tablename__ = "instrument_master"

    instrument_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    market: Mapped[str] = mapped_column(String(10), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str | None] = mapped_column(String(100))
    yfinance_ticker: Mapped[str | None] = mapped_column(String(30))
    is_core_training: Mapped[bool] = mapped_column(Boolean, default=False)
    is_cross_market: Mapped[bool] = mapped_column(Boolean, default=False)
    is_macro_proxy: Mapped[bool] = mapped_column(Boolean, default=False)
    history_policy_years: Mapped[int] = mapped_column(Integer, default=10)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Instrument {self.symbol} ({self.asset_type})>"


# ---------------------------------------------------------------------------
# Table 2: daily_bars
# ---------------------------------------------------------------------------
class DailyBar(Base):
    __tablename__ = "daily_bars"
    __table_args__ = (
        Index("idx_daily_bars_date", "trade_date"),
        Index("idx_daily_bars_instrument", "instrument_id", "trade_date"),
    )

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("instrument_master.instrument_id"), nullable=False
    )
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String(20), default="yfinance")
    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<DailyBar {self.symbol} {self.trade_date}>"


# ---------------------------------------------------------------------------
# Table 3: data_fetch_log
# ---------------------------------------------------------------------------
class DataFetchLog(Base):
    __tablename__ = "data_fetch_log"

    fetch_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    fetch_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    fetch_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    rows_fetched: Mapped[int] = mapped_column(Integer, default=0)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="success")
    error_message: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<FetchLog {self.symbol} {self.fetch_start_date}->{self.fetch_end_date} [{self.status}]>"


# ---------------------------------------------------------------------------
# Table 4: prediction_snapshots
# ---------------------------------------------------------------------------
class PredictionSnapshot(Base):
    __tablename__ = "prediction_snapshots"
    __table_args__ = (
        UniqueConstraint("model_version", "asof_date", "symbol", name="uq_pred_version_date_symbol"),
        Index("idx_pred_asof", "asof_date", "symbol"),
    )

    prediction_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    run_id: Mapped[str | None] = mapped_column(String(50))
    model_name: Mapped[str] = mapped_column(String(50), default="dashboard")
    model_version: Mapped[str | None] = mapped_column(String(50))
    asof_date: Mapped[date] = mapped_column(Date, nullable=False)
    prediction_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    current_close: Mapped[float] = mapped_column(Float, nullable=False)
    y_close_1d_pred: Mapped[float | None] = mapped_column(Float)
    y_close_3d_pred: Mapped[float | None] = mapped_column(Float)
    y_close_5d_pred: Mapped[float | None] = mapped_column(Float)
    feature_manifest_version: Mapped[str | None] = mapped_column(String(50))
    data_freshness_date: Mapped[date | None] = mapped_column(Date)

    def __repr__(self) -> str:
        return f"<Prediction {self.symbol} asof={self.asof_date}>"


# ---------------------------------------------------------------------------
# Table 5: model_runs (Phase 6+)
# ---------------------------------------------------------------------------
class ModelRun(Base):
    __tablename__ = "model_runs"

    run_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(50))
    config_hash: Mapped[str | None] = mapped_column(String(64))
    train_start_date: Mapped[date | None] = mapped_column(Date)
    train_end_date: Mapped[date | None] = mapped_column(Date)
    val_start_date: Mapped[date | None] = mapped_column(Date)
    val_end_date: Mapped[date | None] = mapped_column(Date)
    test_start_date: Mapped[date | None] = mapped_column(Date)
    test_end_date: Mapped[date | None] = mapped_column(Date)
    metrics_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<ModelRun {self.run_id} {self.model_name}>"


# ---------------------------------------------------------------------------
# Table 6: dashboard_metrics_history (Phase 7+)
# ---------------------------------------------------------------------------
class DashboardMetricsHistory(Base):
    __tablename__ = "dashboard_metrics_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("model_runs.run_id")
    )
    model_version: Mapped[str | None] = mapped_column(String(50))
    horizon_name: Mapped[str] = mapped_column(String(20), nullable=False)
    mse: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    mae: Mapped[float | None] = mapped_column(Float)
    mape: Mapped[float | None] = mapped_column(Float)
    r2: Mapped[float | None] = mapped_column(Float)
    n_rows: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<DashboardMetrics {self.horizon_name} run={self.run_id}>"
